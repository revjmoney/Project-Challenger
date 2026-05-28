"""
ARIMA time-series worker.

Maintains a rolling window of price log-returns, refits ARIMA every
N ticks, and predicts 1 step ahead to generate trading signals.
Does NOT load a pre-trained model file — the rolling approach makes
live predictions more relevant than an offline fit.
"""
import os
import sys
import time
from collections import deque

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import CONFIG, compute_signal_threshold
from paper_trader import PaperTrader

_TR = CONFIG["TRAINING"]

# Refit ARIMA every this many new ticks (balance freshness vs. CPU cost)
_REFIT_EVERY = 20


def _post(sq, comp: str, status: str, msg: str = "") -> None:
    if sq is None:
        return
    try:
        sq.put_nowait({"component": comp, "status": status, "message": msg})
    except Exception:
        pass


def run_arima_worker(data_queue, status_queue=None):
    try:
        import psutil
        p     = psutil.Process(os.getpid())
        total = psutil.cpu_count(logical=True)
        if total and total > CONFIG["CPU"]["OS_HEADROOM_CORES"]:
            p.cpu_affinity(list(range(total - CONFIG["CPU"]["OS_HEADROOM_CORES"])))
    except Exception:
        pass

    print(f"[ARIMA_STATS] Worker started (PID {os.getpid()})")

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("[ARIMA_STATS] statsmodels not installed — worker exiting")
        _post(status_queue, "ARIMA_STATS", "ERROR", "statsmodels not installed")
        return

    p_ord = _TR.get("ARIMA_P", 2)
    d_ord = _TR.get("ARIMA_D", 0)
    q_ord = _TR.get("ARIMA_Q", 2)
    order  = (p_ord, d_ord, q_ord)
    window = _TR.get("ARIMA_WINDOW", 500)

    # Minimum log-return observations needed before first fit
    min_obs = max(30, p_ord + q_ord + 5)

    # Per-symbol state: prices window, fitted model, tick counter, trader
    sym_state: dict[str, dict] = {}
    idle_logged = False

    _post(status_queue, "ARIMA_STATS", "WAITING", f"Warming up — need {min_obs} ticks")

    while True:
        if not data_queue.empty():
            idle_logged = False
            data   = data_queue.get()
            symbol = data.get("symbol", "")

            if symbol not in sym_state:
                sym_state[symbol] = {
                    "prices":     deque(maxlen=window + 1),
                    "model":      None,
                    "tick_count": 0,
                    "trader":     PaperTrader("ARIMA_STATS", symbol),
                }

            st    = sym_state[symbol]
            price = data["current_price"]
            st["prices"].append(price)
            st["tick_count"] += 1

            sym_tag    = f"[{symbol}] " if symbol else ""
            n_prices   = len(st["prices"])
            n_log_rets = n_prices - 1

            if n_log_rets < min_obs:
                _post(status_queue, "ARIMA_STATS", "WAITING",
                      f"{sym_tag}Warming up… {min_obs - n_log_rets} more ticks needed")
            else:
                if st["model"] is None or st["tick_count"] % _REFIT_EVERY == 0:
                    try:
                        price_arr = np.array(st["prices"])
                        log_rets  = np.diff(np.log(price_arr))
                        res = ARIMA(log_rets, order=order).fit(
                            method_kwargs={"warn_convergence": False}
                        )
                        st["model"] = res
                        _post(status_queue, "ARIMA_STATS", "OK",
                              f"{sym_tag}ARIMA({p_ord},{d_ord},{q_ord}) fitted | {len(log_rets)} obs")
                    except Exception as e:
                        _post(status_queue, "ARIMA_STATS", "ERROR",
                              f"{sym_tag}Fit error: {e}")
                        st["model"] = None

                if st["model"] is not None:
                    try:
                        fc = float(st["model"].forecast(1)[0])
                        threshold = compute_signal_threshold(data.get("atr_14", 0.0), price)
                        _post(status_queue, "ARIMA_STATS", "INFERRING",
                              f"{sym_tag}Forecast={fc:+.6f}  ${price:,.2f}")

                        if fc > threshold:
                            signal = 1
                            _post(status_queue, "ARIMA_STATS", "TRADING",
                                  f"{sym_tag}BUY  forecast={fc:+.6f}  ${price:,.2f}")
                        elif fc < -threshold:
                            signal = -1
                            _post(status_queue, "ARIMA_STATS", "TRADING",
                                  f"{sym_tag}SELL  forecast={fc:+.6f}  ${price:,.2f}")
                        else:
                            signal = 0
                            _post(status_queue, "ARIMA_STATS", "OK",
                                  f"{sym_tag}HOLD  forecast={fc:+.6f}  ${price:,.2f}")

                        st["trader"].on_signal(signal, price, fc)
                    except Exception as e:
                        _post(status_queue, "ARIMA_STATS", "ERROR", str(e))
        else:
            if not idle_logged:
                _post(status_queue, "ARIMA_STATS", "WAITING", "Waiting for tick data...")
                idle_logged = True

        time.sleep(0.005)
