"""
Meta Prophet time-series worker.

Maintains a rolling window of (timestamp, log-price) pairs, refits
Prophet every N ticks, and forecasts 1 bar ahead (15 min) to generate
trading signals.

Prophet is substantially slower to fit than ARIMA, so the refit interval
is much longer (default: every 50 ticks).
"""
import os
import sys
import time
import logging
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import CONFIG, compute_signal_threshold
from paper_trader import PaperTrader

_TR = CONFIG["TRAINING"]

# Refit every this many ticks — Prophet is slow; tune down if CPU is available
_REFIT_EVERY = 50


def _post(sq, comp: str, status: str, msg: str = "") -> None:
    if sq is None:
        return
    try:
        sq.put_nowait({"component": comp, "status": status, "message": msg})
    except Exception:
        pass


def run_prophet_worker(data_queue, status_queue=None):
    try:
        import psutil
        p     = psutil.Process(os.getpid())
        total = psutil.cpu_count(logical=True)
        if total and total > CONFIG["CPU"]["OS_HEADROOM_CORES"]:
            p.cpu_affinity(list(range(total - CONFIG["CPU"]["OS_HEADROOM_CORES"])))
    except Exception:
        pass

    print(f"[PROPHET_FB] Worker started (PID {os.getpid()})")

    try:
        from prophet import Prophet
    except ImportError:
        print("[PROPHET_FB] prophet not installed — worker exiting (pip install prophet)")
        _post(status_queue, "PROPHET_FB", "ERROR",
              "prophet not installed — pip install prophet")
        return

    # Suppress noisy Stan/Prophet logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    window   = _TR.get("PROPHET_WINDOW", 1000)
    n_cp     = _TR.get("PROPHET_CHANGEPOINTS", 25)
    do_seas  = _TR.get("PROPHET_SEASONALITY", False)

    min_obs     = 50
    # Per-symbol state: prices window, times window, fitted model, tick counter, trader
    sym_state: dict[str, dict] = {}
    idle_logged = False

    _post(status_queue, "PROPHET_FB", "WAITING",
          f"Warming up — need {min_obs} ticks")

    while True:
        if not data_queue.empty():
            idle_logged = False
            data   = data_queue.get()
            symbol = data.get("symbol", "")

            if symbol not in sym_state:
                sym_state[symbol] = {
                    "prices":     deque(maxlen=window),
                    "times":      deque(maxlen=window),
                    "model":      None,
                    "tick_count": 0,
                    "trader":     PaperTrader("PROPHET_FB", symbol),
                }

            st    = sym_state[symbol]
            price = data["current_price"]
            now   = datetime.now(timezone.utc)
            st["prices"].append(price)
            st["times"].append(now)
            st["tick_count"] += 1

            sym_tag = f"[{symbol}] " if symbol else ""

            if len(st["prices"]) < min_obs:
                _post(status_queue, "PROPHET_FB", "WAITING",
                      f"{sym_tag}Warming up… {min_obs - len(st['prices'])} more ticks needed")
            else:
                if st["model"] is None or st["tick_count"] % _REFIT_EVERY == 0:
                    try:
                        price_arr  = np.array(st["prices"])
                        log_prices = np.log(price_arr)
                        ds_col     = [t.replace(tzinfo=None) for t in st["times"]]
                        df_p       = pd.DataFrame({"ds": ds_col, "y": log_prices})
                        m = Prophet(
                            daily_seasonality   = do_seas,
                            weekly_seasonality  = False,
                            yearly_seasonality  = False,
                            n_changepoints      = n_cp,
                            uncertainty_samples = 0,
                        )
                        m.fit(df_p)
                        st["model"] = m
                        _post(status_queue, "PROPHET_FB", "OK",
                              f"{sym_tag}Prophet fitted | {len(st['prices'])} obs")
                    except Exception as e:
                        _post(status_queue, "PROPHET_FB", "ERROR",
                              f"{sym_tag}Fit error: {e}")
                        st["model"] = None

                if st["model"] is not None:
                    try:
                        last_dt  = list(st["times"])[-1].replace(tzinfo=None)
                        next_dt  = last_dt + pd.Timedelta(minutes=15)
                        future   = pd.DataFrame({"ds": [next_dt]})
                        forecast = st["model"].predict(future)
                        pred_log = float(forecast["yhat"].iloc[0])
                        curr_log = np.log(float(price))
                        fc       = pred_log - curr_log

                        threshold = compute_signal_threshold(data.get("atr_14", 0.0), price)
                        _post(status_queue, "PROPHET_FB", "INFERRING",
                              f"{sym_tag}Forecast={fc:+.6f}  ${price:,.2f}")

                        if fc > threshold:
                            signal = 1
                            _post(status_queue, "PROPHET_FB", "TRADING",
                                  f"{sym_tag}BUY  forecast={fc:+.6f}  ${price:,.2f}")
                        elif fc < -threshold:
                            signal = -1
                            _post(status_queue, "PROPHET_FB", "TRADING",
                                  f"{sym_tag}SELL  forecast={fc:+.6f}  ${price:,.2f}")
                        else:
                            signal = 0
                            _post(status_queue, "PROPHET_FB", "OK",
                                  f"{sym_tag}HOLD  forecast={fc:+.6f}  ${price:,.2f}")

                        st["trader"].on_signal(signal, price, fc)
                    except Exception as e:
                        _post(status_queue, "PROPHET_FB", "ERROR", str(e))
        else:
            if not idle_logged:
                _post(status_queue, "PROPHET_FB", "WAITING", "Waiting for tick data...")
                idle_logged = True

        time.sleep(0.005)
