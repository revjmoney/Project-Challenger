"""
Scikit-Learn Ridge regression worker process.
Runs on CPU; one dedicated core assigned by main.py.
"""
import os
import sys
import time
import warnings

import numpy as np
import psutil

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Workers run as spawned processes; project root must be on path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import CONFIG, FEATURE_COLS, compute_signal_threshold
from model_security import load_joblib
from paper_trader import PaperTrader

_PATHS     = CONFIG["PATHS"]
MODEL_NAME = "SKLEARN_LINEAR"


def _apply_cpu_headroom():
    try:
        p     = psutil.Process(os.getpid())
        total = psutil.cpu_count(logical=True)
        if total and total > CONFIG["CPU"]["OS_HEADROOM_CORES"]:
            allowed = list(range(total - CONFIG["CPU"]["OS_HEADROOM_CORES"]))
            p.cpu_affinity(allowed)
    except Exception:
        pass   # cpu_affinity not available on all platforms


def _post(status_queue, component: str, status: str, message: str = "") -> None:
    """Post a status update onto the cross-process status queue (fire-and-forget)."""
    if status_queue is None:
        return
    try:
        status_queue.put_nowait({"component": component, "status": status, "message": message})
    except Exception:
        pass


def run_sklearn_worker(data_queue, status_queue=None):
    _apply_cpu_headroom()
    print(f"[{MODEL_NAME}] Worker started (PID {os.getpid()})")

    _post(status_queue, "SKLEARN", "LOADING", "Loading model & scaler...")
    try:
        model  = load_joblib(_PATHS["SKLEARN_MODEL"])
        scaler = load_joblib(_PATHS["SCALER"])
        print(f"[{MODEL_NAME}] Model loaded.")
        _post(status_queue, "SKLEARN", "OK", "Model ready")
    except FileNotFoundError as e:
        print(f"[{MODEL_NAME}] FATAL: model file missing — {e}")
        _post(status_queue, "SKLEARN", "ERROR", f"Model file missing: {e}")
        return

    # One PaperTrader per symbol — created lazily on first tick for that coin
    traders: dict[str, PaperTrader] = {}
    idle_logged = False

    while True:
        if not data_queue.empty():
            idle_logged = False
            data = data_queue.get()
            try:
                symbol = data.get("symbol", "")
                if symbol not in traders:
                    traders[symbol] = PaperTrader(MODEL_NAME, symbol)

                price = data["current_price"]
                feat  = np.array([[data[f] for f in FEATURE_COLS]], dtype=np.float32)
                threshold = compute_signal_threshold(data.get("atr_14", 0.0), price)

                sym_tag = f"[{symbol}] " if symbol else ""
                _post(status_queue, "SKLEARN", "INFERRING",
                      f"{sym_tag}Inferring @ ${price:,.2f}")
                feat_scaled = scaler.transform(feat)
                prediction  = float(model.predict(feat_scaled)[0])

                if prediction > threshold:
                    signal = 1
                    _post(status_queue, "SKLEARN", "TRADING",
                          f"{sym_tag}BUY  pred={prediction:+.6f}  ${price:,.2f}")
                elif prediction < -threshold:
                    signal = -1
                    _post(status_queue, "SKLEARN", "TRADING",
                          f"{sym_tag}SELL  pred={prediction:+.6f}  ${price:,.2f}")
                else:
                    signal = 0
                    _post(status_queue, "SKLEARN", "OK",
                          f"{sym_tag}HOLD  pred={prediction:+.6f}  ${price:,.2f}")

                traders[symbol].on_signal(signal, price, prediction)

            except Exception as e:
                print(f"[{MODEL_NAME}] Inference error: {e}")
                _post(status_queue, "SKLEARN", "ERROR", str(e))
        else:
            if not idle_logged:
                _post(status_queue, "SKLEARN", "WAITING", "Waiting for tick data...")
                idle_logged = True

        time.sleep(0.005)
