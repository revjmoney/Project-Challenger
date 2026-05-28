"""
Generic sklearn-API inference worker.

Handles any model that exposes a .predict(X) interface and is serialised
with joblib — LightGBM, CatBoost, RandomForest, ExtraTrees, ElasticNet,
SVR, and MLPRegressor all use this same worker.

Usage (via functools.partial):
    partial(run_generic_worker, "LGBM_TREE", CONFIG["PATHS"]["LGBM_MODEL"])
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import psutil

# Inference pipeline passes numpy arrays after scaling; feature-name warning is expected
warnings.filterwarnings("ignore", message="X does not have valid feature names")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import CONFIG, FEATURE_COLS, compute_signal_threshold
from model_security import load_joblib
from paper_trader import PaperTrader


def _apply_cpu_headroom():
    try:
        p     = psutil.Process(os.getpid())
        total = psutil.cpu_count(logical=True)
        if total and total > CONFIG["CPU"]["OS_HEADROOM_CORES"]:
            allowed = list(range(total - CONFIG["CPU"]["OS_HEADROOM_CORES"]))
            p.cpu_affinity(allowed)
    except Exception:
        pass


def _post(status_queue, component: str, status: str, message: str = "") -> None:
    if status_queue is None:
        return
    try:
        status_queue.put_nowait({"component": component, "status": status, "message": message})
    except Exception:
        pass


def run_generic_worker(model_key: str, model_path: str, data_queue, status_queue=None):
    """
    Run inference for any joblib-serialised sklearn-API model.

    Parameters
    ----------
    model_key   : CONFIG["ACTIVE_MODELS"] key (e.g. "LGBM_TREE") — used as
                  the ActivityTracker component name and PaperTrader model name.
    model_path  : Absolute path to the .pkl model file.
    data_queue  : Multiprocessing queue receiving tick feature dicts.
    status_queue: Optional queue for posting ActivityTracker updates.
    """
    _apply_cpu_headroom()
    print(f"[{model_key}] Worker started (PID {os.getpid()})")

    _post(status_queue, model_key, "LOADING", "Loading model & scaler...")
    try:
        model  = load_joblib(model_path)
        scaler = load_joblib(CONFIG["PATHS"]["SCALER"])
        print(f"[{model_key}] Model loaded.")
        _post(status_queue, model_key, "OK", "Model ready")
    except FileNotFoundError as e:
        print(f"[{model_key}] FATAL: model file missing — {e}")
        _post(status_queue, model_key, "ERROR", f"Model file missing: {e}")
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
                    traders[symbol] = PaperTrader(model_key, symbol)

                price = data["current_price"]
                feat  = np.array([[data[f] for f in FEATURE_COLS]], dtype=np.float32)
                threshold = compute_signal_threshold(data.get("atr_14", 0.0), price)

                sym_tag = f"[{symbol}] " if symbol else ""
                _post(status_queue, model_key, "INFERRING",
                      f"{sym_tag}Inferring @ ${price:,.2f}")
                feat_scaled = scaler.transform(feat)
                prediction  = float(model.predict(feat_scaled)[0])

                if prediction > threshold:
                    signal = 1
                    _post(status_queue, model_key, "TRADING",
                          f"{sym_tag}BUY  pred={prediction:+.6f}  ${price:,.2f}")
                elif prediction < -threshold:
                    signal = -1
                    _post(status_queue, model_key, "TRADING",
                          f"{sym_tag}SELL  pred={prediction:+.6f}  ${price:,.2f}")
                else:
                    signal = 0
                    _post(status_queue, model_key, "OK",
                          f"{sym_tag}HOLD  pred={prediction:+.6f}  ${price:,.2f}")

                traders[symbol].on_signal(signal, price, prediction)

            except Exception as e:
                print(f"[{model_key}] Inference error: {e}")
                _post(status_queue, model_key, "ERROR", str(e))
        else:
            if not idle_logged:
                _post(status_queue, model_key, "WAITING", "Waiting for tick data...")
                idle_logged = True

        time.sleep(0.005)
