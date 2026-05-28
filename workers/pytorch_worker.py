"""
PyTorch LSTM worker process.
Auto-detects CUDA and pushes inference to GPU when available.
Maintains a rolling sequence buffer for temporal context.

CPU-only install note
---------------------
If you installed via setup_cpu.bat / setup_cpu.sh, PyTorch has no CUDA
support — torch.cuda.is_available() returns False and this worker runs
entirely on CPU.  Everything works; training and inference are just slower
(expect ~5-15 min per training run on a modern laptop vs <1 min on GPU).

torch.compile() is skipped automatically on CPU-only builds (Triton,
the required JIT backend, is Linux + CUDA only).  The worker falls back
to standard eager execution with no action required.

If LSTM training speed is unacceptable on your hardware, disable the model:
    config.py → ACTIVE_MODELS → "PYTORCH_LSTM": False
Ridge (sklearn) and XGBoost continue running unaffected.
"""
import collections
import os
import sys
import time

import numpy as np
import psutil
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import CONFIG, FEATURE_COLS, compute_signal_threshold
from model_security import load_joblib, load_torch
from paper_trader import PaperTrader
from training import LSTMPredictor   # reuse the class definition

_PATHS     = CONFIG["PATHS"]
_TR        = CONFIG["TRAINING"]
MODEL_NAME = "PYTORCH_LSTM"


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
    """Post a status update onto the cross-process status queue (fire-and-forget)."""
    if status_queue is None:
        return
    try:
        status_queue.put_nowait({"component": component, "status": status, "message": message})
    except Exception:
        pass


def run_pytorch_worker(data_queue, status_queue=None):
    _apply_cpu_headroom()
    print(f"[{MODEL_NAME}] Worker started (PID {os.getpid()})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{MODEL_NAME}] Inference device: {device}")

    _post(status_queue, "PYTORCH", "LOADING", f"Loading LSTM model on {device}...")
    try:
        checkpoint = load_torch(_PATHS["PYTORCH_MODEL"], map_location=device)
        model = LSTMPredictor(**checkpoint["model_cfg"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        scaler = load_joblib(_PATHS["SCALER"])
        print(f"[{MODEL_NAME}] Model loaded on {device}.")
    except FileNotFoundError as e:
        print(f"[{MODEL_NAME}] FATAL: model file missing — {e}")
        _post(status_queue, "PYTORCH", "ERROR", f"Model file missing: {e}")
        return

    # ── torch.compile() — optional inference speedup ──────────────────────────
    # PyTorch 2.0+ can graph-capture and fuse the LSTM forward pass.
    # mode="reduce-overhead"  minimises Python dispatch cost per forward call.
    # On Windows the Triton/Inductor backend is unavailable; we do a warm-run
    # so any backend errors surface here instead of mid-loop, then fall back.
    _compile_tag = "eager"
    seq_len      = _TR["SEQUENCE_LENGTH"]
    n_feat       = len(FEATURE_COLS)

    if hasattr(torch, "compile"):
        _raw_model = model
        try:
            _post(status_queue, "PYTORCH", "LOADING", "Compiling LSTM graph...")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            # Warm-run: forces the backend to JIT-compile now, not on first tick
            with torch.no_grad():
                _dummy = torch.zeros(1, seq_len, n_feat, device=device)
                _ = model(_dummy)
                del _dummy
            _compile_tag = "compiled"
            print(f"[{MODEL_NAME}] torch.compile() active  (reduce-overhead, {device})")
        except Exception as _ce:
            model = _raw_model   # revert to the original eager model
            _compile_tag = "eager"
            print(f"[{MODEL_NAME}] torch.compile() skipped ({type(_ce).__name__}): {_ce}")

    _post(status_queue, "PYTORCH", "OK",
          f"LSTM ready on {device} [{_compile_tag}]")

    # Per-symbol state: each coin gets its own sequence buffer and PaperTrader
    # so that LSTM context windows don't bleed across different markets.
    sym_seq_bufs: dict[str, collections.deque] = {}
    traders:      dict[str, PaperTrader]       = {}
    idle_logged = False

    while True:
        if not data_queue.empty():
            idle_logged = False
            data = data_queue.get()
            try:
                symbol = data.get("symbol", "")
                if symbol not in sym_seq_bufs:
                    sym_seq_bufs[symbol] = collections.deque(maxlen=seq_len)
                    traders[symbol]      = PaperTrader(MODEL_NAME, symbol)

                seq_buf = sym_seq_bufs[symbol]
                price   = data["current_price"]
                feat    = np.array([data[f] for f in FEATURE_COLS], dtype=np.float32)
                feat_scaled = scaler.transform(feat.reshape(1, -1))[0]
                seq_buf.append(feat_scaled)

                sym_tag = f"[{symbol}] " if symbol else ""
                buf_len = len(seq_buf)
                if buf_len < seq_len:
                    _post(status_queue, "PYTORCH", "WAITING",
                          f"{sym_tag}Warming up ({buf_len}/{seq_len})")
                    time.sleep(0.005)
                    continue

                _post(status_queue, "PYTORCH", "INFERRING",
                      f"{sym_tag}LSTM inference @ ${price:,.2f}")
                seq_arr = np.stack(list(seq_buf), axis=0)
                tensor  = torch.from_numpy(seq_arr).unsqueeze(0).to(device)

                with torch.no_grad():
                    prediction = float(model(tensor).item())

                threshold = compute_signal_threshold(data.get("atr_14", 0.0), price)
                if prediction > threshold:
                    signal = 1
                    _post(status_queue, "PYTORCH", "TRADING",
                          f"{sym_tag}BUY  pred={prediction:+.6f}  ${price:,.2f}")
                elif prediction < -threshold:
                    signal = -1
                    _post(status_queue, "PYTORCH", "TRADING",
                          f"{sym_tag}SELL  pred={prediction:+.6f}  ${price:,.2f}")
                else:
                    signal = 0
                    _post(status_queue, "PYTORCH", "OK",
                          f"{sym_tag}HOLD  pred={prediction:+.6f}  ${price:,.2f}")

                traders[symbol].on_signal(signal, price, prediction)

            except Exception as e:
                print(f"[{MODEL_NAME}] Inference error: {e}")
                _post(status_queue, "PYTORCH", "ERROR", str(e))
        else:
            if not idle_logged:
                _post(status_queue, "PYTORCH", "WAITING", "Waiting for tick data...")
                idle_logged = True

        time.sleep(0.005)
