"""
Training pipeline — apples-to-apples scientific design:

  * Folds are computed ONCE via TimeSeriesSplit, then saved to data/cv_folds.pkl.
  * ALL three models receive the SAME fold indices into the SAME aligned dataset,
    so they are evaluated at identical timestamps.
  * sklearn / XGBoost receive X_single[i] = features at timestep i.
  * PyTorch LSTM receives X_seq[i]    = features at timesteps [i-seq_len .. i].
  * Both predict y_aligned[i] = next-candle log return at timestep i.
  * Per-fold metrics (MSE, MAE, direction accuracy) are written to SQLite so
    compare_models.py can read them side-by-side.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_security import sign_artifact

try:
    import lightgbm as _lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostRegressor as _CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    from prophet import Prophet as _Prophet
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False

from config import CONFIG, FEATURE_COLS, TARGET_COL, is_demo_mode
from exchanges import get_exchange
from features import calculate_features
from database import log_training_result, init_db, get_cached_candles, store_candles
from activity import (
    get_tracker, DOWNLOADING, SAVING, LOADING, TRAINING, COMPUTING, 
    IDLE, ERROR, DONE
)

_PATHS  = CONFIG["PATHS"]
_TR     = CONFIG["TRAINING"]
_ACTIVE = CONFIG["ACTIVE_MODELS"]
_PROD   = CONFIG["COINBASE"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _candles_to_df(candles: list[dict]) -> pd.DataFrame:
    rows = [{
        "timestamp": int(c["start"]),
        "open":   float(c["open"]),
        "high":   float(c["high"]),
        "low":    float(c["low"]),
        "close":  float(c["close"]),
        "volume": float(c["volume"]),
    } for c in candles]
    df = pd.DataFrame(rows).sort_values("timestamp").set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="s", utc=True)
    return df


def load_or_fetch_candles() -> pd.DataFrame:
    """Single-coin legacy loader (BTC only). Used as fallback."""
    tracker  = get_tracker()
    csv_path = _PATHS["HISTORICAL_CSV"]
    if os.path.exists(csv_path):
        print(f"[TRAIN] Loading cached candles from {csv_path}")
        tracker.update("TRAINING", LOADING, f"Loading cached candles from CSV...")
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    exchange = get_exchange()
    lookback = _TR["LOOKBACK_DAYS"] * 24
    tracker.update("TRAINING", DOWNLOADING,
                   f"Fetching {lookback}h of candles from {exchange.name}...")
    raw = exchange.fetch_candles(_PROD["PRODUCT_ID"], lookback_hours=lookback)
    df  = _candles_to_df(raw)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    tracker.update("TRAINING", SAVING, f"Saving {len(df)} candles to CSV...")
    df.to_csv(csv_path)
    print(f"[TRAIN] Candles cached -> {csv_path}")
    return df


def _load_training_data() -> pd.DataFrame:
    """
    Load and featurize candles for all training-tier coins.
    Uses the SQLite candle_cache when data is available; fetches from the
    exchange otherwise.  Concatenates feature rows across all coins so every
    model trains on a richer, multi-market dataset.
    """
    from coin_manager import get_coins_for_tier
    from config import CONFIG as _CFG
    tracker  = get_tracker()
    exchange = get_exchange()
    # shuffle=True randomises coin order so the concatenated dataset doesn't
    # cluster by coin; deduplication is guaranteed inside get_coins_for_tier.
    tier    = _CFG["COINS"]["TRAINING_TIER"]
    symbols = get_coins_for_tier(tier, exchange.name, shuffle=True)
    lookback = _TR["LOOKBACK_DAYS"] * 24

    # Single-coin + legacy CSV fast-path
    if len(symbols) == 1 and os.path.exists(_PATHS["HISTORICAL_CSV"]):
        print(f"[TRAIN] Single-coin mode — using legacy CSV for {symbols[0]}")
        tracker.update("TRAINING", LOADING, "Loading cached CSV candles...")
        return load_or_fetch_candles()

    all_dfs: list[pd.DataFrame] = []

    for idx, symbol in enumerate(symbols):
        tracker.update(
            "TRAINING", DOWNLOADING if idx == 0 else COMPUTING,
            f"[{idx+1}/{len(symbols)}] Loading {symbol}...",
        )
        # Try SQLite cache first (populated by DataManager)
        cached = get_cached_candles(exchange.name, symbol, hours=lookback)
        if len(cached) >= 200:
            print(f"[TRAIN] {symbol}: {len(cached):,} cached candles")
            raw = cached
        else:
            print(f"[TRAIN] {symbol}: fetching {lookback}h from {exchange.name}...")
            tracker.update("TRAINING", DOWNLOADING,
                           f"Fetching {symbol} ({lookback}h)...")
            try:
                raw = exchange.fetch_candles(symbol, lookback_hours=lookback)
                if raw:
                    store_candles(raw, exchange.name, symbol)
            except Exception as e:
                print(f"[TRAIN] WARNING: Could not fetch {symbol}: {e} — skipping")
                continue

        if not raw:
            print(f"[TRAIN] WARNING: No data for {symbol} — skipping")
            continue

        df_coin = _candles_to_df(raw)
        df_feat = calculate_features(df_coin).dropna(subset=FEATURE_COLS)
        all_dfs.append(df_feat)
        print(f"[TRAIN] {symbol}: {len(df_feat):,} feature rows ready")

    if not all_dfs:
        # Ultimate fallback: legacy CSV / single-coin fetch
        print("[TRAIN] WARNING: No cached multi-coin data — falling back to single-coin CSV")
        return load_or_fetch_candles()

    if len(all_dfs) == 1:
        return all_dfs[0]

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"[TRAIN] Multi-coin dataset ({len(symbols)} coins): {len(combined):,} rows total")
    return combined


# ---------------------------------------------------------------------------
# Dataset alignment
# ---------------------------------------------------------------------------

def prepare_dataset(X_scaled: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Build sequence-aligned arrays so all models share identical timesteps.

    Returns:
      X_single  (N, n_features)       -- single-timestep features
      X_seq     (N, seq_len, n_features) -- rolling sequence features
      y_aligned (N,)                  -- targets aligned to same N rows

    Index i in all three arrays corresponds to the same wall-clock candle,
    meaning every model is trained and tested on identical timestamps.
    """
    n       = len(X_scaled)
    n_valid = n - seq_len          # first valid prediction is at row seq_len

    X_single  = X_scaled[seq_len:]                                          # (N, F)
    X_seq     = np.stack([X_scaled[i:i + seq_len] for i in range(n_valid)], axis=0)  # (N, L, F)
    y_aligned = y[seq_len:]                                                 # (N,)

    assert len(X_single) == len(X_seq) == len(y_aligned), "Alignment mismatch"
    return X_single.astype(np.float32), X_seq.astype(np.float32), y_aligned.astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch LSTM model definition (imported by pytorch_worker.py too)
# ---------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions where sign matches target (up/down agreement)."""
    correct = np.sign(y_pred) == np.sign(y_true)
    return float(correct.mean())


def _fold_metrics(y_true, y_pred):
    mse      = float(np.mean((y_true - y_pred) ** 2))
    mae      = float(np.mean(np.abs(y_true - y_pred)))
    dir_acc  = _direction_accuracy(y_true, y_pred)
    return mse, mae, dir_acc


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_all_models():
    print("\n" + "=" * 65)
    print("  PROJECT CHALLENGER -- TRAINING PIPELINE")
    print("  Apples-to-apples: all models share identical CV fold indices")
    print("=" * 65)

    tracker = get_tracker()
    tracker.update("TRAINING", TRAINING, "Initialising training pipeline...")

    init_db()

    # 1. Load raw candles -> feature matrix (multi-coin concatenated)
    df = _load_training_data()
    df = df.dropna(subset=FEATURE_COLS)
    print(f"\n[TRAIN] Feature matrix: {len(df):,} rows x {len(FEATURE_COLS)} features")

    X_raw = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df[TARGET_COL].values.astype(np.float32)

    # 2. Fit scaler on all data (shared across every model and live inference)
    tracker.update("TRAINING", COMPUTING, "Fitting feature scaler...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    os.makedirs(os.path.dirname(_PATHS["SCALER"]), exist_ok=True)
    tracker.update("TRAINING", SAVING, f"Saving scaler -> {_PATHS['SCALER']}")
    joblib.dump(scaler, _PATHS["SCALER"])
    sign_artifact(_PATHS["SCALER"])
    print(f"[TRAIN] Scaler saved -> {_PATHS['SCALER']}")

    # 3. Build sequence-aligned dataset (same timestep indices for all models)
    seq_len              = _TR["SEQUENCE_LENGTH"]
    X_single, X_seq, y  = prepare_dataset(X_scaled, y_raw, seq_len)
    print(f"[TRAIN] Aligned dataset: {len(y):,} rows (seq_len={seq_len})")

    # 4. Compute folds ONCE and save — identical for all three models
    tracker.update("TRAINING", COMPUTING,
                   f"Computing {_TR['N_SPLITS']}-fold TimeSeriesSplit...")
    tscv  = TimeSeriesSplit(n_splits=_TR["N_SPLITS"])
    folds = list(tscv.split(X_single))      # list of (train_idx, test_idx) arrays
    joblib.dump(folds, _PATHS["CV_FOLDS"])
    sign_artifact(_PATHS["CV_FOLDS"])
    print(f"[TRAIN] CV folds ({_TR['N_SPLITS']}-split) saved -> {_PATHS['CV_FOLDS']}")

    _print_fold_summary(folds)

    # 5. Train each active model using the shared folds
    # ── sklearn-family ────────────────────────────────────────────────────────
    if _ACTIVE.get("SKLEARN_LINEAR"):
        if os.path.exists(_PATHS["SKLEARN_MODEL"]):
            print("\n[TRAIN] sklearn_model.pkl exists -- skipping. Delete to retrain.")
        else:
            _train_sklearn(X_single, y, folds)

    if _ACTIVE.get("XGBOOST_TREE"):
        if os.path.exists(_PATHS["XGBOOST_MODEL"]):
            print("[TRAIN] xgboost_model.pkl exists -- skipping. Delete to retrain.")
        else:
            _train_xgboost(X_single, y, folds)

    if _ACTIVE.get("LGBM_TREE"):
        if os.path.exists(_PATHS["LGBM_MODEL"]):
            print("[TRAIN] lgbm_model.pkl exists -- skipping. Delete to retrain.")
        else:
            _train_lgbm(X_single, y, folds)

    if _ACTIVE.get("CATBOOST_TREE"):
        if os.path.exists(_PATHS["CATBOOST_MODEL"]):
            print("[TRAIN] catboost_model.pkl exists -- skipping. Delete to retrain.")
        else:
            _train_catboost(X_single, y, folds)

    if _ACTIVE.get("RF_TREE"):
        if os.path.exists(_PATHS["RF_MODEL"]):
            print("[TRAIN] rf_model.pkl exists -- skipping. Delete to retrain.")
        else:
            _train_rf(X_single, y, folds)

    if _ACTIVE.get("ET_TREE"):
        if os.path.exists(_PATHS["ET_MODEL"]):
            print("[TRAIN] et_model.pkl exists -- skipping. Delete to retrain.")
        else:
            _train_et(X_single, y, folds)

    if _ACTIVE.get("ELASTIC_LINEAR"):
        if os.path.exists(_PATHS["ELASTIC_MODEL"]):
            print("[TRAIN] elastic_model.pkl exists -- skipping. Delete to retrain.")
        else:
            _train_elastic(X_single, y, folds)

    if _ACTIVE.get("SVR_KERNEL"):
        if os.path.exists(_PATHS["SVR_MODEL"]):
            print("[TRAIN] svr_model.pkl exists -- skipping. Delete to retrain.")
        else:
            _train_svr(X_single, y, folds)

    if _ACTIVE.get("MLP_NN"):
        if os.path.exists(_PATHS["MLP_MODEL"]):
            print("[TRAIN] mlp_model.pkl exists -- skipping. Delete to retrain.")
        else:
            _train_mlp(X_single, y, folds)

    # ── Time-series models ────────────────────────────────────────────────────
    if _ACTIVE.get("ARIMA_STATS"):
        if os.path.exists(_PATHS["ARIMA_MODEL"]):
            print("[TRAIN] arima_model.pkl exists -- skipping. Delete to retrain.")
        else:
            _train_arima(y, folds)

    if _ACTIVE.get("PROPHET_FB"):
        if os.path.exists(_PATHS["PROPHET_MODEL"]):
            print("[TRAIN] prophet_model.pkl exists -- skipping. Delete to retrain.")
        else:
            _train_prophet(y, folds)

    # ── Sequential / deep ─────────────────────────────────────────────────────
    if _ACTIVE.get("PYTORCH_LSTM"):
        if os.path.exists(_PATHS["PYTORCH_MODEL"]):
            print("[TRAIN] pytorch_model.pt exists -- skipping. Delete to retrain.")
        else:
            _train_pytorch(X_seq, y, folds)

    tracker.update("TRAINING", DONE, "All models trained and saved.")
    print("\n[TRAIN] All models trained and saved.")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------

def _print_fold_summary(folds):
    print(f"\n  {'Fold':<6} {'Train rows':>12} {'Test rows':>10}")
    print("  " + "-" * 30)
    for i, (tr, te) in enumerate(folds):
        print(f"  {i+1:<6} {len(tr):>12,} {len(te):>10,}")
    print()


def _train_sklearn(X, y, folds):
    tracker = get_tracker()
    n_folds = len(folds)
    alpha   = _TR["RIDGE_ALPHA"]
    print(f"\n[TRAIN] -- Scikit-Learn Ridge  (alpha={alpha}) --")
    for fold, (tr, te) in enumerate(folds):
        tracker.update("SKLEARN", TRAINING,
                       f"Ridge fold {fold+1}/{n_folds}  ({len(tr):,} train rows)  alpha={alpha}")
        m     = Ridge(alpha=alpha).fit(X[tr], y[tr])
        preds = m.predict(X[te])
        mse, mae, dir_acc = _fold_metrics(y[te], preds)
        log_training_result("SKLEARN_LINEAR", fold + 1, len(tr), len(te), mse, mae, dir_acc)
        print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")

    tracker.update("SKLEARN", SAVING, "Fitting final Ridge on all data & saving...")
    final = Ridge(alpha=alpha).fit(X, y)
    joblib.dump(final, _PATHS["SKLEARN_MODEL"])
    sign_artifact(_PATHS["SKLEARN_MODEL"])
    tracker.update("SKLEARN", DONE, "Ridge model saved.")
    print(f"  Saved -> {_PATHS['SKLEARN_MODEL']}")


def _xgb_device() -> str:
    """
    Return the XGBoost device string for the current hardware.
    XGBoost 2.0+ accepts 'cuda' or 'cpu'.
    We piggyback on PyTorch's CUDA detection rather than calling nvidia-smi.
    """
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _train_xgboost(X, y, folds):
    tracker    = get_tracker()
    n_folds    = len(folds)
    xgb_dev    = _xgb_device()
    print(f"\n[TRAIN] -- XGBoost  (device={xgb_dev}) --")
    tracker.update("XGBOOST", TRAINING,
                   f"Starting XGBoost training on {xgb_dev}...")

    # Hyperparameters — read from CONFIG so the web/TUI Settings can tune them
    _xgb_params = dict(
        n_estimators   = _TR["XGB_N_ESTIMATORS"],
        max_depth      = _TR["XGB_MAX_DEPTH"],
        learning_rate  = _TR["XGB_LR"],
        subsample      = _TR["XGB_SUBSAMPLE"],
        colsample_bytree = _TR["XGB_COLSAMPLE"],
        verbosity      = 0,
        device         = xgb_dev,
    )
    print(f"  n_estimators={_TR['XGB_N_ESTIMATORS']}  max_depth={_TR['XGB_MAX_DEPTH']}  "
          f"lr={_TR['XGB_LR']}  subsample={_TR['XGB_SUBSAMPLE']}  "
          f"colsample={_TR['XGB_COLSAMPLE']}")

    for fold, (tr, te) in enumerate(folds):
        tracker.update("XGBOOST", TRAINING,
                       f"XGBoost fold {fold+1}/{n_folds}  ({len(tr):,} train rows)  [{xgb_dev}]")
        m = xgb.XGBRegressor(**_xgb_params)
        m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        preds = m.predict(X[te])
        mse, mae, dir_acc = _fold_metrics(y[te], preds)
        log_training_result("XGBOOST_TREE", fold + 1, len(tr), len(te), mse, mae, dir_acc)
        print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")

    tracker.update("XGBOOST", SAVING, "Fitting final XGBoost on all data & saving...")
    final = xgb.XGBRegressor(**_xgb_params).fit(X, y)
    joblib.dump(final, _PATHS["XGBOOST_MODEL"])
    sign_artifact(_PATHS["XGBOOST_MODEL"])
    tracker.update("XGBOOST", DONE, f"XGBoost model saved  [{xgb_dev}].")
    print(f"  Saved -> {_PATHS['XGBOOST_MODEL']}")


def _train_pytorch(X_seq, y, folds):
    tracker = get_tracker()
    n_folds = len(folds)
    print("\n[TRAIN] -- PyTorch LSTM --")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feat = X_seq.shape[2]
    model_cfg = {
        "input_size":  n_feat,
        "hidden_size": _TR["LSTM_HIDDEN_SIZE"],
        "num_layers":  _TR["LSTM_LAYERS"],
        "dropout":     _TR["LSTM_DROPOUT"],
    }
    print(f"  Device: {device}  |  arch: LSTM({n_feat}->{_TR['LSTM_HIDDEN_SIZE']}x{_TR['LSTM_LAYERS']})")

    for fold, (tr, te) in enumerate(folds):
        ds     = TensorDataset(torch.from_numpy(X_seq[tr]), torch.from_numpy(y[tr]))
        loader = DataLoader(ds, batch_size=_TR["LSTM_BATCH_SIZE"], shuffle=False)
        net    = LSTMPredictor(**model_cfg).to(device)
        opt    = torch.optim.Adam(net.parameters(), lr=_TR["LSTM_LR"])
        loss_fn = nn.MSELoss()

        net.train()
        for epoch in range(_TR["LSTM_EPOCHS"]):
            tracker.update("PYTORCH", TRAINING,
                           f"LSTM fold {fold+1}/{n_folds}  epoch {epoch+1}/{_TR['LSTM_EPOCHS']}  ({len(tr):,} rows)")
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()

        net.eval()
        with torch.no_grad():
            preds = net(torch.from_numpy(X_seq[te]).to(device)).cpu().numpy()
        mse, mae, dir_acc = _fold_metrics(y[te], preds)
        log_training_result("PYTORCH_LSTM", fold + 1, len(tr), len(te), mse, mae, dir_acc)
        print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")

    # Final model trained on all data
    tracker.update("PYTORCH", TRAINING,
                   f"Fitting final LSTM on all {len(y):,} rows ({_TR['LSTM_EPOCHS']} epochs)...")
    ds     = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=_TR["LSTM_BATCH_SIZE"], shuffle=False)
    final  = LSTMPredictor(**model_cfg).to(device)
    opt    = torch.optim.Adam(final.parameters(), lr=_TR["LSTM_LR"])
    loss_fn = nn.MSELoss()
    final.train()
    for epoch in range(_TR["LSTM_EPOCHS"]):
        tracker.update("PYTORCH", TRAINING,
                       f"Final LSTM epoch {epoch+1}/{_TR['LSTM_EPOCHS']}...")
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(final(xb), yb).backward()
            opt.step()
    final.eval()

    tracker.update("PYTORCH", SAVING, f"Saving LSTM -> {_PATHS['PYTORCH_MODEL']}")
    torch.save({"model_state_dict": final.state_dict(), "model_cfg": model_cfg},
               _PATHS["PYTORCH_MODEL"])
    sign_artifact(_PATHS["PYTORCH_MODEL"])
    tracker.update("PYTORCH", DONE, "LSTM model saved.")
    print(f"  Saved -> {_PATHS['PYTORCH_MODEL']}")


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def _train_lgbm(X, y, folds):
    tracker = get_tracker()
    if not _HAS_LGBM:
        print("[TRAIN] LightGBM not installed — skip (pip install lightgbm)")
        tracker.update("LGBM", DONE, "Skipped — lightgbm not installed")
        return
    n_folds  = len(folds)
    boosting = _TR.get("LGBM_BOOSTING", "gbdt")
    print(f"\n[TRAIN] -- LightGBM  (boosting={boosting}) --")
    params = dict(
        n_estimators   = _TR.get("LGBM_N_ESTIMATORS", 200),
        max_depth      = _TR.get("LGBM_MAX_DEPTH", -1),
        learning_rate  = _TR.get("LGBM_LR", 0.05),
        subsample      = _TR.get("LGBM_SUBSAMPLE", 0.8),
        colsample_bytree = _TR.get("LGBM_COLSAMPLE", 0.8),
        boosting_type  = boosting,
        verbose        = -1,
    )
    for fold, (tr, te) in enumerate(folds):
        tracker.update("LGBM", TRAINING,
                       f"LightGBM fold {fold+1}/{n_folds}  ({len(tr):,} train rows)  [{boosting}]")
        m = _lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])],
              callbacks=[_lgb.early_stopping(20, verbose=False), _lgb.log_evaluation(-1)])
        preds = m.predict(X[te])
        mse, mae, dir_acc = _fold_metrics(y[te], preds)
        log_training_result("LGBM_TREE", fold + 1, len(tr), len(te), mse, mae, dir_acc)
        print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")
    tracker.update("LGBM", SAVING, "Fitting final LightGBM on all data & saving...")
    final = _lgb.LGBMRegressor(**params).fit(X, y)
    joblib.dump(final, _PATHS["LGBM_MODEL"])
    sign_artifact(_PATHS["LGBM_MODEL"])
    tracker.update("LGBM", DONE, "LightGBM model saved.")
    print(f"  Saved -> {_PATHS['LGBM_MODEL']}")


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def _train_catboost(X, y, folds):
    tracker = get_tracker()
    if not _HAS_CATBOOST:
        print("[TRAIN] CatBoost not installed — skip (pip install catboost)")
        tracker.update("CATBOOST", DONE, "Skipped — catboost not installed")
        return
    n_folds = len(folds)
    loss    = _TR.get("CATBOOST_LOSS", "RMSE")
    print(f"\n[TRAIN] -- CatBoost  (loss={loss}) --")
    params = dict(
        iterations   = _TR.get("CATBOOST_ITERATIONS", 200),
        depth        = _TR.get("CATBOOST_DEPTH", 6),
        learning_rate = _TR.get("CATBOOST_LR", 0.05),
        loss_function = loss,
        verbose       = False,
        allow_writing_files = False,
    )
    for fold, (tr, te) in enumerate(folds):
        tracker.update("CATBOOST", TRAINING,
                       f"CatBoost fold {fold+1}/{n_folds}  ({len(tr):,} train rows)  [{loss}]")
        m = _CatBoostRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=(X[te], y[te]), early_stopping_rounds=20)
        preds = m.predict(X[te])
        mse, mae, dir_acc = _fold_metrics(y[te], preds)
        log_training_result("CATBOOST_TREE", fold + 1, len(tr), len(te), mse, mae, dir_acc)
        print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")
    tracker.update("CATBOOST", SAVING, "Fitting final CatBoost on all data & saving...")
    final = _CatBoostRegressor(**params).fit(X, y)
    joblib.dump(final, _PATHS["CATBOOST_MODEL"])
    sign_artifact(_PATHS["CATBOOST_MODEL"])
    tracker.update("CATBOOST", DONE, "CatBoost model saved.")
    print(f"  Saved -> {_PATHS['CATBOOST_MODEL']}")


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def _train_rf(X, y, folds):
    tracker = get_tracker()
    n_folds = len(folds)
    n_est   = _TR.get("RF_N_ESTIMATORS", 200)
    depth   = _TR.get("RF_MAX_DEPTH", 10)
    print(f"\n[TRAIN] -- Random Forest  (n_estimators={n_est}, max_depth={depth}) --")
    for fold, (tr, te) in enumerate(folds):
        tracker.update("RF", TRAINING,
                       f"RandomForest fold {fold+1}/{n_folds}  ({len(tr):,} train rows)")
        m = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                  n_jobs=-1, random_state=42)
        m.fit(X[tr], y[tr])
        preds = m.predict(X[te])
        mse, mae, dir_acc = _fold_metrics(y[te], preds)
        log_training_result("RF_TREE", fold + 1, len(tr), len(te), mse, mae, dir_acc)
        print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")
    tracker.update("RF", SAVING, "Fitting final RandomForest on all data & saving...")
    final = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                   n_jobs=-1, random_state=42).fit(X, y)
    joblib.dump(final, _PATHS["RF_MODEL"])
    sign_artifact(_PATHS["RF_MODEL"])
    tracker.update("RF", DONE, "RandomForest model saved.")
    print(f"  Saved -> {_PATHS['RF_MODEL']}")


# ---------------------------------------------------------------------------
# Extra Trees
# ---------------------------------------------------------------------------

def _train_et(X, y, folds):
    tracker = get_tracker()
    n_folds = len(folds)
    n_est   = _TR.get("ET_N_ESTIMATORS", 200)
    depth   = _TR.get("ET_MAX_DEPTH", 10)
    print(f"\n[TRAIN] -- Extra Trees  (n_estimators={n_est}, max_depth={depth}) --")
    for fold, (tr, te) in enumerate(folds):
        tracker.update("ET", TRAINING,
                       f"ExtraTrees fold {fold+1}/{n_folds}  ({len(tr):,} train rows)")
        m = ExtraTreesRegressor(n_estimators=n_est, max_depth=depth,
                                n_jobs=-1, random_state=42)
        m.fit(X[tr], y[tr])
        preds = m.predict(X[te])
        mse, mae, dir_acc = _fold_metrics(y[te], preds)
        log_training_result("ET_TREE", fold + 1, len(tr), len(te), mse, mae, dir_acc)
        print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")
    tracker.update("ET", SAVING, "Fitting final ExtraTrees on all data & saving...")
    final = ExtraTreesRegressor(n_estimators=n_est, max_depth=depth,
                                 n_jobs=-1, random_state=42).fit(X, y)
    joblib.dump(final, _PATHS["ET_MODEL"])
    sign_artifact(_PATHS["ET_MODEL"])
    tracker.update("ET", DONE, "ExtraTrees model saved.")
    print(f"  Saved -> {_PATHS['ET_MODEL']}")


# ---------------------------------------------------------------------------
# ElasticNet
# ---------------------------------------------------------------------------

def _train_elastic(X, y, folds):
    tracker  = get_tracker()
    n_folds  = len(folds)
    alpha    = _TR.get("ELASTIC_ALPHA", 0.1)
    l1_ratio = _TR.get("ELASTIC_L1_RATIO", 0.5)
    print(f"\n[TRAIN] -- ElasticNet  (alpha={alpha}, l1_ratio={l1_ratio}) --")
    for fold, (tr, te) in enumerate(folds):
        tracker.update("ELASTIC", TRAINING,
                       f"ElasticNet fold {fold+1}/{n_folds}  ({len(tr):,} train rows)  α={alpha}")
        m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        m.fit(X[tr], y[tr])
        preds = m.predict(X[te])
        mse, mae, dir_acc = _fold_metrics(y[te], preds)
        log_training_result("ELASTIC_LINEAR", fold + 1, len(tr), len(te), mse, mae, dir_acc)
        print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")
    tracker.update("ELASTIC", SAVING, "Fitting final ElasticNet on all data & saving...")
    final = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000).fit(X, y)
    joblib.dump(final, _PATHS["ELASTIC_MODEL"])
    sign_artifact(_PATHS["ELASTIC_MODEL"])
    tracker.update("ELASTIC", DONE, "ElasticNet model saved.")
    print(f"  Saved -> {_PATHS['ELASTIC_MODEL']}")


# ---------------------------------------------------------------------------
# SVR
# ---------------------------------------------------------------------------

def _train_svr(X, y, folds):
    tracker  = get_tracker()
    n_folds  = len(folds)
    C        = _TR.get("SVR_C", 1.0)
    eps      = _TR.get("SVR_EPSILON", 0.1)
    kernel   = _TR.get("SVR_KERNEL_TYPE", "rbf")
    # SVR is O(n²–n³); subsample to ≤ 5000 rows to keep training time reasonable
    MAX_ROWS = 5000
    print(f"\n[TRAIN] -- SVR  (kernel={kernel}, C={C}, ε={eps}) --")
    if len(y) > MAX_ROWS:
        print(f"  SVR: capping training rows to {MAX_ROWS:,} (full dataset has {len(y):,})")
    for fold, (tr, te) in enumerate(folds):
        tracker.update("SVR", TRAINING,
                       f"SVR fold {fold+1}/{n_folds}  ({min(len(tr), MAX_ROWS):,} train rows)  [{kernel}]")
        tr_use = tr[:MAX_ROWS] if len(tr) > MAX_ROWS else tr
        m = SVR(kernel=kernel, C=C, epsilon=eps)
        m.fit(X[tr_use], y[tr_use])
        preds = m.predict(X[te])
        mse, mae, dir_acc = _fold_metrics(y[te], preds)
        log_training_result("SVR_KERNEL", fold + 1, len(tr_use), len(te), mse, mae, dir_acc)
        print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")
    tracker.update("SVR", SAVING, "Fitting final SVR on all data & saving...")
    X_use = X[:MAX_ROWS] if len(X) > MAX_ROWS else X
    y_use = y[:MAX_ROWS] if len(y) > MAX_ROWS else y
    final = SVR(kernel=kernel, C=C, epsilon=eps).fit(X_use, y_use)
    joblib.dump(final, _PATHS["SVR_MODEL"])
    sign_artifact(_PATHS["SVR_MODEL"])
    tracker.update("SVR", DONE, f"SVR model saved  [{kernel}].")
    print(f"  Saved -> {_PATHS['SVR_MODEL']}")


# ---------------------------------------------------------------------------
# MLP Neural Network (sklearn)
# ---------------------------------------------------------------------------

def _parse_hidden_layers(spec) -> tuple:
    """Parse '100,50' or [100, 50] into a tuple for MLPRegressor."""
    if isinstance(spec, (list, tuple)):
        return tuple(int(x) for x in spec)
    return tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())


def _train_mlp(X, y, folds):
    tracker = get_tracker()
    n_folds = len(folds)
    hidden  = _parse_hidden_layers(_TR.get("MLP_HIDDEN_LAYERS", "100,50"))
    max_it  = _TR.get("MLP_MAX_ITER", 500)
    lr      = _TR.get("MLP_LR", 0.001)
    print(f"\n[TRAIN] -- MLP Neural Net  (hidden={hidden}, lr={lr}) --")
    for fold, (tr, te) in enumerate(folds):
        tracker.update("MLP", TRAINING,
                       f"MLP fold {fold+1}/{n_folds}  ({len(tr):,} train rows)  hidden={hidden}")
        m = MLPRegressor(hidden_layer_sizes=hidden, learning_rate_init=lr,
                         max_iter=max_it, random_state=42, early_stopping=True,
                         n_iter_no_change=20, verbose=False)
        m.fit(X[tr], y[tr])
        preds = m.predict(X[te])
        mse, mae, dir_acc = _fold_metrics(y[te], preds)
        log_training_result("MLP_NN", fold + 1, len(tr), len(te), mse, mae, dir_acc)
        print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")
    tracker.update("MLP", SAVING, "Fitting final MLP on all data & saving...")
    final = MLPRegressor(hidden_layer_sizes=hidden, learning_rate_init=lr,
                          max_iter=max_it, random_state=42, verbose=False).fit(X, y)
    joblib.dump(final, _PATHS["MLP_MODEL"])
    sign_artifact(_PATHS["MLP_MODEL"])
    tracker.update("MLP", DONE, "MLP model saved.")
    print(f"  Saved -> {_PATHS['MLP_MODEL']}")


# ---------------------------------------------------------------------------
# ARIMA (statsmodels)
# ---------------------------------------------------------------------------

def _train_arima(y, folds):
    """
    Fit ARIMA on the target time series (log returns).
    The saved model is used for backtesting; live inference uses a rolling refit
    in the arima_worker.
    """
    tracker = get_tracker()
    if not _HAS_STATSMODELS:
        print("[TRAIN] statsmodels not installed — skip ARIMA (pip install statsmodels)")
        tracker.update("ARIMA", DONE, "Skipped — statsmodels not installed")
        return
    p, d, q  = _TR.get("ARIMA_P", 2), _TR.get("ARIMA_D", 0), _TR.get("ARIMA_Q", 2)
    order    = (p, d, q)
    n_folds  = len(folds)
    print(f"\n[TRAIN] -- ARIMA({p},{d},{q}) --")
    for fold, (tr, te) in enumerate(folds):
        tracker.update("ARIMA", TRAINING,
                       f"ARIMA fold {fold+1}/{n_folds}  ({len(tr):,} train rows)")
        try:
            res   = _ARIMA(y[tr], order=order).fit()
            start = len(tr)
            end   = len(tr) + len(te) - 1
            preds = res.predict(start=start, end=end)
            if len(preds) < len(te):
                preds = np.pad(preds, (0, len(te) - len(preds)))
            mse, mae, dir_acc = _fold_metrics(y[te], preds)
            log_training_result("ARIMA_STATS", fold + 1, len(tr), len(te), mse, mae, dir_acc)
            print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")
        except Exception as e:
            print(f"  Fold {fold+1}  ARIMA error: {e}")
    tracker.update("ARIMA", SAVING, "Fitting final ARIMA on all data & saving...")
    try:
        final = _ARIMA(y, order=order).fit()
        joblib.dump({"order": order, "params": final.params.tolist()}, _PATHS["ARIMA_MODEL"])
        sign_artifact(_PATHS["ARIMA_MODEL"])
        tracker.update("ARIMA", DONE, "ARIMA model saved.")
        print(f"  Saved -> {_PATHS['ARIMA_MODEL']}")
    except Exception as e:
        print(f"  ARIMA final fit error: {e}")
        tracker.update("ARIMA", DONE, f"ARIMA save error: {e}")


# ---------------------------------------------------------------------------
# Prophet (Meta/Facebook)
# ---------------------------------------------------------------------------

def _train_prophet(y, folds):
    """
    Fit a Prophet model on the target log-return series for CV scoring.
    Saved for backtest use; live inference uses rolling refit in prophet_worker.
    """
    tracker = get_tracker()
    if not _HAS_PROPHET:
        print("[TRAIN] prophet not installed — skip (pip install prophet)")
        tracker.update("PROPHET", DONE, "Skipped — prophet not installed")
        return
    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    n_folds = len(folds)
    print(f"\n[TRAIN] -- Prophet --")
    for fold, (tr, te) in enumerate(folds):
        tracker.update("PROPHET", TRAINING,
                       f"Prophet fold {fold+1}/{n_folds}  ({len(tr):,} train rows)")
        try:
            # Build a simple datetime index (no real timestamps — use integer rows as proxy)
            dates = pd.date_range("2020-01-01", periods=len(tr), freq="15min")
            df_p  = pd.DataFrame({"ds": dates, "y": y[tr]})
            m = _Prophet(
                daily_seasonality   = _TR.get("PROPHET_SEASONALITY", False),
                weekly_seasonality  = False,
                yearly_seasonality  = False,
                n_changepoints      = _TR.get("PROPHET_CHANGEPOINTS", 25),
                uncertainty_samples = 0,
            )
            m.fit(df_p)
            future = pd.DataFrame({"ds": pd.date_range(
                dates[-1] + pd.Timedelta("15min"), periods=len(te), freq="15min")})
            fc     = m.predict(future)["yhat"].values
            mse, mae, dir_acc = _fold_metrics(y[te], fc)
            log_training_result("PROPHET_FB", fold + 1, len(tr), len(te), mse, mae, dir_acc)
            print(f"  Fold {fold+1}  MSE={mse:.8f}  MAE={mae:.6f}  DirAcc={dir_acc:.3f}")
        except Exception as e:
            print(f"  Fold {fold+1}  Prophet error: {e}")

    tracker.update("PROPHET", SAVING, "Fitting final Prophet on all data & saving...")
    try:
        dates = pd.date_range("2020-01-01", periods=len(y), freq="15min")
        df_p  = pd.DataFrame({"ds": dates, "y": y})
        final = _Prophet(
            daily_seasonality   = _TR.get("PROPHET_SEASONALITY", False),
            weekly_seasonality  = False,
            yearly_seasonality  = False,
            n_changepoints      = _TR.get("PROPHET_CHANGEPOINTS", 25),
            uncertainty_samples = 0,
        )
        final.fit(df_p)
        joblib.dump(final, _PATHS["PROPHET_MODEL"])
        sign_artifact(_PATHS["PROPHET_MODEL"])
        tracker.update("PROPHET", DONE, "Prophet model saved.")
        print(f"  Saved -> {_PATHS['PROPHET_MODEL']}")
    except Exception as e:
        print(f"  Prophet final fit error: {e}")
        tracker.update("PROPHET", DONE, f"Prophet save error: {e}")


if __name__ == "__main__":
    train_all_models()
