"""
Project Challenger configuration.

Copyright (c) 2026 Rev. J. Money.
Non-commercial learning/research use only. See LICENSE and NOTICE.
"""
import os as _os
from secrets_store import load_secrets as _load_secrets
from secrets_store import update_secrets as _update_secrets
_ROOT = _os.path.dirname(_os.path.abspath(__file__))

def _p(rel: str) -> str:
    """Convert a project-relative path to absolute."""
    return _os.path.join(_ROOT, rel)

CONFIG = {
    # ── Active exchange ────────────────────────────────────────────────────────
    # "COINBASE" | "BINANCE" | "KRAKEN"
    # Coinbase uses paper mode when API_KEY is left as the placeholder.
    # Binance and Kraken use public market data (no keys needed for data).
    "EXCHANGE": "COINBASE",

    # ── Coinbase Advanced Trade ────────────────────────────────────────────────
    "COINBASE": {
        # Get keys at: https://www.coinbase.com/settings/api
        # Leave as-is to run in paper mode
        "API_KEY":    "YOUR_API_KEY_HERE",
        "API_SECRET": "YOUR_API_SECRET_HERE",
        "PRODUCT_ID": "BTC-USD",
        "GRANULARITY": "FIFTEEN_MINUTE",   # 15-min bars (was ONE_MINUTE)
        "WS_URI":      "wss://advanced-trade-ws.coinbase.com",
        "REST_BASE":   "https://api.coinbase.com",
    },

    # ── Binance (public market data — no keys required for data feeds) ─────────
    "BINANCE": {
        "API_KEY":    "",          # required only for live order execution
        "API_SECRET": "",
        "SYMBOL":     "BTCUSDT",
        "REST_BASE":  "https://api.binance.com",
        "WS_BASE":    "wss://stream.binance.com:9443",
    },

    # ── Kraken (public market data — no keys required for data feeds) ──────────
    "KRAKEN": {
        "API_KEY":    "",          # required only for live order execution
        "API_SECRET": "",
        "SYMBOL":     "XBT/USD",   # WebSocket symbol format
        "REST_PAIR":  "XBTUSD",    # REST API pair format
        "REST_BASE":  "https://api.kraken.com",
        "WS_URI":     "wss://ws.kraken.com/v2",
    },

    # ── Paper trading ──────────────────────────────────────────────────────────
    "PAPER_TRADING": {
        "INITIAL_CAPITAL":        10_000.0,  # USD starting capital per model
        "POSITION_SIZE_PCT":       0.10,     # 10% of capital per trade
        "SIMULATED_SLIPPAGE_PCT":  0.0005,   # 0.05% market friction
        "COINBASE_FEE_PCT":        0.015,    # default taker fee (auto-fetched from Coinbase if API keys present)
        "SIGNAL_THRESHOLD":        0.008,    # min predicted return to trade on (must cover ~0.6% fee per leg)
        "ZERO_OBI_IN_INFERENCE":   True,     # keep OBI=0 in live inference to match training distribution (recommended)
        "MAX_DRAWDOWN_PCT":        0.0,      # 0 = disabled; e.g. 15.0 = stop bot when any model drawdown hits 15%

        # ── Adaptive signal threshold (ATR-scaled) ───────────────────────────
        # When enabled, threshold = (ATR / price) * ATR_MULTIPLIER, clamped to
        # [ADAPTIVE_THRESHOLD_MIN, ADAPTIVE_THRESHOLD_MAX].
        # When disabled, SIGNAL_THRESHOLD above is used as the fixed threshold.
        "ADAPTIVE_THRESHOLD":      False,
        "ATR_MULTIPLIER":          0.8,
        "ADAPTIVE_THRESHOLD_MIN":  0.002,
        "ADAPTIVE_THRESHOLD_MAX":  0.050,

        # ── Kelly position sizing ────────────────────────────────────────────
        # When enabled, each model sizes its trades using the Kelly criterion
        # derived from its own running win rate and avg win/loss amounts.
        # Uses KELLY_FRACTION * full-Kelly for safety (default = quarter-Kelly).
        # Falls back to POSITION_SIZE_PCT until KELLY_MIN_TRADES trades exist.
        "KELLY_SIZING":      False,
        "KELLY_FRACTION":    0.25,   # fraction of full Kelly (0.25 = quarter-Kelly)
        "KELLY_MIN_TRADES":  20,     # minimum trades before Kelly activates
        "KELLY_MIN_PCT":     0.02,   # floor: never risk less than 2%
        "KELLY_MAX_PCT":     0.30,   # ceiling: never risk more than 30%
    },

    # ── Notifications (Discord / Telegram webhooks) ────────────────────────────
    # All fields are optional — leave blank to disable that channel.
    # Changes take effect immediately without a restart.
    "NOTIFICATIONS": {
        "DISCORD_WEBHOOK":  "",   # full Discord webhook URL
        "TELEGRAM_TOKEN":   "",   # bot token from @BotFather
        "TELEGRAM_CHAT_ID": "",   # numeric chat/channel ID
        "ON_TRADE":         True, # notify on each paper trade close
        "ON_ESTOP":         True, # notify on emergency stop
        "ON_PROMOTE":       True, # notify when a new model is promoted to active
        "ON_DRAWDOWN":      True, # notify when max-drawdown kill switch triggers
    },

    # ── Data cache (rolling OHLCV history in SQLite — up to 365×3 days) ─────────
    "DATA_CACHE": {
        "MAX_HOURS":              365 * 24,  # 8760h = 365 days
        "REFRESH_INTERVAL_MIN":   5,         # fetch new bars every 5 minutes
        "REFRESH_WINDOW_MIN":     30,        # only pull the last 30 min on each refresh
        "BULK_DOWNLOAD_DELAY_SEC": 1.5,      # seconds between coins during bulk download
    },

    # ── Backtesting ────────────────────────────────────────────────────────────
    "BACKTESTING": {
        "INITIAL_CAPITAL":       10_000.0,
        # SIGNAL_THRESHOLD and FEE_PCT intentionally absent — both read from PAPER_TRADING
        "POSITION_SIZE_PCT":     0.10,
        "SLIPPAGE_PCT":          0.0005,
        "LOOKBACK_HOURS":        180 * 24,  # 4320h = 6 months of 15-min bars
        "WALK_FORWARD_LEVELS":   3,          # default walk-forward folds (1–10)
    },

    # ── Live trading (Coinbase only; requires real API keys) ───────────────────
    # SAFETY: The bot will refuse to arm unless both conditions below are met.
    "LIVE_TRADING": {
        "ENABLED":                      False,          # master switch — must be True to arm
        "ARMED_MODEL":                  "XGBOOST_TREE", # auto-updated by promote_best_models()
        "PREFERRED_MODEL":              "XGBOOST_TREE", # model to prefer when P&L is within tolerance
        "PREFERRED_MODEL_TOLERANCE_PCT": 10.0,          # % gap at which preferred beats raw best
        "MIN_PAPER_HOURS":              24,             # minimum paper-trading duration before arming
        "MIN_PAPER_PNL_PCT":            1.0,            # minimum paper return % required before arming
        "POSITION_SIZE_PCT":            0.05,           # 5% per live trade (conservative)
        "MAX_POSITION_USD":             500.0,          # hard cap per trade in USD
    },

    # ── Model selection ────────────────────────────────────────────────────────
    # Toggle each model on/off independently.  Models that need an optional
    # library (lightgbm, catboost, statsmodels, prophet) are skipped silently
    # if that library is not installed — install them with:
    #   pip install lightgbm catboost statsmodels
    #   pip install prophet   # heavy dep — Stan / CmdStan required
    # LSTM training is slower on CPU (~5-15 min); off by default.
    # ARIMA/Prophet train and infer more slowly; off by default.
    "ACTIVE_MODELS": {
        # ── Fast / always-on ──────────────────────────────────────────────
        "SKLEARN_LINEAR": True,    # Ridge regression (near-instant)
        "XGBOOST_TREE":   True,    # XGBoost gradient-boosted trees
        "LGBM_TREE":      True,    # LightGBM (needs: pip install lightgbm)
        "CATBOOST_TREE":  True,    # CatBoost  (needs: pip install catboost)
        "RF_TREE":        True,    # Random Forest
        "ET_TREE":        True,    # Extra Trees
        "ELASTIC_LINEAR": True,    # ElasticNet (L1+L2)
        "SVR_KERNEL":     True,    # Support-Vector Regression
        "MLP_NN":         True,    # MLP Neural Net (sklearn, no CUDA needed)
        # ── Optional / slower ─────────────────────────────────────────────
        "PYTORCH_LSTM":   False,   # LSTM sequential model (slow on CPU)
        "ARIMA_STATS":    False,   # ARIMA  (needs: pip install statsmodels)
        "PROPHET_FB":     False,   # Prophet (needs: pip install prophet)
    },

    # ── CPU affinity ──────────────────────────────────────────────────────────
    "CPU": {
        "OS_HEADROOM_CORES": 2,
    },

    # ── Web server ────────────────────────────────────────────────────────────
    # HOST and PORT are used as argparse defaults; override with --host/--port.
    # LOCALHOST_ONLY: when True, any request whose client IP is not 127.0.0.1
    # or ::1 is rejected with HTTP 403 regardless of which interface uvicorn is
    # bound to.  Set to False only when you need LAN / internet access (and have
    # HTTPS + a firewall in place).
    "SERVER": {
        "HOST":           "127.0.0.1",
        "PORT":           8765,
        "LOCALHOST_ONLY": True,
    },

    # ── Training ──────────────────────────────────────────────────────────────
    "TRAINING": {
        # Data
        "LOOKBACK_DAYS":    180,   # 6 months of 15-min bars (was 365)
        "N_SPLITS":         3,     # TimeSeriesSplit folds

        # LSTM sequence model
        "SEQUENCE_LENGTH":  20,    # rolling input window (candles)
        "LSTM_HIDDEN_SIZE": 64,    # hidden units per layer
        "LSTM_LAYERS":      2,     # stacked LSTM layers
        "LSTM_DROPOUT":     0.2,   # inter-layer dropout (only if LAYERS > 1)
        "LSTM_EPOCHS":      5,     # training epochs per fold + final
        "LSTM_BATCH_SIZE":  64,    # mini-batch size
        "LSTM_LR":          0.001, # Adam learning rate

        # Ridge (scikit-learn)
        "RIDGE_ALPHA":      1.0,   # L2 regularisation strength

        # XGBoost
        "XGB_N_ESTIMATORS": 200,   # boosting rounds
        "XGB_MAX_DEPTH":    4,     # max tree depth
        "XGB_LR":           0.05,  # boosting learning rate (eta)
        "XGB_SUBSAMPLE":    0.8,   # row sub-sampling per tree
        "XGB_COLSAMPLE":    0.8,   # feature sub-sampling per tree

        # LightGBM
        "LGBM_N_ESTIMATORS": 200,
        "LGBM_MAX_DEPTH":    -1,   # -1 = unlimited
        "LGBM_LR":           0.05,
        "LGBM_SUBSAMPLE":    0.8,
        "LGBM_COLSAMPLE":    0.8,
        "LGBM_BOOSTING":     "gbdt",   # variant: gbdt | dart | goss

        # CatBoost
        "CATBOOST_ITERATIONS": 200,
        "CATBOOST_DEPTH":      6,
        "CATBOOST_LR":         0.05,
        "CATBOOST_LOSS":       "RMSE",  # variant: RMSE | MAE | Quantile

        # Random Forest
        "RF_N_ESTIMATORS":   200,
        "RF_MAX_DEPTH":      10,

        # Extra Trees
        "ET_N_ESTIMATORS":   200,
        "ET_MAX_DEPTH":      10,

        # ElasticNet
        "ELASTIC_ALPHA":     0.1,  # overall regularisation strength
        "ELASTIC_L1_RATIO":  0.5,  # 0 = Ridge, 1 = Lasso

        # SVR
        "SVR_C":             1.0,
        "SVR_EPSILON":       0.1,
        "SVR_KERNEL_TYPE":   "rbf",  # variant: rbf | linear | poly

        # MLP (sklearn MLPRegressor)
        "MLP_HIDDEN_LAYERS": "100,50",  # comma-separated layer sizes
        "MLP_MAX_ITER":      500,
        "MLP_LR":            0.001,

        # ARIMA (statsmodels)
        "ARIMA_P":           2,    # autoregressive order
        "ARIMA_D":           0,    # differencing order (0 = already stationary)
        "ARIMA_Q":           2,    # moving-average order
        "ARIMA_WINDOW":      500,  # rolling-window size for live inference

        # Prophet (Meta/Facebook)
        "PROPHET_CHANGEPOINTS": 25,
        "PROPHET_SEASONALITY":  False,
        "PROPHET_WINDOW":       1000,  # rolling-window size for live inference
    },

    # ── Coin selection ────────────────────────────────────────────────────────
    # Controls which coins are used for training, backtesting, and trading.
    # Training tiers:
    #   single   — 1 coin  : BTC
    #   quick    — 2 coins : BTC, ETH
    #   standard — 4 coins : BTC, ETH, LTC, DOGE  ← default
    #   extended — 10 curated coins (BTC, ETH, LTC, DOGE, SOL, XRP, ADA, DOT, AVAX, LINK)
    #   insane   — all USD-quoted coins available on the active exchange
    "COINS": {
        "TRAINING_TIER":  "standard",
        "TRADING_COIN":   "BTC",                      # base coin for live/paper trading
        "BACKTEST_COINS": ["BTC", "ETH", "LTC", "DOGE"],  # matches standard tier
    },

    # ── File paths ─────────────────────────────────────────────────────────────
    "PATHS": {
        "DB":              _p("data/challenger_trades.db"),
        "SKLEARN_MODEL":   _p("models/sklearn_model.pkl"),
        "XGBOOST_MODEL":   _p("models/xgboost_model.pkl"),
        "PYTORCH_MODEL":   _p("models/pytorch_model.pt"),
        "LGBM_MODEL":      _p("models/lgbm_model.pkl"),
        "CATBOOST_MODEL":  _p("models/catboost_model.pkl"),
        "RF_MODEL":        _p("models/rf_model.pkl"),
        "ET_MODEL":        _p("models/et_model.pkl"),
        "ELASTIC_MODEL":   _p("models/elastic_model.pkl"),
        "SVR_MODEL":       _p("models/svr_model.pkl"),
        "MLP_MODEL":       _p("models/mlp_model.pkl"),
        "ARIMA_MODEL":     _p("models/arima_model.pkl"),
        "PROPHET_MODEL":   _p("models/prophet_model.pkl"),
        "SCALER":          _p("models/feature_scaler.pkl"),
        "HISTORICAL_CSV":  _p("data/historical_candles.csv"),
        "CV_FOLDS":        _p("data/cv_folds.pkl"),
    },
}

# Feature columns fed into every model (order matters for LSTM sequences)
FEATURE_COLS = ["log_return", "rsi_14", "atr_14", "ema_diff", "rolling_vol_20", "obi"]
TARGET_COL   = "target_next_return"

# ── User-settable overrides (written by the Settings tab in the TUI) ──────────
# Stored in data/user_settings.json so they survive restarts.
# Keys map directly to CONFIG paths — see save_user_settings() below.
_OVERRIDES_PATH = _p("data/user_settings.json")
_KEYS_PATH = _p("data/api_keys.json")
_SECRETS_PATH = _p("data/.secrets.json")
_SECRET_OVERRIDE_KEYS = {"DISCORD_WEBHOOK", "TELEGRAM_TOKEN"}
_COINBASE_SECRET_KEYS = {
    "api_key": "COINBASE_API_KEY",
    "api_secret": "COINBASE_API_SECRET",
}


def _apply_local_secrets() -> None:
    """Overlay protected local secrets onto CONFIG."""
    secrets = _load_secrets(_SECRETS_PATH)
    key = secrets.get(_COINBASE_SECRET_KEYS["api_key"], "").strip()
    secret = secrets.get(_COINBASE_SECRET_KEYS["api_secret"], "").strip()
    if key and secret:
        CONFIG["COINBASE"]["API_KEY"] = key
        CONFIG["COINBASE"]["API_SECRET"] = secret

    webhook = secrets.get("DISCORD_WEBHOOK", "").strip()
    telegram = secrets.get("TELEGRAM_TOKEN", "").strip()
    if webhook:
        CONFIG["NOTIFICATIONS"]["DISCORD_WEBHOOK"] = webhook
    if telegram:
        CONFIG["NOTIFICATIONS"]["TELEGRAM_TOKEN"] = telegram


def _migrate_plaintext_notification_secrets(overrides: dict) -> bool:
    """Move notification secrets out of user_settings.json."""
    updates: dict[str, str | None] = {}
    changed = False
    for key in _SECRET_OVERRIDE_KEYS:
        if key in overrides:
            val = str(overrides.get(key) or "").strip()
            updates[key] = val or None
            overrides.pop(key, None)
            changed = True
    if updates:
        _update_secrets(_SECRETS_PATH, updates)
    return changed

def _load_overrides():
    import json as _json
    if not _os.path.exists(_OVERRIDES_PATH):
        return
    try:
        with open(_OVERRIDES_PATH) as _f:
            ov = _json.load(_f)

        # ── Data / cache ───────────────────────────────────────────────────────
        if "TRAINING_LOOKBACK_DAYS"   in ov: CONFIG["TRAINING"]["LOOKBACK_DAYS"]           = int(ov["TRAINING_LOOKBACK_DAYS"])
        if "DATA_CACHE_MAX_DAYS"      in ov: CONFIG["DATA_CACHE"]["MAX_HOURS"]              = int(ov["DATA_CACHE_MAX_DAYS"]) * 24
        if "DATA_CACHE_MAX_HOURS"     in ov: CONFIG["DATA_CACHE"]["MAX_HOURS"]              = int(ov["DATA_CACHE_MAX_HOURS"])
        if "DATA_CACHE_REFRESH_MIN"   in ov: CONFIG["DATA_CACHE"]["REFRESH_INTERVAL_MIN"]   = int(ov["DATA_CACHE_REFRESH_MIN"])
        if "BACKTEST_LOOKBACK_DAYS"   in ov: CONFIG["BACKTESTING"]["LOOKBACK_HOURS"]        = int(ov["BACKTEST_LOOKBACK_DAYS"]) * 24
        if "BACKTEST_LOOKBACK_HOURS"  in ov: CONFIG["BACKTESTING"]["LOOKBACK_HOURS"]        = int(ov["BACKTEST_LOOKBACK_HOURS"])
        if "WALK_FORWARD_LEVELS"      in ov: CONFIG["BACKTESTING"]["WALK_FORWARD_LEVELS"]   = max(1, min(10, int(ov["WALK_FORWARD_LEVELS"])))
        if "SIGNAL_THRESHOLD"         in ov: CONFIG["PAPER_TRADING"]["SIGNAL_THRESHOLD"]    = float(ov["SIGNAL_THRESHOLD"])
        if "COINBASE_FEE_PCT"         in ov: CONFIG["PAPER_TRADING"]["COINBASE_FEE_PCT"]    = float(ov["COINBASE_FEE_PCT"])
        if "ZERO_OBI_IN_INFERENCE"    in ov: CONFIG["PAPER_TRADING"]["ZERO_OBI_IN_INFERENCE"] = bool(ov["ZERO_OBI_IN_INFERENCE"])
        if "MAX_DRAWDOWN_PCT"          in ov: CONFIG["PAPER_TRADING"]["MAX_DRAWDOWN_PCT"]       = float(ov["MAX_DRAWDOWN_PCT"])
        if "ADAPTIVE_THRESHOLD"       in ov: CONFIG["PAPER_TRADING"]["ADAPTIVE_THRESHOLD"]     = bool(ov["ADAPTIVE_THRESHOLD"])
        if "ATR_MULTIPLIER"           in ov: CONFIG["PAPER_TRADING"]["ATR_MULTIPLIER"]         = float(ov["ATR_MULTIPLIER"])
        if "ADAPTIVE_THRESHOLD_MIN"   in ov: CONFIG["PAPER_TRADING"]["ADAPTIVE_THRESHOLD_MIN"] = float(ov["ADAPTIVE_THRESHOLD_MIN"])
        if "ADAPTIVE_THRESHOLD_MAX"   in ov: CONFIG["PAPER_TRADING"]["ADAPTIVE_THRESHOLD_MAX"] = float(ov["ADAPTIVE_THRESHOLD_MAX"])
        if "KELLY_SIZING"             in ov: CONFIG["PAPER_TRADING"]["KELLY_SIZING"]           = bool(ov["KELLY_SIZING"])
        if "KELLY_FRACTION"           in ov: CONFIG["PAPER_TRADING"]["KELLY_FRACTION"]         = float(ov["KELLY_FRACTION"])
        if "KELLY_MIN_TRADES"         in ov: CONFIG["PAPER_TRADING"]["KELLY_MIN_TRADES"]       = int(ov["KELLY_MIN_TRADES"])
        if "KELLY_MIN_PCT"            in ov: CONFIG["PAPER_TRADING"]["KELLY_MIN_PCT"]          = float(ov["KELLY_MIN_PCT"])
        if "KELLY_MAX_PCT"            in ov: CONFIG["PAPER_TRADING"]["KELLY_MAX_PCT"]          = float(ov["KELLY_MAX_PCT"])

        # ── Notifications ──────────────────────────────────────────────────────
        if _migrate_plaintext_notification_secrets(ov):
            with open(_OVERRIDES_PATH, "w") as _f:
                _json.dump(ov, _f, indent=2)
        if "TELEGRAM_CHAT_ID"  in ov: CONFIG["NOTIFICATIONS"]["TELEGRAM_CHAT_ID"] = str(ov["TELEGRAM_CHAT_ID"])
        if "NOTIFY_ON_TRADE"   in ov: CONFIG["NOTIFICATIONS"]["ON_TRADE"]         = bool(ov["NOTIFY_ON_TRADE"])
        if "NOTIFY_ON_ESTOP"   in ov: CONFIG["NOTIFICATIONS"]["ON_ESTOP"]         = bool(ov["NOTIFY_ON_ESTOP"])
        if "NOTIFY_ON_PROMOTE" in ov: CONFIG["NOTIFICATIONS"]["ON_PROMOTE"]       = bool(ov["NOTIFY_ON_PROMOTE"])

        if "MIN_PAPER_HOURS"              in ov: CONFIG["LIVE_TRADING"]["MIN_PAPER_HOURS"]                = int(ov["MIN_PAPER_HOURS"])
        if "MIN_PAPER_PNL_PCT"            in ov: CONFIG["LIVE_TRADING"]["MIN_PAPER_PNL_PCT"]              = float(ov["MIN_PAPER_PNL_PCT"])
        if "PREFERRED_MODEL"              in ov: CONFIG["LIVE_TRADING"]["PREFERRED_MODEL"]                = str(ov["PREFERRED_MODEL"])
        if "PREFERRED_MODEL_TOLERANCE_PCT" in ov: CONFIG["LIVE_TRADING"]["PREFERRED_MODEL_TOLERANCE_PCT"] = float(ov["PREFERRED_MODEL_TOLERANCE_PCT"])
        if "EXCHANGE"                 in ov: CONFIG["EXCHANGE"]                             = str(ov["EXCHANGE"])

        # ── CV ─────────────────────────────────────────────────────────────────
        if "TRAINING_N_SPLITS"        in ov: CONFIG["TRAINING"]["N_SPLITS"]                 = int(ov["TRAINING_N_SPLITS"])

        # ── LSTM architecture ──────────────────────────────────────────────────
        if "LSTM_SEQUENCE_LENGTH"     in ov: CONFIG["TRAINING"]["SEQUENCE_LENGTH"]          = int(ov["LSTM_SEQUENCE_LENGTH"])
        if "LSTM_HIDDEN_SIZE"         in ov: CONFIG["TRAINING"]["LSTM_HIDDEN_SIZE"]         = int(ov["LSTM_HIDDEN_SIZE"])
        if "LSTM_LAYERS"              in ov: CONFIG["TRAINING"]["LSTM_LAYERS"]              = int(ov["LSTM_LAYERS"])
        if "LSTM_DROPOUT"             in ov: CONFIG["TRAINING"]["LSTM_DROPOUT"]             = float(ov["LSTM_DROPOUT"])

        # ── LSTM training ──────────────────────────────────────────────────────
        if "LSTM_EPOCHS"              in ov: CONFIG["TRAINING"]["LSTM_EPOCHS"]              = int(ov["LSTM_EPOCHS"])
        if "LSTM_BATCH_SIZE"          in ov: CONFIG["TRAINING"]["LSTM_BATCH_SIZE"]          = int(ov["LSTM_BATCH_SIZE"])
        if "LSTM_LR"                  in ov: CONFIG["TRAINING"]["LSTM_LR"]                  = float(ov["LSTM_LR"])

        # ── Ridge ──────────────────────────────────────────────────────────────
        if "RIDGE_ALPHA"              in ov: CONFIG["TRAINING"]["RIDGE_ALPHA"]              = float(ov["RIDGE_ALPHA"])

        # ── XGBoost ────────────────────────────────────────────────────────────
        if "XGB_N_ESTIMATORS"         in ov: CONFIG["TRAINING"]["XGB_N_ESTIMATORS"]         = int(ov["XGB_N_ESTIMATORS"])
        if "XGB_MAX_DEPTH"            in ov: CONFIG["TRAINING"]["XGB_MAX_DEPTH"]            = int(ov["XGB_MAX_DEPTH"])
        if "XGB_LR"                   in ov: CONFIG["TRAINING"]["XGB_LR"]                   = float(ov["XGB_LR"])
        if "XGB_SUBSAMPLE"            in ov: CONFIG["TRAINING"]["XGB_SUBSAMPLE"]            = float(ov["XGB_SUBSAMPLE"])
        if "XGB_COLSAMPLE"            in ov: CONFIG["TRAINING"]["XGB_COLSAMPLE"]            = float(ov["XGB_COLSAMPLE"])

        # ── Active models ──────────────────────────────────────────────────────
        if "ACTIVE_MODEL_SKLEARN"     in ov: CONFIG["ACTIVE_MODELS"]["SKLEARN_LINEAR"]      = bool(ov["ACTIVE_MODEL_SKLEARN"])
        if "ACTIVE_MODEL_XGBOOST"     in ov: CONFIG["ACTIVE_MODELS"]["XGBOOST_TREE"]        = bool(ov["ACTIVE_MODEL_XGBOOST"])
        if "ACTIVE_MODEL_LSTM"        in ov: CONFIG["ACTIVE_MODELS"]["PYTORCH_LSTM"]        = bool(ov["ACTIVE_MODEL_LSTM"])
        if "ACTIVE_MODEL_LGBM"        in ov: CONFIG["ACTIVE_MODELS"]["LGBM_TREE"]           = bool(ov["ACTIVE_MODEL_LGBM"])
        if "ACTIVE_MODEL_CATBOOST"    in ov: CONFIG["ACTIVE_MODELS"]["CATBOOST_TREE"]       = bool(ov["ACTIVE_MODEL_CATBOOST"])
        if "ACTIVE_MODEL_RF"          in ov: CONFIG["ACTIVE_MODELS"]["RF_TREE"]             = bool(ov["ACTIVE_MODEL_RF"])
        if "ACTIVE_MODEL_ET"          in ov: CONFIG["ACTIVE_MODELS"]["ET_TREE"]             = bool(ov["ACTIVE_MODEL_ET"])
        if "ACTIVE_MODEL_ELASTIC"     in ov: CONFIG["ACTIVE_MODELS"]["ELASTIC_LINEAR"]      = bool(ov["ACTIVE_MODEL_ELASTIC"])
        if "ACTIVE_MODEL_SVR"         in ov: CONFIG["ACTIVE_MODELS"]["SVR_KERNEL"]          = bool(ov["ACTIVE_MODEL_SVR"])
        if "ACTIVE_MODEL_MLP"         in ov: CONFIG["ACTIVE_MODELS"]["MLP_NN"]              = bool(ov["ACTIVE_MODEL_MLP"])
        if "ACTIVE_MODEL_ARIMA"       in ov: CONFIG["ACTIVE_MODELS"]["ARIMA_STATS"]         = bool(ov["ACTIVE_MODEL_ARIMA"])
        if "ACTIVE_MODEL_PROPHET"     in ov: CONFIG["ACTIVE_MODELS"]["PROPHET_FB"]          = bool(ov["ACTIVE_MODEL_PROPHET"])

        # ── LightGBM ───────────────────────────────────────────────────────────
        if "LGBM_N_ESTIMATORS"        in ov: CONFIG["TRAINING"]["LGBM_N_ESTIMATORS"]        = int(ov["LGBM_N_ESTIMATORS"])
        if "LGBM_BOOSTING"            in ov: CONFIG["TRAINING"]["LGBM_BOOSTING"]            = str(ov["LGBM_BOOSTING"])
        if "LGBM_LR"                  in ov: CONFIG["TRAINING"]["LGBM_LR"]                  = float(ov["LGBM_LR"])

        # ── CatBoost ───────────────────────────────────────────────────────────
        if "CATBOOST_ITERATIONS"      in ov: CONFIG["TRAINING"]["CATBOOST_ITERATIONS"]      = int(ov["CATBOOST_ITERATIONS"])
        if "CATBOOST_LR"              in ov: CONFIG["TRAINING"]["CATBOOST_LR"]              = float(ov["CATBOOST_LR"])
        if "CATBOOST_LOSS"            in ov: CONFIG["TRAINING"]["CATBOOST_LOSS"]            = str(ov["CATBOOST_LOSS"])

        # ── RandomForest / ExtraTrees ──────────────────────────────────────────
        if "RF_N_ESTIMATORS"          in ov: CONFIG["TRAINING"]["RF_N_ESTIMATORS"]          = int(ov["RF_N_ESTIMATORS"])
        if "ET_N_ESTIMATORS"          in ov: CONFIG["TRAINING"]["ET_N_ESTIMATORS"]          = int(ov["ET_N_ESTIMATORS"])

        # ── ElasticNet ─────────────────────────────────────────────────────────
        if "ELASTIC_ALPHA"            in ov: CONFIG["TRAINING"]["ELASTIC_ALPHA"]            = float(ov["ELASTIC_ALPHA"])
        if "ELASTIC_L1_RATIO"         in ov: CONFIG["TRAINING"]["ELASTIC_L1_RATIO"]         = float(ov["ELASTIC_L1_RATIO"])

        # ── SVR ────────────────────────────────────────────────────────────────
        if "SVR_C"                    in ov: CONFIG["TRAINING"]["SVR_C"]                    = float(ov["SVR_C"])
        if "SVR_EPSILON"              in ov: CONFIG["TRAINING"]["SVR_EPSILON"]              = float(ov["SVR_EPSILON"])
        if "SVR_KERNEL_TYPE"          in ov: CONFIG["TRAINING"]["SVR_KERNEL_TYPE"]          = str(ov["SVR_KERNEL_TYPE"])

        # ── MLP ────────────────────────────────────────────────────────────────
        if "MLP_HIDDEN_LAYERS"        in ov: CONFIG["TRAINING"]["MLP_HIDDEN_LAYERS"]        = str(ov["MLP_HIDDEN_LAYERS"])
        if "MLP_MAX_ITER"             in ov: CONFIG["TRAINING"]["MLP_MAX_ITER"]             = int(ov["MLP_MAX_ITER"])
        if "MLP_LR"                   in ov: CONFIG["TRAINING"]["MLP_LR"]                  = float(ov["MLP_LR"])

        # ── ARIMA ──────────────────────────────────────────────────────────────
        if "ARIMA_P"                  in ov: CONFIG["TRAINING"]["ARIMA_P"]                  = int(ov["ARIMA_P"])
        if "ARIMA_D"                  in ov: CONFIG["TRAINING"]["ARIMA_D"]                  = int(ov["ARIMA_D"])
        if "ARIMA_Q"                  in ov: CONFIG["TRAINING"]["ARIMA_Q"]                  = int(ov["ARIMA_Q"])

        # ── Web server ─────────────────────────────────────────────────────────
        if "SERVER_HOST"           in ov: CONFIG["SERVER"]["HOST"]           = str(ov["SERVER_HOST"])
        if "SERVER_PORT"           in ov: CONFIG["SERVER"]["PORT"]           = int(ov["SERVER_PORT"])
        if "SERVER_LOCALHOST_ONLY" in ov: CONFIG["SERVER"]["LOCALHOST_ONLY"] = bool(ov["SERVER_LOCALHOST_ONLY"])

        # ── Coin selection ─────────────────────────────────────────────────────
        if "TRAINING_TIER"            in ov: CONFIG["COINS"]["TRAINING_TIER"]               = str(ov["TRAINING_TIER"])
        if "TRADING_COIN"             in ov: CONFIG["COINS"]["TRADING_COIN"]                = str(ov["TRADING_COIN"]).upper()
        if "BACKTEST_COINS"           in ov:
            raw = ov["BACKTEST_COINS"]
            if isinstance(raw, list):
                CONFIG["COINS"]["BACKTEST_COINS"] = [str(c).upper() for c in raw]
            elif isinstance(raw, str):
                CONFIG["COINS"]["BACKTEST_COINS"] = [c.strip().upper() for c in raw.split(",") if c.strip()]

    except Exception as e:
        print(f"[CONFIG] Could not load user_settings.json: {e}")

_load_overrides()
_apply_local_secrets()


# ── API key storage ───────────────────────────────────────────────────────────
# Keys are saved to data/.secrets.json with OS protection where available.
# Legacy data/api_keys.json is still read once and migrated for compatibility.

def _load_api_keys() -> None:
    """
    Load Coinbase key+secret from protected local storage, or migrate the
    legacy plaintext data/api_keys.json if it exists.
    """
    import json as _json
    _apply_local_secrets()
    if (CONFIG["COINBASE"]["API_KEY"] != "YOUR_API_KEY_HERE" and
            CONFIG["COINBASE"]["API_SECRET"] != "YOUR_API_SECRET_HERE"):
        return
    if not _os.path.exists(_KEYS_PATH):
        return
    try:
        with open(_KEYS_PATH) as _f:
            keys = _json.load(_f)
        key    = (keys.get("api_key")    or keys.get("key")    or "").strip()
        secret = (keys.get("api_secret") or keys.get("secret") or "").strip()
        if key and secret:
            _update_secrets(_SECRETS_PATH, {
                _COINBASE_SECRET_KEYS["api_key"]: key,
                _COINBASE_SECRET_KEYS["api_secret"]: secret,
            })
            CONFIG["COINBASE"]["API_KEY"]    = key
            CONFIG["COINBASE"]["API_SECRET"] = secret
            try:
                _os.remove(_KEYS_PATH)
            except OSError:
                pass
    except Exception as e:
        print(f"[CONFIG] Could not load api_keys.json: {e}")


_load_api_keys()


def save_api_keys(api_key: str, api_secret: str) -> None:
    """
    Persist Coinbase API credentials to protected local storage.
    Also updates the in-memory CONFIG immediately so the running bot sees
    the new keys without a restart.
    """
    _update_secrets(_SECRETS_PATH, {
        _COINBASE_SECRET_KEYS["api_key"]: api_key,
        _COINBASE_SECRET_KEYS["api_secret"]: api_secret,
    })
    if _os.path.exists(_KEYS_PATH):
        _os.remove(_KEYS_PATH)
    CONFIG["COINBASE"]["API_KEY"]    = api_key
    CONFIG["COINBASE"]["API_SECRET"] = api_secret


def clear_api_keys() -> None:
    """
    Delete stored Coinbase credentials and revert CONFIG to the demo placeholders.
    The bot will switch back to paper mode immediately.
    """
    _update_secrets(_SECRETS_PATH, {
        _COINBASE_SECRET_KEYS["api_key"]: None,
        _COINBASE_SECRET_KEYS["api_secret"]: None,
    })
    if _os.path.exists(_KEYS_PATH):
        _os.remove(_KEYS_PATH)
    CONFIG["COINBASE"]["API_KEY"]    = "YOUR_API_KEY_HERE"
    CONFIG["COINBASE"]["API_SECRET"] = "YOUR_API_SECRET_HERE"


def save_user_settings(overrides: dict) -> None:
    """
    Persist user-adjustable settings to data/user_settings.json.
    Called from the Settings / Training tabs when the user clicks Apply.

    Accepted keys (all optional — only supplied keys are written):

      Data / cache
        TRAINING_LOOKBACK_DAYS   int    days of history to train on
        DATA_CACHE_MAX_HOURS     int    rolling cache window in hours
        DATA_CACHE_REFRESH_MIN   int    cache refresh interval (minutes)
        BACKTEST_LOOKBACK_HOURS  int    default backtest window
        SIGNAL_THRESHOLD         float  min predicted return to trade on
        EXCHANGE                 str    COINBASE | BINANCE | KRAKEN

      Cross-validation
        TRAINING_N_SPLITS        int    TimeSeriesSplit folds

      LSTM architecture
        LSTM_SEQUENCE_LENGTH     int    rolling input window (candles)
        LSTM_HIDDEN_SIZE         int    hidden units per layer
        LSTM_LAYERS              int    stacked LSTM layers
        LSTM_DROPOUT             float  inter-layer dropout (0–1)

      LSTM training
        LSTM_EPOCHS              int    training epochs per fold + final
        LSTM_BATCH_SIZE          int    mini-batch size
        LSTM_LR                  float  Adam learning rate

      Ridge
        RIDGE_ALPHA              float  L2 regularisation strength

      XGBoost
        XGB_N_ESTIMATORS         int    boosting rounds
        XGB_MAX_DEPTH            int    max tree depth
        XGB_LR                   float  boosting learning rate (eta)
        XGB_SUBSAMPLE            float  row sub-sampling per tree (0–1)
        XGB_COLSAMPLE            float  feature sub-sampling per tree (0–1)

      Active models
        ACTIVE_MODEL_SKLEARN     bool   enable Ridge model
        ACTIVE_MODEL_XGBOOST     bool   enable XGBoost model
        ACTIVE_MODEL_LSTM        bool   enable LSTM model
    """
    import json as _json
    _os.makedirs(_os.path.dirname(_OVERRIDES_PATH), exist_ok=True)
    secret_updates: dict[str, str | None] = {}
    for key in _SECRET_OVERRIDE_KEYS:
        if key in overrides:
            val = str(overrides.pop(key) or "").strip()
            secret_updates[key] = val or None
    if secret_updates:
        _update_secrets(_SECRETS_PATH, secret_updates)

    # Merge with any existing overrides so we don't clobber unrelated keys
    existing = {}
    if _os.path.exists(_OVERRIDES_PATH):
        try:
            with open(_OVERRIDES_PATH) as f:
                existing = _json.load(f)
        except Exception:
            pass
    for key in _SECRET_OVERRIDE_KEYS:
        existing.pop(key, None)
    existing.update(overrides)
    with open(_OVERRIDES_PATH, "w") as f:
        _json.dump(existing, f, indent=2)
    # Apply immediately to the running CONFIG
    _load_overrides()
    _apply_local_secrets()


def compute_signal_threshold(atr: float = 0.0, price: float = 0.0) -> float:
    """
    Return the effective signal threshold for the current tick.

    When ADAPTIVE_THRESHOLD is enabled, the threshold scales with ATR as a
    fraction of price (ATR/price * ATR_MULTIPLIER), clamped to [MIN, MAX].
    Otherwise returns the fixed SIGNAL_THRESHOLD.
    """
    pt = CONFIG["PAPER_TRADING"]
    if pt.get("ADAPTIVE_THRESHOLD", False) and atr > 0 and price > 0:
        raw = (atr / price) * pt.get("ATR_MULTIPLIER", 0.8)
        return max(
            pt.get("ADAPTIVE_THRESHOLD_MIN", 0.002),
            min(pt.get("ADAPTIVE_THRESHOLD_MAX", 0.050), raw),
        )
    return pt["SIGNAL_THRESHOLD"]


def is_demo_mode() -> bool:
    """True when running on Coinbase with no real API keys."""
    return (CONFIG["EXCHANGE"] == "COINBASE" and
            CONFIG["COINBASE"]["API_KEY"] == "YOUR_API_KEY_HERE")


def get_active_symbol() -> str:
    """Return the trading symbol for the currently active exchange."""
    ex   = CONFIG["EXCHANGE"]
    coin = CONFIG.get("COINS", {}).get("TRADING_COIN", "")
    if coin:
        if ex == "BINANCE":
            return f"{coin.upper()}USDT"
        if ex == "KRAKEN":
            _kb = {"BTC": "XBT", "DOGE": "XDG"}
            return f"{_kb.get(coin.upper(), coin.upper())}/USD"
        return f"{coin.upper()}-USD"   # Coinbase
    # Fallback to per-exchange config
    if ex == "BINANCE":
        return CONFIG["BINANCE"]["SYMBOL"]
    if ex == "KRAKEN":
        return CONFIG["KRAKEN"]["SYMBOL"]
    return CONFIG["COINBASE"]["PRODUCT_ID"]
