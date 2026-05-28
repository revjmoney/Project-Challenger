"""
Backtesting engine — replays historical 1-minute candles through the trained models.

Uses the same feature engine, shared scaler, and model files as live inference,
so backtest predictions are identical to what the live system would have produced.
Results are stored in the backtest_results table and returned as BacktestResult objects.
"""
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from model_security import load_joblib, load_torch

from config import CONFIG, FEATURE_COLS, get_active_symbol, compute_signal_threshold
from features import calculate_features
from database import (
    store_backtest_result, get_candle_cache_count,
    get_cached_candles, store_candles,
)
from activity import get_tracker, BACKTESTING, INFERRING, COMPUTING, LOADING, IDLE, ERROR, DONE


# ─────────────────────────────────────────────────── data types ───────────────

@dataclass
class BacktestConfig:
    initial_capital:   float = 10_000.0
    signal_threshold:  float = 0.0002
    position_size_pct: float = 0.10
    slippage_pct:      float = 0.0005
    fee_pct:           float = 0.006
    lookback_hours:    int   = 24

    @classmethod
    def from_config(cls) -> "BacktestConfig":
        bt = CONFIG["BACKTESTING"]
        pt = CONFIG["PAPER_TRADING"]
        return cls(
            initial_capital   = bt["INITIAL_CAPITAL"],
            signal_threshold  = pt["SIGNAL_THRESHOLD"],
            position_size_pct = bt["POSITION_SIZE_PCT"],
            slippage_pct      = bt["SLIPPAGE_PCT"],
            fee_pct           = pt["COINBASE_FEE_PCT"],
            lookback_hours    = bt["LOOKBACK_HOURS"],
        )


@dataclass
class BacktestResult:
    model_name:    str
    total_candles: int    = 0
    total_trades:  int    = 0
    wins:          int    = 0
    losses:        int    = 0
    net_pnl:       float  = 0.0
    gross_profit:  float  = 0.0
    gross_loss:    float  = 0.0
    max_drawdown:  float  = 0.0
    initial_cap:   float  = 10_000.0
    equity_curve:  list   = field(default_factory=list)
    trade_history: list   = field(default_factory=list)
    error: Optional[str]  = None

    @property
    def win_rate(self) -> float:
        t = self.wins + self.losses
        return self.wins / t if t > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else float("inf")

    @property
    def net_return_pct(self) -> float:
        return self.net_pnl / self.initial_cap * 100 if self.initial_cap > 0 else 0.0


# ─────────────────────────────────────────────────── public API ───────────────

def run_backtest(
    model_key: str,
    candles:   list[dict],
    cfg:       BacktestConfig,
    log_fn:    Callable[[str], None] | None = None,
    tracker_component: str = "BACKTESTER",
) -> BacktestResult:
    """Run a backtest for a single model over the provided candle list."""
    _log    = log_fn or print
    tracker = get_tracker()
    result  = BacktestResult(model_name=model_key, initial_cap=cfg.initial_capital)

    tracker.update(tracker_component, BACKTESTING,
                   f"{model_key}: building feature matrix from {len(candles)} candles...")

    # 1. Build DataFrame
    df = _candles_to_df(candles)
    if df is None or len(df) < 50:
        n = len(df) if df is not None else 0
        result.error = f"Not enough data ({n} candles, need 50+)"
        tracker.update(tracker_component, ERROR, result.error)
        return result

    # 2. Calculate features
    tracker.update(tracker_component, COMPUTING,
                   f"{model_key}: calculating features ({len(df)} rows)...")
    df = calculate_features(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < 30:
        result.error = "Not enough rows after feature calculation"
        tracker.update(tracker_component, ERROR, result.error)
        return result

    # 3. Scale
    scaler_path = CONFIG["PATHS"]["SCALER"]
    if not os.path.exists(scaler_path):
        result.error = "Scaler not found — run training first"
        tracker.update(tracker_component, ERROR, result.error)
        return result
    tracker.update(tracker_component, LOADING,
                   f"{model_key}: loading scaler & model weights...")
    scaler = load_joblib(scaler_path)
    X      = scaler.transform(df[FEATURE_COLS].values).astype(np.float32)
    closes = df["close"].values.astype(float)

    # 4. Predict
    tracker.update(tracker_component, INFERRING,
                   f"{model_key}: running inference on {len(X)} rows...")
    preds = _get_predictions(model_key, X, _log, df=df)
    if preds is None:
        result.error = f"{model_key} model file not found — run training first"
        tracker.update(tracker_component, ERROR, result.error)
        return result

    # Align lengths (LSTM pads the front with zeros for its warm-up window)
    min_len = min(len(preds), len(closes))
    preds   = preds[-min_len:]
    closes  = closes[-min_len:]
    atrs    = df["atr_14"].values.astype(float)[-min_len:]

    # 5. Simulate
    tracker.update(tracker_component, BACKTESTING,
                   f"{model_key}: simulating {len(closes)} ticks...")
    result = _simulate(model_key, preds, closes, cfg, atr_series=atrs)
    result.total_candles = len(closes)

    # 6. Persist summary
    try:
        store_backtest_result(result, cfg, CONFIG["EXCHANGE"], get_active_symbol())
    except Exception as e:
        _log(f"[BT] DB store error: {e}")

    tracker.update(tracker_component, DONE,
                   f"{model_key}: done — {result.total_trades} trades, "
                   f"WR={result.win_rate:.0%}, PnL={result.net_pnl:+.2f}")
    return result


def run_all_backtests(
    candles: list[dict],
    cfg:     BacktestConfig,
    log_fn:      Callable[[str], None] | None = None,
    progress_cb: Callable[[str, int, int], None] | None = None,
    model_to_component: dict[str, str] | None = None,
) -> dict[str, BacktestResult]:
    """
    Run backtests for every active model **concurrently**.

    progress_cb(model_key, done_count, total_count) is called each time a
    model finishes so callers can stream progress to the UI.

    Returns {model_key: BacktestResult}.
    """
    _log        = log_fn or print
    active_keys = [k for k, v in CONFIG["ACTIVE_MODELS"].items() if v]
    if not active_keys:
        return {}

    n          = len(active_keys)
    results:   dict[str, BacktestResult] = {}
    done_count = 0
    t0         = time.perf_counter()

    _log(f"[BT] Launching {n} model backtest(s) in parallel...")

    m2c = model_to_component or {}

    with ThreadPoolExecutor(max_workers=n, thread_name_prefix="bt") as pool:
        futures = {
            pool.submit(run_backtest, key, candles, cfg, log_fn, m2c.get(key, "BACKTESTER")): key
            for key in active_keys
        }
        for fut in as_completed(futures):
            key = futures[fut]
            done_count += 1
            try:
                results[key] = fut.result()
            except Exception as exc:
                results[key] = BacktestResult(model_name=key, error=str(exc))
                _log(f"[BT] {key} raised an exception: {exc}")
            if progress_cb:
                progress_cb(key, done_count, n)

    elapsed = time.perf_counter() - t0
    _log(f"[BT] All {n} model(s) finished in {elapsed:.1f}s")
    return results


def run_walk_forward_validation(
    candles:     list[dict],
    cfg:         BacktestConfig,
    n_folds:     int = 3,
    log_fn:      Callable[[str], None] | None = None,
    progress_cb: Callable[[int, int, int, int], None] | None = None,
) -> dict:
    """
    Walk-forward validation — splits candle data into *n_folds* sequential
    time windows and runs a full backtest on each slice with the saved model
    files.  No retraining occurs; the goal is measuring temporal robustness
    (does the model stay profitable across different market regimes?).

    progress_cb(fold_num, total_folds, models_done, models_total) is called
    after each model finishes within a fold.

    Returns:
      {
        "n_folds":   int,
        "folds":     [
          {
            "fold_num":  int,
            "candles":   int,
            "start_ts":  int,   # unix epoch seconds
            "end_ts":    int,
            "results":   {model_key: {net_pnl, total_trades, win_rate,
                                      profit_factor, max_drawdown} | {error}},
          }, …
        ],
        "aggregate": {
          model_key: {
            "avg_pnl", "avg_win_rate", "avg_drawdown",
            "consistency",       # fraction of folds where net_pnl > 0
            "total_pnl", "best_fold_pnl", "worst_fold_pnl", "n_folds",
          }
        },
      }
    or {"error": "…", "n_folds": 0, "folds": [], "aggregate": {}} on failure.
    """
    _log         = log_fn or print
    total        = len(candles)
    min_per_fold = 100   # need ≥100 candles for valid feature calculation
    actual_folds = min(n_folds, total // min_per_fold)

    if actual_folds < 1:
        msg = (f"Not enough data ({total} candles) for {n_folds} fold(s) — "
               f"need at least {min_per_fold * n_folds}")
        _log(f"[WFV] {msg}")
        return {"error": msg, "n_folds": 0, "folds": [], "aggregate": {}}

    if actual_folds < n_folds:
        _log(f"[WFV] Requested {n_folds} folds but only {actual_folds} fit in "
             f"{total} candles (min {min_per_fold} per fold)")

    fold_size    = total // actual_folds
    active_keys  = [k for k, v in CONFIG["ACTIVE_MODELS"].items() if v]
    folds_out: list[dict] = []

    for i in range(actual_folds):
        start        = i * fold_size
        end          = start + fold_size if i < actual_folds - 1 else total
        fold_candles = candles[start:end]

        _log(f"[WFV] Fold {i+1}/{actual_folds}: rows {start}–{end} "
             f"({len(fold_candles)} candles)")

        _fold_i = i + 1   # capture for closure

        def _fold_cb(model_key: str, done: int, total_m: int,
                     _fi=_fold_i, _nf=actual_folds) -> None:
            if progress_cb:
                progress_cb(_fi, _nf, done, total_m)

        model_results = run_all_backtests(
            fold_candles, cfg, log_fn=_log, progress_cb=_fold_cb,
        )

        try:
            start_ts = int(fold_candles[0]["start"])
            end_ts   = int(fold_candles[-1]["start"])
        except (IndexError, KeyError):
            start_ts = end_ts = 0

        fold_serial: dict[str, Any] = {}
        for key, r in model_results.items():
            if r.error:
                fold_serial[key] = {"error": r.error}
            else:
                pf = r.profit_factor
                fold_serial[key] = {
                    "net_pnl":       round(r.net_pnl, 2),
                    "total_trades":  r.total_trades,
                    "win_rate":      round(r.win_rate, 4),
                    "profit_factor": round(pf if pf != float("inf") else 9999.0, 3),
                    "max_drawdown":  round(r.max_drawdown, 4),
                }

        folds_out.append({
            "fold_num": i + 1,
            "candles":  len(fold_candles),
            "start_ts": start_ts,
            "end_ts":   end_ts,
            "results":  fold_serial,
        })

    # ── Aggregate per model ───────────────────────────────────────────────────
    aggregate: dict[str, Any] = {}
    for key in active_keys:
        pnls, wrs, dds = [], [], []
        for fold in folds_out:
            r = fold["results"].get(key, {})
            if r and not r.get("error"):
                pnls.append(r["net_pnl"])
                wrs.append(r["win_rate"])
                dds.append(r["max_drawdown"])
        if not pnls:
            continue
        pos_folds = sum(1 for p in pnls if p > 0)
        aggregate[key] = {
            "avg_pnl":        round(sum(pnls) / len(pnls), 2),
            "avg_win_rate":   round(sum(wrs) / len(wrs), 4),
            "avg_drawdown":   round(sum(dds) / len(dds), 4),
            "consistency":    round(pos_folds / len(pnls), 4),
            "total_pnl":      round(sum(pnls), 2),
            "best_fold_pnl":  round(max(pnls), 2),
            "worst_fold_pnl": round(min(pnls), 2),
            "n_folds":        len(pnls),
        }

    _log(f"[WFV] Done — {actual_folds} folds, "
         f"{len(aggregate)} models aggregated")
    return {
        "n_folds":   actual_folds,
        "folds":     folds_out,
        "aggregate": aggregate,
    }


def run_multi_coin_backtests(
    symbols:     list[str],
    cfg:         BacktestConfig,
    log_fn:      Callable[[str], None] | None = None,
    progress_cb: Callable[[str, int, int, int, int], None] | None = None,
) -> dict[str, dict[str, BacktestResult]]:
    """
    Run backtests for every configured coin × every active model.

    progress_cb(symbol, coin_idx, n_coins, models_done, models_total) is called
    each time a model finishes for the current coin.

    Returns  {symbol: {model_key: BacktestResult}}
    """
    _log     = log_fn or print
    from exchanges import get_exchange
    exchange = get_exchange()

    results:  dict[str, dict[str, BacktestResult]] = {}
    n_coins = len(symbols)

    for coin_idx, symbol in enumerate(symbols):
        cached = get_cached_candles(exchange.name, symbol, hours=cfg.lookback_hours)
        if len(cached) < 50:
            _log(f"[BT] {symbol}: cache empty — fetching {cfg.lookback_hours}h...")
            try:
                fetched = exchange.fetch_candles(symbol, lookback_hours=cfg.lookback_hours)
                if fetched:
                    store_candles(fetched, exchange.name, symbol)
                    cached = fetched
            except Exception as e:
                _log(f"[BT] {symbol}: fetch error: {e}")
        if not cached:
            _log(f"[BT] {symbol}: no data — skipping")
            continue
        _log(f"[BT] {symbol}: {len(cached):,} candles → running models...")

        def _model_cb(model_key: str, done: int, total: int,
                      _sym=symbol, _ci=coin_idx) -> None:
            if progress_cb:
                progress_cb(_sym, _ci, n_coins, done, total)

        coin_results = run_all_backtests(cached, cfg, log_fn=_log, progress_cb=_model_cb)
        results[symbol] = coin_results

    return results


# ─────────────────────────────────────────────────── internals ────────────────

def _candles_to_df(candles: list[dict]) -> Optional[pd.DataFrame]:
    if not candles:
        return None
    rows = []
    for c in candles:
        try:
            rows.append({
                "timestamp": int(c["start"]),
                "open":   float(c["open"]),
                "high":   float(c["high"]),
                "low":    float(c["low"]),
                "close":  float(c["close"]),
                "volume": float(c["volume"]),
            })
        except (KeyError, ValueError):
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("timestamp").set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="s", utc=True)
    return df


# Models that use a standard joblib .pkl + .predict(X) interface
_SKLEARN_COMPAT_KEYS = {
    "SKLEARN_LINEAR": "SKLEARN_MODEL",
    "XGBOOST_TREE":   "XGBOOST_MODEL",
    "LGBM_TREE":      "LGBM_MODEL",
    "CATBOOST_TREE":  "CATBOOST_MODEL",
    "RF_TREE":        "RF_MODEL",
    "ET_TREE":        "ET_MODEL",
    "ELASTIC_LINEAR": "ELASTIC_MODEL",
    "SVR_KERNEL":     "SVR_MODEL",
    "MLP_NN":         "MLP_MODEL",
}


def _get_predictions(
    model_key: str,
    X: np.ndarray,
    log_fn: Callable,
    df: Optional[pd.DataFrame] = None,
) -> Optional[np.ndarray]:
    try:
        # ── Standard sklearn-API models ───────────────────────────────────────
        if model_key in _SKLEARN_COMPAT_KEYS:
            path = CONFIG["PATHS"][_SKLEARN_COMPAT_KEYS[model_key]]
            if not os.path.exists(path):
                return None
            return load_joblib(path).predict(X).astype(np.float32)

        # ── PyTorch LSTM ──────────────────────────────────────────────────────
        elif model_key == "PYTORCH_LSTM":
            import torch
            from training import LSTMPredictor
            path = CONFIG["PATHS"]["PYTORCH_MODEL"]
            if not os.path.exists(path):
                return None
            seq_len    = CONFIG["TRAINING"]["SEQUENCE_LENGTH"]
            checkpoint = load_torch(path, map_location="cpu")
            model      = LSTMPredictor(**checkpoint["model_cfg"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            if len(X) < seq_len:
                return None

            seqs = np.stack(
                [X[i:i + seq_len] for i in range(len(X) - seq_len)], axis=0
            )
            with torch.no_grad():
                preds = model(torch.from_numpy(seqs)).numpy()

            # Pad front so length matches X
            return np.concatenate([np.zeros(seq_len, dtype=np.float32), preds])

        # ── ARIMA — fit on backtest close prices ──────────────────────────────
        elif model_key == "ARIMA_STATS":
            path = CONFIG["PATHS"]["ARIMA_MODEL"]
            if not os.path.exists(path):
                return None
            if df is None or len(df) < 30:
                return None
            try:
                from statsmodels.tsa.arima.model import ARIMA
                p = CONFIG["TRAINING"].get("ARIMA_P", 2)
                d = CONFIG["TRAINING"].get("ARIMA_D", 0)
                q = CONFIG["TRAINING"].get("ARIMA_Q", 2)
                closes   = df["close"].values.astype(float)
                log_rets = np.diff(np.log(closes))
                res      = ARIMA(log_rets, order=(p, d, q)).fit(
                    method_kwargs={"warn_convergence": False})
                # In-sample fitted values, padded to match X length
                fitted = res.fittedvalues.astype(np.float32)
                # np.diff reduced length by 1; pad front with zero twice
                return np.concatenate([np.zeros(2, dtype=np.float32), fitted])[:len(X)]
            except Exception as e:
                log_fn(f"[BT] ARIMA error: {e}")
                return None

        # ── Prophet — fit on backtest close prices ────────────────────────────
        elif model_key == "PROPHET_FB":
            path = CONFIG["PATHS"]["PROPHET_MODEL"]
            if not os.path.exists(path):
                return None
            if df is None or len(df) < 50:
                return None
            try:
                import logging as _logging
                _logging.getLogger("prophet").setLevel(_logging.WARNING)
                _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)
                from prophet import Prophet
                closes     = df["close"].values.astype(float)
                log_prices = np.log(closes)
                dates      = df.index.tz_localize(None) if df.index.tz is not None else df.index
                df_p       = pd.DataFrame({"ds": dates, "y": log_prices})
                m = Prophet(
                    daily_seasonality   = CONFIG["TRAINING"].get("PROPHET_SEASONALITY", False),
                    weekly_seasonality  = False,
                    yearly_seasonality  = False,
                    n_changepoints      = CONFIG["TRAINING"].get("PROPHET_CHANGEPOINTS", 25),
                    uncertainty_samples = 0,
                )
                m.fit(df_p)
                # Predict at each date + 1 bar ahead = in-sample yhat shift
                future  = pd.DataFrame({"ds": dates})
                fc      = m.predict(future)["yhat"].values.astype(float)
                # Convert log-price forecast to log-return (1-bar shift)
                log_ret_fc = np.diff(fc)
                # Pad to X length
                preds = np.concatenate([np.zeros(1, dtype=np.float32),
                                        log_ret_fc.astype(np.float32)])
                return preds[:len(X)]
            except Exception as e:
                log_fn(f"[BT] Prophet error: {e}")
                return None

    except Exception as e:
        log_fn(f"[BT] Error loading {model_key}: {e}")
    return None


def _kelly_size_pct(
    cfg: BacktestConfig,
    wins: int, losses: int,
    win_amt: float, loss_amt: float,
) -> float:
    """Return Kelly-adjusted position size fraction, or fall back to cfg default."""
    pt = CONFIG["PAPER_TRADING"]
    if not pt.get("KELLY_SIZING", False):
        return cfg.position_size_pct
    n = wins + losses
    if n < pt.get("KELLY_MIN_TRADES", 20):
        return cfg.position_size_pct
    win_rate = wins / n
    avg_win  = win_amt / wins   if wins   > 0 else 0.0
    avg_loss = loss_amt / losses if losses > 0 else 0.0
    if avg_loss <= 0:
        return pt.get("KELLY_MIN_PCT", 0.02)
    b = avg_win / avg_loss
    kelly = max(0.0, win_rate - (1.0 - win_rate) / b) * pt.get("KELLY_FRACTION", 0.25)
    return max(pt.get("KELLY_MIN_PCT", 0.02), min(pt.get("KELLY_MAX_PCT", 0.30), kelly))


def _simulate(
    model_key:  str,
    preds:      np.ndarray,
    closes:     np.ndarray,
    cfg:        BacktestConfig,
    atr_series: "np.ndarray | None" = None,
) -> BacktestResult:
    result   = BacktestResult(model_name=model_key, initial_cap=cfg.initial_capital)
    capital  = cfg.initial_capital
    position = None   # {"side": "LONG"|"SHORT", "entry": float, "qty": float}
    peak     = capital
    equity   = []

    # Kelly tracking (reset per simulation so each backtest is independent)
    _k_wins = 0; _k_losses = 0; _k_win_amt = 0.0; _k_loss_amt = 0.0

    for i, (pred, price) in enumerate(zip(preds, closes)):
        if not math.isfinite(price) or price <= 0:
            equity.append(capital)
            continue

        atr = float(atr_series[i]) if atr_series is not None and i < len(atr_series) else 0.0
        threshold = compute_signal_threshold(atr, price) if atr > 0 else cfg.signal_threshold
        signal = (1 if pred > threshold else (-1 if pred < -threshold else 0))
        desired = "LONG" if signal == 1 else ("SHORT" if signal == -1 else None)

        # Close if flipping direction
        if position and desired and position["side"] != desired:
            slip  = price * cfg.slippage_pct
            ex    = (price - slip) if position["side"] == "LONG" else (price + slip)
            fee   = ex * cfg.fee_pct * position["qty"]
            pnl   = ((ex - position["entry"]) if position["side"] == "LONG"
                     else (position["entry"] - ex)) * position["qty"] - fee
            capital += pnl
            result.net_pnl      += pnl
            result.total_trades += 1

            result.trade_history.append({
                "side": position["side"],
                "entry": float(position["entry"]),
                "exit": float(ex),
                "pnl": float(pnl),
                "qty": float(position["qty"])
            })

            if pnl > 0:
                result.wins         += 1
                result.gross_profit += pnl
                _k_wins += 1; _k_win_amt += pnl
            else:
                result.losses       += 1
                result.gross_loss   += abs(pnl)
                _k_losses += 1; _k_loss_amt += abs(pnl)
            position = None

        # Open new position
        if position is None and desired:
            size_pct = _kelly_size_pct(cfg, _k_wins, _k_losses, _k_win_amt, _k_loss_amt)
            trade_cap  = capital * size_pct
            slip       = price * cfg.slippage_pct
            entry      = (price + slip) if desired == "LONG" else (price - slip)
            qty        = trade_cap / entry
            fee        = entry * cfg.fee_pct * qty
            capital   -= fee
            position   = {"side": desired, "entry": entry, "qty": qty}

        # Mark-to-market equity
        if position:
            unreal = ((price - position["entry"]) if position["side"] == "LONG"
                      else (position["entry"] - price)) * position["qty"]
            total  = capital + unreal
        else:
            total = capital

        equity.append(total)
        if total > peak:
            peak = total
        dd = (peak - total) / peak if peak > 0 else 0.0
        if dd > result.max_drawdown:
            result.max_drawdown = dd

    # Close any open position at final price
    if position and len(closes) > 0:
        price = closes[-1]
        slip  = price * cfg.slippage_pct
        ex    = (price - slip) if position["side"] == "LONG" else (price + slip)
        fee   = ex * cfg.fee_pct * position["qty"]
        pnl   = ((ex - position["entry"]) if position["side"] == "LONG"
                 else (position["entry"] - ex)) * position["qty"] - fee
        capital += pnl
        result.net_pnl      += pnl
        result.total_trades += 1

        result.trade_history.append({
            "side": position["side"],
            "entry": float(position["entry"]),
            "exit": float(ex),
            "pnl": float(pnl),
            "qty": float(position["qty"])
        })

        if pnl > 0:
            result.wins         += 1
            result.gross_profit += pnl
        else:
            result.losses       += 1
            result.gross_loss   += abs(pnl)

    # Downsample equity curve to ~60 points for display
    # (Kelly vars not needed after simulation ends)
    step = max(1, len(equity) // 60)
    result.equity_curve = equity[::step]
    return result
