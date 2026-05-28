"""
Model archive — top-3 best-performing models per engine.

After every training run:
  1. Each model is scored via a full backtest on cached candle data.
  2. The model file is copied to models/archive/<engine>/ with the
     timestamp and net P&L baked into the filename.
  3. Each engine's archive is pruned to the top-3 by P&L.
  4. The best archived model for each engine is promoted to the active
     model slot (models/sklearn_model.pkl, etc.).
  5. CONFIG["LIVE_TRADING"]["ARMED_MODEL"] is auto-set to whichever
     single engine has the highest P&L in its archive.

Archive filename format:
    sklearn_20260520_143022_pnl+245.67.pkl
    xgboost_20260520_143022_pnl+312.45.pkl
    pytorch_20260520_143022_pnl+401.12.pt
"""
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Optional

# ── constants ─────────────────────────────────────────────────────────────────

_ROOT        = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR  = os.path.join(_ROOT, "models", "archive")
TOP_K        = 3   # models to keep per engine

_EXT = {
    "SKLEARN_LINEAR": ".pkl",
    "XGBOOST_TREE":   ".pkl",
    "LGBM_TREE":      ".pkl",
    "CATBOOST_TREE":  ".pkl",
    "RF_TREE":        ".pkl",
    "ET_TREE":        ".pkl",
    "ELASTIC_LINEAR": ".pkl",
    "SVR_KERNEL":     ".pkl",
    "MLP_NN":         ".pkl",
    "ARIMA_STATS":    ".pkl",
    "PROPHET_FB":     ".pkl",
    "PYTORCH_LSTM":   ".pt",
}
_PREFIX = {
    "SKLEARN_LINEAR": "sklearn",
    "XGBOOST_TREE":   "xgboost",
    "LGBM_TREE":      "lgbm",
    "CATBOOST_TREE":  "catboost",
    "RF_TREE":        "rf",
    "ET_TREE":        "et",
    "ELASTIC_LINEAR": "elastic",
    "SVR_KERNEL":     "svr",
    "MLP_NN":         "mlp",
    "ARIMA_STATS":    "arima",
    "PROPHET_FB":     "prophet",
    "PYTORCH_LSTM":   "pytorch",
}
_ALL_PREFIXES = "|".join(_PREFIX.values())
# Pattern: prefix_YYYYMMDD_HHMMSS_pnl±NNN.NN.ext
_FNAME_RE = re.compile(
    rf"^(?:{_ALL_PREFIXES})_(\d{{8}}_\d{{6}})_pnl([+-]?\d+(?:\.\d+)?)"
)


# ── public data type ──────────────────────────────────────────────────────────

@dataclass
class ArchivedModel:
    model_key:  str    # e.g. "SKLEARN_LINEAR"
    path:       str    # absolute path to the archive file
    timestamp:  str    # "20260520_143022"
    pnl:        float  # net P&L from the scoring backtest
    is_active:  bool   # True = this file is copied to the active model slot
    is_armed:   bool   # True = this engine is the ARMED live-trading model
    trades:     int = 0
    win_rate:   float = 0.0
    max_dd:     float = 0.0


# ── internal helpers ──────────────────────────────────────────────────────────

def _archive_subdir(model_key: str) -> str:
    d = os.path.join(ARCHIVE_DIR, _PREFIX[model_key])
    os.makedirs(d, exist_ok=True)
    return d


def _parse_filename(fname: str) -> Optional[tuple[str, float]]:
    """
    Parse 'sklearn_20260520_143022_pnl+245.67.pkl' into ('20260520_143022', 245.67).
    Returns None if the filename doesn't match.
    """
    m = _FNAME_RE.search(fname)
    if m:
        return m.group(1), float(m.group(2))
    return None


def _prune(model_key: str) -> None:
    """Delete archive models beyond TOP_K (keeps highest P&L ones)."""
    all_models = list_archive_for(model_key)
    if len(all_models) > TOP_K:
        keep = {m.path for m in all_models[:TOP_K]}   # list is already sorted desc
        for m in all_models:
            if m.path not in keep:
                try:
                    os.remove(m.path)
                except OSError:
                    pass


# ── public API ────────────────────────────────────────────────────────────────

def save_to_archive(
    model_key: str,
    src_path: str,
    pnl: float,
    *,
    result_obj: Optional[object] = None,
    require_improvement: bool = True,
) -> Optional[str]:
    """
    Copy *src_path* into the archive directory for *model_key*, with the
    current timestamp and *pnl* in the filename.  Prunes the engine's
    archive to TOP_K entries afterwards.

    If *require_improvement* is True (default), the model is only archived
    when its *pnl* is strictly better than every model already in the
    archive for this engine.

    If *result_obj* (BacktestResult) is provided, it also saves:
        - filename.json: summary metrics (trades, win_rate, max_drawdown)
        - filename.trades.json: full trade history
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source model not found: {src_path}")
    if not math.isfinite(pnl):
        raise ValueError(f"pnl is {pnl} — backtest produced no valid result")

    if require_improvement:
        existing = list_archive_for(model_key)
        if existing and pnl <= existing[0].pnl:
            return None

    d     = _archive_subdir(model_key)
    ext   = _EXT[model_key]
    pfx   = _PREFIX[model_key]
    ts    = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{pfx}_{ts}_pnl{pnl:+.2f}{ext}"
    dst   = os.path.join(d, fname)
    shutil.copy2(src_path, dst)
    
    # Save companion stats/trades if result_obj is provided
    if result_obj and hasattr(result_obj, "total_trades"):
        try:
            import json
            base = os.path.splitext(dst)[0]
            
            # 1. Summary stats
            stats = {
                "total_trades":  getattr(result_obj, "total_trades", 0),
                "wins":          getattr(result_obj, "wins", 0),
                "losses":        getattr(result_obj, "losses", 0),
                "win_rate":      getattr(result_obj, "win_rate", 0.0),
                "max_drawdown":  getattr(result_obj, "max_drawdown", 0.0),
                "gross_profit":  getattr(result_obj, "gross_profit", 0.0),
                "gross_loss":    getattr(result_obj, "gross_loss", 0.0),
                "net_pnl":       pnl,
            }
            with open(base + ".json", "w") as f:
                json.dump(stats, f, indent=2)
            
            # 2. Trade history (if available — we'll need to add it to BacktestResult)
            # For now, we'll store whatever is in .trade_history if it exists.
            if hasattr(result_obj, "trade_history"):
                with open(base + ".trades.json", "w") as f:
                    json.dump(getattr(result_obj, "trade_history", []), f)
        except Exception:
            pass

    _prune(model_key)
    return dst


def list_archive_for(model_key: str) -> list[ArchivedModel]:
    """
    Return all valid archived models for *model_key*, sorted by P&L descending
    (best model first).
    """
    d   = _archive_subdir(model_key)
    ext = _EXT[model_key]
    result: list[ArchivedModel] = []
    try:
        filenames = os.listdir(d)
    except OSError:
        return result
    for fname in filenames:
        if not fname.endswith(ext):
            continue
        parsed = _parse_filename(fname)
        if parsed:
            ts, pnl = parsed
            
            # Look for companion .json stats if they exist
            stats_path = os.path.join(d, fname.replace(ext, ".json"))
            trades, wr, dd = 0, 0.0, 0.0
            if os.path.exists(stats_path):
                try:
                    import json
                    with open(stats_path, "r") as f:
                        s = json.load(f)
                        trades = s.get("total_trades", 0)
                        wr     = s.get("win_rate", 0.0)
                        dd     = s.get("max_drawdown", 0.0)
                except Exception:
                    pass

            result.append(ArchivedModel(
                model_key=model_key,
                path=os.path.join(d, fname),
                timestamp=ts,
                pnl=pnl,
                is_active=False,
                is_armed=False,
                trades=trades,
                win_rate=wr,
                max_dd=dd,
            ))
    result.sort(key=lambda x: -x.pnl)
    return result


def get_archive_trade_history(model_key: str, filename: str) -> list:
    """
    Return the trade history associated with an archived model.
    Looks for a .trades.json file alongside the model.
    """
    d = _archive_subdir(model_key)
    # Remove extension and look for .trades.json
    base = os.path.splitext(filename)[0]
    path = os.path.join(d, base + ".trades.json")
    if os.path.exists(path):
        try:
            import json
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []



def list_all_archives() -> dict[str, list[ArchivedModel]]:
    """
    Return {model_key: [ArchivedModel, ...]} for all active engines.
    Annotates the top entry of each engine as is_active=True, and marks
    is_armed=True on the single best-P&L model across all engines.
    """
    from config import CONFIG  # lazy import to avoid circular dependency

    armed_key = CONFIG["LIVE_TRADING"].get("ARMED_MODEL", "")
    archives: dict[str, list[ArchivedModel]] = {}

    for key in _EXT:
        models = list_archive_for(key)
        if models:
            models[0].is_active = True
        archives[key] = models

    # Mark is_armed on the single best-overall entry
    best_pnl  = float("-inf")
    best_key  = None
    for key, models in archives.items():
        if models and models[0].pnl > best_pnl:
            best_pnl = models[0].pnl
            best_key = key
    if best_key and archives[best_key]:
        archives[best_key][0].is_armed = True

    return archives


def get_best_model_path(model_key: str) -> Optional[str]:
    """Return path to the highest-P&L archived model for *model_key*, or None."""
    models = list_archive_for(model_key)
    return models[0].path if models else None


def get_best_overall() -> Optional[tuple[str, str, float]]:
    """
    Return (model_key, archive_path, pnl) for the single highest-P&L model
    across all engines.  Returns None if all archives are empty.
    """
    best: Optional[tuple[str, str, float]] = None
    for key in _EXT:
        models = list_archive_for(key)
        if models and (best is None or models[0].pnl > best[2]):
            best = (key, models[0].path, models[0].pnl)
    return best


def select_armed_model() -> tuple[str, float, Optional[str]]:
    """
    Choose which model to arm for live trading using the preferred-model rule:

    1. Find the best archive P&L across all engines.
    2. If PREFERRED_MODEL has an archive entry AND the best model differs from
       PREFERRED_MODEL AND the gap is within PREFERRED_MODEL_TOLERANCE_PCT,
       arm the preferred model but return the best as `rival_key` so the UI
       can show a blinking notice.
    3. Otherwise arm the outright best model.

    Sets CONFIG["LIVE_TRADING"]["ARMED_MODEL"] as a side effect.
    Returns (chosen_key, chosen_pnl, rival_key_or_None).
    """
    from config import CONFIG

    preferred_key   = CONFIG["LIVE_TRADING"].get("PREFERRED_MODEL", "XGBOOST_TREE")
    tolerance_pct   = CONFIG["LIVE_TRADING"].get("PREFERRED_MODEL_TOLERANCE_PCT", 10.0)

    scores: dict[str, float] = {}
    for key in _EXT:
        models = list_archive_for(key)
        if models:
            scores[key] = models[0].pnl

    if not scores:
        return preferred_key, 0.0, None

    best_key  = max(scores, key=lambda k: scores[k])
    best_pnl  = scores[best_key]
    pref_pnl  = scores.get(preferred_key)

    rival_key: Optional[str] = None

    if pref_pnl is not None and best_key != preferred_key:
        # Gap as % of the best P&L (only meaningful when best_pnl > 0)
        if best_pnl > 0:
            gap_pct = (best_pnl - pref_pnl) / best_pnl * 100.0
        else:
            gap_pct = float("inf")

        if gap_pct <= tolerance_pct:
            # Preferred is close enough — use it and flag the rival
            chosen_key = preferred_key
            chosen_pnl = pref_pnl
            rival_key  = best_key
        else:
            # Best is materially better — switch to it
            chosen_key = best_key
            chosen_pnl = best_pnl
    elif pref_pnl is not None:
        # Preferred IS the best
        chosen_key = preferred_key
        chosen_pnl = pref_pnl
    else:
        # Preferred has no archive yet — fall back to best
        chosen_key = best_key
        chosen_pnl = best_pnl

    CONFIG["LIVE_TRADING"]["ARMED_MODEL"] = chosen_key
    return chosen_key, chosen_pnl, rival_key


def activate_archive_model(model_key: str, archive_path: str) -> bool:
    """
    Copy a specific archived model to its active slot and set it as the armed
    model, overriding auto-selection.  Returns True on success.
    """
    from config import CONFIG, save_user_settings
    if not os.path.exists(archive_path):
        return False

    paths = CONFIG["PATHS"]
    active_slots = {
        "SKLEARN_LINEAR": paths["SKLEARN_MODEL"],
        "XGBOOST_TREE":   paths["XGBOOST_MODEL"],
        "LGBM_TREE":      paths["LGBM_MODEL"],
        "CATBOOST_TREE":  paths["CATBOOST_MODEL"],
        "RF_TREE":        paths["RF_MODEL"],
        "ET_TREE":        paths["ET_MODEL"],
        "ELASTIC_LINEAR": paths["ELASTIC_MODEL"],
        "SVR_KERNEL":     paths["SVR_MODEL"],
        "MLP_NN":         paths["MLP_MODEL"],
        "ARIMA_STATS":    paths["ARIMA_MODEL"],
        "PROPHET_FB":     paths["PROPHET_MODEL"],
        "PYTORCH_LSTM":   paths["PYTORCH_MODEL"],
    }
    active_path = active_slots.get(model_key)
    if not active_path:
        return False

    os.makedirs(os.path.dirname(active_path), exist_ok=True)
    shutil.copy2(archive_path, active_path)
    CONFIG["LIVE_TRADING"]["ARMED_MODEL"]    = model_key
    CONFIG["LIVE_TRADING"]["PREFERRED_MODEL"] = model_key
    save_user_settings({"PREFERRED_MODEL": model_key})
    return True


def promote_best_models() -> Optional[tuple[str, float, Optional[str]]]:
    """
    For each engine, copy the top-P&L archived model to its active slot:
        models/sklearn_model.pkl  ← best sklearn archive entry
        models/xgboost_model.pkl  ← best xgboost archive entry
        models/pytorch_model.pt   ← best pytorch archive entry

    Then set CONFIG["LIVE_TRADING"]["ARMED_MODEL"] to the engine whose
    best-archive model has the highest P&L across all three engines.

    Returns (best_model_key, best_pnl) or None if archive is empty.
    """
    from config import CONFIG  # lazy import

    paths = CONFIG["PATHS"]
    active_slots = {
        "SKLEARN_LINEAR": paths["SKLEARN_MODEL"],
        "XGBOOST_TREE":   paths["XGBOOST_MODEL"],
        "LGBM_TREE":      paths["LGBM_MODEL"],
        "CATBOOST_TREE":  paths["CATBOOST_MODEL"],
        "RF_TREE":        paths["RF_MODEL"],
        "ET_TREE":        paths["ET_MODEL"],
        "ELASTIC_LINEAR": paths["ELASTIC_MODEL"],
        "SVR_KERNEL":     paths["SVR_MODEL"],
        "MLP_NN":         paths["MLP_MODEL"],
        "ARIMA_STATS":    paths["ARIMA_MODEL"],
        "PROPHET_FB":     paths["PROPHET_MODEL"],
        "PYTORCH_LSTM":   paths["PYTORCH_MODEL"],
    }

    for key, active_path in active_slots.items():
        best_path = get_best_model_path(key)
        if best_path and os.path.exists(best_path):
            os.makedirs(os.path.dirname(active_path), exist_ok=True)
            shutil.copy2(best_path, active_path)

    chosen_key, chosen_pnl, rival_key = select_armed_model()
    if chosen_pnl == 0.0 and not list_archive_for(chosen_key):
        return None
    return chosen_key, chosen_pnl, rival_key
