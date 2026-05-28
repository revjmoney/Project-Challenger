"""
ActivityTracker — centralised real-time status registry.

Architecture
------------
Main process:   every thread (data_mgr, training, backtester, ws, live_trader)
                calls get_tracker().update(component, status, message) directly.

Worker processes (spawned, separate PIDs): cannot share Python singletons.
                They receive a `status_queue` (multiprocessing.Queue) and put
                raw dicts onto it.  BotManager's _status_router thread reads
                those dicts and forwards them into the main-process tracker.

The Activity tab in controller.py reads get_tracker() every second.
"""
import threading
import time
from collections import deque

# ── Status constants ──────────────────────────────────────────────────────────
IDLE         = "IDLE"
WAITING      = "WAITING"       # queue empty, waiting for next tick
DOWNLOADING  = "DOWNLOADING"   # fetching from remote API
SAVING       = "SAVING"        # writing to SQLite or file
LOADING      = "LOADING"       # reading model / scaler from file
INFERRING    = "INFERRING"     # running model prediction
TRAINING     = "TRAINING"      # fitting a model on data
BACKTESTING  = "BACKTESTING"   # replaying historical candles
CONNECTED    = "CONNECTED"     # WebSocket / API link is live
RECONNECTING = "RECONNECTING"  # disconnected, retrying
TRADING      = "TRADING"       # opening or closing a position
ARMED        = "ARMED"         # live-trading gate cleared
DISARMED     = "DISARMED"      # live trading off
VALIDATING   = "VALIDATING"    # checking model files / gate conditions
CHECKING     = "CHECKING"      # pinging API / verifying connectivity
PRUNING      = "PRUNING"       # deleting old rows from candle cache
COMPUTING    = "COMPUTING"     # feature engineering / scaling
ERROR        = "ERROR"         # something went wrong
DONE         = "DONE"          # completed a long-running task
OK           = "OK"            # all good / idle after success

# Canonical display order for the Activity tab
COMPONENTS = [
    "WS_FEED",
    "DATA_MGR",
    "COINBASE_API",
    "SKLEARN",
    "XGBOOST",
    "PYTORCH",
    "TRAINING",
    "BACKTESTER",
    "LIVE_TRADER",
]

# TUI markup colour per status
_STATUS_COLOR = {
    IDLE:         "dim",
    WAITING:      "dim",
    DOWNLOADING:  "yellow",
    SAVING:       "yellow",
    LOADING:      "yellow",
    INFERRING:    "cyan",
    TRAINING:     "cyan",
    BACKTESTING:  "cyan",
    CONNECTED:    "green",
    RECONNECTING: "yellow",
    TRADING:      "bold blue",
    ARMED:        "bold red",
    DISARMED:     "dim",
    VALIDATING:   "cyan",
    CHECKING:     "yellow",
    PRUNING:      "yellow",
    COMPUTING:    "cyan",
    ERROR:        "bold red",
    OK:           "green",
}


class ActivityTracker:
    """
    Per-process singleton.  In the main process this is the live feed for the
    Activity tab.  In worker processes it is unused — workers use status_queue.
    """
    _instance = None
    _creation_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._lock = threading.Lock()
                    inst._components: dict[str, dict] = {
                        c: {"status": IDLE, "message": "—", "ts": time.time()}
                        for c in COMPONENTS
                    }
                    inst._log: deque = deque(maxlen=500)
                    cls._instance = inst
        return cls._instance

    # ── write API ─────────────────────────────────────────────────────────────

    def update(self, component: str, status: str, message: str = "") -> None:
        """Update a component's status and append to the rolling log."""
        with self._lock:
            self._components[component] = {
                "status":  status,
                "message": message,
                "ts":      time.time(),
            }
            self._log.append({
                "ts":        time.time(),
                "component": component,
                "status":    status,
                "message":   message,
            })

    # ── read API ──────────────────────────────────────────────────────────────

    def get_components(self) -> list[dict]:
        """All components in canonical order as dicts {name, status, message, ts}."""
        with self._lock:
            return [
                {"name": c, **self._components.get(
                    c, {"status": IDLE, "message": "—", "ts": 0}
                )}
                for c in COMPONENTS
            ]

    def get_log(self, n: int = 120) -> list[dict]:
        """Most recent n log entries."""
        with self._lock:
            return list(self._log)[-n:]

    @staticmethod
    def color_for(status: str) -> str:
        return _STATUS_COLOR.get(status, "white")


def get_tracker() -> ActivityTracker:
    """Return the singleton ActivityTracker for this process."""
    return ActivityTracker()
