"""
DataManager — rolling 24-hour OHLCV cache.

On start:  fetches the full configured window from the active exchange.
Every hour: fetches the last 2 hours (overlapping) and upserts new bars.
Cache is stored in SQLite candle_cache table so it persists between restarts.
"""
import threading
import time
from datetime import datetime
from typing import Callable

from config import CONFIG, get_active_symbol
from database import (
    store_candles, get_cached_candles,
    prune_old_candles, get_candle_cache_count,
)
from exchanges import get_exchange
from activity import get_tracker, DOWNLOADING, SAVING, PRUNING, IDLE, ERROR

_CACHE_CFG = CONFIG["DATA_CACHE"]


class DataManager:
    """Manages the rolling historical candle cache for the active exchange."""

    def __init__(self, log_fn: Callable[[str], None] | None = None):
        self._log_fn      = log_fn or print
        self._stop_ev     = threading.Event()
        self._thread: threading.Thread | None = None
        self._exchange    = get_exchange()
        self._symbol      = get_active_symbol()
        self._last_refresh: datetime | None = None
        self._lock        = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """
        Start the cache manager.  Returns immediately — the initial candle
        fetch runs in a background thread so the caller (and the TUI event
        loop) are never blocked.
        """
        self._log(f"[DATA] Starting cache ({self._exchange.name} / {self._symbol})")
        self._stop_ev.clear()
        self._thread = threading.Thread(
            target=self._initial_fetch_then_loop, daemon=True, name="data-mgr"
        )
        self._thread.start()

    def stop(self):
        self._stop_ev.set()

    def get_candles(self, hours: int = 24) -> list[dict]:
        """Return candles from the local cache ordered oldest-first."""
        return get_cached_candles(
            self._exchange.name, self._symbol,
            hours=min(hours, _CACHE_CFG["MAX_HOURS"]),
        )

    def force_refresh(self):
        """Immediately fetch fresh data (called from UI)."""
        threading.Thread(
            target=self._fetch_and_store,
            args=(2,),
            daemon=True,
            name="data-refresh",
        ).start()

    @property
    def last_refresh(self) -> datetime | None:
        return self._last_refresh

    @property
    def candle_count(self) -> int:
        return get_candle_cache_count(self._exchange.name, self._symbol)

    @property
    def exchange_name(self) -> str:
        return self._exchange.name

    @property
    def symbol(self) -> str:
        return self._symbol

    # ── Internal ───────────────────────────────────────────────────────────────

    def _fetch_and_store(self, lookback_hours: int):
        tracker = get_tracker()
        with self._lock:
            try:
                self._log(
                    f"[DATA] Fetching {lookback_hours}h from {self._exchange.name}..."
                )
                tracker.update("DATA_MGR", DOWNLOADING,
                               f"Fetching {lookback_hours}h from {self._exchange.name}...")
                candles = self._exchange.fetch_candles(
                    self._symbol, lookback_hours=lookback_hours
                )
                tracker.update("DATA_MGR", SAVING,
                               f"Storing {len(candles)} candles to SQLite...")
                store_candles(candles, self._exchange.name, self._symbol)
                tracker.update("DATA_MGR", PRUNING,
                               f"Pruning old candles (keep {_CACHE_CFG['MAX_HOURS']}h)...")
                prune_old_candles(
                    self._exchange.name, self._symbol,
                    max_hours=_CACHE_CFG["MAX_HOURS"],
                )
                with threading.Lock():
                    self._last_refresh = datetime.now()
                count = self.candle_count
                self._log(f"[DATA] Cache updated — {count} candles stored.")
                tracker.update("DATA_MGR", IDLE,
                               f"{count} candles cached  ({self._exchange.name}/{self._symbol})")
            except Exception as e:
                self._log(f"[DATA] Refresh error: {e}")
                tracker.update("DATA_MGR", ERROR, str(e))

    def _initial_fetch_then_loop(self):
        """
        Background thread: fetch the full configured window first, then enter
        the periodic 5-minute refresh loop.  Runs entirely off the main thread.
        """
        self._fetch_and_store(lookback_hours=_CACHE_CFG["MAX_HOURS"])
        interval      = _CACHE_CFG["REFRESH_INTERVAL_MIN"] * 60          # default 5 min
        refresh_hours = _CACHE_CFG.get("REFRESH_WINDOW_MIN", 30) / 60    # default 0.5h
        while not self._stop_ev.wait(timeout=interval):
            # Only pull the last 30 min on each refresh (dedup via INSERT OR IGNORE)
            self._fetch_and_store(lookback_hours=refresh_hours)

    def _log(self, msg: str):
        self._log_fn(msg)
