"""
BulkDownloader — one-shot 365-day 15-min candle fetch for all Coinbase USD products.

Uses the public Coinbase Exchange REST API (no authentication required).
Rate-limited to avoid hammering the server.
Resumable: skips symbols whose cache is already ≥85% full.
Progress is reported via log_fn and the activity tracker.
"""
import threading
import time
from typing import Callable, Optional

from config import CONFIG
from database import store_candles, get_candle_cache_count, get_available_coins
from activity import get_tracker, DOWNLOADING, SAVING, IDLE, ERROR

_EXCHANGE_KEY       = "COINBASE"
_LOOKBACK_DAYS      = 365
_MIN_FILL_PCT       = 0.85   # skip coins already ≥85% cached
_DEFAULT_COIN_DELAY = 1.5    # seconds between coins


def _expected_candles(lookback_days: int = _LOOKBACK_DAYS) -> int:
    gran = CONFIG["COINBASE"]["GRANULARITY"]
    secs = {"ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
            "ONE_HOUR": 3600, "ONE_DAY": 86400}.get(gran, 900)
    return int(lookback_days * 86400 / secs)


class BulkDownloader:
    """
    Downloads 365 days of 15-min candles for every USD-quoted Coinbase product.
    Runs in a background daemon thread; designed to be started once at startup
    and optionally re-triggered from the UI.
    """

    def __init__(self, log_fn: Callable[[str], None] | None = None):
        self._log_fn  = log_fn or print
        self._thread:  threading.Thread | None = None
        self._stop_ev = threading.Event()

        # Public status properties read by the TUI
        self.running:      bool = False
        self.total_coins:  int  = 0
        self.done_coins:   int  = 0
        self.skipped:      int  = 0
        self.errors:       int  = 0
        self.current_coin: str  = ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self, force: bool = False) -> bool:
        """
        Start the bulk download in a background thread.
        Returns False if already running.
        force=True re-downloads even coins that appear fully cached.
        """
        if self._thread and self._thread.is_alive():
            return False
        self._stop_ev.clear()
        self._force = force
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="bulk-downloader",
        )
        self._thread.start()
        return True

    def stop(self):
        self._stop_ev.set()

    @property
    def progress_pct(self) -> float:
        if self.total_coins == 0:
            return 0.0
        return self.done_coins / self.total_coins * 100.0

    @property
    def status_line(self) -> str:
        if not self.running and self.total_coins == 0:
            return "[dim]Not started[/dim]"
        if self.running:
            coin = self.current_coin or "..."
            return (
                f"[yellow]Downloading [{self.done_coins}/{self.total_coins}] "
                f"{coin}[/yellow]  "
                f"[dim]skipped={self.skipped}  errors={self.errors}[/dim]"
            )
        if self.errors:
            return (
                f"[green]Done[/green]  "
                f"{self.done_coins}/{self.total_coins} coins  "
                f"[dim]skipped={self.skipped}[/dim]  "
                f"[red]{self.errors} errors[/red]"
            )
        return (
            f"[green]Done[/green]  "
            f"{self.done_coins}/{self.total_coins} coins  "
            f"[dim]skipped={self.skipped}[/dim]"
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run(self):
        self.running      = True
        self.done_coins   = 0
        self.skipped      = 0
        self.errors       = 0
        self.current_coin = ""
        tracker = get_tracker()
        delay   = CONFIG["DATA_CACHE"].get("BULK_DOWNLOAD_DELAY_SEC", _DEFAULT_COIN_DELAY)
        force   = getattr(self, "_force", False)

        try:
            from exchanges.coinbase import CoinbaseExchange
            exchange = CoinbaseExchange()

            # 1. Fetch product list
            self._log("[BULK] Fetching Coinbase product list...")
            products = exchange.fetch_products()
            if not products:
                self._log("[BULK] No products returned — aborting.")
                tracker.update("BULK_DL", ERROR, "No products returned")
                return

            self.total_coins = len(products)
            expected = _expected_candles()
            self._log(
                f"[BULK] {self.total_coins} USD products found.  "
                f"Target: {expected:,} candles/coin ({_LOOKBACK_DAYS}d × 15min)"
            )

            # 2. Download each coin
            for i, p in enumerate(products):
                if self._stop_ev.is_set():
                    self._log("[BULK] Download stopped.")
                    break

                symbol = p["symbol"]
                self.current_coin = symbol
                tracker.update("BULK_DL", DOWNLOADING,
                               f"[{i+1}/{self.total_coins}] {symbol}")

                # Skip if already well-cached (unless force=True)
                if not force:
                    cached = get_candle_cache_count(_EXCHANGE_KEY, symbol)
                    if cached >= int(expected * _MIN_FILL_PCT):
                        self._log(
                            f"[BULK] {symbol}: cached {cached:,}/{expected:,} — skip"
                        )
                        self.skipped  += 1
                        self.done_coins += 1
                        continue

                try:
                    candles = exchange.fetch_candles(
                        symbol, lookback_hours=_LOOKBACK_DAYS * 24
                    )
                    if candles:
                        store_candles(candles, _EXCHANGE_KEY, symbol)
                        self._log(
                            f"[BULK] [{i+1}/{self.total_coins}] {symbol}: "
                            f"{len(candles):,} candles stored"
                        )
                    else:
                        self._log(f"[BULK] {symbol}: no candles returned")
                        self.errors += 1
                except Exception as e:
                    self._log(f"[BULK] ERROR {symbol}: {e}")
                    self.errors += 1

                self.done_coins += 1

                # Rate-limit between coins
                if i < len(products) - 1 and not self._stop_ev.is_set():
                    time.sleep(delay)

            self._log(
                f"[BULK] Complete — {self.done_coins}/{self.total_coins} processed, "
                f"{self.skipped} skipped (already cached), {self.errors} errors"
            )
            tracker.update("BULK_DL", IDLE,
                           f"{self.done_coins}/{self.total_coins} done")

        except Exception as e:
            self._log(f"[BULK] Fatal error: {e}")
            tracker.update("BULK_DL", ERROR, str(e))
        finally:
            self.running      = False
            self.current_coin = ""

    def _log(self, msg: str):
        self._log_fn(msg)
