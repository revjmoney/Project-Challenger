"""
BotManager — single source of truth for all process lifecycle.
Used by both controller.py (TUI) and main.py (headless).

New in this version:
  - DataManager integration (24h rolling cache, hourly refresh)
  - run_backtest() method (runs all active models against cached data)
  - LiveTrader integration (arm/disarm live Coinbase orders)
  - Exchange-aware WebSocket stream via exchanges factory
"""
import asyncio
import math
import multiprocessing
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Callable

import psutil

from config import CONFIG, is_demo_mode, get_active_symbol
from database import init_db
from features import LiveFeatureEngine
from exchanges import get_exchange
from data_manager import DataManager
from backtester import BacktestConfig, run_all_backtests, run_multi_coin_backtests
from live_trader import LiveTrader
from activity import get_tracker
from workers.sklearn_worker import run_sklearn_worker
from workers.xgboost_worker import run_xgboost_worker
from workers.pytorch_worker import run_pytorch_worker
from workers.generic_worker import run_generic_worker
from workers.arima_worker import run_arima_worker
from workers.prophet_worker import run_prophet_worker

# ── Model key → (path config key, worker factory) ─────────────────────────────
# Each entry maps ACTIVE_MODELS key to (PATHS key, worker callable(dq, sq)).
# Generic models are wrapped with partial so run_generic_worker receives the
# model_key and model_path as leading positional arguments.
_M = CONFIG["PATHS"]

WORKER_FN: dict[str, Callable] = {
    "SKLEARN_LINEAR": run_sklearn_worker,
    "XGBOOST_TREE":   run_xgboost_worker,
    "PYTORCH_LSTM":   run_pytorch_worker,
    "LGBM_TREE":      partial(run_generic_worker, "LGBM_TREE",      _M["LGBM_MODEL"]),
    "CATBOOST_TREE":  partial(run_generic_worker, "CATBOOST_TREE",  _M["CATBOOST_MODEL"]),
    "RF_TREE":        partial(run_generic_worker, "RF_TREE",        _M["RF_MODEL"]),
    "ET_TREE":        partial(run_generic_worker, "ET_TREE",        _M["ET_MODEL"]),
    "ELASTIC_LINEAR": partial(run_generic_worker, "ELASTIC_LINEAR", _M["ELASTIC_MODEL"]),
    "SVR_KERNEL":     partial(run_generic_worker, "SVR_KERNEL",     _M["SVR_MODEL"]),
    "MLP_NN":         partial(run_generic_worker, "MLP_NN",         _M["MLP_MODEL"]),
    "ARIMA_STATS":    run_arima_worker,
    "PROPHET_FB":     run_prophet_worker,
}

# Active-model key → ActivityTracker component name
MODEL_KEY_TO_COMPONENT: dict[str, str] = {
    "SKLEARN_LINEAR": "SKLEARN",
    "XGBOOST_TREE":   "XGBOOST",
    "PYTORCH_LSTM":   "PYTORCH",
    "LGBM_TREE":      "LGBM",
    "CATBOOST_TREE":  "CATBOOST",
    "RF_TREE":        "RF",
    "ET_TREE":        "ET",
    "ELASTIC_LINEAR": "ELASTIC",
    "SVR_KERNEL":     "SVR",
    "MLP_NN":         "MLP",
    "ARIMA_STATS":    "ARIMA",
    "PROPHET_FB":     "PROPHET",
}

# Active-model key → PATHS key (used for file-existence checks and archiving)
_MODEL_PATH_KEYS: dict[str, str] = {
    "SKLEARN_LINEAR": "SKLEARN_MODEL",
    "XGBOOST_TREE":   "XGBOOST_MODEL",
    "PYTORCH_LSTM":   "PYTORCH_MODEL",
    "LGBM_TREE":      "LGBM_MODEL",
    "CATBOOST_TREE":  "CATBOOST_MODEL",
    "RF_TREE":        "RF_MODEL",
    "ET_TREE":        "ET_MODEL",
    "ELASTIC_LINEAR": "ELASTIC_MODEL",
    "SVR_KERNEL":     "SVR_MODEL",
    "MLP_NN":         "MLP_MODEL",
    "ARIMA_STATS":    "ARIMA_MODEL",
    "PROPHET_FB":     "PROPHET_MODEL",
}


class _TaggedQueue:
    """
    Thin wrapper around a multiprocessing.Queue that injects a 'symbol' key
    into every dict placed on the queue.  Used so the single shared_q can
    carry candle vectors from multiple parallel WebSocket streams and workers
    can route by coin without any changes to the exchange stream code.
    """
    def __init__(self, real_queue: multiprocessing.Queue, symbol: str):
        self._q      = real_queue
        self._symbol = symbol

    def put_nowait(self, item):
        if isinstance(item, dict):
            item = {**item, "symbol": self._symbol}
        try:
            self._q.put_nowait(item)
        except Exception:
            pass


def models_are_trained() -> bool:
    """Return True when every active model's file (and the shared scaler) exists."""
    paths  = CONFIG["PATHS"]
    active = CONFIG["ACTIVE_MODELS"]
    required = [paths["SCALER"]]
    for key, path_key in _MODEL_PATH_KEYS.items():
        if active.get(key):
            required.append(paths[path_key])
    return all(os.path.exists(p) for p in required)


class BotManager:

    def __init__(self, log_fn: Callable[[str], None] | None = None):
        self._log_fn    = log_fn or print
        self._procs:    dict[str, multiprocessing.Process] = {}
        self._queues:   dict[str, multiprocessing.Queue]   = {}
        self._shared_q:  multiprocessing.Queue | None = None
        self._status_q:  multiprocessing.Queue | None = None   # worker→main activity bridge
        self._stop_ev   = threading.Event()
        self._ws_exec:   ThreadPoolExecutor | None = None
        self._router_t:  threading.Thread   | None = None
        self._wdog_t:    threading.Thread   | None = None
        self._status_rt: threading.Thread   | None = None
        self.running    = False

        # Ensure DB tables exist immediately — on_mount queries candle_cache
        # before start() is ever called, so we can't defer this.
        init_db()

        # Data cache manager — created lazily so log_fn is available
        self.data_mgr    = DataManager(log_fn=self._log)
        self.live_trader = LiveTrader(log_fn=self._log)

    # ─────────────────────────────────────────────────────── public API ───────

    def start(self) -> None:
        if self.running:
            self._log("Bot is already running.")
            return
        init_db()

        # Ensure available coins are cached in DB
        from coin_manager import refresh_available_coins
        refresh_available_coins(log_fn=self._log)

        self._stop_ev.clear()
        self._shared_q = multiprocessing.Queue()
        self._status_q = multiprocessing.Queue()

        # Sync real Coinbase taker fee into config so all workers, paper trader,
        # backtester, and live trader use the same authoritative value.
        if not is_demo_mode():
            try:
                from exchanges.coinbase import CoinbaseExchange
                live_fee = CoinbaseExchange().get_taker_fee_rate()
                old_fee  = CONFIG["PAPER_TRADING"]["COINBASE_FEE_PCT"]
                CONFIG["PAPER_TRADING"]["COINBASE_FEE_PCT"] = live_fee
                self._log(
                    f"[FEE] Coinbase taker fee synced: {live_fee*100:.4f}% "
                    f"(was {old_fee*100:.4f}%)"
                )
            except Exception as e:
                self._log(f"[FEE] Fee sync failed ({e}) — using default "
                          f"{CONFIG['PAPER_TRADING']['COINBASE_FEE_PCT']*100:.4f}%")

        # Start data cache (fetches 24h on first call)
        self.data_mgr = DataManager(log_fn=self._log)
        self.data_mgr.start()

        active = [k for k, v in CONFIG["ACTIVE_MODELS"].items() if v]
        for key in active:
            self._spawn_worker(key)

        self._router_t = threading.Thread(
            target=self._router_loop, daemon=True, name="router"
        )
        self._router_t.start()

        # Status router: reads worker activity updates → main-process ActivityTracker
        self._status_rt = threading.Thread(
            target=self._status_router_loop, daemon=True, name="status-router"
        )
        self._status_rt.start()

        # Stream every selected coin in parallel — each WS thread tags its
        # feature vectors with the coin's symbol via _TaggedQueue before they
        # land on _shared_q, so workers can route signals per coin.
        from coin_manager import get_live_symbols
        live_symbols = get_live_symbols()
        if not live_symbols:
            live_symbols = [get_active_symbol()]  # fallback: at least the default coin

        self._ws_exec = ThreadPoolExecutor(
            max_workers=max(len(live_symbols), 1),
            thread_name_prefix="ws",
        )
        for sym in live_symbols:
            self._ws_exec.submit(self._run_ws, sym)
        self._log(
            f"[WS] Streaming {len(live_symbols)} coin(s): "
            + ", ".join(live_symbols[:8])
            + ("…" if len(live_symbols) > 8 else "")
        )

        self._wdog_t = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="watchdog"
        )
        self._wdog_t.start()

        mode = "PAPER" if is_demo_mode() else get_active_symbol()
        self.running = True
        self._log(
            f"Bot started — {CONFIG['EXCHANGE']} / {mode} — "
            f"{len(active)} model(s) active"
        )

    def stop(self) -> None:
        if not self.running:
            return
        self._log("Stopping bot...")
        self._stop_ev.set()
        if self.live_trader.is_armed:
            self.live_trader.disarm()
        self.data_mgr.stop()
        for key in list(self._procs):
            self._kill_worker(key)
        if self._ws_exec:
            self._ws_exec.shutdown(wait=False)
            self._ws_exec = None
        self.running = False
        self._log("Bot stopped.")

    def emergency_stop(self) -> None:
        """
        Emergency stop — force-closes every open paper position at the
        current market price, disarms live trading, then stops the bot.

        Position details are read from the latest portfolio_snapshots rows.
        Closing trades are written to paper_trades so the history is complete.
        """
        self._log("[E-STOP] ⚡ Emergency stop triggered — closing all positions…")

        # ── Get the latest cached price ───────────────────────────────────────
        current_price = 0.0
        try:
            conn = sqlite3.connect(CONFIG["PATHS"]["DB"], timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            row = conn.execute(
                "SELECT close FROM candle_cache "
                "WHERE exchange=? AND symbol=? ORDER BY ts DESC LIMIT 1",
                (CONFIG["EXCHANGE"], get_active_symbol()),
            ).fetchone()
            conn.close()
            if row:
                current_price = float(row[0])
        except Exception as e:
            self._log(f"[E-STOP] Could not read current price: {e}")

        if current_price <= 0:
            self._log("[E-STOP] WARNING: no cached price — P&L on forced closes will be approximate.")

        # ── Find every open position in the last portfolio snapshot ───────────
        open_positions = []
        try:
            conn = sqlite3.connect(CONFIG["PATHS"]["DB"], timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            rows = conn.execute("""
                SELECT model_name, capital, position_side, position_qty,
                       unrealized_pnl, total_pnl
                FROM portfolio_snapshots
                WHERE id IN (
                    SELECT MAX(id) FROM portfolio_snapshots GROUP BY model_name
                )
                  AND position_side IS NOT NULL
                  AND position_qty  > 0
            """).fetchall()
            conn.close()
            open_positions = rows
        except Exception as e:
            self._log(f"[E-STOP] DB error reading positions: {e}")

        # ── Force-close each position ─────────────────────────────────────────
        _pt   = CONFIG["PAPER_TRADING"]
        slip_pct = _pt["SIMULATED_SLIPPAGE_PCT"]
        fee_pct  = _pt["COINBASE_FEE_PCT"]
        initial  = _pt["INITIAL_CAPITAL"]

        from database import log_trade, log_portfolio

        for row in open_positions:
            model_name, capital, side, qty, unrealized, total_pnl = row
            qty = float(qty or 0)
            if qty <= 0:
                continue

            slip = current_price * slip_pct if current_price > 0 else 0.0
            if side == "LONG":
                exec_price = current_price - slip
                close_side = "SELL"
            else:
                exec_price = current_price + slip
                close_side = "BUY"

            fee = exec_price * fee_pct * qty if exec_price > 0 else 0.0
            pnl = float(unrealized or 0) - fee
            new_capital = float(capital or initial) + pnl

            try:
                log_trade(
                    model_name, close_side,
                    current_price, exec_price,
                    slip * qty, fee, qty, pnl,
                )
                log_portfolio(
                    model_name, new_capital,
                    None, 0.0, 0.0,
                    new_capital - initial,
                )
                self._log(
                    f"[E-STOP] {model_name}: closed {side} "
                    f"{qty:.6f} @ {exec_price:.2f}  PnL={pnl:+.2f}"
                )
            except Exception as e:
                self._log(f"[E-STOP] Error closing {model_name}: {e}")

        if not open_positions:
            self._log("[E-STOP] No open positions found.")

        # ── Close any real live Coinbase position BEFORE stopping ─────────────
        if self.live_trader._position is not None:
            self._log("[E-STOP] Closing real live Coinbase position...")
            msg = self.live_trader.emergency_close()
            self._log(f"[E-STOP] Live Coinbase: {msg}")
        else:
            self._log("[E-STOP] No open live Coinbase position.")

        # ── Stop everything ───────────────────────────────────────────────────
        self.stop()
        self._log("[E-STOP] ⚡ All positions closed. Bot stopped.")

    def start_worker(self, key: str) -> None:
        if key in self._procs and self._procs[key].is_alive():
            return
        if not self.running:
            return
        self._spawn_worker(key)

    def stop_worker(self, key: str) -> None:
        self._kill_worker(key)
        self._log(f"Worker {key} disabled.")

    def retrain(self, done_cb: Callable | None = None) -> None:
        was_running = self.running
        if was_running:
            self.stop()

        paths = CONFIG["PATHS"]
        files_to_clear = [paths["SCALER"], paths["CV_FOLDS"]] + [
            paths[pk] for pk in _MODEL_PATH_KEYS.values()
        ]
        for p in files_to_clear:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        self._log("Model files cleared. Starting training...")

        def _train():
            import sys
            sys.stdout = _LogRedirect(self._log_fn)
            try:
                from training import train_all_models
                train_all_models()
                self._log("Retrain complete.")
            except Exception as e:
                self._log(f"[ERROR] Retrain failed: {e}")
                sys.stdout = sys.__stdout__
                if done_cb:
                    done_cb()
                if was_running:
                    self.start()
                return
            finally:
                sys.stdout = sys.__stdout__
            # Score models and archive them with P&L baked into the filename
            self._archive_trained_models()
            if done_cb:
                done_cb()
            if was_running:
                self.start()

        threading.Thread(target=_train, daemon=True, name="retrain").start()

    def run_backtest(
        self,
        cfg:         BacktestConfig | None = None,
        coins:       list[str] | None = None,
        done_cb:     Callable | None = None,
        progress_cb: Callable | None = None,
    ) -> None:
        """
        Run backtest in a background thread.
        Calls done_cb({'results': flat_dict, 'by_coin': per_coin_dict}) on finish.

        coins: list of exchange symbols to backtest (e.g. ['BTC-USD', 'ETH-USD']).
               Defaults to CONFIG['COINS']['BACKTEST_COINS'] converted to symbols.
        """
        from coin_manager import get_backtest_symbols
        bt_cfg   = cfg   or BacktestConfig.from_config()
        bt_coins = coins or get_backtest_symbols()

        def _bt():
            try:
                n_active = sum(1 for v in CONFIG["ACTIVE_MODELS"].values() if v)
                self._log(
                    f"[BT] Starting ({n_active} model(s) × {len(bt_coins)} coin(s)) — "
                    f"{bt_cfg.lookback_hours}h / "
                    f"threshold={bt_cfg.signal_threshold} / "
                    f"capital=${bt_cfg.initial_capital:,.0f}"
                )

                by_coin = run_multi_coin_backtests(bt_coins, bt_cfg, log_fn=self._log,
                                                   progress_cb=progress_cb)

                # Flatten: pick first coin's results as primary (for backward compat)
                flat: dict = {}
                for _sym, model_results in by_coin.items():
                    for mkey, r in model_results.items():
                        if mkey not in flat:
                            flat[mkey] = r   # first coin wins for the summary
                        if r.error:
                            self._log(f"[BT] {_sym}/{mkey}: ERROR — {r.error}")
                        else:
                            pf = (f"{r.profit_factor:.2f}"
                                  if r.profit_factor != float("inf") else "inf")
                            self._log(
                                f"[BT] {_sym}/{mkey}: {r.total_trades} trades  "
                                f"WR={r.win_rate:.0%}  PF={pf}  "
                                f"PnL={r.net_pnl:+.2f}  DD={r.max_drawdown:.1%}"
                            )

                self._log("[BT] Complete.")
                if done_cb:
                    done_cb({"results": flat, "by_coin": by_coin})

            except Exception as e:
                self._log(f"[BT ERROR] {e}")
                if done_cb:
                    done_cb({"results": {}, "by_coin": {}})

        threading.Thread(target=_bt, daemon=True, name="backtest").start()

    def arm_live_trading(self) -> tuple[bool, str]:
        return self.live_trader.arm()

    def disarm_live_trading(self) -> None:
        self.live_trader.disarm()

    def reset_db(self) -> None:
        from database import _connect
        conn = _connect()
        for tbl in ("paper_trades", "model_signals", "portfolio_snapshots"):
            conn.execute(f"DELETE FROM {tbl}")
        conn.commit()
        conn.close()
        self._log("Database cleared — all paper-trading data wiped.")

    def fetch_fresh_data(self) -> None:
        self.data_mgr.force_refresh()
        self._log("Data refresh triggered.")

    def worker_status(self) -> dict[str, bool]:
        return {k: p.is_alive() for k, p in self._procs.items()}

    def worker_details(self) -> list[dict]:
        return [
            {"key": k, "pid": p.pid, "alive": p.is_alive(), "name": p.name}
            for k, p in self._procs.items()
        ]

    # ─────────────────────────────────────────────────────── private ──────────

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_fn(f"[{ts}]  {msg}")

    def _spawn_worker(self, key: str) -> None:
        q    = multiprocessing.Queue()
        self._queues[key] = q
        proc = multiprocessing.Process(
            target=WORKER_FN[key], args=(q, self._status_q), name=key, daemon=True
        )
        proc.start()
        self._apply_affinity(proc.pid)
        self._procs[key] = proc
        self._log(f"Worker {key} started  (PID {proc.pid})")

    def _kill_worker(self, key: str) -> None:
        proc = self._procs.pop(key, None)
        self._queues.pop(key, None)
        if proc and proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)

    def _apply_affinity(self, pid: int) -> None:
        try:
            p        = psutil.Process(pid)
            total    = psutil.cpu_count(logical=True)
            headroom = CONFIG["CPU"]["OS_HEADROOM_CORES"]
            if total and total > headroom:
                p.cpu_affinity(list(range(total - headroom)))
        except Exception:
            pass

    def _router_loop(self) -> None:
        while not self._stop_ev.is_set():
            if self._shared_q and not self._shared_q.empty():
                vec = self._shared_q.get_nowait()
                for q in list(self._queues.values()):
                    try:
                        q.put_nowait(vec)
                    except Exception:
                        pass
            time.sleep(0.001)

    def _run_ws(self, symbol: str) -> None:
        """
        Run the live WebSocket feed for *symbol*.
        Each stream wraps _shared_q in a _TaggedQueue so that feature vectors
        carry the coin symbol before reaching the router and worker queues.
        Automatically reconnects with exponential back-off (5 s → 120 s cap).
        """
        tracker    = get_tracker()
        tagged_q   = _TaggedQueue(self._shared_q, symbol)
        backoff    = 5   # seconds; doubles on each failure, capped at 120

        while not self._stop_ev.is_set():
            exchange = get_exchange()
            engine   = LiveFeatureEngine()
            loop     = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                tracker.update("ws_feed", "CONNECTED", f"Streaming {symbol}")
                loop.run_until_complete(
                    exchange.stream_live(symbol, engine, tagged_q)
                )
                # stream_live returned cleanly → stop requested
                break
            except Exception as e:
                if self._stop_ev.is_set():
                    break
                self._log(
                    f"[WS] Stream error: {e} — reconnecting in {backoff}s"
                )
                tracker.update(
                    "ws_feed", "RECONNECTING",
                    f"Reconnect in {backoff}s — {str(e)[:60]}"
                )
                # Wait for backoff duration but wake immediately if stop_ev fires
                for _ in range(backoff):
                    if self._stop_ev.is_set():
                        break
                    time.sleep(1)
                backoff = min(backoff * 2, 120)
            finally:
                loop.close()

        tracker.update("ws_feed", "IDLE", "Stream stopped")

    def _status_router_loop(self) -> None:
        """Read status dicts posted by worker processes and forward to the main-process ActivityTracker."""
        tracker = get_tracker()
        while not self._stop_ev.is_set():
            if self._status_q and not self._status_q.empty():
                try:
                    item = self._status_q.get_nowait()
                    tracker.update(
                        item.get("component", "?"),
                        item.get("status", "OK"),
                        item.get("message", ""),
                    )
                except Exception:
                    pass
            time.sleep(0.02)

    def _archive_trained_models(self) -> None:
        """
        Run a scoring backtest on each freshly-trained model, then archive
        the model file with its P&L in the filename.  Promotes the highest
        P&L model per engine to the active slot and sets ARMED_MODEL to
        whichever engine had the best overall P&L.

        Safe to call even if models for some engines are missing — those
        engines are simply skipped.
        """
        from model_archive import save_to_archive, promote_best_models, list_archive_for

        self._log("[ARCHIVE] Scoring models for archive...")
        candles = self.data_mgr.get_candles(
            hours=CONFIG["BACKTESTING"]["LOOKBACK_HOURS"]
        )
        if not candles:
            self._log(
                "[ARCHIVE] No candle data available — skipping archive. "
                "Start the bot and let the data cache warm up, then retrain."
            )
            return

        bt_cfg = BacktestConfig.from_config()
        try:
            results = run_all_backtests(
                candles, bt_cfg, log_fn=self._log,
                model_to_component=MODEL_KEY_TO_COMPONENT
            )
        except Exception as e:
            self._log(f"[ARCHIVE] Backtest scoring failed: {e}")
            return

        paths = CONFIG["PATHS"]
        model_paths = {key: paths[pk] for key, pk in _MODEL_PATH_KEYS.items()}

        for key, result in results.items():
            if result.error:
                self._log(f"[ARCHIVE] {key}: backtest error — {result.error}")
                continue
            src = model_paths.get(key)
            if not src or not os.path.exists(src):
                self._log(f"[ARCHIVE] {key}: model file not found — skipping.")
                continue
            pnl = result.net_pnl
            if not math.isfinite(pnl):
                self._log(
                    f"[ARCHIVE] {key}: backtest returned nan/inf P&L "
                    f"({pnl}) — skipping archive. Check model predictions."
                )
                continue
            try:
                arch_path = save_to_archive(key, src, pnl, result_obj=result)
                if arch_path is None:
                    self._log(
                        f"[ARCHIVE] {key}: P&L {pnl:+.2f} — "
                        f"no improvement over archive best, skipped."
                    )
                else:
                    self._log(
                        f"[ARCHIVE] {key}: new best saved — "
                        f"{os.path.basename(arch_path)}"
                    )
            except Exception as e:
                self._log(f"[ARCHIVE] {key}: archive write failed — {e}")

        promoted = promote_best_models()
        if promoted:
            chosen_key, chosen_pnl, rival_key = promoted
            msg = (
                f"[ARCHIVE] Armed → {chosen_key}  (P&L {chosen_pnl:+.2f})"
            )
            if rival_key:
                rival_models = list_archive_for(rival_key) if rival_key else []
                rival_pnl    = rival_models[0].pnl if rival_models else 0.0
                msg += (
                    f"  *** NOTE: {rival_key} has higher archive P&L "
                    f"({rival_pnl:+.2f}) but is within tolerance — "
                    f"preferred model {chosen_key} selected."
                )
            self._log(msg)
        else:
            self._log("[ARCHIVE] Archive empty — no models promoted.")

    def _watchdog_loop(self) -> None:
        """
        Runs every 10 s while the bot is running.  Two responsibilities:

        1. Worker process watchdog — if any worker process has died,
           restart it and post the event to the ActivityTracker so it
           appears in the Activity panel.

        2. Stale data feed detection — if the newest candle in SQLite is
           older than STALE_THRESHOLD_SECS (default 5 min), post an ERROR
           to the ActivityTracker so the operator knows the feed is dead.
           Posts a recovery message when fresh candles arrive again.
        """
        STALE_THRESHOLD_SECS = 300   # 5 minutes without a new candle = stale
        tracker      = get_tracker()
        stale_warned: set[str] = set()   # symbols currently flagged as stale

        tracker.update("watchdog", "OK", "Monitoring workers & data feed")

        while not self._stop_ev.wait(timeout=10):
            if not self.running:
                break

            # ── Worker process watchdog ──────────────────────────────────────
            for key in list(self._procs):
                proc = self._procs.get(key)
                if proc and not proc.is_alive():
                    self._log(f"[WATCHDOG] Worker {key} crashed — restarting…")
                    tracker.update("watchdog", "RECONNECTING",
                                   f"{key} crashed — restarting")
                    self._spawn_worker(key)
                    self._log(f"[WATCHDOG] Worker {key} restarted (PID "
                              f"{self._procs[key].pid})")
                    tracker.update("watchdog", "OK",
                                   f"{key} restarted successfully")

            # ── Stale data feed detection (all streamed symbols) ─────────────
            try:
                from coin_manager import get_live_symbols
                watch_syms = get_live_symbols() or [get_active_symbol()]
                conn = sqlite3.connect(CONFIG["PATHS"]["DB"], timeout=2.0)
                for sym in watch_syms:
                    row = conn.execute(
                        "SELECT ts FROM candle_cache "
                        "WHERE exchange=? AND symbol=? ORDER BY ts DESC LIMIT 1",
                        (CONFIG["EXCHANGE"], sym),
                    ).fetchone()
                    if row:
                        age = time.time() - float(row[0])
                        if age > STALE_THRESHOLD_SECS:
                            if sym not in stale_warned:
                                self._log(
                                    f"[WATCHDOG] ⚠ Stale feed — no new candle "
                                    f"for {age / 60:.1f} min ({sym})"
                                )
                                tracker.update(
                                    "data_feed", "ERROR",
                                    f"Stale — {age / 60:.1f} min ({sym})"
                                )
                                stale_warned.add(sym)
                        else:
                            if sym in stale_warned:
                                self._log(
                                    f"[WATCHDOG] ✓ Data feed recovered — {sym}"
                                )
                                stale_warned.discard(sym)
                conn.close()
                if not stale_warned:
                    tracker.update("data_feed", "OK",
                                   f"Live — {len(watch_syms)} coin(s)")
            except Exception:
                pass   # DB may not exist yet on first run


class _LogRedirect:
    def __init__(self, fn):
        self._fn = fn
    def write(self, s):
        s = s.rstrip("\n")
        if s:
            self._fn(s)
    def flush(self):
        pass
