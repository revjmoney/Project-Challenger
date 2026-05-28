"""
LiveTrader — gates and executes real market orders via Coinbase Advanced Trade.

Safety requirements (both must be met before arming):
  1. The chosen paper-trading model must have been running for at least
     MIN_PAPER_HOURS (default 24h).
  2. That model's cumulative paper P&L must be positive (above MIN_PAPER_PNL_PCT).

When armed, the LiveTrader monitors the model_signals table and mirrors every
signal as a real Coinbase market order, sized by POSITION_SIZE_PCT × capital
(capped at MAX_POSITION_USD).

Live trades are logged to the live_trades SQLite table.
"""
import threading
import time
from datetime import datetime
from typing import Callable, Optional

from config import CONFIG, get_active_symbol
from database import (
    get_model_stats,
    get_paper_trading_start_time,
    get_latest_signal_id,
    get_signals_after,
    log_live_trade,
    save_live_position,
    load_live_position,
    clear_live_position,
)
from activity import get_tracker, ARMED, DISARMED, TRADING, CHECKING, ERROR, IDLE


# ─────────────────────────────────────────────── eligibility gate ─────────────

class LiveTradingGate:
    """
    Checks whether the preconditions for live trading are satisfied.
    Call .check() to get (eligible: bool, reason: str).
    """

    def __init__(self):
        lt  = CONFIG["LIVE_TRADING"]
        pt  = CONFIG["PAPER_TRADING"]
        self.model_name  = lt["ARMED_MODEL"]
        self.min_hours   = lt["MIN_PAPER_HOURS"]
        self.min_pnl_pct = lt["MIN_PAPER_PNL_PCT"]
        self._init_cap   = pt["INITIAL_CAPITAL"]
        self.symbol      = get_active_symbol()

    def check(self) -> tuple[bool, str]:
        if not CONFIG["LIVE_TRADING"].get("ENABLED", False):
            return False, "Live trading is disabled. Set LIVE_TRADING.ENABLED=True in config.py"

        self.symbol = get_active_symbol()

        # Duration check
        start_ts = get_paper_trading_start_time(self.model_name, self.symbol)
        if start_ts is None:
            return False, f"No paper trades found for {self.model_name} / {self.symbol}. Start paper trading first."

        hours = (time.time() - start_ts) / 3600
        if hours < self.min_hours:
            remaining = self.min_hours - hours
            return False, (
                f"Need {remaining:.1f} more hours of paper trading "
                f"({hours:.1f}h / {self.min_hours}h elapsed)"
            )

        # P&L check
        stats = get_model_stats(self.model_name, self.symbol)
        if not stats or not stats[0]:
            return False, "No completed paper trades yet"

        pnl       = (stats[3] or 0.0)
        pnl_pct   = pnl / self._init_cap * 100
        hours_str = f"{hours:.1f}h"

        if pnl_pct <= self.min_pnl_pct:
            return False, (
                f"Paper P&L {pnl_pct:+.2f}% <= minimum {self.min_pnl_pct:.2f}%. "
                f"Keep paper trading."
            )

        return True, (
            f"Eligible — {self.model_name} / {self.symbol}: "
            f"{hours_str} paper, P&L {pnl_pct:+.2f}%"
        )

    def status_lines(self) -> list[tuple[bool, str]]:
        """Returns list of (passed, description) for each requirement."""
        lt   = CONFIG["LIVE_TRADING"]
        pt   = CONFIG["PAPER_TRADING"]
        out  = []

        # Duration
        self.symbol = get_active_symbol()
        start = get_paper_trading_start_time(self.model_name, self.symbol)
        if start:
            hours  = (time.time() - start) / 3600
            passed = hours >= self.min_hours
            out.append((passed, f"Paper trading {self.symbol} >= {self.min_hours}h  ({hours:.1f}h elapsed)"))
        else:
            out.append((False, f"Paper trading {self.symbol} >= {self.min_hours}h  (no trades yet)"))

        # P&L
        stats = get_model_stats(self.model_name, self.symbol)
        if stats and stats[0]:
            pnl     = (stats[3] or 0.0)
            pnl_pct = pnl / pt["INITIAL_CAPITAL"] * 100
            passed  = pnl_pct > self.min_pnl_pct
            out.append((passed, f"Positive paper P&L  ({pnl_pct:+.2f}%)"))
        else:
            out.append((False, "Positive paper P&L  (no closed trades)"))

        # Config flag
        enabled = lt.get("ENABLED", False)
        out.append((enabled, "LIVE_TRADING.ENABLED = True in config.py"))

        return out


# ─────────────────────────────────────────────── live trader ──────────────────

class LiveTrader:
    """
    Polls model_signals table and mirrors signals as real Coinbase orders.
    Runs in its own daemon thread when armed.
    """

    def __init__(self, log_fn: Callable[[str], None] | None = None):
        lt = CONFIG["LIVE_TRADING"]
        pt = CONFIG["PAPER_TRADING"]
        self._log_fn        = log_fn or print
        self._gate          = LiveTradingGate()
        self._model_name    = lt["ARMED_MODEL"]
        self._pos_size_pct  = lt["POSITION_SIZE_PCT"]
        self._max_usd       = lt["MAX_POSITION_USD"]
        self._init_cap      = pt["INITIAL_CAPITAL"]

        self._armed         = False
        self._stop_ev       = threading.Event()
        self._lock          = threading.Lock()   # guards _position and _capital
        self._thread: Optional[threading.Thread] = None
        self._position      = None     # {"side", "entry", "qty", "size_usd"}
        self._capital       = float(self._init_cap)
        self._last_signal_id = 0
        self._symbol        = get_active_symbol()

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def is_armed(self) -> bool:
        return self._armed

    def arm(self) -> tuple[bool, str]:
        """Attempt to arm live trading. Starts the monitoring thread if eligible."""
        eligible, reason = self._gate.check()
        if not eligible:
            return False, reason

        from exchanges.coinbase import CoinbaseExchange
        exchange = CoinbaseExchange()
        if not exchange.supports_live_trading:
            return False, "Coinbase API keys not configured"

        self._symbol         = get_active_symbol()
        self._armed          = True
        self._last_signal_id = get_latest_signal_id(self._model_name, self._symbol)

        # Restore any position that was open before a crash/restart
        saved = load_live_position()
        if saved:
            saved_model, saved_pos, saved_cap = saved
            self._position = saved_pos
            self._capital  = saved_cap
            self._log(
                f"[LIVE] WARNING: Restored open position from previous session — "
                f"{saved_pos['side']} {saved_pos['qty']:.6f} {self._symbol} @ "
                f"${saved_pos['entry']:,.2f} (model: {saved_model}). "
                f"Verify this matches your actual Coinbase position."
            )
        else:
            self._log("[LIVE] No saved position found — starting flat.")

        # Sync capital to the real Coinbase USD available balance.
        # This is more authoritative than the paper-trading INITIAL_CAPITAL.
        self._sync_capital(exchange)

        self._stop_ev.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="live-trader"
        )
        self._thread.start()
        get_tracker().update("LIVE_TRADER", ARMED,
                             f"ARMED — monitoring {self._model_name} / {self._symbol}")
        self._log(f"[LIVE] *** ARMED *** — {reason}")
        self._log(f"[LIVE] Monitoring {self._model_name} {self._symbol} signals for real orders.")
        return True, reason

    def disarm(self):
        self._armed = False
        self._stop_ev.set()
        get_tracker().update("LIVE_TRADER", DISARMED, "Disarmed — paper trading only.")
        self._log("[LIVE] Disarmed — paper trading only.")

    def gate_status(self) -> list[tuple[bool, str]]:
        return self._gate.status_lines()

    def emergency_close(self) -> str:
        """
        Force-close any open live Coinbase position immediately.
        Called by BotManager.emergency_stop() before shutting everything down.
        Acquires the position lock so it cannot race with _on_signal.
        Returns a human-readable status string.
        """
        with self._lock:
            if not self._position:
                return "No open live position to close."
            try:
                from exchanges.coinbase import CoinbaseExchange
                exchange = CoinbaseExchange()
                symbol   = self._symbol or get_active_symbol()
                price    = exchange.get_best_bid(symbol)
                if price <= 0:
                    return (
                        "ERROR: Could not fetch current price from Coinbase — "
                        "position NOT closed. Close it manually on Coinbase.com."
                    )
                self._close_position(exchange, symbol, price)
                return "Live position force-closed successfully."
            except Exception as e:
                self._log(f"[LIVE E-STOP] emergency_close error: {e}")
                return f"ERROR: {e} — close the position manually on Coinbase.com."

    # ── Signal monitoring loop ─────────────────────────────────────────────────

    def _monitor_loop(self):
        tracker = get_tracker()
        while not self._stop_ev.wait(timeout=2.0):
            if not self._armed:
                break
            try:
                tracker.update("LIVE_TRADER", CHECKING,
                               f"Polling {self._model_name} / {self._symbol} signals...")
                rows = get_signals_after(self._model_name, self._last_signal_id, self._symbol)
                for row_id, direction, price, symbol in rows:
                    self._last_signal_id = row_id
                    if direction != 0:
                        self._on_signal(direction, price, symbol)
                if not rows:
                    tracker.update("LIVE_TRADER", ARMED,
                                   f"Armed — waiting for {self._model_name} / {self._symbol} signal")
            except Exception as e:
                tracker.update("LIVE_TRADER", ERROR, str(e))
                self._log(f"[LIVE] Monitor error: {e}")

    def _on_signal(self, signal: int, price: float, symbol: str = ""):
        with self._lock:
            self._on_signal_locked(signal, price, symbol)

    def _on_signal_locked(self, signal: int, price: float, signal_symbol: str = ""):
        """Called only from _on_signal, which already holds self._lock."""
        try:
            from exchanges.coinbase import CoinbaseExchange
            exchange = CoinbaseExchange()
            symbol   = signal_symbol or self._symbol or get_active_symbol()
            desired  = "LONG" if signal == 1 else "SHORT"

            # Fetch current market price — the signal trigger price can be
            # several minutes stale by the time this worker fires.
            live_price = exchange.get_best_bid(symbol)
            if live_price <= 0:
                self._log(
                    f"[LIVE] Cannot fetch current price from Coinbase "
                    f"(got {live_price}) — skipping signal."
                )
                return

            # Close existing position if flipping
            if self._position and self._position["side"] != desired:
                self._close_position(exchange, symbol, live_price)

            # Open new LONG position only — Coinbase spot cannot short.
            # A SHORT signal closes any open LONG (above) then goes flat.
            if self._position is None and desired == "SHORT":
                self._log(
                    "[LIVE] SHORT signal — going flat. "
                    "Coinbase spot does not support shorting."
                )
                return

            if self._position is None:
                size_usd = min(
                    self._capital * self._pos_size_pct,
                    self._max_usd,
                )
                side = "BUY"
                resp = exchange.place_order(symbol, side, size_usd)
                oid  = (resp.get("order_id")
                        or resp.get("success_response", {}).get("order_id", ""))

                if not oid:
                    self._log("[LIVE] No order_id returned by Coinbase — position NOT opened.")
                    return

                # Verify the order actually filled before tracking any position
                fill = exchange.get_order_fill(oid)
                if fill is None:
                    self._log(
                        f"[LIVE] Order {oid} did not fill (IOC rejected or timed out) "
                        f"— position NOT opened."
                    )
                    return

                fill_price = float(fill.get("average_filled_price") or live_price)
                qty        = float(fill.get("filled_size") or
                                   (size_usd / fill_price if fill_price > 0 else 0))
                fee        = float(fill.get("total_fees") or
                                   size_usd * CONFIG["PAPER_TRADING"]["COINBASE_FEE_PCT"])

                self._position = {
                    "side": desired, "entry": fill_price,
                    "qty": qty, "size_usd": size_usd,
                }
                self._capital -= fee
                save_live_position(self._model_name, self._position, self._capital)
                get_tracker().update(
                    "LIVE_TRADER", TRADING,
                    f"OPENED {desired} {qty:.6f} {symbol} @ ${fill_price:,.2f}  (${size_usd:.2f})"
                )
                log_live_trade(
                    model_name=self._model_name,
                    exchange="COINBASE",
                    symbol=symbol,
                    side=side,
                    order_id=oid,
                    requested_price=live_price,
                    fill_price=fill_price,
                    size_usd=size_usd,
                    quantity=qty,
                    fee_paid=fee,
                    realized_pnl=0.0,
                    status="OPEN",
                )
                self._log(
                    f"[LIVE] OPENED {desired} {qty:.6f} @ ${fill_price:.2f} "
                    f"(signal ${price:.2f} → live ${live_price:.2f})  order={oid}"
                )

        except Exception as e:
            self._log(f"[LIVE ERROR] _on_signal: {e}")

    def _close_position(self, exchange, symbol: str, price: float):
        p    = self._position
        side = "SELL" if p["side"] == "LONG" else "BUY"
        oid  = ""
        fill_price = price
        fill_qty   = p["qty"]

        try:
            resp = exchange.place_order(symbol, side, p["qty"] * price)
            oid  = (resp.get("order_id")
                    or resp.get("success_response", {}).get("order_id", ""))

            if not oid:
                self._log("[LIVE] WARNING: No order_id on close — position may still be open on Coinbase.")
            else:
                fill = exchange.get_order_fill(oid)
                if fill is None:
                    self._log(
                        f"[LIVE] WARNING: Close order {oid} fill unconfirmed — "
                        f"position may still be open on Coinbase. Check manually."
                    )
                else:
                    fill_price = float(fill.get("average_filled_price") or price)
                    fill_qty   = float(fill.get("filled_size") or p["qty"])
        except Exception as e:
            self._log(f"[LIVE] Close order error: {e}")

        fee = float(
            # Use Coinbase-reported fee if we got a fill, else estimate
            0 or fill_price * fill_qty * CONFIG["PAPER_TRADING"]["COINBASE_FEE_PCT"]
        )
        pnl = ((fill_price - p["entry"]) if p["side"] == "LONG"
               else (p["entry"] - fill_price)) * fill_qty - fee

        self._capital  += pnl
        self._position  = None
        clear_live_position()

        # Resync capital to the real Coinbase USD balance after the close settles
        self._sync_capital(exchange)

        log_live_trade(
            model_name=self._model_name,
            exchange="COINBASE",
            symbol=symbol,
            side=side,
            order_id=oid,
            requested_price=price,
            fill_price=fill_price,
            size_usd=fill_qty * fill_price,
            quantity=fill_qty,
            fee_paid=fee,
            realized_pnl=pnl,
            status="CLOSED",
        )
        self._log(
            f"[LIVE] CLOSED {p['side']} @ ${fill_price:.2f} "
            f"(requested ${price:.2f})  PnL: {pnl:+.2f}  order={oid}"
        )

    def _sync_capital(self, exchange) -> None:
        """
        Overwrite self._capital with the real available USD balance from Coinbase.
        Called on arm() and after every position close so drift never compounds.
        """
        try:
            accounts      = exchange.get_accounts()
            usd_available = next(
                (a["available"] for a in accounts if a["currency"] == "USD"), None
            )
            if usd_available is not None:
                old = self._capital
                self._capital = usd_available
                self._log(
                    f"[LIVE] Capital synced from Coinbase: "
                    f"${usd_available:,.2f} available USD "
                    f"(was ${old:,.2f})."
                )
            else:
                self._log(
                    f"[LIVE] WARNING: No USD account found on Coinbase — "
                    f"using tracked capital ${self._capital:,.2f}. "
                    f"Verify your account has sufficient funds before trading."
                )
        except Exception as e:
            self._log(
                f"[LIVE] WARNING: Capital sync failed ({e}) — "
                f"using tracked capital ${self._capital:,.2f}."
            )

    def _log(self, msg: str):
        self._log_fn(msg)
