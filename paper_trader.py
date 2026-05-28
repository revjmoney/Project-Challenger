"""
Shared paper-trading state machine used by every model worker.
Each worker creates its own PaperTrader instance — fully independent state.
"""
from database import log_signal, log_trade, log_portfolio
from config import CONFIG

_PT = CONFIG["PAPER_TRADING"]


def _notify_trade(model_name: str, side: str, price: float, pnl: float) -> None:
    """Send a trade-close notification if configured. Never raises."""
    try:
        if not CONFIG.get("NOTIFICATIONS", {}).get("ON_TRADE", True):
            return
        from notifications import notify
        sign  = "+" if pnl >= 0 else ""
        level = "success" if pnl >= 0 else "danger"
        notify(
            f"{model_name} closed {side} @ ${price:,.2f}  —  P&L: {sign}${pnl:.2f}",
            title="Trade Closed",
            level=level,
        )
    except Exception:
        pass


class PaperTrader:
    def __init__(self, model_name: str, symbol: str = ""):
        self.model_name        = model_name
        self.symbol            = symbol   # exchange symbol this trader watches (e.g. 'BTC-USD')
        self.capital           = _PT["INITIAL_CAPITAL"]
        self.slippage_pct      = _PT["SIMULATED_SLIPPAGE_PCT"]
        self.fee_pct           = _PT["COINBASE_FEE_PCT"]
        self.position_size_pct = _PT["POSITION_SIZE_PCT"]
        self.position          = None   # dict or None
        self.realized_pnl      = 0.0
        self._snap_counter     = 0
        self._peak_capital     = _PT["INITIAL_CAPITAL"]  # for drawdown tracking
        # Kelly sizing accumulators
        self._k_wins     = 0
        self._k_losses   = 0
        self._k_win_amt  = 0.0
        self._k_loss_amt = 0.0

    # -----------------------------------------------------------------------

    def on_signal(self, signal: int, current_price: float, predicted_val: float = 0.0):
        """
        signal: 1 = go LONG, -1 = go SHORT, 0 = stay flat / hold.
        """
        if signal == 0:
            self._maybe_snapshot(current_price)
            return

        log_signal(self.model_name, signal, current_price, predicted_val, symbol=self.symbol)

        desired_side = "LONG" if signal == 1 else "SHORT"

        # Close existing position if we need to flip
        if self.position and self.position["side"] != desired_side:
            self._close_position(current_price)

        # Enter new position if flat
        if self.position is None:
            self._open_position(desired_side, current_price)

        self._maybe_snapshot(current_price)

    # -----------------------------------------------------------------------

    def _kelly_position_size(self) -> float:
        """Return Kelly-adjusted size fraction, or fixed fallback."""
        pt = CONFIG["PAPER_TRADING"]
        if not pt.get("KELLY_SIZING", False):
            return self.position_size_pct
        n = self._k_wins + self._k_losses
        if n < pt.get("KELLY_MIN_TRADES", 20):
            return self.position_size_pct
        win_rate = self._k_wins / n
        avg_win  = self._k_win_amt  / self._k_wins   if self._k_wins   > 0 else 0.0
        avg_loss = self._k_loss_amt / self._k_losses if self._k_losses > 0 else 0.0
        if avg_loss <= 0:
            return pt.get("KELLY_MIN_PCT", 0.02)
        b = avg_win / avg_loss
        kelly = max(0.0, win_rate - (1.0 - win_rate) / b) * pt.get("KELLY_FRACTION", 0.25)
        return max(pt.get("KELLY_MIN_PCT", 0.02), min(pt.get("KELLY_MAX_PCT", 0.30), kelly))

    def _open_position(self, side: str, price: float):
        trade_capital = self.capital * self._kelly_position_size()
        slip          = price * self.slippage_pct
        exec_price    = (price + slip) if side == "LONG" else (price - slip)
        qty           = trade_capital / exec_price
        fee           = exec_price * self.fee_pct * qty

        self.capital  -= fee   # entry fee comes out of capital
        self.position  = {"side": side, "entry_price": exec_price, "quantity": qty}

        log_trade(
            self.model_name,
            "BUY" if side == "LONG" else "SELL",
            price, exec_price, slip * qty, fee, qty,
            symbol=self.symbol,
        )

    def _close_position(self, price: float):
        p    = self.position
        slip = price * self.slippage_pct
        exec_price = (price - slip) if p["side"] == "LONG" else (price + slip)
        fee        = exec_price * self.fee_pct * p["quantity"]

        if p["side"] == "LONG":
            pnl = (exec_price - p["entry_price"]) * p["quantity"] - fee
        else:
            pnl = (p["entry_price"] - exec_price) * p["quantity"] - fee

        self.capital      += pnl
        self.realized_pnl += pnl
        self.position      = None

        if pnl > 0:
            self._k_wins    += 1
            self._k_win_amt += pnl
        else:
            self._k_losses   += 1
            self._k_loss_amt += abs(pnl)

        if self.capital > self._peak_capital:
            self._peak_capital = self.capital

        log_trade(
            self.model_name,
            "SELL" if p["side"] == "LONG" else "BUY",
            price, exec_price, slip * p["quantity"], fee, p["quantity"], pnl,
            symbol=self.symbol,
        )
        _notify_trade(self.model_name, p["side"], price, pnl)

    def unrealized_pnl(self, current_price: float) -> float:
        if self.position is None:
            return 0.0
        p = self.position
        if p["side"] == "LONG":
            return (current_price - p["entry_price"]) * p["quantity"]
        return (p["entry_price"] - current_price) * p["quantity"]

    def _maybe_snapshot(self, price: float):
        self._snap_counter += 1
        if self._snap_counter % 20 == 0:
            unreal = self.unrealized_pnl(price)
            log_portfolio(
                self.model_name,
                self.capital,
                self.position["side"] if self.position else None,
                self.position["quantity"] if self.position else 0.0,
                unreal,
                self.realized_pnl + unreal,
                symbol=self.symbol,
            )
