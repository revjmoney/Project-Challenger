"""
Abstract base class for all exchange adapters.
Every exchange must implement fetch_candles() and stream_live().
Live order placement is optional (raises NotImplementedError by default).
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseExchange(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_candles(self, symbol: str, lookback_hours: int = 24) -> list[dict]:
        """
        Fetch historical 1-minute OHLCV candles.

        Returns a list of dicts ordered oldest-first, each with keys:
            start  : str  — Unix timestamp in seconds
            open   : str  — open price
            high   : str  — high price
            low    : str  — low price
            close  : str  — close price
            volume : str  — volume
        """
        ...

    @abstractmethod
    async def stream_live(
        self,
        symbol: str,
        feature_engine: Any,
        output_queue: Any,
    ) -> None:
        """
        Stream live market data.
        Pushes feature-vector dicts (from LiveFeatureEngine) into output_queue.
        Must run forever and reconnect on errors.
        """
        ...

    # ── Optional live-trading methods ─────────────────────────────────────────

    @property
    def supports_live_trading(self) -> bool:
        return False

    def place_order(self, symbol: str, side: str, size_usd: float) -> dict:
        """Place a market order. side = 'BUY' | 'SELL'. size_usd = notional."""
        raise NotImplementedError(f"Live trading not implemented on {self.name}")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError(f"Live trading not implemented on {self.name}")

    def get_best_bid(self, symbol: str) -> float:
        raise NotImplementedError(f"get_best_bid not implemented on {self.name}")

    # ── Product listing ────────────────────────────────────────────────────────

    def fetch_products(self) -> list[dict]:
        """
        Return all tradable USD-quoted spot products for this exchange.

        Returns a list of dicts: [{'base': 'BTC', 'symbol': 'BTC-USD'}, ...]
        Defaults to an empty list; exchange adapters that support it override this.
        """
        return []
