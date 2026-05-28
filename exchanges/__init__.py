"""
Exchange factory — returns the active exchange adapter based on CONFIG["EXCHANGE"].
"""
from config import CONFIG


def get_exchange():
    """Return the configured exchange adapter instance."""
    key = CONFIG.get("EXCHANGE", "COINBASE")
    if key == "BINANCE":
        from exchanges.binance import BinanceExchange
        return BinanceExchange()
    if key == "KRAKEN":
        from exchanges.kraken import KrakenExchange
        return KrakenExchange()
    from exchanges.coinbase import CoinbaseExchange
    return CoinbaseExchange()
