"""
CoinManager — multi-coin symbol mapping, tier expansion, and product caching.

Symbol formats per exchange:
  Coinbase : BTC  → BTC-USD
  Binance  : BTC  → BTCUSDT
  Kraken   : BTC  → XBT/USD  (WS)

Training tiers (single source of truth — used by training, backtesting, live trading,
and the Web GUI):
  single   — 1 coin  : BTC
  quick    — 2 coins : BTC, ETH
  standard — 4 coins : BTC, ETH, LTC, DOGE  ← default
  extended — 10 curated coins (see EXTENDED_BASES below)
  insane   — ALL available USD-quoted coins on the active exchange
"""
import random
import threading
from typing import Optional

from config import CONFIG
from database import store_available_coins, get_available_coins

# ── Fixed coin lists ──────────────────────────────────────────────────────────
# DEFAULT_BASES is the standard tier and the fallback for all other tiers.
DEFAULT_BASES: list[str] = ["BTC", "ETH", "LTC", "DOGE"]

# EXTENDED_BASES = 10 curated coins common to major exchanges.
EXTENDED_BASES: list[str] = [
    "BTC", "ETH", "LTC", "DOGE", "SOL", "XRP", "ADA", "DOT", "AVAX", "LINK"
]

# ── Tier definitions ──────────────────────────────────────────────────────────
# None  → resolve at runtime from the exchange's available-coin cache.
# list  → fixed base-coin names, always in this order (caller may shuffle).
TIER_COINS: dict[str, Optional[list[str]]] = {
    "single":   ["BTC"],
    "quick":    ["BTC", "ETH"],
    "standard": DEFAULT_BASES,    # 4 coins — default
    "extended": EXTENDED_BASES,   # 10 curated coins
    "insane":   None,             # all available USD-quoted coins on the exchange
}

# ── Kraken base-name remapping ────────────────────────────────────────────────
_KRAKEN_FWD = {"BTC": "XBT", "DOGE": "XDG"}
_KRAKEN_REV = {v: k for k, v in _KRAKEN_FWD.items()}


# ── Symbol helpers ────────────────────────────────────────────────────────────

def coin_to_symbol(base: str, exchange: str) -> str:
    """Convert a base coin name (e.g. 'BTC') to the exchange-specific symbol."""
    b  = base.upper()
    ex = exchange.upper()
    if ex == "BINANCE":
        return f"{b}USDT"
    if ex == "KRAKEN":
        kb = _KRAKEN_FWD.get(b, b)
        return f"{kb}/USD"
    # Coinbase (default)
    return f"{b}-USD"


def symbol_to_base(symbol: str, exchange: str) -> str:
    """Convert an exchange symbol back to a base coin name (e.g. 'BTC-USD' → 'BTC')."""
    ex = exchange.upper()
    s  = symbol.upper()
    if ex == "BINANCE":
        return s.removesuffix("USDT").removesuffix("BTC")
    if ex == "KRAKEN":
        base = s.replace("/USD", "").replace("USD", "")
        return _KRAKEN_REV.get(base, base)
    # Coinbase
    return s.replace("-USD", "").replace("-USDT", "")


def get_coins_for_tier(
    tier:      str,
    exchange:  str,
    shuffle:   bool = False,
) -> list[str]:
    """
    Return the list of exchange symbols for the given training tier.

    shuffle=True randomises the coin order (useful during training so the
    concatenated multi-coin dataset doesn't cluster by coin).  The set is
    always deduplicated — no coin ever appears twice.
    """
    bases = TIER_COINS.get(tier)

    if bases is None:
        # "insane" tier — use the full cached product list for this exchange.
        available = get_available_coins(exchange.upper(), max_age_hours=168)
        if available:
            symbols = list({d["symbol"] for d in available})   # deduplicate
            if shuffle:
                random.shuffle(symbols)
            return symbols
        # Cache empty — fall back to extended list
        bases = EXTENDED_BASES

    symbols = [coin_to_symbol(b, exchange) for b in bases]
    symbols = list(dict.fromkeys(symbols))   # preserve order, remove duplicates
    if shuffle:
        random.shuffle(symbols)
    return symbols


# ── Product-list refresh ──────────────────────────────────────────────────────
_refresh_lock   = threading.Lock()
_CACHE_TTL_HRS  = 24   # re-fetch product list at most once per day


def refresh_available_coins(log_fn=None) -> list[dict]:
    """
    Fetch the product list from the active exchange, store in DB, return it.
    Thread-safe; uses DB cache if < _CACHE_TTL_HRS hours old.
    Returns list of {'base': str, 'symbol': str}.
    """
    _log     = log_fn or print
    exchange = CONFIG["EXCHANGE"]

    with _refresh_lock:
        # Use cached if fresh
        cached = get_available_coins(exchange, max_age_hours=_CACHE_TTL_HRS)
        if cached:
            _log(f"[COINS] {len(cached)} cached products for {exchange}")
            return cached

        _log(f"[COINS] Fetching product list from {exchange}...")
        try:
            from exchanges import get_exchange
            ex      = get_exchange()
            products = ex.fetch_products()
            if products:
                store_available_coins(exchange, products)
                _log(f"[COINS] Cached {len(products)} products for {exchange}")
                return products
        except Exception as e:
            _log(f"[COINS] Could not fetch products: {e}")
        return []


def force_refresh_available_coins(log_fn=None) -> list[dict]:
    """Like refresh_available_coins but always re-fetches from the exchange."""
    exchange = CONFIG["EXCHANGE"]
    _log     = log_fn or print
    with _refresh_lock:
        _log(f"[COINS] Force-refreshing product list from {exchange}...")
        try:
            from exchanges import get_exchange
            ex       = get_exchange()
            products = ex.fetch_products()
            if products:
                store_available_coins(exchange, products)
                _log(f"[COINS] Refreshed {len(products)} products for {exchange}")
                return products
        except Exception as e:
            _log(f"[COINS] Could not refresh products: {e}")
        return []


# ── Convenience accessors ─────────────────────────────────────────────────────

def get_training_symbols() -> list[str]:
    """Return exchange symbols to use for the current training tier."""
    exchange = CONFIG["EXCHANGE"]
    tier     = CONFIG["COINS"]["TRAINING_TIER"]
    return get_coins_for_tier(tier, exchange)


def get_training_bases() -> list[str]:
    """Return base coin names for the current training tier (for display).
    For the 'insane' tier returns the EXTENDED_BASES as a representative sample
    (the actual runtime list comes from the exchange cache)."""
    tier  = CONFIG["COINS"]["TRAINING_TIER"]
    bases = TIER_COINS.get(tier)
    if bases is None:
        return list(EXTENDED_BASES)   # representative display list
    return list(bases)


def get_backtest_symbols() -> list[str]:
    """Return exchange symbols to use for backtesting. Matches training coins."""
    return get_training_symbols()


def get_live_symbols() -> list[str]:
    """
    Return exchange symbols to stream for live paper inference.

    The active trading coin is always first, even if the user removes it from
    the broader training/backtest tier. Additional training-tier symbols follow
    so multi-coin paper tracking still receives live exchange data.
    """
    symbols = [get_trading_symbol()]
    symbols.extend(get_training_symbols())
    return list(dict.fromkeys(s for s in symbols if s))


def get_trading_symbol() -> str:
    """Return the exchange symbol for the active trading coin."""
    from config import get_active_symbol
    return get_active_symbol()
