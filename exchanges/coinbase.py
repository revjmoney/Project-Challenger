"""
Coinbase Advanced Trade exchange adapter.
Wraps the existing coinbase_client logic in the BaseExchange interface.
Supports paper mode when no API keys are configured.
Supports live order placement via the Advanced Trade REST API v3.
"""
import hashlib
import hmac
import json
import math
import random
import time
import uuid
import asyncio
from datetime import datetime, timezone

import requests
import websockets

from config import CONFIG, is_demo_mode
from exchanges.base import BaseExchange
from utils.circuit_breaker import CircuitBreaker

_GRANULARITY_SECONDS = {
    "ONE_MINUTE":     60,
    "FIVE_MINUTE":    300,
    "FIFTEEN_MINUTE": 900,
    "ONE_HOUR":       3600,
    "ONE_DAY":        86400,
}

# Public Coinbase Exchange REST API — no authentication required for market data
_EXCHANGE_BASE = "https://api.exchange.coinbase.com"
_EXCHANGE_HEADERS = {"User-Agent": "challenger-bot/1.0"}

# ── Circuit breakers for resilience ────────────────────────────────────────────
_cb_public_api    = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
_cb_authenticated = CircuitBreaker(failure_threshold=5, recovery_timeout=60)


class CoinbaseExchange(BaseExchange):
    name = "COINBASE"

    def __init__(self):
        self._cfg = CONFIG["COINBASE"]

    # ── Auth ───────────────────────────────────────────────────────────────────

    def _auth_headers(self, method: str, path: str, body: str = "") -> dict:
        ts  = str(int(time.time()))
        msg = ts + method.upper() + path + body
        sig = hmac.new(
            self._cfg["API_SECRET"].encode("utf-8"),
            msg.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        return {
            "CB-ACCESS-KEY":       self._cfg["API_KEY"],
            "CB-ACCESS-SIGN":      sig,
            "CB-ACCESS-TIMESTAMP": ts,
            "Content-Type":        "application/json",
        }

    # ── Historical candles (public Exchange API — no auth required) ──────────────

    def fetch_candles(self, symbol: str, lookback_hours: int = 24) -> list[dict]:
        """
        Fetch OHLCV candles using the public Coinbase Exchange REST API.
        No API keys required — works in paper mode and live mode alike.
        Protected by circuit breaker to prevent cascading failures.
        """
        interval_sec = _GRANULARITY_SECONDS[self._cfg["GRANULARITY"]]
        page_size    = 300
        page_dur     = page_size * interval_sec
        now          = int(time.time())
        start        = now - lookback_hours * 3600

        all_candles: list[dict] = []
        chunk_start = start
        while chunk_start < now:
            chunk_end = min(chunk_start + page_dur, now)
            try:
                page = _cb_public_api.call(
                    self._fetch_page_public, symbol, interval_sec, chunk_start, chunk_end
                )
                all_candles.extend(page)
            except Exception as e:
                print(f"  [COINBASE] candle page error ({symbol}): {e} — retrying in 3s")
                time.sleep(3)
                continue
            chunk_start = chunk_end
            time.sleep(0.25)

        seen: dict[str, dict] = {}
        for c in all_candles:
            seen[c["start"]] = c
        return sorted(seen.values(), key=lambda x: int(x["start"]))

    def _fetch_page_public(
        self, product_id: str, granularity_sec: int, start: int, end: int
    ) -> list[dict]:
        """
        One page of candles from the public Coinbase Exchange API.
        Returns candles in the standard internal dict format:
            {"start": "ts", "open": "...", "high": "...", "low": "...", "close": "...", "volume": "..."}
        Exchange API returns newest-first arrays: [time, low, high, open, close, volume]
        """
        path   = f"/products/{product_id}/candles"
        params = {
            "granularity": granularity_sec,
            "start": datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
            "end":   datetime.fromtimestamp(end,   tz=timezone.utc).isoformat(),
        }
        resp = requests.get(
            _EXCHANGE_BASE + path, params=params,
            headers=_EXCHANGE_HEADERS, timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"[COINBASE EXCHANGE] {resp.status_code}: {resp.text[:200]}")
        raw = resp.json()   # [[time, low, high, open, close, volume], ...]
        candles = []
        for c in raw:
            if len(c) >= 6:
                candles.append({
                    "start":  str(int(c[0])),
                    "low":    str(c[1]),
                    "high":   str(c[2]),
                    "open":   str(c[3]),
                    "close":  str(c[4]),
                    "volume": str(c[5]),
                })
        return candles

    def _fetch_page(self, product_id, granularity, start, end) -> list[dict]:
        """Advanced Trade authenticated candle page (kept for legacy callers)."""
        path    = f"/api/v3/brokerage/market/products/{product_id}/candles"
        params  = {"start": start, "end": end, "granularity": granularity}
        headers = self._auth_headers("GET", path)
        resp    = requests.get(
            self._cfg["REST_BASE"] + path,
            params=params, headers=headers, timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"[COINBASE REST] {resp.status_code}: {resp.text[:200]}")
        return resp.json().get("candles", [])

    # ── Live stream ────────────────────────────────────────────────────────────

    async def stream_live(self, symbol: str, feature_engine, output_queue) -> None:
        """Delegates to the original coinbase_client stream (handles paper mode too)."""
        from coinbase_client import stream_coinbase_features
        await stream_coinbase_features(symbol, feature_engine, output_queue)

    # ── Live trading ───────────────────────────────────────────────────────────

    @property
    def supports_live_trading(self) -> bool:
        return not is_demo_mode()

    def place_order(self, symbol: str, side: str, size_usd: float) -> dict:
        """
        Place a market (IOC) order via Coinbase Advanced Trade v3.
        side = 'BUY' | 'SELL'
        size_usd = notional USD value
        Returns the raw API response dict.
        Protected by circuit breaker.
        """
        path            = "/api/v3/brokerage/orders"
        client_order_id = str(uuid.uuid4())

        if side.upper() == "BUY":
            order_cfg = {"market_market_ioc": {"quote_size": f"{size_usd:.2f}"}}
        else:
            bid = self.get_best_bid(symbol)
            if bid <= 0:
                raise RuntimeError(
                    f"[ORDER] Cannot place SELL for {symbol}: "
                    f"get_best_bid returned {bid}. Position NOT closed."
                )
            base_size = size_usd / bid
            order_cfg = {"market_market_ioc": {"base_size": f"{base_size:.8f}"}}

        body_dict = {
            "client_order_id":     client_order_id,
            "product_id":          symbol,
            "side":                side.upper(),
            "order_configuration": order_cfg,
        }
        body    = json.dumps(body_dict)
        headers = self._auth_headers("POST", path, body)
        
        def _place():
            resp = requests.post(
                self._cfg["REST_BASE"] + path,
                data=body, headers=headers, timeout=15,
            )
            if resp.status_code not in (200, 201):
                raise RuntimeError(f"[ORDER] {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        
        return _cb_authenticated.call(_place)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order via Advanced Trade API. Protected by circuit breaker."""
        path    = "/api/v3/brokerage/orders/batch_cancel"
        body    = json.dumps({"order_ids": [order_id]})
        headers = self._auth_headers("POST", path, body)
        
        def _cancel():
            resp = requests.post(
                self._cfg["REST_BASE"] + path,
                data=body, headers=headers, timeout=10,
            )
            return resp.status_code in (200, 201)
        
        return _cb_authenticated.call(_cancel)

    def fetch_products(self) -> list[dict]:
        """
        Fetch all tradable USD-quoted spot products using the public Exchange API.
        No authentication required — works in demo and live mode.
        Protected by circuit breaker.
        """
        def _fetch():
            resp = requests.get(
                _EXCHANGE_BASE + "/products",
                headers=_EXCHANGE_HEADERS, timeout=15,
            )
            if resp.status_code != 200:
                return []
            result = []
            for p in resp.json():
                pid    = p.get("id", "")
                status = p.get("status", "")
                if not pid.endswith("-USD"):
                    continue
                if status not in ("online", ""):
                    continue
                base = pid.replace("-USD", "")
                if not base or "-" in base:   # skip e.g. "USDC-USD"
                    continue
                result.append({"base": base, "symbol": pid})
            result.sort(key=lambda x: x["base"])
            return result
        
        try:
            return _cb_public_api.call(_fetch)
        except Exception as e:
            print(f"[COINBASE] fetch_products error: {e}")
            return []

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """
        Fetch live prices for a list of symbols using the public Exchange API.
        Returns {symbol: price}.  Skips any symbol that errors or returns 0.
        Protected by circuit breaker per symbol.
        """
        prices: dict[str, float] = {}
        for sym in symbols:
            try:
                def _fetch_price():
                    resp = requests.get(
                        f"{_EXCHANGE_BASE}/products/{sym}/ticker",
                        headers=_EXCHANGE_HEADERS, timeout=5,
                    )
                    if resp.status_code == 200:
                        price = float(resp.json().get("price", 0) or 0)
                        if price > 0:
                            return price
                    return 0.0
                
                price = _cb_public_api.call(_fetch_price)
                if price > 0:
                    prices[sym] = price
            except Exception:
                pass
            time.sleep(0.1)
        return prices

    def get_best_bid(self, symbol: str) -> float:
        """
        Current best-bid price for order-sizing purposes.
        Live mode: uses authenticated Advanced Trade ticker for accuracy.
        Paper/fallback: uses public Exchange API ticker.
        Protected by circuit breaker.
        """
        if not is_demo_mode():
            path = f"/api/v3/brokerage/market/products/{symbol}/ticker"
            try:
                def _fetch_auth_bid():
                    resp = requests.get(
                        self._cfg["REST_BASE"] + path,
                        headers=self._auth_headers("GET", path),
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        bid = float(resp.json().get("best_bid", 0) or 0)
                        if bid > 0:
                            return bid
                    return 0.0
                
                bid = _cb_authenticated.call(_fetch_auth_bid)
                if bid > 0:
                    return bid
            except Exception:
                pass

        # Public fallback
        try:
            def _fetch_public_bid():
                resp = requests.get(
                    f"{_EXCHANGE_BASE}/products/{symbol}/ticker",
                    headers=_EXCHANGE_HEADERS, timeout=5,
                )
                if resp.status_code == 200:
                    price = float(resp.json().get("price", 0) or 0)
                    return price
                return 0.0
            
            return _cb_public_api.call(_fetch_public_bid)
        except Exception:
            pass
        return 0.0

    def get_order_fill(
        self, order_id: str, max_attempts: int = 6, delay: float = 1.0
    ) -> dict | None:
        """
        Poll the order history endpoint until the order is FILLED, fails,
        or max_attempts is reached. Protected by circuit breaker.

        Returns the order dict (with filled_size, average_filled_price,
        total_fees) on a confirmed fill, or None if the order did not fill.

        IOC market orders on Coinbase resolve in milliseconds, so 6 × 1s
        gives plenty of headroom while not blocking the monitor loop long.
        """
        terminal_statuses = {"FILLED", "CANCELLED", "EXPIRED", "FAILED",
                             "UNKNOWN_ORDER_STATUS"}
        path = f"/api/v3/brokerage/orders/historical/{order_id}"
        for attempt in range(max_attempts):
            try:
                def _fetch_order():
                    resp = requests.get(
                        self._cfg["REST_BASE"] + path,
                        headers=self._auth_headers("GET", path),
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        return resp.json().get("order", {})
                    return {}
                
                order = _cb_authenticated.call(_fetch_order)
                status = order.get("status", "")
                if status == "FILLED":
                    return order
                if status in terminal_statuses:
                    print(f"[COINBASE] Order {order_id} terminal status: {status}")
                    return None
                # Still OPEN/PENDING — wait and retry
            except Exception as e:
                print(f"[COINBASE] get_order_fill attempt {attempt + 1}: {e}")
            if attempt < max_attempts - 1:
                time.sleep(delay)
        print(f"[COINBASE] Order {order_id} fill not confirmed after {max_attempts} attempts.")
        return None

    def get_taker_fee_rate(self) -> float:
        """
        Fetch the authenticated user's current taker fee rate from Coinbase.
        Returns the configured PAPER_TRADING.COINBASE_FEE_PCT default on any
        failure (paper mode, missing keys, network error, unexpected response).
        Protected by circuit breaker.
        """
        default = CONFIG["PAPER_TRADING"]["COINBASE_FEE_PCT"]
        if is_demo_mode():
            return default
        path = "/api/v3/brokerage/transaction_summary"
        try:
            def _fetch_fee():
                resp = requests.get(
                    self._cfg["REST_BASE"] + path,
                    headers=self._auth_headers("GET", path),
                    timeout=10,
                )
                if resp.status_code != 200:
                    return default
                rate = resp.json().get("fee_tier", {}).get("taker_fee_rate")
                if rate is not None:
                    return float(rate)
                return default
            
            return _cb_authenticated.call(_fetch_fee)
        except Exception:
            pass
        return default

    def get_accounts(self) -> list[dict]:
        """
        Fetch all account balances via GET /api/v3/brokerage/accounts.

        Returns a list of dicts (sorted by total value descending):
            {currency: str, available: float, hold: float, total: float}

        Returns [] in paper mode or on any error — callers must handle an
        empty list gracefully. Protected by circuit breaker.
        """
        if is_demo_mode():
            return []
        path = "/api/v3/brokerage/accounts"
        try:
            def _fetch_accounts():
                headers = self._auth_headers("GET", path)
                resp    = requests.get(
                    self._cfg["REST_BASE"] + path,
                    headers=headers, timeout=10,
                )
                if resp.status_code != 200:
                    return []
                result = []
                for acct in resp.json().get("accounts", []):
                    bal    = acct.get("available_balance", {})
                    hold   = acct.get("hold", {})
                    avail  = float(bal.get("value",  0) or 0)
                    hold_v = float(hold.get("value", 0) or 0)
                    if avail == 0 and hold_v == 0:
                        continue   # skip truly empty accounts
                    result.append({
                        "currency":  bal.get("currency", acct.get("currency", "?")),
                        "available": avail,
                        "hold":      hold_v,
                        "total":     avail + hold_v,
                    })
                # USD first, then everything else by total value descending
                result.sort(key=lambda x: (x["currency"] != "USD", -x["total"]))
                return result
            
            return _cb_authenticated.call(_fetch_accounts)
        except Exception:
            return []


# ── Synthetic fallback (used only when the public API is unreachable) ─────────

def _demo_candles(n: int, interval_sec: int) -> list[dict]:
    """GBM random-walk candles — only used when the Exchange API is totally unreachable."""
    now   = int(time.time())
    price = 65_000.0
    out   = []
    for i in range(n):
        ts    = now - (n - i) * interval_sec
        ret   = random.gauss(0, 0.0015)
        close = price * math.exp(ret)
        high  = max(price, close) * (1 + abs(random.gauss(0, 0.0005)))
        low   = min(price, close) * (1 - abs(random.gauss(0, 0.0005)))
        vol   = random.uniform(0.5, 10.0)
        out.append({
            "start":  str(ts),
            "open":   str(price),
            "high":   str(high),
            "low":    str(low),
            "close":  str(close),
            "volume": str(vol),
        })
        price = close
    return out
