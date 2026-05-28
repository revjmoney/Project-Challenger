"""
Coinbase Advanced Trade REST + WebSocket client.
Falls back to paper-mode data if API keys are not configured.
"""
import asyncio
import hashlib
import hmac
import json
import math
import random
import time
from datetime import datetime, timezone

import requests
import websockets

from config import CONFIG, is_demo_mode
from features import LiveFeatureEngine
from activity import get_tracker, CONNECTED, RECONNECTING, DOWNLOADING, CHECKING, ERROR, IDLE

_COINBASE = CONFIG["COINBASE"]
_GRANULARITY_SECONDS = {
    "ONE_MINUTE":    60,
    "FIVE_MINUTE":   300,
    "FIFTEEN_MINUTE": 900,
    "ONE_HOUR":      3600,
    "ONE_DAY":       86400,
}


# ---------------------------------------------------------------------------
# REST — historical candles
# ---------------------------------------------------------------------------

def _auth_headers(method: str, path: str, body: str = "") -> dict:
    ts  = str(int(time.time()))
    msg = ts + method.upper() + path + body
    sig = hmac.new(
        _COINBASE["API_SECRET"].encode("utf-8"),
        msg.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    return {
        "CB-ACCESS-KEY":       _COINBASE["API_KEY"],
        "CB-ACCESS-SIGN":      sig,
        "CB-ACCESS-TIMESTAMP": ts,
        "Content-Type":        "application/json",
    }


def fetch_candles_page(product_id: str, granularity: str, start: int, end: int) -> list[dict]:
    """Fetch up to 300 candles from Coinbase REST API."""
    path   = f"/api/v3/brokerage/market/products/{product_id}/candles"
    params = {"start": start, "end": end, "granularity": granularity}
    headers = _auth_headers("GET", path)

    resp = requests.get(
        _COINBASE["REST_BASE"] + path,
        params=params,
        headers=headers,
        timeout=15,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"[REST] {resp.status_code}: {resp.text[:200]}")
    return resp.json().get("candles", [])


def fetch_historical_candles(product_id: str, granularity: str, lookback_days: int) -> list[dict]:
    """
    Pages through the public Coinbase Exchange REST API to retrieve `lookback_days` of candles.
    No authentication required.  Returns a flat list ordered oldest-first.
    """
    interval_sec  = _GRANULARITY_SECONDS[granularity]
    page_size     = 300
    page_duration = page_size * interval_sec
    now           = int(time.time())
    total_start   = now - lookback_days * 86400

    all_candles = []
    chunk_start = total_start

    # Use public Exchange API (no auth) for candle pages
    from exchanges.coinbase import CoinbaseExchange
    _ex = CoinbaseExchange()

    print(f"[REST] Fetching {lookback_days}d of {granularity} candles for {product_id}...")
    while chunk_start < now:
        chunk_end = min(chunk_start + page_duration, now)
        try:
            page = _ex._fetch_page_public(product_id, interval_sec, chunk_start, chunk_end)
            all_candles.extend(page)
            print(f"  fetched {len(page)} candles ({datetime.fromtimestamp(chunk_start, tz=timezone.utc).date()})")
        except Exception as e:
            print(f"  [WARN] page failed: {e} — retrying in 3s")
            time.sleep(3)
            continue
        chunk_start = chunk_end
        time.sleep(0.25)   # stay well within rate limits

    # Deduplicate and sort oldest-first
    seen = {}
    for c in all_candles:
        seen[c["start"]] = c
    ordered = sorted(seen.values(), key=lambda x: int(x["start"]))
    print(f"[REST] Total candles retrieved: {len(ordered)}")
    return ordered


# ---------------------------------------------------------------------------
# Paper-mode data generator
# ---------------------------------------------------------------------------

def _generate_demo_candles(lookback_days: int, interval_sec: int) -> list[dict]:
    """Synthetic BTC-like OHLCV walk for demo / offline use."""
    print("[PAPER] Generating synthetic candle data (no API keys set).")
    n     = lookback_days * 86400 // interval_sec
    now   = int(time.time())
    price = 65_000.0
    out   = []
    for i in range(n):
        ts     = now - (n - i) * interval_sec
        ret    = random.gauss(0, 0.0015)
        close  = price * math.exp(ret)
        high   = max(price, close) * (1 + abs(random.gauss(0, 0.0005)))
        low    = min(price, close) * (1 - abs(random.gauss(0, 0.0005)))
        volume = random.uniform(0.5, 10.0)
        out.append({
            "start":  str(ts),
            "open":   str(price),
            "high":   str(high),
            "low":    str(low),
            "close":  str(close),
            "volume": str(volume),
        })
        price = close
    return out


# ---------------------------------------------------------------------------
# WebSocket stream
# ---------------------------------------------------------------------------

def _ws_subscribe_msg(channel: str, product_ids: list[str]) -> str:
    ts       = str(int(time.time()))
    prod_str = ",".join(product_ids)
    message  = ts + channel + prod_str
    sig      = hmac.new(
        _COINBASE["API_SECRET"].encode("utf-8"),
        message.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    return json.dumps({
        "type":        "subscribe",
        "product_ids": product_ids,
        "channel":     channel,
        "api_key":     _COINBASE["API_KEY"],
        "timestamp":   ts,
        "signature":   sig,
    })


_EXCHANGE_WS_URI = "wss://ws-feed.exchange.coinbase.com"   # public, no auth required


async def stream_coinbase_features(
    product_id: str,
    feature_engine: LiveFeatureEngine,
    output_queue,           # multiprocessing.Queue
):
    """
    Connects to a Coinbase WebSocket and feeds live prices into the feature engine.

    Paper mode (no API keys): public Coinbase Exchange WS — ticker channel only,
                              no authentication required, real market prices.
    Live mode  (real keys):   Coinbase Advanced Trade WS — ticker + level2,
                              HMAC-authenticated for order-book data.
    """
    from features import CandleAggregator
    tracker = get_tracker()

    granularity = _COINBASE["GRANULARITY"]
    interval    = _GRANULARITY_SECONDS.get(granularity, 900)
    aggregator  = CandleAggregator(interval)

    if is_demo_mode():
        await _public_exchange_stream(product_id, feature_engine, output_queue, aggregator)
    else:
        await _advanced_trade_stream(product_id, feature_engine, output_queue, aggregator)


async def _public_exchange_stream(
    product_id: str,
    feature_engine: LiveFeatureEngine,
    output_queue,
    aggregator,
):
    """
    Connects to the PUBLIC Coinbase Exchange WebSocket (no auth).
    Subscribes to the ticker channel for real live prices.
    Message format: {"type": "ticker", "product_id": "BTC-USD", "price": "65000.00", ...}
    """
    tracker = get_tracker()
    sub_msg = json.dumps({
        "type":        "subscribe",
        "product_ids": [product_id],
        "channels":    ["ticker"],
    })

    print(f"[WS] Connecting to public Exchange WS for {product_id}...")
    tracker.update("WS_FEED", RECONNECTING,
                   f"Connecting to public Exchange WS ({product_id})...")

    async for ws in websockets.connect(
        _EXCHANGE_WS_URI, ping_interval=20, ping_timeout=10
    ):
        try:
            await ws.send(sub_msg)
            tracker.update("WS_FEED", CONNECTED,
                           f"Public Exchange WS: {product_id} (ticker)")
            tracker.update("COINBASE_API", CONNECTED, f"Public WS ({product_id})")

            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "ticker" and msg.get("product_id") == product_id:
                    price = float(msg.get("price", 0) or 0)
                    if price > 0:
                        tracker.update("WS_FEED", CONNECTED,
                                       f"Tick ${price:,.2f}  ({product_id})")
                        candle = aggregator.update(price)
                        if candle:
                            feature_engine.push_candle(
                                candle["open"], candle["high"],
                                candle["low"],  candle["close"],
                            )
                            vec = feature_engine.get_feature_vector()
                            if vec:
                                output_queue.put(vec)

        except websockets.ConnectionClosed as e:
            print(f"[WS] Public WS disconnected ({e}) — reconnecting in 5s...")
            tracker.update("WS_FEED", RECONNECTING, f"Disconnected — retrying...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[WS] Public WS error: {e} — retrying in 2s...")
            tracker.update("WS_FEED", ERROR, str(e))
            await asyncio.sleep(2)


async def _advanced_trade_stream(
    product_id: str,
    feature_engine: LiveFeatureEngine,
    output_queue,
    aggregator,
):
    """
    Authenticated Coinbase Advanced Trade WebSocket.
    Subscribes to ticker (price) + level2 (order-book imbalance).
    """
    tracker = get_tracker()
    uri = _COINBASE["WS_URI"]
    print(f"[WS] Connecting to Advanced Trade WS -> {uri}")
    tracker.update("WS_FEED", RECONNECTING, f"Connecting to {uri}...")
    tracker.update("COINBASE_API", CHECKING, "Authenticating WebSocket...")

    async for ws in websockets.connect(uri, ping_interval=20, ping_timeout=10):
        try:
            for ch in ("ticker", "level2"):
                await ws.send(_ws_subscribe_msg(ch, [product_id]))
            print(f"[WS] Subscribed: ticker + level2 for {product_id}")
            tracker.update("WS_FEED", CONNECTED,
                           f"Advanced Trade WS: ticker + level2 ({product_id})")
            tracker.update("COINBASE_API", CONNECTED,
                           f"WebSocket authenticated ({product_id})")

            async for raw in ws:
                msg     = json.loads(raw)
                channel = msg.get("channel", "")
                events  = msg.get("events", [])

                for event in events:
                    if channel == "ticker":
                        for tick in event.get("tickers", []):
                            price = float(tick.get("price", 0) or 0)
                            if price > 0:
                                tracker.update("WS_FEED", CONNECTED,
                                               f"Tick ${price:,.2f}  ({product_id})")
                                candle = aggregator.update(price)
                                if candle:
                                    feature_engine.push_candle(
                                        candle["open"], candle["high"],
                                        candle["low"],  candle["close"],
                                    )
                                    vec = feature_engine.get_feature_vector()
                                    if vec:
                                        output_queue.put(vec)

                    elif channel == "level2":
                        for change in event.get("changes", []):
                            if len(change) == 3:
                                side, _, size_str = change
                                size = float(size_str)
                                if side == "buy":
                                    feature_engine.update_obi(size, feature_engine._ask_sz)
                                elif side == "sell":
                                    feature_engine.update_obi(feature_engine._bid_sz, size)

        except websockets.ConnectionClosed as e:
            print(f"[WS] Disconnected ({e}) — reconnecting in 5s...")
            tracker.update("WS_FEED", RECONNECTING,
                           f"Disconnected ({e}) — retrying in 5s...")
            tracker.update("COINBASE_API", RECONNECTING, "WebSocket reconnecting...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[WS] Error: {e} — retrying in 2s...")
            tracker.update("WS_FEED", ERROR, str(e))
            await asyncio.sleep(2)
