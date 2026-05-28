"""
Kraken exchange adapter.
Uses public market data APIs — no API keys required for data.

REST:  GET /0/public/OHLC — up to 720 1-minute candles per request
WS:    wss://ws.kraken.com/v2 — Kraken WebSocket API v2
"""
import asyncio
import json
import time

import requests
import websockets

from config import CONFIG
from exchanges.base import BaseExchange


class KrakenExchange(BaseExchange):
    name = "KRAKEN"

    def __init__(self):
        self._cfg = CONFIG["KRAKEN"]

    # ── Historical candles ────────────────────────────────────────────────────

    def fetch_candles(self, symbol: str, lookback_hours: int = 24) -> list[dict]:
        """
        Fetch 1-minute OHLC data from Kraken public REST API.
        Kraken returns up to 720 candles per call — paginate for 24h (1440 min).
        `symbol` is ignored here; uses CONFIG["KRAKEN"]["REST_PAIR"].
        """
        pair     = self._cfg["REST_PAIR"]
        url      = self._cfg["REST_BASE"] + "/0/public/OHLC"
        now      = int(time.time())
        start    = now - lookback_hours * 3600
        since    = start

        all_candles = []

        while True:
            params = {"pair": pair, "interval": 1, "since": since}
            resp   = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                raise RuntimeError(f"[KRAKEN REST] {resp.status_code}: {resp.text[:200]}")

            body = resp.json()
            if body.get("error"):
                raise RuntimeError(f"[KRAKEN REST] API error: {body['error']}")

            result  = body.get("result", {})
            pair_key = next((k for k in result if k != "last"), None)
            if not pair_key:
                break

            rows = result[pair_key]
            if not rows:
                break

            for row in rows:
                ts = int(row[0])
                if ts >= now:
                    continue
                all_candles.append({
                    "start":  str(ts),
                    "open":   str(row[1]),
                    "high":   str(row[2]),
                    "low":    str(row[3]),
                    "close":  str(row[4]),
                    "volume": str(row[6]),   # row[5]=vwap, row[6]=volume
                })

            last_ts = int(result.get("last", 0))
            if last_ts <= since or last_ts >= now:
                break
            since = last_ts
            time.sleep(0.5)   # Kraken rate limits: 1 req/s on public endpoints

        # deduplicate and sort
        seen = {}
        for c in all_candles:
            seen[c["start"]] = c
        return sorted(seen.values(), key=lambda x: int(x["start"]))

    def fetch_products(self) -> list[dict]:
        """Fetch all USD-quoted spot pairs from Kraken."""
        url = self._cfg["REST_BASE"] + "/0/public/AssetPairs"
        # Kraken XBT → BTC, XDG → DOGE reverse-map
        _rev = {"XBT": "BTC", "XDG": "DOGE", "XLM": "XLM", "XRP": "XRP"}
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                return []
            pairs = resp.json().get("result", {})
            result = []
            for _key, info in pairs.items():
                wsname = info.get("wsname", "")
                if not wsname.endswith("/USD"):
                    continue
                base_raw = wsname.replace("/USD", "")
                base     = _rev.get(base_raw, base_raw)
                result.append({"base": base, "symbol": wsname})
            result.sort(key=lambda x: x["base"])
            return result
        except Exception as e:
            print(f"[KRAKEN] fetch_products error: {e}")
            return []

    # ── Live stream ───────────────────────────────────────────────────────────

    async def stream_live(self, symbol: str, feature_engine, output_queue) -> None:
        """
        Connect to Kraken WebSocket API v2.
        Subscribe to ticker (for price) and book (for OBI).
        """
        from features import CandleAggregator
        from config import CONFIG as GLOBAL_CONFIG
        granularity = GLOBAL_CONFIG.get("COINBASE", {}).get("GRANULARITY", "FIFTEEN_MINUTE")
        _GRAN = {"ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900}
        interval = _GRAN.get(granularity, 900)
        aggregator = CandleAggregator(interval)

        uri = self._cfg["WS_URI"]
        print(f"[KRAKEN WS] Connecting -> {uri}")

        async for ws in websockets.connect(uri, ping_interval=20, ping_timeout=10):
            try:
                # Subscribe to ticker and order book
                await ws.send(json.dumps({
                    "method": "subscribe",
                    "params": {"channel": "ticker", "symbol": [symbol]},
                }))
                await ws.send(json.dumps({
                    "method": "subscribe",
                    "params": {"channel": "book", "symbol": [symbol], "depth": 10},
                }))
                print(f"[KRAKEN WS] Subscribed: ticker + book for {symbol}")

                async for raw in ws:
                    try:
                        msg     = json.loads(raw)
                        channel = msg.get("channel", "")
                        mtype   = msg.get("type", "")
                        data    = msg.get("data", [])

                        if channel == "ticker" and mtype in ("update", "snapshot"):
                            for tick in data:
                                price = float(tick.get("last", 0) or 0)
                                if price > 0:
                                    candle = aggregator.update(price)
                                    if candle:
                                        feature_engine.push_candle(
                                            candle["open"], candle["high"],
                                            candle["low"], candle["close"]
                                        )
                                        vec = feature_engine.get_feature_vector()
                                        if vec:
                                            output_queue.put(vec)

                        elif channel == "book" and mtype in ("snapshot", "update"):
                            for book in data:
                                bids = book.get("bids", [])
                                asks = book.get("asks", [])
                                if bids and asks:
                                    bid_sz = sum(float(b.get("qty", 0)) for b in bids)
                                    ask_sz = sum(float(a.get("qty", 0)) for a in asks)
                                    feature_engine.update_obi(bid_sz, ask_sz)

                    except Exception as e:
                        print(f"[KRAKEN WS] Message error: {e}")

            except websockets.ConnectionClosed as e:
                print(f"[KRAKEN WS] Disconnected ({e}) — reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"[KRAKEN WS] Error: {e} — retrying in 2s...")
                await asyncio.sleep(2)
