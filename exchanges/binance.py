"""
Binance exchange adapter.
Uses public market data APIs — no API keys required for data.
API keys are only needed if live order placement is added in the future.

REST:  GET /api/v3/klines — up to 1000 1-minute candles per request
WS:    wss://stream.binance.com:9443/stream?streams=<sym>@kline_1m/<sym>@depth5@100ms
"""
import asyncio
import json
import time

import requests
import websockets

from config import CONFIG
from exchanges.base import BaseExchange


class BinanceExchange(BaseExchange):
    name = "BINANCE"

    def __init__(self):
        self._cfg = CONFIG["BINANCE"]

    # ── Historical candles ────────────────────────────────────────────────────

    def fetch_candles(self, symbol: str, lookback_hours: int = 24) -> list[dict]:
        """
        Fetch 1-minute klines from Binance public REST API.
        Paginates automatically — each request returns up to 1000 candles.
        """
        now_ms   = int(time.time() * 1000)
        start_ms = now_ms - lookback_hours * 3_600_000
        url      = self._cfg["REST_BASE"] + "/api/v3/klines"

        all_candles = []
        batch_start = start_ms

        while batch_start < now_ms:
            params = {
                "symbol":    symbol,
                "interval":  "1m",
                "startTime": batch_start,
                "endTime":   now_ms,
                "limit":     1000,
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                raise RuntimeError(f"[BINANCE REST] {resp.status_code}: {resp.text[:200]}")
            rows = resp.json()
            if not rows:
                break

            for row in rows:
                all_candles.append({
                    "start":  str(int(row[0]) // 1000),   # ms -> seconds
                    "open":   str(row[1]),
                    "high":   str(row[2]),
                    "low":    str(row[3]),
                    "close":  str(row[4]),
                    "volume": str(row[5]),
                })

            # next batch starts right after the last candle's open time
            batch_start = int(rows[-1][0]) + 60_000
            if len(rows) < 1000:
                break
            time.sleep(0.1)

        # deduplicate and sort
        seen = {}
        for c in all_candles:
            seen[c["start"]] = c
        return sorted(seen.values(), key=lambda x: int(x["start"]))

    def fetch_products(self) -> list[dict]:
        """Fetch all actively traded USDT-quoted spot pairs from Binance."""
        url = self._cfg["REST_BASE"] + "/api/v3/exchangeInfo"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                return []
            result = []
            for s in resp.json().get("symbols", []):
                if s.get("quoteAsset") != "USDT":
                    continue
                if s.get("status") != "TRADING":
                    continue
                if s.get("isSpotTradingAllowed") is False:
                    continue
                base   = s.get("baseAsset", "")
                symbol = s.get("symbol", "")
                if base and symbol:
                    result.append({"base": base, "symbol": symbol})
            result.sort(key=lambda x: x["base"])
            return result
        except Exception as e:
            print(f"[BINANCE] fetch_products error: {e}")
            return []

    # ── Live stream ───────────────────────────────────────────────────────────

    async def stream_live(self, symbol: str, feature_engine, output_queue) -> None:
        """
        Subscribe to Binance combined stream:
          <sym>@kline_1m  — per-tick OHLC updates
          <sym>@depth5    — top-5 order book (for OBI)
        """
        from features import CandleAggregator
        # Binance kline_1m provides 1m candles. If CONFIG granularity is higher,
        # we still want to aggregate them to match training.
        # But wait, Binance @kline_1m ALREADY provides OHLC.
        # However, if we want 15m candles, we need to aggregate these 1m updates.
        
        # Let's check config for granularity.
        # Binance config doesn't have a GRANULARITY key in the provided config snippet,
        # but the main TRAINING config might.
        
        from config import CONFIG as GLOBAL_CONFIG
        granularity = GLOBAL_CONFIG.get("COINBASE", {}).get("GRANULARITY", "FIFTEEN_MINUTE")
        # For simplicity, we'll assume the same granularity as Coinbase for now, 
        # or we could look it up specifically.
        
        # Mapping for common names
        _GRAN = {"ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900}
        interval = _GRAN.get(granularity, 900)
        aggregator = CandleAggregator(interval)

        sym_lower = symbol.lower()
        streams   = f"{sym_lower}@kline_1m/{sym_lower}@depth5@100ms"
        uri       = f"{self._cfg['WS_BASE']}/stream?streams={streams}"
        print(f"[BINANCE WS] Connecting -> {uri}")

        async for ws in websockets.connect(uri, ping_interval=20, ping_timeout=10):
            try:
                print(f"[BINANCE WS] Connected — {symbol}")
                async for raw in ws:
                    try:
                        msg    = json.loads(raw)
                        stream = msg.get("stream", "")
                        data   = msg.get("data", msg)

                        if "@kline" in stream:
                            k     = data.get("k", {})
                            close = float(k.get("c", 0) or 0)
                            if close > 0:
                                candle = aggregator.update(close)
                                if candle:
                                    feature_engine.push_candle(
                                        candle["open"], candle["high"],
                                        candle["low"], candle["close"]
                                    )
                                    vec = feature_engine.get_feature_vector()
                                    if vec:
                                        output_queue.put(vec)

                        elif "@depth" in stream:
                            bids = data.get("bids", [])
                            asks = data.get("asks", [])
                            if bids and asks:
                                bid_sz = sum(float(b[1]) for b in bids)
                                ask_sz = sum(float(a[1]) for a in asks)
                                feature_engine.update_obi(bid_sz, ask_sz)

                    except Exception as e:
                        print(f"[BINANCE WS] Message error: {e}")

            except websockets.ConnectionClosed as e:
                print(f"[BINANCE WS] Disconnected ({e}) — reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"[BINANCE WS] Error: {e} — retrying in 2s...")
                await asyncio.sleep(2)
