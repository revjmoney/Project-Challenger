# Coinbase Advanced Trade API — Project Challenger Reference

**API Version**: Advanced Trade REST v3 / WebSocket v1  
**Base URL**: `https://api.coinbase.com`  
**WebSocket**: `wss://advanced-trade-ws.coinbase.com`  
**Auth method**: HMAC-SHA256 (legacy API keys) — see §Authentication below  

This document covers every Coinbase Advanced Trade endpoint that is used
by Project Challenger **today**, plus every endpoint that is relevant to
planned features.  Grouped by function area.

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Accounts & Balances](#2-accounts--balances) ✅ implemented
3. [Products & Market Data](#3-products--market-data) ✅ implemented
4. [Historical Candles](#4-historical-candles) ✅ implemented
5. [Orders](#5-orders) ✅ implemented
6. [Fills (Trade History)](#6-fills-trade-history) 🔜 planned
7. [Portfolios](#7-portfolios) 🔜 planned
8. [Fees](#8-fees) 🔜 planned
9. [WebSocket Feeds](#9-websocket-feeds) ✅ implemented
10. [Rate Limits](#10-rate-limits)
11. [Error Codes](#11-error-codes)
12. [Key Management Notes](#12-key-management-notes)

---

## 1. Authentication

All private endpoints (accounts, orders, fills, fees) require three headers:

| Header | Value |
|---|---|
| `CB-ACCESS-KEY` | Your API key string |
| `CB-ACCESS-SIGN` | HMAC-SHA256 signature (see below) |
| `CB-ACCESS-TIMESTAMP` | Unix timestamp (seconds) as string |
| `Content-Type` | `application/json` |

### Signature algorithm

```python
import hashlib, hmac, time

def _auth_headers(api_key, api_secret, method, path, body=""):
    ts  = str(int(time.time()))
    msg = ts + method.upper() + path + body   # body = "" for GET
    sig = hmac.new(
        api_secret.encode("utf-8"),
        msg.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    return {
        "CB-ACCESS-KEY":       api_key,
        "CB-ACCESS-SIGN":      sig,
        "CB-ACCESS-TIMESTAMP": ts,
        "Content-Type":        "application/json",
    }
```

`path` must be the **exact URL path** (e.g. `/api/v3/brokerage/accounts`)
including any query-string **if** the endpoint has required query params in
the signature.  For most REST endpoints, query params are NOT included in
the message; for WebSocket JWT auth they are.

### Public endpoints (no auth needed)

- `GET /api/v3/brokerage/market/products` — product list
- `GET /api/v3/brokerage/market/products/{id}/candles` — candles
- `GET /api/v3/brokerage/market/products/{id}/ticker` — best bid/ask

---

## 2. Accounts & Balances

### `GET /api/v3/brokerage/accounts`
**Auth required**: yes  
**Used by**: `CoinbaseExchange.get_accounts()` → balance panel in Settings tab

Returns all accounts (one per currency) for the authenticated user.

**Query params** (optional):

| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | int | 49 | Max accounts per page (max 250) |
| `cursor` | string | — | Pagination cursor from previous response |

**Response shape**:
```json
{
  "accounts": [
    {
      "uuid": "a1b2c3...",
      "name": "BTC Wallet",
      "currency": "BTC",
      "available_balance": { "value": "0.12345678", "currency": "BTC" },
      "default":   true,
      "active":    true,
      "created_at": "2023-01-01T00:00:00Z",
      "updated_at": "2024-06-01T12:00:00Z",
      "deleted_at": null,
      "type": "ACCOUNT_TYPE_CRYPTO",
      "ready": true,
      "hold": { "value": "0.00000000", "currency": "BTC" }
    }
  ],
  "has_next": false,
  "cursor": "",
  "size": 1
}
```

**Key fields used by the project**:
- `available_balance.value` — free balance (can be traded)
- `available_balance.currency` — currency ticker
- `hold.value` — amount locked in open orders

**Notes**:
- Pagination: if `has_next` is `true`, pass the returned `cursor` value as
  the `cursor` query param in the next call to get remaining accounts.
- Accounts with zero balance are returned but the project filters them out.

---

### `GET /api/v3/brokerage/accounts/{account_uuid}`
**Auth required**: yes  
**Used by**: not yet implemented

Returns a single account by its UUID.  Useful for polling a specific wallet
balance without fetching all accounts.

---

## 3. Products & Market Data

### `GET /api/v3/brokerage/market/products`
**Auth required**: no (public)  
**Used by**: not yet (useful for symbol discovery)

Returns the full list of tradeable products.

**Query params**:

| Param | Description |
|---|---|
| `product_type` | `SPOT` (default) or `FUTURE` |
| `contract_expiry_type` | `UNKNOWN_CONTRACT_EXPIRY_TYPE` / `EXPIRING` / `PERPETUAL` |

**Response**: array of product objects each containing:
- `product_id` — e.g. `"BTC-USD"`
- `base_currency_id`, `quote_currency_id`
- `base_min_size`, `base_max_size`, `quote_min_size`
- `status` — `"online"` / `"offline"` / `"delisted"`
- `price`, `volume_24h`, `price_percentage_change_24h`

**Project use case**: validate that `CONFIG["COINBASE"]["PRODUCT_ID"]` is a
live symbol before starting the bot.

---

### `GET /api/v3/brokerage/market/products/{product_id}/ticker`
**Auth required**: no (public)  
**Used by**: `CoinbaseExchange.get_best_bid()` — used by `place_order(SELL)`
to calculate base-currency quantity

**Response**:
```json
{
  "trades": [...],
  "best_bid": "67842.50",
  "best_ask": "67843.00"
}
```

---

## 4. Historical Candles

### `GET /api/v3/brokerage/market/products/{product_id}/candles`
**Auth required**: no (public)  
**Used by**: `CoinbaseExchange._fetch_page()` → `fetch_candles()` →
`data_manager.py` rolling cache

**Query params** (all required):

| Param | Type | Description |
|---|---|---|
| `start` | int | Start of window, Unix epoch seconds |
| `end` | int | End of window, Unix epoch seconds |
| `granularity` | string | See table below |

**Supported granularities**:

| Value | Interval | Max candles per request |
|---|---|---|
| `ONE_MINUTE` | 60 s | 300 |
| `FIVE_MINUTE` | 300 s | 300 |
| `FIFTEEN_MINUTE` | 900 s | 300 |
| `ONE_HOUR` | 3600 s | 300 |
| `SIX_HOUR` | 21600 s | 300 |
| `ONE_DAY` | 86400 s | 300 |

**Response**:
```json
{
  "candles": [
    {
      "start":  "1700000000",
      "low":    "67200.00",
      "high":   "67900.00",
      "open":   "67400.00",
      "close":  "67800.00",
      "volume": "12.34567"
    }
  ]
}
```

**Pagination strategy used by the project**:  
The 300-candle limit means long lookback windows require multiple requests.
`fetch_candles()` chunks the requested window into 300-candle pages and
concatenates them with deduplication on `start` timestamp.  A 0.25 s sleep
between pages avoids rate-limiting.

---

## 5. Orders

### `POST /api/v3/brokerage/orders`
**Auth required**: yes  
**Used by**: `CoinbaseExchange.place_order()` → `live_trader.py`

Creates a new order.  The project exclusively uses **Market IOC** (immediate-
or-cancel) orders to avoid leaving resting orders on the book.

**Request body**:
```json
{
  "client_order_id": "<uuid>",
  "product_id":      "BTC-USD",
  "side":            "BUY",
  "order_configuration": {
    "market_market_ioc": {
      "quote_size": "50.00"
    }
  }
}
```

For **BUY**: `quote_size` = USD notional amount (e.g. `"50.00"` → buy $50 of BTC).  
For **SELL**: `base_size` = BTC quantity (e.g. `"0.00074"` → sell 0.00074 BTC).

**Response** (success):
```json
{
  "order_id":         "abc123",
  "client_order_id":  "<your uuid>",
  "success":          true,
  "success_response": {
    "order_id":        "abc123",
    "product_id":      "BTC-USD",
    "side":            "BUY",
    "client_order_id": "<uuid>"
  }
}
```

**Other `order_configuration` types** (not yet used):

| Type | Description |
|---|---|
| `limit_limit_gtc` | Limit order, good-till-cancelled |
| `limit_limit_gtd` | Limit order, good-till-date |
| `stop_limit_stop_limit_gtc` | Stop-limit, GTC |
| `sor_limit_ioc` | Smart-order-routing limit IOC |

---

### `GET /api/v3/brokerage/orders/historical/batch`
**Auth required**: yes  
**Used by**: not yet implemented (planned for trade history panel)

Fetches historical orders with filtering.

**Query params**:

| Param | Description |
|---|---|
| `product_id` | Filter by symbol |
| `order_status` | `OPEN`, `FILLED`, `CANCELLED`, `EXPIRED`, `FAILED` |
| `order_type` | `MARKET`, `LIMIT`, `STOP`, `STOP_LIMIT` |
| `start_date` | ISO 8601 datetime |
| `end_date` | ISO 8601 datetime |
| `limit` | Max results (max 1000) |
| `cursor` | Pagination cursor |

**Project use case**: populate a "live trade history" panel showing all real
orders placed by the bot since it was armed.

---

### `POST /api/v3/brokerage/orders/batch_cancel`
**Auth required**: yes  
**Used by**: `CoinbaseExchange.cancel_order()`

Cancels one or more orders by ID.

**Request body**:
```json
{ "order_ids": ["abc123", "def456"] }
```

**Response**:
```json
{
  "results": [
    { "order_id": "abc123", "success": true,  "failure_reason": "" },
    { "order_id": "def456", "success": false, "failure_reason": "ORDER_NOT_FOUND" }
  ]
}
```

---

## 6. Fills (Trade History)

### `GET /api/v3/brokerage/orders/historical/fills`
**Auth required**: yes  
**Used by**: 🔜 planned — "fills" panel to show actual executed prices

Returns the fill records for completed orders (each fill = one execution event).

**Query params**:

| Param | Description |
|---|---|
| `order_id` | Filter by a specific order ID |
| `product_id` | Filter by symbol |
| `start_sequence_timestamp` | ISO 8601 |
| `end_sequence_timestamp` | ISO 8601 |
| `limit` | Max 1000 |
| `cursor` | Pagination |

**Response includes per fill**:
- `entry_id`, `trade_id`, `order_id`
- `trade_time` — exact execution timestamp
- `trade_type` — `FILL` / `REVERSAL` / `CORRECTION` / `SYNTHETIC`
- `price`, `size`, `side`
- `commission` — fees paid on this fill
- `retail_portfolio_id`

**Project use case**: cross-reference live-trader DB records against actual
Coinbase fills; compute real fee drag; validate that orders were fully filled.

---

## 7. Portfolios

### `GET /api/v3/brokerage/portfolios`
**Auth required**: yes  
**Used by**: 🔜 planned — portfolio summary panel

Returns all portfolios for the account.  Advanced Trade accounts have a
default "Default" portfolio; Pro users may have multiple.

**Response per portfolio**:
- `uuid`, `name`, `type` (`DEFAULT` / `CONSUMER` / `INTX`)
- `deleted` (boolean)

---

### `GET /api/v3/brokerage/portfolios/{portfolio_uuid}`
**Auth required**: yes  
**Used by**: 🔜 planned

Returns a detailed breakdown including:
- `breakdown.portfolio` — name, type
- `breakdown.portfolio_balances` — total / available / hold in both BTC and USD
- `breakdown.spot_positions` — per-asset holdings with cost basis, unrealised PnL
- `breakdown.perp_positions` — (if applicable)
- `breakdown.futures_positions` — (if applicable)

**Key fields for the planned balance panel upgrade**:
```json
{
  "breakdown": {
    "portfolio_balances": {
      "total_balance":          { "value": "12345.67", "currency": "USD" },
      "total_futures_balance":  { "value": "0.00",     "currency": "USD" },
      "total_cash_equivalent_balance": { "value": "...", "currency": "USD" },
      "unrealized_pnl":         { "value": "-23.45",   "currency": "USD" }
    },
    "spot_positions": [
      {
        "asset":            "BTC",
        "account_uuid":     "...",
        "total_balance_fiat": "11890.12",
        "total_balance_crypto": "0.17523",
        "avg_entry_price":  "67800.00",
        "cost_basis":       { "value": "11884.74", "currency": "USD" },
        "unrealized_pnl":   { "value": "5.38",     "currency": "USD" }
      }
    ]
  }
}
```

This endpoint is richer than `GET /accounts` and is better for a holdings
panel because it includes cost basis and unrealised PnL per position.

---

## 8. Fees

### `GET /api/v3/brokerage/transaction_summary`
**Auth required**: yes  
**Used by**: 🔜 planned — fee-tier display in Settings tab

Returns the authenticated user's 30-day trading volume and current fee tier.

**Query params**: `start_date`, `end_date`, `user_native_currency`, `product_type`

**Response**:
```json
{
  "total_volume":    167432.50,
  "total_fees":      1005.12,
  "fee_tier": {
    "pricing_tier":     ">= $0",
    "usd_from":         "0",
    "usd_to":           "10000",
    "taker_fee_rate":   "0.006",
    "maker_fee_rate":   "0.004"
  },
  "goods_and_services_tax": { "rate": "0", "inclusive": false },
  "advanced_trade_only_volume": 167432.50,
  "advanced_trade_only_fees":   1005.12,
  "coinbase_pro_volume":         0,
  "coinbase_pro_fees":           0
}
```

**Project use case**: display the user's actual taker/maker fee rate in the
Settings tab so the backtest fee parameter can be pre-filled with the real
value rather than the default 0.6%.  Also shows 30-day volume for context.

---

## 9. WebSocket Feeds

**URL**: `wss://advanced-trade-ws.coinbase.com`  
**Used by**: `coinbase_client.stream_coinbase_features()` (and
`exchanges/coinbase.py` → `stream_live()`)

All subscriptions follow the same message shape:
```json
{
  "type":        "subscribe",
  "product_ids": ["BTC-USD"],
  "channel":     "<channel_name>",
  "api_key":     "<your key>",
  "timestamp":   "1700000000",
  "signature":   "<hmac sig>"
}
```

The WebSocket signature uses the same HMAC algorithm as REST but with the
message format: `timestamp + channel + product_ids_joined_by_comma`.

### Channels used by the project

| Channel | Data | Used for |
|---|---|---|
| `ticker` | Best bid, ask, last price, 24h stats | `current_price`, `obi` feature |
| `level2` | Full order book depth updates | `obi` (order book imbalance) feature |

### Channels available but not yet used

| Channel | Data | Potential use |
|---|---|---|
| `market_trades` | Individual trade executions (real fills) | More accurate price signal |
| `user` | **Your** order fills and position changes | Confirm live order completion |
| `heartbeats` | Keeps connection alive | Better reconnect logic |

### `ticker` message shape
```json
{
  "channel": "ticker",
  "client_id": "",
  "timestamp": "2024-01-01T12:00:00Z",
  "sequence_num": 1234,
  "events": [
    {
      "type": "update",
      "tickers": [
        {
          "type":              "ticker",
          "product_id":        "BTC-USD",
          "price":             "67842.50",
          "volume_24_h":       "1234.56",
          "low_52_w":          "38000.00",
          "high_52_w":         "73750.00",
          "price_percent_chg_24_h": "1.23",
          "best_bid":          "67841.00",
          "best_ask":          "67843.00",
          "best_bid_quantity": "0.05",
          "best_ask_quantity": "0.12"
        }
      ]
    }
  ]
}
```

### `level2` message shape
```json
{
  "channel": "l2_data",
  "events": [
    {
      "type":       "snapshot",
      "product_id": "BTC-USD",
      "updates": [
        { "side": "bid", "event_time": "...", "price_level": "67841.00", "new_quantity": "0.5" },
        { "side": "offer", "event_time": "...", "price_level": "67843.00", "new_quantity": "0.3" }
      ]
    }
  ]
}
```

`type` is `"snapshot"` on first connect (full book), then `"update"` for deltas.

### `user` channel — live order tracking
```json
{
  "channel": "user",
  "events": [
    {
      "type":   "snapshot",
      "orders": [
        {
          "order_id":     "abc123",
          "client_order_id": "...",
          "cumulative_quantity": "0.00074",
          "leaves_quantity":    "0.00000",
          "avg_price":          "67842.50",
          "total_fees":         "0.30",
          "status":             "FILLED",
          "product_id":         "BTC-USD",
          "creation_time":      "2024-01-01T12:00:00Z",
          "order_side":         "BUY",
          "order_type":         "MARKET"
        }
      ]
    }
  ]
}
```

This is the recommended way to confirm that a live order was fully filled
and get the actual average fill price and fees.

---

## 10. Rate Limits

| Category | Limit |
|---|---|
| REST — private endpoints | 30 requests / second |
| REST — public endpoints | 10 requests / second |
| WebSocket connections | 8 simultaneous per key |
| WebSocket subscriptions | 2500 channels per connection |
| Candle endpoint | 300 candles per request |

**Project behaviour**:
- `fetch_candles()` sleeps 0.25 s between page requests to stay under limits
- The WebSocket reconnect loop uses exponential backoff (5 s base)
- Balance panel polls every 30 s (well under the 30 req/s limit)

---

## 11. Error Codes

### HTTP status codes

| Code | Meaning | Common cause |
|---|---|---|
| `200` | OK | Success |
| `400` | Bad Request | Malformed JSON, missing required field |
| `401` | Unauthorized | Bad API key, wrong signature, expired timestamp |
| `403` | Forbidden | Key lacks required permission |
| `404` | Not Found | Unknown product ID, order ID |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Coinbase-side issue; retry with backoff |

### Error response body
```json
{
  "error":        "INVALID_ARGUMENT",
  "message":      "Invalid product_id",
  "error_details": "...",
  "preview_failure_reason": "",
  "new_order_failure_reason": ""
}
```

### Common `error` values

| Value | Description |
|---|---|
| `INVALID_ARGUMENT` | Bad parameter value |
| `UNAUTHENTICATED` | Auth header missing or invalid |
| `PERMISSION_DENIED` | Key doesn't have the required scope |
| `NOT_FOUND` | Resource doesn't exist |
| `RESOURCE_EXHAUSTED` | Rate limit hit |
| `INSUFFICIENT_FUND` | Not enough balance to place order |
| `ORDER_NOT_FOUND` | Cancel on unknown order ID |

---

## 12. Key Management Notes

### API key permissions (scopes)

When creating an Advanced Trade API key at https://www.coinbase.com/settings/api
you must enable the following permissions for Project Challenger:

| Permission | Required for |
|---|---|
| `wallet:accounts:read` | Balance panel (`GET /accounts`) |
| `wallet:trades:read` | Fill history |
| `wallet:transactions:read` | Transaction summary / fees |
| `trade` (Advanced Trade) | Placing and cancelling orders |
| `view` (Advanced Trade) | Candles, ticker, products |

For a **read-only** / paper-trading setup you only need `view` and
`wallet:accounts:read`.  The `trade` scope is only needed when live trading
is armed.

### Storing keys securely

Keys are stored in `data/.secrets.json` using local OS protection where
available. Legacy `data/api_keys.json` files are migrated on startup and both
paths are listed in `.gitignore`. Supported import formats:

- **Coinbase JSON export** (`{"api_key": "...", "api_secret": "..."}`)
- **Coinbase CDP export** (`{"name": "organizations/.../apiKeys/xyz", "privateKey": "..."}`)
- **Plain text** — two lines (key, secret) or `API_KEY=...` / `API_SECRET=...`
- **ZIP archive** containing any of the above

Import via: Settings tab → API KEYS → Load from File.

### Timestamp tolerance

Coinbase rejects requests where the `CB-ACCESS-TIMESTAMP` header differs
from the server clock by more than **30 seconds**.  If you see `UNAUTHENTICATED`
errors despite a valid key, check that your system clock is synchronised
(Windows: `w32tm /resync`, Linux: `timedatectl` / `ntpdate`).

---

*Last updated: 2026-05-20 — Project Challenger v0.12.0*
