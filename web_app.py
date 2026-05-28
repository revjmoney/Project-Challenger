"""
Project Challenger — Web GUI server
====================================
FastAPI backend + single-page browser UI mirroring all six TUI tabs.

Copyright (c) 2026 Rev. J. Money.
Non-commercial learning/research use only. See LICENSE and NOTICE.

Start:
    python web_app.py              # http://127.0.0.1:8765  (localhost only by default)
    python web_app.py --port 9000  # custom port
    python web_app.py --host 0.0.0.0  # bind all interfaces (also set SERVER_LOCALHOST_ONLY=false)

Then open  http://127.0.0.1:8765  in your browser.

No extra config needed — shares the same database, model files, and
config.py as controller.py and main.py.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import datetime
import io
import json
import math
import os
import sqlite3
import sys
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ── FastAPI / Uvicorn ─────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    import uvicorn
except ImportError:
    sys.exit(
        "[web_app] FastAPI or uvicorn not installed.\n"
        "Run:  pip install fastapi uvicorn\n"
        "or use the web requirements file:\n"
        "  pip install -r requirements_web.txt"
    )

# ── Project imports ───────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import auth as _auth
from config import (
    CONFIG, save_api_keys, clear_api_keys,
    save_user_settings, is_demo_mode, get_active_symbol,
)
from bot_manager import BotManager, models_are_trained
from database import init_db
from activity import get_tracker
from model_archive import list_all_archives
from backtester import BacktestConfig

# ── Globals ───────────────────────────────────────────────────────────────────
_bot: Optional[BotManager] = None
_bot_log: collections.deque = collections.deque(maxlen=500)

# Set to True via --secure-cookies CLI flag; requires HTTPS
_SECURE_COOKIES: bool = False

_backtest_state: dict[str, Any] = {
    "running": False, "results": None, "by_coin": None, "error": None,
    "pct": 0, "message": "", "models_done": 0, "models_total": 0,
    "coins_done": 0, "coins_total": 0, "model_statuses": {},
}
_bt_start_time: float = 0.0       # epoch seconds when last backtest began
_retrain_state: dict[str, Any] = {
    "running": False, "done": False, "error": None,
}
_train_start_time: float = 0.0    # epoch seconds when last retrain began

_wfv_state: dict[str, Any] = {
    "running": False, "results": None, "error": None,
    "pct": 0, "message": "",
    "fold_num": 0, "n_folds": 0,
    "models_done": 0, "models_total": 0,
}
_wfv_start_time: float = 0.0      # epoch seconds when last WFV began

WEB_DIR  = _ROOT / "web"
DATA_DIR = _ROOT / "data"


from startup import check_install
check_install()


# ── Drawdown monitor ─────────────────────────────────────────────────────────

_dd_stop = threading.Event()


def _drawdown_monitor() -> None:
    """
    Background thread: checks every 60 s whether any model has breached the
    MAX_DRAWDOWN_PCT threshold.  When triggered, fires a notification and
    stops the bot — but only once per bot session to avoid repeated stops.
    """
    triggered: set[str] = set()
    while not _dd_stop.wait(60):
        max_dd = CONFIG["PAPER_TRADING"].get("MAX_DRAWDOWN_PCT", 0.0)
        if max_dd <= 0 or not _bot or not _bot.running:
            triggered.clear()
            continue
        try:
            conn  = _db()
            peaks = {r[0]: r[1] for r in conn.execute(
                "SELECT model_name, MAX(capital) FROM portfolio_snapshots GROUP BY model_name"
            ).fetchall()}
            currents = {r[0]: r[1] for r in conn.execute(
                "SELECT model_name, capital FROM portfolio_snapshots "
                "WHERE id IN (SELECT MAX(id) FROM portfolio_snapshots GROUP BY model_name)"
            ).fetchall()}
            initial = CONFIG["PAPER_TRADING"]["INITIAL_CAPITAL"]
            for model, peak in peaks.items():
                if model in triggered:
                    continue
                peak    = max(peak, initial)
                current = currents.get(model, peak)
                dd_pct  = (peak - current) / peak * 100 if peak > 0 else 0.0
                if dd_pct >= max_dd:
                    triggered.add(model)
                    msg = (f"{model} drawdown {dd_pct:.1f}% ≥ limit {max_dd:.1f}% — "
                           f"bot stopped automatically.")
                    print(f"[DRAWDOWN] {msg}")
                    _append_log(f"[DRAWDOWN] {msg}")
                    try:
                        from notifications import notify
                        if CONFIG.get("NOTIFICATIONS", {}).get("ON_DRAWDOWN", True):
                            notify(msg, title="Max Drawdown Hit", level="danger")
                    except Exception:
                        pass
                    if _bot and _bot.running:
                        threading.Thread(target=_bot.stop, daemon=True,
                                         name="dd-stop").start()
                    break
        except Exception as e:
            print(f"[DRAWDOWN] monitor error: {e}")


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bot
    init_db()
    _bot = BotManager(log_fn=_append_log)

    # ── Auto-train on first run ──────────────────────────────────────────────
    # If no model files exist when the server starts, kick off training
    # automatically so the user just has to wait in the browser rather than
    # running a separate command.  After training finishes the bot starts
    # automatically so data begins streaming immediately.
    if not models_are_trained():
        global _train_start_time
        _train_start_time = time.time()
        active_models = [k for k, v in CONFIG["ACTIVE_MODELS"].items() if v]
        _retrain_state.update({
            "running": True, "done": False, "error": None,
            "active_models": active_models
        })

        def _auto_done():
            _retrain_state.update({"running": False, "done": True})
            # Auto-start the bot so live data begins right away
            if _bot:
                try:
                    time.sleep(1)
                    _bot.start()
                except Exception:
                    pass

        def _delayed_retrain():
            # Give Uvicorn 2 s to finish binding before training touches disk
            time.sleep(2)
            if _bot:
                _bot.retrain(done_cb=_auto_done)

        threading.Thread(
            target=_delayed_retrain, daemon=True, name="auto-train"
        ).start()

    _dd_stop.clear()
    threading.Thread(
        target=_drawdown_monitor, daemon=True, name="dd-monitor"
    ).start()

    yield

    _dd_stop.set()
    if _bot and _bot.running:
        _bot.stop()


app = FastAPI(title="Project Challenger", lifespan=lifespan)


# ── CSRF guard ───────────────────────────────────────────────────────────────

_CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


@app.middleware("http")
async def _csrf_guard(request: Request, call_next):
    if request.method not in _CSRF_SAFE_METHODS:
        origin = request.headers.get("origin")
        if origin:
            origin_host = urlparse(origin).netloc
            server_host = request.headers.get("host", "")
            if origin_host != server_host:
                return JSONResponse({"error": "CSRF check failed."}, status_code=403)
    return await call_next(request)


# ── Security headers ─────────────────────────────────────────────────────────

_SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    ),
}


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    for header, value in _SECURITY_HEADERS.items():
        response.headers[header] = value
    return response


# ── Login rate limiter ────────────────────────────────────────────────────────

_LOGIN_MAX_ATTEMPTS = 5      # failures before lockout
_LOGIN_WINDOW_SECS  = 900    # sliding window (15 min)
_LOGIN_LOCKOUT_SECS = 600    # lockout duration (10 min)

_login_attempts: dict[str, list[float]] = collections.defaultdict(list)
_login_lock = threading.Lock()


def _login_check_rate(ip: str) -> int:
    """Return seconds remaining in lockout; 0 if not locked out."""
    now = time.time()
    with _login_lock:
        _login_attempts[ip] = [t for t in _login_attempts[ip]
                                if now - t < _LOGIN_WINDOW_SECS]
        recent = _login_attempts[ip]
        if len(recent) >= _LOGIN_MAX_ATTEMPTS:
            remaining = int(_LOGIN_LOCKOUT_SECS - (now - recent[0]))
            if remaining > 0:
                return remaining
            _login_attempts[ip] = []
    return 0


def _login_record_failure(ip: str) -> None:
    with _login_lock:
        _login_attempts[ip].append(time.time())


def _login_clear(ip: str) -> None:
    with _login_lock:
        _login_attempts.pop(ip, None)


# ── Auth middleware ───────────────────────────────────────────────────────────
# All /api/* paths require a valid session cookie EXCEPT the auth endpoints
# themselves and the SSE stream (which also accepts ?token= as fallback).

_AUTH_PUBLIC = {
    "/",
    "/api/auth/status",
    "/api/auth/setup",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/register",
}


@app.middleware("http")
async def _auth_guard(request: Request, call_next):
    if request.url.path in _AUTH_PUBLIC or request.method == "OPTIONS":
        return await call_next(request)

    # Cookie only — EventSource is same-origin so the browser sends it
    # automatically.  Never accept tokens in the URL (they leak to server logs).
    token = request.cookies.get("auth_token", "")

    if not _auth.validate_session(token):
        return JSONResponse(
            {"error": "unauthorized", "need_auth": True}, status_code=401
        )

    return await call_next(request)


# ── Localhost-only guard ──────────────────────────────────────────────────────
# Registered last so it executes first (Starlette runs outermost middleware
# first; the last @app.middleware is the outermost wrapper).
# When SERVER.LOCALHOST_ONLY is True, all requests from non-loopback IPs are
# rejected with 403 before any other processing.

_LOOPBACK_IPS = {"127.0.0.1", "::1"}


@app.middleware("http")
async def _localhost_guard(request: Request, call_next):
    if CONFIG["SERVER"].get("LOCALHOST_ONLY", True):
        client_ip = request.client.host if request.client else ""
        if client_ip not in _LOOPBACK_IPS:
            return JSONResponse({"error": "Access restricted to localhost."}, status_code=403)
    return await call_next(request)


# ── Authentication endpoints ──────────────────────────────────────────────────

@app.get("/api/auth/status")
async def api_auth_status():
    """Check whether credentials have been configured (first-run detection)."""
    return {
        "configured": _auth.has_credentials(),
        "registered": (DATA_DIR / ".registered").exists(),
    }


@app.post("/api/auth/setup")
async def api_auth_setup(request: Request):
    """
    First-run only — create the single username + password.
    Returns 400 if credentials already exist (use /login instead).
    Sets an auth_token cookie on success.
    """
    if _auth.has_credentials():
        raise HTTPException(400, "Credentials already configured — use /api/auth/login.")
    body     = await request.json()
    username = (body.get("username") or "").strip()
    password =  body.get("password") or ""
    if not username or not password:
        raise HTTPException(400, "username and password are required.")
    try:
        _auth.save_credentials(username, password)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    token    = _auth.create_session(username)
    resp     = JSONResponse({"status": "ok", "username": username})
    resp.set_cookie("auth_token", token, httponly=True, samesite="lax",
                    max_age=86400 * 30, secure=_SECURE_COOKIES)
    return resp


@app.post("/api/register")
async def api_register(request: Request):
    """
    One-time install registration — fires after first-run account setup.
    Collects optional user info + HW/SW metadata and POSTs to the
    revjmoney.com ping endpoint. Writes data/.registered so it never runs twice.
    Always returns ok (failure is silent — don't block the user).
    """
    import platform, multiprocessing, urllib.request as _ureq, urllib.error as _uerr
    _PING_URL   = "https://jmscnc.com/revjmoney/yo/"
    _FLAG_FILE  = DATA_DIR / ".registered"

    if _FLAG_FILE.exists():
        return JSONResponse({"status": "ok", "note": "already registered"})

    try:
        body = await request.json()
    except Exception:
        body = {}

    sys_info = {
        "sys_os":         platform.system(),
        "sys_os_release": platform.release(),
        "sys_machine":    platform.machine(),
        "sys_python":     sys.version.split()[0],
        "sys_cpu_count":  multiprocessing.cpu_count(),
        "app_version":    "0.28.6-cdx",
    }
    payload = {**body, **sys_info}

    async def _fire_and_forget():
        raw = json.dumps(payload).encode()
        for attempt in range(1, 4):
            try:
                req = _ureq.Request(_PING_URL, data=raw,
                                    headers={"Content-Type": "application/json"})
                with _ureq.urlopen(req, timeout=8):
                    pass
                break  # success
            except Exception:
                if attempt < 3:
                    await asyncio.sleep(attempt * 2)  # 2s, 4s
        try:
            _FLAG_FILE.write_text("1")
        except Exception:
            pass

    import asyncio
    asyncio.create_task(_fire_and_forget())
    return JSONResponse({"status": "ok"})


@app.post("/api/auth/login")
async def api_auth_login(request: Request):
    """Validate credentials and issue a session cookie."""
    ip = request.client.host if request.client else "unknown"
    retry_after = _login_check_rate(ip)
    if retry_after > 0:
        return JSONResponse(
            {"error": f"Too many failed attempts. Try again in {retry_after} seconds."},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    body     = await request.json()
    username = (body.get("username") or "").strip()
    password =  body.get("password") or ""
    if not _auth.check_credentials(username, password):
        _login_record_failure(ip)
        raise HTTPException(401, "Invalid username or password.")
    _login_clear(ip)
    token    = _auth.create_session(username)
    resp     = JSONResponse({"status": "ok", "username": username})
    resp.set_cookie("auth_token", token, httponly=True, samesite="lax",
                    max_age=86400 * 30, secure=_SECURE_COOKIES)
    return resp


@app.post("/api/auth/logout")
async def api_auth_logout(request: Request):
    """Destroy the session and clear the auth cookie."""
    token = request.cookies.get("auth_token", "")
    _auth.destroy_session(token)
    resp  = JSONResponse({"status": "ok"})
    resp.delete_cookie("auth_token")
    return resp


@app.post("/api/auth/change_password")
async def api_auth_change_password(request: Request):
    """Change the password.  Requires the current password for verification."""
    token    = request.cookies.get("auth_token", "")
    username = _auth.get_username(token)
    if not username:
        raise HTTPException(401, "Not authenticated.")
    body       = await request.json()
    current_pw = body.get("current_password") or ""
    new_pw     = body.get("new_password")     or ""
    if not _auth.check_credentials(username, current_pw):
        raise HTTPException(403, "Current password is incorrect.")
    try:
        _auth.save_credentials(username, new_pw)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    # Invalidate all sessions and issue a fresh token
    _auth.destroy_all_sessions()
    new_token = _auth.create_session(username)
    resp      = JSONResponse({"status": "ok"})
    resp.set_cookie("auth_token", new_token, httponly=True, samesite="lax",
                    max_age=86400 * 30, secure=_SECURE_COOKIES)
    return resp


# ── Internal helpers ──────────────────────────────────────────────────────────

def _append_log(msg: str) -> None:
    _bot_log.append(msg)   # deque(maxlen=500) auto-evicts oldest entry


def _parse_signals(n: int = 50) -> list[dict]:
    tracker = get_tracker()
    signals = []
    for e in tracker.get_log(500):
        if e.get("status") != "TRADING":
            continue
        msg = e.get("message", "")
        side = "BUY" if msg.upper().startswith("BUY") else "SELL"
        signals.append({
            "ts":        e["ts"],
            "component": e["component"],
            "side":      side,
            "message":   msg,
        })
    return signals[-n:]


_db_local = threading.local()   # one persistent connection per thread


def _db() -> sqlite3.Connection:
    """
    Return a persistent thread-local SQLite connection.
    WAL mode + row_factory are set once at creation; the connection is
    kept alive for the lifetime of the thread (Uvicorn reuses its thread
    pool, so this avoids the open + PRAGMA overhead on every API request).
    Callers must NOT call conn.close() — the connection is shared.
    """
    conn = getattr(_db_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(
            CONFIG["PATHS"]["DB"], timeout=5.0, check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        _db_local.conn = conn
    return conn


def _latest_price() -> float:
    try:
        row = _db().execute(
            "SELECT close FROM candle_cache "
            "WHERE exchange=? AND symbol=? ORDER BY ts DESC LIMIT 1",
            (CONFIG["EXCHANGE"], get_active_symbol()),
        ).fetchone()
        return float(row["close"]) if row else 0.0
    except Exception:
        return 0.0


def _candle_count() -> int:
    try:
        row = _db().execute(
            "SELECT COUNT(*) AS n FROM candle_cache WHERE exchange=? AND symbol=?",
            (CONFIG["EXCHANGE"], get_active_symbol()),
        ).fetchone()
        return int(row["n"]) if row else 0
    except Exception:
        return 0


def _paper_metrics() -> dict:
    """Latest capital + P&L + open position per model from portfolio_snapshots."""
    try:
        rows = _db().execute("""
            SELECT model_name, capital, total_pnl,
                   position_side, position_qty, unrealized_pnl
            FROM portfolio_snapshots
            WHERE id IN (
                SELECT MAX(id) FROM portfolio_snapshots GROUP BY model_name
            )
        """).fetchall()
        return {r["model_name"]: {
            "capital":        r["capital"],
            "total_pnl":      r["total_pnl"],
            "position_side":  r["position_side"],
            "position_qty":   r["position_qty"]   or 0.0,
            "unrealized_pnl": r["unrealized_pnl"] or 0.0,
        } for r in rows}
    except Exception:
        return {}


def _trade_stats() -> dict:
    """Trade count, wins, cumulative P&L per model from paper_trades."""
    try:
        rows = _db().execute("""
            SELECT model_name,
                   COUNT(*) AS count,
                   SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                   SUM(realized_pnl) AS total_pnl
            FROM paper_trades
            GROUP BY model_name
        """).fetchall()
        return {r["model_name"]: {
            "count":     r["count"],
            "wins":      int(r["wins"] or 0),
            "total_pnl": float(r["total_pnl"] or 0.0),
        } for r in rows}
    except Exception:
        return {}


def _compute_sharpe(model_name: str) -> Optional[float]:
    """
    Annualised Sharpe ratio from paper trades, using daily P&L buckets.
    Returns None when there are fewer than 3 days of trade data.
    """
    try:
        rows = _db().execute(
            "SELECT timestamp, realized_pnl FROM paper_trades "
            "WHERE model_name=? ORDER BY timestamp ASC",
            (model_name,),
        ).fetchall()
        if len(rows) < 6:
            return None
        daily: dict[str, float] = {}
        for ts, pnl in rows:
            day = datetime.date.fromtimestamp(float(ts)).isoformat()
            daily[day] = daily.get(day, 0.0) + (float(pnl) if pnl else 0.0)
        vals = list(daily.values())
        if len(vals) < 3:
            return None
        n    = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std  = math.sqrt(variance)
        if std == 0:
            return None
        return round(mean / std * math.sqrt(252), 2)
    except Exception:
        return None


def _cv_summary() -> list[dict]:
    """Most recent CV fold rows from training_results."""
    try:
        rows = _db().execute("""
            SELECT model_name, fold, direction_acc, test_mse
            FROM training_results
            WHERE trained_at = (SELECT MAX(trained_at) FROM training_results)
            ORDER BY model_name, fold
        """).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _serialize_bt(results: dict) -> dict:
    out = {}
    for key, r in results.items():
        if r.error:
            out[key] = {"error": r.error}
        else:
            pf = r.profit_factor
            out[key] = {
                "total_trades":  r.total_trades,
                "wins":          r.wins,
                "win_rate":      r.win_rate,
                "profit_factor": pf if pf != float("inf") else 9999.0,
                "net_pnl":       r.net_pnl,
                "max_drawdown":  r.max_drawdown,
            }
    return out


def _serialize_bt_by_coin(by_coin: dict) -> dict:
    """Serialize {symbol: {model_key: BacktestResult}} for JSON transport."""
    out: dict = {}
    for sym, model_results in by_coin.items():
        out[sym] = _serialize_bt(model_results)
    return out


def _archive_json() -> dict:
    try:
        archives = list_all_archives()
        return {
            key: [
                {
                    "timestamp": m.timestamp,
                    "pnl":       m.pnl,
                    "is_active": m.is_active,
                    "is_armed":  m.is_armed,
                    "filename":  os.path.basename(m.path),
                    "trades":    m.trades,
                    "win_rate":  m.win_rate,
                    "max_dd":    m.max_dd,
                }
                for m in models
            ]
            for key, models in archives.items()
        }
    except Exception:
        return {}


# ── Index ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    path = WEB_DIR / "index.html"
    if not path.exists():
        raise HTTPException(500, "web/index.html not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


# ── Archive Trade History ─────────────────────────────────────────────────────

@app.get("/api/archive/trades/{model_key}/{filename}")
async def api_archive_trades(model_key: str, filename: str):
    """Return trade history for a specific archived model."""
    from model_archive import get_archive_trade_history
    return get_archive_trade_history(model_key, filename)


# ── Ticker — latest candle close for each coin with cached data ───────────────

_TICKER_TTL   = 30.0   # seconds between DB reads
_ticker_cache: dict[str, Any] = {"data": [], "ts": 0.0}
_ticker_lock  = threading.Lock()


@app.get("/api/ticker")
async def api_ticker():
    """Return latest close price and 1-candle change for all training-tier coins
    from the SQLite candle_cache. Requires a valid session; result is cached for
    30 seconds so rapid polls don't hammer the database."""
    now = time.time()
    with _ticker_lock:
        if now - _ticker_cache["ts"] < _TICKER_TTL:
            return _ticker_cache["data"]

    from coin_manager import get_training_symbols, symbol_to_base

    symbols  = get_training_symbols()
    exchange = CONFIG["EXCHANGE"]
    result: list[dict] = []

    try:
        conn = _db()
        for sym in symbols:
            rows = conn.execute(
                "SELECT close, ts FROM candle_cache "
                "WHERE exchange=? AND symbol=? "
                "ORDER BY ts DESC LIMIT 2",
                (exchange, sym),
            ).fetchall()
            if not rows:
                continue
            price  = float(rows[0][0])
            change = 0.0
            if len(rows) == 2 and rows[1][0]:
                old = float(rows[1][0])
                if old > 0:
                    change = round((price - old) / old * 100, 3)
            base = symbol_to_base(sym, exchange)
            result.append({"symbol": base, "price": price, "change24h": change})
    except Exception as e:
        print(f"[api/ticker] {e}")

    with _ticker_lock:
        _ticker_cache["data"] = result
        _ticker_cache["ts"]   = time.time()

    return result


# ── Status & metrics ──────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    ws = _bot.worker_status() if _bot else {}
    return {
        "running":        _bot.running if _bot else False,
        "demo_mode":      is_demo_mode(),
        "paper_mode":     is_demo_mode(),
        "exchange":       CONFIG["EXCHANGE"],
        "symbol":         get_active_symbol(),
        "price":          _latest_price(),
        "candle_count":   _candle_count(),
        "models_trained": models_are_trained(),
        "workers":        ws,
        "live_armed":     _bot.live_trader.is_armed if _bot else False,
        "armed_model":    CONFIG["LIVE_TRADING"]["ARMED_MODEL"],
    }


@app.get("/api/metrics")
async def api_metrics():
    portfolio = _paper_metrics()
    trades    = _trade_stats()
    initial   = CONFIG["PAPER_TRADING"]["INITIAL_CAPITAL"]
    active    = CONFIG["ACTIVE_MODELS"]
    models: dict[str, Any] = {}
    # Emit metrics for every configured model (active or not) so the UI
    # can show zeroed-out cards for inactive models too.
    for key in active:
        p   = portfolio.get(key, {})
        t   = trades.get(key, {})
        cap = p.get("capital", initial)
        pnl = p.get("total_pnl", 0.0)
        cnt = t.get("count", 0)
        win = t.get("wins", 0)
        models[key] = {
            "capital":        cap,
            "total_pnl":      pnl,
            "pnl_pct":        (cap - initial) / initial * 100,
            "trades":         cnt,
            "wins":           win,
            "win_rate":       (win / cnt * 100) if cnt > 0 else 0.0,
            "position_side":  p.get("position_side"),
            "position_qty":   p.get("position_qty",   0.0),
            "unrealized_pnl": p.get("unrealized_pnl", 0.0),
            "active":         bool(active.get(key)),
            "sharpe":         _compute_sharpe(key),
        }
    return {"models": models, "initial_capital": initial}


@app.get("/api/equity")
async def api_equity():
    """
    Cumulative P&L time series per model for equity curve charts.
    Returns {MODEL_KEY: {times: [...], values: [...]}} sorted oldest-first.
    """
    try:
        rows = _db().execute(
            "SELECT model_name, timestamp, realized_pnl "
            "FROM paper_trades ORDER BY timestamp ASC"
        ).fetchall()
    except Exception:
        return JSONResponse({})
    series: dict[str, dict] = {}
    for r in rows:
        key  = r["model_name"]
        ts   = float(r["timestamp"])
        pnl  = float(r["realized_pnl"] or 0.0)
        if key not in series:
            series[key] = {"times": [], "values": [], "_cum": 0.0}
        series[key]["_cum"] += pnl
        series[key]["times"].append(round(ts, 1))
        series[key]["values"].append(round(series[key]["_cum"], 2))
    for v in series.values():
        del v["_cum"]
    return JSONResponse(series)


@app.get("/api/trades")
async def api_trades(limit: int = 50, model: str = ""):
    """
    Return the most recent paper trades.
    Optional ?model=SKLEARN_LINEAR to filter by model.
    Optional ?limit=N (max 200).
    """
    limit = min(max(limit, 1), 200)
    try:
        conn = _db()
        if model:
            rows = conn.execute(
                "SELECT id, timestamp, model_name, side, execution_price, "
                "quantity, fee_paid, realized_pnl "
                "FROM paper_trades WHERE model_name=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (model, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, timestamp, model_name, side, execution_price, "
                "quantity, fee_paid, realized_pnl "
                "FROM paper_trades ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {
            "trades": [
                {
                    "id":             r["id"],
                    "timestamp":      r["timestamp"],
                    "model_name":     r["model_name"],
                    "side":           r["side"],
                    "price":          r["execution_price"],
                    "quantity":       r["quantity"],
                    "fee":            r["fee_paid"],
                    "realized_pnl":   r["realized_pnl"],
                }
                for r in rows
            ]
        }
    except Exception as e:
        print(f"[api/trades] {e}")
        return {"trades": [], "error": "Failed to load trades."}


@app.get("/api/trades/export")
async def api_trades_export(model: str = ""):
    """Download all paper trades as a CSV file."""
    try:
        conn = _db()
        if model:
            rows = conn.execute(
                "SELECT id, timestamp, model_name, side, execution_price, "
                "quantity, fee_paid, realized_pnl "
                "FROM paper_trades WHERE model_name=? ORDER BY timestamp ASC",
                (model,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, timestamp, model_name, side, execution_price, "
                "quantity, fee_paid, realized_pnl "
                "FROM paper_trades ORDER BY timestamp ASC",
            ).fetchall()
    except Exception:
        rows = []

    buf    = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "datetime", "model", "side", "price", "quantity", "fee", "realized_pnl"])
    for r in rows:
        ts = datetime.datetime.fromtimestamp(float(r[1])).strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([r[0], ts, r[2], r[3], r[4], r[5], r[6], r[7]])

    fname = f"trades_{'all' if not model else model}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


@app.get("/api/notifications")
async def api_get_notifications():
    """Return current notification settings (secrets masked)."""
    cfg = CONFIG.get("NOTIFICATIONS", {})
    token = cfg.get("TELEGRAM_TOKEN", "")
    return {
        "discord_webhook":  cfg.get("DISCORD_WEBHOOK", ""),
        "telegram_token":   ("*" * 6 + token[-4:]) if len(token) > 4 else ("*" * len(token)),
        "telegram_chat_id": cfg.get("TELEGRAM_CHAT_ID", ""),
        "on_trade":         cfg.get("ON_TRADE",    True),
        "on_estop":         cfg.get("ON_ESTOP",    True),
        "on_promote":       cfg.get("ON_PROMOTE",  True),
        "max_drawdown_pct": CONFIG["PAPER_TRADING"].get("MAX_DRAWDOWN_PCT", 0.0),
    }


@app.post("/api/notifications")
async def api_save_notifications(request: Request):
    """Persist notification settings and reload CONFIG immediately."""
    body = await request.json()
    overrides: dict = {}
    if "discord_webhook"  in body: overrides["DISCORD_WEBHOOK"]  = str(body["discord_webhook"]).strip()
    if "telegram_token"   in body: overrides["TELEGRAM_TOKEN"]   = str(body["telegram_token"]).strip()
    if "telegram_chat_id" in body: overrides["TELEGRAM_CHAT_ID"] = str(body["telegram_chat_id"]).strip()
    if "on_trade"         in body: overrides["NOTIFY_ON_TRADE"]  = bool(body["on_trade"])
    if "on_estop"         in body: overrides["NOTIFY_ON_ESTOP"]  = bool(body["on_estop"])
    if "on_promote"       in body: overrides["NOTIFY_ON_PROMOTE"] = bool(body["on_promote"])
    if "max_drawdown_pct" in body: overrides["MAX_DRAWDOWN_PCT"] = float(body["max_drawdown_pct"])
    if overrides:
        save_user_settings(overrides)
    return {"status": "saved"}


@app.post("/api/notifications/test")
async def api_test_notification():
    """Send a test notification to verify webhook configuration."""
    try:
        from notifications import notify
        notify(
            "Webhook is working! Project Challenger will send alerts here.",
            title="Test Notification",
            level="info",
        )
        return {"status": "sent"}
    except Exception as e:
        _append_log(f"[NOTIFY] test delivery failed: {e}")
        raise HTTPException(500, "Notification delivery failed. Check webhook URL and token.")


@app.get("/api/archive")
async def api_archive():
    return _archive_json()


@app.get("/api/signals")
async def api_signals(limit: int = 50):
    limit = min(max(limit, 1), 200)
    return {"signals": _parse_signals(limit)}


@app.get("/api/activity")
async def api_activity():
    tracker = get_tracker()
    return {
        "components": tracker.get_components(),
        "workers":    _bot.worker_details() if _bot else [],
        "log":        tracker.get_log(100),
        "bot_log":    list(_bot_log)[-100:],
    }


@app.get("/api/cv")
async def api_cv():
    return {"folds": _cv_summary()}


@app.get("/api/settings")
async def api_settings():
    # Serialise training dict — convert any list values to JSON-safe types
    training_out = {}
    for k, v in CONFIG["TRAINING"].items():
        training_out[k] = v if not isinstance(v, (list, tuple)) else str(v)
    return {
        "paper_trading":   CONFIG["PAPER_TRADING"],
        "backtesting":     CONFIG["BACKTESTING"],
        "training":        training_out,
        "active_models":   CONFIG["ACTIVE_MODELS"],
        "data_cache":      CONFIG["DATA_CACHE"],
        "live_trading":    CONFIG["LIVE_TRADING"],
        "exchange":        CONFIG["EXCHANGE"],
        "notifications":   CONFIG.get("NOTIFICATIONS", {}),
        "walk_forward_levels": CONFIG["BACKTESTING"].get("WALK_FORWARD_LEVELS", 3),
    }


@app.get("/api/backtest/status")
async def api_backtest_status():
    return _backtest_state


@app.get("/api/training")
async def api_training():
    """Detailed training status — model file presence, last CV run, retrain state."""
    from bot_manager import _MODEL_PATH_KEYS
    paths = CONFIG["PATHS"]
    # Always include scaler + every known model path
    files: dict[str, Any] = {
        "scaler":  os.path.exists(paths["SCALER"]),
        # Legacy keys for backwards compatibility
        "sklearn": os.path.exists(paths["SKLEARN_MODEL"]),
        "xgboost": os.path.exists(paths["XGBOOST_MODEL"]),
        "lstm":    os.path.exists(paths["PYTORCH_MODEL"]),
    }
    # Add per-model entries keyed by ACTIVE_MODELS key (lower-case)
    for model_key, path_key in _MODEL_PATH_KEYS.items():
        files[model_key.lower()] = os.path.exists(paths[path_key])
    last_trained: Optional[float] = None
    try:
        row = _db().execute(
            "SELECT MAX(trained_at) AS t FROM training_results"
        ).fetchone()
        if row and row["t"]:
            last_trained = float(row["t"])
    except Exception:
        pass

    return {
        "retrain_state": _retrain_state,
        "model_files":   files,
        "last_trained":  last_trained,
        "cv_folds":      _cv_summary(),
    }


@app.get("/api/retrain/status")
async def api_retrain_status():
    return _retrain_state


# ── Bot control ───────────────────────────────────────────────────────────────

@app.post("/api/bot/start")
async def api_bot_start():
    if not _bot:
        raise HTTPException(500, "BotManager not initialized")
    if _bot.running:
        return {"status": "already_running"}
    _bot.start()
    return {"status": "started"}


@app.post("/api/bot/stop")
async def api_bot_stop():
    if not _bot:
        raise HTTPException(500, "BotManager not initialized")
    _bot.stop()
    return {"status": "stopped"}


@app.post("/api/bot/estop")
async def api_estop():
    """
    Emergency stop — force-close all open paper positions at current market
    price, disarm live trading, and stop the bot.
    """
    if _bot:
        threading.Thread(
            target=_bot.emergency_stop, daemon=True, name="estop"
        ).start()
    return {"status": "stopping"}


@app.get("/api/pids")
async def api_pids():
    """Return the server PID, all live worker PIDs, and training/backtest status."""
    workers = []
    if _bot:
        for key, proc in list(_bot._procs.items()):
            workers.append({
                "key":   key,
                "pid":   proc.pid,
                "alive": proc.is_alive(),
            })
    return {
        "server_pid":      os.getpid(),
        "workers":         workers,
        "training_active": _retrain_state.get("running", False),
        "backtest_active": _backtest_state.get("running", False),
    }


@app.post("/api/bot/killall")
async def api_bot_killall():
    """Stop the bot and terminate all managed worker processes."""
    if _bot and _bot.running:
        _bot.stop()
    return {"status": "killed"}


@app.post("/api/bot/retrain")
async def api_bot_retrain():
    global _train_start_time
    if _retrain_state["running"]:
        return {"status": "already_running"}
    _train_start_time = time.time()
    active_models = [k for k, v in CONFIG["ACTIVE_MODELS"].items() if v]
    _retrain_state.update({
        "running": True, "done": False, "error": None,
        "active_models": active_models
    })

    def _done():
        _retrain_state.update({"running": False, "done": True})

    if _bot:
        # retrain() calls stop() which joins worker processes (up to 3 s each).
        # Running it in an executor keeps the asyncio event loop free so SSE
        # and other API calls don't time out while workers are being torn down.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: _bot.retrain(done_cb=_done))
    return {"status": "started"}


@app.post("/api/bot/refresh_data")
async def api_refresh_data():
    if _bot:
        _bot.fetch_fresh_data()
    return {"status": "triggered"}


@app.post("/api/bot/reset_db")
async def api_reset_db():
    if _bot:
        _bot.reset_db()
    return {"status": "ok"}


# ── Backtest ──────────────────────────────────────────────────────────────────

@app.post("/api/backtest/run")
async def api_backtest_run(request: Request):
    if _backtest_state["running"]:
        return {"status": "already_running"}

    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    cfg = BacktestConfig.from_config()
    if "lookback_hours" in body:
        cfg = BacktestConfig(
            lookback_hours    = int(body.get("lookback_hours",   cfg.lookback_hours)),
            signal_threshold  = float(body.get("signal_threshold", cfg.signal_threshold)),
            initial_capital   = float(body.get("initial_capital",  cfg.initial_capital)),
            position_size_pct = cfg.position_size_pct,
            slippage_pct      = cfg.slippage_pct,
            fee_pct           = cfg.fee_pct,
        )

    # Optional: coin override from request body
    coins: list[str] | None = None
    if "coins" in body:
        from coin_manager import coin_to_symbol
        exchange = CONFIG["EXCHANGE"]
        coins = [coin_to_symbol(c.strip().upper(), exchange)
                 for c in body["coins"] if c.strip()]

    # Count active models for progress tracking
    from coin_manager import get_backtest_symbols as _gbs
    n_models = sum(1 for v in CONFIG["ACTIVE_MODELS"].values() if v)
    n_coins  = len(coins) if coins else len(_gbs())
    active_keys = [k for k, v in CONFIG["ACTIVE_MODELS"].items() if v]

    global _bt_start_time
    _bt_start_time = time.time()
    _backtest_state.update({
        "running": True, "results": None, "by_coin": None, "error": None,
        "pct": 0, "message": "Initializing…",
        "models_done": 0, "models_total": n_models,
        "coins_done": 0,  "coins_total": n_coins,
        "model_statuses": {k: "running" for k in active_keys},
    })

    def _progress(symbol: str, coin_idx: int, n_coin: int, models_done: int, models_total: int):
        coins_done = coin_idx  # coin_idx is 0-based; current coin not yet complete
        total_steps = n_coin * models_total
        done_steps  = coins_done * models_total + models_done
        pct = int(done_steps / total_steps * 95) if total_steps > 0 else 0
        _backtest_state["pct"]          = pct
        _backtest_state["models_done"]  = models_done
        _backtest_state["coins_done"]   = coins_done
        _backtest_state["message"]      = (
            f"{symbol} — model {models_done}/{models_total}"
            + (f", coin {coin_idx+1}/{n_coin}" if n_coin > 1 else "")
        )

    def _done(payload: dict):
        _backtest_state["running"]       = False
        _backtest_state["pct"]           = 100
        _backtest_state["message"]       = "Complete"
        flat    = payload.get("results", {})
        by_coin = payload.get("by_coin", {})
        if flat or by_coin:
            _backtest_state["results"] = _serialize_bt(flat)
            _backtest_state["by_coin"] = _serialize_bt_by_coin(by_coin)
            # Mark all models done
            for k in active_keys:
                _backtest_state["model_statuses"][k] = "done"
        else:
            _backtest_state["error"] = "No results returned"

    if _bot:
        _bot.run_backtest(cfg=cfg, coins=coins, done_cb=_done, progress_cb=_progress)
    return {"status": "started"}


# ── Walk-forward validation ───────────────────────────────────────────────────

@app.post("/api/backtest/walkforward")
async def api_walkforward_run(request: Request):
    """
    Run walk-forward validation.

    Body (all optional):
      n_folds          int   1–10  (overrides CONFIG default)
      lookback_hours   int
      signal_threshold float
      initial_capital  float
    """
    if _wfv_state["running"]:
        return {"status": "already_running"}

    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    n_folds = int(body.get("n_folds",
                            CONFIG["BACKTESTING"].get("WALK_FORWARD_LEVELS", 3)))
    n_folds = max(1, min(10, n_folds))   # hard clamp 1–10

    cfg = BacktestConfig.from_config()
    if "lookback_hours" in body or "signal_threshold" in body or "initial_capital" in body:
        cfg = BacktestConfig(
            lookback_hours    = int(body.get("lookback_hours",    cfg.lookback_hours)),
            signal_threshold  = float(body.get("signal_threshold", cfg.signal_threshold)),
            initial_capital   = float(body.get("initial_capital",  cfg.initial_capital)),
            position_size_pct = cfg.position_size_pct,
            slippage_pct      = cfg.slippage_pct,
            fee_pct           = cfg.fee_pct,
        )

    n_models = sum(1 for v in CONFIG["ACTIVE_MODELS"].values() if v)

    global _wfv_start_time
    _wfv_start_time = time.time()
    _wfv_state.update({
        "running": True, "results": None, "error": None,
        "pct": 0, "message": "Initializing…",
        "fold_num": 0, "n_folds": n_folds,
        "models_done": 0, "models_total": n_models,
    })

    def _progress(fold_num: int, total_folds: int,
                  models_done: int, models_total: int) -> None:
        total_steps = total_folds * max(models_total, 1)
        done_steps  = (fold_num - 1) * max(models_total, 1) + models_done
        pct = int(done_steps / total_steps * 95)
        _wfv_state.update({
            "fold_num":    fold_num,
            "n_folds":     total_folds,
            "models_done": models_done,
            "models_total": models_total,
            "pct":         pct,
            "message":     f"Fold {fold_num}/{total_folds} — model {models_done}/{models_total}",
        })

    def _run_wfv() -> None:
        try:
            from coin_manager import get_backtest_symbols
            from backtester import run_walk_forward_validation
            from database import get_cached_candles, store_candles
            from exchanges import get_exchange

            symbols  = get_backtest_symbols()
            symbol   = symbols[0] if symbols else get_active_symbol()
            exchange = get_exchange()

            _append_log(f"[WFV] Starting {n_folds}-fold validation on {symbol} "
                        f"({cfg.lookback_hours}h lookback)")

            cached = get_cached_candles(exchange.name, symbol,
                                        hours=cfg.lookback_hours)
            if len(cached) < 50:
                _append_log(f"[WFV] {symbol}: cache thin — fetching "
                            f"{cfg.lookback_hours}h from exchange…")
                try:
                    fetched = exchange.fetch_candles(
                        symbol, lookback_hours=cfg.lookback_hours)
                    if fetched:
                        store_candles(fetched, exchange.name, symbol)
                        cached = fetched
                except Exception as exc:
                    _append_log(f"[WFV] fetch error: {exc}")

            if not cached:
                _wfv_state.update({
                    "running": False, "pct": 0,
                    "error": f"No candle data available for {symbol}",
                })
                return

            result = run_walk_forward_validation(
                cached, cfg,
                n_folds     = n_folds,
                log_fn      = _append_log,
                progress_cb = _progress,
            )

            if result.get("error"):
                _wfv_state.update({
                    "running": False, "pct": 0,
                    "error": result["error"],
                })
            else:
                actual = result["n_folds"]
                _wfv_state.update({
                    "running": False, "pct": 100,
                    "message": f"Complete — {actual} folds on {symbol}",
                    "results": result,
                    "symbol":  symbol,
                })
        except Exception as exc:
            _append_log(f"[WFV] unhandled error: {exc}")
            _wfv_state.update({
                "running": False, "pct": 0,
                "error": "Walk-forward validation failed. Check the Activity log for details.",
            })

    threading.Thread(target=_run_wfv, daemon=True, name="wfv").start()
    return {"status": "started", "n_folds": n_folds}


@app.get("/api/backtest/walkforward/status")
async def api_walkforward_status():
    """Return current walk-forward validation state."""
    return _wfv_state


# ── Coin management ──────────────────────────────────────────────────────────

@app.get("/api/coins")
async def api_get_coins():
    """Return current coin config + cached product list."""
    from coin_manager import (
        TIER_COINS, get_training_bases, get_backtest_symbols, get_trading_symbol,
    )
    from database import get_available_coins
    exchange = CONFIG["EXCHANGE"]
    cfg      = CONFIG["COINS"]
    available = get_available_coins(exchange, max_age_hours=24)
    return {
        "training_tier":   cfg["TRAINING_TIER"],
        "trading_coin":    cfg["TRADING_COIN"],
        "backtest_coins":  cfg["BACKTEST_COINS"],
        "training_coins":  get_training_bases(),
        "trading_symbol":  get_trading_symbol(),
        "available":       available,          # [{base, symbol}, ...]
        "available_count": len(available),
        "tiers": {
            k: v if v is not None else "all"
            for k, v in TIER_COINS.items()
        },
    }


@app.post("/api/coins")
async def api_save_coins(request: Request):
    """
    Persist coin settings (training_tier, backtest_coins).
    backtest_coins is now the universal coin list used for training,
    backtesting, and live/paper trading.  TRADING_COIN is auto-set to
    the first coin in the list (used by real live-order execution).
    """
    body = await request.json()
    overrides: dict = {}
    if "training_tier" in body:
        overrides["TRAINING_TIER"] = body["training_tier"]
    if "backtest_coins" in body:
        coins = body["backtest_coins"]
        overrides["BACKTEST_COINS"] = coins
        # Keep TRADING_COIN in sync — first selected coin becomes the live
        # execution target (real Coinbase orders always use a single symbol).
        if coins:
            overrides["TRADING_COIN"] = str(coins[0]).upper()
    # Legacy: accept explicit trading_coin override from old clients
    if "trading_coin" in body:
        overrides["TRADING_COIN"] = str(body["trading_coin"]).upper()
    if overrides:
        save_user_settings(overrides)
    return {"status": "saved", "coins": CONFIG["COINS"]}


@app.post("/api/coins/refresh")
async def api_refresh_coins():
    """Force-refresh the product list from the exchange."""
    def _refresh():
        from coin_manager import force_refresh_available_coins
        force_refresh_available_coins(log_fn=_append_log)

    threading.Thread(target=_refresh, daemon=True, name="coin-refresh").start()
    return {"status": "refreshing"}


# ── Settings & keys ───────────────────────────────────────────────────────────

@app.post("/api/settings")
async def api_save_settings(request: Request):
    body = await request.json()
    save_user_settings(body)
    return {"status": "saved"}


@app.get("/api/accounts")
async def api_accounts():
    """
    Fetch Coinbase account balances.
    Returns [] in paper mode or when Binance/Kraken is active.
    """
    if CONFIG["EXCHANGE"] != "COINBASE" or is_demo_mode():
        return {"accounts": [], "demo": is_demo_mode(), "exchange": CONFIG["EXCHANGE"]}
    try:
        from exchanges.coinbase import CoinbaseExchange
        accts = CoinbaseExchange().get_accounts()
        return {"accounts": accts, "demo": False, "exchange": "COINBASE"}
    except Exception as e:
        print(f"[api/accounts] {e}")
        return {"accounts": [], "error": "Failed to fetch accounts.", "exchange": "COINBASE"}


@app.post("/api/keys")
async def api_save_keys(request: Request):
    body = await request.json()
    key    = (body.get("api_key")    or "").strip()
    secret = (body.get("api_secret") or "").strip()
    if not key or not secret:
        raise HTTPException(400, "api_key and api_secret are required")
    save_api_keys(key, secret)
    return {"status": "saved"}


@app.delete("/api/keys")
async def api_clear_keys():
    clear_api_keys()
    return {"status": "cleared"}


# ── SSE live stream ───────────────────────────────────────────────────────────

@app.get("/api/stream")
async def event_stream(request: Request):
    """
    Server-Sent Events — pushes a JSON payload every 2 s containing:
      components   – per-component status from ActivityTracker
      new_log      – bot log lines added since the last tick
      bot_running  – bool
      price        – latest candle close
      retrain      – {running, done}
      bt_running   – bool
    """
    last_log_idx = len(_bot_log)

    async def generate():
        nonlocal last_log_idx
        while True:
            if await request.is_disconnected():
                break
            tracker = get_tracker()
            new_lines = list(_bot_log)[last_log_idx:]
            last_log_idx = len(_bot_log)

            # System resource stats (psutil is already a project dependency)
            cpu_pct = ram_pct = 0.0
            if _HAS_PSUTIL:
                try:
                    cpu_pct = _psutil.cpu_percent(interval=None)
                    ram_pct = _psutil.virtual_memory().percent
                except Exception:
                    pass

            # Elapsed training time (only meaningful while retraining)
            train_elapsed = (
                time.time() - _train_start_time
                if _retrain_state["running"] and _train_start_time > 0
                else 0.0
            )
            bt_elapsed = (
                time.time() - _bt_start_time
                if _backtest_state["running"] and _bt_start_time > 0
                else 0.0
            )
            wfv_elapsed = (
                time.time() - _wfv_start_time
                if _wfv_state["running"] and _wfv_start_time > 0
                else 0.0
            )

            payload = {
                "components":     tracker.get_components(),
                "workers":        _bot.worker_details() if _bot else [],
                "new_log":        new_lines,
                "bot_running":    _bot.running if _bot else False,
                "price":          _latest_price(),
                "retrain":        {
                    "running":       _retrain_state["running"],
                    "done":          _retrain_state["done"],
                    "active_models": _retrain_state.get("active_models", []),
                },
                "bt_running":      _backtest_state["running"],
                "bt_pct":          _backtest_state["pct"],
                "bt_message":      _backtest_state["message"],
                "bt_models_done":  _backtest_state["models_done"],
                "bt_models_total": _backtest_state["models_total"],
                "bt_coins_done":   _backtest_state["coins_done"],
                "bt_coins_total":  _backtest_state["coins_total"],
                "bt_model_statuses": _backtest_state["model_statuses"],
                "bt_elapsed_s":    bt_elapsed,
                "registered":      (DATA_DIR / ".registered").exists(),
                "wfv_running":     _wfv_state["running"],
                "wfv_pct":         _wfv_state["pct"],
                "wfv_message":     _wfv_state["message"],
                "wfv_fold_num":    _wfv_state["fold_num"],
                "wfv_n_folds":     _wfv_state["n_folds"],
                "wfv_elapsed_s":   wfv_elapsed,
                "cpu_pct":         cpu_pct,
                "ram_pct":         ram_pct,
                "train_elapsed_s": train_elapsed,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(2.0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Project Challenger Web GUI")
    p.add_argument("--host",           default=CONFIG["SERVER"]["HOST"])
    p.add_argument("--port",           type=int, default=CONFIG["SERVER"]["PORT"])
    p.add_argument("--reload",         action="store_true", help="Dev hot-reload (not for prod)")
    p.add_argument("--secure-cookies", action="store_true",
                   help="Mark auth cookies Secure (requires HTTPS)")
    args = p.parse_args()

    _SECURE_COOKIES = args.secure_cookies  # type: ignore[assignment]

    scheme = "https" if _SECURE_COOKIES else "http"
    localhost_only = CONFIG["SERVER"].get("LOCALHOST_ONLY", True)
    print(f"\n  Project Challenger Web GUI")
    print(f"  Listening on  {scheme}://{args.host}:{args.port}")
    print(f"  Open          {scheme}://127.0.0.1:{args.port}  in your browser")
    print(f"  Localhost-only guard: {'ENABLED' if localhost_only else 'DISABLED — external access allowed'}")
    if not localhost_only:
        print(f"\n  WARNING: SERVER_LOCALHOST_ONLY is False. Requests from any IP")
        print(f"  will be accepted. Ensure you have a firewall or HTTPS reverse proxy.")
    if not _SECURE_COOKIES:
        print(f"\n  WARNING: --secure-cookies is not set. Auth cookies will be")
        print(f"  transmitted over plain HTTP. Use --secure-cookies when running")
        print(f"  behind an HTTPS reverse proxy.")
    print()

    uvicorn.run(
        "web_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
