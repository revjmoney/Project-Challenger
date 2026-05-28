"""
SQLite database layer — thread-safe reads/writes with WAL mode.
Tables:
  model_signals          — per-tick signals from each model worker
  paper_trades           — completed paper trades
  portfolio_snapshots    — periodic equity snapshots
  training_results       — per-fold CV metrics logged during training
  candle_cache           — rolling 24-hour OHLCV history (all exchanges)
  backtest_results       — summary of each backtest run
  live_trades            — real orders placed via Coinbase API
"""
import sqlite3
import time
from config import CONFIG

DB_PATH = CONFIG["PATHS"]["DB"]


# ─────────────────────────────────────────────────────────────── init ─────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS model_signals (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        REAL    NOT NULL,
            model_name       TEXT    NOT NULL,
            symbol           TEXT    NOT NULL DEFAULT '',
            signal_direction INTEGER NOT NULL,
            predicted_value  REAL,
            trigger_price    REAL    NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        REAL    NOT NULL,
            model_name       TEXT    NOT NULL,
            symbol           TEXT    NOT NULL DEFAULT '',
            side             TEXT    NOT NULL,
            target_price     REAL    NOT NULL,
            execution_price  REAL    NOT NULL,
            slippage_cost    REAL    NOT NULL,
            fee_paid         REAL    NOT NULL,
            quantity         REAL    NOT NULL,
            realized_pnl     REAL    DEFAULT 0.0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       REAL    NOT NULL,
            model_name      TEXT    NOT NULL,
            symbol          TEXT    NOT NULL DEFAULT '',
            capital         REAL    NOT NULL,
            position_side   TEXT,
            position_qty    REAL    DEFAULT 0.0,
            unrealized_pnl  REAL    DEFAULT 0.0,
            total_pnl       REAL    DEFAULT 0.0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS training_results (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at    REAL    NOT NULL,
            model_name    TEXT    NOT NULL,
            fold          INTEGER NOT NULL,
            train_rows    INTEGER NOT NULL,
            test_rows     INTEGER NOT NULL,
            test_mse      REAL    NOT NULL,
            test_mae      REAL    NOT NULL,
            direction_acc REAL    NOT NULL
        )
    """)

    # Rolling OHLCV cache — populated by DataManager, queried by backtester
    c.execute("""
        CREATE TABLE IF NOT EXISTS candle_cache (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT    NOT NULL,
            symbol   TEXT    NOT NULL,
            ts       INTEGER NOT NULL,
            open     REAL    NOT NULL,
            high     REAL    NOT NULL,
            low      REAL    NOT NULL,
            close    REAL    NOT NULL,
            volume   REAL    NOT NULL,
            UNIQUE(exchange, symbol, ts)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_candle_cache ON candle_cache(exchange, symbol, ts)")

    # Backtest run summaries
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ran_at           REAL    NOT NULL,
            model_name       TEXT    NOT NULL,
            exchange         TEXT    NOT NULL,
            symbol           TEXT    NOT NULL,
            lookback_hours   INTEGER NOT NULL,
            initial_capital  REAL    NOT NULL,
            signal_threshold REAL    NOT NULL,
            total_candles    INTEGER NOT NULL,
            total_trades     INTEGER NOT NULL,
            wins             INTEGER NOT NULL,
            losses           INTEGER NOT NULL,
            net_pnl          REAL    NOT NULL,
            win_rate         REAL    NOT NULL,
            profit_factor    REAL    NOT NULL,
            max_drawdown     REAL    NOT NULL
        )
    """)

    # Real orders placed via Coinbase API
    c.execute("""
        CREATE TABLE IF NOT EXISTS live_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       REAL    NOT NULL,
            model_name      TEXT    NOT NULL,
            exchange        TEXT    NOT NULL,
            symbol          TEXT    NOT NULL,
            side            TEXT    NOT NULL,
            order_id        TEXT,
            requested_price REAL    NOT NULL,
            fill_price      REAL    NOT NULL,
            size_usd        REAL    NOT NULL,
            quantity        REAL    NOT NULL,
            fee_paid        REAL    NOT NULL,
            realized_pnl    REAL    DEFAULT 0.0,
            status          TEXT    NOT NULL
        )
    """)

    # Single-row table that persists the live Coinbase position across restarts.
    # Uses a fixed id=1 so INSERT OR REPLACE always overwrites the same row.
    c.execute("""
        CREATE TABLE IF NOT EXISTS live_position_state (
            id          INTEGER PRIMARY KEY,
            model_name  TEXT NOT NULL,
            side        TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity    REAL NOT NULL,
            size_usd    REAL NOT NULL,
            capital     REAL NOT NULL,
            opened_at   REAL NOT NULL
        )
    """)

    # Available products fetched from each exchange (cached for 24h)
    c.execute("""
        CREATE TABLE IF NOT EXISTS available_coins (
            exchange   TEXT NOT NULL,
            base       TEXT NOT NULL,
            symbol     TEXT NOT NULL,
            fetched_at REAL NOT NULL,
            PRIMARY KEY (exchange, symbol)
        )
    """)

    conn.commit()
    conn.close()
    _migrate_symbol_columns()
    print("[DB] Database initialized.")


def _migrate_symbol_columns():
    """
    Non-destructive migration: add the 'symbol' column to signal/trade tables
    if it is missing (existing rows get an empty string).
    Safe to call on every startup — does nothing when columns already exist.
    """
    conn = _connect()
    try:
        for table in ("model_signals", "paper_trades", "portfolio_snapshots"):
            existing = {r[1] for r in conn.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()}
            if "symbol" not in existing:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN symbol TEXT NOT NULL DEFAULT ''"
                )
                conn.commit()
    except Exception as e:
        print(f"[DB MIGRATE] symbol column: {e}")
    finally:
        conn.close()


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ───────────────────────────────────────────────── paper-trading writes ───────

def log_signal(model_name, direction, trigger_price, predicted_val=0.0, symbol=""):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO model_signals "
            "(timestamp,model_name,symbol,signal_direction,predicted_value,trigger_price) "
            "VALUES (?,?,?,?,?,?)",
            (time.time(), model_name, symbol, direction, predicted_val, trigger_price),
        )
        conn.commit()
    except Exception as e:
        print(f"[DB SIGNAL ERROR] {model_name}: {e}")
    finally:
        conn.close()


def log_trade(model_name, side, target_price, execution_price,
              slippage, fee, qty, pnl=0.0, symbol=""):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO paper_trades "
            "(timestamp,model_name,symbol,side,target_price,execution_price,"
            "slippage_cost,fee_paid,quantity,realized_pnl) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (time.time(), model_name, symbol, side,
             target_price, execution_price, slippage, fee, qty, pnl),
        )
        conn.commit()
    except Exception as e:
        print(f"[DB TRADE ERROR] {model_name}: {e}")
    finally:
        conn.close()


def log_portfolio(model_name, capital, position_side, position_qty,
                  unrealized_pnl, total_pnl, symbol=""):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO portfolio_snapshots "
            "(timestamp,model_name,symbol,capital,position_side,"
            "position_qty,unrealized_pnl,total_pnl) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), model_name, symbol, capital,
             position_side, position_qty, unrealized_pnl, total_pnl),
        )
        conn.commit()
    except Exception as e:
        print(f"[DB PORTFOLIO ERROR] {model_name}: {e}")
    finally:
        conn.close()


def log_training_result(model_name, fold, train_rows, test_rows, mse, mae, direction_acc):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO training_results "
            "(trained_at,model_name,fold,train_rows,test_rows,test_mse,test_mae,direction_acc) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), model_name, fold, train_rows, test_rows, mse, mae, direction_acc),
        )
        conn.commit()
    except Exception as e:
        print(f"[DB TRAIN ERROR] {model_name}: {e}")
    finally:
        conn.close()


# ───────────────────────────────────────────────── candle cache ───────────────

def store_candles(candles: list[dict], exchange: str, symbol: str):
    """Upsert candles into the cache. Ignores conflicts (existing rows stay)."""
    if not candles:
        return
    conn = _connect()
    try:
        conn.executemany(
            "INSERT OR IGNORE INTO candle_cache "
            "(exchange,symbol,ts,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)",
            [
                (exchange, symbol, int(c["start"]),
                 float(c["open"]), float(c["high"]), float(c["low"]),
                 float(c["close"]), float(c["volume"]))
                for c in candles
            ],
        )
        conn.commit()
    except Exception as e:
        print(f"[DB CACHE ERROR] {e}")
    finally:
        conn.close()


def get_cached_candles(exchange: str, symbol: str, hours: int = 24) -> list[dict]:
    """Return candles from the cache as normalised dicts, oldest-first."""
    since = int(time.time()) - hours * 3600
    conn  = _connect()
    try:
        rows = conn.execute(
            "SELECT ts,open,high,low,close,volume FROM candle_cache "
            "WHERE exchange=? AND symbol=? AND ts>=? ORDER BY ts ASC",
            (exchange, symbol, since),
        ).fetchall()
        return [
            {"start": str(r[0]), "open": str(r[1]), "high": str(r[2]),
             "low": str(r[3]), "close": str(r[4]), "volume": str(r[5])}
            for r in rows
        ]
    finally:
        conn.close()


def prune_old_candles(exchange: str, symbol: str, max_hours: int = 24):
    """Delete candles older than max_hours from the cache."""
    cutoff = int(time.time()) - max_hours * 3600
    conn   = _connect()
    try:
        conn.execute(
            "DELETE FROM candle_cache WHERE exchange=? AND symbol=? AND ts<?",
            (exchange, symbol, cutoff),
        )
        conn.commit()
    finally:
        conn.close()


def get_candle_cache_count(exchange: str, symbol: str) -> int:
    conn = _connect()
    try:
        r = conn.execute(
            "SELECT COUNT(*) FROM candle_cache WHERE exchange=? AND symbol=?",
            (exchange, symbol),
        ).fetchone()
        return r[0] if r else 0
    finally:
        conn.close()


def get_latest_prices(exchange: str) -> list[tuple[str, float, float]]:
    """
    Return (symbol, last_close, prev_close) for every symbol that has cached
    candles on the given exchange, ordered by symbol name.
    prev_close is the second-most-recent close (used for % change display).
    """
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT symbol,
                   MAX(CASE WHEN rn = 1 THEN close END) AS last_close,
                   MAX(CASE WHEN rn = 2 THEN close END) AS prev_close
            FROM (
                SELECT symbol, close,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
                FROM candle_cache
                WHERE exchange = ?
            )
            WHERE rn <= 2
            GROUP BY symbol
            ORDER BY symbol
            """,
            (exchange,),
        ).fetchall()
        return [(r[0], r[1] or 0.0, r[2] or 0.0) for r in rows if r[1]]
    except Exception:
        return []
    finally:
        conn.close()


# ───────────────────────────────────────────────── backtest results ───────────

def store_backtest_result(result, cfg, exchange: str, symbol: str):
    """Persist a BacktestResult summary to the database."""
    conn = _connect()
    try:
        pf = result.profit_factor
        if pf == float("inf"):
            pf = 9999.0
        conn.execute(
            "INSERT INTO backtest_results "
            "(ran_at,model_name,exchange,symbol,lookback_hours,initial_capital,"
            "signal_threshold,total_candles,total_trades,wins,losses,"
            "net_pnl,win_rate,profit_factor,max_drawdown) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                time.time(), result.model_name, exchange, symbol,
                cfg.lookback_hours, cfg.initial_capital, cfg.signal_threshold,
                result.total_candles, result.total_trades,
                result.wins, result.losses,
                result.net_pnl, result.win_rate, pf, result.max_drawdown,
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"[DB BT ERROR] {e}")
    finally:
        conn.close()


def get_backtest_results(limit: int = 20) -> list:
    """Return the most recent backtest results."""
    conn = _connect()
    try:
        return conn.execute(
            "SELECT model_name,ran_at,exchange,symbol,lookback_hours,"
            "total_trades,wins,losses,net_pnl,win_rate,profit_factor,max_drawdown "
            "FROM backtest_results ORDER BY ran_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()


# ───────────────────────────────────────────────── live trade writes ──────────

def log_live_trade(model_name, exchange, symbol, side, order_id,
                   requested_price, fill_price, size_usd, quantity,
                   fee_paid, realized_pnl=0.0, status="OPEN"):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO live_trades "
            "(timestamp,model_name,exchange,symbol,side,order_id,"
            "requested_price,fill_price,size_usd,quantity,fee_paid,realized_pnl,status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), model_name, exchange, symbol, side, order_id,
             requested_price, fill_price, size_usd, quantity,
             fee_paid, realized_pnl, status),
        )
        conn.commit()
    except Exception as e:
        print(f"[DB LIVE TRADE ERROR] {e}")
    finally:
        conn.close()


def get_live_trade_stats(model_name: str):
    """Returns (count, wins, losses, net_pnl, total_fees) for live trades."""
    conn = _connect()
    try:
        return conn.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN realized_pnl>0 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN realized_pnl<0 THEN 1 ELSE 0 END), "
            "SUM(realized_pnl), SUM(fee_paid) "
            "FROM live_trades WHERE model_name=? AND status='CLOSED'",
            (model_name,),
        ).fetchone()
    finally:
        conn.close()


def get_paper_trading_start_time(model_name: str, symbol: str | None = None):
    """Returns Unix timestamp of the first paper trade for model_name, or None."""
    conn = _connect()
    try:
        if symbol:
            row = conn.execute(
                "SELECT MIN(timestamp) FROM paper_trades WHERE model_name=? AND symbol=?",
                (model_name, symbol),
            ).fetchone()
            return row[0] if row and row[0] else None
        row = conn.execute(
            "SELECT MIN(timestamp) FROM paper_trades WHERE model_name=?",
            (model_name,),
        ).fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()


def get_latest_signal_id(model_name: str, symbol: str | None = None):
    """Returns the MAX signal id for the given model (for live-trade polling)."""
    conn = _connect()
    try:
        if symbol:
            row = conn.execute(
                "SELECT MAX(id) FROM model_signals WHERE model_name=? AND symbol=?",
                (model_name, symbol),
            ).fetchone()
            return row[0] if row and row[0] else 0
        row = conn.execute(
            "SELECT MAX(id) FROM model_signals WHERE model_name=?",
            (model_name,),
        ).fetchone()
        return row[0] if row and row[0] else 0
    finally:
        conn.close()


def get_signals_after(model_name: str, after_id: int, symbol: str | None = None) -> list:
    """Fetch model signals newer than after_id."""
    conn = _connect()
    try:
        if symbol:
            return conn.execute(
                "SELECT id,signal_direction,trigger_price,symbol FROM model_signals "
                "WHERE model_name=? AND symbol=? AND id>? ORDER BY id ASC",
                (model_name, symbol, after_id),
            ).fetchall()
        return conn.execute(
            "SELECT id,signal_direction,trigger_price FROM model_signals "
            "WHERE model_name=? AND id>? ORDER BY id ASC",
            (model_name, after_id),
        ).fetchall()
    finally:
        conn.close()


# ───────────────────────────────────────────────── read helpers ───────────────

def get_model_stats(model_name: str, symbol: str | None = None):
    """Returns (count, wins, losses, pnl, fees, slippage, gross_profit, gross_loss)."""
    conn = _connect()
    try:
        if symbol:
            return conn.execute("""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END),
                    SUM(realized_pnl),
                    SUM(fee_paid),
                    SUM(slippage_cost),
                    SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END),
                    SUM(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) ELSE 0 END)
                FROM paper_trades WHERE model_name = ? AND symbol = ?
            """, (model_name, symbol)).fetchone()
        return conn.execute("""
            SELECT
                COUNT(*),
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END),
                SUM(realized_pnl),
                SUM(fee_paid),
                SUM(slippage_cost),
                SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END),
                SUM(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) ELSE 0 END)
            FROM paper_trades WHERE model_name = ?
        """, (model_name,)).fetchone()
    finally:
        conn.close()


def get_latest_portfolio(model_name: str):
    conn = _connect()
    try:
        return conn.execute(
            "SELECT capital,position_side,unrealized_pnl,total_pnl "
            "FROM portfolio_snapshots WHERE model_name=? ORDER BY timestamp DESC LIMIT 1",
            (model_name,),
        ).fetchone()
    finally:
        conn.close()


def get_all_model_names():
    conn = _connect()
    try:
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT model_name FROM paper_trades"
        ).fetchall()]
    finally:
        conn.close()


def get_training_results():
    conn = _connect()
    try:
        return conn.execute(
            "SELECT model_name,fold,train_rows,test_rows,test_mse,test_mae,direction_acc,trained_at "
            "FROM training_results ORDER BY trained_at DESC, model_name, fold"
        ).fetchall()
    finally:
        conn.close()


def store_available_coins(exchange: str, coins: list[dict]) -> None:
    """
    Cache the product list for an exchange.
    Each dict must have 'base' (e.g. 'BTC') and 'symbol' (e.g. 'BTC-USD').
    Replaces the entire cached list for that exchange.
    """
    if not coins:
        return
    now  = time.time()
    conn = _connect()
    try:
        conn.execute("DELETE FROM available_coins WHERE exchange=?", (exchange,))
        conn.executemany(
            "INSERT OR REPLACE INTO available_coins (exchange, base, symbol, fetched_at) "
            "VALUES (?, ?, ?, ?)",
            [(exchange, c["base"], c["symbol"], now) for c in coins],
        )
        conn.commit()
    except Exception as e:
        print(f"[DB COINS ERROR] {e}")
    finally:
        conn.close()


def get_available_coins(exchange: str, max_age_hours: int = 24) -> list[dict]:
    """
    Return the cached product list for an exchange.
    Returns [] if the cache is empty or older than max_age_hours.
    Each entry: {'base': str, 'symbol': str}
    """
    cutoff = time.time() - max_age_hours * 3600
    conn   = _connect()
    try:
        rows = conn.execute(
            "SELECT base, symbol, fetched_at FROM available_coins "
            "WHERE exchange=? AND fetched_at>=? ORDER BY base ASC",
            (exchange, cutoff),
        ).fetchall()
        return [{"base": r[0], "symbol": r[1]} for r in rows]
    finally:
        conn.close()


def get_training_summary():
    conn = _connect()
    try:
        return conn.execute("""
            SELECT model_name,
                   COUNT(*)           AS folds,
                   AVG(test_mse)      AS avg_mse,
                   AVG(test_mae)      AS avg_mae,
                   AVG(direction_acc) AS avg_dir_acc,
                   MAX(trained_at)    AS last_trained
            FROM training_results
            GROUP BY model_name
            ORDER BY avg_mse ASC
        """).fetchall()
    finally:
        conn.close()


# ─────────────────────────────────────── live position state persistence ───────

def save_live_position(model_name: str, position: dict, capital: float) -> None:
    """Persist the current live Coinbase position so it survives restarts."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO live_position_state "
            "(id, model_name, side, entry_price, quantity, size_usd, capital, opened_at) "
            "VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
            (
                model_name,
                position["side"],
                position["entry"],
                position["qty"],
                position["size_usd"],
                capital,
                time.time(),
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"[DB LIVE POS ERROR] {e}")
    finally:
        conn.close()


def load_live_position() -> tuple | None:
    """
    Return (model_name, position_dict, capital) if a saved position exists,
    otherwise None.  Called on arm() to restore state after a crash/restart.
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT model_name, side, entry_price, quantity, size_usd, capital "
            "FROM live_position_state WHERE id=1"
        ).fetchone()
        if not row:
            return None
        model_name, side, entry, qty, size_usd, capital = row
        position = {"side": side, "entry": entry, "qty": qty, "size_usd": size_usd}
        return model_name, position, capital
    finally:
        conn.close()


def clear_live_position() -> None:
    """Delete the persisted position record after a successful close."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM live_position_state WHERE id=1")
        conn.commit()
    except Exception as e:
        print(f"[DB LIVE POS CLEAR ERROR] {e}")
    finally:
        conn.close()
