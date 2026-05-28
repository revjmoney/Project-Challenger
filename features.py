"""
Feature engineering for historical DataFrames and live tick-by-tick streams.
"""
import time
import numpy as np
import pandas as pd

from config import CONFIG, FEATURE_COLS, TARGET_COL


# ---------------------------------------------------------------------------
# Core math helpers (no pandas-ta dependency required)
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Batch feature calculation (used by training.py)
# ---------------------------------------------------------------------------

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : DataFrame with columns [open, high, low, close, volume].
    Output: Same DataFrame with FEATURE_COLS and TARGET_COL appended.
    Rows with NaN features are dropped.
    """
    df = df.copy().sort_index()

    df["log_return"]     = np.log(df["close"] / df["close"].shift(1))
    df["rsi_14"]         = _rsi(df["close"], 14)
    df["atr_14"]         = _atr(df["high"], df["low"], df["close"], 14)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["ema_diff"]       = (ema12 - ema26) / ema26.replace(0, np.nan)

    df["rolling_vol_20"] = df["log_return"].rolling(20).std()

    if "obi" not in df.columns:
        df["obi"] = 0.0

    df[TARGET_COL] = df["log_return"].shift(-1)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


# ---------------------------------------------------------------------------
# Candle Aggregator (Tick-to-Candle bucketing)
# ---------------------------------------------------------------------------

class CandleAggregator:
    """
    Buckets incoming ticks into OHLCV candles of a fixed granularity.
    Returns a candle dict when a new bucket starts, otherwise None.
    """
    def __init__(self, granularity_seconds: int):
        self.interval = granularity_seconds
        self.current_bucket = None  # (ts_start, open, high, low, close, volume)
        self._last_tick_time = 0

    def update(self, price: float, volume: float = 0.0, ts: float | None = None) -> dict | None:
        """
        Push a tick. Returns a completed candle dict {start, open, high, low, close, volume}
        only when the tick moves into a NEW time bucket.
        """
        now = ts or time.time()
        bucket_start = int(now // self.interval) * self.interval

        completed_candle = None

        if self.current_bucket is None:
            # First tick ever
            self.current_bucket = {
                "start": bucket_start,
                "open":  price,
                "high":  price,
                "low":   price,
                "close": price,
                "volume": volume
            }
        elif bucket_start > self.current_bucket["start"]:
            # We've moved into a new interval. Return the finished one.
            completed_candle = self.current_bucket.copy()
            # Start the next one
            self.current_bucket = {
                "start": bucket_start,
                "open":  price,
                "high":  price,
                "low":   price,
                "close": price,
                "volume": volume
            }
        else:
            # Still in the same bucket. Update H/L/C/V.
            self.current_bucket["high"]  = max(self.current_bucket["high"], price)
            self.current_bucket["low"]   = min(self.current_bucket["low"], price)
            self.current_bucket["close"]  = price
            self.current_bucket["volume"] += volume

        return completed_candle


# ---------------------------------------------------------------------------
# Live streaming feature engine
# ---------------------------------------------------------------------------

class LiveFeatureEngine:
    """
    Maintains rolling OHLCV buffers and OBI state for real-time inference.
    Call push_candle() each time a new 1-min candle closes,
    update_obi() each time a level2 book update arrives.
    """

    def __init__(self, buffer_size: int = 120):
        self.buffer_size = buffer_size
        self._closes  = []
        self._highs   = []
        self._lows    = []
        self._bid_sz  = 1.0
        self._ask_sz  = 1.0

    # ---- public API --------------------------------------------------------

    def update_obi(self, bid_size: float, ask_size: float):
        self._bid_sz = bid_size
        self._ask_sz = ask_size

    def push_candle(self, open_: float, high: float, low: float, close: float):
        self._closes.append(close)
        self._highs.append(high)
        self._lows.append(low)
        if len(self._closes) > self.buffer_size:
            self._closes.pop(0)
            self._highs.pop(0)
            self._lows.pop(0)

    def get_feature_vector(self) -> dict | None:
        """Returns feature dict keyed by FEATURE_COLS, or None if not enough data."""
        if len(self._closes) < 32:   # need at least 30 + warm-up
            return None

        s = pd.Series(self._closes, dtype=float)
        h = pd.Series(self._highs,  dtype=float)
        l = pd.Series(self._lows,   dtype=float)

        log_return = float(np.log(s.iloc[-1] / s.iloc[-2]))

        rsi_series = _rsi(s, 14)
        rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

        atr_series = _atr(h, l, s, 14)
        atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0

        ema12    = float(s.ewm(span=12, adjust=False).mean().iloc[-1])
        ema26    = float(s.ewm(span=26, adjust=False).mean().iloc[-1])
        ema_diff = (ema12 - ema26) / ema26 if ema26 != 0 else 0.0

        log_rets = np.log(s / s.shift(1)).dropna()
        vol = float(log_rets.rolling(20).std().iloc[-1])
        if np.isnan(vol):
            vol = 0.0

        total = self._bid_sz + self._ask_sz
        live_obi = (self._bid_sz - self._ask_sz) / total if total > 0 else 0.0

        # When ZERO_OBI_IN_INFERENCE=True (default), send 0.0 to match the
        # all-zero OBI distribution the models were trained on.
        # Set to False only if you retrain with real order-book data.
        obi = 0.0 if CONFIG["PAPER_TRADING"].get("ZERO_OBI_IN_INFERENCE", True) else live_obi

        return {
            "current_price":    self._closes[-1],
            "log_return":       log_return,
            "rsi_14":           rsi,
            "atr_14":           atr,
            "ema_diff":         ema_diff,
            "rolling_vol_20":   vol,
            "obi":              obi,
        }

    def as_feature_array(self) -> "np.ndarray | None":
        """Returns a 1-D numpy array in FEATURE_COLS order, or None."""
        vec = self.get_feature_vector()
        if vec is None:
            return None
        return np.array([vec[f] for f in FEATURE_COLS], dtype=np.float32)
