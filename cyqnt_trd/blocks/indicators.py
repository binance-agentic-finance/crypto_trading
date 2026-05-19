"""Technical indicators implemented in pure pandas / numpy.

All indicators accept either a ``pandas.Series`` (for single-series
indicators like :func:`sma`) or a ``pandas.DataFrame`` with the standard
OHLCV columns (for multi-series indicators like :func:`adx`).

All public functions are pure (no side effects) and look-ahead-safe (a
value at index ``i`` only depends on values at indices ``<= i``).

Examples
--------
>>> from cyqnt_trd.blocks import indicators as ind
>>> ma20 = ind.sma(df["close"], 20)
>>> macd_line, signal_line, hist = ind.macd(df["close"])
>>> adx_val, plus_di, minus_di = ind.adx(df, period=14)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ._utils import (
    SeriesLike,
    ensure_df,
    ensure_series,
    positive_int,
    rolling_max,
    rolling_min,
    safe_divide,
)

__all__ = [
    "sma",
    "ema",
    "wma",
    "rma",
    "rsi",
    "macd",
    "adx",
    "atr",
    "true_range",
    "bollinger",
    "bollinger_bands",
    "donchian",
    "stochastic",
    "vwap",
    "obv",
    "volume_ma",
    "volume_zscore",
    "ma_direction",
    "ma_alignment",
    "swing_high",
    "swing_low",
    "highest",
    "lowest",
    "price_change_pct",
    "supertrend",
    "ichimoku",
    "parabolic_sar",
    "rolling_zscore",
    "rolling_quantile",
    # convenience re-exports
    "candle_lower_shadow",
    "candle_upper_shadow",
]


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


def sma(series: SeriesLike, period: int) -> pd.Series:
    """Simple moving average over *period* bars."""
    period = positive_int(period, "period")
    return ensure_series(series).rolling(window=period, min_periods=period).mean()


def ema(series: SeriesLike, period: int) -> pd.Series:
    """Exponential moving average (standard ``alpha = 2 / (period + 1)``)."""
    period = positive_int(period, "period")
    return ensure_series(series).ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: SeriesLike, period: int) -> pd.Series:
    """Linearly weighted moving average."""
    period = positive_int(period, "period")
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    s = ensure_series(series)
    return s.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights), raw=True
    )


def rma(series: SeriesLike, period: int) -> pd.Series:
    """Wilder's smoothing (RMA) — used by RSI / ADX / ATR."""
    period = positive_int(period, "period")
    s = ensure_series(series).astype(float)
    # Wilder's: equivalent to EMA with alpha = 1/period
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# RSI / MACD
# ---------------------------------------------------------------------------


def rsi(series: SeriesLike, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder).

    Returns values in ``[0, 100]``. The first ``period`` values are NaN.
    """
    period = positive_int(period, "period")
    s = ensure_series(series).astype(float)
    delta = s.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)
    rs = safe_divide(avg_gain, avg_loss, fill=0.0)
    out = 100.0 - (100.0 / (1.0 + rs))
    # Special cases: avg_loss == 0 with non-zero gain → RSI = 100; avg_gain == 0
    # with non-zero loss → RSI = 0. Importantly, we must NOT overwrite the early
    # leading-NaN region (where avg_gain / avg_loss are themselves NaN) — leaving
    # it NaN is the correct, look-ahead-safe behaviour.
    valid = avg_gain.notna() & avg_loss.notna()
    only_gain = valid & (avg_loss == 0) & (avg_gain > 0)
    only_loss = valid & (avg_gain == 0) & (avg_loss > 0)
    out = out.where(~only_gain, 100.0)
    out = out.where(~only_loss, 0.0)
    out = out.where(valid, np.nan)
    return out


def macd(
    series: SeriesLike,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram.

    Returns
    -------
    (macd_line, signal_line, histogram) : tuple of pandas.Series
        ``macd_line   = EMA(fast) - EMA(slow)``
        ``signal_line = EMA(macd_line, signal)``
        ``histogram   = macd_line - signal_line``
    """
    fast = positive_int(fast, "fast")
    slow = positive_int(slow, "slow")
    signal = positive_int(signal, "signal")
    if slow <= fast:
        raise ValueError(f"slow ({slow}) must be greater than fast ({fast})")
    s = ensure_series(series).astype(float)
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ---------------------------------------------------------------------------
# ADX / ATR
# ---------------------------------------------------------------------------


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's True Range = max(H-L, |H-Cprev|, |L-Cprev|)."""
    df = ensure_df(df, required=("high", "low", "close"))
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    period = positive_int(period, "period")
    return rma(true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index plus +DI / -DI.

    Returns
    -------
    (adx, plus_di, minus_di) : tuple of pandas.Series
    """
    period = positive_int(period, "period")
    df = ensure_df(df, required=("high", "low", "close"))
    high = df["high"]
    low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)
    tr = true_range(df)
    atr_ = rma(tr, period)
    plus_di = 100.0 * safe_divide(rma(plus_dm, period), atr_, fill=0.0)
    minus_di = 100.0 * safe_divide(rma(minus_dm, period), atr_, fill=0.0)
    dx = 100.0 * safe_divide((plus_di - minus_di).abs(), (plus_di + minus_di), fill=0.0)
    adx_ = rma(dx, period)
    return adx_, plus_di, minus_di


# ---------------------------------------------------------------------------
# Volatility / channel indicators
# ---------------------------------------------------------------------------


def bollinger(
    series: SeriesLike, period: int = 20, std_mult: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns
    -------
    (upper, middle, lower) : tuple of pandas.Series
    """
    period = positive_int(period, "period")
    if std_mult <= 0:
        raise ValueError(f"std_mult must be > 0, got {std_mult}")
    s = ensure_series(series).astype(float)
    mid = sma(s, period)
    sd = s.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    return upper, mid, lower


def donchian(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channel: rolling high / low / midpoint over *period* bars.

    Returns
    -------
    (upper, lower, middle) : tuple of pandas.Series
    """
    period = positive_int(period, "period")
    df = ensure_df(df, required=("high", "low"))
    upper = df["high"].rolling(window=period, min_periods=period).max()
    lower = df["low"].rolling(window=period, min_periods=period).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator (slow).

    ``%K = SMA(((close - LL) / (HH - LL) * 100), smooth_k)``
    ``%D = SMA(%K, d_period)``
    """
    k_period = positive_int(k_period, "k_period")
    d_period = positive_int(d_period, "d_period")
    smooth_k = positive_int(smooth_k, "smooth_k")
    df = ensure_df(df, required=("high", "low", "close"))
    hh = df["high"].rolling(window=k_period, min_periods=k_period).max()
    ll = df["low"].rolling(window=k_period, min_periods=k_period).min()
    raw_k = 100.0 * safe_divide(df["close"] - ll, hh - ll, fill=50.0)
    k = sma(raw_k, smooth_k)
    d = sma(k, d_period)
    return k, d


# ---------------------------------------------------------------------------
# Volume-based
# ---------------------------------------------------------------------------


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP (running, not session-reset).

    Use ``vwap(df.groupby(...).apply(lambda x: vwap(x)))`` for a session
    VWAP if you have a session column.
    """
    df = ensure_df(df, required=("high", "low", "close", "volume"))
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv_cum = (typical * df["volume"]).cumsum()
    vol_cum = df["volume"].cumsum()
    return safe_divide(pv_cum, vol_cum, fill=0.0)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    df = ensure_df(df, required=("close", "volume"))
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    df = ensure_df(df, required=("volume",))
    return sma(df["volume"], period)


def volume_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume z-score using rolling mean & std-dev."""
    df = ensure_df(df, required=("volume",))
    return rolling_zscore(df["volume"], period)


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------


def ma_direction(ma_series: SeriesLike, lookback: int = 5, flat_threshold_bps: float = 5.0) -> pd.Series:
    """Classify MA slope into ``"up"`` / ``"down"`` / ``"flat"``.

    *flat_threshold_bps* is the absolute pct-change threshold (in basis
    points, 1 bp = 0.01%) below which the MA is considered flat.
    """
    lookback = positive_int(lookback, "lookback")
    s = ensure_series(ma_series).astype(float)
    pct = s.pct_change(periods=lookback) * 10_000.0  # bps
    flat = pct.abs() <= flat_threshold_bps
    direction = pd.Series(np.where(pct > 0, "up", "down"), index=s.index)
    direction = direction.where(~flat, "flat")
    direction = direction.where(s.notna(), other=np.nan)
    return direction


def ma_alignment(*ma_series: pd.Series) -> pd.Series:
    """Classify MA stack into ``"bullish"`` / ``"bearish"`` / ``"mixed"``.

    Bullish = ``ma1 > ma2 > ... > maN`` (typically fast > medium > slow).
    """
    if len(ma_series) < 2:
        raise ValueError("need at least two MA series")
    df = pd.concat(list(ma_series), axis=1)
    diffs = df.diff(axis=1)
    # diff(axis=1) yields NaN for the first column; drop it
    rest = diffs.iloc[:, 1:]
    bullish = (rest < 0).all(axis=1)
    bearish = (rest > 0).all(axis=1)
    out = pd.Series("mixed", index=df.index)
    out = out.where(~bullish, "bullish")
    out = out.where(~bearish, "bearish")
    return out


def swing_high(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Rolling N-bar swing high (highest high in the last N bars)."""
    return rolling_max(ensure_df(df, required=("high",))["high"], lookback)


def swing_low(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Rolling N-bar swing low (lowest low in the last N bars)."""
    return rolling_min(ensure_df(df, required=("low",))["low"], lookback)


def highest(series: SeriesLike, period: int) -> pd.Series:
    """Rolling N-bar maximum of a generic series."""
    return rolling_max(series, period)


def lowest(series: SeriesLike, period: int) -> pd.Series:
    """Rolling N-bar minimum of a generic series."""
    return rolling_min(series, period)


def price_change_pct(series: SeriesLike, periods: int = 1) -> pd.Series:
    """Simple percent change over *periods* bars."""
    return ensure_series(series).pct_change(periods=periods)


# ---------------------------------------------------------------------------
# Composite indicators
# ---------------------------------------------------------------------------


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """SuperTrend indicator (canonical implementation).

    Returns
    -------
    (supertrend_value, direction) : tuple
        ``direction``: +1 means uptrend (price above ST), -1 means downtrend.
    """
    period = positive_int(period, "period")
    df = ensure_df(df, required=("high", "low", "close"))
    n = len(df)
    atr_arr = atr(df, period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy()
    close = df["close"].to_numpy()
    upper_basic = hl2 + multiplier * atr_arr
    lower_basic = hl2 - multiplier * atr_arr

    final_upper = upper_basic.copy()
    final_lower = lower_basic.copy()
    direction_arr = np.zeros(n, dtype=int)
    st_arr = np.full(n, np.nan)
    started = False

    for i in range(1, n):
        if np.isnan(upper_basic[i]) or np.isnan(lower_basic[i]):
            continue
        if started:
            if not (upper_basic[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]):
                final_upper[i] = final_upper[i - 1]
            if not (lower_basic[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]):
                final_lower[i] = final_lower[i - 1]
            prev_dir = direction_arr[i - 1]
            if prev_dir == 1 and close[i] < final_lower[i]:
                direction_arr[i] = -1
            elif prev_dir == -1 and close[i] > final_upper[i]:
                direction_arr[i] = 1
            else:
                direction_arr[i] = prev_dir
        else:
            direction_arr[i] = 1 if close[i] > final_upper[i] else -1
            started = True
        st_arr[i] = final_lower[i] if direction_arr[i] == 1 else final_upper[i]

    return (
        pd.Series(st_arr, index=df.index),
        pd.Series(direction_arr, index=df.index),
    )


def ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """Ichimoku Cloud (5 components).

    Returns a DataFrame with columns
    ``tenkan_sen``, ``kijun_sen``, ``senkou_span_a``,
    ``senkou_span_b``, ``chikou_span``.
    """
    df = ensure_df(df, required=("high", "low", "close"))
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tenkan_sen = (rolling_max(high, tenkan) + rolling_min(low, tenkan)) / 2.0
    kijun_sen = (rolling_max(high, kijun) + rolling_min(low, kijun)) / 2.0
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0).shift(displacement)
    senkou_b_ = ((rolling_max(high, senkou_b) + rolling_min(low, senkou_b)) / 2.0).shift(
        displacement
    )
    chikou = close.shift(-displacement)
    return pd.DataFrame(
        {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_a,
            "senkou_span_b": senkou_b_,
            "chikou_span": chikou,
        }
    )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def rolling_zscore(series: SeriesLike, period: int) -> pd.Series:
    """Rolling z-score: ``(x - mean) / std`` over *period* bars."""
    period = positive_int(period, "period")
    s = ensure_series(series).astype(float)
    mu = s.rolling(window=period, min_periods=period).mean()
    sd = s.rolling(window=period, min_periods=period).std(ddof=0)
    return safe_divide(s - mu, sd, fill=0.0)


def rolling_quantile(series: SeriesLike, period: int, q: float) -> pd.Series:
    """Rolling quantile (0 <= q <= 1)."""
    period = positive_int(period, "period")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be within [0, 1], got {q}")
    return ensure_series(series).rolling(window=period, min_periods=period).quantile(q)


# ---------------------------------------------------------------------------
# Parabolic SAR
# ---------------------------------------------------------------------------


def parabolic_sar(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> Tuple[pd.Series, pd.Series]:
    """Parabolic SAR (Wilder).

    Returns
    -------
    (sar, direction) : tuple of pandas.Series
        ``direction``: +1 means uptrend (SAR below price), -1 means
        downtrend (SAR above price).

    Example
    -------
    >>> sar_v, sar_dir = ind.parabolic_sar(df)
    >>> long_signal = (sar_dir == 1) & (df["close"] > sar_v)
    """
    df = ensure_df(df, required=("high", "low", "close"))
    n = len(df)
    if n < 2:
        return (
            pd.Series(np.nan, index=df.index),
            pd.Series(0, index=df.index, dtype=int),
        )

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    sar = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    # Initialise: assume uptrend if first bar's close > prev bar's close.
    direction[0] = 1 if close[1] >= close[0] else -1
    if direction[0] == 1:
        sar[0] = low[0]
        ep = high[0]
    else:
        sar[0] = high[0]
        ep = low[0]
    af = af_start

    for i in range(1, n):
        prev_dir = direction[i - 1]
        prev_sar = sar[i - 1]
        # Tentative SAR for this bar
        cand = prev_sar + af * (ep - prev_sar)

        if prev_dir == 1:
            # Uptrend: SAR cannot exceed the previous two bars' lows
            cand = min(cand, low[i - 1], low[max(i - 2, 0)])
            if low[i] < cand:
                # Trend reversal to downtrend
                direction[i] = -1
                sar[i] = ep  # SAR snaps to prior EP
                ep = low[i]
                af = af_start
            else:
                direction[i] = 1
                sar[i] = cand
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            # Downtrend
            cand = max(cand, high[i - 1], high[max(i - 2, 0)])
            if high[i] > cand:
                # Trend reversal to uptrend
                direction[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                direction[i] = -1
                sar[i] = cand
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return (
        pd.Series(sar, index=df.index),
        pd.Series(direction, index=df.index),
    )


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

#: Alias for :func:`bollinger` — many LLM-generated scripts call it
#: ``bollinger_bands`` (the more descriptive name).
bollinger_bands = bollinger


def __getattr__(name: str):
    """Lazy re-export of pattern helpers commonly mis-attributed to indicators."""
    if name == "candle_lower_shadow":
        from .patterns import candle_lower_shadow as _impl
        return _impl
    if name == "candle_upper_shadow":
        from .patterns import candle_upper_shadow as _impl
        return _impl
    raise AttributeError(f"module 'cyqnt_trd.blocks.indicators' has no attribute {name!r}")
