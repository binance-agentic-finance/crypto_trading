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
    crossover,        # ADD THIS
    crossunder,       # ADD THIS
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
    # new: TradingView-style indicators (path A — hand-written, no extra deps)
    "vwma",
    "hma",
    "mfi",
    "cci",
    "williams_r",
    "keltner",
    "heikin_ashi",
    "cmf",
    # mid-priority TradingView indicators (path A)
    "tema",
    "dema",
    "aroon",
    "trix",
    "awesome_oscillator",
    "pivot_points",
    "zigzag",
    "pvt",
    # new: ported from atomic_strategy_lib
    "stochrsi",
    "rsi_zone",
    "atr_ratio",
    "bb_bandwidth",
    "bb_pct_b",
    "bb_squeeze",
    "dual_speed_atr",
    "volume_surge_ratio",
    "volume_trend",
    "ema_cross_signal",
    "trend_strength",
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


# ---------------------------------------------------------------------------
# Stochastic RSI
# ---------------------------------------------------------------------------

def stochrsi(
    series: SeriesLike,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic RSI — momentum oscillator combining RSI and Stochastic.

    Returns
    -------
    (k, d) : tuple of pandas.Series
        ``%K`` and ``%D`` values in ``[0, 100]``.
    """
    rsi_s = rsi(series, rsi_period)
    rsi_min = rsi_s.rolling(window=stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi_s.rolling(window=stoch_period, min_periods=stoch_period).max()
    stoch = safe_divide(rsi_s - rsi_min, rsi_max - rsi_min, fill=0.5) * 100.0
    k = sma(stoch, k_period)
    d = sma(k, d_period)
    return k, d


# ---------------------------------------------------------------------------
# RSI Zone classifier
# ---------------------------------------------------------------------------

def rsi_zone(
    rsi_series: SeriesLike,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> pd.Series:
    """Classify RSI into ``"OVERBOUGHT"`` / ``"OVERSOLD"`` / ``"NEUTRAL"`` at each bar.

    Vectorised equivalent of :func:`atomic.signals.momentum.rsi_zone_detect`.
    """
    r = ensure_series(rsi_series).astype(float)
    out = pd.Series("NEUTRAL", index=r.index)
    out = out.where(~(r >= overbought), "OVERBOUGHT")
    out = out.where(~(r <= oversold), "OVERSOLD")
    return out


# ---------------------------------------------------------------------------
# ATR ratio
# ---------------------------------------------------------------------------

def atr_ratio(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as a percentage of the current close price.

    Equivalent of :func:`atomic.signals.volatility.atr_ratio`.
    """
    period = positive_int(period, "period")
    df = ensure_df(df, required=("high", "low", "close"))
    atr_s = atr(df, period)
    return safe_divide(atr_s, df["close"].astype(float), fill=0.0) * 100.0


# ---------------------------------------------------------------------------
# Bollinger Band bandwidth / %B / squeeze
# ---------------------------------------------------------------------------

def bb_bandwidth(
    series: SeriesLike, period: int = 20, std_mult: float = 2.0
) -> pd.Series:
    """Bollinger Band bandwidth: ``(upper - lower) / middle * 100``."""
    period = positive_int(period, "period")
    upper, mid, lower = bollinger(series, period, std_mult)
    return safe_divide(upper - lower, mid, fill=0.0) * 100.0


def bb_pct_b(
    series: SeriesLike, period: int = 20, std_mult: float = 2.0
) -> pd.Series:
    r"""Bollinger ``%B``: position of price within the band, ``(price - lower) / (upper - lower)``.

    0 = at lower band, 1 = at upper band, 0.5 = at middle.
    """
    period = positive_int(period, "period")
    s = ensure_series(series).astype(float)
    upper, mid, lower = bollinger(s, period, std_mult)
    return safe_divide(s - lower, upper - lower, fill=0.5)


def bb_squeeze(
    series: SeriesLike,
    period: int = 20,
    std_mult: float = 2.0,
    lookback: int = 20,
    squeeze_threshold: float = 2.0,
) -> pd.Series:
    """Bollinger Band squeeze: True when bandwidth is historically compressed.

    Squeeze is active when *current_bw <= squeeze_threshold* OR
    *current_bw / avg_bw_lookback < 0.5*.

    Vectorised equivalent of :func:`atomic.signals.volatility.bb_squeeze_detect`.
    """
    bw = bb_bandwidth(series, period, std_mult)
    avg_bw = bw.rolling(window=lookback, min_periods=lookback).mean()
    ratio = safe_divide(bw, avg_bw, fill=1.0)
    return ((bw <= squeeze_threshold) | (ratio < 0.5)).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Dual-speed ATR (volatility expansion/contraction)
# ---------------------------------------------------------------------------

def dual_speed_atr(
    df: pd.DataFrame,
    fast_period: int = 7,
    slow_period: int = 21,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compare fast vs slow ATR to detect volatility expansion / contraction.

    Returns
    -------
    (fast_atr, slow_atr, ratio, expanding) : tuple of pandas.Series
        ``ratio = fast_atr / slow_atr``.  ``expanding`` is True when ratio > 1.
    """
    fast_period = positive_int(fast_period, "fast_period")
    slow_period = positive_int(slow_period, "slow_period")
    fast_s = atr(df, fast_period)
    slow_s = atr(df, slow_period)
    ratio = safe_divide(fast_s, slow_s, fill=1.0)
    expanding = (ratio > 1.0).fillna(False).astype(bool)
    return fast_s, slow_s, ratio, expanding


# ---------------------------------------------------------------------------
# Volume surge ratio
# ---------------------------------------------------------------------------

def volume_surge_ratio(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Volume ratio: latest bar volume / rolling average of prior *lookback* bars.

    Values > 1 mean above-average volume; ``surge_threshold`` (typically 2.0)
    separates normal from surge.  Vectorised equivalent of
    :func:`atomic.signals.volume.volume_surge_detect`.
    """
    lookback = positive_int(lookback, "lookback")
    df = ensure_df(df, required=("volume",))
    vol = df["volume"].astype(float)
    avg = vol.shift(1).rolling(window=lookback, min_periods=lookback).mean()
    return safe_divide(vol, avg, fill=1.0)


# ---------------------------------------------------------------------------
# Volume trend (accumulation / distribution)
# ---------------------------------------------------------------------------

def volume_trend(
    df: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
) -> pd.Series:
    """Classify volume trend into ``"ACCUMULATING"`` / ``"DISTRIBUTING"`` / ``"DRYING_UP"`` / ``"NEUTRAL"``.

    Logic mirrors :func:`atomic.signals.volume.volume_trend`:

    * ``vol_ratio > 1.2 and buy_ratio > 0.6``  → ACCUMULATING
    * ``vol_ratio > 1.2 and buy_ratio < 0.4``  → DISTRIBUTING
    * ``vol_ratio < 0.5``                       → DRYING_UP
    * otherwise                                 → NEUTRAL

    *buy_ratio* is the fraction of volume on bullish bars (green candles)
    within the short window.
    """
    short_window = positive_int(short_window, "short_window")
    long_window = positive_int(long_window, "long_window")
    df = ensure_df(df, required=("open", "close", "volume"))
    vol = df["volume"].astype(float)
    is_green = (df["close"] >= df["open"]).astype(float)

    short_avg = vol.rolling(window=short_window, min_periods=short_window).mean()
    long_avg = vol.rolling(window=long_window, min_periods=long_window).mean()

    buy_vol = (vol * is_green).rolling(window=short_window, min_periods=short_window).sum()
    total_vol = vol.rolling(window=short_window, min_periods=short_window).sum()
    buy_ratio = safe_divide(buy_vol, total_vol, fill=0.5)

    vol_ratio = safe_divide(short_avg, long_avg, fill=1.0)

    out = pd.Series("NEUTRAL", index=df.index)
    # Apply in priority order to avoid overwriting with lower-priority rule
    drying = vol_ratio < 0.5
    distributing = (vol_ratio > 1.2) & (buy_ratio < 0.4) & ~drying
    accumulating = (vol_ratio > 1.2) & (buy_ratio > 0.6) & ~drying & ~distributing
    out = out.where(~drying, "DRYING_UP")
    out = out.where(~distributing, "DISTRIBUTING")
    out = out.where(~accumulating, "ACCUMULATING")
    return out


# ---------------------------------------------------------------------------
# EMA cross signal (GOLDEN / DEATH / NONE)
# ---------------------------------------------------------------------------

def ema_cross_signal(
    series: SeriesLike,
    fast_period: int = 9,
    slow_period: int = 21,
) -> pd.Series:
    """Vectorised EMA cross detector.

    Returns a string Series:

    * ``"GOLDEN"`` on the bar where fast EMA crosses above slow EMA
    * ``"DEATH"``  on the bar where fast EMA crosses below slow EMA
    * ``"NONE"``   otherwise

    Vectorised equivalent of :func:`atomic.signals.trend.ema_cross_detect`.
    """
    fast_period = positive_int(fast_period, "fast_period")
    slow_period = positive_int(slow_period, "slow_period")
    fast_s = ema(series, fast_period)
    slow_s = ema(series, slow_period)
    out = pd.Series("NONE", index=fast_s.index)
    out = out.where(~crossover(fast_s, slow_s), "GOLDEN")
    out = out.where(~crossunder(fast_s, slow_s), "DEATH")
    return out


# ---------------------------------------------------------------------------
# Trend strength composite
# ---------------------------------------------------------------------------

def trend_strength(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Composite trend strength: 0.0 (choppy) → 1.0 (strong directional move).

    Combines rolling green-candle ratio (60%) and average body-to-range
    fraction normalised by 80% (40%).  Mirrors
    :func:`atomic.signals.trend.trend_strength`.
    """
    lookback = positive_int(lookback, "lookback")
    df = ensure_df(df, required=("open", "high", "low", "close"))
    is_green = (df["close"] >= df["open"]).astype(float)
    body = (df["close"] - df["open"]).abs()
    hl_range = df["high"] - df["low"]
    body_pct = safe_divide(body, hl_range, fill=0.0) * 100.0

    green_ratio = is_green.rolling(window=lookback, min_periods=lookback).mean()
    avg_body = body_pct.rolling(window=lookback, min_periods=lookback).mean()

    return (green_ratio * 0.6 + (avg_body / 80.0).clip(upper=1.0) * 0.4).rename(
        "trend_strength"
    )


# ---------------------------------------------------------------------------
# TradingView-style indicators (hand-written, pandas/numpy)
# ---------------------------------------------------------------------------


def vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume-weighted moving average — Σ(price × volume) / Σ(volume).

    Equivalent to TradingView's ``ta.vwma``. Common use: weight a moving
    average toward bars with higher participation. Less smooth than a
    plain SMA on low-volume bars and reacts faster on high-volume ones.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``close`` and ``volume``.
    period : int, default 20
        Window length.

    Returns
    -------
    pd.Series
        VWMA, NaN for first ``period - 1`` bars.
    """
    period = positive_int(period, "period")
    df = ensure_df(df, required=("close", "volume"))
    pv = df["close"].astype(float) * df["volume"].astype(float)
    return safe_divide(
        pv.rolling(window=period, min_periods=period).sum(),
        df["volume"].rolling(window=period, min_periods=period).sum(),
        fill=float("nan"),
    )


def hma(series: SeriesLike, period: int = 20) -> pd.Series:
    """Hull Moving Average — WMA(2*WMA(n/2) − WMA(n), sqrt(n)).

    Equivalent to TradingView's ``ta.hma``. Designed by Alan Hull to
    reduce lag of a traditional MA while keeping smoothness.

    Parameters
    ----------
    series : pd.Series or DataFrame column
    period : int, default 20

    Returns
    -------
    pd.Series
        HMA values; NaN until ``period + sqrt(period) - 2`` bars are
        available.
    """
    period = positive_int(period, "period")
    half = max(1, period // 2)
    sqrt_n = max(1, int(round(period ** 0.5)))
    s = ensure_series(series).astype(float)
    raw = 2.0 * wma(s, half) - wma(s, period)
    return wma(raw, sqrt_n)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — RSI computed on typical price × volume.

    Equivalent to TradingView's ``ta.mfi``. Reads 0-100; >80 typically
    signals overbought, <20 oversold. Distinguishes accumulation vs
    distribution by incorporating volume.

    Formula:
        TP        = (high + low + close) / 3
        RawFlow   = TP × volume
        PosFlow   = sum(RawFlow where TP rose) over `period`
        NegFlow   = sum(RawFlow where TP fell) over `period`
        MFI       = 100 - 100 / (1 + PosFlow/NegFlow)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``, ``volume``.
    period : int, default 14

    Returns
    -------
    pd.Series
        MFI in [0, 100]; NaN for the first ``period`` bars.
    """
    period = positive_int(period, "period")
    df = ensure_df(df, required=("high", "low", "close", "volume"))
    tp = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    raw_flow = tp * df["volume"].astype(float)
    delta = tp.diff()

    pos_flow = raw_flow.where(delta > 0, 0.0)
    neg_flow = raw_flow.where(delta < 0, 0.0)
    pos_sum = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(window=period, min_periods=period).sum()

    money_ratio = safe_divide(pos_sum, neg_sum, fill=float("inf"))
    mfi_series = 100.0 - 100.0 / (1.0 + money_ratio)
    # Where neg_sum was zero (all positive), MFI saturates at 100
    return mfi_series.where(neg_sum > 0, 100.0).where(pos_sum + neg_sum > 0, float("nan"))


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index — Lambert (1980).

    Equivalent to TradingView's ``ta.cci``. Measures how far the
    typical price is from its moving average in units of mean deviation.
    Often capped at ±100 for "overbought/oversold" signal generation.

    Formula:
        TP        = (high + low + close) / 3
        SMA_TP    = SMA(TP, period)
        MeanDev   = mean(|TP − SMA_TP|) over `period`
        CCI       = (TP − SMA_TP) / (0.015 × MeanDev)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.
    period : int, default 20

    Returns
    -------
    pd.Series
        CCI; NaN for the first ``period - 1`` bars.
    """
    period = positive_int(period, "period")
    df = ensure_df(df, required=("high", "low", "close"))
    tp = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    sma_tp = sma(tp, period)
    mean_dev = (tp - sma_tp).abs().rolling(window=period, min_periods=period).mean()
    return safe_divide(tp - sma_tp, 0.015 * mean_dev, fill=float("nan"))


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R — momentum oscillator from highest-high range.

    Equivalent to TradingView's ``ta.wpr``. Outputs values in [-100, 0]:
        ≥ -20 → overbought
        ≤ -80 → oversold

    Formula:
        Highest = max(high, period)
        Lowest  = min(low, period)
        %R      = (Highest − Close) / (Highest − Lowest) × -100

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.
    period : int, default 14

    Returns
    -------
    pd.Series
        Williams %R in [-100, 0]; NaN for the first ``period - 1`` bars.
    """
    period = positive_int(period, "period")
    df = ensure_df(df, required=("high", "low", "close"))
    hh = rolling_max(df["high"].astype(float), period)
    ll = rolling_min(df["low"].astype(float), period)
    return safe_divide(hh - df["close"].astype(float), hh - ll, fill=float("nan")) * -100.0


def keltner(
    df: pd.DataFrame,
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channel — EMA-centred bands at ``±multiplier × ATR``.

    Equivalent to TradingView's ``ta.keltner`` (with the EMA midline
    convention; some old references use SMA). Wider/narrower than
    Bollinger because volatility is measured by ATR (range-based)
    rather than std (close-based).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.
    period : int, default 20
        EMA period for the midline.
    atr_period : int, default 10
        ATR period.
    multiplier : float, default 2.0
        Band width factor.

    Returns
    -------
    (upper, middle, lower) : tuple of pd.Series
    """
    period = positive_int(period, "period")
    atr_period = positive_int(atr_period, "atr_period")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be positive, got {multiplier}")
    df = ensure_df(df, required=("high", "low", "close"))
    middle = ema(df["close"], period)
    atr_series = atr(df, period=atr_period)
    upper = middle + multiplier * atr_series
    lower = middle - multiplier * atr_series
    return upper, middle, lower


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin Ashi candle transformation.

    Equivalent to TradingView's ``ta.heikinashi`` view (or built-in
    Heikin Ashi candle type). Smooths price action by averaging OHLC
    against the prior smoothed open. Helps to identify clean trends
    but lags reversals by 1-2 bars.

    Formula:
        HA_Close = (O + H + L + C) / 4
        HA_Open  = (prev_HA_Open + prev_HA_Close) / 2  (seed: (O[0]+C[0])/2)
        HA_High  = max(H, HA_Open, HA_Close)
        HA_Low   = min(L, HA_Open, HA_Close)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``open``, ``high``, ``low``, ``close``.

    Returns
    -------
    pd.DataFrame
        Columns: ``ha_open``, ``ha_high``, ``ha_low``, ``ha_close``.
    """
    df = ensure_df(df, required=("open", "high", "low", "close"))
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    low = df["low"].astype(float)
    c = df["close"].astype(float)

    ha_close = (o + h + low + c) / 4.0

    # ha_open needs recurrence — fill iteratively (numpy loop, ~250 bars trivial)
    ha_open_arr = np.empty(len(df), dtype=float)
    ha_open_arr[0] = (o.iloc[0] + c.iloc[0]) / 2.0
    ha_close_arr = ha_close.to_numpy()
    for i in range(1, len(df)):
        ha_open_arr[i] = (ha_open_arr[i - 1] + ha_close_arr[i - 1]) / 2.0
    ha_open = pd.Series(ha_open_arr, index=df.index)

    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)

    return pd.DataFrame(
        {
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
            "ha_close": ha_close,
        }
    )


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow — accumulation/distribution over `period`.

    Equivalent to TradingView's ``ta.cmf``. Reads in [-1, 1]:
        > 0 → net accumulation (buyers in control)
        < 0 → net distribution (sellers in control)

    Formula:
        MFM     = ((C − L) − (H − C)) / (H − L)         # close position in range
        MFV     = MFM × volume
        CMF     = sum(MFV, period) / sum(volume, period)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``, ``volume``.
    period : int, default 20

    Returns
    -------
    pd.Series
        CMF in [-1, 1]; NaN for the first ``period - 1`` bars.
    """
    period = positive_int(period, "period")
    df = ensure_df(df, required=("high", "low", "close", "volume"))
    h = df["high"].astype(float)
    low = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    hl_range = h - low
    mfm = safe_divide((c - low) - (h - c), hl_range, fill=0.0)
    mfv = mfm * v
    return safe_divide(
        mfv.rolling(window=period, min_periods=period).sum(),
        v.rolling(window=period, min_periods=period).sum(),
        fill=0.0,
    )


# ---------------------------------------------------------------------------
# TradingView-style indicators — mid-priority batch
# ---------------------------------------------------------------------------


def tema(series: SeriesLike, period: int = 20) -> pd.Series:
    """Triple Exponential Moving Average.

    Equivalent to TradingView's ``ta.tema``. Designed by Patrick Mulloy
    to reduce EMA lag while keeping smoothness, with stronger reactivity
    than DEMA.

    Formula:
        EMA1 = EMA(close, n)
        EMA2 = EMA(EMA1, n)
        EMA3 = EMA(EMA2, n)
        TEMA = 3*EMA1 − 3*EMA2 + EMA3

    Parameters
    ----------
    series : pd.Series or DataFrame column
    period : int, default 20

    Returns
    -------
    pd.Series
    """
    period = positive_int(period, "period")
    s = ensure_series(series).astype(float)
    e1 = ema(s, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3.0 * e1 - 3.0 * e2 + e3


def dema(series: SeriesLike, period: int = 20) -> pd.Series:
    """Double Exponential Moving Average.

    Equivalent to TradingView's ``ta.dema``. Less lag than EMA but more
    responsive than TEMA. Computed as ``2*EMA1 - EMA(EMA1)``.

    Parameters
    ----------
    series : pd.Series or DataFrame column
    period : int, default 20

    Returns
    -------
    pd.Series
    """
    period = positive_int(period, "period")
    s = ensure_series(series).astype(float)
    e1 = ema(s, period)
    e2 = ema(e1, period)
    return 2.0 * e1 - e2


def aroon(
    df: pd.DataFrame, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Aroon Up, Aroon Down, and Aroon Oscillator (Tushar Chande, 1995).

    Equivalent to TradingView's ``ta.aroon``. Measures how recently the
    highest high (Aroon Up) or lowest low (Aroon Down) occurred within
    the lookback window. Each ranges in [0, 100]; the oscillator
    (= up − down) ranges in [-100, +100].

    Interpretation:
        Aroon Up   ≥ 70 → strong uptrend
        Aroon Down ≥ 70 → strong downtrend
        Aroon Osc  > 0 → bullish bias, < 0 → bearish bias

    Formula:
        Aroon Up   = 100 * (period - bars_since_highest_high) / period
        Aroon Down = 100 * (period - bars_since_lowest_low)  / period
        Aroon Osc  = Aroon Up - Aroon Down

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high`` and ``low``.
    period : int, default 14

    Returns
    -------
    (aroon_up, aroon_down, aroon_osc) : tuple of pd.Series
    """
    period = positive_int(period, "period")
    df = ensure_df(df, required=("high", "low"))
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # Argmax / argmin position within rolling window (counting from window start)
    def _bars_since_extreme(series: pd.Series, mode: str) -> pd.Series:
        # rolling apply: position of max/min within window, 0..period
        if mode == "high":
            pos = series.rolling(window=period + 1, min_periods=period + 1).apply(
                lambda x: float(np.argmax(x[::-1])), raw=True
            )
        else:  # low
            pos = series.rolling(window=period + 1, min_periods=period + 1).apply(
                lambda x: float(np.argmin(x[::-1])), raw=True
            )
        return pos

    bars_since_high = _bars_since_extreme(high, "high")
    bars_since_low = _bars_since_extreme(low, "low")
    aroon_up = 100.0 * (period - bars_since_high) / period
    aroon_down = 100.0 * (period - bars_since_low) / period
    aroon_osc = aroon_up - aroon_down
    return aroon_up, aroon_down, aroon_osc


def trix(series: SeriesLike, period: int = 14) -> pd.Series:
    """TRIX — triple-smoothed exponential rate of change (Jack Hutson).

    Equivalent to TradingView's ``ta.trix``. A triple-EMA-smoothed
    momentum oscillator. Good for filtering out short-term noise from
    rate-of-change signals. Sign change at zero line is the canonical
    entry/exit cue.

    Formula:
        EMA3 = EMA(EMA(EMA(close, n), n), n)
        TRIX = (EMA3 − EMA3.shift(1)) / EMA3.shift(1) × 10000   (basis points)

    Parameters
    ----------
    series : pd.Series or DataFrame column
    period : int, default 14

    Returns
    -------
    pd.Series
        TRIX in basis points (1.0 = 0.01% per bar).
    """
    period = positive_int(period, "period")
    s = ensure_series(series).astype(float)
    e1 = ema(s, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return safe_divide(e3 - e3.shift(1), e3.shift(1), fill=0.0) * 10000.0


def awesome_oscillator(df: pd.DataFrame) -> pd.Series:
    """Awesome Oscillator (Bill Williams).

    Equivalent to TradingView's ``ta.ao``. Measures the difference
    between a 5-period and 34-period SMA of the median price (H+L)/2.
    Sign reveals momentum direction; histogram colour change marks
    momentum shifts.

    Formula:
        median = (high + low) / 2
        AO     = SMA(median, 5) − SMA(median, 34)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high`` and ``low``.

    Returns
    -------
    pd.Series
    """
    df = ensure_df(df, required=("high", "low"))
    median = (df["high"].astype(float) + df["low"].astype(float)) / 2.0
    return sma(median, 5) - sma(median, 34)


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Standard (Floor) Pivot Points + R1/R2/R3 + S1/S2/S3.

    Equivalent to TradingView's ``ta.pivot_point_levels`` (Standard /
    Floor variant). Uses the *previous* bar's high, low, close to
    project today's pivot levels. Works at any timeframe; for "daily
    pivots" pass a daily-resampled DataFrame.

    Formula (Standard):
        PP = (prev_high + prev_low + prev_close) / 3
        R1 = 2 × PP - prev_low      ;  S1 = 2 × PP - prev_high
        R2 = PP + (prev_high - prev_low) ; S2 = PP - (prev_high - prev_low)
        R3 = prev_high + 2 × (PP - prev_low)
        S3 = prev_low  - 2 × (prev_high - PP)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.

    Returns
    -------
    pd.DataFrame
        Columns: pp, r1, r2, r3, s1, s2, s3. First row is NaN
        (no previous bar available).
    """
    df = ensure_df(df, required=("high", "low", "close"))
    ph = df["high"].astype(float).shift(1)
    pl = df["low"].astype(float).shift(1)
    pc = df["close"].astype(float).shift(1)
    pp = (ph + pl + pc) / 3.0
    range_ = ph - pl
    r1 = 2.0 * pp - pl
    s1 = 2.0 * pp - ph
    r2 = pp + range_
    s2 = pp - range_
    r3 = ph + 2.0 * (pp - pl)
    s3 = pl - 2.0 * (ph - pp)
    return pd.DataFrame(
        {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}
    )


def zigzag(series: SeriesLike, deviation_pct: float = 5.0) -> pd.Series:
    """ZigZag pivot detector by percentage deviation.

    Approximates TradingView's built-in ``ZigZag`` indicator. Marks
    confirmed swing pivots — the previous trend extreme is locked in
    only after price reverses by ``deviation_pct`` from it.

    .. note::

        Like TradingView's reference implementation, the *most-recent*
        pivot is **tentative** and may shift forward as new bars arrive
        (it represents "the current swing leg"). Pivots before the
        last reversal are stable. Use only completed (non-final) pivots
        for backtesting to remain look-ahead-safe.

    Parameters
    ----------
    series : pd.Series or DataFrame column
        Typically ``close`` or ``(high+low)/2``.
    deviation_pct : float, default 5.0
        Percentage move required to confirm a reversal.

    Returns
    -------
    pd.Series
        Pivot prices at confirmed/tentative pivot bars; NaN otherwise.
    """
    if deviation_pct <= 0:
        raise ValueError(f"deviation_pct must be positive, got {deviation_pct}")
    s = ensure_series(series).astype(float).reset_index(drop=True)
    n = len(s)
    pivots = pd.Series(np.nan, index=s.index)
    if n < 2:
        return pivots

    last_pivot_idx = 0
    last_pivot_price = s.iloc[0]
    pivots.iloc[0] = last_pivot_price
    direction = 0  # 0 = unknown, +1 = up, -1 = down

    for i in range(1, n):
        price = s.iloc[i]
        change_pct = (price - last_pivot_price) / last_pivot_price * 100.0

        if direction == 0:
            if abs(change_pct) >= deviation_pct:
                pivots.iloc[i] = price
                last_pivot_idx = i
                last_pivot_price = price
                direction = 1 if change_pct > 0 else -1
        elif direction == 1:  # uptrend
            if price > last_pivot_price:
                pivots.iloc[last_pivot_idx] = np.nan
                pivots.iloc[i] = price
                last_pivot_idx = i
                last_pivot_price = price
            elif change_pct <= -deviation_pct:
                pivots.iloc[i] = price
                last_pivot_idx = i
                last_pivot_price = price
                direction = -1
        else:  # direction == -1, downtrend
            if price < last_pivot_price:
                pivots.iloc[last_pivot_idx] = np.nan
                pivots.iloc[i] = price
                last_pivot_idx = i
                last_pivot_price = price
            elif change_pct >= deviation_pct:
                pivots.iloc[i] = price
                last_pivot_idx = i
                last_pivot_price = price
                direction = 1

    # Restore original index
    pivots.index = ensure_series(series).index
    return pivots


def pvt(df: pd.DataFrame) -> pd.Series:
    """Price-Volume Trend (PVT, Joseph Granville's modification of OBV).

    Equivalent to TradingView's ``ta.pvt``. Cumulative volume weighted
    by relative price change. Less binary than OBV — captures the
    *magnitude* of the move, not just direction. Divergences from price
    often signal reversal.

    Formula:
        PVT[i] = PVT[i-1] + ((close[i] − close[i-1]) / close[i-1]) × volume[i]

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``close`` and ``volume``.

    Returns
    -------
    pd.Series
        Cumulative PVT. First bar = 0 (seed).
    """
    df = ensure_df(df, required=("close", "volume"))
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    pct_change = close.pct_change().fillna(0.0)
    return (pct_change * volume).cumsum()


def __getattr__(name: str):
    """Lazy re-export of pattern helpers commonly mis-attributed to indicators."""
    if name == "candle_lower_shadow":
        from .patterns import candle_lower_shadow as _impl
        return _impl
    if name == "candle_upper_shadow":
        from .patterns import candle_upper_shadow as _impl
        return _impl
    raise AttributeError(f"module 'cyqnt_trd.blocks.indicators' has no attribute {name!r}")
