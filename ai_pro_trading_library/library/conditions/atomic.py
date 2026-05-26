"""Look-ahead-safe atomic conditions for strategy composition.

Source attribution: rewritten from `cyqnt_trd.blocks.conditions` into a
library-first module.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def cross_above(left: pd.Series, right: pd.Series) -> pd.Series:
    return ((left > right) & (left.shift(1) <= right.shift(1))).fillna(False)


def cross_below(left: pd.Series, right: pd.Series) -> pd.Series:
    return ((left < right) & (left.shift(1) >= right.shift(1))).fillna(False)


def price_above(close: pd.Series, reference: pd.Series | float) -> pd.Series:
    return (close > reference).fillna(False)


def price_below(close: pd.Series, reference: pd.Series | float) -> pd.Series:
    return (close < reference).fillna(False)


def rsi_in_range(values: pd.Series, lower: float, upper: float) -> pd.Series:
    if lower > upper:
        raise ValueError("lower must be <= upper")
    return values.between(float(lower), float(upper), inclusive="both").fillna(False)


def breakout_high(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    lookback = _positive_int(lookback, "lookback")
    required = {"close", "high"}
    if not required.issubset(df.columns):
        raise ValueError(f"dataframe is missing required columns: {sorted(required - set(df.columns))}")
    prior_high = df["high"].shift(1).rolling(window=lookback, min_periods=lookback).max()
    return (df["close"] > prior_high).fillna(False)


def breakout_low(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    lookback = _positive_int(lookback, "lookback")
    required = {"close", "low"}
    if not required.issubset(df.columns):
        raise ValueError(f"dataframe is missing required columns: {sorted(required - set(df.columns))}")
    prior_low = df["low"].shift(1).rolling(window=lookback, min_periods=lookback).min()
    return (df["close"] < prior_low).fillna(False)


def volume_surge(volume: pd.Series, reference_volume: pd.Series, multiplier: float = 1.5) -> pd.Series:
    return (volume >= float(multiplier) * reference_volume).fillna(False)


# ---------------------------------------------------------------------------
# Bounce / retest / consolidation
# ---------------------------------------------------------------------------


def price_bounce_ma(df: pd.DataFrame, ma: pd.Series, direction: str = "long") -> pd.Series:
    """Touch-and-reject of MA: price wicked beyond MA but closed back through it."""
    required = {"low", "high", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    if direction == "long":
        return ((df["low"] <= ma) & (df["close"] > ma)).fillna(False)
    if direction == "short":
        return ((df["high"] >= ma) & (df["close"] < ma)).fillna(False)
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def retest_after_breakout(
    df: pd.DataFrame,
    lookback: int = 20,
    retest_window: int = 5,
) -> pd.Series:
    """Retest of breakout level within `retest_window` bars."""
    required = {"close", "high", "low"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    lookback = _positive_int(lookback, "lookback")
    retest_window = _positive_int(retest_window, "retest_window")

    breakout = breakout_high(df, lookback)
    breakout_level = df["high"].shift(1).rolling(window=lookback, min_periods=lookback).max()
    breakout_level_at_breakout = breakout_level.where(breakout).ffill(limit=retest_window)
    bars_since = breakout.cumsum()
    bars_since_at_break = bars_since.where(breakout).ffill(limit=retest_window)
    is_within_window = (bars_since - bars_since_at_break) <= retest_window
    retest = (
        is_within_window
        & (df["low"] <= breakout_level_at_breakout * 1.005)
        & (df["close"] >= breakout_level_at_breakout)
    )
    return retest.fillna(False)


def consolidation_range(
    df: pd.DataFrame,
    period: int = 20,
    max_range_pct: float = 0.03,
) -> pd.Series:
    """(high-low)/mid over `period` bars below `max_range_pct`."""
    required = {"high", "low"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    period = _positive_int(period, "period")
    hh = df["high"].rolling(window=period, min_periods=period).max()
    ll = df["low"].rolling(window=period, min_periods=period).min()
    rng_pct = (hh - ll) / ((hh + ll) / 2.0)
    return (rng_pct <= float(max_range_pct)).fillna(False).astype(bool)


def range_detection(df: pd.DataFrame, period: int = 20, max_range_pct: float = 0.03) -> pd.Series:
    """Alias for `consolidation_range`."""
    return consolidation_range(df, period=period, max_range_pct=max_range_pct)


def volume_shrink(
    volume: pd.Series,
    reference_volume: pd.Series,
    bars: int = 3,
    multiplier: float = 1.0,
) -> pd.Series:
    """All of the last `bars` bars have volume < multiplier * reference."""
    bars = _positive_int(bars, "bars")
    cond = volume < float(multiplier) * reference_volume
    return cond.rolling(window=bars, min_periods=bars).sum().eq(bars).fillna(False)


# ---------------------------------------------------------------------------
# MACD conditions
# ---------------------------------------------------------------------------


def macd_golden_cross(macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
    return cross_above(macd_line, signal_line)


def macd_death_cross(macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
    return cross_below(macd_line, signal_line)


def macd_above_zero(macd_line: pd.Series) -> pd.Series:
    return (macd_line > 0).fillna(False).astype(bool)


def macd_below_zero(macd_line: pd.Series) -> pd.Series:
    return (macd_line < 0).fillna(False).astype(bool)


def macd_bullish_divergence(
    price: pd.Series,
    macd_line: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Price LL while MACD HL over `lookback`."""
    lookback = _positive_int(lookback, "lookback")
    p_low = price.rolling(window=lookback, min_periods=lookback).min()
    m_low = macd_line.rolling(window=lookback, min_periods=lookback).min()
    return ((price == p_low) & (macd_line > m_low)).fillna(False)


def macd_bearish_divergence(
    price: pd.Series,
    macd_line: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Price HH while MACD LH over `lookback`."""
    lookback = _positive_int(lookback, "lookback")
    p_high = price.rolling(window=lookback, min_periods=lookback).max()
    m_high = macd_line.rolling(window=lookback, min_periods=lookback).max()
    return ((price == p_high) & (macd_line < m_high)).fillna(False)


# ---------------------------------------------------------------------------
# RSI thresholds
# ---------------------------------------------------------------------------


def rsi_overbought(rsi_series: pd.Series, threshold: float = 70.0) -> pd.Series:
    return (rsi_series >= float(threshold)).fillna(False).astype(bool)


def rsi_oversold(rsi_series: pd.Series, threshold: float = 30.0) -> pd.Series:
    return (rsi_series <= float(threshold)).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# ADX conditions
# ---------------------------------------------------------------------------


def adx_trending(adx_series: pd.Series, threshold: float = 25.0) -> pd.Series:
    return (adx_series >= float(threshold)).fillna(False).astype(bool)


def adx_ranging(adx_series: pd.Series, threshold: float = 20.0) -> pd.Series:
    return (adx_series < float(threshold)).fillna(False).astype(bool)


def adx_direction_long(plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    return (plus_di > minus_di).fillna(False).astype(bool)


def adx_direction_short(plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    return (minus_di > plus_di).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Price-vs-MA (df + bars window form)
# ---------------------------------------------------------------------------


def price_above_ma(df: pd.DataFrame, ma: pd.Series, bars: int = 1) -> pd.Series:
    """Close has been above `ma` for the last `bars` bars."""
    if "close" not in df.columns:
        raise ValueError("dataframe missing 'close' column")
    bars = _positive_int(bars, "bars")
    above = df["close"] > ma
    return above.rolling(window=bars, min_periods=bars).sum().eq(bars).fillna(False)


def price_below_ma(df: pd.DataFrame, ma: pd.Series, bars: int = 1) -> pd.Series:
    """Close has been below `ma` for the last `bars` bars."""
    if "close" not in df.columns:
        raise ValueError("dataframe missing 'close' column")
    bars = _positive_int(bars, "bars")
    below = df["close"] < ma
    return below.rolling(window=bars, min_periods=bars).sum().eq(bars).fillna(False)


def ema_deviation_within(price: pd.Series, ema_series: pd.Series, max_pct: float) -> pd.Series:
    """|price-ema| / ema <= max_pct."""
    deviation = (price - ema_series).abs() / ema_series.replace(0.0, np.nan)
    return (deviation <= float(max_pct)).fillna(False)


# ---------------------------------------------------------------------------
# Bar shape with body-pct override
# ---------------------------------------------------------------------------


def is_bullish_bar(df: pd.DataFrame, min_body_pct: float = 0.0) -> pd.Series:
    """Close > open AND (close - open) / open * 100 >= min_body_pct."""
    if not {"open", "close"}.issubset(df.columns):
        raise ValueError("dataframe missing 'open'/'close' columns")
    body_pct = (df["close"] - df["open"]) / df["open"].replace(0.0, np.nan) * 100.0
    return ((df["close"] > df["open"]) & (body_pct >= float(min_body_pct))).fillna(False)


def is_bearish_bar(df: pd.DataFrame, min_body_pct: float = 0.0) -> pd.Series:
    """Close < open AND (open - close) / open * 100 >= min_body_pct."""
    if not {"open", "close"}.issubset(df.columns):
        raise ValueError("dataframe missing 'open'/'close' columns")
    body_pct = (df["open"] - df["close"]) / df["open"].replace(0.0, np.nan) * 100.0
    return ((df["close"] < df["open"]) & (body_pct >= float(min_body_pct))).fillna(False)


# ---------------------------------------------------------------------------
# Time / funding filters
# ---------------------------------------------------------------------------


def time_filter(
    timestamps_ms: pd.Series,
    start_hour: int,
    end_hour: int,
    tz_offset_hours: int = 8,
) -> pd.Series:
    """Boolean: True when local hour is in [start_hour, end_hour). Default UTC+8."""
    ts = pd.to_datetime(timestamps_ms.astype("int64"), unit="ms", utc=True)
    local = ts + pd.Timedelta(hours=int(tz_offset_hours))
    hour = local.dt.hour
    if start_hour <= end_hour:
        out = (hour >= start_hour) & (hour < end_hour)
    else:
        out = (hour >= start_hour) | (hour < end_hour)
    return out.fillna(False).astype(bool)


def funding_window_safe(
    timestamps_ms: pd.Series,
    settle_hours_utc: Iterable[int] = (0, 8, 16),
    buffer_min: int = 15,
) -> pd.Series:
    """True on bars outside `buffer_min` of funding settlement times."""
    if buffer_min < 0:
        raise ValueError(f"buffer_min must be >= 0, got {buffer_min}")
    ts = pd.to_datetime(timestamps_ms.astype("int64"), unit="ms", utc=True)
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    settle_minutes = {int(h) * 60 for h in settle_hours_utc}
    distance = minute_of_day.apply(
        lambda m: min(min(abs(m - s), 24 * 60 - abs(m - s)) for s in settle_minutes)
    )
    return (distance > buffer_min).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Multi-timeframe / structure
# ---------------------------------------------------------------------------


def multi_timeframe_alignment(*signals: pd.Series) -> pd.Series:
    """All boolean signals True at the same time."""
    if not signals:
        raise ValueError("at least one signal required")
    out = signals[0].fillna(False).astype(bool)
    for s in signals[1:]:
        out = out & s.fillna(False).astype(bool)
    return out


def higher_high(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Current high is rolling max over `lookback` AND > high lookback bars ago."""
    if "high" not in df.columns:
        raise ValueError("dataframe missing 'high' column")
    lookback = _positive_int(lookback, "lookback")
    rolling_h = df["high"].rolling(window=lookback, min_periods=lookback).max()
    return ((df["high"] == rolling_h) & (df["high"] > df["high"].shift(lookback))).fillna(False)


def higher_low(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    if "low" not in df.columns:
        raise ValueError("dataframe missing 'low' column")
    lookback = _positive_int(lookback, "lookback")
    rolling_l = df["low"].rolling(window=lookback, min_periods=lookback).min()
    return ((df["low"] == rolling_l) & (df["low"] > df["low"].shift(lookback))).fillna(False)


def lower_high(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    if "high" not in df.columns:
        raise ValueError("dataframe missing 'high' column")
    lookback = _positive_int(lookback, "lookback")
    rolling_h = df["high"].rolling(window=lookback, min_periods=lookback).max()
    return ((df["high"] == rolling_h) & (df["high"] < df["high"].shift(lookback))).fillna(False)


def lower_low(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    if "low" not in df.columns:
        raise ValueError("dataframe missing 'low' column")
    lookback = _positive_int(lookback, "lookback")
    rolling_l = df["low"].rolling(window=lookback, min_periods=lookback).min()
    return ((df["low"] == rolling_l) & (df["low"] < df["low"].shift(lookback))).fillna(False)


def liquidity_sweep_high(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Wick poke above prior swing high but close back below it."""
    required = {"high", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    lookback = _positive_int(lookback, "lookback")
    prior_high = df["high"].shift(1).rolling(window=lookback, min_periods=lookback).max()
    return ((df["high"] > prior_high) & (df["close"] <= prior_high)).fillna(False)


def liquidity_sweep_low(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Wick poke below prior swing low but close back above it."""
    required = {"low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    lookback = _positive_int(lookback, "lookback")
    prior_low = df["low"].shift(1).rolling(window=lookback, min_periods=lookback).min()
    return ((df["low"] < prior_low) & (df["close"] >= prior_low)).fillna(False)


# ---------------------------------------------------------------------------
# close_above / close_below / price_touch_or_cross
# ---------------------------------------------------------------------------


def close_above(df: pd.DataFrame, ref, bars: int = 1) -> pd.Series:
    """Close has been above `ref` (scalar or series) for the last `bars` bars."""
    if "close" not in df.columns:
        raise ValueError("dataframe missing 'close' column")
    bars = _positive_int(bars, "bars")
    if isinstance(ref, (int, float)):
        ref_s = pd.Series(float(ref), index=df.index)
    else:
        ref_s = ref.reindex(df.index)
    above = df["close"] > ref_s
    return above.rolling(window=bars, min_periods=bars).sum().eq(bars).fillna(False).astype(bool)


def close_below(df: pd.DataFrame, ref, bars: int = 1) -> pd.Series:
    """Close has been below `ref` for the last `bars` bars."""
    if "close" not in df.columns:
        raise ValueError("dataframe missing 'close' column")
    bars = _positive_int(bars, "bars")
    if isinstance(ref, (int, float)):
        ref_s = pd.Series(float(ref), index=df.index)
    else:
        ref_s = ref.reindex(df.index)
    below = df["close"] < ref_s
    return below.rolling(window=bars, min_periods=bars).sum().eq(bars).fillna(False).astype(bool)


def price_touch_or_cross(df: pd.DataFrame, level, direction: str = "any") -> pd.Series:
    """Bar's high-low range touches or crosses `level`."""
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    if isinstance(level, (int, float)):
        level_s = pd.Series(float(level), index=df.index)
    else:
        level_s = level.reindex(df.index)
    touched = (df["low"] <= level_s) & (df["high"] >= level_s)
    if direction == "any":
        return touched.fillna(False).astype(bool)
    prev_close = df["close"].shift(1)
    if direction == "up":
        return (touched & (prev_close < level_s)).fillna(False).astype(bool)
    if direction == "down":
        return (touched & (prev_close > level_s)).fillna(False).astype(bool)
    raise ValueError(f"direction must be 'any' / 'up' / 'down', got {direction!r}")

