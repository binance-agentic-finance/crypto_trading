"""Atomic boolean entry conditions.

Each function returns a boolean ``pandas.Series`` aligned to the input
index. Conditions can be combined with ``&`` / ``|`` / ``~`` directly,
or via :mod:`cyqnt_trd.blocks.entry` combinators (``all_of`` / ``any_of``
/ ``score_entry``).

Examples
--------
>>> from cyqnt_trd.blocks import indicators as ind, conditions as cond
>>> ma20 = ind.sma(df["close"], 20)
>>> ma60 = ind.sma(df["close"], 60)
>>> long_signal = cond.ma_cross_above(ma20, ma60) & cond.rsi_in_range(ind.rsi(df["close"]), 50, 75)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ._utils import (
    SeriesLike,
    crossover,
    crossunder,
    ensure_df,
    ensure_series,
    positive_int,
    rolling_max,
    rolling_min,
)

__all__ = [
    # Crossover
    "ma_cross_above",
    "ma_cross_below",
    # Bounce / breakout / retest
    "price_bounce_ma",
    "breakout_high",
    "breakout_low",
    "retest_after_breakout",
    "consolidation_range",
    "range_detection",
    # Volume
    "volume_surge",
    "volume_shrink",
    # MACD
    "macd_golden_cross",
    "macd_death_cross",
    "macd_above_zero",
    "macd_below_zero",
    "macd_bullish_divergence",
    "macd_bearish_divergence",
    # RSI
    "rsi_overbought",
    "rsi_oversold",
    "rsi_in_range",
    # StochRSI (faster than RSI)
    "stochrsi_oversold",
    "stochrsi_overbought",
    "stochrsi_cross_above",
    # ADX
    "adx_trending",
    "adx_ranging",
    "adx_direction_long",
    "adx_direction_short",
    # Aroon (trend strength)
    "aroon_up_strong",
    "aroon_down_strong",
    "aroon_oscillator_above",
    # PSAR (trend flip detector)
    "psar_flip_up",
    "psar_flip_down",
    # MA position
    "price_above_ma",
    "price_below_ma",
    "ema_deviation_within",
    "close_above",
    "close_below",
    "price_touch_or_cross",
    # Bar shape
    "is_bullish_bar",
    "is_bearish_bar",
    # Time / funding
    "time_filter",
    "funding_window_safe",
    # Multi-frame / structure
    "multi_timeframe_alignment",
    "higher_high",
    "higher_low",
    "lower_high",
    "lower_low",
    "liquidity_sweep_high",
    "liquidity_sweep_low",
    # Convenience aliases re-exported from sibling modules
    "ma_alignment",
    "consecutive",
    "candle_lower_shadow",
    "candle_upper_shadow",
]


# ---------------------------------------------------------------------------
# Crossovers
# ---------------------------------------------------------------------------


def ma_cross_above(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on the bar *fast* MA crosses above *slow* MA (golden cross)."""
    return crossover(ensure_series(fast), ensure_series(slow))


def ma_cross_below(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on the bar *fast* MA crosses below *slow* MA (death cross)."""
    return crossunder(ensure_series(fast), ensure_series(slow))


# ---------------------------------------------------------------------------
# Bounce / breakout / retest
# ---------------------------------------------------------------------------


def price_bounce_ma(df: pd.DataFrame, ma: pd.Series, direction: str = "long") -> pd.Series:
    """Detect a "touch and reject" of *ma*.

    For ``direction="long"``: bar low touches/penetrates MA from above and
    bar closes back above MA (i.e. bullish rejection).

    For ``direction="short"``: bar high touches/penetrates MA from below
    and bar closes back below MA.
    """
    df = ensure_df(df, required=("low", "high", "close"))
    ma = ensure_series(ma)
    if direction == "long":
        return ((df["low"] <= ma) & (df["close"] > ma)).fillna(False)
    if direction == "short":
        return ((df["high"] >= ma) & (df["close"] < ma)).fillna(False)
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def breakout_high(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Bar close exceeds the highest high of the previous *lookback* bars."""
    df = ensure_df(df, required=("close", "high"))
    lookback = positive_int(lookback, "lookback")
    prior_high = rolling_max(df["high"].shift(1), lookback)
    return (df["close"] > prior_high).fillna(False)


def breakout_low(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Bar close falls below the lowest low of the previous *lookback* bars."""
    df = ensure_df(df, required=("close", "low"))
    lookback = positive_int(lookback, "lookback")
    prior_low = rolling_min(df["low"].shift(1), lookback)
    return (df["close"] < prior_low).fillna(False)


def retest_after_breakout(
    df: pd.DataFrame, lookback: int = 20, retest_window: int = 5
) -> pd.Series:
    """Detect a retest of the breakout level within *retest_window* bars after a breakout."""
    df = ensure_df(df, required=("close", "high", "low"))
    lookback = positive_int(lookback, "lookback")
    retest_window = positive_int(retest_window, "retest_window")

    breakout = breakout_high(df, lookback)
    breakout_level = rolling_max(df["high"].shift(1), lookback)
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
    df: pd.DataFrame, period: int = 20, max_range_pct: float = 0.03
) -> pd.Series:
    """True on bars where (high-low)/mid over *period* is below *max_range_pct*."""
    df = ensure_df(df, required=("high", "low"))
    period = positive_int(period, "period")
    hh = rolling_max(df["high"], period)
    ll = rolling_min(df["low"], period)
    rng_pct = (hh - ll) / ((hh + ll) / 2.0)
    return (rng_pct <= max_range_pct).fillna(False).astype(bool)


def range_detection(
    df: pd.DataFrame, period: int = 20, max_range_pct: float = 0.03
) -> pd.Series:
    """Alias for :func:`consolidation_range` (matches user vocabulary)."""
    return consolidation_range(df, period=period, max_range_pct=max_range_pct)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def volume_surge(df: pd.DataFrame, ref_volume: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Bar volume >= *multiplier* * *ref_volume* (e.g. volume MA)."""
    df = ensure_df(df, required=("volume",))
    ref = ensure_series(ref_volume)
    return (df["volume"] >= multiplier * ref).fillna(False)


def volume_shrink(
    df: pd.DataFrame, ref_volume: pd.Series, bars: int = 3, multiplier: float = 1.0
) -> pd.Series:
    """All of the last *bars* bars have volume < *multiplier* * *ref_volume*."""
    df = ensure_df(df, required=("volume",))
    bars = positive_int(bars, "bars")
    ref = ensure_series(ref_volume)
    cond = df["volume"] < multiplier * ref
    return cond.rolling(window=bars, min_periods=bars).sum().eq(bars).fillna(False)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd_golden_cross(macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
    """MACD line crosses above signal line."""
    return crossover(macd_line, signal_line)


def macd_death_cross(macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
    """MACD line crosses below signal line."""
    return crossunder(macd_line, signal_line)


def macd_above_zero(macd_line: pd.Series) -> pd.Series:
    """MACD line is above zero."""
    return (ensure_series(macd_line) > 0).fillna(False).astype(bool)


def macd_below_zero(macd_line: pd.Series) -> pd.Series:
    """MACD line is below zero."""
    return (ensure_series(macd_line) < 0).fillna(False).astype(bool)


def macd_bullish_divergence(
    price: pd.Series, macd_line: pd.Series, lookback: int = 20
) -> pd.Series:
    """Price LL while MACD HL — classical bullish divergence over *lookback* bars."""
    lookback = positive_int(lookback, "lookback")
    p = ensure_series(price)
    m = ensure_series(macd_line)
    p_low = p.rolling(window=lookback, min_periods=lookback).min()
    m_low = m.rolling(window=lookback, min_periods=lookback).min()
    return ((p == p_low) & (m > m_low)).fillna(False)


def macd_bearish_divergence(
    price: pd.Series, macd_line: pd.Series, lookback: int = 20
) -> pd.Series:
    """Price HH while MACD LH — classical bearish divergence."""
    lookback = positive_int(lookback, "lookback")
    p = ensure_series(price)
    m = ensure_series(macd_line)
    p_high = p.rolling(window=lookback, min_periods=lookback).max()
    m_high = m.rolling(window=lookback, min_periods=lookback).max()
    return ((p == p_high) & (m < m_high)).fillna(False)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def rsi_overbought(rsi_series: pd.Series, threshold: float = 70.0) -> pd.Series:
    """RSI above overbought threshold."""
    return (ensure_series(rsi_series) >= threshold).fillna(False).astype(bool)


def rsi_oversold(rsi_series: pd.Series, threshold: float = 30.0) -> pd.Series:
    """RSI below oversold threshold."""
    return (ensure_series(rsi_series) <= threshold).fillna(False).astype(bool)


def rsi_in_range(rsi_series: pd.Series, low: float = 40.0, high: float = 60.0) -> pd.Series:
    """RSI in ``[low, high]`` (inclusive)."""
    s = ensure_series(rsi_series)
    return ((s >= low) & (s <= high)).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# StochRSI (faster than RSI)
# ---------------------------------------------------------------------------


def stochrsi_oversold(stochrsi_series: pd.Series, threshold: float = 20.0) -> pd.Series:
    """StochRSI below oversold threshold."""
    return (ensure_series(stochrsi_series) <= threshold).fillna(False).astype(bool)


def stochrsi_overbought(stochrsi_series: pd.Series, threshold: float = 80.0) -> pd.Series:
    """StochRSI above overbought threshold."""
    return (ensure_series(stochrsi_series) >= threshold).fillna(False).astype(bool)


def stochrsi_cross_above(
    stochrsi_series: pd.Series, lookback: int = 14, threshold: float = 20.0
) -> pd.Series:
    """StochRSI crosses above *threshold*."""
    lookback = positive_int(lookback, "lookback")
    s = ensure_series(stochrsi_series)
    return crossover(s, pd.Series(threshold, index=s.index)).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


def adx_trending(adx_series: pd.Series, threshold: float = 25.0) -> pd.Series:
    """ADX >= threshold (market in trend regime)."""
    return (ensure_series(adx_series) >= threshold).fillna(False).astype(bool)


def adx_ranging(adx_series: pd.Series, threshold: float = 20.0) -> pd.Series:
    """ADX < threshold (market in range regime)."""
    return (ensure_series(adx_series) < threshold).fillna(False).astype(bool)


def adx_direction_long(plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    """``+DI > -DI`` — directional bias is long."""
    return (ensure_series(plus_di) > ensure_series(minus_di)).fillna(False).astype(bool)


def adx_direction_short(plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    """``-DI > +DI`` — directional bias is short."""
    return (ensure_series(minus_di) > ensure_series(plus_di)).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Aroon (trend strength)
# ---------------------------------------------------------------------------


def aroon_up_strong(aroon_up_series: pd.Series, threshold: float = 70.0) -> pd.Series:
    """Aroon Up >= threshold."""
    return (ensure_series(aroon_up_series) >= threshold).fillna(False).astype(bool)


def aroon_down_strong(aroon_down_series: pd.Series, threshold: float = 70.0) -> pd.Series:
    """Aroon Down >= threshold."""
    return (ensure_series(aroon_down_series) >= threshold).fillna(False).astype(bool)


def aroon_oscillator_above(
    aroon_up_series: pd.Series, aroon_down_series: pd.Series, threshold: float = 20.0
) -> pd.Series:
    """Aroon Up - Aroon Down >= threshold."""
    return (
        (ensure_series(aroon_up_series) - ensure_series(aroon_down_series))
        >= threshold
    ).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# PSAR (trend flip detector)
# ---------------------------------------------------------------------------


def psar_flip_up(psar_series: pd.Series, close_series: pd.Series) -> pd.Series:
    """PSAR was above close (downtrend) and now flips below close (uptrend).

    Triggers on the bar where the trend reversal completes — the canonical
    long entry signal from the Parabolic SAR system.

    Parameters
    ----------
    psar_series : pd.Series
        SAR value series (first element of ``indicators.parabolic_sar()``).
    close_series : pd.Series
        Close price series.
    """
    psar = ensure_series(psar_series)
    close = ensure_series(close_series)
    return (
        (psar.shift(1) > close.shift(1))   # was downtrend
        & (psar < close)                    # now uptrend
    ).fillna(False).astype(bool)


def psar_flip_down(psar_series: pd.Series, close_series: pd.Series) -> pd.Series:
    """PSAR was below close (uptrend) and now flips above close (downtrend).

    Triggers on the bar where the trend reversal completes — the canonical
    short entry / long exit signal from the Parabolic SAR system.
    """
    psar = ensure_series(psar_series)
    close = ensure_series(close_series)
    return (
        (psar.shift(1) < close.shift(1))   # was uptrend
        & (psar > close)                    # now downtrend
    ).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# MA position
# ---------------------------------------------------------------------------


def price_above_ma(df: pd.DataFrame, ma: pd.Series, bars: int = 1) -> pd.Series:
    """Close has been above *ma* for the last *bars* bars."""
    df = ensure_df(df, required=("close",))
    bars = positive_int(bars, "bars")
    ma = ensure_series(ma)
    above = df["close"] > ma
    return above.rolling(window=bars, min_periods=bars).sum().eq(bars).fillna(False)


def price_below_ma(df: pd.DataFrame, ma: pd.Series, bars: int = 1) -> pd.Series:
    """Close has been below *ma* for the last *bars* bars."""
    df = ensure_df(df, required=("close",))
    bars = positive_int(bars, "bars")
    ma = ensure_series(ma)
    below = df["close"] < ma
    return below.rolling(window=bars, min_periods=bars).sum().eq(bars).fillna(False)


def ema_deviation_within(
    price: pd.Series, ema_series: pd.Series, max_pct: float
) -> pd.Series:
    """``|price - ema| / ema <= max_pct`` — used to reject over-extended entries."""
    p = ensure_series(price)
    e = ensure_series(ema_series)
    deviation = (p - e).abs() / e.replace(0.0, np.nan)
    return (deviation <= max_pct).fillna(False)


# ---------------------------------------------------------------------------
# Bar shape
# ---------------------------------------------------------------------------


def is_bullish_bar(df: pd.DataFrame, min_body_pct: float = 0.0) -> pd.Series:
    """Close > open and ``(close - open) / open >= min_body_pct / 100``."""
    df = ensure_df(df, required=("open", "close"))
    body_pct = (df["close"] - df["open"]) / df["open"].replace(0.0, np.nan) * 100.0
    return ((df["close"] > df["open"]) & (body_pct >= min_body_pct)).fillna(False)


def is_bearish_bar(df: pd.DataFrame, min_body_pct: float = 0.0) -> pd.Series:
    """Close < open and ``(open - close) / open >= min_body_pct / 100``."""
    df = ensure_df(df, required=("open", "close"))
    body_pct = (df["open"] - df["close"]) / df["open"].replace(0.0, np.nan) * 100.0
    return ((df["close"] < df["open"]) & (body_pct >= min_body_pct)).fillna(False)


# ---------------------------------------------------------------------------
# Time filters
# ---------------------------------------------------------------------------


def time_filter(
    timestamps_ms: SeriesLike,
    start_hour: int,
    end_hour: int,
    tz_offset_hours: int = 8,
) -> pd.Series:
    """Boolean series: True when local hour is in ``[start_hour, end_hour)``.

    *tz_offset_hours* is added to UTC. Default is UTC+8 (Asia/Singapore /
    Beijing / Taipei) — matches typical user requests in the dataset.
    """
    ts = pd.to_datetime(ensure_series(timestamps_ms).astype("int64"), unit="ms", utc=True)
    local = ts + pd.Timedelta(hours=tz_offset_hours)
    hour = local.dt.hour
    if start_hour <= end_hour:
        out = (hour >= start_hour) & (hour < end_hour)
    else:
        # cross-midnight window
        out = (hour >= start_hour) | (hour < end_hour)
    return out.fillna(False).astype(bool)


def funding_window_safe(
    timestamps_ms: SeriesLike,
    settle_hours_utc: Iterable[int] = (0, 8, 16),
    buffer_min: int = 15,
) -> pd.Series:
    """True on bars *outside* the buffer around funding-rate settlement.

    Default Binance USDT-M settles every 8h at UTC 00:00 / 08:00 / 16:00;
    we exclude bars that fall within ``buffer_min`` of those times.
    """
    if buffer_min < 0:
        raise ValueError(f"buffer_min must be >= 0, got {buffer_min}")
    ts = pd.to_datetime(ensure_series(timestamps_ms).astype("int64"), unit="ms", utc=True)
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    settle_minutes = {h * 60 for h in settle_hours_utc}
    distance = minute_of_day.apply(
        lambda m: min(min(abs(m - s), 24 * 60 - abs(m - s)) for s in settle_minutes)
    )
    return (distance > buffer_min).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Multi-timeframe / structure
# ---------------------------------------------------------------------------


def multi_timeframe_alignment(*signals: pd.Series) -> pd.Series:
    """All boolean signals are True at the same time."""
    if not signals:
        raise ValueError("at least one signal required")
    out = ensure_series(signals[0]).fillna(False).astype(bool)
    for s in signals[1:]:
        out = out & ensure_series(s).fillna(False).astype(bool)
    return out


def higher_high(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Current high is the highest in the last *lookback* bars and exceeds the previous swing high."""
    df = ensure_df(df, required=("high",))
    lookback = positive_int(lookback, "lookback")
    rolling_h = rolling_max(df["high"], lookback)
    return ((df["high"] == rolling_h) & (df["high"] > df["high"].shift(lookback))).fillna(False)


def higher_low(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Current low is the lowest *low* in the last *lookback* bars but exceeds the previous swing low.

    A "higher low" structurally — the latest trough is above the trough
    *lookback* bars ago.
    """
    df = ensure_df(df, required=("low",))
    lookback = positive_int(lookback, "lookback")
    rolling_l = rolling_min(df["low"], lookback)
    return ((df["low"] == rolling_l) & (df["low"] > df["low"].shift(lookback))).fillna(False)


def lower_high(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Current high is the highest *high* in the last *lookback* bars but below the previous swing high.

    A "lower high" structurally — the latest peak is below the peak
    *lookback* bars ago.
    """
    df = ensure_df(df, required=("high",))
    lookback = positive_int(lookback, "lookback")
    rolling_h = rolling_max(df["high"], lookback)
    return ((df["high"] == rolling_h) & (df["high"] < df["high"].shift(lookback))).fillna(False)


def lower_low(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Current low is the lowest in the last *lookback* bars and below the previous swing low."""
    df = ensure_df(df, required=("low",))
    lookback = positive_int(lookback, "lookback")
    rolling_l = rolling_min(df["low"], lookback)
    return ((df["low"] == rolling_l) & (df["low"] < df["low"].shift(lookback))).fillna(False)


def liquidity_sweep_high(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Wick poke above prior swing high but close back below it (failed breakout)."""
    df = ensure_df(df, required=("high", "close"))
    lookback = positive_int(lookback, "lookback")
    prior_high = rolling_max(df["high"].shift(1), lookback)
    return ((df["high"] > prior_high) & (df["close"] <= prior_high)).fillna(False)


def liquidity_sweep_low(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Wick poke below prior swing low but close back above it (failed breakdown)."""
    df = ensure_df(df, required=("low", "close"))
    lookback = positive_int(lookback, "lookback")
    prior_low = rolling_min(df["low"].shift(1), lookback)
    return ((df["low"] < prior_low) & (df["close"] >= prior_low)).fillna(False)


# ---------------------------------------------------------------------------
# Convenience aliases for LLM-generated code
# ---------------------------------------------------------------------------
# These names appear frequently in LLM-generated code that mistakenly reaches
# into ``conditions`` for things actually defined elsewhere. We re-export them
# here as a forgiveness layer so that auto-generated strategies don't break
# on cosmetic naming differences.


def close_above(
    df: pd.DataFrame, ref, bars: int = 1
) -> pd.Series:
    """Bar close has been above *ref* for the last *bars* bars.

    *ref* can be either a constant scalar (e.g. 100.0) or a pandas Series
    (e.g. an MA / EMA). Equivalent to :func:`price_above_ma` with a more
    intuitive name.
    """
    df = ensure_df(df, required=("close",))
    bars = positive_int(bars, "bars")
    if isinstance(ref, (int, float)):
        ref_s = pd.Series(float(ref), index=df.index)
    else:
        ref_s = ensure_series(ref).reindex(df.index)
    above = df["close"] > ref_s
    return above.rolling(window=bars, min_periods=bars).sum().eq(bars).fillna(False).astype(bool)


def close_below(
    df: pd.DataFrame, ref, bars: int = 1
) -> pd.Series:
    """Bar close has been below *ref* for the last *bars* bars."""
    df = ensure_df(df, required=("close",))
    bars = positive_int(bars, "bars")
    if isinstance(ref, (int, float)):
        ref_s = pd.Series(float(ref), index=df.index)
    else:
        ref_s = ensure_series(ref).reindex(df.index)
    below = df["close"] < ref_s
    return below.rolling(window=bars, min_periods=bars).sum().eq(bars).fillna(False).astype(bool)


def price_touch_or_cross(
    df: pd.DataFrame, level, direction: str = "any"
) -> pd.Series:
    """Bar's high-low range crosses or touches the level.

    Use this for "did price reach this MA / S/R level on this bar" tests.

    Parameters
    ----------
    df: DataFrame with at least ``high`` and ``low`` columns.
    level: scalar or pandas.Series — the target price level.
    direction: "any" (default), "up" (came from below) or "down" (came from above).
    """
    df = ensure_df(df, required=("high", "low", "close"))
    if isinstance(level, (int, float)):
        level_s = pd.Series(float(level), index=df.index)
    else:
        level_s = ensure_series(level).reindex(df.index)
    touched = (df["low"] <= level_s) & (df["high"] >= level_s)
    if direction == "any":
        return touched.fillna(False).astype(bool)
    prev_close = df["close"].shift(1)
    if direction == "up":
        return (touched & (prev_close < level_s)).fillna(False).astype(bool)
    if direction == "down":
        return (touched & (prev_close > level_s)).fillna(False).astype(bool)
    raise ValueError(f"direction must be 'any' / 'up' / 'down', got {direction!r}")


# Re-export names that LLMs often expect in the `conditions` namespace
# even though they live elsewhere.
def __getattr__(name: str):
    if name == "ma_alignment":
        from .indicators import ma_alignment as _impl
        return _impl
    if name == "consecutive":
        from .entry import consecutive as _impl
        return _impl
    if name == "candle_lower_shadow":
        from .patterns import candle_lower_shadow as _impl
        return _impl
    if name == "candle_upper_shadow":
        from .patterns import candle_upper_shadow as _impl
        return _impl
    raise AttributeError(f"module 'cyqnt_trd.blocks.conditions' has no attribute {name!r}")
