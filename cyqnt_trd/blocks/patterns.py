"""Candlestick pattern detectors.

Each function takes a DataFrame with OHLC columns and returns a boolean
``pandas.Series`` aligned to the DataFrame's index — ``True`` on bars
where the pattern is present.

Pattern definitions follow Steve Nison & TA-Lib conventions. Definitions
that depend on multi-bar context (e.g. ``morning_star``) require at least
the trailing N bars to be present; earlier rows return ``False``.

Examples
--------
>>> from cyqnt_trd.blocks import patterns as pat
>>> bullish = pat.bullish_engulfing(df)
>>> hammer = pat.hammer(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import ensure_df, safe_divide

__all__ = [
    "candle_body",
    "candle_range",
    "candle_body_pct",
    "candle_upper_shadow",
    "candle_lower_shadow",
    "candle_upper_shadow_pct",
    "candle_lower_shadow_pct",
    "is_bullish",
    "is_bearish",
    "doji",
    "hammer",
    "inverted_hammer",
    "shooting_star",
    "hanging_man",
    "marubozu",
    "spinning_top",
    "bullish_engulfing",
    "bearish_engulfing",
    "bullish_harami",
    "bearish_harami",
    "morning_star",
    "evening_star",
    "three_white_soldiers",
    "three_black_crows",
    "piercing_line",
    "dark_cloud_cover",
    "tweezer_top",
    "tweezer_bottom",
    "gap_up",
    "gap_down",
]


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------


def candle_body(df: pd.DataFrame) -> pd.Series:
    """Absolute body size: ``|close - open|``."""
    df = ensure_df(df, required=("open", "close"))
    return (df["close"] - df["open"]).abs()


def candle_range(df: pd.DataFrame) -> pd.Series:
    """Bar high-low range."""
    df = ensure_df(df, required=("high", "low"))
    return df["high"] - df["low"]


def candle_body_pct(df: pd.DataFrame) -> pd.Series:
    """Body as a fraction of bar range. ``[0, 1]``."""
    return safe_divide(candle_body(df), candle_range(df), fill=0.0)


def candle_upper_shadow(df: pd.DataFrame) -> pd.Series:
    """Length of the upper wick: ``high - max(open, close)``."""
    df = ensure_df(df, required=("open", "high", "close"))
    return df["high"] - df[["open", "close"]].max(axis=1)


def candle_lower_shadow(df: pd.DataFrame) -> pd.Series:
    """Length of the lower wick: ``min(open, close) - low``."""
    df = ensure_df(df, required=("open", "low", "close"))
    return df[["open", "close"]].min(axis=1) - df["low"]


def candle_upper_shadow_pct(df: pd.DataFrame) -> pd.Series:
    """Upper shadow as fraction of range."""
    return safe_divide(candle_upper_shadow(df), candle_range(df), fill=0.0)


def candle_lower_shadow_pct(df: pd.DataFrame) -> pd.Series:
    """Lower shadow as fraction of range."""
    return safe_divide(candle_lower_shadow(df), candle_range(df), fill=0.0)


def is_bullish(df: pd.DataFrame) -> pd.Series:
    """Boolean series: True where ``close > open``."""
    df = ensure_df(df, required=("open", "close"))
    return (df["close"] > df["open"]).fillna(False).astype(bool)


def is_bearish(df: pd.DataFrame) -> pd.Series:
    """Boolean series: True where ``close < open``."""
    df = ensure_df(df, required=("open", "close"))
    return (df["close"] < df["open"]).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Single-bar patterns
# ---------------------------------------------------------------------------


def doji(df: pd.DataFrame, body_to_range_max: float = 0.1) -> pd.Series:
    """Doji: very small body relative to range.

    *body_to_range_max* is the maximum allowed ``body / range`` ratio
    (default 10%).
    """
    return (candle_body_pct(df) <= body_to_range_max).fillna(False).astype(bool)


def marubozu(df: pd.DataFrame, body_to_range_min: float = 0.95) -> pd.Series:
    """Marubozu: body fills almost the whole bar (no shadows)."""
    return (candle_body_pct(df) >= body_to_range_min).fillna(False).astype(bool)


def spinning_top(df: pd.DataFrame, body_to_range_max: float = 0.3) -> pd.Series:
    """Spinning top: small body with both upper & lower shadows."""
    body_pct = candle_body_pct(df)
    upper_pct = candle_upper_shadow_pct(df)
    lower_pct = candle_lower_shadow_pct(df)
    return (
        (body_pct <= body_to_range_max) & (upper_pct >= 0.2) & (lower_pct >= 0.2)
    ).fillna(False).astype(bool)


def hammer(
    df: pd.DataFrame,
    lower_shadow_to_body_min: float = 2.0,
    upper_shadow_to_body_max: float = 0.3,
) -> pd.Series:
    """Hammer: small body near top, long lower shadow.

    Typically appears at the bottom of a downtrend (use with regime/
    trend filter to confirm context).
    """
    body = candle_body(df)
    lower = candle_lower_shadow(df)
    upper = candle_upper_shadow(df)
    eps = 1e-12
    return (
        (lower >= lower_shadow_to_body_min * body)
        & (upper <= upper_shadow_to_body_max * body)
        & (body > eps)
    ).fillna(False).astype(bool)


def inverted_hammer(
    df: pd.DataFrame,
    upper_shadow_to_body_min: float = 2.0,
    lower_shadow_to_body_max: float = 0.3,
) -> pd.Series:
    """Inverted hammer: small body near bottom, long upper shadow."""
    body = candle_body(df)
    lower = candle_lower_shadow(df)
    upper = candle_upper_shadow(df)
    eps = 1e-12
    return (
        (upper >= upper_shadow_to_body_min * body)
        & (lower <= lower_shadow_to_body_max * body)
        & (body > eps)
    ).fillna(False).astype(bool)


def shooting_star(df: pd.DataFrame) -> pd.Series:
    """Shooting star: bearish version of inverted hammer (close < open)."""
    return (inverted_hammer(df) & is_bearish(df)).fillna(False).astype(bool)


def hanging_man(df: pd.DataFrame) -> pd.Series:
    """Hanging man: bearish version of hammer (close < open)."""
    return (hammer(df) & is_bearish(df)).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Two-bar patterns
# ---------------------------------------------------------------------------


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish engulfing: prev bearish + current bullish whose body engulfs prev."""
    df = ensure_df(df, required=("open", "close"))
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    cur_open = df["open"]
    cur_close = df["close"]
    prev_bearish = prev_close < prev_open
    cur_bullish = cur_close > cur_open
    engulfs = (cur_open <= prev_close) & (cur_close >= prev_open)
    return (prev_bearish & cur_bullish & engulfs).fillna(False)


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bearish engulfing: prev bullish + current bearish whose body engulfs prev."""
    df = ensure_df(df, required=("open", "close"))
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    cur_open = df["open"]
    cur_close = df["close"]
    prev_bullish = prev_close > prev_open
    cur_bearish = cur_close < cur_open
    engulfs = (cur_open >= prev_close) & (cur_close <= prev_open)
    return (prev_bullish & cur_bearish & engulfs).fillna(False)


def bullish_harami(df: pd.DataFrame) -> pd.Series:
    """Bullish harami: prev big bearish + current small bullish inside prev body."""
    df = ensure_df(df, required=("open", "close"))
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    cur_open = df["open"]
    cur_close = df["close"]
    prev_bearish = prev_close < prev_open
    cur_bullish = cur_close > cur_open
    inside = (cur_open >= prev_close) & (cur_close <= prev_open)
    return (prev_bearish & cur_bullish & inside).fillna(False)


def bearish_harami(df: pd.DataFrame) -> pd.Series:
    """Bearish harami: prev big bullish + current small bearish inside prev body."""
    df = ensure_df(df, required=("open", "close"))
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    cur_open = df["open"]
    cur_close = df["close"]
    prev_bullish = prev_close > prev_open
    cur_bearish = cur_close < cur_open
    inside = (cur_open <= prev_close) & (cur_close >= prev_open)
    return (prev_bullish & cur_bearish & inside).fillna(False)


def piercing_line(df: pd.DataFrame) -> pd.Series:
    """Piercing line: prev bearish + current bullish opens below prev low and closes above prev midpoint."""
    df = ensure_df(df, required=("open", "high", "low", "close"))
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_low = df["low"].shift(1)
    cur_open = df["open"]
    cur_close = df["close"]
    prev_mid = (prev_open + prev_close) / 2.0
    return (
        (prev_close < prev_open)
        & (cur_open < prev_low)
        & (cur_close > prev_mid)
        & (cur_close < prev_open)
    ).fillna(False)


def dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    """Dark cloud cover: prev bullish + current bearish opens above prev high and closes below prev midpoint."""
    df = ensure_df(df, required=("open", "high", "low", "close"))
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_high = df["high"].shift(1)
    cur_open = df["open"]
    cur_close = df["close"]
    prev_mid = (prev_open + prev_close) / 2.0
    return (
        (prev_close > prev_open)
        & (cur_open > prev_high)
        & (cur_close < prev_mid)
        & (cur_close > prev_open)
    ).fillna(False)


def tweezer_top(df: pd.DataFrame, tol_pct: float = 0.001) -> pd.Series:
    """Tweezer top: two consecutive bars share approximately the same high."""
    df = ensure_df(df, required=("high",))
    diff = (df["high"] - df["high"].shift(1)).abs()
    avg = (df["high"] + df["high"].shift(1)) / 2.0
    return (diff <= tol_pct * avg).fillna(False)


def tweezer_bottom(df: pd.DataFrame, tol_pct: float = 0.001) -> pd.Series:
    """Tweezer bottom: two consecutive bars share approximately the same low."""
    df = ensure_df(df, required=("low",))
    diff = (df["low"] - df["low"].shift(1)).abs()
    avg = (df["low"] + df["low"].shift(1)) / 2.0
    return (diff <= tol_pct * avg).fillna(False)


# ---------------------------------------------------------------------------
# Three-bar patterns
# ---------------------------------------------------------------------------


def morning_star(df: pd.DataFrame, body_size_min: float = 0.6) -> pd.Series:
    """Morning star: bearish + indecision + bullish that closes past prev[1] midpoint.

    *body_size_min* is the minimum body/range fraction required for the
    first and third bars to count as "big" candles.
    """
    df = ensure_df(df, required=("open", "high", "low", "close"))
    body_pct = candle_body_pct(df)
    bullish = is_bullish(df)
    bearish = is_bearish(df)

    cond1 = bearish.shift(2) & (body_pct.shift(2) >= body_size_min)
    # middle bar small body, gap-down vs prev close
    middle_body = body_pct.shift(1) <= 0.3
    cond2 = middle_body
    cond3 = bullish & (body_pct >= body_size_min) & (
        df["close"] > ((df["open"].shift(2) + df["close"].shift(2)) / 2.0)
    )
    return (cond1 & cond2 & cond3).fillna(False)


def evening_star(df: pd.DataFrame, body_size_min: float = 0.6) -> pd.Series:
    """Evening star: bullish + indecision + bearish that closes past prev[1] midpoint."""
    df = ensure_df(df, required=("open", "high", "low", "close"))
    body_pct = candle_body_pct(df)
    bullish = is_bullish(df)
    bearish = is_bearish(df)

    cond1 = bullish.shift(2) & (body_pct.shift(2) >= body_size_min)
    middle_body = body_pct.shift(1) <= 0.3
    cond2 = middle_body
    cond3 = bearish & (body_pct >= body_size_min) & (
        df["close"] < ((df["open"].shift(2) + df["close"].shift(2)) / 2.0)
    )
    return (cond1 & cond2 & cond3).fillna(False)


def three_white_soldiers(df: pd.DataFrame, body_size_min: float = 0.6) -> pd.Series:
    """Three white soldiers: three consecutive bullish bars each closing higher."""
    df = ensure_df(df, required=("open", "close"))
    body_pct = candle_body_pct(df)
    bullish = is_bullish(df)
    cond = (
        bullish.shift(2)
        & bullish.shift(1)
        & bullish
        & (body_pct.shift(2) >= body_size_min)
        & (body_pct.shift(1) >= body_size_min)
        & (body_pct >= body_size_min)
        & (df["close"] > df["close"].shift(1))
        & (df["close"].shift(1) > df["close"].shift(2))
    )
    return cond.fillna(False)


def three_black_crows(df: pd.DataFrame, body_size_min: float = 0.6) -> pd.Series:
    """Three black crows: three consecutive bearish bars each closing lower."""
    df = ensure_df(df, required=("open", "close"))
    body_pct = candle_body_pct(df)
    bearish = is_bearish(df)
    cond = (
        bearish.shift(2)
        & bearish.shift(1)
        & bearish
        & (body_pct.shift(2) >= body_size_min)
        & (body_pct.shift(1) >= body_size_min)
        & (body_pct >= body_size_min)
        & (df["close"] < df["close"].shift(1))
        & (df["close"].shift(1) < df["close"].shift(2))
    )
    return cond.fillna(False)


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


def gap_up(df: pd.DataFrame, min_gap_pct: float = 0.5) -> pd.Series:
    """Bar opens at least *min_gap_pct* above the previous close."""
    df = ensure_df(df, required=("open", "close"))
    prev_close = df["close"].shift(1)
    gap = safe_divide(df["open"] - prev_close, prev_close, fill=0.0) * 100.0
    return (gap >= min_gap_pct).fillna(False)


def gap_down(df: pd.DataFrame, min_gap_pct: float = 0.5) -> pd.Series:
    """Bar opens at least *min_gap_pct* below the previous close."""
    df = ensure_df(df, required=("open", "close"))
    prev_close = df["close"].shift(1)
    gap = safe_divide(prev_close - df["open"], prev_close, fill=0.0) * 100.0
    return (gap >= min_gap_pct).fillna(False)
