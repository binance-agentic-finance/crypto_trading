"""Candle pattern detectors — full parity vs `cyqnt_trd.blocks.patterns`.

Detectors are boolean Series; the simpler `library.features.detectors.doji` /
`engulfing` exist as merge variants — these patterns are the canonical
parity-tested set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"dataframe missing columns: {missing}")


def _safe_divide(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------


def candle_body(df: pd.DataFrame) -> pd.Series:
    _require(df, ("open", "close"))
    return (df["close"] - df["open"]).abs()


def candle_range(df: pd.DataFrame) -> pd.Series:
    _require(df, ("high", "low"))
    return df["high"] - df["low"]


def candle_body_pct(df: pd.DataFrame) -> pd.Series:
    return _safe_divide(candle_body(df), candle_range(df), fill=0.0)


def candle_upper_shadow(df: pd.DataFrame) -> pd.Series:
    _require(df, ("open", "high", "close"))
    return df["high"] - df[["open", "close"]].max(axis=1)


def candle_lower_shadow(df: pd.DataFrame) -> pd.Series:
    _require(df, ("open", "low", "close"))
    return df[["open", "close"]].min(axis=1) - df["low"]


def candle_upper_shadow_pct(df: pd.DataFrame) -> pd.Series:
    return _safe_divide(candle_upper_shadow(df), candle_range(df), fill=0.0)


def candle_lower_shadow_pct(df: pd.DataFrame) -> pd.Series:
    return _safe_divide(candle_lower_shadow(df), candle_range(df), fill=0.0)


def is_bullish(df: pd.DataFrame) -> pd.Series:
    _require(df, ("open", "close"))
    return (df["close"] > df["open"]).fillna(False).astype(bool)


def is_bearish(df: pd.DataFrame) -> pd.Series:
    _require(df, ("open", "close"))
    return (df["close"] < df["open"]).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Single-bar patterns
# ---------------------------------------------------------------------------


def doji(df: pd.DataFrame, body_to_range_max: float = 0.1) -> pd.Series:
    return (candle_body_pct(df) <= float(body_to_range_max)).fillna(False).astype(bool)


def marubozu(df: pd.DataFrame, body_to_range_min: float = 0.95) -> pd.Series:
    return (candle_body_pct(df) >= float(body_to_range_min)).fillna(False).astype(bool)


def spinning_top(df: pd.DataFrame, body_to_range_max: float = 0.3) -> pd.Series:
    body_pct = candle_body_pct(df)
    upper_pct = candle_upper_shadow_pct(df)
    lower_pct = candle_lower_shadow_pct(df)
    return (
        (body_pct <= float(body_to_range_max))
        & (upper_pct >= 0.2)
        & (lower_pct >= 0.2)
    ).fillna(False).astype(bool)


def hammer(
    df: pd.DataFrame,
    lower_shadow_to_body_min: float = 2.0,
    upper_shadow_to_body_max: float = 0.3,
) -> pd.Series:
    body = candle_body(df)
    lower = candle_lower_shadow(df)
    upper = candle_upper_shadow(df)
    eps = 1e-12
    return (
        (lower >= float(lower_shadow_to_body_min) * body)
        & (upper <= float(upper_shadow_to_body_max) * body)
        & (body > eps)
    ).fillna(False).astype(bool)


def inverted_hammer(
    df: pd.DataFrame,
    upper_shadow_to_body_min: float = 2.0,
    lower_shadow_to_body_max: float = 0.3,
) -> pd.Series:
    body = candle_body(df)
    lower = candle_lower_shadow(df)
    upper = candle_upper_shadow(df)
    eps = 1e-12
    return (
        (upper >= float(upper_shadow_to_body_min) * body)
        & (lower <= float(lower_shadow_to_body_max) * body)
        & (body > eps)
    ).fillna(False).astype(bool)


def shooting_star(df: pd.DataFrame) -> pd.Series:
    return (inverted_hammer(df) & is_bearish(df)).fillna(False).astype(bool)


def hanging_man(df: pd.DataFrame) -> pd.Series:
    return (hammer(df) & is_bearish(df)).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Two-bar patterns
# ---------------------------------------------------------------------------


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    _require(df, ("open", "close"))
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    cur_open = df["open"]
    cur_close = df["close"]
    prev_bearish = prev_close < prev_open
    cur_bullish = cur_close > cur_open
    engulfs = (cur_open <= prev_close) & (cur_close >= prev_open)
    return (prev_bearish & cur_bullish & engulfs).fillna(False)


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    _require(df, ("open", "close"))
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    cur_open = df["open"]
    cur_close = df["close"]
    prev_bullish = prev_close > prev_open
    cur_bearish = cur_close < cur_open
    engulfs = (cur_open >= prev_close) & (cur_close <= prev_open)
    return (prev_bullish & cur_bearish & engulfs).fillna(False)


def bullish_harami(df: pd.DataFrame) -> pd.Series:
    _require(df, ("open", "close"))
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    cur_open = df["open"]
    cur_close = df["close"]
    prev_bearish = prev_close < prev_open
    cur_bullish = cur_close > cur_open
    inside = (cur_open >= prev_close) & (cur_close <= prev_open)
    return (prev_bearish & cur_bullish & inside).fillna(False)


def bearish_harami(df: pd.DataFrame) -> pd.Series:
    _require(df, ("open", "close"))
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    cur_open = df["open"]
    cur_close = df["close"]
    prev_bullish = prev_close > prev_open
    cur_bearish = cur_close < cur_open
    inside = (cur_open <= prev_close) & (cur_close >= prev_open)
    return (prev_bullish & cur_bearish & inside).fillna(False)


def piercing_line(df: pd.DataFrame) -> pd.Series:
    _require(df, ("open", "high", "low", "close"))
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
    _require(df, ("open", "high", "low", "close"))
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
    _require(df, ("high",))
    diff = (df["high"] - df["high"].shift(1)).abs()
    avg = (df["high"] + df["high"].shift(1)) / 2.0
    return (diff <= float(tol_pct) * avg).fillna(False)


def tweezer_bottom(df: pd.DataFrame, tol_pct: float = 0.001) -> pd.Series:
    _require(df, ("low",))
    diff = (df["low"] - df["low"].shift(1)).abs()
    avg = (df["low"] + df["low"].shift(1)) / 2.0
    return (diff <= float(tol_pct) * avg).fillna(False)


# ---------------------------------------------------------------------------
# Three-bar patterns
# ---------------------------------------------------------------------------


def morning_star(df: pd.DataFrame, body_size_min: float = 0.6) -> pd.Series:
    _require(df, ("open", "high", "low", "close"))
    body_pct = candle_body_pct(df)
    bullish = is_bullish(df)
    bearish = is_bearish(df)
    cond1 = bearish.shift(2) & (body_pct.shift(2) >= float(body_size_min))
    cond2 = body_pct.shift(1) <= 0.3
    cond3 = bullish & (body_pct >= float(body_size_min)) & (
        df["close"] > ((df["open"].shift(2) + df["close"].shift(2)) / 2.0)
    )
    return (cond1 & cond2 & cond3).fillna(False)


def evening_star(df: pd.DataFrame, body_size_min: float = 0.6) -> pd.Series:
    _require(df, ("open", "high", "low", "close"))
    body_pct = candle_body_pct(df)
    bullish = is_bullish(df)
    bearish = is_bearish(df)
    cond1 = bullish.shift(2) & (body_pct.shift(2) >= float(body_size_min))
    cond2 = body_pct.shift(1) <= 0.3
    cond3 = bearish & (body_pct >= float(body_size_min)) & (
        df["close"] < ((df["open"].shift(2) + df["close"].shift(2)) / 2.0)
    )
    return (cond1 & cond2 & cond3).fillna(False)


def three_white_soldiers(df: pd.DataFrame, body_size_min: float = 0.6) -> pd.Series:
    _require(df, ("open", "close"))
    body_pct = candle_body_pct(df)
    bullish = is_bullish(df)
    cond = (
        bullish.shift(2)
        & bullish.shift(1)
        & bullish
        & (body_pct.shift(2) >= float(body_size_min))
        & (body_pct.shift(1) >= float(body_size_min))
        & (body_pct >= float(body_size_min))
        & (df["close"] > df["close"].shift(1))
        & (df["close"].shift(1) > df["close"].shift(2))
    )
    return cond.fillna(False)


def three_black_crows(df: pd.DataFrame, body_size_min: float = 0.6) -> pd.Series:
    _require(df, ("open", "close"))
    body_pct = candle_body_pct(df)
    bearish = is_bearish(df)
    cond = (
        bearish.shift(2)
        & bearish.shift(1)
        & bearish
        & (body_pct.shift(2) >= float(body_size_min))
        & (body_pct.shift(1) >= float(body_size_min))
        & (body_pct >= float(body_size_min))
        & (df["close"] < df["close"].shift(1))
        & (df["close"].shift(1) < df["close"].shift(2))
    )
    return cond.fillna(False)


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


def gap_up(df: pd.DataFrame, min_gap_pct: float = 0.5) -> pd.Series:
    _require(df, ("open", "close"))
    prev_close = df["close"].shift(1)
    gap = _safe_divide(df["open"] - prev_close, prev_close, fill=0.0) * 100.0
    return (gap >= float(min_gap_pct)).fillna(False)


def gap_down(df: pd.DataFrame, min_gap_pct: float = 0.5) -> pd.Series:
    _require(df, ("open", "close"))
    prev_close = df["close"].shift(1)
    gap = _safe_divide(prev_close - df["open"], prev_close, fill=0.0) * 100.0
    return (gap >= float(min_gap_pct)).fillna(False)


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
    "marubozu",
    "spinning_top",
    "hammer",
    "inverted_hammer",
    "shooting_star",
    "hanging_man",
    "bullish_engulfing",
    "bearish_engulfing",
    "bullish_harami",
    "bearish_harami",
    "piercing_line",
    "dark_cloud_cover",
    "tweezer_top",
    "tweezer_bottom",
    "morning_star",
    "evening_star",
    "three_white_soldiers",
    "three_black_crows",
    "gap_up",
    "gap_down",
]
