"""Discrete feature detectors.

Indicators return continuous values. Detectors return boolean or categorical
signals that describe market structure, patterns, regime, or microstructure.
"""

from __future__ import annotations

import pandas as pd

from ai_pro_trading_library.library.conditions.atomic import (
    breakout_high,
    breakout_low,
    cross_above,
    cross_below,
    price_above,
    price_below,
    rsi_in_range,
    volume_surge,
)


def bullish_candle(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ("open", "close"))
    return (df["close"] > df["open"]).fillna(False)


def bearish_candle(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ("open", "close"))
    return (df["close"] < df["open"]).fillna(False)


def range_compression(df: pd.DataFrame, period: int = 20, max_range_pct: float = 0.03) -> pd.Series:
    _require_columns(df, ("high", "low"))
    high = df["high"].rolling(window=period, min_periods=period).max()
    low = df["low"].rolling(window=period, min_periods=period).min()
    mid = (high + low) / 2.0
    return (((high - low) / mid) <= float(max_range_pct)).fillna(False)


def trend_regime(close: pd.Series, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    """Return categorical regime labels: `uptrend`, `downtrend`, or `range`."""
    up = (close > fast_ma) & (fast_ma > slow_ma)
    down = (close < fast_ma) & (fast_ma < slow_ma)
    regime = pd.Series("range", index=close.index, dtype="object")
    regime.loc[up.fillna(False)] = "uptrend"
    regime.loc[down.fillna(False)] = "downtrend"
    return regime


def doji(df: pd.DataFrame, body_ratio: float = 0.1) -> pd.Series:
    """True when |close - open| / (high - low) <= body_ratio.

    Empty-range bars (high == low) are False.
    """
    _require_columns(df, ("open", "high", "low", "close"))
    spread = (df["high"] - df["low"]).astype(float)
    body = (df["close"] - df["open"]).abs().astype(float)
    safe = spread.where(spread > 0, other=float("nan"))
    ratio = body / safe
    return (ratio <= float(body_ratio)).fillna(False)


def engulfing(df: pd.DataFrame) -> pd.Series:
    """Categorical engulfing label per bar: `bullish`, `bearish`, or `none`.

    Bullish engulfing: prev bar bearish, current bar bullish AND
        current open <= prev close AND current close >= prev open.
    Bearish engulfing: mirror.
    """
    _require_columns(df, ("open", "close"))
    open_, close = df["open"].astype(float), df["close"].astype(float)
    prev_open, prev_close = open_.shift(1), close.shift(1)
    prev_bear = prev_close < prev_open
    prev_bull = prev_close > prev_open
    cur_bull = close > open_
    cur_bear = close < open_
    bullish = (prev_bear & cur_bull & (open_ <= prev_close) & (close >= prev_open)).fillna(False)
    bearish = (prev_bull & cur_bear & (open_ >= prev_close) & (close <= prev_open)).fillna(False)
    label = pd.Series("none", index=df.index, dtype="object")
    label.loc[bullish] = "bullish"
    label.loc[bearish] = "bearish"
    return label


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"dataframe is missing required columns: {missing}")


__all__ = [
    "bearish_candle",
    "breakout_high",
    "breakout_low",
    "bullish_candle",
    "cross_above",
    "cross_below",
    "doji",
    "engulfing",
    "price_above",
    "price_below",
    "range_compression",
    "rsi_in_range",
    "trend_regime",
    "volume_surge",
]

