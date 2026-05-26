"""Regime classifiers — full parity vs `cyqnt_trd.blocks.regime`."""

from __future__ import annotations

import pandas as pd

from ai_pro_trading_library.library.conditions.atomic import consolidation_range


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def adx_regime(
    adx_series: pd.Series,
    trend_threshold: float = 25.0,
    range_threshold: float = 20.0,
) -> pd.Series:
    """Categorise each bar as `trend` / `range` / `transition`."""
    if range_threshold > trend_threshold:
        raise ValueError(
            f"range_threshold ({range_threshold}) must be <= trend_threshold ({trend_threshold})"
        )
    out = pd.Series("transition", index=adx_series.index)
    out = out.where(~(adx_series >= float(trend_threshold)), "trend")
    out = out.where(~(adx_series < float(range_threshold)), "range")
    return out


def volatility_regime(
    df: pd.DataFrame,
    period: int = 20,
    high_quantile: float = 0.8,
    low_quantile: float = 0.2,
) -> pd.Series:
    """Categorise rolling realised vol as `high` / `normal` / `low`."""
    if "close" not in df.columns:
        raise ValueError("dataframe missing 'close'")
    period = _positive_int(period, "period")
    rets = df["close"].pct_change()
    vol = rets.rolling(window=period, min_periods=period).std(ddof=0)
    high_t = vol.expanding(min_periods=period).quantile(high_quantile)
    low_t = vol.expanding(min_periods=period).quantile(low_quantile)
    out = pd.Series("normal", index=df.index)
    out = out.where(~(vol >= high_t), "high")
    out = out.where(~(vol <= low_t), "low")
    return out


def is_range_regime(df: pd.DataFrame, period: int = 20, max_range_pct: float = 0.03) -> pd.Series:
    return consolidation_range(df, period=period, max_range_pct=max_range_pct)


def range_regime(df: pd.DataFrame, period: int = 20, max_range_pct: float = 0.03) -> pd.Series:
    """Categorise as `range` / `trending`."""
    flag = is_range_regime(df, period=period, max_range_pct=max_range_pct)
    out = pd.Series("trending", index=flag.index)
    out = out.where(~flag, "range")
    return out


def trend_regime_ma(df: pd.DataFrame, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    """Categorise as `uptrend` / `downtrend` / `sideways`."""
    if "close" not in df.columns:
        raise ValueError("dataframe missing 'close'")
    out = pd.Series("sideways", index=df.index)
    up = (df["close"] > fast_ma) & (fast_ma > slow_ma)
    down = (df["close"] < fast_ma) & (fast_ma < slow_ma)
    out = out.where(~up, "uptrend")
    out = out.where(~down, "downtrend")
    return out


__all__ = [
    "adx_regime",
    "is_range_regime",
    "range_regime",
    "trend_regime_ma",
    "volatility_regime",
]
