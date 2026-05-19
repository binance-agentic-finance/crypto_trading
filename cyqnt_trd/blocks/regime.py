"""Market-regime classifiers.

Useful for building strategies that switch behaviour based on market
state (e.g. trend-following in trending markets, mean-reversion in
ranging markets — case #9 in the user dataset).

Examples
--------
>>> from cyqnt_trd.blocks import regime, indicators as ind
>>> adx_v, *_ = ind.adx(df, 14)
>>> regime_label = regime.adx_regime(adx_v)
>>> # regime_label is a series of "trend" / "range" / "transition"
"""

from __future__ import annotations

import pandas as pd

from ._utils import ensure_df, ensure_series, positive_int

__all__ = [
    "adx_regime",
    "volatility_regime",
    "range_regime",
    "is_range_regime",
    "trend_regime_ma",
]


def adx_regime(
    adx_series: pd.Series,
    trend_threshold: float = 25.0,
    range_threshold: float = 20.0,
) -> pd.Series:
    """Classify each bar as ``"trend"`` / ``"range"`` / ``"transition"``.

    * ``adx >= trend_threshold`` → trend
    * ``adx < range_threshold`` → range
    * else → transition (don't switch yet)
    """
    if range_threshold > trend_threshold:
        raise ValueError(
            f"range_threshold ({range_threshold}) must be <= trend_threshold ({trend_threshold})"
        )
    s = ensure_series(adx_series)
    out = pd.Series("transition", index=s.index)
    out = out.where(~(s >= trend_threshold), "trend")
    out = out.where(~(s < range_threshold), "range")
    return out


def volatility_regime(
    df: pd.DataFrame,
    period: int = 20,
    high_quantile: float = 0.8,
    low_quantile: float = 0.2,
) -> pd.Series:
    """Classify by realised volatility (rolling pct-change std-dev).

    Returns ``"high"`` / ``"normal"`` / ``"low"``.
    """
    df = ensure_df(df, required=("close",))
    period = positive_int(period, "period")
    rets = df["close"].pct_change()
    vol = rets.rolling(window=period, min_periods=period).std(ddof=0)
    high_t = vol.expanding(min_periods=period).quantile(high_quantile)
    low_t = vol.expanding(min_periods=period).quantile(low_quantile)
    out = pd.Series("normal", index=df.index)
    out = out.where(~(vol >= high_t), "high")
    out = out.where(~(vol <= low_t), "low")
    return out


def range_regime(
    df: pd.DataFrame, period: int = 20, max_range_pct: float = 0.03
) -> pd.Series:
    """Classify each bar as ``"range"`` / ``"trending"`` based on rolling high-low spread.

    Returns a string-valued Series for consistency with the other
    ``*_regime`` classifiers in this module.
    """
    flag = is_range_regime(df, period=period, max_range_pct=max_range_pct)
    out = pd.Series("trending", index=flag.index)
    out = out.where(~flag, "range")
    return out


def is_range_regime(
    df: pd.DataFrame, period: int = 20, max_range_pct: float = 0.03
) -> pd.Series:
    """Boolean series — True when the rolling high-low range is below threshold.

    Use this when you want a bool directly. For a string label use
    :func:`range_regime`.
    """
    from .conditions import consolidation_range  # local import to avoid cycles

    return consolidation_range(df, period=period, max_range_pct=max_range_pct)


def trend_regime_ma(
    df: pd.DataFrame, fast_ma: pd.Series, slow_ma: pd.Series
) -> pd.Series:
    """Classify trend by MA position: ``"uptrend"``/``"downtrend"``/``"sideways"``.

    * close > fast > slow → uptrend
    * close < fast < slow → downtrend
    * else → sideways
    """
    df = ensure_df(df, required=("close",))
    fast = ensure_series(fast_ma)
    slow = ensure_series(slow_ma)
    out = pd.Series("sideways", index=df.index)
    up = (df["close"] > fast) & (fast > slow)
    down = (df["close"] < fast) & (fast < slow)
    out = out.where(~up, "uptrend")
    out = out.where(~down, "downtrend")
    return out
