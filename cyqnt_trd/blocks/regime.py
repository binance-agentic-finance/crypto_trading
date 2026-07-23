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

OOS-survival filters
--------------------
Empirically derived from XRPUSDT 5m evolutionary discovery (2026-06-05).
These boolean filter helpers gate strategy entries to *avoid* regimes
where the strategy is statistically known to lose money:

>>> from cyqnt_trd.blocks import regime, indicators as ind
>>> atr_v = ind.atr(df, 14)
>>> # only allow entries when ATR is in the bottom 80% of its 200-bar history
>>> regime_ok = regime.atr_below_percentile(atr_v, window=200, percentile=0.80)
>>> long_signal = base_entry & regime_ok
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
    # OOS-survival filters
    "atr_below_percentile",
    "atr_above_percentile",
    "atr_ratio_below_threshold",
    "atr_ratio_above_threshold",
    "ma_slope_positive",
    "ma_slope_negative",
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


# ---------------------------------------------------------------------------
# OOS-survival filters
#
# Empirically validated from the XRPUSDT 5m evolutionary discovery run
# (2026-06-05). These return boolean Series to AND-combine with a base
# entry signal so the strategy stops trading in regimes where it is
# statistically known to bleed.
# ---------------------------------------------------------------------------


def atr_below_percentile(
    atr_series: pd.Series,
    window: int = 200,
    percentile: float = 0.80,
) -> pd.Series:
    """True when ATR is in the *lower* fraction of its rolling history.

    Use this to **block trades during extreme-volatility regimes** (the
    top 1 - percentile chunk). Empirically the single most important
    filter for OOS-survival on a 5m XRPUSDT crash window: filtering out
    the top 20% of ATR removed virtually all of the catastrophic OOS
    losses without sacrificing the bulk of in-sample edge.

    Parameters
    ----------
    atr_series : pd.Series
        ATR values from :func:`cyqnt_trd.blocks.indicators.atr`.
    window : int, default 200
        Rolling window for the percentile baseline.
    percentile : float in (0, 1), default 0.80
        Block bars where ATR exceeds this percentile of the rolling window.
    """
    window = positive_int(window, "window")
    if not 0 < percentile < 1:
        raise ValueError(f"percentile must be in (0, 1), got {percentile}")
    a = ensure_series(atr_series)
    threshold = a.rolling(window=window, min_periods=max(20, window // 2)).quantile(percentile)
    return (a < threshold).fillna(False).astype(bool)


def atr_above_percentile(
    atr_series: pd.Series,
    window: int = 200,
    percentile: float = 0.40,
) -> pd.Series:
    """True when ATR is in the *upper* fraction of its rolling history.

    Use this to **require minimum volatility** before trading — strategies
    that capture small moves get destroyed by fees + slippage during
    dead-vol periods. Pair with :func:`atr_below_percentile` to require
    a vol "Goldilocks zone".
    """
    window = positive_int(window, "window")
    if not 0 < percentile < 1:
        raise ValueError(f"percentile must be in (0, 1), got {percentile}")
    a = ensure_series(atr_series)
    threshold = a.rolling(window=window, min_periods=max(20, window // 2)).quantile(percentile)
    return (a > threshold).fillna(False).astype(bool)


def atr_ratio_below_threshold(
    atr_series: pd.Series,
    close_series: pd.Series,
    threshold: float = 0.012,
) -> pd.Series:
    """True when ATR / close < threshold (low *normalised* volatility).

    Equivalent to ``atr_below_percentile`` but with an explicit numeric
    threshold (e.g. 1.2% for crypto 5m). Cleaner for documentation and
    cross-symbol consistency since the threshold is regime-explicit.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    a = ensure_series(atr_series)
    c = ensure_series(close_series)
    return ((a / c) < threshold).fillna(False).astype(bool)


def atr_ratio_above_threshold(
    atr_series: pd.Series,
    close_series: pd.Series,
    threshold: float = 0.003,
) -> pd.Series:
    """True when ATR / close > threshold (minimum normalised volatility).

    Use as an ATR floor — strategies need enough vol to overcome fees.
    For 5m crypto a typical floor is 0.25-0.35% (i.e. 0.0025-0.0035).
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    a = ensure_series(atr_series)
    c = ensure_series(close_series)
    return ((a / c) > threshold).fillna(False).astype(bool)


def ma_slope_positive(
    ma_series: pd.Series,
    lookback: int = 5,
) -> pd.Series:
    """True when MA is rising over *lookback* bars.

    Faster than waiting for ``close > 200EMA``-style filters because it
    detects regime change as soon as the MA stops going up, not when
    price has already broken through. Recommended ``lookback`` for 5m:
    5 (fast kill switch) or 12 (medium-term confirmation).

    Parameters
    ----------
    ma_series : pd.Series
        Any moving average (EMA / SMA / HMA / DEMA …).
    lookback : int, default 5
        Number of bars to compare against. Larger = slower & more stable.
    """
    lookback = positive_int(lookback, "lookback")
    s = ensure_series(ma_series)
    return (s > s.shift(lookback)).fillna(False).astype(bool)


def ma_slope_negative(
    ma_series: pd.Series,
    lookback: int = 5,
) -> pd.Series:
    """True when MA is falling over *lookback* bars.

    Symmetric to :func:`ma_slope_positive`. Use as a short-only entry
    gate or as a long-side exit/avoid signal.
    """
    lookback = positive_int(lookback, "lookback")
    s = ensure_series(ma_series)
    return (s < s.shift(lookback)).fillna(False).astype(bool)
