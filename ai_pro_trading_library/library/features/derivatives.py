"""Derivative analytics — full parity vs `cyqnt_trd.blocks.derivatives`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _safe_divide(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


# ---------------------------------------------------------------------------
# OI
# ---------------------------------------------------------------------------


def oi_change_pct(oi: pd.Series, periods: int = 1) -> pd.Series:
    return oi.pct_change(periods=periods)


def oi_price_divergence(price: pd.Series, oi: pd.Series, lookback: int = 20) -> pd.Series:
    """Categorise price-vs-OI as bullish_buildup / bearish_buildup / long_squeeze / short_squeeze / none."""
    lookback = _positive_int(lookback, "lookback")
    p = price.astype(float).pct_change(lookback)
    o = oi.astype(float).pct_change(lookback)
    out = pd.Series("none", index=p.index)
    out = out.where(~((p > 0) & (o > 0)), "bullish_buildup")
    out = out.where(~((p < 0) & (o > 0)), "bearish_buildup")
    out = out.where(~((p > 0) & (o < 0)), "long_squeeze")
    out = out.where(~((p < 0) & (o < 0)), "short_squeeze")
    return out


# ---------------------------------------------------------------------------
# Long/short ratio
# ---------------------------------------------------------------------------


def long_short_ratio_state(
    ratio: pd.Series,
    crowded_threshold: float = 2.5,
    contrarian_threshold: float = 0.5,
) -> pd.Series:
    r = ratio.astype(float)
    out = pd.Series("neutral", index=r.index)
    out = out.where(~(r >= float(crowded_threshold)), "crowded_long")
    out = out.where(~(r <= float(contrarian_threshold)), "crowded_short")
    return out


def long_short_ratio_signal(
    ratio: pd.Series,
    crowded_threshold: float = 2.5,
    contrarian_threshold: float = 0.5,
) -> pd.Series:
    """Deprecated alias for `long_short_ratio_state`."""
    warnings.warn(
        "long_short_ratio_signal is deprecated; use long_short_ratio_state",
        DeprecationWarning,
        stacklevel=2,
    )
    return long_short_ratio_state(ratio, crowded_threshold, contrarian_threshold)


def taker_buy_sell_state(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    threshold: float = 1.5,
) -> pd.Series:
    b = buy_volume.astype(float)
    s = sell_volume.astype(float)
    ratio = _safe_divide(b, s, fill=1.0)
    out = pd.Series("balanced", index=b.index)
    out = out.where(~(ratio >= float(threshold)), "aggressive_buy")
    out = out.where(~(ratio <= 1.0 / float(threshold)), "aggressive_sell")
    return out


def taker_buy_sell_signal(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    threshold: float = 1.5,
) -> pd.Series:
    """Deprecated alias for `taker_buy_sell_state`."""
    warnings.warn(
        "taker_buy_sell_signal is deprecated; use taker_buy_sell_state",
        DeprecationWarning,
        stacklevel=2,
    )
    return taker_buy_sell_state(buy_volume, sell_volume, threshold)


# ---------------------------------------------------------------------------
# Cumulative volume delta
# ---------------------------------------------------------------------------


def cvd(buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
    return (buy_volume.astype(float) - sell_volume.astype(float)).cumsum()


def cvd_divergence(price: pd.Series, cvd_series: pd.Series, lookback: int = 20) -> pd.Series:
    lookback = _positive_int(lookback, "lookback")
    p = price.astype(float)
    c = cvd_series.astype(float)
    p_min = p.rolling(window=lookback, min_periods=lookback).min()
    p_max = p.rolling(window=lookback, min_periods=lookback).max()
    c_min = c.rolling(window=lookback, min_periods=lookback).min()
    c_max = c.rolling(window=lookback, min_periods=lookback).max()
    out = pd.Series("none", index=p.index)
    bullish = (p == p_min) & (c > c_min)
    bearish = (p == p_max) & (c < c_max)
    out = out.where(~bullish, "bullish")
    out = out.where(~bearish, "bearish")
    return out


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------


def basis(spot_close: pd.Series, futures_close: pd.Series) -> pd.Series:
    """(futures - spot) / spot * 10000 (basis points)."""
    s = spot_close.astype(float)
    f = futures_close.astype(float)
    return _safe_divide(f - s, s, fill=0.0) * 10_000.0


def basis_zscore(spot_close: pd.Series, futures_close: pd.Series, period: int = 96) -> pd.Series:
    period = _positive_int(period, "period")
    bps = basis(spot_close, futures_close)
    mu = bps.rolling(window=period, min_periods=period).mean()
    sd = bps.rolling(window=period, min_periods=period).std(ddof=0)
    return _safe_divide(bps - mu, sd, fill=0.0)


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------


def funding_rate_state(
    funding: pd.Series,
    high_threshold_bps: float = 5.0,
    low_threshold_bps: float = -5.0,
) -> pd.Series:
    """Categorise funding into bullish_squeeze / bearish_squeeze / neutral.

    Funding input is fraction (0.0001 == 1 bp); converted to bps internally.
    """
    f = funding.astype(float) * 10_000.0
    out = pd.Series("neutral", index=f.index)
    out = out.where(~(f >= float(high_threshold_bps)), "bullish_squeeze")
    out = out.where(~(f <= float(low_threshold_bps)), "bearish_squeeze")
    return out


# ---------------------------------------------------------------------------
# Liquidations
# ---------------------------------------------------------------------------


def liquidation_imbalance(
    long_liq_usd: pd.Series,
    short_liq_usd: pd.Series,
    lookback: int = 12,
) -> pd.Series:
    lookback = _positive_int(lookback, "lookback")
    L = long_liq_usd.astype(float).rolling(window=lookback, min_periods=1).sum()
    S = short_liq_usd.astype(float).rolling(window=lookback, min_periods=1).sum()
    total = L + S
    return _safe_divide(L - S, total, fill=0.0)


def liquidation_clusters(
    long_liq_usd: pd.Series,
    short_liq_usd: pd.Series,
    threshold_usd: float = 1_000_000.0,
    lookback: int = 12,
) -> tuple[pd.Series, pd.Series]:
    lookback = _positive_int(lookback, "lookback")
    L = long_liq_usd.astype(float).rolling(window=lookback, min_periods=1).sum()
    S = short_liq_usd.astype(float).rolling(window=lookback, min_periods=1).sum()
    long_cluster = L >= float(threshold_usd)
    short_cluster = S >= float(threshold_usd)
    return long_cluster.fillna(False), short_cluster.fillna(False)


__all__ = [
    "basis",
    "basis_zscore",
    "cvd",
    "cvd_divergence",
    "funding_rate_state",
    "liquidation_clusters",
    "liquidation_imbalance",
    "long_short_ratio_signal",
    "long_short_ratio_state",
    "oi_change_pct",
    "oi_price_divergence",
    "taker_buy_sell_signal",
    "taker_buy_sell_state",
]
