"""Crypto-derivatives-specific building blocks.

Functions for working with futures-only data: open interest, funding
rate, top long/short ratio, taker buy/sell volume, basis, CVD, and
liquidation streams.

These are *transformation* helpers that take pre-fetched series — to
download the raw data from Binance, see :mod:`cyqnt_trd.blocks.data`.

Examples
--------
>>> from cyqnt_trd.blocks import derivatives as deriv, data
>>> oi = data.fetch_oi("BTCUSDT", period="5m", limit=288)
>>> divergence = deriv.oi_price_divergence(price=df["close"], oi=oi["oi"])
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ._utils import ensure_series, positive_int, safe_divide

__all__ = [
    "oi_change_pct",
    "oi_price_divergence",
    "long_short_ratio_state",
    "taker_buy_sell_state",
    "long_short_ratio_signal",  # deprecated alias
    "taker_buy_sell_signal",    # deprecated alias
    "cvd",
    "cvd_divergence",
    "basis",
    "basis_zscore",
    "funding_rate_state",
    "liquidation_imbalance",
    "liquidation_clusters",
]


# ---------------------------------------------------------------------------
# Open Interest
# ---------------------------------------------------------------------------


def oi_change_pct(oi: pd.Series, periods: int = 1) -> pd.Series:
    """Percent change of open interest over *periods* bars."""
    return ensure_series(oi).pct_change(periods=periods)


def oi_price_divergence(price: pd.Series, oi: pd.Series, lookback: int = 20) -> pd.Series:
    """Detect price-vs-OI divergence using N-bar slopes.

    Returns a string-valued ``pandas.Series`` with possible values:

    * ``"bullish_buildup"`` — price up, OI up (sustainable trend)
    * ``"bearish_buildup"`` — price down, OI up (new shorts, downside continuation)
    * ``"long_squeeze"`` — price up, OI down (shorts covering, may stall)
    * ``"short_squeeze"`` — price down, OI down (longs covering, may stall)
    * ``"none"`` — flat or insufficient history
    """
    lookback = positive_int(lookback, "lookback")
    p = ensure_series(price).astype(float).pct_change(lookback)
    o = ensure_series(oi).astype(float).pct_change(lookback)
    out = pd.Series("none", index=p.index)
    out = out.where(~((p > 0) & (o > 0)), "bullish_buildup")
    out = out.where(~((p < 0) & (o > 0)), "bearish_buildup")
    out = out.where(~((p > 0) & (o < 0)), "long_squeeze")
    out = out.where(~((p < 0) & (o < 0)), "short_squeeze")
    return out


# ---------------------------------------------------------------------------
# Long/Short ratio
# ---------------------------------------------------------------------------


def long_short_ratio_state(
    ratio: pd.Series,
    crowded_threshold: float = 2.5,
    contrarian_threshold: float = 0.5,
) -> pd.Series:
    """Categorise a long/short ratio series into ``crowded_long`` / ``crowded_short`` / ``neutral``.

    Trader sentiment is contrarian: when the ratio is *very high*, retail
    is over-leveraged long and a flush is more likely; when *very low*,
    a short squeeze is more likely.

    Returns a string-valued Series.
    """
    r = ensure_series(ratio).astype(float)
    out = pd.Series("neutral", index=r.index)
    out = out.where(~(r >= crowded_threshold), "crowded_long")
    out = out.where(~(r <= contrarian_threshold), "crowded_short")
    return out


def long_short_ratio_signal(
    ratio: pd.Series,
    crowded_threshold: float = 2.5,
    contrarian_threshold: float = 0.5,
) -> pd.Series:
    """Deprecated alias for :func:`long_short_ratio_state`. Kept for backward compatibility."""
    import warnings

    warnings.warn(
        "long_short_ratio_signal is deprecated; use long_short_ratio_state instead "
        "(returns a string Series).",
        DeprecationWarning,
        stacklevel=2,
    )
    return long_short_ratio_state(ratio, crowded_threshold, contrarian_threshold)


def taker_buy_sell_state(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    threshold: float = 1.5,
) -> pd.Series:
    """Categorise taker buy/sell flow into ``aggressive_buy``/``aggressive_sell``/``balanced``.

    Returns a string Series. ``threshold`` is the ratio above which one
    side dominates.
    """
    b = ensure_series(buy_volume).astype(float)
    s = ensure_series(sell_volume).astype(float)
    ratio = safe_divide(b, s, fill=1.0)
    out = pd.Series("balanced", index=b.index)
    out = out.where(~(ratio >= threshold), "aggressive_buy")
    out = out.where(~(ratio <= 1.0 / threshold), "aggressive_sell")
    return out


def taker_buy_sell_signal(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    threshold: float = 1.5,
) -> pd.Series:
    """Deprecated alias for :func:`taker_buy_sell_state`. Kept for backward compatibility."""
    import warnings

    warnings.warn(
        "taker_buy_sell_signal is deprecated; use taker_buy_sell_state instead "
        "(returns a string Series).",
        DeprecationWarning,
        stacklevel=2,
    )
    return taker_buy_sell_state(buy_volume, sell_volume, threshold)


# ---------------------------------------------------------------------------
# Cumulative Volume Delta
# ---------------------------------------------------------------------------


def cvd(buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
    """Cumulative Volume Delta = ``cumsum(buy - sell)``."""
    b = ensure_series(buy_volume).astype(float)
    s = ensure_series(sell_volume).astype(float)
    return (b - s).cumsum()


def cvd_divergence(price: pd.Series, cvd_series: pd.Series, lookback: int = 20) -> pd.Series:
    """Bullish/bearish CVD divergence vs price over a rolling window.

    Returns a string-valued ``pandas.Series`` with possible values:

    * ``"bullish"``  — price makes lower low, CVD makes higher low
    * ``"bearish"``  — price makes higher high, CVD makes lower high
    * ``"none"``     — no divergence
    """
    lookback = positive_int(lookback, "lookback")
    p = ensure_series(price).astype(float)
    c = ensure_series(cvd_series).astype(float)
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
# Basis (perpetual vs spot)
# ---------------------------------------------------------------------------


def basis(spot_close: pd.Series, futures_close: pd.Series) -> pd.Series:
    """Basis in basis points: ``(futures - spot) / spot * 10_000``."""
    s = ensure_series(spot_close).astype(float)
    f = ensure_series(futures_close).astype(float)
    return safe_divide(f - s, s, fill=0.0) * 10_000.0


def basis_zscore(spot_close: pd.Series, futures_close: pd.Series, period: int = 96) -> pd.Series:
    """Rolling z-score of the basis (1d on 15m bars when period=96)."""
    period = positive_int(period, "period")
    bps = basis(spot_close, futures_close)
    mu = bps.rolling(window=period, min_periods=period).mean()
    sd = bps.rolling(window=period, min_periods=period).std(ddof=0)
    return safe_divide(bps - mu, sd, fill=0.0)


# ---------------------------------------------------------------------------
# Funding rate
# ---------------------------------------------------------------------------


def funding_rate_state(
    funding: pd.Series,
    high_threshold_bps: float = 5.0,
    low_threshold_bps: float = -5.0,
) -> pd.Series:
    """Categorise funding into ``bullish_squeeze`` / ``bearish_squeeze`` / ``neutral``.

    Note: Binance returns funding as a fraction (e.g. 0.0001 == 1 bp);
    we convert to bps. ``bullish_squeeze`` (high positive funding) means
    longs are paying — over-extended longs, contrarian short bias.
    """
    f = ensure_series(funding).astype(float) * 10_000.0  # to bps
    out = pd.Series("neutral", index=f.index)
    out = out.where(~(f >= high_threshold_bps), "bullish_squeeze")
    out = out.where(~(f <= low_threshold_bps), "bearish_squeeze")
    return out


# ---------------------------------------------------------------------------
# Liquidations
# ---------------------------------------------------------------------------


def liquidation_imbalance(
    long_liq_usd: pd.Series, short_liq_usd: pd.Series, lookback: int = 12
) -> pd.Series:
    """Rolling long-liq vs short-liq imbalance ratio.

    Returns a value in ``[-1, 1]``:

    * ``+1`` — only long liquidations (cascade of long flushes, can mark a bottom)
    * ``-1`` — only short liquidations (short squeeze, can mark a top)
    * ``0``  — balanced
    """
    lookback = positive_int(lookback, "lookback")
    L = ensure_series(long_liq_usd).astype(float).rolling(window=lookback, min_periods=1).sum()
    S = ensure_series(short_liq_usd).astype(float).rolling(window=lookback, min_periods=1).sum()
    total = L + S
    return safe_divide(L - S, total, fill=0.0)


def liquidation_clusters(
    long_liq_usd: pd.Series,
    short_liq_usd: pd.Series,
    threshold_usd: float = 1_000_000.0,
    lookback: int = 12,
) -> Tuple[pd.Series, pd.Series]:
    """Identify long- and short-liquidation cluster events over rolling sums.

    Returns
    -------
    (long_cluster, short_cluster) : tuple of boolean ``pandas.Series``
    """
    lookback = positive_int(lookback, "lookback")
    L = ensure_series(long_liq_usd).astype(float).rolling(window=lookback, min_periods=1).sum()
    S = ensure_series(short_liq_usd).astype(float).rolling(window=lookback, min_periods=1).sum()
    long_cluster = L >= threshold_usd
    short_cluster = S >= threshold_usd
    return long_cluster.fillna(False), short_cluster.fillna(False)
