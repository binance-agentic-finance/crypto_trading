"""Smart Money Concepts (SMC) – Liquidity analysis.

Three look-ahead-safe indicators built on confirmed fractal pivots:

* :func:`liquidity_sweep_detect`  — bearish / bullish liquidity sweeps.
* :func:`equal_highs_lows`        — equal-highs / equal-lows price clusters.
* :func:`premium_discount_zone`   — PREMIUM / DISCOUNT / EQUILIBRIUM labels.

Pivot confirmation carries a ``swing_lookback``-bar lag: a fractal pivot at
bar *i* is only considered known at bar *i + swing_lookback*, so no future
data is consumed at inference time.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ._utils import ensure_df, ensure_series, positive_int, safe_divide

try:
    from .smc_structure import fractal_pivot_high, fractal_pivot_low
except ImportError:  # pragma: no cover – available after smc_structure lands
    fractal_pivot_high = None  # type: ignore[assignment]
    fractal_pivot_low = None   # type: ignore[assignment]

__all__ = ["liquidity_sweep_detect", "equal_highs_lows", "premium_discount_zone"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _confirmed_pivots(raw: pd.Series, lookback: int) -> pd.Series:
    """Shift *raw* pivot series by *lookback* bars to enforce look-ahead safety.

    A fractal pivot at bar *i* requires observing bars up to *i + lookback*,
    so the value is shifted to appear at bar *i + lookback* — the earliest
    bar at which it could be acted upon in a live stream.
    """
    return ensure_series(raw).shift(lookback)


def _best_cluster(pivots: list[float], tol: float) -> Tuple[float, int]:
    """Return ``(mean_price, count)`` of the largest equal-level cluster.

    Anchors on the most recently added pivot.  Any existing pivot within
    *tol* (fractional, not percent) of the anchor is included.  Returns
    ``(nan, 0)`` when fewer than two levels match.
    """
    if not pivots:
        return float("nan"), 0

    anchor = pivots[-1]
    cluster = [p for p in pivots if abs(p - anchor) / anchor <= tol]
    if len(cluster) < 2:
        return float("nan"), 0
    return float(np.mean(cluster)), len(cluster)


# ---------------------------------------------------------------------------
# 1. Liquidity Sweep Detection
# ---------------------------------------------------------------------------

def liquidity_sweep_detect(
    df: pd.DataFrame,
    swing_lookback: int = 5,
) -> pd.DataFrame:
    """Detect bearish and bullish liquidity sweeps.

    *Bearish sweep*: the bar's high wicks above the last confirmed swing high
    but the close falls back below it — sellers absorbed the breakout.
    *Bullish sweep*: mirror logic on the low side.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    swing_lookback:
        Left/right bar count for fractal-pivot confirmation (look-ahead lag).

    Returns
    -------
    pd.DataFrame
        ``sweep_direction`` (``"BULL"`` / ``"BEAR"`` / NaN),
        ``sweep_level`` (swept swing extreme),
        ``sweep_wick_pct`` (wick as % of close price),
        ``sweep_strength`` (0–1, ``min(wick_pct / 2, 1.0)``).
    """
    df = ensure_df(df)
    swing_lookback = positive_int(swing_lookback, "swing_lookback")

    # Confirmed swing levels (lookahead-safe: shifted by swing_lookback).
    last_sh = _confirmed_pivots(
        fractal_pivot_high(df, swing_lookback), swing_lookback
    ).ffill()
    last_sl = _confirmed_pivots(
        fractal_pivot_low(df, swing_lookback), swing_lookback
    ).ffill()

    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # --- Sweep condition masks ---
    bear_mask = last_sh.notna() & (high > last_sh) & (close < last_sh)
    bull_mask = last_sl.notna() & (low  < last_sl) & (close > last_sl)

    # Wick sizes (full-bar; clipped to avoid negative values on bad data).
    bear_wick = (high  - close).clip(lower=0.0)
    bull_wick = (close - low  ).clip(lower=0.0)

    # Wick percentage (safe division — close should never be 0, but guard it).
    bear_wick_pct = safe_divide(bear_wick, close) * 100.0
    bull_wick_pct = safe_divide(bull_wick, close) * 100.0

    # --- Build output columns ---
    direction = pd.Series(np.nan, index=df.index, dtype=object)
    level     = pd.Series(np.nan, index=df.index, dtype=float)
    wick_pct  = pd.Series(np.nan, index=df.index, dtype=float)
    strength  = pd.Series(np.nan, index=df.index, dtype=float)

    # Bearish fills.
    direction[bear_mask] = "BEAR"
    level[bear_mask]     = last_sh[bear_mask]
    wick_pct[bear_mask]  = bear_wick_pct[bear_mask]
    strength[bear_mask]  = (bear_wick_pct[bear_mask] / 2.0).clip(upper=1.0)

    # Bullish fills (overwrite if both fire on the same bar — edge-case).
    direction[bull_mask] = "BULL"
    level[bull_mask]     = last_sl[bull_mask]
    wick_pct[bull_mask]  = bull_wick_pct[bull_mask]
    strength[bull_mask]  = (bull_wick_pct[bull_mask] / 2.0).clip(upper=1.0)

    return pd.DataFrame(
        {
            "sweep_direction": direction,
            "sweep_level":     level,
            "sweep_wick_pct":  wick_pct,
            "sweep_strength":  strength,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# 2. Equal Highs / Equal Lows
# ---------------------------------------------------------------------------

def equal_highs_lows(
    df: pd.DataFrame,
    swing_lookback: int = 5,
    tolerance_pct: float = 0.1,
    max_pivots: int = 20,
) -> pd.DataFrame:
    """Identify equal-highs and equal-lows clusters in a rolling pivot window.

    A rolling window of the last *max_pivots* confirmed pivot levels is
    maintained separately for highs and lows.  Whenever a new pivot is
    confirmed the newest pivot is used as an anchor; all pivots within
    *tolerance_pct* % of the anchor form a cluster.  The cluster's mean price
    and member count are reported and carried forward on subsequent bars.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    swing_lookback:
        Left/right bar count for fractal-pivot confirmation.
    tolerance_pct:
        Maximum relative distance (%) for two pivot levels to be equal.
    max_pivots:
        Maximum number of recent confirmed pivots retained per side.

    Returns
    -------
    pd.DataFrame
        ``eqh_level`` (float), ``eqh_count`` (int),
        ``eql_level`` (float), ``eql_count`` (int).
    """
    df         = ensure_df(df)
    swing_lookback = positive_int(swing_lookback, "swing_lookback")
    max_pivots     = positive_int(max_pivots,     "max_pivots")
    tol = float(tolerance_pct) / 100.0

    conf_ph = _confirmed_pivots(
        fractal_pivot_high(df, swing_lookback), swing_lookback
    )
    conf_pl = _confirmed_pivots(
        fractal_pivot_low(df, swing_lookback), swing_lookback
    )

    n = len(df)
    eqh_lvl = np.full(n, np.nan, dtype=float)
    eqh_cnt = np.zeros(n, dtype=np.int64)
    eql_lvl = np.full(n, np.nan, dtype=float)
    eql_cnt = np.zeros(n, dtype=np.int64)

    # Rolling pivot windows (FIFO, capped at max_pivots).
    ph_window: list[float] = []
    pl_window: list[float] = []

    # Last known cluster carried forward on bars with no new pivot.
    last_eqh_lvl: float = np.nan
    last_eqh_cnt: int   = 0
    last_eql_lvl: float = np.nan
    last_eql_cnt: int   = 0

    for i in range(n):
        new_ph = conf_ph.iloc[i]
        new_pl = conf_pl.iloc[i]

        # --- Update high pivot window ---
        if not np.isnan(new_ph):
            ph_window.append(float(new_ph))
            if len(ph_window) > max_pivots:
                ph_window.pop(0)
            last_eqh_lvl, last_eqh_cnt = _best_cluster(ph_window, tol)

        # --- Update low pivot window ---
        if not np.isnan(new_pl):
            pl_window.append(float(new_pl))
            if len(pl_window) > max_pivots:
                pl_window.pop(0)
            last_eql_lvl, last_eql_cnt = _best_cluster(pl_window, tol)

        eqh_lvl[i] = last_eqh_lvl
        eqh_cnt[i] = last_eqh_cnt
        eql_lvl[i] = last_eql_lvl
        eql_cnt[i] = last_eql_cnt

    return pd.DataFrame(
        {
            "eqh_level": eqh_lvl,
            "eqh_count": eqh_cnt,
            "eql_level": eql_lvl,
            "eql_count": eql_cnt,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# 3. Premium / Discount Zone
# ---------------------------------------------------------------------------

def premium_discount_zone(
    df: pd.DataFrame,
    swing_lookback: int = 5,
) -> pd.DataFrame:
    """Label each bar as PREMIUM, DISCOUNT, or EQUILIBRIUM.

    The *swing range* is bounded by the most recent confirmed swing high
    (range top) and the most recent confirmed swing low (range bottom).
    Equilibrium is the midpoint of that range.  A ±0.1 % tolerance band
    around equilibrium is labelled ``"EQUILIBRIUM"``; above is
    ``"PREMIUM"``; below is ``"DISCOUNT"``.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    swing_lookback:
        Left/right bar count for fractal-pivot confirmation.

    Returns
    -------
    pd.DataFrame
        ``premium_top``, ``discount_bottom``, ``equilibrium``,
        ``current_zone``.
    """
    df = ensure_df(df)
    swing_lookback = positive_int(swing_lookback, "swing_lookback")

    range_top = _confirmed_pivots(
        fractal_pivot_high(df, swing_lookback), swing_lookback
    ).ffill()
    range_bottom = _confirmed_pivots(
        fractal_pivot_low(df, swing_lookback), swing_lookback
    ).ffill()

    equilibrium = (range_top + range_bottom) / 2.0
    eq_band     = equilibrium * 0.001          # ±0.1 % tolerance around mid

    close      = df["close"]
    both_known = range_top.notna() & range_bottom.notna()

    zone = pd.Series(np.nan, index=df.index, dtype=object)
    zone[both_known & (close > equilibrium + eq_band)] = "PREMIUM"
    zone[both_known & (close < equilibrium - eq_band)] = "DISCOUNT"
    zone[
        both_known
        & (close >= equilibrium - eq_band)
        & (close <= equilibrium + eq_band)
    ] = "EQUILIBRIUM"

    return pd.DataFrame(
        {
            "premium_top":     range_top,
            "discount_bottom": range_bottom,
            "equilibrium":     equilibrium,
            "current_zone":    zone,
        },
        index=df.index,
    )
