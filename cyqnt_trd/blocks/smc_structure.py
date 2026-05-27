"""Smart Money Concepts (SMC) structural analysis.

Provides fractal pivot detection, Fair Value Gap identification, Order Block
detection, and Break of Structure / Change of Character (BOS/CHoCH) analysis.

All public functions are pure and lookahead-safe: a value at index ``i``
depends only on data at indices ``<= i``.  Fractal pivot detection requires
``lookback`` future bars for confirmation; see individual docstrings.
"""

from __future__ import annotations

from typing import Tuple  # noqa: F401 — kept for consistency with package style

import numpy as np
import pandas as pd

from ._utils import (
    SeriesLike,
    ensure_df,
    ensure_series,
    positive_int,
    rolling_max,
    rolling_min,
    safe_divide,
)

__all__ = [
    "fractal_pivot_high",
    "fractal_pivot_low",
    "fair_value_gap",
    "order_block_detect",
    "bos_choch_detect",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fwd_max(series: SeriesLike, k: int) -> pd.Series:
    """Max of the *k* bars strictly **after** each position."""
    s = ensure_series(series)
    return pd.concat([s.shift(-j) for j in range(1, k + 1)], axis=1).max(axis=1)


def _fwd_min(series: SeriesLike, k: int) -> pd.Series:
    """Min of the *k* bars strictly **after** each position."""
    s = ensure_series(series)
    return pd.concat([s.shift(-j) for j in range(1, k + 1)], axis=1).min(axis=1)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def fractal_pivot_high(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Return confirmed fractal pivot highs.

    Bar *i* is a pivot high iff ``high[i]`` is **strictly greater** than each
    of the previous *lookback* bars **and** greater-or-equal to each of the
    next *lookback* bars.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    lookback:
        Number of bars on each side required for confirmation.

    Returns
    -------
    pd.Series
        ``high`` value at confirmed pivots, ``NaN`` elsewhere.

    Notes
    -----
    **Lag warning**: a pivot at bar *i* cannot be confirmed until bar
    *i + lookback* closes.  The value is placed at bar *i* for accurate
    historical labelling but is **not available in real-time** at bar *i*.
    """
    df = ensure_df(df)
    lookback = positive_int(lookback, "lookback")
    high = df["high"]

    # max of exactly *lookback* bars before i  →  rolling_max(.., lookback).shift(1)
    bwd = rolling_max(high, lookback).shift(1)
    fwd = _fwd_max(high, lookback)

    is_pivot = (high > bwd) & (high >= fwd)
    return high.where(is_pivot)


def fractal_pivot_low(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Return confirmed fractal pivot lows.

    Bar *i* is a pivot low iff ``low[i]`` is strictly less than each of the
    previous *lookback* bars and less-or-equal to the next *lookback* bars.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    lookback : int
        Bars on each side for confirmation.  Same ``lookback``-bar lag as
        :func:`fractal_pivot_high`.

    Returns
    -------
    pd.Series
        ``low`` value at confirmed pivots, ``NaN`` elsewhere.
    """
    df = ensure_df(df)
    lookback = positive_int(lookback, "lookback")
    low = df["low"]

    bwd = rolling_min(low, lookback).shift(1)
    fwd = _fwd_min(low, lookback)

    is_pivot = (low < bwd) & (low <= fwd)
    return low.where(is_pivot)


def fair_value_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Identify Fair Value Gap (FVG / imbalance) zones.

    For each bar *i* (i >= 2):

    * **Bullish FVG**: ``low[i] > high[i-2]``  →  zone ``[high[i-2], low[i]]``
    * **Bearish FVG**: ``high[i] < low[i-2]``  →  zone ``[high[i], low[i-2]]``

    Fully vectorised.  No lookahead: conditions reference only bars *i* and
    *i-2*.

    Parameters
    ----------
    df:
        OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame
        Same index as *df* with columns:

        * ``fvg_top``       – upper edge of the gap (NaN if no FVG)
        * ``fvg_bottom``    – lower edge of the gap
        * ``fvg_direction`` – ``"BULL"`` / ``"BEAR"`` / NaN
        * ``fvg_size_pct``  – ``(top - bottom) / close * 100``
    """
    df = ensure_df(df)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    high_2 = high.shift(2)  # high[i-2]
    low_2 = low.shift(2)    # low[i-2]

    bull_mask = low > high_2   # low[i] > high[i-2]
    bear_mask = high < low_2   # high[i] < low[i-2]

    fvg_top = pd.Series(np.nan, index=df.index, dtype=float)
    fvg_bottom = pd.Series(np.nan, index=df.index, dtype=float)
    fvg_direction = pd.Series(np.nan, index=df.index, dtype=object)

    # Bullish FVG: zone upper = low[i], lower = high[i-2]
    fvg_top = fvg_top.where(~bull_mask, low)
    fvg_bottom = fvg_bottom.where(~bull_mask, high_2)
    fvg_direction = fvg_direction.where(~bull_mask, "BULL")

    # Bearish FVG: zone upper = low[i-2], lower = high[i]
    # Logically mutually exclusive with bullish but bearish overrides if overlap
    fvg_top = fvg_top.where(~bear_mask, low_2)
    fvg_bottom = fvg_bottom.where(~bear_mask, high)
    fvg_direction = fvg_direction.where(~bear_mask, "BEAR")

    # Gap size as percentage of close; safe_divide guards against zero close
    fvg_size_pct = safe_divide((fvg_top - fvg_bottom) * 100.0, close)
    # Mask out bars with no FVG
    fvg_size_pct = fvg_size_pct.where(fvg_direction.notna())

    return pd.DataFrame(
        {
            "fvg_top": fvg_top,
            "fvg_bottom": fvg_bottom,
            "fvg_direction": fvg_direction,
            "fvg_size_pct": fvg_size_pct,
        },
        index=df.index,
    )


def order_block_detect(df: pd.DataFrame, swing_lookback: int = 5) -> pd.DataFrame:
    """Detect Order Blocks (OB) formed at Break-of-Structure events.

    Confirms fractal pivots, tracks the last swing high/low, then on a BOS
    walks backward to find the most recent opposing-colour candle as the OB.
    Broken levels are consumed so each level fires at most once.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    swing_lookback : int
        Lookback for fractal pivot confirmation (``swing_lookback``-bar lag).

    Returns
    -------
    pd.DataFrame
        Columns: ``ob_index`` (float bar index), ``ob_top``, ``ob_bottom``,
        ``ob_direction`` (``"BULL"``/``"BEAR"``/NaN).

    Notes
    -----
    OB mitigation tracking is **out of scope**; implement as a stateful
    filter on top of this output.
    """
    df = ensure_df(df)
    swing_lookback = positive_int(swing_lookback, "swing_lookback")
    n = len(df)

    pivot_high = fractal_pivot_high(df, swing_lookback)
    pivot_low = fractal_pivot_low(df, swing_lookback)

    opens_ = df["open"].values
    highs_ = df["high"].values
    lows_ = df["low"].values
    closes_ = df["close"].values
    ph = pivot_high.values
    pl = pivot_low.values

    ob_index_ = np.full(n, np.nan, dtype=float)
    ob_top_ = np.full(n, np.nan, dtype=float)
    ob_bottom_ = np.full(n, np.nan, dtype=float)
    ob_direction_ = np.full(n, np.nan, dtype=object)

    last_sh: float = np.nan
    last_sh_idx: int = -1
    last_sl: float = np.nan
    last_sl_idx: int = -1

    for i in range(n):
        # Absorb newly confirmed pivot highs/lows
        if not np.isnan(ph[i]):
            last_sh = ph[i]
            last_sh_idx = i
        if not np.isnan(pl[i]):
            last_sl = pl[i]
            last_sl_idx = i

        # --- Bullish BOS ---
        if not np.isnan(last_sh) and closes_[i] > last_sh:
            start = max(last_sh_idx + 1, 0)
            ob_bar = -1
            for j in range(i - 1, start - 1, -1):
                if closes_[j] < opens_[j]:  # bearish (red) candle
                    ob_bar = j
                    break
            if ob_bar >= 0:
                ob_index_[i] = float(ob_bar)
                ob_top_[i] = highs_[ob_bar]
                ob_bottom_[i] = lows_[ob_bar]
                ob_direction_[i] = "BULL"
            # Consume level — next signal requires a new confirmed swing high
            last_sh = np.nan
            last_sh_idx = -1

        # --- Bearish BOS ---
        elif not np.isnan(last_sl) and closes_[i] < last_sl:
            start = max(last_sl_idx + 1, 0)
            ob_bar = -1
            for j in range(i - 1, start - 1, -1):
                if closes_[j] > opens_[j]:  # bullish (green) candle
                    ob_bar = j
                    break
            if ob_bar >= 0:
                ob_index_[i] = float(ob_bar)
                ob_top_[i] = highs_[ob_bar]
                ob_bottom_[i] = lows_[ob_bar]
                ob_direction_[i] = "BEAR"
            # Consume level
            last_sl = np.nan
            last_sl_idx = -1

    return pd.DataFrame(
        {
            "ob_index": ob_index_,
            "ob_top": ob_top_,
            "ob_bottom": ob_bottom_,
            "ob_direction": ob_direction_,
        },
        index=df.index,
    )


def bos_choch_detect(df: pd.DataFrame, swing_lookback: int = 5) -> pd.DataFrame:
    """Detect Break of Structure (BOS) and Change of Character (CHoCH) events.

    Three-state trend machine (``"UP"``/``"DOWN"``/``"NEUTRAL"``), init
    ``"NEUTRAL"``.  On each bar:

    * ``close > last_swing_high`` → ``"BOS_BULL"`` (if UP) or ``"CHOCH_BULL"``
      (else, trend → UP).
    * ``close < last_swing_low``  → ``"BOS_BEAR"`` (if DOWN) or
      ``"CHOCH_BEAR"`` (else, trend → DOWN).

    Broken levels are consumed to prevent repeated signals.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    swing_lookback : int
        Lookback for fractal pivot confirmation.

    Returns
    -------
    pd.DataFrame
        Columns: ``structure_event``, ``broken_level``,
        ``trend_state`` (state *after* the bar's event).
    """
    df = ensure_df(df)
    swing_lookback = positive_int(swing_lookback, "swing_lookback")
    n = len(df)

    pivot_high = fractal_pivot_high(df, swing_lookback)
    pivot_low = fractal_pivot_low(df, swing_lookback)

    closes_ = df["close"].values
    ph = pivot_high.values
    pl = pivot_low.values

    structure_event_ = np.full(n, np.nan, dtype=object)
    broken_level_ = np.full(n, np.nan, dtype=float)
    trend_state_ = np.full(n, "NEUTRAL", dtype=object)

    last_sh: float = np.nan
    last_sl: float = np.nan
    trend: str = "NEUTRAL"

    for i in range(n):
        # Absorb confirmed pivots
        if not np.isnan(ph[i]):
            last_sh = ph[i]
        if not np.isnan(pl[i]):
            last_sl = pl[i]

        # Evaluate structural events — bullish break takes priority
        if not np.isnan(last_sh) and closes_[i] > last_sh:
            if trend == "UP":
                structure_event_[i] = "BOS_BULL"
            else:
                structure_event_[i] = "CHOCH_BULL"
                trend = "UP"
            broken_level_[i] = last_sh
            last_sh = np.nan  # consume level

        elif not np.isnan(last_sl) and closes_[i] < last_sl:
            if trend == "DOWN":
                structure_event_[i] = "BOS_BEAR"
            else:
                structure_event_[i] = "CHOCH_BEAR"
                trend = "DOWN"
            broken_level_[i] = last_sl
            last_sl = np.nan  # consume level

        trend_state_[i] = trend

    return pd.DataFrame(
        {
            "structure_event": structure_event_,
            "broken_level": broken_level_,
            "trend_state": trend_state_,
        },
        index=df.index,
    )
