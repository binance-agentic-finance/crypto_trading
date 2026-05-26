"""Exit / stop-loss / take-profit helpers.

Provides both **functional** helpers (compute exit prices for a given
entry) and **declarative spec classes** (describe an exit rule that the
backtester or live engine can evaluate per bar).

Functional examples (single-trade math)::

    from cyqnt_trd.blocks import exit as ex
    stop_price = ex.fixed_stop_price(entry=100.0, pct=0.02, side="long")  # 98.0
    tp_price = ex.fixed_tp_price(entry=100.0, pct=0.05, side="long")      # 105.0
    rr = ex.risk_reward(entry=100.0, stop=98.0, tp=105.0)                 # 2.5

Declarative examples (multi-bar evaluation)::

    rule = ex.AtrTrailingStop(atr=ind.atr(df, 14), multiplier=2.2)
    exit_signal = rule.evaluate(df, entry_price=100.0, side="long")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ._utils import (
    crossover,
    crossunder,
    ensure_df,
    ensure_series,
    non_negative_float,
    positive_int,
    safe_divide,
)

__all__ = [
    # Functional helpers
    "fixed_stop_price",
    "fixed_tp_price",
    "atr_stop_price",
    "risk_reward",
    "passes_min_rr",
    "compute_partial_close_levels",
    # Declarative classes
    "ExitRule",
    "FixedStopLoss",
    "FixedTakeProfit",
    "AtrTrailingStop",
    "MaCrossExit",
    "SwingExit",
    "EmaTrailingTp",
    "MacdReversalExit",
    "GapExit",
    "TimeBasedExit",
    "BreakevenMove",
    "DualStop",
    "MultiTp",
]


_LONG = "long"
_SHORT = "short"


def _check_side(side: str) -> str:
    if side not in (_LONG, _SHORT):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    return side


# ---------------------------------------------------------------------------
# Single-trade math
# ---------------------------------------------------------------------------


def fixed_stop_price(entry: float, pct: float, side: str = _LONG) -> float:
    """Return the absolute stop-loss price for a *pct* (e.g. 0.02 = 2%) stop."""
    pct = non_negative_float(pct, "pct")
    side = _check_side(side)
    if side == _LONG:
        return float(entry) * (1.0 - pct)
    return float(entry) * (1.0 + pct)


def fixed_tp_price(entry: float, pct: float, side: str = _LONG) -> float:
    """Return the absolute take-profit price for a *pct* gain."""
    pct = non_negative_float(pct, "pct")
    side = _check_side(side)
    if side == _LONG:
        return float(entry) * (1.0 + pct)
    return float(entry) * (1.0 - pct)


def atr_stop_price(entry: float, atr_value: float, multiplier: float = 2.0, side: str = _LONG) -> float:
    """Return ATR-distance stop price."""
    atr_value = non_negative_float(atr_value, "atr_value")
    multiplier = non_negative_float(multiplier, "multiplier")
    side = _check_side(side)
    if side == _LONG:
        return float(entry) - multiplier * atr_value
    return float(entry) + multiplier * atr_value


def risk_reward(entry: float, stop: float, tp: float) -> float:
    """Return the reward-to-risk ratio: ``|tp - entry| / |entry - stop|``."""
    risk = abs(float(entry) - float(stop))
    reward = abs(float(tp) - float(entry))
    if risk == 0.0:
        return float("inf") if reward > 0 else 0.0
    return reward / risk


def passes_min_rr(entry: float, stop: float, tp: float, min_rr: float = 1.5) -> bool:
    """Return True if reward-to-risk ratio meets *min_rr*."""
    return risk_reward(entry, stop, tp) >= float(min_rr)


def compute_partial_close_levels(
    entry: float,
    targets: List[Tuple[float, float]],
    side: str = _LONG,
) -> List[Tuple[float, float]]:
    """Convert (gain_pct, close_ratio) pairs into (price, close_ratio) pairs.

    Example::

        compute_partial_close_levels(100.0, [(0.03, 0.5), (0.06, 0.5)], "long")
        # -> [(103.0, 0.5), (106.0, 0.5)]
    """
    side = _check_side(side)
    out: List[Tuple[float, float]] = []
    for gain_pct, close_ratio in targets:
        if not 0.0 < close_ratio <= 1.0:
            raise ValueError(f"close_ratio must be within (0, 1], got {close_ratio}")
        out.append((fixed_tp_price(entry, gain_pct, side), float(close_ratio)))
    return out


# ---------------------------------------------------------------------------
# Graduated take-profit (multi-step partial close at fixed return targets)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TakeProfitStep:
    """One stage of a graduated take-profit ladder.

    Parameters
    ----------
    target_price:
        Absolute price at which this stage triggers.
    close_fraction:
        Fraction of the *original* position to close at this stage.
        Sum of all close_fractions in a ladder must equal 1.0.
    """

    target_price: float
    close_fraction: float


def graduated_take_profit(
    entry_price: float,
    direction: str,
    target_returns: Tuple[float, ...],
    close_fractions: Tuple[float, ...],
) -> Tuple[TakeProfitStep, ...]:
    """Build a multi-stage take-profit ladder at fixed return targets.

    Sourced from ``ai_pro_trading_library.library.risk.targets.graduated_take_profit``.

    Parameters
    ----------
    entry_price:
        Trade entry price.
    direction:
        ``"long"`` or ``"short"`` (case-sensitive).
    target_returns:
        Tuple of fractional returns at which each stage triggers
        (e.g. ``(0.05, 0.10, 0.20)`` for 5 %, 10 %, 20 %).
    close_fractions:
        Tuple of fractions of the position to close at each stage. Must
        sum to 1.0 and have the same length as *target_returns*.

    Returns
    -------
    Tuple[TakeProfitStep, ...]
        Sorted ladder of (target_price, close_fraction) pairs.

    Examples
    --------
    >>> ladder = graduated_take_profit(
    ...     entry_price=100.0,
    ...     direction="long",
    ...     target_returns=(0.05, 0.10, 0.20),
    ...     close_fractions=(0.30, 0.30, 0.40),
    ... )
    >>> [round(s.target_price, 2) for s in ladder]
    [105.0, 110.0, 120.0]
    """
    if len(target_returns) != len(close_fractions):
        raise ValueError("target_returns and close_fractions must have the same length")
    if not 0.999 <= sum(close_fractions) <= 1.001:
        raise ValueError("close_fractions must sum to 1.0")
    if direction not in {"long", "short"}:
        raise ValueError("direction must be 'long' or 'short'")
    multiplier = 1.0 if direction == "long" else -1.0
    return tuple(
        TakeProfitStep(entry_price * (1.0 + multiplier * ret), fraction)
        for ret, fraction in zip(target_returns, close_fractions)
    )


# ---------------------------------------------------------------------------
# Declarative exit rules
# ---------------------------------------------------------------------------


@dataclass
class ExitRule:
    """Base class for declarative exit rules.

    Subclasses implement :meth:`evaluate` which returns a boolean
    ``pandas.Series`` aligned to *df* — True on bars where the rule
    fires (i.e. the position should be closed).

    The ``entry_index`` parameter is the index value of the bar on which
    the trade was opened. When provided, time-based / trailing rules
    only consider bars at-or-after that index. When omitted, they fall
    back to using the whole DataFrame (legacy behaviour) — emit a
    DeprecationWarning so that callers update.
    """

    def evaluate(
        self,
        df: pd.DataFrame,
        entry_price: float,
        side: str = _LONG,
        *,
        entry_index=None,
    ) -> pd.Series:  # pragma: no cover — abstract
        raise NotImplementedError


def _post_entry_mask(df: pd.DataFrame, entry_index) -> pd.Series:
    """Return a boolean mask: True for bars at-or-after the entry bar."""
    if entry_index is None:
        # Legacy behaviour: assume the first bar is the entry bar.
        # This is mathematically equivalent to passing entry_index=df.index[0]
        # but signals to the caller that they should be explicit.
        return pd.Series(True, index=df.index)
    return pd.Series(df.index >= entry_index, index=df.index)


@dataclass
class FixedStopLoss(ExitRule):
    """Stop the trade if price moves *pct* against entry."""

    pct: float

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        df = ensure_df(df, required=("close",))
        side = _check_side(side)
        stop = fixed_stop_price(entry_price, self.pct, side)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return ((df["close"] <= stop) & post).fillna(False)
        return ((df["close"] >= stop) & post).fillna(False)


@dataclass
class FixedTakeProfit(ExitRule):
    """Take profit when price moves *pct* in favour of entry."""

    pct: float

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        df = ensure_df(df, required=("close",))
        side = _check_side(side)
        tp = fixed_tp_price(entry_price, self.pct, side)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return ((df["close"] >= tp) & post).fillna(False)
        return ((df["close"] <= tp) & post).fillna(False)


@dataclass
class AtrTrailingStop(ExitRule):
    """Trail a stop *multiplier* * ATR below the running high (long) or above the low (short).

    The trailing peak is computed from the entry bar onward, not from
    the start of the DataFrame.
    """

    atr: pd.Series
    multiplier: float = 2.0

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        df = ensure_df(df, required=("high", "low", "close"))
        side = _check_side(side)
        atr_s = ensure_series(self.atr).reindex(df.index)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            highest = df["high"].where(post).cummax()
            stop = highest - self.multiplier * atr_s
            return ((df["close"] <= stop) & post).fillna(False)
        lowest = df["low"].where(post).cummin()
        stop = lowest + self.multiplier * atr_s
        return ((df["close"] >= stop) & post).fillna(False)


@dataclass
class MaCrossExit(ExitRule):
    """Exit on MA crossover (death cross for longs, golden for shorts)."""

    fast_ma: pd.Series
    slow_ma: pd.Series

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        side = _check_side(side)
        fast = ensure_series(self.fast_ma).reindex(df.index)
        slow = ensure_series(self.slow_ma).reindex(df.index)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return (crossunder(fast, slow) & post).fillna(False)
        return (crossover(fast, slow) & post).fillna(False)


@dataclass
class SwingExit(ExitRule):
    """Exit when price breaks the previous N-bar swing in the unfavourable direction."""

    lookback: int = 10

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        df = ensure_df(df, required=("high", "low", "close"))
        side = _check_side(side)
        n = positive_int(self.lookback, "lookback")
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            swing_low_ = df["low"].shift(1).rolling(window=n, min_periods=n).min()
            return ((df["close"] < swing_low_) & post).fillna(False)
        swing_high_ = df["high"].shift(1).rolling(window=n, min_periods=n).max()
        return ((df["close"] > swing_high_) & post).fillna(False)


@dataclass
class EmaTrailingTp(ExitRule):
    """Exit when price closes through an EMA (used as trailing TP for trend trades)."""

    ema: pd.Series

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        df = ensure_df(df, required=("close",))
        side = _check_side(side)
        ema_s = ensure_series(self.ema).reindex(df.index)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return ((df["close"] < ema_s) & post).fillna(False)
        return ((df["close"] > ema_s) & post).fillna(False)


@dataclass
class MacdReversalExit(ExitRule):
    """Exit when MACD crosses against the trade direction."""

    macd_line: pd.Series
    signal_line: pd.Series

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        side = _check_side(side)
        m = ensure_series(self.macd_line).reindex(df.index)
        s = ensure_series(self.signal_line).reindex(df.index)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return (crossunder(m, s) & post).fillna(False)
        return (crossover(m, s) & post).fillna(False)


@dataclass
class GapExit(ExitRule):
    """Exit on an adverse gap of at least *gap_pct* (e.g. 0.01 = 1%)."""

    gap_pct: float = 0.01

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        df = ensure_df(df, required=("open", "close"))
        side = _check_side(side)
        prev_close = df["close"].shift(1)
        gap = safe_divide(df["open"] - prev_close, prev_close, fill=0.0)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return ((gap <= -float(self.gap_pct)) & post).fillna(False)
        return ((gap >= float(self.gap_pct)) & post).fillna(False)


@dataclass
class TimeBasedExit(ExitRule):
    """Exit after holding for *max_bars* bars from the entry bar."""

    max_bars: int

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        n = positive_int(self.max_bars, "max_bars")
        out = pd.Series(False, index=df.index)
        if entry_index is None:
            # Legacy: count from the start of the DataFrame.
            if len(out) > n:
                out.iloc[n:] = True
        else:
            # Bars-since-entry: mark True on the (n+1)-th bar at or after entry.
            try:
                pos = df.index.get_loc(entry_index)
                target = pos + n
                if target < len(out):
                    out.iloc[target:] = True
            except KeyError:
                # entry_index not in df.index — leave all-False.
                pass
        return out


@dataclass
class BreakevenMove(ExitRule):
    """Move the stop to *lock_pct* once price moves *trigger_pct* in favour.

    Returns a boolean series; True when the breakeven stop is hit.
    Useful when chained inside a :class:`DualStop` to combine with a
    fixed initial stop.
    """

    trigger_pct: float
    lock_pct: float = 0.0

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        df = ensure_df(df, required=("high", "low", "close"))
        side = _check_side(side)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            triggered = df["high"].where(post).cummax() >= entry_price * (1.0 + float(self.trigger_pct))
            stop = entry_price * (1.0 + float(self.lock_pct))
            return (triggered & (df["close"] <= stop) & post).fillna(False)
        triggered = df["low"].where(post).cummin() <= entry_price * (1.0 - float(self.trigger_pct))
        stop = entry_price * (1.0 - float(self.lock_pct))
        return (triggered & (df["close"] >= stop) & post).fillna(False)


@dataclass
class DualStop(ExitRule):
    """Combine multiple stops; fire when *any* of them fires (logical OR)."""

    stops: List[ExitRule] = field(default_factory=list)

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        if not self.stops:
            raise ValueError("DualStop requires at least one stop rule")
        out = self.stops[0].evaluate(df, entry_price, side, entry_index=entry_index).fillna(False)
        for stop in self.stops[1:]:
            out = out | stop.evaluate(df, entry_price, side, entry_index=entry_index).fillna(False)
        return out


@dataclass
class MultiTp(ExitRule):
    """Multi-level take-profit. Returns True on each bar where a level *first* fires.

    Note: the framework expects the integration layer to track which TP
    levels have already triggered. This rule only computes the per-bar
    "level N triggered now" mask. For full close-by-close behaviour you
    typically combine with a position-tracking layer outside the rule.
    """

    targets: List[Tuple[float, float]]
    """List of ``(gain_pct, close_ratio)`` tuples."""

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        df = ensure_df(df, required=("close",))
        side = _check_side(side)
        levels = compute_partial_close_levels(entry_price, self.targets, side)
        post = _post_entry_mask(df, entry_index)
        any_hit = pd.Series(False, index=df.index)
        for tp_price, _ratio in levels:
            if side == _LONG:
                any_hit = any_hit | (df["close"] >= tp_price)
            else:
                any_hit = any_hit | (df["close"] <= tp_price)
        return (any_hit & post).fillna(False)
