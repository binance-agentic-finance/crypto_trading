"""Stateful exit-rule classes — L1 parity vs `cyqnt_trd.blocks.exit`.

Each rule's `evaluate(df, entry_price, side, *, entry_index=None)` returns a
boolean Series aligned to `df.index`; True on bars where the rule fires.
The `entry_index` keyword restricts the rule to bars at-or-after the entry
bar; when omitted, behaviour matches the blocks legacy (whole DataFrame).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_pro_trading_library.library.conditions.atomic import cross_above, cross_below
from ai_pro_trading_library.library.risk.targets import (
    compute_partial_close_levels,
    fixed_stop_price,
    fixed_tp_price,
)


_LONG = "long"
_SHORT = "short"


def _check_side(side: str) -> str:
    side = str(side).lower()
    if side not in {_LONG, _SHORT}:
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    return side


def _post_entry_mask(df: pd.DataFrame, entry_index) -> pd.Series:
    if entry_index is None:
        return pd.Series(True, index=df.index)
    return pd.Series(df.index >= entry_index, index=df.index)


def _safe_divide(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


@dataclass
class ExitRule:
    """Base class for declarative exit rules."""

    def evaluate(
        self,
        df: pd.DataFrame,
        entry_price: float,
        side: str = _LONG,
        *,
        entry_index=None,
    ) -> pd.Series:
        raise NotImplementedError


@dataclass
class FixedStopLoss(ExitRule):
    pct: float

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        if "close" not in df.columns:
            raise ValueError("dataframe missing 'close' column")
        side = _check_side(side)
        stop = fixed_stop_price(entry_price, self.pct, side)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return ((df["close"] <= stop) & post).fillna(False)
        return ((df["close"] >= stop) & post).fillna(False)


@dataclass
class FixedTakeProfit(ExitRule):
    pct: float

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        if "close" not in df.columns:
            raise ValueError("dataframe missing 'close' column")
        side = _check_side(side)
        tp = fixed_tp_price(entry_price, self.pct, side)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return ((df["close"] >= tp) & post).fillna(False)
        return ((df["close"] <= tp) & post).fillna(False)


@dataclass
class AtrTrailingStop(ExitRule):
    atr: pd.Series
    multiplier: float = 2.0

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        for col in ("high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"dataframe missing '{col}' column")
        side = _check_side(side)
        atr_s = self.atr.reindex(df.index)
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
    fast_ma: pd.Series
    slow_ma: pd.Series

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        side = _check_side(side)
        fast = self.fast_ma.reindex(df.index)
        slow = self.slow_ma.reindex(df.index)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return (cross_below(fast, slow) & post).fillna(False)
        return (cross_above(fast, slow) & post).fillna(False)


@dataclass
class SwingExit(ExitRule):
    lookback: int = 10

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        for col in ("high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"dataframe missing '{col}' column")
        side = _check_side(side)
        n = int(self.lookback)
        if n <= 0:
            raise ValueError("lookback must be positive")
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            swing_low_ = df["low"].shift(1).rolling(window=n, min_periods=n).min()
            return ((df["close"] < swing_low_) & post).fillna(False)
        swing_high_ = df["high"].shift(1).rolling(window=n, min_periods=n).max()
        return ((df["close"] > swing_high_) & post).fillna(False)


@dataclass
class EmaTrailingTp(ExitRule):
    ema: pd.Series

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        if "close" not in df.columns:
            raise ValueError("dataframe missing 'close' column")
        side = _check_side(side)
        ema_s = self.ema.reindex(df.index)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return ((df["close"] < ema_s) & post).fillna(False)
        return ((df["close"] > ema_s) & post).fillna(False)


@dataclass
class MacdReversalExit(ExitRule):
    macd_line: pd.Series
    signal_line: pd.Series

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        side = _check_side(side)
        m = self.macd_line.reindex(df.index)
        s = self.signal_line.reindex(df.index)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return (cross_below(m, s) & post).fillna(False)
        return (cross_above(m, s) & post).fillna(False)


@dataclass
class GapExit(ExitRule):
    gap_pct: float = 0.01

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        if not {"open", "close"}.issubset(df.columns):
            raise ValueError("dataframe missing 'open'/'close' columns")
        side = _check_side(side)
        prev_close = df["close"].shift(1)
        gap = _safe_divide(df["open"] - prev_close, prev_close, fill=0.0)
        post = _post_entry_mask(df, entry_index)
        if side == _LONG:
            return ((gap <= -float(self.gap_pct)) & post).fillna(False)
        return ((gap >= float(self.gap_pct)) & post).fillna(False)


@dataclass
class TimeBasedExit(ExitRule):
    max_bars: int

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        n = int(self.max_bars)
        if n <= 0:
            raise ValueError("max_bars must be positive")
        out = pd.Series(False, index=df.index)
        if entry_index is None:
            if len(out) > n:
                out.iloc[n:] = True
        else:
            try:
                pos = df.index.get_loc(entry_index)
                target = pos + n
                if target < len(out):
                    out.iloc[target:] = True
            except KeyError:
                pass
        return out


@dataclass
class BreakevenMove(ExitRule):
    trigger_pct: float
    lock_pct: float = 0.0

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        for col in ("high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"dataframe missing '{col}' column")
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
    stops: list = field(default_factory=list)

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        if not self.stops:
            raise ValueError("DualStop requires at least one stop rule")
        out = self.stops[0].evaluate(df, entry_price, side, entry_index=entry_index).fillna(False)
        for stop in self.stops[1:]:
            out = out | stop.evaluate(df, entry_price, side, entry_index=entry_index).fillna(False)
        return out


@dataclass
class MultiTp(ExitRule):
    targets: list = field(default_factory=list)

    def evaluate(self, df, entry_price, side=_LONG, *, entry_index=None):
        if "close" not in df.columns:
            raise ValueError("dataframe missing 'close' column")
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


__all__ = [
    "AtrTrailingStop",
    "BreakevenMove",
    "DualStop",
    "EmaTrailingTp",
    "ExitRule",
    "FixedStopLoss",
    "FixedTakeProfit",
    "GapExit",
    "MaCrossExit",
    "MacdReversalExit",
    "MultiTp",
    "SwingExit",
    "TimeBasedExit",
]
