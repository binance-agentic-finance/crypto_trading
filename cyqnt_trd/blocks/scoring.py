"""Multi-factor scoring system.

A reusable scoring engine — extremely common in the user dataset. Users
typically describe rules like:

    Trigger if:
      4H RSI in 60-75:  +1.5
      4H EMA20 > EMA50: +1.0
      Volume >= 2x avg: +1.0
      Daily EMA bearish: -1.0
      Total >= 3.0 → BUY

Use :class:`ScoringSystem` to register rules with optional weights and
caps, then evaluate against each bar of a DataFrame.

Examples
--------
>>> from cyqnt_trd.blocks import scoring, indicators as ind, conditions as cond
>>> rsi = ind.rsi(df["close"], 14)
>>> ma20 = ind.sma(df["close"], 20)
>>> ma50 = ind.sma(df["close"], 50)
>>> system = scoring.ScoringSystem()
>>> system.add_rule("rsi_zone", cond.rsi_in_range(rsi, 50, 70), weight=2.0)
>>> system.add_rule("trend",     cond.price_above_ma(df, ma20), weight=1.5)
>>> system.add_rule("trend_bear", cond.price_below_ma(df, ma50), weight=-1.0)
>>> entry_signal = system.signal(threshold=2.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ._utils import ensure_series

__all__ = ["Rule", "ScoringSystem"]


@dataclass
class Rule:
    """A single scoring rule."""

    name: str
    condition: pd.Series
    weight: float = 1.0
    cap: Optional[float] = None
    """Maximum positive contribution (e.g. avoid double-counting). None = no cap."""


class ScoringSystem:
    """Compose multiple weighted boolean rules into a numeric score.

    Rules are evaluated independently and summed. Use :meth:`evaluate`
    to obtain the per-bar numeric score and :meth:`signal` to obtain a
    boolean entry signal at a given threshold.
    """

    def __init__(self) -> None:
        self._rules: List[Rule] = []

    def add_rule(
        self,
        name: str,
        condition: pd.Series,
        weight: float = 1.0,
        cap: Optional[float] = None,
    ) -> "ScoringSystem":
        """Register a new rule. Returns ``self`` for chaining."""
        self._rules.append(
            Rule(name=name, condition=condition, weight=float(weight), cap=cap)
        )
        return self

    @property
    def rules(self) -> Tuple[Rule, ...]:
        return tuple(self._rules)

    def evaluate(self) -> pd.Series:
        """Sum weighted contributions across all rules."""
        if not self._rules:
            raise ValueError("no rules registered — call add_rule first")
        total: Optional[pd.Series] = None
        for rule in self._rules:
            cond = ensure_series(rule.condition).fillna(False).astype(bool)
            contribution = cond.astype(float) * float(rule.weight)
            if rule.cap is not None:
                contribution = contribution.clip(upper=float(rule.cap))
            if total is None:
                total = contribution
            else:
                total = total.add(contribution, fill_value=0.0)
        assert total is not None
        return total

    def signal(self, threshold: float) -> pd.Series:
        """Return a boolean series: True where score >= threshold."""
        return self.evaluate() >= float(threshold)

    def breakdown(self) -> pd.DataFrame:
        """Return a DataFrame with one column per rule's contribution.

        Useful for debugging which rules fire on which bars.
        """
        if not self._rules:
            raise ValueError("no rules registered")
        cols: Dict[str, pd.Series] = {}
        for rule in self._rules:
            cond = ensure_series(rule.condition).fillna(False).astype(bool)
            contribution = cond.astype(float) * float(rule.weight)
            if rule.cap is not None:
                contribution = contribution.clip(upper=float(rule.cap))
            cols[rule.name] = contribution
        return pd.DataFrame(cols)
