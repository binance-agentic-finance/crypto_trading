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

Module-level combinator functions (pandas-vectorized, L3-01 to L3-05)
----------------------------------------------------------------------
These are direct ports of ``atomic.scoring.combinators`` that accept
pandas Series (one value per bar) instead of atomic Signal objects:

* :func:`additive_combine` — sum signed contributions
* :func:`weighted_composite` — per-signal multipliers
* :func:`sequential_filter` — count signals passing strength threshold
* :func:`majority_vote` — (bullish – bearish) / total per bar
* :func:`hierarchical_combine` — tier-weighted nested scoring

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

import numpy as np
import pandas as pd

from ._utils import ensure_series

__all__ = [
    "Rule",
    "ScoringSystem",
    # Module-level combinator functions (pandas-vectorized, L3-01 to L3-05)
    "additive_combine",
    "weighted_composite",
    "sequential_filter",
    "majority_vote",
    "hierarchical_combine",
]


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

    # ------------------------------------------------------------------
    # Combinator methods (L3-01 to L3-05) — ported from atomic.scoring
    # ------------------------------------------------------------------

    def additive_combine_with(self, other: "ScoringSystem") -> pd.Series:
        """Sum this system's total score with another ScoringSystem's score.

        Analogous to calling ``additive_combine({...})`` on the merged rule
        contributions of two separate systems.

        Args:
            other: A second :class:`ScoringSystem` whose rules have been
                   evaluated on the same-length Series.

        Returns:
            pd.Series with per-bar sum of both systems' scores, rounded to 2 dp.
        """
        return self.evaluate().add(other.evaluate(), fill_value=0.0).round(2)

    def sequential_filter(self, min_strength: float = 0.3) -> pd.Series:
        """Count of rules whose weighted |contribution| >= *min_strength* per bar.

        Analogous to ``atomic.scoring.combinators.sequential_filter``: for each
        bar, counts how many "strong-enough" signals pass the threshold.

        Args:
            min_strength: minimum absolute contribution for a rule to count.
                          Defaults to 0.3.

        Returns:
            pd.Series of float counts (0 … n_rules).
        """
        bd = self.breakdown()
        passes = (bd.abs() >= float(min_strength)).sum(axis=1)
        return passes.astype(float)

    def majority_vote(self) -> pd.Series:
        """Net direction vote: (positive_rules − negative_rules) / total per bar.

        Analogous to ``atomic.scoring.combinators.majority_vote``.
        Returns values in **[−1.0, 1.0]**.
        Positive = bullish majority, negative = bearish majority.

        Returns:
            pd.Series rounded to 2 dp.
        """
        bd = self.breakdown()
        if bd.empty:
            raise ValueError("no rules registered")
        total = len(bd.columns)
        bullish = (bd > 0).sum(axis=1)
        bearish = (bd < 0).sum(axis=1)
        return ((bullish - bearish) / total).round(2)

    def hierarchical(
        self,
        tiers: List[Tuple[str, List[str], float]],
    ) -> pd.Series:
        """Tier-weighted scoring using named subsets of registered rules.

        Analogous to ``atomic.scoring.combinators.hierarchical_combine``.

        Args:
            tiers: list of ``(tier_name, rule_names, tier_weight)`` tuples.
                   *rule_names* must match names given to :meth:`add_rule`.

        Returns:
            pd.Series of tier-weighted aggregate scores, rounded to 2 dp.

        Raises:
            ValueError: if none of the tier rule_names match any registered rule.
        """
        bd = self.breakdown()
        total: Optional[pd.Series] = None
        for tier_name, rule_names, tier_weight in tiers:
            cols = [r for r in rule_names if r in bd.columns]
            if not cols:
                continue
            tier_score = bd[cols].sum(axis=1)
            weighted = tier_score * float(tier_weight)
            total = weighted if total is None else total.add(weighted, fill_value=0.0)
        if total is None:
            raise ValueError(
                "no matching rules found for any tier — check rule names passed to hierarchical()"
            )
        return total.round(2)


# ---------------------------------------------------------------------------
# Module-level combinator functions (pandas-vectorized, L3-01 to L3-05)
# ---------------------------------------------------------------------------


def additive_combine(signals: Dict[str, pd.Series]) -> pd.Series:
    """Additive sum of signed score contributions per bar.

    Port of ``atomic.scoring.combinators.additive_combine()``.

    Args:
        signals: dict mapping signal name → signed contribution Series.
                 Positive values = bullish contribution,
                 negative values = bearish contribution.

    Returns:
        pd.Series of per-bar aggregated scores, rounded to 2 dp.

    Raises:
        ValueError: if *signals* is empty.

    Example::

        sigs = {
            "rsi": pd.Series([10.0, -5.0, 0.0]),
            "ema": pd.Series([ 5.0,  5.0, 2.0]),
        }
        additive_combine(sigs)  # → Series([15.0, 0.0, 2.0])
    """
    if not signals:
        raise ValueError("signals dict must not be empty")
    total: Optional[pd.Series] = None
    for _name, s in signals.items():
        s = ensure_series(s)
        total = s if total is None else total.add(s, fill_value=0.0)
    assert total is not None
    return total.round(2)


def weighted_composite(
    signals: Dict[str, pd.Series],
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Weighted sum of signed score contributions per bar.

    Port of ``atomic.scoring.combinators.weighted_composite()``.

    Args:
        signals: dict mapping signal name → signed contribution Series.
        weights: optional dict of per-signal multipliers; default weight = 1.0.

    Returns:
        pd.Series of per-bar weighted aggregated scores, rounded to 2 dp.

    Example::

        sigs = {"a": pd.Series([2.0, 4.0]), "b": pd.Series([1.0, 1.0])}
        weighted_composite(sigs, weights={"a": 0.5, "b": 2.0})
        # → Series([3.0, 4.0])
    """
    if not signals:
        raise ValueError("signals dict must not be empty")
    if weights is None:
        weights = {}
    total: Optional[pd.Series] = None
    for name, s in signals.items():
        w = float(weights.get(name, 1.0))
        contrib = ensure_series(s) * w
        total = contrib if total is None else total.add(contrib, fill_value=0.0)
    assert total is not None
    return total.round(2)


def sequential_filter(
    signals: List[pd.Series],
    min_strength: float = 0.3,
) -> pd.Series:
    """Count of signals passing the minimum strength threshold per bar.

    Port of ``atomic.scoring.combinators.sequential_filter()``.
    For each bar, counts how many signal Series have ``|value| >= min_strength``.

    Args:
        signals: list of signed score Series.
        min_strength: minimum absolute value for a signal to count (default 0.3).

    Returns:
        pd.Series of float counts (0 … len(signals)).

    Example::

        sigs = [pd.Series([0.5, 0.1, 0.4]), pd.Series([0.6, 0.5, 0.1])]
        sequential_filter(sigs, min_strength=0.4)
        # → Series([2.0, 1.0, 1.0])
    """
    if not signals:
        raise ValueError("signals list must not be empty")
    count: Optional[pd.Series] = None
    for s in signals:
        s = ensure_series(s)
        passes = (s.abs() >= float(min_strength)).astype(float)
        count = passes if count is None else count.add(passes, fill_value=0.0)
    assert count is not None
    return count.fillna(0.0)


def majority_vote(signals: List[pd.Series]) -> pd.Series:
    """Majority direction vote per bar.

    Port of ``atomic.scoring.combinators.majority_vote()``.
    Returns ``(bullish_count − bearish_count) / total`` in **[−1.0, 1.0]**.
    Positive = bullish majority, negative = bearish majority.

    Args:
        signals: list of signed score Series (positive = bullish, negative = bearish).

    Returns:
        pd.Series rounded to 2 dp.

    Example::

        sigs = [pd.Series([1.0, -1.0, 0.0]), pd.Series([1.0, 1.0, -1.0])]
        majority_vote(sigs)
        # → Series([1.0, 0.0, -0.5])
    """
    if not signals:
        raise ValueError("signals list must not be empty")
    n = len(signals)
    s0 = ensure_series(signals[0])
    bullish = pd.Series(0.0, index=s0.index)
    bearish = pd.Series(0.0, index=s0.index)
    for s in signals:
        sr = ensure_series(s).reindex(s0.index, fill_value=0.0)
        bullish = bullish + (sr > 0).astype(float)
        bearish = bearish + (sr < 0).astype(float)
    return ((bullish - bearish) / n).round(2)


def hierarchical_combine(
    tiers: List[Tuple[str, Dict[str, pd.Series], float]],
) -> pd.Series:
    """Hierarchical tiered scoring per bar.

    Port of ``atomic.scoring.combinators.hierarchical_combine()``.

    Args:
        tiers: list of ``(tier_name, signals_dict, tier_weight)`` tuples.
               *signals_dict* maps signal name → signed contribution Series.

    Returns:
        pd.Series of per-bar tier-weighted aggregate scores, rounded to 2 dp.

    Example::

        tiers = [
            ("trend",  {"ema": pd.Series([2.0])}, 0.5),  # 2.0 × 0.5 = 1.0
            ("volume", {"vol": pd.Series([4.0])}, 1.0),  # 4.0 × 1.0 = 4.0
        ]
        hierarchical_combine(tiers)  # → Series([5.0])
    """
    if not tiers:
        raise ValueError("tiers list must not be empty")
    total: Optional[pd.Series] = None
    for _tier_name, signals, tier_weight in tiers:
        tier_score = additive_combine(signals)
        weighted = tier_score * float(tier_weight)
        total = weighted if total is None else total.add(weighted, fill_value=0.0)
    assert total is not None
    return total.round(2)
