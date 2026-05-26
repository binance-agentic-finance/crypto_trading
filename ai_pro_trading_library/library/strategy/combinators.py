"""Entry-condition combinators — L1 parity vs `cyqnt_trd.blocks.entry`.

These differ from `library.scoring.combinators` (which acts on `Signal`
dataclasses): these helpers act on raw boolean / mapping inputs and are
the convenience surface for hand-written or LLM-generated strategies.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Tuple

import pandas as pd


def _bool(s: pd.Series) -> pd.Series:
    return pd.Series(s).fillna(False).astype(bool)


def all_of(conditions: Iterable[pd.Series]) -> pd.Series:
    """True where every condition is True."""
    conditions = list(conditions)
    if not conditions:
        raise ValueError("at least one condition is required")
    out = _bool(conditions[0])
    for c in conditions[1:]:
        out = out & _bool(c)
    return out


def any_of(conditions: Iterable[pd.Series]) -> pd.Series:
    """True where any condition is True."""
    conditions = list(conditions)
    if not conditions:
        raise ValueError("at least one condition is required")
    out = _bool(conditions[0])
    for c in conditions[1:]:
        out = out | _bool(c)
    return out


def exclude_when(base: pd.Series, exclusions: Iterable[pd.Series]) -> pd.Series:
    """`base & ~any_of(exclusions)`."""
    return _bool(base) & ~any_of(exclusions)


def weighted_score(
    rules: Mapping[str, Tuple[pd.Series, float]],
) -> pd.Series:
    """Sum of `bool(cond) * weight` over a mapping of named rules."""
    if not rules:
        raise ValueError("at least one rule is required")
    items = list(rules.items())
    total: pd.Series | None = None
    for _name, (cond, weight) in items:
        contribution = _bool(cond).astype(float) * float(weight)
        total = contribution if total is None else total + contribution
    return total  # type: ignore[return-value]


def score_entry(
    rules: Mapping[str, Tuple[pd.Series, float]],
    threshold: float,
) -> pd.Series:
    """True where `weighted_score(rules) >= threshold`."""
    return weighted_score(rules) >= float(threshold)


def regime_switch(
    regime: pd.Series,
    mapping: dict[str, pd.Series],
) -> pd.Series:
    """Pick active signal based on regime label per bar."""
    out = pd.Series(False, index=regime.index)
    for label, signal in mapping.items():
        s = _bool(signal)
        out = out.where(regime != label, s)
    return out


def consecutive(condition: pd.Series, n: int) -> pd.Series:
    """True when condition has been True for n consecutive bars."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    c = _bool(condition)
    return c.rolling(window=n, min_periods=n).sum().eq(n).fillna(False)


def adaptive_switch(
    indicator_value: pd.Series,
    rules: Iterable[Tuple[Callable[[float], bool], pd.Series]],
    default: pd.Series,
) -> pd.Series:
    """Pick a signal based on rule evaluators on an indicator value."""
    iv = pd.Series(indicator_value)
    out = _bool(default).copy()
    for predicate, signal in rules:
        mask = iv.apply(lambda x: bool(predicate(x)) if pd.notna(x) else False)
        out = out.where(~mask, _bool(signal))
    return out


__all__ = [
    "adaptive_switch",
    "all_of",
    "any_of",
    "consecutive",
    "exclude_when",
    "regime_switch",
    "score_entry",
    "weighted_score",
]
