"""Entry-condition combinators.

Functional helpers to combine multiple boolean entry conditions into a
final entry signal. Conditions can also be combined with the standard
pandas ``&`` / ``|`` / ``~`` operators — these helpers exist so that
LLM-generated and human-written code can stay readable when there are
many conditions.

Examples
--------
>>> from cyqnt_trd.blocks import entry
>>> entry.all_of([cond1, cond2, cond3])
>>> entry.any_of([cond1, cond2])
>>> entry.score_entry({"trend": (cond_trend, 2.0), "volume": (cond_vol, 1.0)}, threshold=2.5)
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Tuple, Union

import pandas as pd

from ._utils import ensure_series

__all__ = [
    "all_of",
    "any_of",
    "exclude_when",
    "score_entry",
    "weighted_score",
    "regime_switch",
    "consecutive",
]

ConditionInput = Union[pd.Series, Iterable]


def _bool(s: pd.Series) -> pd.Series:
    return ensure_series(s).fillna(False).astype(bool)


def all_of(conditions: Iterable[pd.Series]) -> pd.Series:
    """Return a boolean series that is True when *all* conditions are True.

    Equivalent to ``cond1 & cond2 & ... & condN`` but accepts an iterable
    so that LLM-generated code can dynamically build the list.
    """
    conditions = list(conditions)
    if not conditions:
        raise ValueError("at least one condition is required")
    out = _bool(conditions[0])
    for c in conditions[1:]:
        out = out & _bool(c)
    return out


def any_of(conditions: Iterable[pd.Series]) -> pd.Series:
    """Return a boolean series that is True when *any* condition is True."""
    conditions = list(conditions)
    if not conditions:
        raise ValueError("at least one condition is required")
    out = _bool(conditions[0])
    for c in conditions[1:]:
        out = out | _bool(c)
    return out


def exclude_when(base: pd.Series, exclusions: Iterable[pd.Series]) -> pd.Series:
    """Return ``base & ~any_of(exclusions)`` — apply a list of veto conditions."""
    return _bool(base) & ~any_of(exclusions)


def weighted_score(
    rules: Mapping[str, Tuple[pd.Series, float]],
) -> pd.Series:
    """Sum boolean rules weighted by their score.

    *rules* is a mapping ``name -> (boolean_series, weight)``.
    Returns a numeric score series.
    """
    if not rules:
        raise ValueError("at least one rule is required")
    items = list(rules.items())
    total: pd.Series
    for i, (_, (cond, weight)) in enumerate(items):
        contribution = _bool(cond).astype(float) * float(weight)
        if i == 0:
            total = contribution
        else:
            total = total + contribution
    return total


def score_entry(
    rules: Mapping[str, Tuple[pd.Series, float]],
    threshold: float,
) -> pd.Series:
    """Return a boolean series: True where the weighted score >= threshold.

    Useful for "scoring system" entries that use a sum-of-evidence rule
    (very common in the user dataset, e.g. trend-momentum strategies).
    """
    return weighted_score(rules) >= float(threshold)


def regime_switch(
    regime: pd.Series, mapping: Dict[str, pd.Series]
) -> pd.Series:
    """Pick the active signal based on a regime label.

    *regime* is a string-valued series (e.g. from
    :func:`cyqnt_trd.blocks.regime.adx_regime`). For each bar, the
    output equals the boolean signal in ``mapping[regime[bar]]``, or
    False if the regime is not in the mapping.
    """
    regime = ensure_series(regime)
    out = pd.Series(False, index=regime.index)
    for label, signal in mapping.items():
        s = _bool(signal)
        out = out.where(regime != label, s)
    return out


def consecutive(condition: pd.Series, n: int) -> pd.Series:
    """True where *condition* has been True for *n* consecutive bars."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    c = _bool(condition)
    return c.rolling(window=n, min_periods=n).sum().eq(n).fillna(False)


def adaptive_switch(
    indicator_value: pd.Series,
    rules: Iterable[Tuple[Callable[[float], bool], pd.Series]],
    default: pd.Series,
) -> pd.Series:
    """Pick a signal based on rule evaluators on an indicator value.

    Example: switch between trend & range strategies based on ADX::

        from cyqnt_trd.blocks import indicators as ind, entry

        adx_v, *_ = ind.adx(df, 14)
        signal = entry.adaptive_switch(
            adx_v,
            rules=[
                (lambda x: x >= 25, trend_signal),
                (lambda x: x < 20, range_signal),
            ],
            default=pd.Series(False, index=df.index),
        )
    """
    iv = ensure_series(indicator_value)
    out = _bool(default).copy()
    for predicate, signal in rules:
        mask = iv.apply(lambda x: bool(predicate(x)) if pd.notna(x) else False)
        out = out.where(~mask, _bool(signal))
    return out
