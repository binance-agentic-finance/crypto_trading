"""Weighted scoring primitives.

Source attribution: rewritten from `cyqnt_trd.blocks.scoring`; gates and verdicts
are added as canonical library concepts for migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str
    condition: pd.Series
    weight: float = 1.0
    cap: float | None = None


@dataclass(frozen=True)
class Verdict:
    label: Literal["pass", "fail"]
    score: float
    threshold: float
    reason: str = ""


class ScoringSystem:
    def __init__(self) -> None:
        self._rules: list[Rule] = []

    @property
    def rules(self) -> tuple[Rule, ...]:
        return tuple(self._rules)

    def add_rule(
        self,
        name: str,
        condition: pd.Series,
        weight: float = 1.0,
        cap: float | None = None,
    ) -> "ScoringSystem":
        self._rules.append(Rule(name=name, condition=condition, weight=float(weight), cap=cap))
        return self

    def evaluate(self) -> pd.Series:
        if not self._rules:
            raise ValueError("no rules registered")
        total: pd.Series | None = None
        for rule in self._rules:
            contribution = rule.condition.fillna(False).astype(bool).astype(float) * rule.weight
            if rule.cap is not None:
                contribution = contribution.clip(upper=rule.cap)
            total = contribution if total is None else total.add(contribution, fill_value=0.0)
        assert total is not None
        return total

    def signal(self, threshold: float) -> pd.Series:
        return self.evaluate() >= float(threshold)

    def verdict_at(self, index: int | str, threshold: float) -> Verdict:
        score = float(self.evaluate().loc[index])
        passed = score >= float(threshold)
        return Verdict(
            label="pass" if passed else "fail",
            score=score,
            threshold=float(threshold),
            reason="score threshold met" if passed else "score threshold not met",
        )

