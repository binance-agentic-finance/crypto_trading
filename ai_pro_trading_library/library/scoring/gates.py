"""Gate definitions for converting continuous features into discrete signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_pro_trading_library.library.scoring.rules import Verdict


Operator = Literal["gt", "gte", "lt", "lte", "eq", "between"]


@dataclass(frozen=True)
class Gate:
    label: str
    operator: Operator
    threshold: float | tuple[float, float]
    weight: float = 1.0

    def evaluate(self, value: float) -> bool:
        if self.operator == "gt":
            return value > float(self.threshold)
        if self.operator == "gte":
            return value >= float(self.threshold)
        if self.operator == "lt":
            return value < float(self.threshold)
        if self.operator == "lte":
            return value <= float(self.threshold)
        if self.operator == "eq":
            return value == float(self.threshold)
        if self.operator == "between":
            low, high = self.threshold
            return float(low) <= value <= float(high)
        raise ValueError(f"unsupported gate operator: {self.operator}")


def verdict_classify(score: float, threshold: float, reason: str = "") -> Verdict:
    passed = float(score) >= float(threshold)
    return Verdict(
        label="pass" if passed else "fail",
        score=float(score),
        threshold=float(threshold),
        reason=reason or ("score threshold met" if passed else "score threshold not met"),
    )
