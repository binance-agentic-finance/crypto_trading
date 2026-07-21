"""StrategyDecision — single-symbol template output + flat_decision helper.

UI card: "template output" (single-symbol).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyDecision:
    side:     str                # "long" | "short" | "flat"
    strength: float = 1.0        # 0.0..1.0, optional confidence hint
    reason:   str = ""           # human-readable explanation


def flat_decision(reason: str = "no signal") -> StrategyDecision:
    """Convenience: 'do nothing this bar'. Prefer this over raising."""
    return StrategyDecision(side="flat", strength=0.0, reason=reason)
