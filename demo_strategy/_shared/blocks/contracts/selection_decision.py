"""SelectionDecision — cross-sectional template output.

UI card: "template output" (selection type).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SelectionDecision:
    weights:  dict[str, float] = field(default_factory=dict)
    reason:   str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
