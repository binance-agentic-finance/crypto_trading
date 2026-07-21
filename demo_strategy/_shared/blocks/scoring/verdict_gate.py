"""VerdictGateBlock — bucket a total score into a labeled verdict."""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class VerdictGateBlock(Block):
    NAME  = "verdict_gate"
    LAYER = "scoring"
    DEFAULT_PARAMS: ClassVar[dict] = {
        "thresholds": {"STRONG_CANDIDATE": 8, "CANDIDATE": 4, "WATCHLIST": 1},
        "skip_label": "SKIP",
    }

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = int total OR dict from HierarchicalScoreBlock."""
        p = self._merge(kwargs)
        total = inputs.get("total") if isinstance(inputs, dict) else int(inputs)

        ordered = sorted(p["thresholds"].items(), key=lambda kv: -kv[1])
        for label, thr in ordered:
            if total >= thr:
                return {"verdict": label, "total": int(total)}
        return {"verdict": p["skip_label"], "total": int(total)}
