"""AttentionDeepBlock — score depth-fetch mention count."""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class AttentionDeepBlock(Block):
    NAME  = "attention_deep"
    LAYER = "signals"
    DEFAULT_PARAMS: ClassVar[dict] = {
        "thresholds":  {"strong": 3, "mixed": 1},
        "tier_scores": {"strong": 2, "mixed": 1, "none": -1},
    }

    def compute(self, inputs, **kwargs) -> dict:
        p = self._merge(kwargs)
        n = (inputs or {}).get("deep_mentions", 0)
        thr = p["thresholds"]
        sc = p["tier_scores"]
        if n >= thr["strong"]:
            return {"score": sc["strong"], "tier": "strong",
                    "reason": f"deep×{n}", "count": n}
        if n >= thr["mixed"]:
            return {"score": sc["mixed"], "tier": "mixed",
                    "reason": f"deep×{n}", "count": n}
        return {"score": sc["none"], "tier": "none",
                "reason": "no deep signal", "count": n}
