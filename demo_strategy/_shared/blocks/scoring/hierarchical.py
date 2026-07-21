"""HierarchicalScoreBlock — reduce a list of tier scores → single int.

Modes:
    additive (default): sum(tier_scores)
    weighted:           sum(w_i * s_i), needs weights={name: w}
    max:                max(tier_scores)
"""
from __future__ import annotations

from typing import ClassVar, Iterable

from demo_strategy._shared.blocks.base import Block


class HierarchicalScoreBlock(Block):
    NAME  = "hierarchical"
    LAYER = "scoring"
    DEFAULT_PARAMS: ClassVar[dict] = {"mode": "additive"}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = iterable of numbers OR iterable of dicts with 'score' key.
        Returns {'total': int, 'mode': str}."""
        p = self._merge(kwargs)
        mode = p["mode"]
        scores = self._extract_scores(inputs)

        if not scores:
            return {"total": 0, "mode": mode}
        if mode == "additive":
            return {"total": int(sum(scores)), "mode": mode}
        if mode == "max":
            return {"total": int(max(scores)), "mode": mode}
        if mode == "weighted":
            weights = kwargs.get("weights") or p.get("weights") or {}
            # scores were extracted as list of int; weights maps name→float
            # If caller passed tier dicts, prefer name-weighted total
            if isinstance(inputs, Iterable) and all(isinstance(x, dict) for x in inputs):
                total = 0.0
                for tier in inputs:
                    w = weights.get(tier.get("name"), 1.0)
                    total += w * tier.get("score", 0)
                return {"total": int(total), "mode": mode}
            return {"total": int(sum(scores)), "mode": mode}
        raise ValueError(f"unknown scoring mode {mode!r}")

    @staticmethod
    def _extract_scores(inputs) -> list[int]:
        out = []
        for x in inputs or []:
            if isinstance(x, (int, float)):
                out.append(int(x))
            elif isinstance(x, dict) and "score" in x:
                out.append(int(x["score"]))
        return out
