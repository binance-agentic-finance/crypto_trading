"""AttentionFrequencyBlock — score how "everywhere" a token is on Square.

Input:  {locales: [...], sections: [...], en_cn_overlap: bool,
         section_overlap: bool, ...}   ← from square_buzz_screener's
         universe_source.get_universe()
Output: {score, tier, reason}
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class AttentionFrequencyBlock(Block):
    NAME  = "attention_frequency"
    LAYER = "signals"
    DEFAULT_PARAMS: ClassVar[dict] = {
        "tiers": {"en_cn_overlap": 3, "section_overlap": 2,
                  "single_source": 1, "isolated": 0},
    }

    def compute(self, inputs, **kwargs) -> dict:
        p = self._merge(kwargs)
        tiers = p["tiers"]
        att = inputs or {}

        if att.get("en_cn_overlap"):
            return {"score": tiers["en_cn_overlap"], "tier": "en_cn_overlap",
                    "reason": "EN+CN cross-locale"}
        if att.get("section_overlap"):
            return {"score": tiers["section_overlap"], "tier": "section_overlap",
                    "reason": "Multi-section (searched + rapid_riser)"}
        if att.get("locales"):
            return {"score": tiers["single_source"], "tier": "single_source",
                    "reason": "Single-source mention"}
        return {"score": tiers["isolated"], "tier": "isolated",
                "reason": "No Square mention"}
