"""AtrDynamicStopLossBlock — ATR-scaled stop distance.

UI card: "risk · stop-loss · atr_dynamic" — one slider (atr_mult).
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class AtrDynamicStopLossBlock(Block):
    NAME  = "stop_loss/atr_dynamic"
    LAYER = "risk"
    DEFAULT_PARAMS: ClassVar[dict] = {"atr_mult": 2.0}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'entry': float, 'side': str, 'atr': float}."""
        p = self._merge(kwargs)
        entry = float(inputs.get("entry", 0))
        side  = inputs.get("side", "long")
        atr   = float(inputs.get("atr") or 0)
        offset = p["atr_mult"] * atr
        stop = entry - offset if side == "long" else entry + offset
        # equivalent stop_pct for reporting
        stop_pct = (offset / entry * 100) if entry > 0 else 0
        return {"stop_price": stop, "stop_pct": stop_pct,
                "side": side, "atr_mult": p["atr_mult"]}
