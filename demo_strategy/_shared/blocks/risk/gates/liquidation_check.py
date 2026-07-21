"""LiquidationCheckGateBlock — margin-buffer sanity check.

UI card: "risk · gate · liquidation_check".
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class LiquidationCheckGateBlock(Block):
    NAME  = "gates/liquidation_check"
    LAYER = "risk"
    DEFAULT_PARAMS: ClassVar[dict] = {"margin_buffer_pct": 20.0}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'buffer_pct': float}."""
        p = self._merge(kwargs)
        buf = float(inputs.get("buffer_pct") or 100.0)
        if buf < p["margin_buffer_pct"]:
            return {"allow": False,
                    "reason": f"margin_buffer {buf:.1f}% < min {p['margin_buffer_pct']}%"}
        return {"allow": True, "reason": f"margin_buffer {buf:.1f}%"}
