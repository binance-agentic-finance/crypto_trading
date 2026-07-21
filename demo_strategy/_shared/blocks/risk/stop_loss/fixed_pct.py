"""FixedPctStopLossBlock — compute stop price at a fixed % from entry.

UI card: "risk · stop-loss · fixed_pct" — one slider (stop_pct).
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class FixedPctStopLossBlock(Block):
    NAME  = "stop_loss/fixed_pct"
    LAYER = "risk"
    DEFAULT_PARAMS: ClassVar[dict] = {"stop_pct": 5.0}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'entry': float, 'side': 'long'|'short'}."""
        p = self._merge(kwargs)
        entry = float(inputs.get("entry", 0))
        side  = inputs.get("side", "long")
        pct   = p["stop_pct"] / 100.0
        stop = entry * (1 - pct) if side == "long" else entry * (1 + pct)
        return {"stop_price": stop, "stop_pct": p["stop_pct"], "side": side}
