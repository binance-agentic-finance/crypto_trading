"""TrailingStopLossBlock — offset from since-entry extreme close.

UI card: "risk · stop-loss · trailing" — one slider (offset_pct).
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class TrailingStopLossBlock(Block):
    NAME  = "stop_loss/trailing"
    LAYER = "risk"
    DEFAULT_PARAMS: ClassVar[dict] = {"offset_pct": 3.0}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'side': str, 'extreme_since_entry': float}.

        `extreme_since_entry` is caller-tracked (daemon holds it).
        """
        p = self._merge(kwargs)
        extreme = float(inputs.get("extreme_since_entry", 0))
        side = inputs.get("side", "long")
        pct = p["offset_pct"] / 100.0
        stop = extreme * (1 - pct) if side == "long" else extreme * (1 + pct)
        return {"stop_price": stop, "offset_pct": p["offset_pct"], "side": side}
