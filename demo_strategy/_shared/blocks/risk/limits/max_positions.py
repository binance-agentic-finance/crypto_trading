"""MaxPositionsLimitBlock — gate that blocks new entries once N positions open.

UI card: "risk · limit · max_positions".
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class MaxPositionsLimitBlock(Block):
    NAME  = "limits/max_positions"
    LAYER = "risk"
    DEFAULT_PARAMS: ClassVar[dict] = {"max_open": 3}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'open_positions': int}."""
        p = self._merge(kwargs)
        n = int(inputs.get("open_positions", 0))
        if n >= p["max_open"]:
            return {"allow": False,
                    "reason": f"open_positions {n} ≥ max_open {p['max_open']}"}
        return {"allow": True, "reason": f"open_positions {n}"}
