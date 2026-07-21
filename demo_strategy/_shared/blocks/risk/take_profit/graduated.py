"""GraduatedTakeProfitBlock — pick the next TP stage given current pnl_pct.

UI card: "risk · take-profit · graduated" — stages table.
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class GraduatedTakeProfitBlock(Block):
    NAME  = "take_profit/graduated"
    LAYER = "risk"
    DEFAULT_PARAMS: ClassVar[dict] = {
        "stages": [
            {"profit_pct":  5, "close_pct":  5},
            {"profit_pct": 10, "close_pct": 10},
            {"profit_pct": 20, "close_pct": 15},
        ],
    }

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'pnl_pct': float, 'stages_hit': list[int]}.
        Return {'trigger': bool, 'close_pct': int, 'stage_index': int}."""
        p = self._merge(kwargs)
        pnl_pct = float(inputs.get("pnl_pct", 0))
        hit = set(inputs.get("stages_hit") or [])
        for i, s in enumerate(p["stages"]):
            if i in hit:
                continue
            if pnl_pct >= s["profit_pct"]:
                return {"trigger": True, "close_pct": s["close_pct"],
                        "stage_index": i}
        return {"trigger": False, "close_pct": 0, "stage_index": -1}
