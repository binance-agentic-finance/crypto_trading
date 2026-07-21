"""DailyLossLimitBlock — gate that halts new entries when today's loss > threshold.

UI card: "risk · limit · daily_loss" — single field (max_pct).
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class DailyLossLimitBlock(Block):
    NAME  = "limits/daily_loss"
    LAYER = "risk"
    DEFAULT_PARAMS: ClassVar[dict] = {"max_pct": 10.0}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'today_pnl_pct': float}.
        Returns {'allow': bool, 'reason': str}."""
        p = self._merge(kwargs)
        pnl = float(inputs.get("today_pnl_pct", 0))
        if pnl <= -abs(p["max_pct"]):
            return {"allow": False,
                    "reason": f"daily_loss {pnl:.1f}% exceeds -{p['max_pct']}%"}
        return {"allow": True, "reason": f"daily_pnl {pnl:.1f}%"}
