"""MaxExposureLimitBlock — cap per-symbol notional as % of balance.

UI card: "risk · limit · max_exposure".
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class MaxExposureLimitBlock(Block):
    NAME  = "limits/max_exposure"
    LAYER = "risk"
    DEFAULT_PARAMS: ClassVar[dict] = {"max_position_pct": 30.0}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'balance': float, 'notional_usdt': float}."""
        p = self._merge(kwargs)
        balance = float(inputs.get("balance", 0))
        notional = float(inputs.get("notional_usdt", 0))
        if balance <= 0:
            return {"allow": True, "reason": "no balance data"}
        pct = notional / balance * 100
        if pct > p["max_position_pct"]:
            return {"allow": False,
                    "reason": f"notional {pct:.1f}% > cap {p['max_position_pct']}%"}
        return {"allow": True, "reason": f"notional {pct:.1f}%"}
