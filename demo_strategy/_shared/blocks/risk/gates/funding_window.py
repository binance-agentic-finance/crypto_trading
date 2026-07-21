"""FundingWindowGateBlock — block new entries near funding settlement.

Perpetual futures charge funding at 0h / 8h / 16h UTC. This gate blocks
new entries within ±window_min minutes of these times.

UI card: "risk · gate · funding_window".
"""
from __future__ import annotations

import time
from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class FundingWindowGateBlock(Block):
    NAME  = "gates/funding_window"
    LAYER = "risk"
    DEFAULT_PARAMS: ClassVar[dict] = {"window_min": 15}
    _FUNDING_HOURS = (0, 8, 16)

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'now_ts': int (optional ms epoch, default = now)}."""
        p = self._merge(kwargs)
        now = inputs.get("now_ts") or (time.time() * 1000)
        now_sec = float(now) / 1000
        t = time.gmtime(now_sec)
        min_of_day = t.tm_hour * 60 + t.tm_min
        w = int(p["window_min"])
        for h in self._FUNDING_HOURS:
            settle = h * 60
            if abs(min_of_day - settle) <= w or \
               abs(min_of_day - (settle + 1440)) <= w or \
               abs(min_of_day - (settle - 1440)) <= w:
                return {"allow": False,
                        "reason": f"within ±{w}min of funding {h:02d}:00 UTC"}
        return {"allow": True, "reason": f"outside funding window"}
