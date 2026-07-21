"""CooldownLimitBlock — block new entries for N seconds after last stop-out.

UI card: "risk · limit · cooldown".
"""
from __future__ import annotations

import time
from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class CooldownLimitBlock(Block):
    NAME  = "limits/cooldown"
    LAYER = "risk"
    DEFAULT_PARAMS: ClassVar[dict] = {"cooldown_sec": 3600}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'last_stop_out_ts': int (ms epoch)}."""
        p = self._merge(kwargs)
        last = int(inputs.get("last_stop_out_ts") or 0)
        if last <= 0:
            return {"allow": True, "reason": "no stop-out on record"}
        elapsed = time.time() - last / 1000
        if elapsed < p["cooldown_sec"]:
            remaining = int(p["cooldown_sec"] - elapsed)
            return {"allow": False,
                    "reason": f"cooldown {remaining}s remaining"}
        return {"allow": True, "reason": f"cooldown cleared ({int(elapsed)}s ago)"}
