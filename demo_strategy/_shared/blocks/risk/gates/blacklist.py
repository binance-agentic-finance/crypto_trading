"""SymbolBlacklistGateBlock — refuse orders on blacklisted symbols.

UI card: "risk · gate · blacklist".
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class SymbolBlacklistGateBlock(Block):
    NAME  = "gates/blacklist"
    LAYER = "risk"
    DEFAULT_PARAMS: ClassVar[dict] = {"symbols": []}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'symbol': str}."""
        p = self._merge(kwargs)
        sym = str(inputs.get("symbol") or "").upper()
        bl = {s.upper() for s in (p["symbols"] or [])}
        if sym in bl:
            return {"allow": False, "reason": f"{sym} on blacklist"}
        return {"allow": True, "reason": ""}
