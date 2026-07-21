"""FeeModelBlock — apply fee + slippage to a trade PnL.

UI card: "backtest · fee_model".
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class FeeModelBlock(Block):
    NAME  = "fee_model"
    LAYER = "backtest"
    DEFAULT_PARAMS: ClassVar[dict] = {"fee_bps": 4.0, "slip_bps": 2.0}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'gross_pnl': float, 'notional': float}."""
        p = self._merge(kwargs)
        gross = float(inputs.get("gross_pnl", 0))
        notional = float(inputs.get("notional", 0))
        total_bps = (p["fee_bps"] + p["slip_bps"]) * 2  # round-trip: entry + exit
        cost = notional * (total_bps / 10_000.0)
        return {"gross_pnl": gross, "cost": cost, "net_pnl": gross - cost}
