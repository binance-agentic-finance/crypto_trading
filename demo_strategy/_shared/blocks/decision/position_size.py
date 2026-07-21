"""PositionSizeBlock — derive USDT notional + leverage from balance + risk."""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class PositionSizeBlock(Block):
    NAME  = "position_size"
    LAYER = "decision"
    DEFAULT_PARAMS: ClassVar[dict] = {
        "risk_pct":     0.03,
        "stop_pct":     0.05,
        "max_leverage": 5,
    }

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = float account balance (USDT).

        Returns:
            qty_usdt, leverage, stop_pct, risk_usdt
        """
        p = self._merge(kwargs)
        balance = float(inputs or 0)
        if balance <= 0 or p["stop_pct"] <= 0:
            return {"qty_usdt": 0.0, "leverage": 1,
                    "stop_pct": p["stop_pct"], "risk_usdt": 0.0}
        qty_usdt = balance * p["risk_pct"] / p["stop_pct"]
        need_lev = qty_usdt / balance
        leverage = max(1, min(int(p["max_leverage"]), int(need_lev + 0.5) + 1))
        return {
            "qty_usdt":  round(qty_usdt, 2),
            "leverage":  leverage,
            "stop_pct":  p["stop_pct"],
            "risk_usdt": round(balance * p["risk_pct"], 2),
        }
