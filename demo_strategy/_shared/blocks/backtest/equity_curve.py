"""EquityCurveBlock — accumulate trade net-PnL into an equity time series.

UI card: "backtest · equity_curve" — starting balance + save path.
"""
from __future__ import annotations

from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class EquityCurveBlock(Block):
    NAME  = "equity_curve"
    LAYER = "backtest"
    DEFAULT_PARAMS: ClassVar[dict] = {"initial_usdt": 10_000, "save_path": None}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = list of trade dicts with 'net_pnl' + 'ts' fields."""
        p = self._merge(kwargs)
        equity = float(p["initial_usdt"])
        curve = [{"ts": None, "equity": equity}]
        for tr in (inputs or []):
            equity += float(tr.get("net_pnl", 0))
            curve.append({"ts": tr.get("ts"), "equity": equity})
        return {"curve": curve, "final_equity": equity,
                "return_pct": (equity / p["initial_usdt"] - 1) * 100
                              if p["initial_usdt"] else 0}
