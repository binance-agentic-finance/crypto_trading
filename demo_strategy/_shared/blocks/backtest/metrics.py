"""MetricsBlock — compute standard backtest metrics from trades + equity.

UI card: "backtest · metrics" — checkbox list for which metrics to compute.
"""
from __future__ import annotations

import math
from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class MetricsBlock(Block):
    NAME  = "metrics"
    LAYER = "backtest"
    DEFAULT_PARAMS: ClassVar[dict] = {
        "compute": ["total_return", "max_drawdown", "sharpe",
                    "win_rate", "avg_win_loss", "n_trades"],
    }

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = {'curve': list[{ts, equity}], 'trades': list[{net_pnl,...}]}."""
        p = self._merge(kwargs)
        want = set(p["compute"] or [])

        curve  = inputs.get("curve") or []
        trades = inputs.get("trades") or []
        out: dict = {}

        if "total_return" in want and curve:
            start, end = curve[0]["equity"], curve[-1]["equity"]
            out["total_return_pct"] = (end / start - 1) * 100 if start else 0

        if "max_drawdown" in want and curve:
            peak = -math.inf
            max_dd = 0.0
            for row in curve:
                e = row["equity"]
                if e > peak:
                    peak = e
                if peak > 0:
                    dd = (e - peak) / peak * 100
                    if dd < max_dd:
                        max_dd = dd
            out["max_drawdown_pct"] = max_dd

        if "n_trades" in want:
            out["n_trades"] = len(trades)

        if "win_rate" in want and trades:
            wins = sum(1 for t in trades if t.get("net_pnl", 0) > 0)
            out["win_rate_pct"] = wins / len(trades) * 100

        if "avg_win_loss" in want and trades:
            wins = [t["net_pnl"] for t in trades if t.get("net_pnl", 0) > 0]
            losses = [t["net_pnl"] for t in trades if t.get("net_pnl", 0) <= 0]
            out["avg_win"]  = sum(wins) / len(wins) if wins else 0
            out["avg_loss"] = sum(losses) / len(losses) if losses else 0
            out["avg_win_loss_ratio"] = (out["avg_win"] / abs(out["avg_loss"])
                                          if out["avg_loss"] else None)

        if "sharpe" in want and curve and len(curve) > 2:
            rets = []
            for i in range(1, len(curve)):
                prev, cur = curve[i-1]["equity"], curve[i]["equity"]
                if prev > 0:
                    rets.append(cur / prev - 1)
            if rets:
                mu = sum(rets) / len(rets)
                var = sum((r - mu) ** 2 for r in rets) / len(rets)
                sd = math.sqrt(var)
                # annualization factor = sqrt(365 * bars_per_day); we don't know
                # the timeframe here so leave raw (caller multiplies if needed)
                out["sharpe_raw"] = (mu / sd) if sd > 0 else 0
        return out
