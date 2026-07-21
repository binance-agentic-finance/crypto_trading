"""AtrBlock — Wilder ATR + %-of-close + dual-speed expansion regime."""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from demo_strategy._shared.blocks.base import Block


class AtrBlock(Block):
    NAME  = "atr"
    LAYER = "signals"
    DEFAULT_PARAMS: ClassVar[dict] = {
        "period":                 14,
        "fast_atr":               7,
        "slow_atr":               21,
        "expand_ratio_threshold": 1.05,
    }

    def compute(self, inputs, **kwargs) -> dict:
        p = self._merge(kwargs)
        bars = self._to_bars(inputs)
        if bars is None or len(bars) < max(p["period"], p["slow_atr"]) + 1:
            return {"atr": None, "atr_pct": None,
                    "expanding": False, "ratio": None}

        a = self._atr(bars, p["period"])
        af = self._atr(bars, p["fast_atr"])
        as_ = self._atr(bars, p["slow_atr"])
        close = float(bars["close"].iloc[-1])
        atr_pct = None if not (a and close) else float(a / close * 100)
        ratio = None if not (af and as_) else float(af / as_)
        expanding = bool(ratio is not None and ratio > p["expand_ratio_threshold"])

        return {
            "atr":       a,
            "atr_pct":   atr_pct,
            "expanding": expanding,
            "ratio":     ratio,
        }

    @staticmethod
    def _to_bars(x) -> pd.DataFrame | None:
        if isinstance(x, pd.DataFrame):
            need = {"high", "low", "close"}
            return x if need.issubset(x.columns) else None
        return None

    @staticmethod
    def _atr(bars: pd.DataFrame, period: int) -> float | None:
        if len(bars) < period + 1:
            return None
        h, l, c = bars["high"], bars["low"], bars["close"]
        prev = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev).abs(),
                        (l - prev).abs()], axis=1).max(axis=1)
        return float(tr.ewm(alpha=1.0/period, adjust=False).mean().iloc[-1])
