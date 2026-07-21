"""RsiBlock — Wilder RSI + zone label.

Zones:  OVERBOUGHT / OVERSOLD / BULLISH_NEUTRAL / BEARISH_NEUTRAL / NEUTRAL
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from demo_strategy._shared.blocks.base import Block


class RsiBlock(Block):
    NAME  = "rsi"
    LAYER = "signals"
    DEFAULT_PARAMS: ClassVar[dict] = {
        "period":     14,
        "overbought": 70,
        "oversold":   30,
    }

    def compute(self, inputs, **kwargs) -> dict:
        p = self._merge(kwargs)
        closes = self._to_closes(inputs)
        if len(closes) < p["period"] + 1:
            return {"value": None, "zone": "UNKNOWN"}

        s = pd.Series(closes)
        delta = s.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1.0/p["period"], adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1.0/p["period"], adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        val = (100 - 100 / (1 + rs)).iloc[-1]
        v = float(val) if pd.notna(val) else 50.0

        return {"value": v, "zone": self._zone(v, p["overbought"], p["oversold"])}

    @staticmethod
    def _zone(v: float, overbought: float, oversold: float) -> str:
        if v >= overbought:  return "OVERBOUGHT"
        if v <= oversold:    return "OVERSOLD"
        if v > 50:           return "BULLISH_NEUTRAL"
        if v < 50:           return "BEARISH_NEUTRAL"
        return "NEUTRAL"

    @staticmethod
    def _to_closes(x):
        if isinstance(x, pd.DataFrame):
            return x["close"].tolist() if "close" in x.columns else []
        if isinstance(x, (list, tuple)):
            return list(x)
        return []
