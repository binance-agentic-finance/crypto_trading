"""EmaBlock — EMA(fast/mid/slow) values + alignment against last close.

Inputs:  bars (pd.DataFrame with 'close' column) OR list[float]
Output:  {ema_fast, ema_mid, ema_slow, aligned_count, direction}
         direction ∈ BULLISH / BEARISH / NEUTRAL
"""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from demo_strategy._shared.blocks.base import Block


class EmaBlock(Block):
    NAME  = "ema"
    LAYER = "signals"
    DEFAULT_PARAMS: ClassVar[dict] = {
        "periods": {"fast": 20, "mid": 60, "slow": 200},
    }

    def compute(self, inputs, **kwargs) -> dict:
        params = self._merge(kwargs)
        periods = params["periods"]

        closes = self._to_closes(inputs)
        if not closes:
            return {"ema_fast": None, "ema_mid": None, "ema_slow": None,
                    "aligned_count": 0, "direction": "NEUTRAL"}
        price = closes[-1]

        ema_fast = self._ema(closes, periods["fast"])
        ema_mid  = self._ema(closes, periods["mid"])
        ema_slow = self._ema(closes, periods["slow"])

        aligned = 0
        direction = "NEUTRAL"
        if ema_fast and ema_mid and ema_slow and price:
            above = [price > ema_fast, price > ema_mid, price > ema_slow]
            aligned = sum(above)
            if aligned >= 2:      direction = "BULLISH"
            elif not any(above):  direction = "BEARISH"

        return {
            "ema_fast":       ema_fast,
            "ema_mid":        ema_mid,
            "ema_slow":       ema_slow,
            "aligned_count":  aligned,
            "direction":      direction,
        }

    # ── helpers ──
    @staticmethod
    def _to_closes(x) -> list[float]:
        if isinstance(x, pd.DataFrame):
            return x["close"].tolist() if "close" in x.columns else []
        if isinstance(x, (list, tuple)):
            return list(x)
        return []

    @staticmethod
    def _ema(closes, period: int) -> float | None:
        if len(closes) < period:
            return None
        return float(pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1])
