"""MacdBlock — MACD line + signal + histogram + direction hint."""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from demo_strategy._shared.blocks.base import Block


class MacdBlock(Block):
    NAME  = "macd"
    LAYER = "signals"
    DEFAULT_PARAMS: ClassVar[dict] = {"fast": 12, "slow": 26, "signal": 9}

    def compute(self, inputs, **kwargs) -> dict:
        p = self._merge(kwargs)
        closes = self._to_closes(inputs)
        if len(closes) < p["slow"] + p["signal"]:
            return {"macd": None, "signal": None, "histogram": None,
                    "hist_increasing": False}

        s = pd.Series(closes)
        fast = s.ewm(span=p["fast"], adjust=False).mean()
        slow = s.ewm(span=p["slow"], adjust=False).mean()
        m = fast - slow
        sig = m.ewm(span=p["signal"], adjust=False).mean()
        hist = m - sig

        return {
            "macd":            float(m.iloc[-1]),
            "signal":          float(sig.iloc[-1]),
            "histogram":       float(hist.iloc[-1]),
            "hist_increasing": bool(len(hist) >= 2 and hist.iloc[-1] > hist.iloc[-2]),
        }

    @staticmethod
    def _to_closes(x):
        if isinstance(x, pd.DataFrame):
            return x["close"].tolist() if "close" in x.columns else []
        if isinstance(x, (list, tuple)):
            return list(x)
        return []
