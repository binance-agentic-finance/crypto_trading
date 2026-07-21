"""VolumeSurgeBlock — current bar volume vs rolling-mean baseline."""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from demo_strategy._shared.blocks.base import Block


class VolumeSurgeBlock(Block):
    NAME  = "volume_surge"
    LAYER = "signals"
    DEFAULT_PARAMS: ClassVar[dict] = {"multiplier": 2.0, "lookback": 24}

    def compute(self, inputs, **kwargs) -> dict:
        p = self._merge(kwargs)
        bars = self._to_bars(inputs)
        if bars is None or len(bars) < p["lookback"] + 1:
            return {"surge": False, "ratio": None}
        recent = bars["volume"].iloc[-1]
        base = bars["volume"].iloc[-p["lookback"]-1:-1].mean()
        if base == 0:
            return {"surge": False, "ratio": None}
        r = float(recent / base)
        return {"surge": r >= p["multiplier"], "ratio": r}

    @staticmethod
    def _to_bars(x) -> pd.DataFrame | None:
        if isinstance(x, pd.DataFrame) and "volume" in x.columns:
            return x
        return None
