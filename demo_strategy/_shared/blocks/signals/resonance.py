"""ResonanceBlock — cross-timeframe direction alignment.

Two modes:
  A) inputs = {tf_name: bars_df} → per-tf trend_classify → resonance
  B) inputs = pd.DataFrame or list[float] → proxy resonance over
     short/mid/long slices of the same series (single-tf fallback)
"""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from demo_strategy._shared.blocks.base import Block


class ResonanceBlock(Block):
    NAME  = "resonance"
    LAYER = "signals"
    DEFAULT_PARAMS: ClassVar[dict] = {
        "proxy_windows":       {"short": 6, "mid": 20, "long": 60},
        "direction_threshold": 0.005,
    }

    def compute(self, inputs, **kwargs) -> dict:
        p = self._merge(kwargs)
        thr = p["direction_threshold"]

        if isinstance(inputs, dict):
            # Mode A — per-tf bars
            dirs = {}
            for tf, bars in inputs.items():
                closes = self._closes(bars)
                if len(closes) >= 3:
                    dirs[tf] = self._trend(closes, thr)
            return self._resonate(dirs)

        # Mode B — single-tf proxy
        closes = self._closes(inputs)
        if len(closes) < p["proxy_windows"]["long"]:
            return {"aligned": False, "dominant": "UNKNOWN",
                    "tf_directions": {}}
        w = p["proxy_windows"]
        dirs = {
            "short": self._trend(closes[-w["short"]:], thr),
            "mid":   self._trend(closes[-w["mid"]:],   thr),
            "long":  self._trend(closes[-w["long"]:],  thr),
        }
        return self._resonate(dirs)

    # ── helpers ──
    @staticmethod
    def _closes(x):
        if isinstance(x, pd.DataFrame):
            return x["close"].tolist() if "close" in x.columns else []
        if isinstance(x, (list, tuple)):
            return list(x)
        return []

    @staticmethod
    def _trend(closes, threshold: float) -> str:
        if len(closes) < 3:
            return "UNKNOWN"
        first, last = closes[0], closes[-1]
        if last > first * (1 + threshold): return "BULLISH"
        if last < first * (1 - threshold): return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _resonate(tf_directions: dict) -> dict:
        if not tf_directions:
            return {"aligned": False, "dominant": "UNKNOWN",
                    "tf_directions": {}}
        counts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0, "UNKNOWN": 0}
        for d in tf_directions.values():
            counts[d] = counts.get(d, 0) + 1
        # Ignore UNKNOWN when picking the dominant direction
        directional = {k: v for k, v in counts.items() if k != "UNKNOWN"}
        dominant = max(directional, key=directional.get) if directional else "UNKNOWN"
        aligned = dominant != "UNKNOWN" and counts[dominant] == len(tf_directions)
        return {"aligned": aligned, "dominant": dominant,
                "tf_directions": dict(tf_directions)}
