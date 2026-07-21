"""BacktestLoaderBlock — slice historical bars for a backtest window.

UI card: "backtest · loader" — date range + warmup.
"""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from demo_strategy._shared.blocks.base import Block


class BacktestLoaderBlock(Block):
    NAME  = "loader"
    LAYER = "backtest"
    DEFAULT_PARAMS: ClassVar[dict] = {
        "start_date":  None,
        "end_date":    None,
        "warmup_bars": 100,
    }

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = pd.DataFrame with datetime-like index (or `close_time`
        ms epoch as index). Returns {'bars': sliced_df, 'warmup_end_idx': int}."""
        p = self._merge(kwargs)
        df = inputs
        if df is None or df.empty:
            return {"bars": df, "warmup_end_idx": 0}

        # Slice by date range if configured
        start, end = p["start_date"], p["end_date"]
        if start or end:
            idx = pd.to_datetime(df.index, unit="ms", utc=True) \
                  if df.index.dtype.kind == "i" else df.index
            mask = pd.Series(True, index=df.index)
            if start:
                mask &= idx >= pd.Timestamp(start, tz="UTC")
            if end:
                mask &= idx <= pd.Timestamp(end,   tz="UTC")
            df = df.loc[mask]

        warmup = min(int(p["warmup_bars"]), len(df) - 1)
        return {"bars": df, "warmup_end_idx": warmup}
