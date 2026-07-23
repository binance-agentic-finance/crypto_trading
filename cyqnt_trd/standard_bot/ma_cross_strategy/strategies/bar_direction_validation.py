"""
Bar-direction validation strategy.

Designed for end-to-end pipeline verification:
  - very frequent target changes on 1m bars
  - simple, fully deterministic logic
  - no-lookahead and next-bar-open compatible

Logic:
  - long when the latest close is above the previous close
  - short when the latest close is below the previous close
"""

from __future__ import annotations

import pandas as pd

from cyqnt_trd.blocks import strategy


def make_signals(df: pd.DataFrame):
    if len(df) < 2:
        false_series = pd.Series(False, index=df.index)
        return false_series, false_series

    prev_close = df["close"].shift(1)
    long_signal = df["close"] > prev_close
    short_signal = df["close"] < prev_close
    return long_signal.fillna(False), short_signal.fillna(False)


strategy.register("bar_direction_validation", make_signals)
