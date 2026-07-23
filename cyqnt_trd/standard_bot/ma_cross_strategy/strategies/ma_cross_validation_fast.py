"""
High-frequency validation MA cross strategy.

Purpose:
  Give the OpenClaw / Docker validation flow a faster-moving strategy so that
  paper/live paths can be observed within minutes instead of waiting hours for
  a new crossover on 1h settings.

Signal logic:
  - long: SMA(3) crosses above SMA(9)
  - short: SMA(3) crosses below SMA(9)
  - next-bar-open execution, no lookahead (via shift(1))
"""

from __future__ import annotations

import pandas as pd

from cyqnt_trd.blocks import indicators as ind, strategy


FAST_PERIOD: int = 3
SLOW_PERIOD: int = 9


def make_signals(df: pd.DataFrame):
    if len(df) < SLOW_PERIOD + 1:
        false_series = pd.Series(False, index=df.index)
        return false_series, false_series

    fast_ma = ind.sma(df["close"], FAST_PERIOD)
    slow_ma = ind.sma(df["close"], SLOW_PERIOD)

    prev_fast = fast_ma.shift(1)
    prev_slow = slow_ma.shift(1)

    long_signal = (prev_fast < prev_slow) & (fast_ma > slow_ma)
    short_signal = (prev_fast > prev_slow) & (fast_ma < slow_ma)
    return long_signal.fillna(False), short_signal.fillna(False)


strategy.register("ma_cross_validation_fast", make_signals)
