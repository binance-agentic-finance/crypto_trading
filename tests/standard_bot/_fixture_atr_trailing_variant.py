"""Test fixture: MA-cross strategy with native atr_trailing_stop exit_cfg.

Used to verify cyqnt-trd Bug #3 fix end-to-end via mvp_backtest CLI.
Mirrors ``_fixture_atr_stop_variant.py`` but with a trailing stop.
"""

import os
import pandas as pd
from cyqnt_trd.blocks import indicators as ind, strategy

TRAIL_MULT = float(os.environ.get("FIXTURE_ATR_TRAIL_MULT", "2.0"))
ATR_PERIOD = int(os.environ.get("FIXTURE_ATR_TRAIL_PERIOD", "14"))


def make_signals(df: pd.DataFrame):
    fast = ind.sma(df["close"], 20)
    slow = ind.sma(df["close"], 60)
    long_signal = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    short_signal = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    return long_signal.fillna(False).astype(bool), short_signal.fillna(False).astype(bool)


strategy.register(
    "fixture_ma_cross_atr_trailing",
    make_signals,
    exit_cfg={
        "type": "atr_trailing_stop",
        "atr_period": ATR_PERIOD,
        "trail_mult": TRAIL_MULT,
        "max_bars": 9999,
    },
)
