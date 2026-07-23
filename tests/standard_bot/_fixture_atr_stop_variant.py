"""Test fixture: MA-cross strategy with native atr_stop_tp exit_cfg.

Used to verify cyqnt-trd Bug #1 fix end-to-end via mvp_backtest CLI.
This is intentionally separate from auto_opt_experiments.variants/H002_*
which had been switched to pct_stop_tp as a workaround.
"""

import os
import pandas as pd
from cyqnt_trd.blocks import indicators as ind, strategy

STOP_MULT = float(os.environ.get("FIXTURE_ATR_STOP_MULT", "0.5"))
TP_MULT = float(os.environ.get("FIXTURE_ATR_TP_MULT", "999.0"))


def make_signals(df: pd.DataFrame):
    fast = ind.sma(df["close"], 20)
    slow = ind.sma(df["close"], 60)
    long_signal = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    short_signal = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    return long_signal.fillna(False).astype(bool), short_signal.fillna(False).astype(bool)


strategy.register(
    "fixture_ma_cross_atr_stop",
    make_signals,
    exit_cfg={
        "type": "atr_stop_tp",
        "atr_period": 14,
        "stop_mult": STOP_MULT,
        "tp_mult": TP_MULT,
        "max_bars": 9999,
    },
)
