"""Example factors — copy one of these and change the body.

A factor is a function ``Panel -> DataFrame`` (dates × symbols). Anything the
frames can express is allowed; what matters is that **row t may only use data
known at the close of day t**. The engine adds the execution lag for you (entry
is ``open[t+2]``), so do not shift your own signal forward "to be safe" — that
double-lags it and quietly costs you a day of edge.

Four of the five below are deliberately *not* good. They are here because the
interesting part of an evaluation matrix is which gate each one fails.
"""
from __future__ import annotations

import numpy as np


def reversal_5d(p):
    """Short-term reversal: yesterday's losers, today's longs."""
    return -np.log(p.close / p.close.shift(5))


def low_volatility(p):
    """Prefer the calmest names. The most persistent crypto cross-sectional style."""
    ret = np.log(p.close / p.close.shift(1))
    return -ret.rolling(10, min_periods=10).std()


def intraday_close_strength(p):
    """Where the close sits inside the day's range (Alpha#101 in the reference study)."""
    return (p.close - p.open) / ((p.high - p.low) + 1e-9)


def volume_shock(p):
    """Today's dollar volume against its own 20-day norm."""
    adv = p.quote_volume.rolling(20, min_periods=20).mean()
    return p.quote_volume / adv.replace(0, np.nan)


def look_ahead_trap(p):
    """**Deliberately broken**: returns the very window the engine trades.

    Entry is ``open[t+2]`` and exit ``open[t+2+h]``, so this hands the scorer the
    answer. It exists as the harness's own smoke test: if this ever stops posting
    an absurd IC, the measurement path has drifted.

    Its high IC is a measurement positive control, not a leakage detector.
    G0 separately rejects it because truncating future panel rows changes past
    signals. That check also rejects ``p.close.shift(-1)`` even when entry delay
    removes any apparent profit from the leak. Sampled checks cannot certify
    upstream data availability or future information hidden in external state.
    """
    h = 3
    return p.open.shift(-(2 + h)) / p.open.shift(-2) - 1.0


DEMO = ["reversal_5d", "low_volatility", "intraday_close_strength",
        "volume_shock", "look_ahead_trap"]
