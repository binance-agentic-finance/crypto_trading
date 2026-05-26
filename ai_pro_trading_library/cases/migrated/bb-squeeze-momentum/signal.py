"""Bollinger squeeze + breakout long signal."""

from __future__ import annotations

import pandas as pd

from ai_pro_trading_library.library.core.protocols import StrategySpec
from ai_pro_trading_library.library.features.indicators import bollinger_bands


def build_signal(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    period = int(spec.parameters.get("bb_period", 20))
    stddev = float(spec.parameters.get("bb_stddev", 2.0))
    close = data["close"].astype(float)
    lower, _middle, upper = bollinger_bands(close, period, stddev)
    bandwidth = (upper - lower) / close
    squeeze_q = bandwidth.rolling(window=period, min_periods=period).quantile(0.35)
    squeeze = bandwidth <= squeeze_q
    return (squeeze & (close > upper.shift(1))).fillna(False)
