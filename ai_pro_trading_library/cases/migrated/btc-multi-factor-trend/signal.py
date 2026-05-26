"""BTC multi-factor trend long signal: close > fast_ema > slow_ema."""

from __future__ import annotations

import pandas as pd

from ai_pro_trading_library.library.core.protocols import StrategySpec
from ai_pro_trading_library.library.features.indicators import ema


def build_signal(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    close = data["close"].astype(float)
    fast = ema(close, int(spec.parameters.get("fast_ema", 20)))
    slow = ema(close, int(spec.parameters.get("slow_ema", 50)))
    return ((close > fast) & (fast > slow)).fillna(False)
