"""Structure breakout long signal: close > prior_high(lookback)."""

from __future__ import annotations

import pandas as pd

from ai_pro_trading_library.library.core.protocols import StrategySpec
from ai_pro_trading_library.library.conditions.atomic import breakout_high


def build_signal(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    lookback = int(spec.parameters.get("breakout_lookback", 20))
    if "high" not in data.columns:
        return pd.Series(False, index=data.index)
    return breakout_high(data, lookback=lookback)
