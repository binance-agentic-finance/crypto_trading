"""RSI mean-reversion long signal.

Long when RSI(period) <= oversold; flat otherwise.
"""

from __future__ import annotations

import pandas as pd

from ai_pro_trading_library.library.core.protocols import StrategySpec
from ai_pro_trading_library.library.features.indicators import rsi


def build_signal(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    period = int(spec.parameters.get("rsi_period", 14))
    oversold = float(spec.parameters.get("oversold", 30))
    return (rsi(data["close"].astype(float), period) <= oversold).fillna(False)
