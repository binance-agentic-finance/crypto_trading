"""MarketData — framework-fetched OHLCV slice handed to a template.

UI card: "market data schema" — shows symbol/timeframe binding + preview
of the bars column layout.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketData:
    bars:      pd.DataFrame        # OHLCV, indexed by close_time ASC
    symbol:    str
    timeframe: str
