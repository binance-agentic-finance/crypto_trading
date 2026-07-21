"""BdpBotAdapter — data source when running inside bdp-ai-trading-bot.

UI card: "data source · bdp-bot" — enabled when
`BDP_BOT_MARKET_HOT_STORE` env var is set (auto-detected by factory).
"""
from __future__ import annotations

import pandas as pd


class BdpBotAdapter:
    """Uses `strategy_service.market.fetch_bars` (Redis-backed hot store)."""

    def __init__(self):
        from strategy_service.market import fetch_bars   # noqa: PLC0415
        self._fetch = fetch_bars

    def fetch_bars(self, symbol: str, timeframe: str,
                   limit: int = 200) -> pd.DataFrame:
        return self._fetch(symbol=symbol, interval=timeframe, limit=limit)
