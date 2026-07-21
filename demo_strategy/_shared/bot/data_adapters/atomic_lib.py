"""AtomicLibAdapter — data source when running with atomic_strategy_lib.

UI card: "data source · atomic_strategy_lib" — for local dev / backtest
where binance-cli is available.
"""
from __future__ import annotations

import pandas as pd


class AtomicLibAdapter:
    """Uses `atomic_strategy_lib.data.market_bundle.kline_fetch`."""

    def __init__(self):
        from atomic_strategy_lib.data.market_bundle import kline_fetch  # noqa: PLC0415
        self._fetch = kline_fetch

    def fetch_bars(self, symbol: str, timeframe: str,
                   limit: int = 200) -> pd.DataFrame:
        candles = self._fetch(symbol, interval=timeframe, limit=limit,
                              market="spot") or []
        rows = [
            {"open": c.open, "high": c.high, "low": c.low,
             "close": c.close, "volume": c.volume,
             "close_time": c.close_time}
            for c in candles
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index("close_time", inplace=True)
        return df
