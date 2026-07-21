"""DirectRESTAdapter — public Binance kline endpoint.

UI card: "data source · direct REST" — always-available fallback,
no auth needed. Used when neither bdp-bot nor atomic_strategy_lib is
importable (fresh laptop, CI, etc).
"""
from __future__ import annotations

import pandas as pd


class DirectRESTAdapter:
    """Hits `https://fapi.binance.com/fapi/v1/klines` directly."""

    def fetch_bars(self, symbol: str, timeframe: str,
                   limit: int = 200) -> pd.DataFrame:
        import requests   # noqa: PLC0415
        url = "https://fapi.binance.com/fapi/v1/klines"
        r = requests.get(url, params={
            "symbol":   symbol.upper(),
            "interval": timeframe,
            "limit":    limit,
        }, timeout=10)
        r.raise_for_status()
        raw = r.json()

        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "trades", "tbbav", "tbqav", "_"]
        df = pd.DataFrame(raw, columns=cols)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        df["close_time"] = df["close_time"].astype("int64")
        df.set_index("close_time", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
