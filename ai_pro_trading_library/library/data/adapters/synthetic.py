"""Synthetic OHLCV adapter — deterministic Bar generator for tests/smoke.

Replaces `runtime/smoke.py::synthetic_ohlcv` (DataFrame) with the canonical
`MarketDataAdapter` Protocol. Same deterministic output, but `list[Bar]`
shape so any code written against an adapter Protocol works with this for
free.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai_pro_trading_library.library.core.protocols import Bar
from ai_pro_trading_library.library.data.interfaces import (
    FundingRequest,
    KlineRequest,
    OIRequest,
    TickerRequest,
)


@dataclass
class SyntheticAdapter:
    """Deterministic OHLCV / funding / OI / ticker for tests.

    The synthetic series is the same shape `runtime/smoke.py` has used since
    Phase 6, just exposed as canonical `Bar` lists.
    """

    base_price: float = 100.0
    drift: float = 0.4
    noise_amplitude: float = 0.7
    timestamp_step_ms: int = 3_600_000  # 1h bars by default
    base_timestamp_ms: int = 1_700_000_000_000

    def fetch_klines(self, req: KlineRequest) -> list[Bar]:
        rows = max(int(req.limit), 1)
        step = self.timestamp_step_ms
        bars: list[Bar] = []
        for i in range(rows):
            close = self.base_price + i * self.drift + ((i % 9) - 4) * self.noise_amplitude
            open_ = close - self.drift
            high = close + 1.5
            low = close - 1.5
            ts = self.base_timestamp_ms + (i + 1) * step
            bars.append(
                Bar(
                    instrument_id=req.instrument_id.upper(),
                    timeframe=req.timeframe,
                    timestamp=ts,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=1000.0 + (i % 13) * 25,
                    quote_volume=(1000.0 + (i % 13) * 25) * close,
                    trades=50 + (i % 7),
                    taker_buy_base=500.0,
                    taker_buy_quote=500.0 * close,
                    open_time=self.base_timestamp_ms + i * step,
                )
            )
        return bars

    def fetch_funding_history(self, req: FundingRequest) -> pd.DataFrame:
        rows = max(int(req.limit), 1)
        step = 8 * 3_600_000  # funding settles every 8h
        return pd.DataFrame(
            [
                {
                    "timestamp": self.base_timestamp_ms + i * step,
                    "funding_rate": 0.0001 + (i % 5 - 2) * 0.00002,
                }
                for i in range(rows)
            ]
        )

    def fetch_oi_history(self, req: OIRequest) -> pd.DataFrame:
        rows = max(int(req.limit), 1)
        step = self.timestamp_step_ms
        return pd.DataFrame(
            [
                {
                    "timestamp": self.base_timestamp_ms + i * step,
                    "oi": 10_000.0 + i * 5.0,
                    "oi_value": (10_000.0 + i * 5.0) * (self.base_price + i * self.drift),
                }
                for i in range(rows)
            ]
        )

    def fetch_24h_tickers(self, req: TickerRequest) -> pd.DataFrame:
        symbols = (
            [req.instrument_id.upper()]
            if req.instrument_id
            else ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        )
        return pd.DataFrame(
            [
                {
                    "symbol": s,
                    "lastPrice": self.base_price + (i + 1) * 10,
                    "priceChangePercent": (i + 1) * 0.5,
                    "volume": 100_000.0 * (i + 1),
                    "quoteVolume": 100_000.0 * self.base_price * (i + 1),
                }
                for i, s in enumerate(symbols)
            ]
        )

    def fetch_ticker_price(self, req: TickerRequest) -> pd.DataFrame:
        symbols = (
            [req.instrument_id.upper()]
            if req.instrument_id
            else ["BTCUSDT", "ETHUSDT"]
        )
        return pd.DataFrame(
            [
                {"symbol": s, "price": self.base_price + (i + 1) * 10}
                for i, s in enumerate(symbols)
            ]
        )


__all__ = ["SyntheticAdapter"]
