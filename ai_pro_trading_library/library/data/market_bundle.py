"""Case-friendly façade over a `MarketDataAdapter`.

Replaces `atomic_strategy_lib.data.market_bundle` cache-aware accessors
with a single class that takes any adapter (cached or not) and exposes
one-line accessors:

    from library.data.adapters.binance_rest import BinanceRestAdapter
    from library.data.adapters.market_cache import CachedAdapter
    from library.data.market_bundle import MarketBundleClient

    client = MarketBundleClient(CachedAdapter(BinanceRestAdapter()))
    bars = client.klines("BTCUSDT", "1h", limit=500)
    funding = client.funding("BTCUSDT")
    tickers = client.tickers()

The façade does not own a cache itself — the cache is the adapter
decoration. Callers who don't want caching pass the raw `BinanceRestAdapter`.

Note: the dataclass `library.core.protocols.MarketBundle` (which represents
a `Dict[(symbol, tf), List[Bar]]` data bundle for backtest snapshots) is a
**different concept** — that's a value object, this is the accessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ai_pro_trading_library.library.core.protocols import Bar
from ai_pro_trading_library.library.data.interfaces import (
    FundingRequest,
    KlineRequest,
    LongShortRequest,
    OIRequest,
    OrderbookRequest,
    TickerRequest,
)


@dataclass
class MarketBundleClient:
    """Case-friendly accessor over a `MarketDataAdapter`."""

    adapter: Any  # any adapter implementing the Protocols
    market_type: str = "futures"

    def klines(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Bar]:
        return self.adapter.fetch_klines(
            KlineRequest(
                instrument_id=symbol,
                timeframe=timeframe,
                limit=limit,
                start_ms=start_ms,
                end_ms=end_ms,
                market_type=self.market_type,
            )
        )

    def klines_multi_tf(
        self,
        symbol: str,
        timeframes: list[str],
        limit: int = 200,
    ) -> dict[str, list[Bar]]:
        if hasattr(self.adapter, "fetch_klines_multi_tf"):
            return self.adapter.fetch_klines_multi_tf(
                symbol, timeframes, limit=limit, market_type=self.market_type
            )
        return {tf: self.klines(symbol, tf, limit=limit) for tf in timeframes}

    def funding(
        self,
        symbol: str,
        limit: int = 200,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        return self.adapter.fetch_funding_history(
            FundingRequest(symbol, limit=limit, start_ms=start_ms, end_ms=end_ms)
        )

    def funding_current(self, symbol: str) -> pd.DataFrame:
        return self.adapter.fetch_premium_index(TickerRequest(symbol))

    def open_interest(self, symbol: str) -> dict:
        return self.adapter.fetch_open_interest(TickerRequest(symbol))

    def oi_history(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 500,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        return self.adapter.fetch_oi_history(
            OIRequest(
                instrument_id=symbol,
                period=period,
                limit=limit,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )

    def long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 500,
        scope: str = "top_account",
    ) -> pd.DataFrame:
        return self.adapter.fetch_long_short_ratio(
            LongShortRequest(symbol, period=period, limit=limit, scope=scope)
        )

    def taker_buy_sell(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 500,
    ) -> pd.DataFrame:
        return self.adapter.fetch_taker_buy_sell(
            OIRequest(instrument_id=symbol, period=period, limit=limit)
        )

    def tickers(self, symbol: str | None = None) -> pd.DataFrame:
        return self.adapter.fetch_24h_tickers(
            TickerRequest(instrument_id=symbol, market_type=self.market_type)
        )

    def ticker_price(self, symbol: str | None = None) -> pd.DataFrame:
        return self.adapter.fetch_ticker_price(
            TickerRequest(instrument_id=symbol, market_type=self.market_type)
        )

    def exchange_info(self) -> dict:
        return self.adapter.fetch_exchange_info(
            TickerRequest(market_type=self.market_type)
        )

    def orderbook(self, symbol: str, limit: int = 100) -> dict:
        return self.adapter.fetch_orderbook(OrderbookRequest(symbol, limit=limit))


__all__ = ["MarketBundleClient"]
