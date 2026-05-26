"""Data-layer request contracts and adapter protocols.

Per `docs/architecture/data-layer-design.md`:

- L1 canonical Bar lives in `library.core.protocols.Bar`.
- L2 adapter Protocols are defined here.
- L3 concrete adapters live under `library.data.adapters.*`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ai_pro_trading_library.library.core.protocols import Bar


# ---------------------------------------------------------------------------
# Request dataclasses (one per endpoint family)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataRequest:
    """Generic data request — shape kept for backwards compatibility with
    existing test fixtures. Most adapter methods take the more specific
    `KlineRequest` / `FundingRequest` / etc. below."""

    symbol: str
    timeframe: str
    source: str = "local"
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class KlineRequest:
    instrument_id: str
    timeframe: str
    limit: int = 500
    start_ms: int | None = None
    end_ms: int | None = None
    market_type: str = "futures"  # "futures" or "spot"


@dataclass(frozen=True)
class FundingRequest:
    instrument_id: str
    limit: int = 200
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass(frozen=True)
class OIRequest:
    instrument_id: str
    period: str = "5m"
    limit: int = 500
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass(frozen=True)
class LongShortRequest:
    instrument_id: str
    period: str = "5m"
    limit: int = 500
    scope: str = "top_account"  # "top_account" / "top_position" / "global" / "taker"


@dataclass(frozen=True)
class TickerRequest:
    instrument_id: str | None = None
    market_type: str = "futures"


@dataclass(frozen=True)
class OrderbookRequest:
    instrument_id: str
    limit: int = 100  # 5 / 10 / 20 / 50 / 100 / 500 / 1000


# ---------------------------------------------------------------------------
# Adapter Protocols
# ---------------------------------------------------------------------------


class MarketDataAdapter(Protocol):
    """Klines / OHLCV data."""

    def fetch_klines(self, req: KlineRequest) -> list[Bar]: ...


class DerivativesDataAdapter(Protocol):
    """Funding / OI / long-short / taker volume."""

    def fetch_funding_history(self, req: FundingRequest) -> pd.DataFrame: ...
    def fetch_premium_index(self, req: TickerRequest) -> pd.DataFrame: ...
    def fetch_open_interest(self, req: TickerRequest) -> dict: ...
    def fetch_oi_history(self, req: OIRequest) -> pd.DataFrame: ...
    def fetch_long_short_ratio(self, req: LongShortRequest) -> pd.DataFrame: ...
    def fetch_taker_buy_sell(self, req: OIRequest) -> pd.DataFrame: ...


class TickerDataAdapter(Protocol):
    """24h ticker / latest price / exchange info."""

    def fetch_24h_tickers(self, req: TickerRequest) -> pd.DataFrame: ...
    def fetch_ticker_price(self, req: TickerRequest) -> pd.DataFrame: ...
    def fetch_exchange_info(self, req: TickerRequest) -> dict: ...


class OrderbookDataAdapter(Protocol):
    """Order book snapshots."""

    def fetch_orderbook(self, req: OrderbookRequest) -> dict: ...


__all__ = [
    "DataRequest",
    "DerivativesDataAdapter",
    "FundingRequest",
    "KlineRequest",
    "LongShortRequest",
    "MarketDataAdapter",
    "OIRequest",
    "OrderbookDataAdapter",
    "OrderbookRequest",
    "TickerDataAdapter",
    "TickerRequest",
]
