"""Direct-REST market data adapter for Binance fapi/api.

Replaces:
- `cyqnt_trd.blocks.data.fetch_*` (already direct REST — same contract)
- `atomic_strategy_lib.data.{kline, funding, open_interest, ticker, orderbook}`
  CLI subprocess wrappers (dropped per `dependency-strategy.md`)
- `cyqnt_trd.standard_bot.data.adapters.BinanceRestMarketDataAdapter` (parsed
  klines into `Bar` already; same shape as us).

L2 parity vs `blocks.data` (since blocks already does direct REST). The
adapter implements the `MarketDataAdapter` / `DerivativesDataAdapter` /
`TickerDataAdapter` / `OrderbookDataAdapter` Protocols.

`requests` is imported lazily — only required when actually fetching
(install via `pip install ai-pro-trading-library[data]`). The
`_request_json` instance method is the single network seam, so tests
can monkey-patch it without touching `requests`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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


# Endpoint base URLs for Binance USDT-margined futures + spot
_FAPI_BASE = "https://fapi.binance.com"
_API_BASE = "https://api.binance.com"

_FAPI_KLINES = f"{_FAPI_BASE}/fapi/v1/klines"
_FAPI_24H = f"{_FAPI_BASE}/fapi/v1/ticker/24hr"
_FAPI_PRICE = f"{_FAPI_BASE}/fapi/v1/ticker/price"
_FAPI_FUNDING = f"{_FAPI_BASE}/fapi/v1/fundingRate"
_FAPI_PREMIUM = f"{_FAPI_BASE}/fapi/v1/premiumIndex"
_FAPI_EXINFO = f"{_FAPI_BASE}/fapi/v1/exchangeInfo"
_FAPI_DEPTH = f"{_FAPI_BASE}/fapi/v1/depth"
_FAPI_OI = f"{_FAPI_BASE}/futures/data/openInterestHist"
_FAPI_OI_NOW = f"{_FAPI_BASE}/fapi/v1/openInterest"
_FAPI_TOPLS_ACC = f"{_FAPI_BASE}/futures/data/topLongShortAccountRatio"
_FAPI_TOPLS_POS = f"{_FAPI_BASE}/futures/data/topLongShortPositionRatio"
_FAPI_GLOBALLS = f"{_FAPI_BASE}/futures/data/globalLongShortAccountRatio"
_FAPI_TAKER = f"{_FAPI_BASE}/futures/data/takerlongshortRatio"

_API_KLINES = f"{_API_BASE}/api/v3/klines"
_API_24H = f"{_API_BASE}/api/v3/ticker/24hr"
_API_PRICE = f"{_API_BASE}/api/v3/ticker/price"
_API_EXINFO = f"{_API_BASE}/api/v3/exchangeInfo"
_API_DEPTH = f"{_API_BASE}/api/v3/depth"


@dataclass
class BinanceRestAdapter:
    """REST adapter for Binance public endpoints.

    Implements `MarketDataAdapter`, `DerivativesDataAdapter`,
    `TickerDataAdapter`, and `OrderbookDataAdapter` Protocols.
    """

    timeout: float = 15.0
    retries: int = 3
    user_agent: str = "ai-pro-trading-library/0.1"
    market_type: str = "futures"  # default; per-call override available

    # ---------------------------------------------------------------------
    # Network seam — override or monkey-patch in tests
    # ---------------------------------------------------------------------

    def _request_json(self, url: str, params: dict[str, Any]) -> Any:
        """GET `url` with `params`, retrying on transient errors. Lazy-imports
        `requests`. Single seam for testability."""
        try:
            import requests  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "requests is not installed. Install via "
                "'pip install ai-pro-trading-library[data]' or "
                "'pip install requests'."
            ) from e

        last_exc: BaseException | None = None
        for attempt in range(self.retries):
            try:
                resp = requests.get(
                    url,
                    params={k: v for k, v in params.items() if v is not None},
                    timeout=self.timeout,
                    headers={"User-Agent": self.user_agent},
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as exc:  # type: ignore[attr-defined]
                last_exc = exc
                if attempt < self.retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(
            f"GET {url} failed after {self.retries} attempts: {last_exc}"
        )

    # ---------------------------------------------------------------------
    # MarketDataAdapter
    # ---------------------------------------------------------------------

    def fetch_klines(self, req: KlineRequest) -> list[Bar]:
        """Klines from fapi or api → canonical `list[Bar]`."""
        if req.market_type not in ("futures", "spot"):
            raise ValueError(f"market_type must be 'futures' or 'spot', got {req.market_type!r}")
        url = _FAPI_KLINES if req.market_type == "futures" else _API_KLINES
        params = {
            "symbol": req.instrument_id.upper(),
            "interval": req.timeframe,
            "limit": min(int(req.limit), 1500),
            "startTime": req.start_ms,
            "endTime": req.end_ms,
        }
        raw = self._request_json(url, params)
        return [self._parse_kline_row(req.instrument_id, req.timeframe, row) for row in raw]

    @staticmethod
    def _parse_kline_row(instrument_id: str, timeframe: str, row: list) -> Bar:
        """Binance klines schema:
        [open_time, open, high, low, close, volume, close_time, quote_volume,
         trades, taker_buy_base, taker_buy_quote, ignore]
        """
        return Bar(
            instrument_id=instrument_id.upper(),
            timeframe=timeframe,
            timestamp=int(row[6]),  # close_time
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            quote_volume=float(row[7]) if len(row) > 7 else None,
            trades=int(row[8]) if len(row) > 8 else None,
            taker_buy_base=float(row[9]) if len(row) > 9 else None,
            taker_buy_quote=float(row[10]) if len(row) > 10 else None,
            open_time=int(row[0]),
            confirmed=True,
        )

    def fetch_klines_multi_tf(
        self,
        instrument_id: str,
        timeframes: list[str],
        limit: int = 200,
        market_type: str = "futures",
    ) -> dict[str, list[Bar]]:
        """For-loop convenience over `fetch_klines`."""
        out: dict[str, list[Bar]] = {}
        for tf in timeframes:
            out[tf] = self.fetch_klines(
                KlineRequest(
                    instrument_id=instrument_id,
                    timeframe=tf,
                    limit=limit,
                    market_type=market_type,
                )
            )
        return out

    # ---------------------------------------------------------------------
    # DerivativesDataAdapter
    # ---------------------------------------------------------------------

    def fetch_funding_history(self, req: FundingRequest) -> pd.DataFrame:
        """Historical funding rate. Columns: `timestamp, funding_rate`."""
        raw = self._request_json(
            _FAPI_FUNDING,
            {
                "symbol": req.instrument_id.upper(),
                "limit": min(int(req.limit), 1000),
                "startTime": req.start_ms,
                "endTime": req.end_ms,
            },
        )
        return pd.DataFrame(
            [
                {
                    "timestamp": int(r["fundingTime"]),
                    "funding_rate": float(r["fundingRate"]),
                }
                for r in raw
            ]
        )

    def fetch_premium_index(self, req: TickerRequest) -> pd.DataFrame:
        """Mark / index / latest funding snapshot."""
        params = {"symbol": req.instrument_id.upper()} if req.instrument_id else {}
        raw = self._request_json(_FAPI_PREMIUM, params)
        if not isinstance(raw, list):
            raw = [raw]
        df = pd.DataFrame(raw)
        if df.empty:
            return df
        numeric = ["markPrice", "indexPrice", "lastFundingRate", "interestRate",
                   "estimatedSettlePrice"]
        for c in numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def fetch_open_interest(self, req: TickerRequest) -> dict:
        """Latest open interest for a symbol."""
        if not req.instrument_id:
            raise ValueError("OI fetch requires instrument_id")
        return self._request_json(
            _FAPI_OI_NOW, {"symbol": req.instrument_id.upper()}
        )

    def fetch_oi_history(self, req: OIRequest) -> pd.DataFrame:
        """Historical OI. Columns: `timestamp, oi, oi_value`."""
        raw = self._request_json(
            _FAPI_OI,
            {
                "symbol": req.instrument_id.upper(),
                "period": req.period,
                "limit": min(int(req.limit), 500),
                "startTime": req.start_ms,
                "endTime": req.end_ms,
            },
        )
        return pd.DataFrame(
            [
                {
                    "timestamp": int(r["timestamp"]),
                    "oi": float(r["sumOpenInterest"]),
                    "oi_value": float(r["sumOpenInterestValue"]),
                }
                for r in raw
            ]
        )

    def fetch_long_short_ratio(self, req: LongShortRequest) -> pd.DataFrame:
        """Long/short ratio (top_account / top_position / global)."""
        urls = {
            "top_account": _FAPI_TOPLS_ACC,
            "top_position": _FAPI_TOPLS_POS,
            "global": _FAPI_GLOBALLS,
        }
        if req.scope not in urls:
            raise ValueError(
                f"scope must be one of {sorted(urls)}, got {req.scope!r}"
            )
        raw = self._request_json(
            urls[req.scope],
            {
                "symbol": req.instrument_id.upper(),
                "period": req.period,
                "limit": min(int(req.limit), 500),
            },
        )
        return pd.DataFrame(
            [
                {
                    "timestamp": int(r["timestamp"]),
                    "long_short_ratio": float(r["longShortRatio"]),
                    "long_account": float(
                        r.get("longAccount", r.get("longPosition", "nan"))
                    ),
                    "short_account": float(
                        r.get("shortAccount", r.get("shortPosition", "nan"))
                    ),
                }
                for r in raw
            ]
        )

    def fetch_taker_buy_sell(self, req: OIRequest) -> pd.DataFrame:
        """Taker buy / sell volume ratio."""
        raw = self._request_json(
            _FAPI_TAKER,
            {
                "symbol": req.instrument_id.upper(),
                "period": req.period,
                "limit": min(int(req.limit), 500),
            },
        )
        return pd.DataFrame(
            [
                {
                    "timestamp": int(r["timestamp"]),
                    "buy_sell_ratio": float(r["buySellRatio"]),
                    "buy_volume": float(r["buyVol"]),
                    "sell_volume": float(r["sellVol"]),
                }
                for r in raw
            ]
        )

    # ---------------------------------------------------------------------
    # TickerDataAdapter
    # ---------------------------------------------------------------------

    def fetch_24h_tickers(self, req: TickerRequest) -> pd.DataFrame:
        """Full 24h ticker snapshot for the universe (or a single symbol)."""
        if req.market_type == "futures":
            url = _FAPI_24H
        elif req.market_type == "spot":
            url = _API_24H
        else:
            raise ValueError(
                f"market_type must be 'futures' or 'spot', got {req.market_type!r}"
            )
        params = {"symbol": req.instrument_id.upper()} if req.instrument_id else {}
        raw = self._request_json(url, params)
        if not isinstance(raw, list):
            raw = [raw]
        df = pd.DataFrame(raw)
        if df.empty:
            return df
        numeric = ["priceChange", "priceChangePercent", "weightedAvgPrice",
                   "lastPrice", "openPrice", "highPrice", "lowPrice",
                   "volume", "quoteVolume"]
        for c in numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def fetch_ticker_price(self, req: TickerRequest) -> pd.DataFrame:
        """Latest price snapshot. Columns: `symbol, price`."""
        if req.market_type == "futures":
            url = _FAPI_PRICE
        elif req.market_type == "spot":
            url = _API_PRICE
        else:
            raise ValueError(
                f"market_type must be 'futures' or 'spot', got {req.market_type!r}"
            )
        params = {"symbol": req.instrument_id.upper()} if req.instrument_id else {}
        raw = self._request_json(url, params)
        if not isinstance(raw, list):
            raw = [raw]
        df = pd.DataFrame(raw)
        if df.empty:
            return df
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df

    def fetch_exchange_info(self, req: TickerRequest) -> dict:
        """Symbol metadata (lot size, min notional, …)."""
        if req.market_type == "futures":
            url = _FAPI_EXINFO
        elif req.market_type == "spot":
            url = _API_EXINFO
        else:
            raise ValueError(
                f"market_type must be 'futures' or 'spot', got {req.market_type!r}"
            )
        return self._request_json(url, {})

    # ---------------------------------------------------------------------
    # OrderbookDataAdapter
    # ---------------------------------------------------------------------

    def fetch_orderbook(self, req: OrderbookRequest) -> dict:
        """Top-of-book depth snapshot. Returns the raw `{lastUpdateId, bids, asks}`
        Binance shape (downstream code can compute imbalance / spread)."""
        url = _FAPI_DEPTH if self.market_type == "futures" else _API_DEPTH
        return self._request_json(
            url,
            {
                "symbol": req.instrument_id.upper(),
                "limit": min(int(req.limit), 1000),
            },
        )


__all__ = ["BinanceRestAdapter"]
