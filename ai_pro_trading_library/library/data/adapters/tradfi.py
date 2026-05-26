"""TradFi (US stock / Yahoo Finance) adapter.

`atomic_strategy_lib.data.tradfi` reaches `query2.finance.yahoo.com`
directly. This adapter mirrors the same endpoints in pure REST form
(no `yfinance` dep required — yfinance is just a wrapper around the
same query2 endpoints).

Used by:
- ondo-daily case (Ondo tokenized stock background data)
- earnings-tradfi-scanner (US stock earnings dates + price quotes)
- macro-event-driven-scanner (FOMC / CPI / NFP impact assessment)
- black-swan-insurance (correlation with macro hedge assets)

For premium / corporate-action / fundamentals data the cases mostly
fall back to `BinanceProAdapter.tokenized_dynamic` which already
exposes the dataset via Binance's RWA endpoints (no Yahoo auth needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_YAHOO_QUERY = "https://query2.finance.yahoo.com"
_YAHOO_QUERY1 = "https://query1.finance.yahoo.com"


@dataclass
class YahooFinanceAdapter:
    """Thin wrapper for query2.finance.yahoo.com endpoints used by tradfi cases."""

    base_url: str = _YAHOO_QUERY
    base_url_v1: str = _YAHOO_QUERY1
    timeout: int = 30
    session: Any = None
    user_agent: str = "Mozilla/5.0 (compatible; ai-pro-trading-library)"

    def quote(self, symbol: str) -> Any:
        """`/v7/finance/quote?symbols=AAPL` — current price + 24h stats."""
        return self._get(
            f"{self.base_url}/v7/finance/quote",
            params={"symbols": symbol.upper()},
        )

    def chart(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        range_: str = "1mo",
    ) -> Any:
        """`/v8/finance/chart/AAPL?interval=1d&range=1mo` — OHLCV history."""
        return self._get(
            f"{self.base_url}/v8/finance/chart/{symbol.upper()}",
            params={"interval": interval, "range": range_},
        )

    def search(self, query: str, *, quotes_count: int = 10) -> Any:
        """`/v1/finance/search?q=Apple` — symbol lookup."""
        return self._get(
            f"{self.base_url_v1}/v1/finance/search",
            params={"q": query, "quotesCount": int(quotes_count), "newsCount": 0},
        )

    def options_calendar(self, symbol: str) -> Any:
        """`/v7/finance/options/AAPL` — option expirations + chain."""
        return self._get(f"{self.base_url}/v7/finance/options/{symbol.upper()}")

    def fundamentals(self, symbol: str, modules: tuple[str, ...] = ("earnings",)) -> Any:
        """`/v10/finance/quoteSummary/AAPL?modules=earnings,...` — fundamentals."""
        return self._get(
            f"{self.base_url}/v10/finance/quoteSummary/{symbol.upper()}",
            params={"modules": ",".join(modules)},
        )

    def _get(self, url: str, params: dict | None = None) -> Any:
        session = self._session()
        response = session.get(
            url,
            params=params or {},
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _session(self):
        if self.session is not None:
            return self.session
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "requests is required; install with "
                "`pip install ai-pro-trading-library[data]`"
            ) from exc
        return requests


__all__ = ["YahooFinanceAdapter"]
