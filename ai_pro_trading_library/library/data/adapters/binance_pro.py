"""Binance Pro REST adapter — `bapi/bigdata` + `bapi/defi` endpoints.

Endpoint URLs taken from `binance-pro-cli/src/core.ts` and
`binance-pro-cli/src/web3-core.ts`. All endpoints in this adapter are
`publicRead: true` per the source TypeScript registry — no API key
required, just a JSON POST/GET to `https://www.binance.com` or
`https://web3.binance.com`.

Method patterns:
- bigdata search/skill endpoints (token-report, market-indicators,
  hot ranks, AI signals): POST to `https://www.binance.com<path>`
  with JSON body. Response is unwrapped via `.data` field.
- web3.binance.com endpoints: mix of GET and POST (see registry).
- RWA (Ondo tokenized) endpoints under `/wallet/market/token/rwa/*` —
  used by the new ondo-daily / earnings-tradfi / macro-event-driven
  cases.

The adapter is a thin pass-through: it knows the endpoint paths and
the standard "extract `data` field from the response" idiom. Caller
controls request bodies because each endpoint takes different params.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_BINANCE_BASE = "https://www.binance.com"
_WEB3_BASE = "https://web3.binance.com"


# ---------- bigdata search / skill registry ----------
# Source: binance-pro-cli/src/core.ts capabilityRegistry

BIGDATA_ENDPOINTS = {
    "token_report": "/bapi/bigdata/v1/public/bigdata/search/skill/token-report",
    "market_indicators": "/bapi/bigdata/v1/public/bigdata/search/skill/market-indicators",
    "ai_trending": "/bapi/bigdata/v1/public/bigdata/search/skill/getAITrending",
    "spot_hot_rank": "/bapi/bigdata/v1/public/bigdata/search/skill/getSpotHotRank",
    "futures_hot_rank": "/bapi/bigdata/v1/public/bigdata/search/skill/getFuturesHotRank",
    "alpha_hot_rank": "/bapi/bigdata/v1/public/bigdata/search/skill/getAlphaHotRank",
    "web3_hot_rank": "/bapi/bigdata/v1/public/bigdata/search/skill/getWeb3HotRank",
    "ai_signal": "/bapi/bigdata/v1/public/bigdata/search/skill/getAISignalByToken",
    "ai_signal_rank": "/bapi/bigdata/v1/public/bigdata/search/skill/getTradeSignalTopRank",
    "trading_bot_top": "/bapi/futures/v1/public/future/common/strategy/landing-page/queryTopStrategy",
    "trading_bot_top_um_dca": "/bapi/futures/v1/public/future/common/strategy/landing-page/queryTopUmDcaStrategy",
}


# ---------- web3.binance.com / RWA endpoints ----------
# Source: binance-pro-cli/src/web3-core.ts web3CapabilityRegistry

@dataclass(frozen=True)
class _Web3Endpoint:
    base: str  # "web3" | "www"
    path: str
    method: str  # "GET" | "POST"


WEB3_ENDPOINTS: dict[str, _Web3Endpoint] = {
    "web3_signal": _Web3Endpoint(
        "web3",
        "/bapi/defi/v1/public/wallet-direct/buw/wallet/web/signal/smart-money/ai",
        "POST",
    ),
    "web3_rank_social": _Web3Endpoint(
        "web3",
        "/bapi/defi/v1/public/wallet-direct/buw/wallet/market/token/pulse/social/hype/rank/leaderboard/ai",
        "GET",
    ),
    "web3_rank_token": _Web3Endpoint(
        "web3",
        "/bapi/defi/v1/public/wallet-direct/buw/wallet/market/token/pulse/unified/rank/list/ai",
        "POST",
    ),
    "web3_rank_smart_inflow": _Web3Endpoint(
        "web3",
        "/bapi/defi/v1/public/wallet-direct/tracker/wallet/token/inflow/rank/query/ai",
        "POST",
    ),
    "web3_rank_meme": _Web3Endpoint(
        "web3",
        "/bapi/defi/v1/public/wallet-direct/buw/wallet/market/token/pulse/exclusive/rank/list/ai",
        "GET",
    ),
    "web3_rank_pnl": _Web3Endpoint(
        "web3",
        "/bapi/defi/v1/public/wallet-direct/market/leaderboard/query/ai",
        "GET",
    ),
    "web3_address_pnl": _Web3Endpoint(
        "web3",
        "/bapi/defi/v3/public/wallet-direct/buw/wallet/address/pnl/active-position-list/ai",
        "GET",
    ),
    # RWA / Ondo tokenized stock endpoints — under www.binance.com
    "tokenized_list": _Web3Endpoint(
        "www",
        "/bapi/defi/v1/public/wallet-direct/buw/wallet/market/token/rwa/stock/detail/list/ai",
        "GET",
    ),
    "tokenized_meta": _Web3Endpoint(
        "www",
        "/bapi/defi/v1/public/wallet-direct/buw/wallet/market/token/rwa/meta/ai",
        "GET",
    ),
    "tokenized_market": _Web3Endpoint(
        "www",
        "/bapi/defi/v1/public/wallet-direct/buw/wallet/market/token/rwa/market/status/ai",
        "GET",
    ),
    "tokenized_asset": _Web3Endpoint(
        "www",
        "/bapi/defi/v1/public/wallet-direct/buw/wallet/market/token/rwa/asset/market/status/ai",
        "GET",
    ),
    "tokenized_dynamic": _Web3Endpoint(
        "www",
        "/bapi/defi/v2/public/wallet-direct/buw/wallet/market/token/rwa/dynamic/ai",
        "GET",
    ),
    "tokenized_kline": _Web3Endpoint(
        "www",
        "/bapi/defi/v1/public/wallet-direct/buw/wallet/dex/market/token/kline/ai",
        "GET",
    ),
}


def _extract_data(response_json: Any) -> Any:
    """Mirror `extractData` in binance-pro-cli core.ts: return the `data`
    field if present, otherwise return the whole payload."""
    if isinstance(response_json, dict) and "data" in response_json:
        return response_json["data"]
    return response_json


@dataclass
class BinanceProAdapter:
    """Pro REST endpoints (publicRead) for cases that need hot ranks,
    AI signals, on-chain data, or tokenized stock info.

    No auth needed for the publicRead capabilities ported here; subclass
    if/when the auth-gated endpoints get added.
    """

    base_url: str = _BINANCE_BASE
    web3_base_url: str = _WEB3_BASE
    timeout: int = 30
    session: Any = None

    # ---------- bigdata helpers ----------

    def call_bigdata(self, capability: str, body: dict[str, Any] | None = None) -> Any:
        if capability not in BIGDATA_ENDPOINTS:
            raise KeyError(f"unknown bigdata capability: {capability}")
        url = self.base_url.rstrip("/") + BIGDATA_ENDPOINTS[capability]
        return _extract_data(self._post(url, body or {}))

    def token_report(self, *, token: str, product: str = "spot") -> Any:
        return self.call_bigdata("token_report", {"token": token.upper(), "product": product})

    def market_indicators(self, *, token: str, product: str = "spot") -> Any:
        return self.call_bigdata("market_indicators", {"token": token.upper(), "product": product})

    def hot_rank(self, *, kind: str = "spot", country_code: str | None = None) -> Any:
        """`kind` ∈ {"spot", "futures", "alpha", "web3"}."""
        cap = f"{kind}_hot_rank"
        body: dict[str, Any] = {}
        if country_code:
            body["countryCode"] = country_code
        return self.call_bigdata(cap, body)

    def ai_signal(self, *, token: str) -> Any:
        return self.call_bigdata("ai_signal", {"token": token.upper()})

    def ai_signal_rank(self, *, top_n: int = 10) -> Any:
        return self.call_bigdata("ai_signal_rank", {"topN": int(top_n)})

    def ai_trending(self) -> Any:
        return self.call_bigdata("ai_trending", {})

    def trading_bot_top(self, *, dca: bool = False) -> Any:
        cap = "trading_bot_top_um_dca" if dca else "trading_bot_top"
        return self.call_bigdata(cap, {})

    # ---------- web3 helpers ----------

    def call_web3(
        self,
        capability: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        if capability not in WEB3_ENDPOINTS:
            raise KeyError(f"unknown web3 capability: {capability}")
        ep = WEB3_ENDPOINTS[capability]
        base = self.web3_base_url if ep.base == "web3" else self.base_url
        url = base.rstrip("/") + ep.path
        if ep.method == "POST":
            return _extract_data(self._post(url, body or {}))
        return _extract_data(self._get(url, params or {}))

    def smart_money_signals(self, *, body: dict[str, Any] | None = None) -> Any:
        return self.call_web3("web3_signal", body=body or {})

    def social_hype_rank(self, *, params: dict[str, Any] | None = None) -> Any:
        return self.call_web3("web3_rank_social", params=params or {})

    def unified_token_rank(self, *, body: dict[str, Any] | None = None) -> Any:
        return self.call_web3("web3_rank_token", body=body or {})

    def smart_money_inflow(self, *, body: dict[str, Any] | None = None) -> Any:
        return self.call_web3("web3_rank_smart_inflow", body=body or {})

    def meme_rank(self, *, params: dict[str, Any] | None = None) -> Any:
        return self.call_web3("web3_rank_meme", params=params or {})

    def trader_leaderboard(self, *, params: dict[str, Any] | None = None) -> Any:
        return self.call_web3("web3_rank_pnl", params=params or {})

    def address_holdings(self, *, address: str) -> Any:
        return self.call_web3("web3_address_pnl", params={"address": address})

    def tokenized_stock_list(self, *, params: dict[str, Any] | None = None) -> Any:
        """Ondo tokenized US-stock list. Used by ondo-daily case."""
        return self.call_web3("tokenized_list", params=params or {})

    def tokenized_meta(self, *, symbol: str) -> Any:
        # Tokenized stock symbols preserve "AAPLon"-style mixed casing
        return self.call_web3("tokenized_meta", params={"symbol": symbol})

    def tokenized_market_status(self) -> Any:
        return self.call_web3("tokenized_market")

    def tokenized_asset_status(self, *, symbol: str) -> Any:
        return self.call_web3("tokenized_asset", params={"symbol": symbol})

    def tokenized_dynamic(self, *, symbol: str) -> Any:
        return self.call_web3("tokenized_dynamic", params={"symbol": symbol})

    def tokenized_kline(
        self,
        *,
        chain_id: str | None = None,
        contract_address: str | None = None,
        symbol: str | None = None,
        interval: str = "1d",
        limit: int = 30,
    ) -> Any:
        """Tokenized stock klines.

        The endpoint identifies the token by `(chainId, contractAddress)` —
        not by ticker. `symbol` is accepted for ergonomic callers but is
        currently ignored by the upstream API. Pass chain_id +
        contract_address from the row in `tokenized_stock_list()`.
        """
        params: dict[str, Any] = {"interval": interval, "limit": int(limit)}
        if chain_id is not None:
            params["chainId"] = str(chain_id)
        if contract_address is not None:
            params["contractAddress"] = contract_address
        if symbol is not None:
            params["symbol"] = symbol
        return self.call_web3("tokenized_kline", params=params)

    # ---------- internals ----------

    def _post(self, url: str, body: dict[str, Any]) -> Any:
        session = self._session()
        response = session.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _get(self, url: str, params: dict[str, Any]) -> Any:
        session = self._session()
        response = session.get(url, params=params, timeout=self.timeout)
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


__all__ = [
    "BIGDATA_ENDPOINTS",
    "BinanceProAdapter",
    "WEB3_ENDPOINTS",
]
