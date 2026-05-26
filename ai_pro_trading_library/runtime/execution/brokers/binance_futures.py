"""Binance USD-M Futures broker (testnet + mainnet).

Mirrors `cyqnt_trd.standard_bot.execution.binance_futures_testnet` and
its mainnet sibling, adapted to the new repo's `ExecutionIntent` /
`OrderIntent` shapes.

## Safety design

- `testnet=True` is the default. Constructing without overrides hits
  `https://testnet.binancefuture.com`.
- Mainnet (`testnet=False`) requires `confirm_live=True` — without it,
  construction raises ValueError.
- Optional `risk_planner: ExecutionPlanner` runs intents through risk
  rules before any signed REST call (same pattern as `PaperBroker`).
- Optional `idempotency_store: IdempotencyStore` persists
  `client_order_id → order_id` mapping; resubmitting an intent with a
  previously seen client_order_id returns the cached id without a new
  signed REST call. Survives process restarts.
- Same `submit_intent(intent, *, price, snapshot)` API as `PaperBroker`,
  so cases can swap broker via dependency injection.

## Endpoints used

- `POST /fapi/v1/order` — place order
- `DELETE /fapi/v1/order` — cancel by client_order_id
- `GET /fapi/v2/account` — account snapshot
- `POST /fapi/v1/leverage` — set per-symbol leverage
- `POST /fapi/v1/marginType` — set ISOLATED / CROSSED margin mode
- `POST /fapi/v1/listenKey` — start user-data stream (returns listenKey)
- `PUT /fapi/v1/listenKey` — keepalive (extend 60-min TTL)
- `DELETE /fapi/v1/listenKey` — close stream

All signed with HMAC-SHA256 on the query string per Binance spec.

## Deployment-specific confirmation policy

`confirm_live=True` is a static flag — appropriate for a developer
acknowledging mainnet risk at construction time. Production deployments
that need a per-trade approval gate (chat ack, signed webhook,
scheduled-window check) should compose the broker behind their own
gate Protocol; see the deployment notes in
`docs/migration/live-broker-deferred.md`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

if TYPE_CHECKING:
    from ai_pro_trading_library.library.core.protocols import (
        AccountSnapshot,
        ExecutionIntent,
    )
    from ai_pro_trading_library.runtime.execution.rules import ExecutionPlanner


_TESTNET_BASE = "https://testnet.binancefuture.com"
_MAINNET_BASE = "https://fapi.binance.com"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _format_quantity(value: float) -> str:
    text = format(float(value), ".12f").rstrip("0").rstrip(".")
    return text or "0"


def _truncate_client_order_id(value: str) -> str:
    return value[:36]


@dataclass
class IdempotencyStore:
    """Persistent client_order_id → order_id map backed by a JSON file.

    The store is *additive* — successful submissions are written through
    immediately. A resubmission with the same client_order_id returns the
    cached order_id without calling the exchange. This survives process
    restarts.

    Concurrency: this is a single-process store. Multi-process
    deployments should swap in a Redis or database-backed implementation
    that satisfies the same interface (`has`, `get`, `record`, `clear`).
    """

    path: Path
    _mapping: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            try:
                self._mapping = dict(json.loads(self.path.read_text()))
            except (json.JSONDecodeError, OSError):
                self._mapping = {}

    def has(self, client_order_id: str) -> bool:
        return client_order_id in self._mapping

    def get(self, client_order_id: str) -> str | None:
        return self._mapping.get(client_order_id)

    def record(self, client_order_id: str, order_id: str) -> None:
        self._mapping[client_order_id] = str(order_id)
        self._flush()

    def clear(self) -> None:
        self._mapping.clear()
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._mapping, indent=2, sort_keys=True))


@dataclass
class BinanceFuturesBroker:
    """Live USD-M Futures broker. Defaults to testnet for safety.

    Construct with API credentials. To target mainnet, pass both
    `testnet=False` AND `confirm_live=True`. Anything else raises.

    Submit orders via `submit_intent(ExecutionIntent, *, price, snapshot)`.
    The broker:
    1. Runs the intent through `risk_planner` (if set) and rejects
       on the first failed rule.
    2. Sizes the order: `quantity = intent.notional / price`.
    3. Sends a signed POST `/fapi/v1/order`.
    4. Returns the Binance `orderId` on success, None if risk rejects.
    """

    api_key: str
    api_secret: str
    testnet: bool = True
    confirm_live: bool = False
    risk_planner: "ExecutionPlanner | None" = None
    idempotency_store: IdempotencyStore | None = None
    base_url_override: str | None = None
    recv_window: int = 5000
    timeout: int = 30
    session: Any = None  # requests.Session-compatible
    rejections: list[tuple["ExecutionIntent", str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.testnet and not self.confirm_live:
            raise ValueError(
                "Mainnet trading requires confirm_live=True. "
                "Pass testnet=True (default) for testnet, or "
                "explicit confirm_live=True to acknowledge mainnet risk."
            )
        if not self.api_key or not self.api_secret:
            raise ValueError("api_key and api_secret are required")

    @property
    def base_url(self) -> str:
        if self.base_url_override:
            return self.base_url_override
        return _TESTNET_BASE if self.testnet else _MAINNET_BASE

    # ---------- public API ----------

    def submit_intent(
        self,
        intent: "ExecutionIntent",
        *,
        price: float,
        snapshot: "AccountSnapshot | None" = None,
    ) -> str | None:
        """Risk-gate, size, and place a signed order. Returns orderId or None.

        If an `idempotency_store` is configured and the intent's
        `client_order_id` was previously submitted successfully, returns
        the cached order_id without a new REST call.
        """
        if self.risk_planner is not None:
            approved, rejected = self.risk_planner.plan([intent], snapshot)
            if rejected:
                self.rejections.extend(rejected)
                return None
        if price <= 0:
            raise ValueError("price must be > 0 to size the order")

        coid = (
            _truncate_client_order_id(intent.client_order_id)
            if intent.client_order_id else None
        )
        if coid and self.idempotency_store and self.idempotency_store.has(coid):
            return self.idempotency_store.get(coid)

        quantity = intent.notional / price
        params = {
            "symbol": intent.symbol.upper(),
            "side": "BUY" if intent.side.lower() == "buy" else "SELL",
            "type": intent.order_type.upper(),
            "quantity": _format_quantity(quantity),
        }
        if coid:
            params["newClientOrderId"] = coid
        if intent.reduce_only:
            params["reduceOnly"] = "true"
        if intent.order_type.upper() == "MARKET":
            params["newOrderRespType"] = "RESULT"
        payload = self._signed_request("POST", "/fapi/v1/order", params)
        order_id = payload.get("orderId")
        if order_id is None:
            return None
        order_id_str = str(order_id)
        if coid and self.idempotency_store:
            self.idempotency_store.record(coid, order_id_str)
        return order_id_str

    def cancel_order(self, *, symbol: str, client_order_id: str) -> dict[str, Any]:
        return self._signed_request("DELETE", "/fapi/v1/order", {
            "symbol": symbol.upper(),
            "origClientOrderId": _truncate_client_order_id(client_order_id),
        })

    # ---------- leverage / margin ----------

    def set_leverage(self, *, symbol: str, leverage: int) -> dict[str, Any]:
        """POST /fapi/v1/leverage. Returns the response payload (includes
        the post-change leverage and max notional)."""
        if leverage < 1 or leverage > 125:
            raise ValueError(f"leverage must be in [1, 125], got {leverage}")
        return self._signed_request("POST", "/fapi/v1/leverage", {
            "symbol": symbol.upper(),
            "leverage": int(leverage),
        })

    def set_margin_type(self, *, symbol: str, margin_type: str) -> dict[str, Any]:
        """POST /fapi/v1/marginType. `margin_type` ∈ {ISOLATED, CROSSED}."""
        upper = margin_type.upper()
        if upper not in ("ISOLATED", "CROSSED"):
            raise ValueError(f"margin_type must be ISOLATED or CROSSED, got {margin_type!r}")
        return self._signed_request("POST", "/fapi/v1/marginType", {
            "symbol": symbol.upper(),
            "marginType": upper,
        })

    # ---------- user-data stream lifecycle (REST only) ----------
    #
    # The websocket consumer that ingests fills/order updates from a
    # listenKey is intentionally not part of this broker — it would
    # introduce a long-running async loop that belongs in a runtime
    # supervisor, not here. These three methods cover the listenKey
    # life-cycle so downstream code can plug in its own consumer.

    def start_user_data_stream(self) -> str:
        """POST /fapi/v1/listenKey. Returns the new listenKey (60-min TTL)."""
        payload = self._unsigned_post("/fapi/v1/listenKey")
        listen_key = payload.get("listenKey")
        if not isinstance(listen_key, str) or not listen_key:
            raise RuntimeError(f"unexpected listenKey response: {payload!r}")
        return listen_key

    def keepalive_user_data_stream(self, listen_key: str | None = None) -> dict[str, Any]:
        """PUT /fapi/v1/listenKey. Extends the listenKey for another 60 min.

        Per Binance docs, the listenKey is per-API-key on USD-M futures;
        passing a specific key is informational only — the request is
        scoped to the API key in the X-MBX-APIKEY header.
        """
        return self._unsigned_request("PUT", "/fapi/v1/listenKey")

    def close_user_data_stream(self, listen_key: str | None = None) -> dict[str, Any]:
        """DELETE /fapi/v1/listenKey. Tells Binance to invalidate the key."""
        return self._unsigned_request("DELETE", "/fapi/v1/listenKey")

    def account_snapshot(self) -> "AccountSnapshot":
        """Fetch /fapi/v2/account and project to AccountSnapshot."""
        from ai_pro_trading_library.library.core.protocols import (
            AccountPosition,
            AccountSnapshot,
        )

        payload = self._signed_request("GET", "/fapi/v2/account", {})
        positions = tuple(
            AccountPosition(
                symbol=p["symbol"],
                side="long" if float(p.get("positionAmt", 0)) >= 0 else "short",
                notional=abs(float(p.get("positionAmt", 0))) * float(p.get("entryPrice", 0)),
                unrealized_pnl=float(p.get("unrealizedProfit", 0)),
            )
            for p in payload.get("positions", [])
            if float(p.get("positionAmt", 0)) != 0.0
        )
        return AccountSnapshot(
            equity=float(payload.get("totalWalletBalance", 0)),
            available_balance=float(payload.get("availableBalance", 0)),
            positions=positions,
        )

    # ---------- internals ----------

    def _sign(self, query: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _signed_request(self, method: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "requests is required; install with "
                "`pip install ai-pro-trading-library[data]`"
            ) from exc
        session = self.session or requests
        body = dict(params)
        body["timestamp"] = _now_ms()
        body["recvWindow"] = self.recv_window
        query = urlencode(body, doseq=True)
        signature = self._sign(query)
        url = f"{self.base_url}{path}?{query}&signature={signature}"
        headers = {"X-MBX-APIKEY": self.api_key}
        response = session.request(method, url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _unsigned_request(self, method: str, path: str) -> dict[str, Any]:
        """Listen-key endpoints are scoped by API key header but do not
        require an HMAC signature."""
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "requests is required; install with "
                "`pip install ai-pro-trading-library[data]`"
            ) from exc
        session = self.session or requests
        url = f"{self.base_url}{path}"
        headers = {"X-MBX-APIKEY": self.api_key}
        response = session.request(method, url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        text = response.text or ""
        if not text:
            return {}
        try:
            return response.json()
        except (ValueError, json.JSONDecodeError):
            return {}

    def _unsigned_post(self, path: str) -> dict[str, Any]:
        return self._unsigned_request("POST", path)


__all__ = ["BinanceFuturesBroker", "IdempotencyStore"]
