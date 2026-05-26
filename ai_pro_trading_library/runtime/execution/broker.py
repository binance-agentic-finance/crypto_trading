"""Execution boundary for paper/live brokers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ai_pro_trading_library.library.core.protocols import (
        AccountSnapshot,
        ExecutionIntent,
    )
    from ai_pro_trading_library.runtime.execution.rules import ExecutionPlanner


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    reduce_only: bool = False
    client_order_id: str | None = None


class Broker(Protocol):
    def submit(self, order: OrderIntent) -> str:
        ...


class PaperBroker:
    """In-memory paper broker.

    Idempotent on `client_order_id`: re-submitting the same order id returns
    the original ack id without producing a new order.

    When constructed with `risk_planner`, calling `submit_intent` runs the
    `ExecutionIntent` through the planner before placing the order. Rejected
    intents are recorded in `rejections` and `submit_intent` returns None
    instead of an ack id. This satisfies the standard_bot acceptance gate
    "execution planner blocks invalid orders before any broker adapter call".
    """

    def __init__(self, *, risk_planner: "ExecutionPlanner | None" = None) -> None:
        self.orders: list[OrderIntent] = []
        self._by_client_id: dict[str, str] = {}
        self.risk_planner = risk_planner
        self.rejections: list[tuple["ExecutionIntent", str]] = []

    def submit(self, order: OrderIntent) -> str:
        if order.quantity <= 0:
            raise ValueError("order quantity must be > 0")
        if order.client_order_id and order.client_order_id in self._by_client_id:
            return self._by_client_id[order.client_order_id]
        self.orders.append(order)
        ack = f"paper-{len(self.orders)}"
        if order.client_order_id:
            self._by_client_id[order.client_order_id] = ack
        return ack

    def submit_intent(
        self,
        intent: "ExecutionIntent",
        *,
        price: float,
        snapshot: "AccountSnapshot | None" = None,
    ) -> str | None:
        """Run intent through risk_planner; if approved, place an OrderIntent.

        Returns the ack id on success, or None if the planner rejected the
        intent. The rejection reason is appended to `self.rejections`.

        `price` is required to convert `intent.notional` → order quantity.
        """
        if self.risk_planner is not None:
            approved, rejected = self.risk_planner.plan([intent], snapshot)
            if rejected:
                self.rejections.extend(rejected)
                return None
        if price <= 0:
            raise ValueError("price must be > 0 to size the order")
        order = OrderIntent(
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.notional / price,
            order_type=intent.order_type,
            reduce_only=intent.reduce_only,
            client_order_id=intent.client_order_id,
        )
        return self.submit(order)
