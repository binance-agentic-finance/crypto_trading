"""BdpBotKafkaRouter — publish to bdp-ai-trading-bot's Kafka OrderCommand topic.

UI card: "execution · bdp-bot Kafka" — production path; picked when
`BDP_BOT_KAFKA_ORDER_TOPIC` env is set.
"""
from __future__ import annotations

from typing import Any


class BdpBotKafkaRouter:
    """Publishes OrderCommandV1 / PortfolioTargetCommandV1 to Kafka."""

    def __init__(self):
        from strategy_service.engine.order_stream.kafka import publish   # noqa: PLC0415
        from strategy_service.engine.order_stream.contract import (       # noqa: PLC0415
            OrderCommandV1, PortfolioTargetCommandV1,
            stable_client_order_id, stable_idempotency_key,
            stable_portfolio_client_order_id, stable_portfolio_idempotency_key,
        )
        self._publish = publish
        self._OrderCommandV1           = OrderCommandV1
        self._PortfolioTargetCommandV1 = PortfolioTargetCommandV1
        self._coid  = stable_client_order_id
        self._iidk  = stable_idempotency_key
        self._pcoid = stable_portfolio_client_order_id
        self._piidk = stable_portfolio_idempotency_key

    def submit_single(self, *, symbol, side, qty, entry, stop, mode, metadata):
        cmd = self._OrderCommandV1(
            symbol=symbol, side=side.upper(), qty=qty, price=entry,
            stop_price=stop, mode=mode,
            client_order_id=self._coid(metadata),
            idempotency_key=self._iidk(metadata),
            metadata=metadata,
        )
        ack = self._publish(cmd)
        return {"status": "published", "ack": ack, "cmd_type": "OrderCommandV1"}

    def submit_portfolio(self, *, weights, notional_usdt, mode, metadata):
        cmd = self._PortfolioTargetCommandV1(
            weights=weights, notional_usdt=notional_usdt, mode=mode,
            client_order_id=self._pcoid(metadata),
            idempotency_key=self._piidk(metadata),
            metadata=metadata,
        )
        ack = self._publish(cmd)
        return {"status": "published", "ack": ack,
                "cmd_type": "PortfolioTargetCommandV1"}
