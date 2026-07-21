"""build_order_router — pick the router matching the runtime mode.

UI card: "execution · resolver" — read-only diagnostic.

Rules:
    mode = 'paper'    → DryRunRouter
    mode = 'live'     → BdpBotKafkaRouter (if BDP_BOT_KAFKA_ORDER_TOPIC set),
                         else AtomicLibRouter, else DryRunRouter with warning
    mode = 'backtest' → DryRunRouter
"""
from __future__ import annotations

import logging
import os

from .protocol   import OrderRouter
from .bdp_kafka  import BdpBotKafkaRouter
from .atomic_lib import AtomicLibRouter
from .dry_run    import DryRunRouter

log = logging.getLogger(__name__)


def build_order_router(mode: str = "paper") -> OrderRouter:
    if mode != "live":
        return DryRunRouter()

    if os.environ.get("BDP_BOT_KAFKA_ORDER_TOPIC"):
        try:
            log.info("OrderRouter → bdp-bot Kafka")
            return BdpBotKafkaRouter()
        except Exception as e:   # noqa: BLE001
            log.debug("BdpBotKafkaRouter unavailable: %s", e)

    try:
        log.info("OrderRouter → atomic_strategy_lib (local binance-cli)")
        return AtomicLibRouter()
    except Exception as e:   # noqa: BLE001
        log.warning("AtomicLibRouter unavailable: %s — DRY RUN", e)

    return DryRunRouter()
