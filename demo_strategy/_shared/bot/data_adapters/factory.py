"""build_data_adapter — pick the highest-priority backend for this env.

UI card: "data source · resolver" — read-only diagnostic showing which
adapter was picked and why.

Priority (first-working wins):
    1. BdpBotAdapter    — env `BDP_BOT_MARKET_HOT_STORE` set
    2. AtomicLibAdapter — atomic_strategy_lib importable
    3. DirectRESTAdapter — always available
"""
from __future__ import annotations

import logging
import os

from .protocol    import DataAdapter
from .bdp_bot     import BdpBotAdapter
from .atomic_lib  import AtomicLibAdapter
from .direct_rest import DirectRESTAdapter

log = logging.getLogger(__name__)


def build_data_adapter() -> DataAdapter:
    """Return the highest-priority adapter that works in this env."""
    if os.environ.get("BDP_BOT_MARKET_HOT_STORE"):
        try:
            log.info("DataAdapter → bdp-bot (strategy_service.market)")
            return BdpBotAdapter()
        except Exception as e:  # noqa: BLE001
            log.debug("BdpBotAdapter unavailable: %s", e)

    try:
        log.info("DataAdapter → atomic_strategy_lib")
        return AtomicLibAdapter()
    except Exception as e:  # noqa: BLE001
        log.debug("AtomicLibAdapter unavailable: %s", e)

    log.info("DataAdapter → direct REST fallback")
    return DirectRESTAdapter()
