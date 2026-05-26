"""
cyqnt_trd.standard_bot.monitoring
==================================

Runtime monitoring utilities — PnL computation, signal logging, and stop
order health checks.  These are thin runtime helpers called by daemons or
external scripts; they do **not** belong in ``blocks`` (which is for pure
pandas strategy logic).

Modules
-------
pnl        Unrealized PnL and portfolio aggregation.
signals    Signal recording, outcome checking, performance reporting.
stops      Stale-stop detection and open-order querying via binance-cli.
"""

from .pnl import (
    unrealized_pnl_compute,
    portfolio_pnl_aggregate,
)
from .signals import (
    signal_record,
    signal_outcome_check,
    performance_report,
)
from .stops import (
    stale_stop_detect,
    query_open_orders,
    STOP_TYPES,
)

__all__ = [
    # pnl
    "unrealized_pnl_compute",
    "portfolio_pnl_aggregate",
    # signals
    "signal_record",
    "signal_outcome_check",
    "performance_report",
    # stops
    "stale_stop_detect",
    "query_open_orders",
    "STOP_TYPES",
]
