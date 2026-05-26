"""cyqnt_trd.exec_cli — binance-cli execution wrappers.

Port of atomic_strategy_lib.execution into the cyqnt_trd package.

Provides order placement, position setup, and exchange filter utilities
via binance-cli subprocess calls.

⚠️  SAFETY CONTRACT — all order/position functions default to dry_run=True.
    You MUST pass ``dry_run=False`` explicitly to execute live operations.

Quickstart
----------
    from cyqnt_trd.exec_cli import market_order, set_leverage, quantize

    # Inspect what would be sent (no real order):
    result = market_order("BTCUSDT", "BUY", qty=0.01)          # dry_run=True
    print(result)

    # Place a real order:
    result = market_order("BTCUSDT", "BUY", qty=0.01, dry_run=False)

Modules
-------
orders.py   — market_order, limit_order, stop_market_order, cancel_all,
               partial_close
position.py — set_leverage, set_margin_type
filters.py  — exchange_filter_fetch, quantize, round_to_tick
"""

from ._subprocess import CLIError, ExchangeFilter, OrderResult
from .filters import exchange_filter_fetch, quantize, round_to_tick
from .orders import (
    cancel_all,
    limit_order,
    market_order,
    partial_close,
    stop_market_order,
)
from .position import set_leverage, set_margin_type

__all__ = [
    # orders
    "market_order",
    "limit_order",
    "stop_market_order",
    "cancel_all",
    "partial_close",
    # position
    "set_leverage",
    "set_margin_type",
    # filters
    "exchange_filter_fetch",
    "quantize",
    "round_to_tick",
    # types
    "OrderResult",
    "ExchangeFilter",
    "CLIError",
]
