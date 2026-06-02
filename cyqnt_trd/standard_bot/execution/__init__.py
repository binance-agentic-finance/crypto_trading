"""
Execution layer exports for the standard bot architecture.
"""

from .binance_futures_mainnet import BinanceFuturesMainnetBrokerAdapter
from .binance_futures_testnet import BinanceFuturesTestnetBrokerAdapter
from .cli_executor import BinanceCliExecutor
from .idempotency import build_client_tag
from .interfaces import BrokerAdapter, ExecutionPlanner, RiskRule
from .paper import PaperBrokerAdapter
from .rules import (
    InstrumentWhitelistRule,
    LongOnlySinglePositionRule,
    MaxAbsoluteNotionalRule,
    MaxPositionFractionRule,
)

__all__ = [
    "BinanceCliExecutor",
    "BinanceFuturesMainnetBrokerAdapter",
    "BinanceFuturesTestnetBrokerAdapter",
    "BrokerAdapter",
    "ExecutionPlanner",
    "build_client_tag",
    "InstrumentWhitelistRule",
    "LongOnlySinglePositionRule",
    "MaxAbsoluteNotionalRule",
    "MaxPositionFractionRule",
    "PaperBrokerAdapter",
    "RiskRule",
]
