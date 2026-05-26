"""Backtest runtime facade."""

from ai_pro_trading_library.runtime.backtest.engine import (
    BacktestResult,
    BacktestRunner,
    build_signal_from_spec,
    run_vector_backtest,
)
from ai_pro_trading_library.runtime.backtest.execution_kernels import (
    TARGET_FLAT,
    TARGET_KEEP,
    TARGET_LONG,
    TARGET_SHORT,
    simulate_target_positions_next_open,
)
from ai_pro_trading_library.runtime.backtest.long_only_engine import (
    EquityPoint,
    LongOnlyResult,
    TradeRecord,
    run_long_only_backtest,
)
from ai_pro_trading_library.runtime.backtest.metrics import (
    compute_equity_metrics,
    compute_full_metrics,
    compute_trade_metrics,
)
from ai_pro_trading_library.runtime.backtest.snapshot_runner import (
    SnapshotBacktestRunner,
)

__all__ = [
    "BacktestResult",
    "BacktestRunner",
    "EquityPoint",
    "LongOnlyResult",
    "SnapshotBacktestRunner",
    "TARGET_FLAT",
    "TARGET_KEEP",
    "TARGET_LONG",
    "TARGET_SHORT",
    "TradeRecord",
    "build_signal_from_spec",
    "compute_equity_metrics",
    "compute_full_metrics",
    "compute_trade_metrics",
    "run_long_only_backtest",
    "run_vector_backtest",
    "simulate_target_positions_next_open",
]
