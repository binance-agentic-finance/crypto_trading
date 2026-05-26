"""Monitoring state, PnL, signal-tracking, stops, watcher compat."""

from ai_pro_trading_library.library.monitoring.pnl import (
    portfolio_pnl_aggregate,
    unrealized_pnl_compute,
)
from ai_pro_trading_library.library.monitoring.signals_log import (
    performance_report,
    signal_outcome_check,
    signal_record,
    stale_stop_detect,
)
from ai_pro_trading_library.library.monitoring.state import (
    EventRecord,
    MonitoringState,
    TradeRecord,
)
from ai_pro_trading_library.library.monitoring.watcher_compat import (
    WATCHER_OPTIONAL_KEYS,
    WATCHER_REQUIRED_KEYS,
    monitoring_state_to_watcher_payload,
    validate_watcher_payload,
)

__all__ = [
    "EventRecord",
    "MonitoringState",
    "TradeRecord",
    "WATCHER_OPTIONAL_KEYS",
    "WATCHER_REQUIRED_KEYS",
    "monitoring_state_to_watcher_payload",
    "performance_report",
    "portfolio_pnl_aggregate",
    "signal_outcome_check",
    "signal_record",
    "stale_stop_detect",
    "unrealized_pnl_compute",
    "validate_watcher_payload",
]
