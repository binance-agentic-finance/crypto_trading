"""cyqnt_trd.compat — backward-compatible shim for atomic_strategy_lib types.

Drop-in replacement for ``atomic_strategy_lib.core.types``.  All dataclass
types and helpers are available at the package level::

    # Old (atomic_strategy_lib)
    from atomic_strategy_lib.core.types import Candle, Signal, Verdict, TradePlan

    # New (cyqnt_trd)
    from cyqnt_trd.compat import Candle, Signal, Verdict, TradePlan

Field names, types, and defaults are **identical** to the originals so no
downstream code changes are needed.

DataFrame adapters are also available::

    from cyqnt_trd.compat import df_to_candles, candles_to_df

See ``cyqnt_trd/compat/README.md`` for the full migration guide.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export atomic dataclass types
# ---------------------------------------------------------------------------
from .types import (
    # L1 — data
    Candle,
    Ticker,
    FundingRate,
    OpenInterest,
    OIHistoryPoint,
    OrderBookLevel,
    OrderBook,
    Balance,
    Position,
    ExchangeFilter,
    # L2-L3 — signal & scoring
    Signal,
    Score,
    Verdict,
    # L4 — decision
    TradePlan,
    # L5 — execution
    OrderResult,
    # velocity
    VelocityMetrics,
    # helper
    extract_closes,
)

# ---------------------------------------------------------------------------
# Re-export adapters
# ---------------------------------------------------------------------------
from .adapters import (
    df_to_candles,
    candles_to_df,
    signal_series_to_signals,
    signals_to_series,
    trade_plan_to_dict,
    dict_to_trade_plan,
)

# ---------------------------------------------------------------------------
# Re-export config loader (atomic_strategy_lib.core.config replacement)
# ---------------------------------------------------------------------------
from .config import (
    DEFAULT_CONFIG,
    load_config,
    merge_config,
    get_section,
    resolve_workspace_path,
)

__all__ = [
    # types
    "Candle",
    "Ticker",
    "FundingRate",
    "OpenInterest",
    "OIHistoryPoint",
    "OrderBookLevel",
    "OrderBook",
    "Balance",
    "Position",
    "ExchangeFilter",
    "Signal",
    "Score",
    "Verdict",
    "TradePlan",
    "OrderResult",
    "VelocityMetrics",
    "extract_closes",
    # adapters
    "df_to_candles",
    "candles_to_df",
    "signal_series_to_signals",
    "signals_to_series",
    "trade_plan_to_dict",
    "dict_to_trade_plan",
    # config (atomic.core.config drop-in)
    "DEFAULT_CONFIG",
    "load_config",
    "merge_config",
    "get_section",
    "resolve_workspace_path",
]
