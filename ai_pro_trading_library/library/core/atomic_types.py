"""Atomic-shape dataclasses — local mirrors of `atomic_strategy_lib.core.types`.

Per `docs/migration/overlap-policy.md` Class B: atomic uses dataclasses
with their own field set (e.g. `Ticker.change_pct_24h` vs canonical
`AccountPosition` having no such field). Atomic-shape consumers get
these locally so they don't need `import atomic_strategy_lib`.

Already covered elsewhere (NOT re-defined here):
- `Candle` — duck-typed via `library.data.conversions.bars_to_atomic_candles`
- `Signal` (atomic shape) — `library.scoring.atomic_compat.AtomicSignal`
- `Score` — `library.scoring.atomic_compat.AtomicScore`
- `Verdict` — `library.scoring.atomic_compat.AtomicVerdict`
- `VelocityMetrics` — `library.features.atomic_signals.computes.VelocityMetrics`

Atomic `TradePlan` shape differs from canonical
`library.core.protocols.TradePlan` (the latter holds an `ExecutionIntent`
tuple, atomic holds verdict + score + stop_pct + leverage + max_loss as
flat scalars). Both are kept; atomic version is exposed as
`AtomicTradePlan` here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Ticker:
    """24h ticker snapshot."""

    symbol: str
    price: float
    change_pct_24h: float | None = None
    high_24h: float | None = None
    low_24h: float | None = None
    volume_base_24h: float | None = None
    volume_quote_24h: float | None = None
    trades_24h: int = 0
    weighted_avg_price: float | None = None


@dataclass
class FundingRate:
    """Funding rate snapshot for a single symbol."""

    symbol: str
    rate: float
    timestamp: int | None = None
    mark_price: float | None = None
    index_price: float | None = None
    next_funding_time: int | None = None


@dataclass
class OpenInterest:
    """Latest open interest for a symbol."""

    symbol: str
    oi_base: float
    oi_value: float  # quote currency (USDT)
    timestamp: int | None = None


@dataclass
class OIHistoryPoint:
    """Single OI history point — used by `atomic_signals.derivatives_detectors.oi_anomaly_detect`."""

    timestamp: int
    oi_value: float
    oi_base: float = 0.0


@dataclass
class OrderBookLevel:
    price: float
    quantity: float


@dataclass
class OrderBook:
    """Order book snapshot."""

    symbol: str
    bids: list  # list[OrderBookLevel]
    asks: list  # list[OrderBookLevel]
    timestamp: int | None = None


@dataclass
class Balance:
    """Account balance for one asset."""

    asset: str
    free: float
    locked: float
    total: float


@dataclass
class AtomicPosition:
    """Atomic-shape position record. Differs from canonical
    `library.core.protocols.AccountPosition` (which has fewer fields)."""

    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    unrealized_pnl: float
    leverage: int
    margin_type: str  # "ISOLATED" or "CROSS"
    notional: float = 0.0
    mark_price: float = 0.0


@dataclass
class ExchangeFilter:
    """Exchange-info constraints (LOT_SIZE / PRICE_FILTER / MIN_NOTIONAL)."""

    symbol: str
    step_size: float  # LOT_SIZE stepSize
    tick_size: float  # PRICE_FILTER tickSize
    min_notional: float
    min_qty: float
    qty_precision: int
    price_precision: int


@dataclass
class OrderResult:
    """Result of a (paper or live) order submission."""

    success: bool
    order_id: str | None = None
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    quantity: float = 0.0
    price: float = 0.0
    status: str = ""
    dry_run: bool = False
    raw: dict = field(default_factory=dict)


@dataclass
class AtomicTradePlan:
    """Atomic-shape trade plan. Different field set from canonical
    `library.core.protocols.TradePlan`."""

    symbol: str
    direction: str  # "LONG" / "SHORT" / "NO_TRADE"
    verdict: str
    score: float
    entry_price: float
    stop_price: float
    stop_pct: float
    leverage: float
    notional: float
    max_loss: float
    reasons: list = field(default_factory=list)


__all__ = [
    "AtomicPosition",
    "AtomicTradePlan",
    "Balance",
    "ExchangeFilter",
    "FundingRate",
    "OIHistoryPoint",
    "OpenInterest",
    "OrderBook",
    "OrderBookLevel",
    "OrderResult",
    "Ticker",
]
