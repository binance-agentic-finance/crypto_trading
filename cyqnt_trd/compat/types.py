"""Backward-compatible dataclass re-definitions mirroring atomic_strategy_lib.core.types.

Every type here is **identical** in field names, types, and defaults to the
original atomic source so that existing code that does::

    from atomic_strategy_lib.core.types import Candle, Signal, Verdict, TradePlan

can migrate to::

    from cyqnt_trd.compat import Candle, Signal, Verdict, TradePlan

with zero functional changes.

Note on ``Verdict``
-------------------
Atomic ``Verdict`` is intentionally a plain class (not an Enum) with string
constants and a ``RANK`` dict.  We replicate that shape exactly so that code
like ``plan.verdict == Verdict.CANDIDATE`` and ``Verdict.RANK[plan.verdict]``
continues to work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# L1 — Data types
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    """OHLCV bar from atomic_strategy_lib.core.types.

    ``timestamp`` is milliseconds since epoch (same as Binance kline close_time).
    """
    timestamp: int          # unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float           # base-asset volume
    quote_volume: float = 0.0
    trades: int = 0

    # ------------------------------------------------------------------
    # Convenience helpers not in original atomic — added for cyqnt_trd
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trades": self.trades,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Candle":
        return cls(
            timestamp=int(d["timestamp"]),
            open=float(d["open"]),
            high=float(d["high"]),
            low=float(d["low"]),
            close=float(d["close"]),
            volume=float(d["volume"]),
            quote_volume=float(d.get("quote_volume", 0.0)),
            trades=int(d.get("trades", 0)),
        )


@dataclass
class Ticker:
    symbol: str
    price: float
    change_pct_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    volume_base_24h: Optional[float] = None
    volume_quote_24h: Optional[float] = None
    trades_24h: int = 0
    weighted_avg_price: Optional[float] = None


@dataclass
class FundingRate:
    symbol: str
    rate: float
    timestamp: Optional[int] = None
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    next_funding_time: Optional[int] = None


@dataclass
class OpenInterest:
    symbol: str
    oi_base: float
    oi_value: float         # quote currency (USDT)
    timestamp: Optional[int] = None


@dataclass
class OIHistoryPoint:
    timestamp: int
    oi_value: float
    oi_base: float = 0.0


@dataclass
class OrderBookLevel:
    price: float
    quantity: float


@dataclass
class OrderBook:
    symbol: str
    bids: list              # list[OrderBookLevel]
    asks: list              # list[OrderBookLevel]
    timestamp: Optional[int] = None


@dataclass
class Balance:
    asset: str
    free: float
    locked: float
    total: float


@dataclass
class Position:
    """Open futures position — mirrors atomic_strategy_lib.core.types.Position."""
    symbol: str
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    unrealized_pnl: float
    leverage: int
    margin_type: str        # "ISOLATED" or "CROSS"
    notional: float = 0.0
    mark_price: float = 0.0


@dataclass
class ExchangeFilter:
    symbol: str
    step_size: float        # LOT_SIZE stepSize
    tick_size: float        # PRICE_FILTER tickSize
    min_notional: float
    min_qty: float
    qty_precision: int
    price_precision: int


# ---------------------------------------------------------------------------
# L2-L3 — Signal & scoring types
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    name: str
    value: float
    direction: str          # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float = 0.0   # 0.0 to 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "direction": self.direction,
            "strength": self.strength,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Signal":
        return cls(
            name=str(d["name"]),
            value=float(d["value"]),
            direction=str(d["direction"]),
            strength=float(d.get("strength", 0.0)),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class Score:
    value: float
    reasons: list = field(default_factory=list)   # list[str]
    factors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"value": self.value, "reasons": self.reasons, "factors": self.factors}

    @classmethod
    def from_dict(cls, d: dict) -> "Score":
        return cls(
            value=float(d["value"]),
            reasons=list(d.get("reasons", [])),
            factors=dict(d.get("factors", {})),
        )


class Verdict:
    """String-constant verdict class — intentionally NOT an Enum (mirrors atomic).

    Usage::

        if plan.verdict == Verdict.STRONG_CANDIDATE:
            ...
        rank = Verdict.RANK[plan.verdict]   # lower = better
    """
    STRONG_CANDIDATE = "STRONG_CANDIDATE"
    CANDIDATE = "CANDIDATE"
    WATCHLIST = "WATCHLIST"
    SKIP = "SKIP"
    AVOID = "AVOID"

    RANK: Dict[str, int] = {
        "STRONG_CANDIDATE": 0,
        "CANDIDATE": 1,
        "WATCHLIST": 2,
        "SKIP": 3,
        "AVOID": 4,
    }

    # Convenience: all valid string values
    ALL: List[str] = [
        "STRONG_CANDIDATE",
        "CANDIDATE",
        "WATCHLIST",
        "SKIP",
        "AVOID",
    ]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls.RANK


# ---------------------------------------------------------------------------
# L4 — Decision types
# ---------------------------------------------------------------------------

@dataclass
class TradePlan:
    symbol: str
    direction: str          # "LONG", "SHORT", "NO_TRADE"
    verdict: str
    score: float
    entry_price: float
    stop_price: float
    stop_pct: float
    leverage: float
    notional: float
    max_loss: float
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "verdict": self.verdict,
            "score": self.score,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "stop_pct": self.stop_pct,
            "leverage": self.leverage,
            "notional": self.notional,
            "max_loss": self.max_loss,
            "reasons": self.reasons,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TradePlan":
        return cls(
            symbol=str(d["symbol"]),
            direction=str(d["direction"]),
            verdict=str(d["verdict"]),
            score=float(d["score"]),
            entry_price=float(d["entry_price"]),
            stop_price=float(d["stop_price"]),
            stop_pct=float(d["stop_pct"]),
            leverage=float(d["leverage"]),
            notional=float(d["notional"]),
            max_loss=float(d["max_loss"]),
            reasons=list(d.get("reasons", [])),
        )


# ---------------------------------------------------------------------------
# L5 — Execution types
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    quantity: float = 0.0
    price: float = 0.0
    status: str = ""
    dry_run: bool = False
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status,
            "dry_run": self.dry_run,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrderResult":
        return cls(
            success=bool(d["success"]),
            order_id=d.get("order_id"),
            symbol=str(d.get("symbol", "")),
            side=str(d.get("side", "")),
            order_type=str(d.get("order_type", "")),
            quantity=float(d.get("quantity", 0.0)),
            price=float(d.get("price", 0.0)),
            status=str(d.get("status", "")),
            dry_run=bool(d.get("dry_run", False)),
            raw=dict(d.get("raw", {})),
        )


# ---------------------------------------------------------------------------
# L2 — Velocity metrics
# ---------------------------------------------------------------------------

@dataclass
class VelocityMetrics:
    timeframe: str
    vol_vs_avg: Optional[float] = None
    vol_accel_3v3: Optional[float] = None
    range_vs_avg: Optional[float] = None
    green_streak: int = 0
    red_streak: int = 0
    price_change_pct: float = 0.0
    last3_change_pct: float = 0.0


# ---------------------------------------------------------------------------
# Helpers (verbatim from atomic.core.types)
# ---------------------------------------------------------------------------

def extract_closes(values: list) -> list:
    """Extract close prices from Candle list, or return as-is if already floats."""
    if values and hasattr(values[0], "close"):
        return [v.close for v in values]
    return values


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # L1
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
    # L2-L3
    "Signal",
    "Score",
    "Verdict",
    # L4
    "TradePlan",
    # L5
    "OrderResult",
    # Velocity
    "VelocityMetrics",
    # helpers
    "extract_closes",
]
