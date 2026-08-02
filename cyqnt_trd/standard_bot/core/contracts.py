"""
Core contracts for the standard trading bot architecture.

These dataclasses are the first version of the protocol layer described in
``standard_trading_bot_pdf.md``. The goal is to make the new architecture
concrete without breaking the existing modules yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SourceStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    SKIPPED = "skipped"
    ERROR = "error"


class SignalKind(str, Enum):
    TRADE = "trade"
    SELECTION = "selection"
    ALERT = "alert"
    NOOP = "noop"


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    FLAT = "flat"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class ExecutionStatus(str, Enum):
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class TriggerType(str, Enum):
    CRON = "cron"
    HTTP = "http"
    QUEUE = "queue"
    MANUAL = "manual"


InstrumentId = str
Timeframe = str
RunId = str
TraceId = str


@dataclass
class BundleMeta:
    data_source: str = ""
    fetched_at: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeRange:
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    tail_bars: Optional[int] = None


@dataclass
class QueryOptions:
    partial_ok: bool = True
    max_staleness_sec: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketQuery:
    instruments: List[InstrumentId]
    timeframes: List[Timeframe]
    time_range: TimeRange = field(default_factory=TimeRange)


@dataclass
class SocialQuery:
    time_range: TimeRange = field(default_factory=TimeRange)
    keywords: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    instruments: List[InstrumentId] = field(default_factory=list)
    language: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnChainQuery:
    time_range: TimeRange = field(default_factory=TimeRange)
    watchlist_ref: Optional[str] = None
    chains: List[str] = field(default_factory=list)
    instruments: List[InstrumentId] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameQuery:
    """Declared need for one named long-form frame (advisory/monitor route).

    ``schema`` is the frame contract id (e.g. ``MarketMetricFrame@1.0``), so an
    assembler knows what to normalize into ``DataSnapshot.frames[name]``.
    """

    schema: str
    required: bool = True
    field_keys: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQuery:
    market: Optional[MarketQuery] = None
    social: Optional[SocialQuery] = None
    onchain: Optional[OnChainQuery] = None
    options: QueryOptions = field(default_factory=QueryOptions)
    frames: Dict[str, FrameQuery] = field(default_factory=dict)


@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: int
    instrument_id: InstrumentId
    timeframe: Timeframe
    confirmed: bool
    quote_volume: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketBundle:
    bars: Dict[str, List[Bar]]
    meta: BundleMeta = field(default_factory=BundleMeta)

    @staticmethod
    def key(instrument_id: InstrumentId, timeframe: Timeframe) -> str:
        return "%s|%s" % (instrument_id, timeframe)


@dataclass
class SocialDocument:
    doc_id: str
    source_type: str
    published_at: int
    text: str
    raw_ref: Dict[str, Any]
    observable_at: Optional[int] = None
    author_id: Optional[str] = None
    author_handle: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    language: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    sentiment: Optional[float] = None


@dataclass
class SocialFeedBundle:
    items: List[SocialDocument]
    query: Optional[SocialQuery] = None
    meta: BundleMeta = field(default_factory=BundleMeta)


@dataclass
class ChainObservation:
    obs_id: str
    chain: str
    timestamp: int
    event_type: str
    addresses: Dict[str, Any]
    raw_ref: Dict[str, Any]
    tokens: List[Dict[str, Any]] = field(default_factory=list)
    direction: Optional[str] = None
    confidence: Optional[float] = None
    related_instruments: List[InstrumentId] = field(default_factory=list)
    confirmed_at: Optional[int] = None


@dataclass
class OnChainSignalBundle:
    observations: List[ChainObservation]
    watchlist_ref: Optional[Dict[str, Any]] = None
    meta: BundleMeta = field(default_factory=BundleMeta)


@dataclass
class UniverseBundle:
    """Cross-sectional selection inputs (many symbols at a single ``as_of``).

    Trade strategies read a single-instrument per-bar DataFrame; SELECTION
    strategies instead rank a *universe* of symbols. This bundle carries the
    24h ticker universe table and the Square ticker-rank mention/sentiment
    aggregates a selection strategy ranks over, plus (for delta-based
    strategies) the previous ticker-rank snapshot and per-candidate klines.

    The tables are kept opaque (``Any`` = pandas DataFrame) so ``core`` has no
    hard pandas import. All frames are expected to be PIT-gated (only rows
    whose availability time <= ``as_of``) by the assembler that builds them.
    """

    as_of: int = 0
    universe: Any = None          # 24h ticker table (DataFrame)
    ticker_rank: Any = None       # Square mention/sentiment aggregates (DataFrame)
    ticker_rank_prev: Any = None  # previous ticker-rank snapshot (delta selection, N2)
    klines: Dict[str, Any] = field(default_factory=dict)  # per-candidate klines (N2)
    meta: BundleMeta = field(default_factory=BundleMeta)


@dataclass
class SnapshotMeta:
    snapshot_id: str
    assembled_at: int
    partial_ok: bool = True
    source_status: Dict[str, SourceStatus] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    decision_as_of: Optional[int] = None
    primary_timeframe: Optional[Timeframe] = None
    alignment_policy: Optional[str] = None
    trace_id: Optional[TraceId] = None


@dataclass
class DataSnapshot:
    version: str
    market: Optional[MarketBundle] = None
    social: Optional[SocialFeedBundle] = None
    onchain: Optional[OnChainSignalBundle] = None
    meta: SnapshotMeta = field(
        default_factory=lambda: SnapshotMeta(snapshot_id="", assembled_at=0)
    )
    # Optional cross-sectional selection inputs for SELECTION-kind strategies.
    # Defaults to None so every existing market/social/onchain assembler path
    # and the 8 pre-existing strategies are byte-identical.
    universe: Optional["UniverseBundle"] = None
    # Extensible named long-form frames for data that is not naturally
    # OHLCV/social/on-chain. ALERT-kind advisory bots read normalized pandas
    # frames from here (``market_metrics`` / ``news_events``) so they ride the
    # same DataSnapshot -> SignalBatch pipeline as trade/selection strategies.
    frames: Dict[str, Any] = field(default_factory=dict)
    # The same frames with their canonical shape metadata.  Keeping this on the
    # snapshot prevents a JSON replay from degrading a Metric/Event/Position
    # frame back into an untyped DataFrame before a StandardBot sees it.
    typed: Dict[str, Any] = field(default_factory=dict)
    # Account and run context are part of cyqnt.input/v1.  They used to be
    # serialised into the bundle and silently discarded while loading it.
    positions: Dict[str, float] = field(default_factory=dict)
    equity: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    trace_id: str = ""

    def require_market(self) -> MarketBundle:
        if self.market is None:
            raise ValueError("DataSnapshot.market is required for this operation")
        return self.market

    def require_frame(self, name: str) -> Any:
        if name not in self.frames:
            raise ValueError("DataSnapshot.frames[%r] is required for this operation" % name)
        return self.frames[name]


@dataclass
class SignalContext:
    account_snapshot: Optional["AccountSnapshot"] = None
    blacklist: List[InstrumentId] = field(default_factory=list)
    previous_positions: Dict[InstrumentId, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalProvenance:
    plugin_id: str
    plugin_version: str
    config_hash: str
    input_fingerprint: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalEnvelope:
    version: str
    signal_id: str
    kind: SignalKind
    strength: float
    provenance: SignalProvenance
    instrument_id: Optional[InstrumentId] = None
    side: Optional[TradeSide] = None
    time_horizon: Optional[str] = None
    valid_until: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalBatch:
    signals: List[SignalEnvelope]
    batch_id: str
    created_at: int

    def trade_signals(self) -> List[SignalEnvelope]:
        return [signal for signal in self.signals if signal.kind == SignalKind.TRADE]

    def selection_signals(self) -> List[SignalEnvelope]:
        return [signal for signal in self.signals if signal.kind == SignalKind.SELECTION]


@dataclass
class AccountPosition:
    instrument_id: InstrumentId
    quantity: float
    avg_entry_price: Optional[float] = None
    side: Optional[str] = None


@dataclass
class AccountSnapshot:
    account_id: str
    balances: Dict[str, float] = field(default_factory=dict)
    positions: List[AccountPosition] = field(default_factory=list)
    fetched_at: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionIntent:
    intent_id: str
    instrument_id: InstrumentId
    side: TradeSide
    order_type: OrderType
    quantity: Optional[float] = None
    notional: Optional[float] = None
    price: Optional[float] = None
    time_in_force: Optional[str] = None
    reduce_only: bool = False
    client_tag: Optional[str] = None
    risk_hints: Dict[str, Any] = field(default_factory=dict)
    source_signal_id: Optional[str] = None


@dataclass
class Fill:
    price: float
    quantity: float
    fee: float = 0.0
    filled_at: Optional[int] = None


@dataclass
class ExecutionReport:
    intent_id: str
    status: ExecutionStatus
    external_ids: Dict[str, str] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    reason: Optional[str] = None
    run_id: Optional[RunId] = None


@dataclass
class SignalPipelineSpec:
    plugin_chain: List[Dict[str, Any]]
    pipeline_hash: Optional[str] = None


@dataclass
class BacktestRequest:
    request_id: str
    instruments: List[InstrumentId]
    primary_timeframe: Timeframe
    start_ts: int
    end_ts: int
    signal_pipeline: SignalPipelineSpec
    initial_capital: float = 10000.0
    fee_model: Dict[str, Any] = field(default_factory=dict)
    slippage_model: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EquityPoint:
    timestamp: int
    equity: float
    cash: Optional[float] = None
    drawdown: Optional[float] = None


@dataclass
class BacktestResult:
    request_id: str
    total_return: float
    equity_curve: List[EquityPoint]
    execution_reports: List[ExecutionReport] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    signal_batches: List[SignalBatch] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorTrigger:
    trigger_type: TriggerType
    payload: Dict[str, Any] = field(default_factory=dict)
    instruments_override: List[InstrumentId] = field(default_factory=list)
    signal_only: bool = False


@dataclass
class RunSummary:
    run_id: RunId
    status: str
    signal_count: int
    signals: List[SignalEnvelope] = field(default_factory=list)
    execution_reports: List[ExecutionReport] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
