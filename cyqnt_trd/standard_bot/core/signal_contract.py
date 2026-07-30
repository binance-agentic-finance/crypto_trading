"""``cyqnt.signal/v2`` — the complete standard-bot output contract.

What was missing
----------------
``SignalEnvelope`` carries ``kind`` / ``side`` / ``instrument_id`` and a
free-form ``payload``. That is enough to open a position and nothing else: the
payload shape differed per strategy, and **there was no way to say "close the
long"** as opposed to "open a short". Those are different instructions — one
reduces existing exposure, the other creates new opposite exposure, and on a
spot account only the first is even possible — but both used to arrive as
``side=SELL`` and the executor had to guess from context.

This module makes the whole position lifecycle explicit:

* :class:`PositionIntent` — open / add / reduce / **close** / flip / flat / hold,
  each with an unambiguous direction and a ``reduce_only`` flag.
* :class:`ExitPlan` — stop, take-profit ladder, trailing and time stop as one
  declared object that travels with the signal instead of living only in the
  daemon's memory.
* :class:`StandardSignal` — identity, instrument, decision, entry, exit, sizing,
  risk limits, horizon, explanation, data quality and provenance in one row.

Every field a downstream consumer has had to infer is now stated. The one thing
deliberately **not** here is the execution idempotency key: it is
``instance:node:event_ref``, and only the executor knows run identity. A
strategy that minted its own would make two independent runs collide and a
replay of one run not collide — exactly backwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "SCHEMA_VERSION",
    "PositionIntent",
    "Direction",
    "AdvisoryAction",
    "MarketScope",
    "SignalKind",  # re-exported from .contracts — one vocabulary, not two
    "SizeMode",
    "EntryType",
    "DataQuality",
    "TimeHorizon",
    "StopSpec",
    "TakeProfitLeg",
    "TimeStop",
    "ExitPlan",
    "EntrySpec",
    "SizeSpec",
    "RiskLimits",
    "Evidence",
    "Provenance",
    "SelectionCandidate",
    "StandardSignal",
]

SCHEMA_VERSION = "cyqnt.signal/v2"


class PositionIntent(str, Enum):
    """What this signal asks the book to do. The core of the contract.

    ``OPEN_*`` / ``ADD_*`` create or increase exposure; ``REDUCE_*`` /
    ``CLOSE_*`` only ever decrease it; ``FLIP_*`` closes and reverses in one
    instruction; ``FLAT`` means "hold no position"; ``HOLD`` is an explicit
    no-change (useful so a monitor can say "still valid" rather than go silent).
    """

    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    ADD_LONG = "add_long"
    ADD_SHORT = "add_short"
    REDUCE_LONG = "reduce_long"
    REDUCE_SHORT = "reduce_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    FLIP_TO_LONG = "flip_to_long"
    FLIP_TO_SHORT = "flip_to_short"
    FLAT = "flat"
    HOLD = "hold"

    # ---- derived properties: no consumer should re-derive these ----

    @property
    def is_entry(self) -> bool:
        return self in (
            PositionIntent.OPEN_LONG, PositionIntent.OPEN_SHORT,
            PositionIntent.ADD_LONG, PositionIntent.ADD_SHORT,
            PositionIntent.FLIP_TO_LONG, PositionIntent.FLIP_TO_SHORT,
        )

    @property
    def is_exit(self) -> bool:
        return self in (
            PositionIntent.REDUCE_LONG, PositionIntent.REDUCE_SHORT,
            PositionIntent.CLOSE_LONG, PositionIntent.CLOSE_SHORT,
            PositionIntent.FLIP_TO_LONG, PositionIntent.FLIP_TO_SHORT,
            PositionIntent.FLAT,
        )

    @property
    def reduce_only(self) -> bool:
        """True when the instruction may only shrink an existing position."""
        return self in (
            PositionIntent.REDUCE_LONG, PositionIntent.REDUCE_SHORT,
            PositionIntent.CLOSE_LONG, PositionIntent.CLOSE_SHORT,
            PositionIntent.FLAT,
        )

    @property
    def target_side(self) -> Optional[str]:
        """Side of the position AFTER this instruction: long / short / flat."""
        if self in (PositionIntent.OPEN_LONG, PositionIntent.ADD_LONG,
                    PositionIntent.FLIP_TO_LONG):
            return "long"
        if self in (PositionIntent.OPEN_SHORT, PositionIntent.ADD_SHORT,
                    PositionIntent.FLIP_TO_SHORT):
            return "short"
        if self in (PositionIntent.CLOSE_LONG, PositionIntent.CLOSE_SHORT,
                    PositionIntent.FLAT):
            return "flat"
        if self in (PositionIntent.REDUCE_LONG,):
            return "long"
        if self in (PositionIntent.REDUCE_SHORT,):
            return "short"
        return None

    @property
    def closes_side(self) -> Optional[str]:
        """Which existing side is being closed/reduced — the 平仓方向.

        ``CLOSE_LONG`` closes a long (sells); ``CLOSE_SHORT`` closes a short
        (buys). ``FLAT`` closes whatever is open, so it returns ``"any"``.
        """
        if self in (PositionIntent.CLOSE_LONG, PositionIntent.REDUCE_LONG,
                    PositionIntent.FLIP_TO_SHORT):
            return "long"
        if self in (PositionIntent.CLOSE_SHORT, PositionIntent.REDUCE_SHORT,
                    PositionIntent.FLIP_TO_LONG):
            return "short"
        if self is PositionIntent.FLAT:
            return "any"
        return None

    @property
    def order_side(self) -> Optional[str]:
        """Exchange-level side needed to carry this out: buy / sell / none."""
        if self in (PositionIntent.OPEN_LONG, PositionIntent.ADD_LONG,
                    PositionIntent.FLIP_TO_LONG, PositionIntent.CLOSE_SHORT,
                    PositionIntent.REDUCE_SHORT):
            return "buy"
        if self in (PositionIntent.OPEN_SHORT, PositionIntent.ADD_SHORT,
                    PositionIntent.FLIP_TO_SHORT, PositionIntent.CLOSE_LONG,
                    PositionIntent.REDUCE_LONG):
            return "sell"
        return None

    @property
    def requires_short_capability(self) -> bool:
        """True when carrying this out means holding negative exposure.

        Closing a long is a sell but needs no short capability — the
        distinction a spot account depends on.
        """
        return self in (
            PositionIntent.OPEN_SHORT, PositionIntent.ADD_SHORT,
            PositionIntent.FLIP_TO_SHORT,
        )


class Direction(str, Enum):
    """Informational bias. NOT an order side — advisory bots emit this alone."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class AdvisoryAction(str, Enum):
    ALERT = "alert"
    WATCH = "watch"
    AVOID = "avoid"
    INVESTIGATE = "investigate"


class MarketScope(str, Enum):
    SINGLE = "single"          # one instrument
    CROSS_SECTION = "cross_section"   # a ranked universe
    GLOBAL = "global"          # market-wide, no single instrument


# ``kind`` — the field a consumer switches on to know WHICH shape it got.
#
# ``cyqnt.signal/v1`` carried one and v2's first cut dropped it, leaving
# consumers to infer the shape from ``market_scope`` plus a non-empty
# ``candidates``. That inference is not safe: ``market_scope`` defaults to
# ``single`` on :class:`BotSpec`, so a selection bot that never overrode it
# published a cross-sectional basket labelled ``single``. Putting trade and
# selection output in one envelope is only useful if one field says which it is,
# so it is back — reusing the SAME enum ``SignalEnvelope`` already uses rather
# than minting a second vocabulary for the same idea. ``NOOP`` is deliberately
# never emitted by v2: ``intent=hold`` already says "no change", and two ways to
# say it is one way to disagree.
from .contracts import SignalKind  # noqa: E402  (values: trade/selection/alert/noop)


class SizeMode(str, Enum):
    QUANTITY = "quantity"      # base units
    QUOTE_AMOUNT = "quote_amount"   # quote currency notional
    EQUITY_PCT = "equity_pct"  # fraction of equity
    RISK_PCT = "risk_pct"      # fraction of equity risked to the stop
    POSITION_PCT = "position_pct"   # fraction of the EXISTING position (exits)


class EntryType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    LIMIT_MAKER = "limit_maker"


class DataQuality(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


class TimeHorizon(str, Enum):
    SCALP = "scalp"
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"


# ---------------------------------------------------------------------------
# exit plan — the part that used to live only in the daemon's head
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StopSpec:
    """Protective stop. Exactly one of price / pct / atr_mult should be set."""

    price: Optional[float] = None
    pct: Optional[float] = None
    atr_mult: Optional[float] = None
    atr_value: Optional[float] = None
    trailing: bool = False
    #: place at the venue rather than evaluating in-process. When False the
    #: position is unprotected if the daemon dies — state it, do not imply it.
    exchange_managed: bool = False

    def __post_init__(self) -> None:
        given = [v for v in (self.price, self.pct, self.atr_mult) if v is not None]
        if len(given) > 1:
            raise ValueError("StopSpec: set only one of price / pct / atr_mult")
        if self.atr_mult is not None and self.atr_value is None:
            raise ValueError("StopSpec: atr_mult needs atr_value to resolve a price")
        for name, value in (("pct", self.pct), ("atr_mult", self.atr_mult)):
            if value is not None and value <= 0:
                raise ValueError("StopSpec: %s must be positive" % name)

    def resolve_price(self, *, entry_price: float, side: str) -> Optional[float]:
        """Absolute stop price for a position opened at ``entry_price``."""
        sign = -1.0 if side == "long" else 1.0
        if self.price is not None:
            return float(self.price)
        if self.pct is not None:
            return float(entry_price) * (1.0 + sign * float(self.pct))
        if self.atr_mult is not None and self.atr_value is not None:
            return float(entry_price) + sign * float(self.atr_mult) * float(self.atr_value)
        return None


@dataclass(frozen=True)
class TakeProfitLeg:
    """One rung of a take-profit ladder."""

    close_pct: float                      # fraction of the position to close
    price: Optional[float] = None
    pct: Optional[float] = None
    atr_mult: Optional[float] = None
    atr_value: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.close_pct <= 1.0:
            raise ValueError("TakeProfitLeg.close_pct must be in (0, 1]")
        given = [v for v in (self.price, self.pct, self.atr_mult) if v is not None]
        if len(given) != 1:
            raise ValueError("TakeProfitLeg: set exactly one of price / pct / atr_mult")

    def resolve_price(self, *, entry_price: float, side: str) -> Optional[float]:
        sign = 1.0 if side == "long" else -1.0
        if self.price is not None:
            return float(self.price)
        if self.pct is not None:
            return float(entry_price) * (1.0 + sign * float(self.pct))
        if self.atr_mult is not None and self.atr_value is not None:
            return float(entry_price) + sign * float(self.atr_mult) * float(self.atr_value)
        return None


@dataclass(frozen=True)
class TimeStop:
    """Close on elapsed time regardless of price."""

    max_bars: Optional[int] = None
    max_seconds: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_bars is None and self.max_seconds is None:
            raise ValueError("TimeStop needs max_bars or max_seconds")
        for name, value in (("max_bars", self.max_bars), ("max_seconds", self.max_seconds)):
            if value is not None and value <= 0:
                raise ValueError("TimeStop.%s must be positive" % name)


@dataclass(frozen=True)
class ExitPlan:
    """Everything about how this position is meant to end.

    Carried on the signal so paper, live and backtest read the same plan, and so
    an executor that can place venue-side protective orders has the numbers to
    do it instead of relying on an in-process monitor.
    """

    stop_loss: Optional[StopSpec] = None
    take_profit: Tuple[TakeProfitLeg, ...] = ()
    time_stop: Optional[TimeStop] = None
    #: close when the entry condition inverts (e.g. opposite MA cross)
    exit_on_opposite_signal: bool = True
    #: free-text reason this plan exists, for the user-facing card
    note: str = ""

    def __post_init__(self) -> None:
        total = sum(leg.close_pct for leg in self.take_profit)
        if total > 1.0 + 1e-9:
            raise ValueError(
                "ExitPlan: take-profit legs close %.3f of the position (>1)" % total
            )

    @property
    def has_protection(self) -> bool:
        return self.stop_loss is not None or self.time_stop is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stop_loss": asdict(self.stop_loss) if self.stop_loss else None,
            "take_profit": [asdict(leg) for leg in self.take_profit],
            "time_stop": asdict(self.time_stop) if self.time_stop else None,
            "exit_on_opposite_signal": self.exit_on_opposite_signal,
            "note": self.note,
        }


@dataclass(frozen=True)
class EntrySpec:
    """How to get in (or, for a reduce/close, how to get out)."""

    type: EntryType = EntryType.MARKET
    price: Optional[float] = None
    zone: Optional[Tuple[float, float]] = None
    time_in_force: str = "GTC"
    post_only: bool = False

    def __post_init__(self) -> None:
        needs_price = self.type in (EntryType.LIMIT, EntryType.STOP,
                                    EntryType.STOP_LIMIT, EntryType.LIMIT_MAKER)
        if needs_price and self.price is None and self.zone is None:
            raise ValueError("EntrySpec: %s needs a price or zone" % self.type.value)
        if self.time_in_force not in ("GTC", "IOC", "FOK"):
            raise ValueError("EntrySpec.time_in_force must be GTC / IOC / FOK")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "price": self.price,
            "zone": list(self.zone) if self.zone else None,
            "time_in_force": self.time_in_force,
            "post_only": self.post_only,
        }


@dataclass(frozen=True)
class SizeSpec:
    mode: SizeMode = SizeMode.EQUITY_PCT
    value: float = 0.0
    leverage: Optional[float] = None
    max_notional_quote: Optional[float] = None
    #: set by the intent, not by the author — see StandardSignal.__post_init__
    reduce_only: bool = False

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("SizeSpec.value must be >= 0")
        if self.mode in (SizeMode.EQUITY_PCT, SizeMode.RISK_PCT, SizeMode.POSITION_PCT):
            if not 0.0 <= self.value <= 1.0:
                raise ValueError("SizeSpec: %s expects a fraction in [0, 1]" % self.mode.value)
        if self.leverage is not None and self.leverage <= 0:
            raise ValueError("SizeSpec.leverage must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value, "value": self.value, "leverage": self.leverage,
            "max_notional_quote": self.max_notional_quote, "reduce_only": self.reduce_only,
        }


@dataclass(frozen=True)
class RiskLimits:
    max_loss_quote: Optional[float] = None
    max_position_quote: Optional[float] = None
    max_leverage: Optional[float] = None
    #: distance to liquidation at the proposed size, as a fraction of price
    liquidation_buffer_pct: Optional[float] = None
    daily_loss_cap_quote: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Evidence:
    source: str
    observed: Dict[str, Any] = field(default_factory=dict)
    available_time: Optional[int] = None
    url: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Provenance:
    strategy_id: str
    strategy_version: str = "v1"
    snapshot_id: str = ""
    config_hash: str = ""
    inputs: Tuple[str, ...] = ()
    run_id: str = ""
    trace_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["inputs"] = list(self.inputs)
        return out


@dataclass(frozen=True)
class SelectionCandidate:
    """One row of a cross-sectional ranking."""

    symbol: str
    rank: int
    score: float
    direction: Direction = Direction.NEUTRAL
    reason: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    #: optional per-candidate trade plan, so a selector can hand over a
    #: ready-to-execute instruction instead of just a watchlist
    trade: Optional["StandardSignal"] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol, "rank": self.rank, "score": self.score,
            "direction": self.direction.value, "reason": self.reason,
            "features": dict(self.features),
            "trade": self.trade.to_dict() if self.trade else None,
        }


# ---------------------------------------------------------------------------
# the signal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandardSignal:
    """One complete instruction or observation from a standard bot."""

    # ---- identity ----
    bot_id: str
    decision_time: int                       # epoch ms, == snapshot decision_as_of
    provenance: Provenance
    signal_id: str = ""
    schema: str = SCHEMA_VERSION
    bot_version: str = "v1"
    #: trade / selection / advisory. Leave as None and it is derived from the
    #: payload, so it cannot disagree with the fields it describes.
    kind: Optional[SignalKind] = None

    # ---- instrument (对应标的) ----
    symbol: Optional[str] = None
    venue: str = "binance"
    product: str = "usd_m_perpetual"         # spot | usd_m_perpetual | coin_m | margin | option
    base_asset: str = ""
    quote_asset: str = ""
    market_scope: MarketScope = MarketScope.SINGLE

    # ---- decision ----
    intent: PositionIntent = PositionIntent.HOLD
    direction: Direction = Direction.NEUTRAL
    advisory_action: Optional[AdvisoryAction] = None
    score: float = 0.0                       # 0-100 strength
    confidence: float = 0.0                  # 0-1 evidence confidence

    # ---- execution plan ----
    entry: Optional[EntrySpec] = None
    exit_plan: Optional[ExitPlan] = None
    size: Optional[SizeSpec] = None
    risk: RiskLimits = field(default_factory=RiskLimits)

    # ---- validity ----
    time_horizon: TimeHorizon = TimeHorizon.INTRADAY
    horizon_seconds: int = 3600
    valid_until: Optional[int] = None

    # ---- explanation ----
    topic: str = ""
    reason_codes: Tuple[str, ...] = ()
    summary: str = ""
    recommended_behavior: str = ""
    evidence: Tuple[Evidence, ...] = ()

    # ---- data quality ----
    data_quality: DataQuality = DataQuality.GOOD
    source_status: Dict[str, str] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()

    # ---- cross-section payload ----
    candidates: Tuple[SelectionCandidate, ...] = ()
    universe_size: int = 0

    # ---- safety ----
    auto_trade_eligible: bool = False
    requires_confirmation: bool = True
    dedup_key: str = ""

    def __post_init__(self) -> None:
        if not self.bot_id:
            raise ValueError("StandardSignal.bot_id is required")
        # NaN fails every comparison, so a range check alone lets it through --
        # and worse, `min(100.0, nan)` returns 100.0, so a NaN computed upstream
        # can arrive already laundered into MAXIMUM strength. Name it.
        for field_name in ("score", "confidence"):
            value = getattr(self, field_name)
            if value != value:
                raise ValueError(
                    "%s is NaN. That means the number it was computed from was "
                    "missing; emit no signal, or say so in warnings — do not "
                    "publish a strength derived from absent data." % field_name
                )
        if not 0.0 <= self.score <= 100.0:
            raise ValueError("score must be in [0, 100]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")

        actionable = self.intent is not PositionIntent.HOLD
        if actionable and self.market_scope is MarketScope.SINGLE and not self.symbol:
            raise ValueError(
                "intent=%s needs a symbol — an instruction with no instrument is "
                "not executable" % self.intent.value
            )

        # reduce_only is a property of the intent, never an independent claim
        if self.size is not None and self.size.reduce_only != self.intent.reduce_only:
            object.__setattr__(
                self, "size",
                SizeSpec(
                    mode=self.size.mode, value=self.size.value,
                    leverage=self.size.leverage,
                    max_notional_quote=self.size.max_notional_quote,
                    reduce_only=self.intent.reduce_only,
                ),
            )

        # an entry with no way out is the single most common way a bot loses money
        if self.intent.is_entry and self.exit_plan is None:
            raise ValueError(
                "intent=%s has no exit_plan: state the stop/time-stop explicitly, "
                "or pass ExitPlan(exit_on_opposite_signal=True) to say so on "
                "purpose" % self.intent.value
            )

        # spot cannot hold negative exposure; closing a long is fine
        if self.product == "spot" and self.intent.requires_short_capability:
            raise ValueError(
                "intent=%s requires short capability but product=spot; use "
                "CLOSE_LONG/REDUCE_LONG to sell an existing position"
                % self.intent.value
            )

        if self.market_scope is MarketScope.CROSS_SECTION and not self.candidates:
            # An empty basket is a real answer: "nothing qualified this round" is
            # how a rebalancing selector says hold nothing, and a consumer needs
            # to receive it — going silent is indistinguishable from a crash.
            #
            # But it must be an answer, not an absence. The vacuous case this
            # guards is a bot that evaluated NOTHING and still reports all-clear:
            # empty basket, empty universe, and quality GOOD. Either a universe
            # was actually ranked (universe_size > 0) or the read failed and the
            # signal says so (data_quality != GOOD) — plus a stated reason either
            # way, so a reader can tell "none qualified" from "we broke".
            evaluated = self.universe_size > 0 or self.data_quality is not DataQuality.GOOD
            stated = bool(self.reason_codes or self.summary or self.warnings)
            if not (evaluated and stated):
                raise ValueError(
                    "cross-section signal carries no candidates. An empty basket "
                    "is allowed, but it must show it is an answer: set "
                    "universe_size to what was actually ranked (or set "
                    "data_quality to degraded/insufficient if the universe could "
                    "not be read), AND state a reason_codes/summary. Got "
                    "universe_size=%d data_quality=%s reason=%r"
                    % (self.universe_size, self.data_quality.value,
                       tuple(self.reason_codes) or self.summary)
                )

        # ---- kind: one field a consumer can switch on -------------------- #
        # Derived, not trusted. A producer that forgets to set market_scope
        # would otherwise publish a basket labelled "single", which is exactly
        # how a consumer ends up parsing a selection result as a trade.
        derived = (
            SignalKind.SELECTION if self.candidates
            else SignalKind.ALERT if self.advisory_action is not None
            else SignalKind.TRADE
        )
        if self.kind is None:
            object.__setattr__(self, "kind", derived)
        elif self.kind is not derived:
            raise ValueError(
                "kind=%s contradicts the payload (candidates=%d, advisory_action=%s) "
                "which describes a %s signal. Leave kind unset and it is derived."
                % (self.kind.value, len(self.candidates),
                   self.advisory_action.value if self.advisory_action else None,
                   derived.value)
            )
        # keep the scope consistent with the kind: a basket is cross-sectional
        # by definition, whatever the bot's spec happened to default to.
        if self.kind is SignalKind.SELECTION and self.market_scope is MarketScope.SINGLE:
            object.__setattr__(self, "market_scope", MarketScope.CROSS_SECTION)

        if self.auto_trade_eligible and self.advisory_action is not None:
            raise ValueError("an advisory signal cannot be auto-trade eligible")

        if self.direction is Direction.NEUTRAL and self.intent.target_side in ("long", "short"):
            # keep the informational field consistent with the instruction
            object.__setattr__(
                self, "direction",
                Direction.LONG if self.intent.target_side == "long" else Direction.SHORT,
            )

        if not self.signal_id:
            object.__setattr__(self, "signal_id", self._derive_signal_id())
        if self.valid_until is None:
            object.__setattr__(
                self, "valid_until", int(self.decision_time) + self.horizon_seconds * 1000
            )
        if not self.dedup_key:
            object.__setattr__(self, "dedup_key", self._derive_dedup_key())

    # ---- derived ----

    def _derive_signal_id(self) -> str:
        import uuid

        namespace = uuid.UUID("7e1c4a2b-8f36-5d07-9b14-3a6e5c8d0f92")
        return str(
            uuid.uuid5(
                namespace,
                "%s|%s|%s|%s|%s|%s"
                % (self.bot_id, self.bot_version, self.symbol or "-",
                   self.product, self.intent.value, self.decision_time),
            )
        )

    def _derive_dedup_key(self) -> str:
        """Notification de-duplication key. NOT an execution idempotency key."""
        return "%s:%s:%s:%s:%s" % (
            self.bot_id, self.venue, self.product,
            self.symbol or "GLOBAL", self.intent.value,
        )

    @property
    def is_actionable(self) -> bool:
        return self.intent is not PositionIntent.HOLD and self.advisory_action is None

    @property
    def closes_side(self) -> Optional[str]:
        """平仓方向 — which existing side this reduces, if any."""
        return self.intent.closes_side

    @property
    def order_side(self) -> Optional[str]:
        return self.intent.order_side

    def resolved_exit_prices(self, entry_price: float) -> Dict[str, Any]:
        """Absolute stop / take-profit prices for a fill at ``entry_price``."""
        side = self.intent.target_side or "long"
        if self.exit_plan is None:
            return {"stop_loss_price": None, "take_profit": []}
        stop = (
            self.exit_plan.stop_loss.resolve_price(entry_price=entry_price, side=side)
            if self.exit_plan.stop_loss
            else None
        )
        legs = [
            {
                "price": leg.resolve_price(entry_price=entry_price, side=side),
                "close_pct": leg.close_pct,
            }
            for leg in self.exit_plan.take_profit
        ]
        return {"stop_loss_price": stop, "take_profit": legs}

    # ---- serialisation ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value if self.kind else SignalKind.TRADE.value,
            "bot_id": self.bot_id,
            "bot_version": self.bot_version,
            "signal_id": self.signal_id,
            "decision_time": self.decision_time,
            "symbol": self.symbol,
            "venue": self.venue,
            "product": self.product,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "market_scope": self.market_scope.value,
            "intent": self.intent.value,
            "target_side": self.intent.target_side,
            "closes_side": self.intent.closes_side,
            "order_side": self.intent.order_side,
            "reduce_only": self.intent.reduce_only,
            "direction": self.direction.value,
            "advisory_action": self.advisory_action.value if self.advisory_action else None,
            "score": self.score,
            "confidence": self.confidence,
            "entry": self.entry.to_dict() if self.entry else None,
            "exit_plan": self.exit_plan.to_dict() if self.exit_plan else None,
            "size": self.size.to_dict() if self.size else None,
            "risk": self.risk.to_dict(),
            "time_horizon": self.time_horizon.value,
            "horizon_seconds": self.horizon_seconds,
            "valid_until": self.valid_until,
            "topic": self.topic,
            "reason_codes": list(self.reason_codes),
            "summary": self.summary,
            "recommended_behavior": self.recommended_behavior,
            "evidence": [item.to_dict() for item in self.evidence],
            "data_quality": self.data_quality.value,
            "source_status": dict(self.source_status),
            "warnings": list(self.warnings),
            "candidates": [item.to_dict() for item in self.candidates],
            "universe_size": self.universe_size,
            "auto_trade_eligible": self.auto_trade_eligible,
            "requires_confirmation": self.requires_confirmation,
            "dedup_key": self.dedup_key,
            "provenance": self.provenance.to_dict(),
        }

    def to_execution_request(self) -> Dict[str, Any]:
        """Everything the order layer needs, minus what only it can supply.

        Maps onto the venue-neutral order contract
        (``venue_class`` / ``intent_type`` / ``instrument`` / ``side`` /
        ``size`` / ``params``). Two fields are deliberately absent:

        * ``idempotency_key`` — ``instance:node:event_ref``; only the executor
          knows run identity.
        * ``strategy_instance_id`` / ``node_id`` — likewise run-scoped.

        Raises for a non-actionable signal, so an ALERT can never be walked
        into an order by a careless caller.
        """
        if self.advisory_action is not None:
            raise ValueError(
                "advisory signal (action=%s) is not executable" % self.advisory_action.value
            )
        if not self.is_actionable:
            raise ValueError("intent=hold is not executable")

        entry = self.entry or EntrySpec()
        venue_class = _VENUE_CLASS.get(self.product, "CEX_PERP")
        params: Dict[str, Any] = {"time_in_force": entry.time_in_force}
        if entry.price is not None:
            params["price"] = entry.price
        if entry.zone is not None:
            params["zone"] = list(entry.zone)
        if self.exit_plan is not None and self.exit_plan.stop_loss is not None:
            resolved = self.resolved_exit_prices(entry.price or 0.0)
            params["bracket"] = {
                "stop": resolved["stop_loss_price"],
                "take_profit": [leg["price"] for leg in resolved["take_profit"]],
                "exchange_managed": self.exit_plan.stop_loss.exchange_managed,
            }
        return {
            "venue_class": venue_class,
            "intent_type": _INTENT_TYPE.get(entry.type, "MARKET"),
            "instrument": self.symbol,
            "side": (self.intent.order_side or "").upper() or None,
            "position_intent": self.intent.value,
            "reduce_only": self.intent.reduce_only,
            "size": self.size.to_dict() if self.size else None,
            "params": params,
            "client_tag": self.dedup_key[:36],
            "source_signal_id": self.signal_id,
            "requires_confirmation": self.requires_confirmation,
        }


_VENUE_CLASS = {
    "spot": "CEX_SPOT",
    "margin": "CEX_MARGIN",
    "usd_m_perpetual": "CEX_PERP",
    "coin_m_perpetual": "CEX_PERP",
    "option": "CEX_OPTION",
}

_INTENT_TYPE = {
    EntryType.MARKET: "MARKET",
    EntryType.LIMIT: "LIMIT",
    EntryType.STOP: "STOP_LOSS",
    EntryType.STOP_LIMIT: "STOP_LOSS_LIMIT",
    EntryType.LIMIT_MAKER: "LIMIT_MAKER",
}
