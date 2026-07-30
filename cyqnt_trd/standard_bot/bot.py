"""``StandardBot`` — one base class, complete input in, complete output out.

The repo grew three bot shapes independently: ``BlockStrategyPlugin`` (per-bar
booleans on one instrument), ``SelectionStrategyPlugin`` (rank a universe) and
``AdvisoryBot`` (emit alerts off named frames). They already share the registry
and ``run_pipeline_step``, but each declares its inputs differently and each
returns a differently-shaped payload — so "what can this bot read?" and "what
can it tell me to do?" had three answers.

``StandardBot`` is the single answer:

**Input** — a bot declares :class:`DataRequest` entries against the data-node
catalog. The framework fetches them under one decision time, records per-node
status, and hands over a :class:`BotContext`. A bot cannot read a node it did
not declare, and a declared node that failed is visible as status rather than
absent data.

**Output** — :class:`StandardSignal`, which states the instrument, the position
intent (including **which side is being closed**), the entry, the full exit
plan, sizing, risk limits, validity, reasoning, evidence, data quality and
provenance. One shape covers trade, selection and advisory; the difference is
which intents a bot is *allowed* to emit, and that is declared, checked, and
impossible to violate by accident.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .core import (
    AdvisoryAction,
    DataQuality,
    FrameKind,
    MarketScope,
    PositionIntent,
    Provenance,
    SignalBatch,
    SignalEnvelope,
    SignalKind,
    SignalProvenance,
    SizeMode,
    StandardSignal,
    TradeSide,
)

__all__ = [
    "BotKind",
    "DataRequest",
    "BotSpec",
    "BotContext",
    "StandardBot",
    "CapabilityError",
]

_BOT_NS = uuid.UUID("4b2e9d17-5c80-51f3-a6d4-8e0b7c93f215")


class BotKind(str, Enum):
    """What a bot is permitted to emit.

    This is a *capability*, not a label: :meth:`StandardBot.run` refuses a
    signal the bot's kind does not allow, so an advisory bot cannot emit an
    executable instruction even if its author writes one.
    """

    TRADE = "trade"          # position instructions on named instruments
    SELECTION = "selection"  # cross-sectional ranking
    ADVISORY = "advisory"    # alerts only, never actionable


class CapabilityError(RuntimeError):
    """A bot tried to emit something its declared kind does not permit."""


@dataclass(frozen=True)
class DataRequest:
    """One declared input: a catalog node plus the params to call it with."""

    node: str
    params: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    #: name the bot reads it back under; defaults to the node name
    alias: str = ""

    @property
    def key(self) -> str:
        return self.alias or self.node


@dataclass(frozen=True)
class BotSpec:
    """Everything static about a bot — the half a UI or registry can read."""

    bot_id: str
    kind: BotKind
    display_name: str = ""
    version: str = "v1"
    description: str = ""
    #: products this bot is valid for; a spot-only bot cannot emit short intents
    products: Tuple[str, ...] = ("usd_m_perpetual",)
    market_scope: MarketScope = MarketScope.SINGLE
    #: intents the bot may emit. Empty = derive from ``kind``.
    allowed_intents: Tuple[PositionIntent, ...] = ()
    default_horizon_seconds: int = 3600
    #: True only when the bot actually observes current exposure — either by
    #: declaring a PositionFrame input, or because the caller supplies
    #: ``positions``. A bot that does NOT read positions may not emit
    #: CLOSE_* / REDUCE_* / FLIP_* / FLAT: those assert that a position exists,
    #: and a bot that never looked is guessing. This is the single most
    #: expensive thing to get wrong, so it is a declaration, not a convention.
    reads_positions: bool = False

    def resolved_intents(self) -> Tuple[PositionIntent, ...]:
        if self.allowed_intents:
            return self.allowed_intents
        if self.kind is BotKind.ADVISORY:
            return (PositionIntent.HOLD,)
        if self.kind is BotKind.SELECTION:
            return (PositionIntent.HOLD, PositionIntent.OPEN_LONG, PositionIntent.OPEN_SHORT)
        spot_only = tuple(self.products) == ("spot",)
        if spot_only:
            # a spot book can buy, add, trim and close — it cannot go negative
            allowed = (
                PositionIntent.OPEN_LONG, PositionIntent.ADD_LONG,
                PositionIntent.REDUCE_LONG, PositionIntent.CLOSE_LONG,
                PositionIntent.FLAT, PositionIntent.HOLD,
            )
        else:
            allowed = tuple(PositionIntent)
        if not self.reads_positions:
            allowed = tuple(
                intent for intent in allowed
                if not (intent.is_exit or intent in (PositionIntent.ADD_LONG,
                                                     PositionIntent.ADD_SHORT))
            )
        return allowed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "kind": self.kind.value,
            "display_name": self.display_name or self.bot_id,
            "version": self.version,
            "description": self.description,
            "products": list(self.products),
            "market_scope": self.market_scope.value,
            "allowed_intents": [item.value for item in self.resolved_intents()],
            "reads_positions": self.reads_positions,
            "default_horizon_seconds": self.default_horizon_seconds,
        }


@dataclass
class BotContext:
    """The complete input handed to :meth:`StandardBot.decide` (``cyqnt.input/v1``).

    Two views of the same data:

    ``frames``  raw, exactly as the source returned it — the escape hatch.
    ``typed``   normalised to a canonical shape (:class:`TypedFrame`), so a bot
                reading funding, OI and news uses one column vocabulary
                (``instrument_id`` / ``event_time`` / ``available_time`` /
                ``metric`` / ``value``) instead of three.

    ``source_status`` covers every declared node including the ones that failed,
    so "I did not read it" and "I read it and it was empty" stay
    distinguishable.
    """

    decision_time: int
    frames: Dict[str, Any] = field(default_factory=dict)
    typed: Dict[str, Any] = field(default_factory=dict)
    source_status: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    #: current exposure per symbol, signed: >0 long, <0 short, 0/absent flat
    positions: Dict[str, float] = field(default_factory=dict)
    equity: Optional[float] = None
    snapshot_id: str = ""
    run_id: str = ""
    trace_id: str = ""

    # ---- reading ----

    def frame(self, key: str, default: Any = None) -> Any:
        """Raw frame, as the source returned it."""
        return self.frames.get(key, default)

    def view(self, key: str):
        """Normalised :class:`TypedFrame`, or ``None`` when the node is RAW."""
        return self.typed.get(key)

    def require_view(self, key: str):
        typed = self.typed.get(key)
        if typed is None:
            raise KeyError(
                "input %r has no canonical shape (status=%s). Either it was not "
                "fetched, or its node declares emits=RAW — use ctx.frame(%r)."
                % (key, self.source_status.get(key, "not requested"), key)
            )
        return typed

    def metric(self, key: str, name: str, *, instrument: Optional[str] = None):
        """Latest value of one metric from a MetricFrame input.

        Returns ``None`` when the source failed, the metric is absent, or the
        value is not numeric — never 0. A missing reading and a reading of zero
        are different facts, and this is where they stay different.
        """
        typed = self.typed.get(key)
        return None if typed is None else typed.metric(name, instrument=instrument)

    def series(self, key: str, column: str, *, instrument: Optional[str] = None):
        typed = self.typed.get(key)
        if typed is None:
            import pandas as pd

            return pd.Series(dtype=float)
        return typed.series(column, instrument=instrument)

    def views_of(self, kind: "FrameKind") -> Dict[str, Any]:
        """Every typed input of one canonical shape, keyed by input name."""
        return {key: typed for key, typed in self.typed.items() if typed.kind is kind}

    def require(self, key: str) -> Any:
        if key not in self.frames:
            raise KeyError(
                "input %r was not fetched (status=%s). Declare it in required_data()."
                % (key, self.source_status.get(key, "not requested"))
            )
        return self.frames[key]

    def has(self, key: str) -> bool:
        frame = self.frames.get(key)
        if frame is None:
            return False
        return not getattr(frame, "empty", False)

    def position(self, symbol: str) -> float:
        return float(self.positions.get(str(symbol).upper(), 0.0))

    def adopt_positions_from(self, key: str) -> int:
        """Fill ``positions`` from a PositionFrame input. Returns rows used.

        A bot that declares ``contract_positions`` should not also be handed a
        separately-built position dict — that is two sources of truth for "am I
        long or short", the one question the exit path cannot get wrong.
        """
        typed = self.typed.get(key)
        if typed is None or typed.kind is not FrameKind.POSITION or typed.empty:
            return 0
        import pandas as pd

        used = 0
        for _, row in typed.frame.iterrows():
            symbol = str(row.get("instrument_id", "")).upper()
            if not symbol:
                continue
            quantity = pd.to_numeric(row.get("quantity"), errors="coerce")
            if pd.isna(quantity):
                continue
            side = str(row.get("side", "")).lower()
            signed = -abs(float(quantity)) if side == "short" else float(quantity)
            self.positions[symbol] = signed
            used += 1
        return used

    def side_of(self, symbol: str) -> str:
        qty = self.position(symbol)
        return "long" if qty > 0 else ("short" if qty < 0 else "flat")

    # ---- quality ----

    @property
    def degraded_inputs(self) -> List[str]:
        return sorted(k for k, v in self.source_status.items() if v != "ok")

    @property
    def inferred_availability(self) -> List[str]:
        """Inputs whose ``available_time`` was assumed rather than read.

        Each is a replay that may be optimistic by that source's real
        publication lag.
        """
        return sorted(
            key for key, typed in self.typed.items()
            if getattr(typed, "available_time_inferred", False)
        )

    def data_quality(self, *, required: Sequence[str] = ()) -> DataQuality:
        """GOOD when everything read cleanly; INSUFFICIENT when a required
        input failed; DEGRADED in between."""
        for key in required:
            if self.source_status.get(key, "error") == "error":
                return DataQuality.INSUFFICIENT
        return DataQuality.GOOD if not self.degraded_inputs else DataQuality.DEGRADED


def _frames_from_snapshot(snapshot: Any) -> Dict[str, Any]:
    """Expose ``DataSnapshot.market`` / ``.universe`` as ``BotContext`` frames.

    ``DataSnapshot`` is the one input object for every bot shape, but its two
    oldest slots are typed bundles (``MarketBundle`` of ``Bar`` objects,
    ``UniverseBundle`` of tables) while a :class:`StandardBot` reads named
    DataFrames off ``ctx.frames``. Without this bridge ``_coerce_context`` saw
    only ``snapshot.frames`` — so a v2 bot handed a fully populated snapshot got
    **empty inputs**, and the OHLCV a block strategy reads happily was invisible
    to it.

    Keys follow the data-node catalog so ``DataRequest("klines")`` and
    ``ctx.frame("klines")`` line up:

    ``klines``                      primary instrument/timeframe series
    ``klines:<SYMBOL>:<TIMEFRAME>`` every series, for multi-instrument bots
    ``universe`` / ``ticker_rank``  cross-sectional tables (selection bots)

    Derivative columns (``funding_rate`` / ``open_interest``) ride inside the
    bar frames exactly as they do for block strategies — same spill, same names.
    """
    out: Dict[str, Any] = {}
    market = getattr(snapshot, "market", None)
    if market is not None and getattr(market, "bars", None):
        from ..blocks.data import bars_to_df  # local: keep core import-light

        primary_tf = getattr(getattr(snapshot, "meta", None), "primary_timeframe", None)
        primary_key = None
        for key, bars in market.bars.items():
            if not bars:
                continue
            frame = bars_to_df(bars)
            # BarFrame@1.0 requires available_time — the PIT gate, i.e. when we
            # could first have KNOWN the row, as distinct from when it happened.
            # A confirmed bar becomes knowable at its close, so close_time is the
            # honest value. bars_to_df does not emit it (nothing needed it before
            # the input contract existed), and without it every kline frame fails
            # its own declared shape.
            if "available_time" not in frame.columns and "close_time" in frame.columns:
                frame = frame.copy()
                frame["available_time"] = frame["close_time"]
            out["klines:%s" % key.replace("|", ":")] = frame
            if primary_key is None or (primary_tf and key.endswith("|%s" % primary_tf)):
                if primary_key is None or key.endswith("|%s" % primary_tf):
                    primary_key, out["klines"] = key, frame

    universe = getattr(snapshot, "universe", None)
    if universe is not None:
        if getattr(universe, "universe", None) is not None:
            out["universe"] = universe.universe
        if getattr(universe, "ticker_rank", None) is not None:
            out["ticker_rank"] = universe.ticker_rank
        if getattr(universe, "ticker_rank_prev", None) is not None:
            out["ticker_rank_prev"] = universe.ticker_rank_prev
        for symbol, frame in (getattr(universe, "klines", None) or {}).items():
            out.setdefault("klines:%s" % str(symbol).upper(), frame)
    return out


#: Payload keys the pre-v2 engines read directly. ``cyqnt.signal/v2`` names the
#: same concepts differently (``exit_plan`` not ``exit_spec``; ``size`` is a
#: SizeSpec object, not a float), so without a translation at this boundary a v2
#: bot cannot drive them: ``SnapshotBacktestRunner`` does
#: ``float(payload["size"])`` — a TypeError on a dict — and reads
#: ``payload["exit_spec"]``, so a v2 ExitPlan would be silently ignored and the
#: position would run with no stop.
#:
#: The translation lives here rather than in the engines: v2 is the semantic
#: contract, SignalEnvelope is the transport, and the transport boundary is
#: where shapes get adapted. The full v2 dict is still carried untouched, so a
#: v2-aware consumer never sees the compat keys as authoritative.
_ENGINE_COMPAT_KEYS = ("engine_size", "exit_spec", "target_position", "risk_hints")


def _exit_plan_to_exit_spec(signal: StandardSignal) -> Optional[Dict[str, Any]]:
    """Translate a v2 ``ExitPlan`` into the engines' ``exit_spec`` dict.

    Returns ``None`` when the plan carries no price/time exit, which the engines
    read as "exit on the opposite signal only" — the same meaning as
    ``ExitPlan.exit_on_opposite_signal`` with nothing else set.
    """
    plan = signal.exit_plan
    if plan is None:
        return None
    stop, legs, tstop = plan.stop_loss, list(plan.take_profit), plan.time_stop
    max_bars = int(tstop.max_bars) if (tstop and tstop.max_bars) else 9999
    side = "long" if (signal.intent.target_side or "") == "long" else "short"

    # Only the FIRST take-profit rung is representable: no simulation engine
    # supports partial closes yet, so a ladder is flattened and the drop is
    # recorded rather than silently truncated.
    tp = legs[0] if legs else None
    out: Dict[str, Any] = {"max_bars": max_bars, "side": side}
    if len(legs) > 1:
        out["_dropped_tp_legs"] = len(legs) - 1

    if stop is None and tp is None:
        return {"type": "time_only", "max_bars": max_bars} if tstop else None

    # ATR-based (trailing takes precedence — it is a different engine type)
    if stop is not None and stop.atr_mult is not None and stop.atr_value:
        out["atr_at_entry"] = float(stop.atr_value)
        if stop.trailing:
            out.update(type="atr_trailing_stop", trail_mult=float(stop.atr_mult))
            return out
        out.update(type="atr_stop_tp", stop_mult=float(stop.atr_mult))
        out["tp_mult"] = float(tp.atr_mult) if (tp and tp.atr_mult is not None) else 0.0
        return out

    # Percent-based
    if (stop is not None and stop.pct is not None) or (tp is not None and tp.pct is not None):
        out.update(type="pct_stop_tp")
        if stop is not None and stop.pct is not None:
            out["stop_pct"] = float(stop.pct)
        if tp is not None and tp.pct is not None:
            out["tp_pct"] = float(tp.pct)
        return out

    # Absolute prices — the runner compares these directly against bar high/low.
    if (stop is not None and stop.price is not None) or (tp is not None and tp.price is not None):
        out.update(type="pct_stop_tp")
        if stop is not None and stop.price is not None:
            out["stop_loss_price"] = float(stop.price)
        if tp is not None and tp.price is not None:
            out["take_profit_price"] = float(tp.price)
        return out

    return {"type": "time_only", "max_bars": max_bars} if tstop else None


def _size_to_equity_fraction(signal: StandardSignal) -> Optional[float]:
    """Best-effort SizeSpec → the engines' equity-fraction float.

    Returns ``None`` when the mode cannot be expressed as a fraction of equity
    without information the signal does not carry. Guessing would be worse than
    declining: the engines default an absent ``size`` to 1.0, so a wrong guess
    silently changes position size.

    * ``EQUITY_PCT``   — exact, used directly.
    * ``RISK_PCT``     — exact once the stop distance is known: risking ``r`` of
      equity with a stop ``d`` away means deploying ``r / d``. Capped at 1.0.
    * ``QUANTITY`` / ``QUOTE_AMOUNT`` — absolute; needs equity and mark price.
    * ``POSITION_PCT`` — a fraction of the *existing* position, not of equity.
    """
    spec = signal.size
    if spec is None:
        return None
    if spec.mode is SizeMode.EQUITY_PCT:
        return float(spec.value)
    if spec.mode is SizeMode.RISK_PCT:
        plan = signal.exit_plan
        stop = plan.stop_loss if plan else None
        if stop is None or not spec.value:
            return None
        stop_pct: Optional[float] = None
        if stop.pct is not None:
            stop_pct = float(stop.pct)
        elif stop.atr_mult is not None and stop.atr_value:
            ref = _entry_reference_price(signal)
            if ref:
                stop_pct = float(stop.atr_mult) * float(stop.atr_value) / float(ref)
        if not stop_pct:
            return None
        return min(1.0, float(spec.value) / stop_pct)
    return None


def _entry_reference_price(signal: StandardSignal) -> Optional[float]:
    """A price to measure an ATR stop distance against, if the signal states one.

    ``EntrySpec`` carries ``price`` (limit/stop orders) or ``zone`` (a range);
    a market entry states neither, in which case the distance is unknowable at
    signal time and the caller must decline rather than assume.
    """
    entry = signal.entry
    if entry is None:
        return None
    if entry.price is not None:
        return float(entry.price)
    if entry.zone:
        lo, hi = entry.zone
        return (float(lo) + float(hi)) / 2.0
    return None


def _engine_compat_payload(signal: StandardSignal) -> Dict[str, Any]:
    """Keys the pre-v2 engines need, derived from the v2 signal. See above."""
    _side = signal.intent.target_side or ""
    target = 1 if _side == "long" else (-1 if _side == "short" else 0)
    out: Dict[str, Any] = {
        "target_position": target,
        "risk_hints": {"target_position": target},
        "bar_timestamp": signal.decision_time,
    }
    spec = _exit_plan_to_exit_spec(signal)
    if spec is not None:
        out["exit_spec"] = spec
    frac = _size_to_equity_fraction(signal)
    if frac is not None:
        # NOT under "size": that key belongs to v2's SizeSpec object, and
        # overwriting it with a float would make the payload fail its own
        # schema. The engines read engine_size first.
        out["engine_size"] = frac
    elif signal.size is not None:
        # The v2 SizeSpec dict must NOT be left under "size": the engines call
        # float() on it. Drop the key entirely (so they fall back to their own
        # default) and say so loudly — an unnoticed fallback to size=1.0 means
        # trading full equity, which is exactly the kind of silent difference
        # this contract exists to remove.
        out["engine_size"] = None
        out["size_unresolved"] = (
            "SizeSpec(mode=%s) cannot be expressed as an equity fraction from the "
            "signal alone — needs account equity and/or a stop distance. Resolve "
            "it in the executor, or emit EQUITY_PCT for engine-driven sizing."
            % signal.size.mode.value
        )
    return out


class StandardBot(ABC):
    """Base class. Subclasses implement :meth:`required_data` and :meth:`decide`."""

    spec: BotSpec

    # ---- identity (SignalPlugin protocol surface) ----

    @property
    def plugin_id(self) -> str:
        return self.spec.bot_id

    @property
    def plugin_version(self) -> str:
        return self.spec.version

    def required_inputs(self) -> Dict[str, bool]:
        """Standard plugin surface: declared input key -> required."""
        base = {
            "market": False, "social": False, "onchain": False,
            "derivatives": False, "universe": False,
        }
        for request in self.required_data():
            base[request.key] = request.required
        return base

    def needed_timeframes(self) -> List[str]:
        """Bar timeframes the assembler must fetch, from the declared inputs.

        The last piece of the ``SignalPlugin`` protocol a bot was missing, and
        the reason no entrypoint could run one: the registry asks every plugin
        this before assembling a snapshot, so a bot that could not answer could
        not be scheduled. Read off ``required_data`` rather than configured
        separately — two places to state the same timeframe is one place to get
        it wrong.
        """
        out: List[str] = []
        for request in self.required_data():
            timeframe = (request.params or {}).get("interval") or (request.params or {}).get(
                "timeframe")
            if timeframe and timeframe not in out:
                out.append(str(timeframe))
        return out

    # ---- the two things a subclass writes ----

    @abstractmethod
    def required_data(self) -> Sequence[DataRequest]:
        """Declare every catalog node this bot reads."""

    @abstractmethod
    def decide(self, ctx: BotContext) -> Iterable[StandardSignal]:
        """Turn the complete input into zero or more complete signals."""

    # ---- config ----

    def normalize_config(self, config: Any) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, Mapping):
            return dict(config)
        return dict(getattr(config, "__dict__", {}) or {})

    # ---- framework ----

    def fetch_context(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        decision_time: Optional[int] = None,
        positions: Optional[Mapping[str, float]] = None,
        equity: Optional[float] = None,
        frames: Optional[Mapping[str, Any]] = None,
        require_backtestable: bool = False,
    ) -> BotContext:
        """Fetch every declared node under one decision time.

        ``frames`` short-circuits the fetch for offline/replay use; anything it
        does not supply is still fetched.
        """
        from .core import TypedFrame
        from .data.catalog import get_node
        from .runtime import data as data_runtime

        supplied = dict(frames or {})
        collected: Dict[str, Any] = {}
        typed: Dict[str, Any] = {}
        status: Dict[str, str] = {}
        warnings: List[str] = []

        with data_runtime.session(
            as_of_ms=decision_time, require_backtestable=require_backtestable
        ) as sess:
            as_of = sess.as_of_ms
            for request in self.required_data():
                if request.key in supplied:
                    collected[request.key] = supplied[request.key]
                    status[request.key] = "ok"
                else:
                    try:
                        collected[request.key] = sess.call(request.node, **request.params)
                        status[request.key] = sess.results[request.node].status
                    except data_runtime.DataUnavailable as exc:
                        status[request.key] = "error"
                        warnings.append("%s: %s" % (request.key, exc.reason))
                        # keep going: a bot may still abstain intelligently, and
                        # the caller can see exactly what was missing.
                        continue
                view = self._normalize(
                    request, collected[request.key],
                    status=status[request.key], as_of=as_of, warnings=warnings,
                )
                if view is not None:
                    typed[request.key] = view
            warnings.extend(sess.warnings())

        context = BotContext(
            decision_time=int(decision_time or as_of),
            frames=collected,
            typed=typed,
            source_status=status,
            warnings=warnings,
            config=self.normalize_config(config),
            positions={str(k).upper(): float(v) for k, v in (positions or {}).items()},
            equity=equity,
            snapshot_id=str(uuid.uuid5(_BOT_NS, "%s|%s" % (self.plugin_id, as_of))),
        )
        # A declared PositionFrame input IS the position source. Adopting it here
        # means the bot never has two answers to "am I long or short" — the one
        # question the exit path cannot afford to get wrong. An explicit
        # ``positions=`` argument still wins, so a backtest can drive it.
        if not positions:
            for key, view in typed.items():
                if view.kind is FrameKind.POSITION:
                    context.adopt_positions_from(key)
        return context

    def _normalize(self, request, frame, *, status: str, as_of: int, warnings: List[str]):
        """Vendor frame -> canonical shape, per the node's declared contract.

        A node that has not declared a shape (``emits=RAW``) yields no typed
        view; the bot reads it through ``ctx.frame()`` and knows it is on its
        own. A frame that fails validation is reported and downgraded rather
        than silently handed over as if it matched.
        """
        from .core import FrameValidationError, TypedFrame
        from .data.catalog import DataUnavailable, get_node

        try:
            spec = get_node(request.node)
        except KeyError:
            return None
        if not spec.typed:
            return None
        try:
            normalized, notes, inferred = spec.normalize(
                frame, available_time=as_of, params=request.params
            )
        except FrameValidationError as exc:
            warnings.append("%s: %s" % (request.key, exc))
            return TypedFrame(
                node=request.node, kind=spec.emits, frame=frame, status="degraded",
                as_of=as_of, availability=spec.availability.value,
                pit_hazard=spec.pit_hazard, warnings=(str(exc),),
            )
        warnings.extend("%s: %s" % (request.key, note) for note in notes)
        return TypedFrame(
            node=request.node, kind=spec.emits, frame=normalized, status=status,
            as_of=as_of, availability=spec.availability.value,
            pit_hazard=spec.pit_hazard, warnings=tuple(notes),
            available_time_inferred=inferred,
        )

    def decide_checked(self, ctx: BotContext) -> List[StandardSignal]:
        """Run ``decide`` and enforce every declared capability."""
        allowed = set(self.spec.resolved_intents())
        signals: List[StandardSignal] = []
        for signal in self.decide(ctx) or []:
            if not isinstance(signal, StandardSignal):
                raise TypeError(
                    "%s.decide must yield StandardSignal, got %r"
                    % (type(self).__name__, type(signal).__name__)
                )
            if signal.intent not in allowed:
                if not self.spec.reads_positions and (
                    signal.intent.is_exit
                    or signal.intent in (PositionIntent.ADD_LONG, PositionIntent.ADD_SHORT)
                ):
                    raise CapabilityError(
                        "%s emitted intent=%s but declares reads_positions=False. "
                        "That instruction asserts an existing position this bot "
                        "never observed. Either declare a PositionFrame input "
                        "(e.g. DataRequest('contract_positions')) and set "
                        "reads_positions=True, or emit only OPEN_*/HOLD."
                        % (self.plugin_id, signal.intent.value)
                    )
                raise CapabilityError(
                    "%s (kind=%s, products=%s) may not emit intent=%s; allowed: %s"
                    % (self.plugin_id, self.spec.kind.value, list(self.spec.products),
                       signal.intent.value,
                       ", ".join(sorted(item.value for item in allowed)))
                )
            if self.spec.kind is BotKind.ADVISORY:
                if signal.advisory_action is None:
                    raise CapabilityError(
                        "%s is an advisory bot; every signal must carry an "
                        "advisory_action" % self.plugin_id
                    )
                if signal.auto_trade_eligible:
                    raise CapabilityError(
                        "%s is advisory; auto_trade_eligible must be false" % self.plugin_id
                    )
            if signal.product not in self.spec.products:
                raise CapabilityError(
                    "%s is declared for products %s but emitted product=%s"
                    % (self.plugin_id, list(self.spec.products), signal.product)
                )
            signals.append(self._stamp(signal, ctx))
        return signals

    def _stamp(self, signal: StandardSignal, ctx: BotContext) -> StandardSignal:
        """Fill provenance and data quality from the context, not from the author."""
        from dataclasses import replace

        provenance = Provenance(
            strategy_id=self.plugin_id,
            strategy_version=self.plugin_version,
            snapshot_id=ctx.snapshot_id,
            config_hash=signal.provenance.config_hash,
            inputs=tuple(request.key for request in self.required_data()),
            run_id=ctx.run_id,
            trace_id=ctx.trace_id,
        )
        return replace(
            signal,
            provenance=provenance,
            source_status=dict(ctx.source_status),
            warnings=tuple(signal.warnings) + tuple(ctx.warnings),
        )

    # ---- SignalPlugin protocol ----

    def run(self, snapshot: Any, config: Any = None, context: Any = None) -> SignalBatch:
        """Standard entry point. ``snapshot`` may be a :class:`BotContext` or a
        ``DataSnapshot``; anything else is treated as "fetch it yourself"."""
        ctx = self._coerce_context(snapshot, config)
        signals = self.decide_checked(ctx)
        return self._to_batch(ctx, signals)

    def step(self, snapshot: Any, state: Any, config: Any = None, context: Any = None):
        from .signal.interfaces import SignalState, StepSignalResult

        ctx = self._coerce_context(snapshot, config)
        signals = self.decide_checked(ctx)
        seen = list((getattr(state, "values", {}) or {}).get("seen_signal_ids", []))
        known = set(seen)
        fresh = [signal for signal in signals if signal.signal_id not in known]
        seen.extend(signal.signal_id for signal in fresh)
        next_state = SignalState(
            plugin_id=self.plugin_id,
            values={"last_snapshot_id": ctx.snapshot_id, "seen_signal_ids": seen[-5000:]},
        )
        return StepSignalResult(
            state=next_state, signals=[self._to_envelope(item) for item in fresh]
        )

    def initialize_state(self):
        from .signal.interfaces import SignalState

        return SignalState(plugin_id=self.plugin_id, values={"seen_signal_ids": []})

    # ---- helpers ----

    def _coerce_context(self, snapshot: Any, config: Any) -> BotContext:
        if isinstance(snapshot, BotContext):
            if config is not None and not snapshot.config:
                snapshot.config = self.normalize_config(config)
            return snapshot
        meta = getattr(snapshot, "meta", None)
        if meta is not None:
            frames = dict(getattr(snapshot, "frames", {}) or {})
            frames.update(_frames_from_snapshot(snapshot))
            return BotContext(
                decision_time=int(getattr(meta, "decision_as_of", 0)
                                  or getattr(meta, "assembled_at", 0)),
                frames=frames,
                source_status={
                    key: (value.value if hasattr(value, "value") else str(value))
                    for key, value in (getattr(meta, "source_status", {}) or {}).items()
                },
                warnings=list(getattr(meta, "warnings", []) or []),
                config=self.normalize_config(config),
                snapshot_id=str(getattr(meta, "snapshot_id", "")),
            )
        return self.fetch_context(config=config)

    def typed_inputs(self) -> Dict[str, Any]:
        """``{input key: canonical schema}`` — the bot's declared input contract."""
        from .data.catalog import get_node

        described: Dict[str, Any] = {}
        for request in self.required_data():
            try:
                spec = get_node(request.node)
            except KeyError:
                described[request.key] = None
                continue
            schema = spec.input_schema
            described[request.key] = {
                "node": request.node,
                "required": request.required,
                "emits": spec.emits.value,
                "schema": schema.name if schema else None,
                "availability": spec.availability.value,
                "pit_hazard": spec.pit_hazard,
            }
        return described

    _KIND_MAP = {
        BotKind.TRADE: SignalKind.TRADE,
        BotKind.SELECTION: SignalKind.SELECTION,
        BotKind.ADVISORY: SignalKind.ALERT,
    }

    _SIDE_MAP = {"buy": TradeSide.BUY, "sell": TradeSide.SELL}

    def _to_envelope(self, signal: StandardSignal) -> SignalEnvelope:
        """Wrap a StandardSignal in the transport envelope the registry moves."""
        payload = signal.to_dict()
        payload.update(_engine_compat_payload(signal))
        return SignalEnvelope(
            version=signal.schema,
            signal_id=signal.signal_id,
            kind=self._KIND_MAP[self.spec.kind],
            strength=signal.score / 100.0,
            provenance=SignalProvenance(
                plugin_id=signal.provenance.strategy_id,
                plugin_version=signal.provenance.strategy_version,
                config_hash=signal.provenance.config_hash,
                input_fingerprint=signal.provenance.snapshot_id,
            ),
            instrument_id=signal.symbol,
            side=self._SIDE_MAP.get(signal.order_side or "", TradeSide.FLAT),
            time_horizon=signal.time_horizon.value,
            valid_until=signal.valid_until,
            payload=payload,
        )

    def _to_batch(self, ctx: BotContext, signals: Sequence[StandardSignal]) -> SignalBatch:
        envelopes = [self._to_envelope(item) for item in signals]
        return SignalBatch(
            signals=envelopes,
            batch_id=str(
                uuid.uuid5(
                    _BOT_NS,
                    "%s|%s|%d" % (self.plugin_id, ctx.snapshot_id, len(envelopes)),
                )
            ),
            created_at=int(time.time() * 1000),
        )
