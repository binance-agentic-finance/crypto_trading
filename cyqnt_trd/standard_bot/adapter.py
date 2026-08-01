"""The bridge between the two ways of writing a strategy.

The repo grew two paths that never met:

* **block path** — ``make_signals(df) -> (long, short)`` registered with
  ``blocks.strategy.register``. Every CLI runs it: backtest, paper, live. Its
  output is a :class:`SignalEnvelope` (``block/v1``, 10 keys, a free-form
  ``payload``), which is not the contract published to consumers.
* **v2 path** — ``StandardBot`` with ``required_data()`` + ``decide(ctx)``. Its
  output is a :class:`StandardSignal` (``cyqnt.signal/v2``, 42 keys, the
  published contract), it can read every declared data source, and it does coin
  selection. No CLI could run it.

Each was missing exactly what the other had, so this module supplies the two
translations and nothing else:

``envelope_to_signal``
    block/YAML output → ``cyqnt.signal/v2``. A YAML strategy now publishes the
    same JSON as a v2 bot.

``register_standard_bot``
    a ``StandardBot`` → the registry the entrypoints read, so ``mvp_backtest
    --strategy <bot_id>`` runs a v2 bot.

Both are deliberately lossy in one direction only, and say where. An envelope
does not know the account's position, so ``target_position`` can be translated
to "open long" but never to "close the short" — the information was never
there, and inventing it is how an executor ends up flattening a position nobody
asked it to touch.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence

from .core import (Direction, EntrySpec, EntryType, ExitPlan, MarketScope,
                   PositionIntent, Provenance, SelectionCandidate, SignalEnvelope,
                   SignalKind, SizeMode, SizeSpec, StandardSignal, StopSpec,
                   TakeProfitLeg, TimeStop)

__all__ = [
    "envelope_to_signal",
    "batch_to_signals",
    "register_standard_bot",
    "AdapterError",
]

_ADAPTER_NS = uuid.UUID("6f2b1d84-9c3a-5e17-b0d5-2a7c4e9f1b63")


class AdapterError(ValueError):
    """Raised when an envelope cannot be expressed as a v2 signal."""


# --------------------------------------------------------------------------- #
# block / YAML  ->  cyqnt.signal/v2                                            #
# --------------------------------------------------------------------------- #


def _intent_from_target(target: Optional[int], side: Optional[str]) -> PositionIntent:
    """``target_position`` → intent, without pretending to know the position.

    The block path emits a *target*: +1 long, -1 short, 0 flat. That is strictly
    less information than v2's intent, which distinguishes opening from adding
    and closing from reversing. Mapping +1 to ``OPEN_LONG`` is the honest
    reading — "the book should be long" — and it is what every engine already
    does with the same number. ``FLAT`` is likewise exact: "hold no position".

    What is NOT derivable is ``CLOSE_LONG`` vs ``OPEN_SHORT``, or ``ADD_*``,
    because that depends on current exposure the envelope never carried.
    """
    if target is None:
        if side == "buy":
            return PositionIntent.OPEN_LONG
        if side == "sell":
            return PositionIntent.OPEN_SHORT
        return PositionIntent.HOLD
    if target > 0:
        return PositionIntent.OPEN_LONG
    if target < 0:
        return PositionIntent.OPEN_SHORT
    return PositionIntent.FLAT


def _exit_plan_from_spec(spec: Optional[Dict[str, Any]]) -> Optional[ExitPlan]:
    """Invert ``_exit_plan_to_exit_spec``.

    The engines' ``exit_spec`` is a flat dict with a ``type`` discriminator;
    ``ExitPlan`` is structured. Round-tripping is exact for the fields the
    engines actually support, and the one thing they cannot represent — a
    multi-rung take-profit ladder — was already flattened on the way out, so
    nothing is lost here that was not lost there.
    """
    if not spec:
        return None
    kind = str(spec.get("type") or "")
    max_bars = spec.get("max_bars")
    time_stop = TimeStop(max_bars=int(max_bars)) if max_bars else None

    if kind == "time_only":
        return ExitPlan(time_stop=time_stop, exit_on_opposite_signal=True)

    if kind in ("atr_stop_tp", "atr_trailing_stop"):
        atr = spec.get("atr_at_entry")
        trailing = kind == "atr_trailing_stop"
        mult = spec.get("trail_mult") if trailing else spec.get("stop_mult")
        stop = StopSpec(atr_mult=float(mult) if mult is not None else None,
                        atr_value=float(atr) if atr is not None else None,
                        trailing=trailing)
        legs = ()
        tp_mult = spec.get("tp_mult")
        if tp_mult:
            legs = (TakeProfitLeg(close_pct=1.0, atr_mult=float(tp_mult)),)
        return ExitPlan(stop_loss=stop, take_profit=legs, time_stop=time_stop)

    if kind == "pct_stop_tp":
        stop = None
        if spec.get("stop_pct") is not None:
            stop = StopSpec(pct=float(spec["stop_pct"]))
        elif spec.get("stop_loss_price") is not None:
            stop = StopSpec(price=float(spec["stop_loss_price"]))
        legs: tuple = ()
        if spec.get("tp_pct") is not None:
            legs = (TakeProfitLeg(close_pct=1.0, pct=float(spec["tp_pct"])),)
        elif spec.get("take_profit_price") is not None:
            legs = (TakeProfitLeg(close_pct=1.0, price=float(spec["take_profit_price"])),)
        return ExitPlan(stop_loss=stop, take_profit=legs, time_stop=time_stop)

    if kind == "ma_cross_exit":
        return ExitPlan(time_stop=time_stop, exit_on_opposite_signal=True,
                        note="ma_cross_exit(period=%s, ma_type=%s) is evaluated by the "
                             "engine; it has no ExitPlan representation"
                             % (spec.get("period"), spec.get("ma_type")))

    # opposite_signal, or a type this translation has not been taught
    return ExitPlan(time_stop=time_stop, exit_on_opposite_signal=True)


def _candidates_from_payload(rows: Sequence[Dict[str, Any]]) -> tuple:
    out: List[SelectionCandidate] = []
    for position, row in enumerate(rows, start=1):
        raw = str(row.get("side") or row.get("direction") or "neutral").lower()
        direction = (Direction.LONG if raw == "long"
                     else Direction.SHORT if raw == "short" else Direction.NEUTRAL)
        score = row.get("score")
        out.append(SelectionCandidate(
            symbol=str(row.get("symbol") or row.get("instrument_id") or ""),
            rank=int(row.get("rank") or position),
            score=float(score) if score is not None else 0.0,
            direction=direction,
            reason=str(row.get("reason") or ""),
            features=dict(row.get("features") or {}),
        ))
    return tuple(out)


def envelope_to_signal(
    envelope: SignalEnvelope,
    *,
    decision_time: Optional[int] = None,
    venue: str = "binance",
    product: str = "usd_m_perpetual",
    bot_version: Optional[str] = None,
) -> StandardSignal:
    """Express a block/YAML ``SignalEnvelope`` as ``cyqnt.signal/v2``.

    ``decision_time`` defaults to the envelope's own ``bar_timestamp``; pass it
    explicitly when replaying, since a signal and the input it was computed from
    must agree on the clock.
    """
    payload: Dict[str, Any] = dict(envelope.payload or {})

    # Already a v2 signal riding the envelope (a StandardBot's own output):
    # re-deriving it from the compat keys would be lossy for no reason.
    if payload.get("schema") == StandardSignal.__dataclass_fields__["schema"].default:
        return _signal_from_v2_payload(payload)

    ts = decision_time
    if ts is None:
        ts = payload.get("bar_timestamp") or envelope.valid_until
    if ts is None:
        raise AdapterError(
            "cannot date the signal: the envelope carries no payload "
            "'bar_timestamp' and no 'valid_until'. Pass decision_time="
        )

    provenance = Provenance(
        strategy_id=getattr(envelope.provenance, "plugin_id", "") or envelope.signal_id,
        strategy_version=bot_version or getattr(envelope.provenance, "plugin_version", "v1"),
        snapshot_id=getattr(envelope.provenance, "input_fingerprint", "") or "",
        config_hash=getattr(envelope.provenance, "config_hash", "") or "",
    )

    kind = getattr(envelope.kind, "value", envelope.kind)
    side = getattr(envelope.side, "value", envelope.side)

    if kind == SignalKind.SELECTION.value:
        candidates = _candidates_from_payload(payload.get("candidates") or [])
        if not candidates:
            raise AdapterError(
                "selection envelope %r carries no candidates; a cross-sectional "
                "signal with an empty basket says nothing" % envelope.signal_id)
        return StandardSignal(
            bot_id=provenance.strategy_id, decision_time=int(ts), provenance=provenance,
            signal_id=envelope.signal_id, bot_version=provenance.strategy_version,
            venue=venue, product=product, market_scope=MarketScope.CROSS_SECTION,
            intent=PositionIntent.HOLD, direction=Direction.NEUTRAL,
            candidates=candidates,
            universe_size=int(payload.get("universe_size") or len(candidates)),
            score=min(100.0, max(0.0, float(envelope.strength) * 100.0)),
            confidence=min(1.0, max(0.0, float(envelope.strength))),
            summary=str(payload.get("summary") or ""),
            time_horizon=_horizon(envelope.time_horizon),
            valid_until=envelope.valid_until,
        )

    intent = _intent_from_target(payload.get("target_position"), side)
    exit_plan = _exit_plan_from_spec(payload.get("exit_spec"))
    if intent.is_entry and exit_plan is None:
        # The contract refuses an entry with no stated way out. An envelope
        # without an exit_spec means exactly "exit on the opposite signal",
        # which is a legitimate plan — so say it, rather than fail.
        exit_plan = ExitPlan(exit_on_opposite_signal=True)

    size_value = payload.get("engine_size", payload.get("size"))
    size = None
    if isinstance(size_value, (int, float)):
        size = SizeSpec(mode=SizeMode.EQUITY_PCT, value=float(size_value))

    entry_price = payload.get("entry_price") or payload.get("reference_price")
    entry = (EntrySpec(type=EntryType.LIMIT, price=float(entry_price))
             if isinstance(entry_price, (int, float)) else None)

    return StandardSignal(
        bot_id=provenance.strategy_id, decision_time=int(ts), provenance=provenance,
        signal_id=envelope.signal_id, bot_version=provenance.strategy_version,
        symbol=envelope.instrument_id, venue=venue, product=product,
        intent=intent, entry=entry, exit_plan=exit_plan, size=size,
        score=min(100.0, max(0.0, float(envelope.strength) * 100.0)),
        confidence=min(1.0, max(0.0, float(envelope.strength))),
        time_horizon=_horizon(envelope.time_horizon),
        valid_until=envelope.valid_until,
        summary=str(payload.get("summary") or ""),
    )


def _horizon(value: Any):
    from .core import TimeHorizon

    try:
        return TimeHorizon(str(value))
    except (ValueError, TypeError):
        return TimeHorizon.INTRADAY


def _signal_from_v2_payload(payload: Dict[str, Any]) -> StandardSignal:
    """Rebuild a StandardSignal a StandardBot already serialised into a payload."""
    return StandardSignal.from_dict(payload)


def batch_to_signals(batch: Any, **kwargs: Any) -> List[StandardSignal]:
    """Every envelope in a ``SignalBatch`` (or a bare list) as v2 signals."""
    envelopes = getattr(batch, "signals", batch) or []
    return [envelope_to_signal(item, **kwargs) for item in envelopes]


# --------------------------------------------------------------------------- #
# StandardBot  ->  the registry every entrypoint reads                         #
# --------------------------------------------------------------------------- #


def register_standard_bot(bot: Any, *, strategy_id: Optional[str] = None) -> str:
    """Make a ``StandardBot`` runnable by ``mvp_backtest --strategy <id>``.

    ``StandardBot`` already satisfies the ``SignalPlugin`` protocol — it has
    ``required_inputs`` / ``initialize_state`` / ``run`` / ``step`` and emits
    ``SignalEnvelope`` through its own ``_to_envelope``. What it never had was a
    way into the registry the entrypoints consult, so no CLI could name it.

    This routes it through the same pending-registration mechanism
    ``blocks.strategy.register`` uses, which is what ``build_strategy_pipeline``
    looks up. Returns the id it registered under.

    Paper and live are NOT covered: ``PythonLivePaperSession`` type-checks for a
    ``BlockStrategyPlugin`` before it will run anything. That gate is a separate
    change, and pretending otherwise here would mean a bot that backtests and
    then refuses to trade.
    """
    from cyqnt_trd.blocks import strategy as _blocks_strategy

    plugin_id = strategy_id or getattr(bot, "plugin_id", None) or bot.spec.bot_id
    if not plugin_id:
        raise AdapterError("the bot has no plugin_id / spec.bot_id to register under")

    if not hasattr(bot, "needed_timeframes"):
        raise AdapterError(
            "%r does not implement needed_timeframes(); the registry asks every "
            "plugin which timeframes to assemble" % plugin_id)

    def _factory(raw: Dict[str, Any]) -> Any:
        """Config factory, not a plugin factory — the registry builds the config.

        It must hand back a **mapping**: every shipped bot's ``normalize_config``
        does ``merged.update(config)``, so the ``SimpleNamespace`` the block path
        uses raises ``'X' object is not iterable`` on the first bar. That is why
        the pre-existing ``bind_to_signal_registry`` could not run three of the
        shipped bots.
        """
        config = dict(raw or {})
        instrument = config.get("instrument_id")
        if isinstance(instrument, str):
            config["instrument_id"] = instrument.upper()
        return config

    _blocks_strategy._PENDING_REGISTRATIONS.append((bot, _factory))
    _blocks_strategy._KNOWN_BLOCK_STRATEGY_IDS.add(plugin_id)
    _blocks_strategy._KNOWN_BLOCK_PLUGINS[plugin_id] = bot
    return plugin_id
