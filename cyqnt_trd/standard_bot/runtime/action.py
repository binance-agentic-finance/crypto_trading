"""``action.*`` — the DAG-spec action-node surface.

Codegen ends a strategy with ``action.order(signal=trade_signal)``. What that
call must NOT be is a direct exchange write: the standard bot's whole execution
discipline (signal envelope → intent → risk rules → idempotent submit) lives
between a strategy's decision and an order, and a generated script that calls
the exchange itself walks around all of it.

So ``action.order`` emits a :class:`SignalEnvelope` — the same object every
registered strategy produces — and hands it to the configured sink. The default
sink records; only an explicitly installed executor sink can place an order.

    from cyqnt_trd.standard_bot.runtime import action

    action.order(signal=trade_signal, exit_config=exit_config)   # records
    action.alert(signal=advisory_decision)                       # never trades

Order-side contract note: the downstream order payload
(``venue_class / intent_type / idempotency_key / instrument / side / size /
params``) is built by the execution layer from the envelope, not by the
strategy. A strategy supplies *intent* (side, size hint, exit levels); the
executor owns ``idempotency_key`` (``instance:node:event_ref``) because only it
knows the run identity. Never synthesise an idempotency key here — two DAG runs
that computed the same signal must not collide on it by accident, and a replay
of the same run must.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..core import (
    SignalBatch,
    SignalEnvelope,
    SignalKind,
    SignalProvenance,
    TradeSide,
)

__all__ = [
    "ActionSink",
    "RecordingSink",
    "set_sink",
    "get_sink",
    "current_batch",
    "reset",
    "order",
    "alert",
    "close_position",
    "notify",
]

_ACTION_NS = uuid.UUID("c3d81f0a-6b27-5e94-8a15-0d7c9b3e2f48")

_SIDE_MAP = {
    "long": TradeSide.BUY,
    "buy": TradeSide.BUY,
    "short": TradeSide.SELL,
    "sell": TradeSide.SELL,
    "flat": TradeSide.FLAT,
    "close": TradeSide.FLAT,
}


class ActionSink:
    """Where emitted signals go. Subclass to route to a real executor."""

    def emit(self, envelope: SignalEnvelope) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class RecordingSink(ActionSink):
    """Default sink: keep the envelopes, place nothing.

    This is deliberately the default. A generated script that runs with no sink
    configured produces signals and no orders — the safe direction to fail.
    """

    signals: List[SignalEnvelope] = field(default_factory=list)

    def emit(self, envelope: SignalEnvelope) -> None:
        self.signals.append(envelope)


_SINK: ActionSink = RecordingSink()


def set_sink(sink: ActionSink) -> ActionSink:
    """Install the sink for this process. Returns the previous one."""
    global _SINK
    previous = _SINK
    _SINK = sink
    return previous


def get_sink() -> ActionSink:
    return _SINK


def reset() -> None:
    """Restore the default recording sink (tests, REPL)."""
    set_sink(RecordingSink())


def current_batch(batch_id: str = "") -> SignalBatch:
    """Everything the recording sink has collected, as a standard SignalBatch."""
    sink = get_sink()
    signals = list(getattr(sink, "signals", []))
    return SignalBatch(
        signals=signals,
        batch_id=batch_id or str(uuid.uuid5(_ACTION_NS, "batch|%d" % len(signals))),
        created_at=int(time.time() * 1000),
    )


def _config_hash(payload: Dict[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _emit(
    *,
    kind: SignalKind,
    payload: Dict[str, Any],
    instrument_id: Optional[str],
    side: Optional[TradeSide],
    strength: float,
    strategy_id: str,
    strategy_version: str,
    valid_until: Optional[int],
) -> SignalEnvelope:
    signal_id = str(
        uuid.uuid5(
            _ACTION_NS,
            "%s|%s|%s|%s" % (strategy_id, kind.value, instrument_id or "-", _config_hash(payload)),
        )
    )
    envelope = SignalEnvelope(
        version=strategy_version,
        signal_id=signal_id,
        kind=kind,
        strength=strength,
        provenance=SignalProvenance(
            plugin_id=strategy_id,
            plugin_version=strategy_version,
            config_hash=_config_hash(payload),
        ),
        instrument_id=instrument_id,
        side=side,
        valid_until=valid_until,
        payload=payload,
    )
    get_sink().emit(envelope)
    return envelope


def order(
    signal: Optional[Dict[str, Any]],
    *,
    exit_config: Optional[Dict[str, Any]] = None,
    strategy_id: str = "dag_strategy",
    strategy_version: str = "v1",
) -> Optional[SignalEnvelope]:
    """Emit a TRADE signal from a merge node's assembled dict.

    ``signal=None`` is the "gate said no" branch and is a no-op, not an error —
    codegen emits it for the else-branch of a condition node.

    ``exit_config`` is the compiled top-level ``exit:`` block. It travels on the
    envelope payload so paper/live can monitor it; note that on the current
    execution path stops are evaluated in-process, so a stop that must survive
    the daemon dying has to be placed at the exchange by the executor.
    """
    if signal is None:
        return None
    if not isinstance(signal, dict):
        raise TypeError("action.order expects the merge-node dict, got %r" % type(signal))

    symbol = signal.get("symbol") or signal.get("instrument_id")
    raw_side = str(signal.get("side", "")).lower()
    side = _SIDE_MAP.get(raw_side)
    if symbol and side is None:
        raise ValueError(
            "action.order: signal has symbol=%r but side=%r; side must be one of %s"
            % (symbol, signal.get("side"), sorted(_SIDE_MAP))
        )

    payload: Dict[str, Any] = dict(signal)
    if exit_config:
        payload["exit_spec"] = dict(exit_config)
    payload.setdefault("emitted_at", int(time.time() * 1000))

    return _emit(
        kind=SignalKind.TRADE,
        payload=payload,
        instrument_id=str(symbol) if symbol else None,
        side=side,
        strength=float(signal.get("confidence", 1.0) or 1.0),
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        valid_until=signal.get("valid_until"),
    )


def close_position(
    symbol: str,
    *,
    reason: str = "",
    strategy_id: str = "dag_strategy",
    strategy_version: str = "v1",
) -> SignalEnvelope:
    """Emit a flat-target TRADE signal (reduce to zero)."""
    return _emit(
        kind=SignalKind.TRADE,
        payload={"symbol": symbol, "side": "flat", "target_position": 0, "reason": reason},
        instrument_id=symbol,
        side=TradeSide.FLAT,
        strength=1.0,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        valid_until=None,
    )


def alert(
    signal: Dict[str, Any],
    *,
    strategy_id: str = "dag_monitor",
    strategy_version: str = "v1",
) -> SignalEnvelope:
    """Emit an ALERT signal — advisory only, never routable to an order.

    ``auto_trade_eligible`` is forced false and the kind is ALERT, which is not
    in ``SignalBatch.trade_signals()``, so the execution planner cannot build an
    intent from it no matter what the payload says.
    """
    payload = dict(signal)
    payload["auto_trade_eligible"] = False
    return _emit(
        kind=SignalKind.ALERT,
        payload=payload,
        instrument_id=payload.get("symbol"),
        side=_SIDE_MAP.get(str(payload.get("direction", "")).lower(), TradeSide.FLAT),
        strength=float(payload.get("score", 0.0) or 0.0) / 100.0,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        valid_until=payload.get("valid_until"),
    )


def notify(
    message: str,
    *,
    channel: str = "in_app",
    symbol: Optional[str] = None,
    strategy_id: str = "dag_strategy",
    strategy_version: str = "v1",
) -> SignalEnvelope:
    """Emit a NOOP-kind envelope carrying a user-facing message.

    Notification delivery is an external sink; the strategy only states what it
    would say. Codegen compiles an f-string message into this call.
    """
    return _emit(
        kind=SignalKind.NOOP,
        payload={"message": message, "channel": channel, "symbol": symbol},
        instrument_id=symbol,
        side=None,
        strength=0.0,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        valid_until=None,
    )
