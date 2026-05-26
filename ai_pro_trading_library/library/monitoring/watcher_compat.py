"""§E5 watcher_template payload compatibility.

Per `docs/plans/acceptance-plan.md` §E5: the new repo's
`runtime.daemon.PaperDaemon` writes `state.json` with our canonical
`MonitoringState` shape (run_id / status / equity / trades / events).
The legacy `standard_bot.entrypoints.mvp_monitor_http.summary_to_payload`
expects a different shape (run_id / status / signal_count / errors /
signals / execution_reports).

Both encode the same conceptual "what is the daemon doing right now?"
just at different layers of detail. This module bridges:

- `monitoring_state_to_watcher_payload(MonitoringState)` →
  watcher-compatible dict.
- `WATCHER_REQUIRED_KEYS` — strict key set of the legacy payload.
- `validate_watcher_payload(dict)` — schema validation helper.

Use these when an old `mvp_monitor_http` consumer needs to read our
state.json. New consumers should target `MonitoringState.to_dict()` or
`runtime.watcher.project_from_path` directly.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ai_pro_trading_library.library.monitoring.state import MonitoringState


# Keys required by `standard_bot.entrypoints.mvp_monitor_http.summary_to_payload`
WATCHER_REQUIRED_KEYS: tuple[str, ...] = (
    "run_id",
    "status",
    "signal_count",
    "errors",
    "signals",
    "execution_reports",
)

# Optional keys our daemon adds for richer monitoring (not required by
# the legacy watcher but harmless extras).
WATCHER_OPTIONAL_KEYS: tuple[str, ...] = (
    "equity",
    "trade_count",
    "event_count",
)


def monitoring_state_to_watcher_payload(state: MonitoringState) -> dict[str, Any]:
    """Convert canonical `MonitoringState` to the legacy watcher dict shape.

    Mappings:
    - `state.run_id` → payload["run_id"]
    - `state.status` → payload["status"]
    - `state.events` (level=="error") → payload["errors"]  (list[str])
    - `state.events` (level!="error") → payload["signals"] (one per event,
       shaped to the legacy SignalEnvelope projection)
    - `state.trades` → payload["execution_reports"] (one per trade)
    - `len(payload["signals"])` → payload["signal_count"]

    Optional keys (`equity`, `trade_count`, `event_count`) are added so
    callers wanting richer info don't need a second fetch.
    """
    errors = [e.message for e in state.events if e.level.lower() == "error"]
    signals = [
        {
            "signal_id": e.payload.get("signal_id", f"evt-{i}"),
            "kind": e.payload.get("kind", "info"),
            "instrument_id": e.payload.get("instrument_id"),
            "side": e.payload.get("side"),
            "strength": e.payload.get("strength", 0.0),
            "time_horizon": e.payload.get("time_horizon"),
            "valid_until": e.payload.get("valid_until"),
            "payload": {**e.payload, "ts": e.ts, "message": e.message},
        }
        for i, e in enumerate(state.events)
        if e.level.lower() != "error"
    ]
    execution_reports = [
        {
            "intent_id": f"trade-{i}",
            "status": "filled",
            "reason": None,
            "external_ids": {"order_id": str(i)},
            "fills": [asdict(t)],
        }
        for i, t in enumerate(state.trades)
    ]
    return {
        "run_id": state.run_id,
        "status": state.status,
        "signal_count": len(signals),
        "errors": errors,
        "signals": signals,
        "execution_reports": execution_reports,
        # Optional extras
        "equity": state.equity,
        "trade_count": len(state.trades),
        "event_count": len(state.events),
    }


def validate_watcher_payload(payload: dict[str, Any]) -> None:
    """Raise ValueError if `payload` is missing any WATCHER_REQUIRED_KEYS."""
    missing = [k for k in WATCHER_REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(
            f"watcher payload missing required keys: {missing}; "
            f"present keys: {sorted(payload)}"
        )
    if not isinstance(payload["signals"], list):
        raise ValueError("watcher payload 'signals' must be a list")
    if not isinstance(payload["execution_reports"], list):
        raise ValueError("watcher payload 'execution_reports' must be a list")
    if not isinstance(payload["errors"], list):
        raise ValueError("watcher payload 'errors' must be a list")


__all__ = [
    "WATCHER_OPTIONAL_KEYS",
    "WATCHER_REQUIRED_KEYS",
    "monitoring_state_to_watcher_payload",
    "validate_watcher_payload",
]
