"""Runtime state file protocol."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ai_pro_trading_library.library.monitoring.state import EventRecord, MonitoringState, TradeRecord


def write_state_json(state: MonitoringState, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(state), indent=2, sort_keys=True), encoding="utf-8")
    return target


def read_state_json(path: str | Path) -> MonitoringState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MonitoringState(
        run_id=payload["run_id"],
        status=payload["status"],
        trades=[TradeRecord(**trade) for trade in payload.get("trades", [])],
        events=[EventRecord(**event) for event in payload.get("events", [])],
        equity=payload.get("equity"),
    )


def append_jsonl(record: EventRecord | TradeRecord, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return target
