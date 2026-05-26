"""State projection records written by runtime and consumed by watcher."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TradeRecord:
    ts: str
    symbol: str
    side: str
    quantity: float
    price: float
    pnl: float | None = None


@dataclass(frozen=True)
class EventRecord:
    ts: str
    level: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringState:
    run_id: str
    status: str
    trades: list[TradeRecord] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    equity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "equity": self.equity,
            "trades": [asdict(t) for t in self.trades],
            "events": [asdict(e) for e in self.events],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MonitoringState":
        return cls(
            run_id=payload["run_id"],
            status=payload["status"],
            equity=payload.get("equity"),
            trades=[TradeRecord(**t) for t in payload.get("trades", [])],
            events=[EventRecord(**e) for e in payload.get("events", [])],
        )

    def dump(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "MonitoringState":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
