"""Paper session state container."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_pro_trading_library.library.monitoring.state import EventRecord, MonitoringState


@dataclass
class PaperSession:
    run_id: str
    status: str = "created"
    events: list[EventRecord] = field(default_factory=list)

    def start(self, ts: str) -> MonitoringState:
        self.status = "running"
        self.events.append(EventRecord(ts=ts, level="info", message="paper session started"))
        return MonitoringState(run_id=self.run_id, status=self.status, events=list(self.events))

    def stop(self, ts: str) -> MonitoringState:
        self.status = "stopped"
        self.events.append(EventRecord(ts=ts, level="info", message="paper session stopped"))
        return MonitoringState(run_id=self.run_id, status=self.status, events=list(self.events))

