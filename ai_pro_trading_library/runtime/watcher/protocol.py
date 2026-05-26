"""Watcher-facing state projection — in-memory and disk-backed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_pro_trading_library.library.monitoring.state import MonitoringState


@dataclass(frozen=True)
class WatcherSnapshot:
    run_id: str
    status: str
    trade_count: int
    event_count: int
    equity: float | None


def project_status(state: MonitoringState) -> WatcherSnapshot:
    return WatcherSnapshot(
        run_id=state.run_id,
        status=state.status,
        trade_count=len(state.trades),
        event_count=len(state.events),
        equity=state.equity,
    )


def project_from_path(path: str | Path) -> WatcherSnapshot:
    """Read a state.json file and project to a WatcherSnapshot."""
    return project_status(MonitoringState.load(path))
