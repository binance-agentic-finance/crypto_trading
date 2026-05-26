"""Canonical bootstrap runtime round-trip."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_pro_trading_library.library.core.protocols import StrategySpec
from ai_pro_trading_library.library.monitoring.state import EventRecord, MonitoringState
from ai_pro_trading_library.runtime.backtest import BacktestRunner
from ai_pro_trading_library.runtime.daemon import write_state_json
from ai_pro_trading_library.runtime.paper import PaperSession
from ai_pro_trading_library.runtime.watcher import WatcherSnapshot, project_status


def run_backtest_paper_watcher_round_trip(
    spec: StrategySpec,
    data: pd.DataFrame,
    output_dir: str | Path,
) -> WatcherSnapshot:
    backtest = BacktestRunner().run(spec, data)
    session = PaperSession(run_id=f"{spec.strategy_id}-smoke")
    state = session.start("2026-05-23T00:00:00Z")
    state.equity = 1.0 + backtest.total_return
    state.events.append(
        EventRecord(
            ts="2026-05-23T00:00:01Z",
            level="info",
            message="backtest result projected to paper state",
            payload={"trades": backtest.trades, "total_return": backtest.total_return},
        )
    )
    state = MonitoringState(
        run_id=state.run_id,
        status=state.status,
        trades=state.trades,
        events=state.events,
        equity=state.equity,
    )
    write_state_json(state, Path(output_dir) / "state.json")
    return project_status(state)
