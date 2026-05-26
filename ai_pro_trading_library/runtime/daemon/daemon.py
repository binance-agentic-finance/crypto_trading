"""Bounded paper-trading daemon.

Runs `max_iterations` ticks of a `StrategySpec`, persisting state to
`<output_dir>/<run_id>/state.json` and append-only `trades.jsonl`,
`events.jsonl`. The loop is intentionally synchronous so tests are
deterministic; production deployments would wrap this in a scheduler.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ai_pro_trading_library.library.core.protocols import StrategySpec
from ai_pro_trading_library.library.monitoring.state import (
    EventRecord,
    MonitoringState,
    TradeRecord,
)
from ai_pro_trading_library.runtime.backtest import BacktestRunner
from ai_pro_trading_library.runtime.daemon.state import append_jsonl


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _synthetic_bar(seed: int) -> pd.DataFrame:
    base = 100.0 + (seed % 7) * 0.3
    rows = [
        {
            "open": base + i * 0.1,
            "high": base + i * 0.1 + 0.5,
            "low": base + i * 0.1 - 0.5,
            "close": base + i * 0.1 + ((i + seed) % 5 - 2) * 0.05,
            "volume": 1000 + i * 5,
        }
        for i in range(60)
    ]
    return pd.DataFrame(rows)


class PaperDaemon:
    """Synchronous paper daemon that writes state.json + jsonl streams."""

    def __init__(self, output_dir: str | Path, run_id: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id or f"paper-{uuid.uuid4().hex[:8]}"
        self.run_dir = self.output_dir / self.run_id
        self.runner = BacktestRunner()

    def start(self, spec: StrategySpec, max_iterations: int = 1) -> MonitoringState:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        spec.require_symbols()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state = MonitoringState(run_id=self.run_id, status="running", equity=1.0)
        ts = _now_iso()
        start_evt = EventRecord(ts=ts, level="info", message="paper daemon started",
                                payload={"strategy_id": spec.strategy_id})
        state.events.append(start_evt)
        append_jsonl(start_evt, self.run_dir / "events.jsonl")

        for tick in range(max_iterations):
            data = _synthetic_bar(seed=tick)
            result = self.runner.run(spec, data)
            equity = (state.equity or 1.0) * (1.0 + result.total_return)
            tick_evt = EventRecord(
                ts=_now_iso(),
                level="info",
                message=f"tick {tick} backtest complete",
                payload={
                    "trades": result.trades,
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                },
            )
            state.events.append(tick_evt)
            append_jsonl(tick_evt, self.run_dir / "events.jsonl")

            if result.trades > 0:
                trade = TradeRecord(
                    ts=_now_iso(),
                    symbol=spec.symbols[0],
                    side="buy",
                    quantity=1.0,
                    price=float(data["close"].iloc[-1]),
                    pnl=float(result.total_return),
                )
                state.trades.append(trade)
                append_jsonl(trade, self.run_dir / "trades.jsonl")

            state.equity = float(equity)

        state.status = "stopped"
        state.events.append(
            EventRecord(ts=_now_iso(), level="info", message="paper daemon stopped")
        )
        state.dump(self.run_dir / "state.json")
        return state
