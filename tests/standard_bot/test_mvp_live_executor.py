from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyqnt_trd.standard_bot.entrypoints.mvp_live_executor import (
    _executor_state_path,
    _format_pending_transition,
    _reconcile_only,
    _run_preflight,
)


class _FakeExecutor:
    def __init__(
        self,
        *,
        balance: float = 5000.0,
        price: float = 2000.0,
        step_size: float = 0.001,
        actual_direction: str = "flat",
        max_notional: float = 200.0,
        notional_fraction: float = 0.95,
    ) -> None:
        self.balance = balance
        self.price = price
        self.step_size = step_size
        self.actual_direction = actual_direction
        self.max_notional = max_notional
        self.notional_fraction = notional_fraction
        self.symbol = "ETHUSDT"

    def _get_usdt_balance(self) -> float:
        return self.balance

    def _get_current_price(self) -> float:
        return self.price

    def _get_step_size(self) -> float:
        return self.step_size

    def _round_step(self, qty: float, step: float) -> float:
        if step <= 0:
            return qty
        return int(qty / step) * step

    def _get_position_direction(self) -> str:
        return self.actual_direction

    def _load_executor_state(self, path: Path) -> dict:
        if not path.exists():
            return {"schema_version": 1, "pending_transition": None}
        return json.loads(path.read_text())

    def _save_executor_state(self, path: Path, state: dict) -> None:
        path.write_text(json.dumps(state))

    def _has_pending_transition(self, state: dict) -> bool:
        return bool(state.get("pending_transition"))

    def _advance_pending_transition(self, state: dict, _executions_path: Path) -> dict:
        state["pending_transition"] = None
        return state


def _prepare_state_dir(tmp_path: Path) -> Path:
    state_dir = tmp_path / "watcher" / "RUN"
    state_dir.mkdir(parents=True)
    (state_dir / "trades.jsonl").write_text("")
    (state_dir / "state.json").write_text(json.dumps({"status": "running"}))
    return state_dir


def test_run_preflight_requires_trades_jsonl(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "watcher" / "RUN"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(json.dumps({"status": "running"}))
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/binance-cli")

    with pytest.raises(FileNotFoundError):
        _run_preflight(state_dir, _FakeExecutor())


def test_run_preflight_rejects_qty_rounds_to_zero(tmp_path, monkeypatch) -> None:
    state_dir = _prepare_state_dir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/binance-cli")
    executor = _FakeExecutor(
        balance=100.0,
        price=1_000_000.0,
        step_size=0.1,
        max_notional=10.0,
    )

    with pytest.raises(RuntimeError, match="rounds to zero quantity"):
        _run_preflight(state_dir, executor)


def test_reconcile_only_returns_zero_when_nothing_pending(tmp_path) -> None:
    state_dir = _prepare_state_dir(tmp_path)
    executor = _FakeExecutor()
    _executor_state_path(state_dir).write_text(
        json.dumps({"schema_version": 1, "pending_transition": None})
    )

    assert _reconcile_only(state_dir, executor, max_cycles=3, sleep_sec=0) == 0


def test_reconcile_only_clears_pending_transition(tmp_path) -> None:
    state_dir = _prepare_state_dir(tmp_path)
    executor = _FakeExecutor()
    _executor_state_path(state_dir).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pending_transition": {
                    "target_direction": "short",
                    "source_action": "flip_to_short",
                    "attempt_count": 1,
                    "last_attempt_action": "close_long",
                    "last_error": "Server error: 502",
                },
            }
        )
    )

    assert _reconcile_only(state_dir, executor, max_cycles=3, sleep_sec=0) == 0
    saved = json.loads(_executor_state_path(state_dir).read_text())
    assert saved["pending_transition"] is None


def test_format_pending_transition_is_human_readable() -> None:
    text = _format_pending_transition(
        {
            "target_direction": "short",
            "source_action": "flip_to_short",
            "attempt_count": 2,
            "last_attempt_action": "close_long",
            "last_error": "Server error: 502",
        }
    )
    assert "target=short" in text
    assert "last_action=close_long" in text
    assert "502" in text
