from __future__ import annotations

from unittest.mock import patch

from cyqnt_trd.standard_bot.execution.cli_executor import BinanceCliExecutor


def _pending_transition(target_direction: str = "short") -> dict:
    return {
        "origin_fill_id": "fill-1",
        "source_action": "flip_to_short" if target_direction == "short" else "flip_to_long",
        "target_direction": target_direction,
        "created_at": "2026-06-03T00:00:00+00:00",
        "updated_at": "2026-06-03T00:00:00+00:00",
        "attempt_count": 0,
        "last_attempt_action": None,
        "last_attempt_at": None,
        "last_error": None,
        "last_observed_direction": "flat",
    }


def test_flip_failure_persists_pending_transition(tmp_path) -> None:
    executor = BinanceCliExecutor(symbol="ETHUSDT")
    state = {"schema_version": 1, "pending_transition": None}

    with patch.object(executor, "_get_position_direction", side_effect=["long"]), patch.object(
        executor,
        "_place_single_order",
        return_value={"status": "failed", "error": "Server error: 502"},
    ):
        updated = executor.handle_fill(
            {"fill_id": "fill-1", "action": "flip_to_short"},
            tmp_path / "executions.jsonl",
            state,
        )

    pending = updated["pending_transition"]
    assert pending is not None
    assert pending["target_direction"] == "short"
    assert pending["last_attempt_action"] == "close_long"
    assert pending["attempt_count"] == 1
    assert "502" in pending["last_error"]


def test_pending_transition_reconciles_close_then_open(tmp_path) -> None:
    executor = BinanceCliExecutor(symbol="ETHUSDT")
    state = {"schema_version": 1, "pending_transition": _pending_transition("short")}
    actions: list[str] = []

    def fake_place(action: str, _path):
        actions.append(action)
        return {"status": "NEW", "action": action}

    with patch.object(executor, "_place_single_order", side_effect=fake_place), patch.object(
        executor,
        "_get_position_direction",
        side_effect=["long", "flat", "flat", "short"],
    ), patch(
        "cyqnt_trd.standard_bot.execution.cli_executor.time.sleep",
        return_value=None,
    ):
        updated = executor._advance_pending_transition(
            state, tmp_path / "executions.jsonl"
        )

    assert actions == ["close_long", "open_short"]
    assert updated["pending_transition"] is None


def test_reconcile_uses_open_when_account_already_flat(tmp_path) -> None:
    executor = BinanceCliExecutor(symbol="ETHUSDT")
    state = {"schema_version": 1, "pending_transition": _pending_transition("short")}
    actions: list[str] = []

    def fake_place(action: str, _path):
        actions.append(action)
        return {"status": "NEW", "action": action}

    with patch.object(executor, "_place_single_order", side_effect=fake_place), patch.object(
        executor,
        "_get_position_direction",
        side_effect=["flat", "short"],
    ), patch(
        "cyqnt_trd.standard_bot.execution.cli_executor.time.sleep",
        return_value=None,
    ):
        updated = executor._advance_pending_transition(
            state, tmp_path / "executions.jsonl"
        )

    assert actions == ["open_short"]
    assert updated["pending_transition"] is None
