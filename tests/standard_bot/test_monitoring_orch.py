"""Tests for cyqnt_trd.standard_bot.monitoring and .orchestration.

Covers every public function with at least one focused unit test.
No network access is required — all external calls are mocked.

Run with::

    pytest tests/standard_bot/test_monitoring_orch.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# monitoring.pnl
# ---------------------------------------------------------------------------
from cyqnt_trd.standard_bot.monitoring.pnl import (
    portfolio_pnl_aggregate,
    unrealized_pnl_compute,
)

# ---------------------------------------------------------------------------
# monitoring.signals
# ---------------------------------------------------------------------------
from cyqnt_trd.standard_bot.monitoring.signals import (
    performance_report,
    signal_outcome_check,
    signal_record,
)

# ---------------------------------------------------------------------------
# monitoring.stops
# ---------------------------------------------------------------------------
from cyqnt_trd.standard_bot.monitoring.stops import (
    STOP_TYPES,
    query_open_orders,
    stale_stop_detect,
)

# ---------------------------------------------------------------------------
# orchestration.pipeline
# ---------------------------------------------------------------------------
from cyqnt_trd.standard_bot.orchestration.pipeline import (
    Verdict,
    candidate_filter_rank,
    scan_rank_select,
    sequential_workflow,
    two_stage_pipeline,
)

# ---------------------------------------------------------------------------
# orchestration.scheduler
# ---------------------------------------------------------------------------
from cyqnt_trd.standard_bot.orchestration.scheduler import (
    fixed_interval_loop,
    rescan_loop,
    rescan_pass,
)

# ---------------------------------------------------------------------------
# orchestration.state
# ---------------------------------------------------------------------------
from cyqnt_trd.standard_bot.orchestration.state import (
    position_close,
    position_create,
    position_list,
    position_open_list,
    position_update,
    signal_log_append,
    signal_log_load,
    signal_log_save,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


# ======================================================================
# monitoring.pnl
# ======================================================================

class TestUnrealizedPnlCompute:
    def test_long_in_profit(self):
        r = unrealized_pnl_compute(100.0, 110.0, "LONG", leverage=2, notional=1000.0)
        assert r["pnl_pct"] == 10.0
        assert r["pnl_leveraged_pct"] == 20.0
        assert r["pnl_abs"] == 100.0
        assert r["in_profit"] is True

    def test_short_in_profit(self):
        r = unrealized_pnl_compute(100.0, 90.0, "SHORT", leverage=1, notional=500.0)
        assert r["pnl_pct"] == 10.0
        assert r["in_profit"] is True

    def test_long_at_loss(self):
        r = unrealized_pnl_compute(100.0, 80.0, "LONG")
        assert r["pnl_pct"] == -20.0
        assert r["in_profit"] is False

    def test_zero_entry_price(self):
        r = unrealized_pnl_compute(0.0, 100.0, "LONG")
        assert r["pnl_pct"] == 0
        assert r["in_profit"] is False

    def test_case_insensitive_direction(self):
        r = unrealized_pnl_compute(100.0, 110.0, "long")
        assert r["in_profit"] is True


class TestPortfolioPnlAggregate:
    def test_empty_positions(self):
        r = portfolio_pnl_aggregate([])
        assert r["positions_count"] == 0
        assert r["best"] is None
        assert r["worst"] is None

    def test_single_position(self):
        positions = [
            {
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "entry_price": 40_000.0,
                "current_price": 44_000.0,
                "leverage": 5,
                "notional": 1_000.0,
            }
        ]
        r = portfolio_pnl_aggregate(positions)
        assert r["positions_count"] == 1
        assert r["total_notional"] == 1_000.0
        assert r["best"]["symbol"] == "BTCUSDT"
        assert r["worst"]["symbol"] == "BTCUSDT"

    def test_best_and_worst(self):
        positions = [
            {"symbol": "A", "direction": "LONG", "entry_price": 100, "current_price": 120, "notional": 100},
            {"symbol": "B", "direction": "LONG", "entry_price": 100, "current_price": 80,  "notional": 100},
        ]
        r = portfolio_pnl_aggregate(positions)
        assert r["best"]["symbol"] == "A"
        assert r["worst"]["symbol"] == "B"


# ======================================================================
# monitoring.signals
# ======================================================================

class TestSignalRecord:
    def test_creates_jsonl_file(self, tmp_dir: Path):
        out = str(tmp_dir / "signals.jsonl")
        sig = signal_record(
            symbol="BTCUSDT",
            direction="LONG",
            verdict="STRONG_CANDIDATE",
            score=4.5,
            entry_price=50_000.0,
            signals_file=out,
        )
        assert sig["symbol"] == "BTCUSDT"
        assert Path(out).exists()

    def test_appends_multiple(self, tmp_dir: Path):
        out = str(tmp_dir / "signals.jsonl")
        for _ in range(3):
            signal_record("ETHUSDT", "SHORT", "CANDIDATE", 2.0, 3000.0, signals_file=out)
        lines = Path(out).read_text().strip().splitlines()
        assert len(lines) == 3

    def test_metadata_included(self, tmp_dir: Path):
        out = str(tmp_dir / "signals.jsonl")
        sig = signal_record(
            "X", "LONG", "SKIP", 0.0, 1.0,
            metadata={"strategy": "rsi_bb"},
            signals_file=out,
        )
        assert sig["metadata"]["strategy"] == "rsi_bb"


class TestSignalOutcomeCheck:
    def test_long_profit(self):
        r = signal_outcome_check(100.0, 110.0, "LONG", leverage=2.0, stop_price=95.0)
        assert r["pnl_pct"] == 10.0
        assert r["pnl_leveraged_pct"] == 20.0
        assert r["stop_hit"] is False

    def test_long_stop_hit(self):
        r = signal_outcome_check(100.0, 94.0, "LONG", stop_price=95.0)
        assert r["stop_hit"] is True

    def test_short_profit(self):
        r = signal_outcome_check(100.0, 90.0, "SHORT")
        assert r["pnl_pct"] == 10.0

    def test_invalid_prices(self):
        r = signal_outcome_check(0.0, 100.0, "LONG")
        assert r["pnl_pct"] is None

    def test_no_stop_tracked(self):
        r = signal_outcome_check(100.0, 110.0, "LONG")
        assert r["stop_hit"] is None


class TestPerformanceReport:
    def _make_signal(self, direction: str, pnl: float) -> dict:
        return {
            "direction": direction,
            "verdict": "CANDIDATE",
            "score": 3.0,
            "outcomes": {"24h": {"pnl_pct": pnl}},
        }

    def test_empty(self):
        r = performance_report([])
        assert r == {"total": 0, "evaluated": 0}

    def test_all_wins(self):
        signals = [self._make_signal("LONG", 5.0) for _ in range(4)]
        r = performance_report(signals)
        assert r["win_rate"] == 100.0
        assert r["wins"] == 4
        assert r["losses"] == 0

    def test_mixed(self):
        signals = [
            self._make_signal("LONG", 5.0),
            self._make_signal("LONG", -3.0),
            self._make_signal("SHORT", 2.0),
        ]
        r = performance_report(signals)
        assert r["evaluated"] == 3
        assert r["wins"] == 2
        assert r["losses"] == 1
        assert abs(r["win_rate"] - 66.7) < 0.1

    def test_no_outcomes(self):
        signals = [{"direction": "LONG", "verdict": "SKIP", "score": 0.0, "outcomes": {}}]
        r = performance_report(signals)
        assert r["evaluated"] == 0


# ======================================================================
# monitoring.stops
# ======================================================================

class TestStaleStopDetect:
    def test_no_stops(self):
        orders = [{"type": "LIMIT", "orderId": 1}]
        r = stale_stop_detect(orders)
        assert r["has_stop"] is False
        assert r["stop_count"] == 0

    def test_with_stop(self):
        orders = [
            {"type": "STOP_MARKET", "orderId": 42, "stopPrice": "41000", "side": "SELL"},
            {"type": "LIMIT", "orderId": 43},
        ]
        r = stale_stop_detect(orders)
        assert r["has_stop"] is True
        assert r["stop_count"] == 1
        assert r["stop_orders"][0]["orderId"] == 42

    def test_all_stop_types_recognised(self):
        for stype in STOP_TYPES:
            orders = [{"type": stype}]
            r = stale_stop_detect(orders)
            assert r["has_stop"] is True, f"{stype} not recognized"


class TestQueryOpenOrders:
    """query_open_orders must not raise; returns [] on CLI failure."""

    def test_returns_list_on_missing_binary(self):
        result = query_open_orders("BTCUSDT", binary="nonexistent-cli-binary-xyz")
        assert isinstance(result, list)

    def test_mock_cli_returns_list(self):
        fake_orders = [{"orderId": 1, "type": "LIMIT", "symbol": "BTCUSDT"}]
        mock_run = _make_subprocess_mock(json.dumps(fake_orders))
        with patch("subprocess.run", mock_run):
            result = query_open_orders("BTCUSDT")
        assert result == fake_orders

    def test_mock_cli_error_returns_empty(self):
        mock_run = _make_subprocess_mock("{}", returncode=1)
        with patch("subprocess.run", mock_run):
            result = query_open_orders("BTCUSDT")
        assert result == []


def _make_subprocess_mock(stdout: str, returncode: int = 0):
    """Return a mock for subprocess.run with the given stdout."""
    from unittest.mock import MagicMock

    def _mock_run(cmd, **kwargs):
        m = MagicMock()
        m.returncode = returncode
        m.stdout = stdout
        m.stderr = ""
        return m

    return _mock_run


# ======================================================================
# orchestration.pipeline
# ======================================================================

class TestVerdict:
    def test_constants(self):
        assert Verdict.STRONG_CANDIDATE == "STRONG_CANDIDATE"
        assert Verdict.RANK["STRONG_CANDIDATE"] < Verdict.RANK["CANDIDATE"]
        assert Verdict.RANK["CANDIDATE"] < Verdict.RANK["SKIP"]


class TestSequentialWorkflow:
    def test_simple_chain(self):
        def step_a(d):
            d["a"] = 1
            return d

        def step_b(d):
            d["b"] = d["a"] + 1
            return d

        result = sequential_workflow([step_a, step_b], {})
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["_completed"] is True

    def test_stops_on_error(self):
        def step_ok(d):
            d["ok"] = True
            return d

        def step_fail(d):
            raise ValueError("oops")

        def step_never(d):
            d["never"] = True
            return d

        result = sequential_workflow([step_ok, step_fail, step_never], {})
        assert result["ok"] is True
        assert "never" not in result
        assert result["_completed"] is False
        assert result["_errors"][0]["step"] == "step_fail"

    def test_initial_data_preserved(self):
        result = sequential_workflow([], {"seed": 42})
        assert result["seed"] == 42
        assert result["_completed"] is True


class TestTwoStagePipeline:
    def test_basic(self):
        scanned = []

        def scan():
            return ["BTCUSDT", "ETHUSDT"]

        def validate(sym):
            scanned.append(sym)
            return {"symbol": sym, "verdict": "CANDIDATE"}

        results = two_stage_pipeline(scan, validate)
        assert len(results) == 2
        assert scanned == ["BTCUSDT", "ETHUSDT"]

    def test_max_candidates_capped(self):
        def scan():
            return [f"SYM{i}" for i in range(20)]

        def validate(sym):
            return {"symbol": sym}

        results = two_stage_pipeline(scan, validate, max_candidates=5)
        assert len(results) == 5

    def test_dict_candidates(self):
        def scan():
            return [{"symbol": "BTCUSDT"}, {"symbol": "ETHUSDT"}]

        def validate(sym):
            return {"symbol": sym, "ok": True}

        results = two_stage_pipeline(scan, validate)
        assert results[0]["symbol"] == "BTCUSDT"


class TestScanRankSelect:
    def _make_candidates(self):
        return [
            {"symbol": "A", "verdict": "STRONG_CANDIDATE", "direction": "LONG", "score": 5.0},
            {"symbol": "B", "verdict": "CANDIDATE",        "direction": "LONG", "score": 3.5},
            {"symbol": "C", "verdict": "WATCHLIST",        "direction": "LONG", "score": 2.0},
            {"symbol": "D", "verdict": "SKIP",             "direction": "LONG", "score": 1.0},
            {"symbol": "E", "verdict": "CANDIDATE",        "direction": "NO_TRADE", "score": 4.0},
        ]

    def test_filters_by_min_verdict(self):
        results = scan_rank_select(self._make_candidates(), min_verdict="CANDIDATE", max_select=10)
        symbols = [r["symbol"] for r in results]
        assert "D" not in symbols  # SKIP
        assert "C" not in symbols  # WATCHLIST worse than CANDIDATE
        assert "E" not in symbols  # NO_TRADE direction

    def test_sorted_by_score(self):
        results = scan_rank_select(self._make_candidates(), max_select=2)
        assert results[0]["symbol"] == "A"
        assert results[1]["symbol"] == "B"

    def test_max_select(self):
        results = scan_rank_select(self._make_candidates(), max_select=1)
        assert len(results) == 1

    def test_empty(self):
        assert scan_rank_select([]) == []


class TestCandidateFilterRank:
    def test_filter_and_sort(self):
        candidates = [
            {"score": 3.0, "vol": 1000},
            {"score": 5.0, "vol": 500},
            {"score": 2.0, "vol": 800},
        ]
        result = candidate_filter_rank(
            candidates,
            filters=[lambda c: c["vol"] >= 800],
            sort_key="score",
        )
        assert len(result) == 2
        assert result[0]["score"] == 3.0  # sorted descending

    def test_no_filters(self):
        candidates = [{"score": i} for i in range(5)]
        result = candidate_filter_rank(candidates)
        assert result[0]["score"] == 4

    def test_limit(self):
        candidates = [{"score": i} for i in range(10)]
        result = candidate_filter_rank(candidates, limit=3)
        assert len(result) == 3

    def test_empty(self):
        assert candidate_filter_rank([]) == []


# ======================================================================
# orchestration.scheduler
# ======================================================================

class TestFixedIntervalLoop:
    def test_runs_n_times(self):
        calls: list[int] = []

        def fn():
            calls.append(1)

        with patch("time.sleep"):
            count = fixed_interval_loop(fn, interval_seconds=1, max_iterations=5)

        assert count == 5
        assert len(calls) == 5

    def test_keyboard_interrupt(self):
        call_count = [0]

        def fn():
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt

        with patch("time.sleep"):
            count = fixed_interval_loop(fn, interval_seconds=0)

        assert count == 1  # second call raises, loop exits after 1 completed


class TestRescanPass:
    def test_basic_pass(self):
        positions = [
            {"symbol": "BTCUSDT", "direction": "LONG", "entry_price": 40_000.0},
        ]

        def price_fetcher(sym):
            return 42_000.0

        def trend_classifier(sym):
            return "BULLISH"

        def decision_fn(pos, price, trend):
            return {"action": "HOLD", "reason": "in profit"}

        results = rescan_pass(positions, price_fetcher, trend_classifier, decision_fn)
        assert len(results) == 1
        assert results[0]["symbol"] == "BTCUSDT"
        assert results[0]["current_price"] == 42_000.0
        assert results[0]["trend"] == "BULLISH"

    def test_skips_on_bad_price(self):
        positions = [{"symbol": "BTCUSDT"}]

        results = rescan_pass(
            positions,
            price_fetcher=lambda s: 0.0,
            trend_classifier=lambda s: "MIXED",
            decision_fn=lambda p, pr, t: {"action": "HOLD"},
        )
        assert results[0]["action"] == "SKIP"

    def test_action_fn_called(self):
        acted: list[str] = []
        positions = [{"symbol": "SOLUSDT", "direction": "LONG", "entry_price": 100.0}]

        def price_fetcher(sym):
            return 90.0

        def trend_classifier(sym):
            return "BEARISH"

        def decision_fn(pos, price, trend):
            return {"action": "CLOSE", "reason": "stop hit"}

        def action_fn(pos, decision, price):
            acted.append(pos["symbol"])
            return {"closed": True}

        results = rescan_pass(
            positions, price_fetcher, trend_classifier, decision_fn, action_fn
        )
        assert "SOLUSDT" in acted
        assert results[0]["action_result"]["closed"] is True


class TestRescanLoop:
    def test_runs_n_times(self):
        calls: list[int] = []

        def pass_fn():
            calls.append(1)
            return []

        with patch("time.sleep"):
            count = rescan_loop(pass_fn, interval_seconds=0, max_iterations=3)

        assert count == 3
        assert len(calls) == 3

    def test_on_complete_called(self):
        received: list[list] = []

        def pass_fn():
            return [{"symbol": "X"}]

        def on_complete(decisions):
            received.append(decisions)

        with patch("time.sleep"):
            rescan_loop(pass_fn, max_iterations=2, on_complete=on_complete)

        assert len(received) == 2
        assert received[0][0]["symbol"] == "X"


# ======================================================================
# orchestration.state
# ======================================================================

class TestPositionCRUD:
    def _positions_file(self, tmp_dir: Path) -> str:
        return str(tmp_dir / "positions.json")

    def test_create_and_list(self, tmp_dir):
        pf = self._positions_file(tmp_dir)
        pos = position_create(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=40_000.0,
            stop_price=38_000.0,
            stop_pct=5.0,
            leverage=5,
            notional=1_000.0,
            max_loss=50.0,
            verdict="CANDIDATE",
            score=3.5,
            positions_file=pf,
        )
        assert pos["symbol"] == "BTCUSDT"
        assert pos["status"] == "open"

        all_pos = position_list(pf)
        assert len(all_pos) == 1
        assert all_pos[0]["id"] == pos["id"]

    def test_update(self, tmp_dir):
        pf = self._positions_file(tmp_dir)
        pos = position_create("ETHUSDT", "SHORT", 3_000.0, 3_100.0, 3.3, 3, 500.0, 15.0, positions_file=pf)
        ok = position_update(pos["id"], {"peak_profit_pct": 5.5}, pf)
        assert ok is True
        updated = position_list(pf)[0]
        assert updated["peak_profit_pct"] == 5.5

    def test_update_unknown_id(self, tmp_dir):
        pf = self._positions_file(tmp_dir)
        assert position_update("nonexistent", {"x": 1}, pf) is False

    def test_close(self, tmp_dir):
        pf = self._positions_file(tmp_dir)
        pos = position_create("SOLUSDT", "LONG", 150.0, 140.0, 6.7, 10, 300.0, 20.0, positions_file=pf)
        ok = position_close(pos["id"], pf)
        assert ok is True
        closed = position_list(pf)[0]
        assert closed["status"] == "closed"
        assert "closed_at" in closed

    def test_open_list_filters_closed(self, tmp_dir):
        pf = self._positions_file(tmp_dir)
        pos1 = position_create("A", "LONG", 100.0, 95.0, 5.0, 1, 100.0, 5.0, positions_file=pf)
        pos2 = position_create("B", "SHORT", 200.0, 210.0, 5.0, 1, 200.0, 10.0, positions_file=pf)
        position_close(pos1["id"], pf)

        open_pos = position_open_list(pf)
        assert len(open_pos) == 1
        assert open_pos[0]["symbol"] == "B"

    def test_empty_file_returns_empty_list(self, tmp_dir):
        pf = self._positions_file(tmp_dir)
        assert position_list(pf) == []


class TestSignalLogCRUD:
    def _signals_file(self, tmp_dir: Path) -> str:
        return str(tmp_dir / "signals.jsonl")

    def test_append_and_load(self, tmp_dir):
        sf = self._signals_file(tmp_dir)
        sig = {"symbol": "BTCUSDT", "verdict": "CANDIDATE", "outcomes": {}}
        signal_log_append(sig, sf)
        loaded = signal_log_load(sf)
        assert len(loaded) == 1
        assert loaded[0]["symbol"] == "BTCUSDT"

    def test_save_overwrites(self, tmp_dir):
        sf = self._signals_file(tmp_dir)
        # Write 3 signals
        for i in range(3):
            signal_log_append({"id": i}, sf)
        # Save only 1
        signal_log_save([{"id": 99}], sf)
        loaded = signal_log_load(sf)
        assert len(loaded) == 1
        assert loaded[0]["id"] == 99

    def test_load_missing_file(self, tmp_dir):
        sf = self._signals_file(tmp_dir)
        assert signal_log_load(sf) == []

    def test_atomic_write_survives_crash_simulation(self, tmp_dir):
        """Verify that after a normal save the tmp file is cleaned up."""
        sf = self._signals_file(tmp_dir)
        signal_log_save([{"x": 1}], sf)
        tmp_file = Path(sf + ".tmp")
        assert not tmp_file.exists(), ".tmp file should be removed after atomic write"


# ======================================================================
# Import smoke test
# ======================================================================

class TestModuleImports:
    def test_monitoring_package_importable(self):
        from cyqnt_trd.standard_bot import monitoring  # noqa: F401

    def test_orchestration_package_importable(self):
        from cyqnt_trd.standard_bot import orchestration  # noqa: F401

    def test_monitoring_all_exports(self):
        import cyqnt_trd.standard_bot.monitoring as m
        for name in m.__all__:
            assert hasattr(m, name), f"missing export: {name}"

    def test_orchestration_all_exports(self):
        import cyqnt_trd.standard_bot.orchestration as o
        for name in o.__all__:
            assert hasattr(o, name), f"missing export: {name}"
