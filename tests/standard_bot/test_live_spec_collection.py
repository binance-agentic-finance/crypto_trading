"""Network-free regressions for spec-aware multi-pass live collection."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import (
    BundleRunError,
    collect_live_bundle_for_spec,
    run_bundle,
)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import load_spec, validate_spec


ROOT = Path(__file__).parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "universe_derivatives.json"
SPEC_OI = ROOT / "docs" / "strategy_yaml_spec" / "example_open_interest_screen.yaml"
SPEC_RUNUP = (
    ROOT / "docs" / "strategy_yaml_spec" / "example_three_month_runup_screen.yaml"
)


def _fixture_collector(monkeypatch):
    """Serve only the requested fixture frames, one deterministic pass at a time."""
    from cyqnt_trd.standard_bot.data import live_snapshot

    frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
    calls = []

    def fake_build_live_snapshot(**kwargs):
        requests = list(kwargs.get("requests") or ())
        assert requests, "the spec collector must make every pass explicit"
        calls.append(copy.deepcopy(kwargs))
        number = len(calls)
        keys = [key for _node, _params, key in requests]
        bundle = {
            "schema": "cyqnt.input/v1",
            "snapshot_id": "fixture-pass-%d" % number,
            "decision_time": int(frozen["decision_time"]) + number,
            "decision_time_basis": "collection_complete",
            "market_type": frozen["market_type"],
            "instruments": list(frozen["instruments"]),
            "primary_timeframe": frozen["primary_timeframe"],
            "frames": {
                key: copy.deepcopy(frozen["frames"][key]) for key in keys
            },
            "source_status": {
                key: frozen["source_status"][key] for key in keys
            },
            "warnings": ["fixture pass %d: %s" % (number, ",".join(keys))],
            "positions": {},
            "equity": None,
        }
        return None, bundle

    monkeypatch.setattr(
        live_snapshot, "build_live_snapshot", fake_build_live_snapshot
    )
    return frozen, calls


def _request(calls, key):
    return next(
        request
        for call in calls
        for request in call["requests"]
        if request[2] == key
    )


def test_open_interest_fan_out_uses_each_specs_own_surviving_prefix(monkeypatch):
    frozen, calls = _fixture_collector(monkeypatch)
    spec = load_spec(str(SPEC_OI))

    bundle = collect_live_bundle_for_spec(
        spec, run_id="run-oi-live", trace_id="trace-oi-live"
    )

    assert [[request[2] for request in call["requests"]] for call in calls] == [
        ["universe", "contract_meta"],
        ["open_interest_snapshot"],
        ["oi_change_snapshot"],
    ]
    # The free exchange-info filters leave 127 symbols. The OI floor then leaves
    # 41, so the second fan-out is not charged for the 86 rows already rejected.
    assert len(_request(calls, "open_interest_snapshot")[1]["symbols"]) == 127
    assert len(_request(calls, "oi_change_snapshot")[1]["symbols"]) == 41
    assert set(bundle["frames"]) == {
        "universe", "contract_meta", "open_interest_snapshot",
        "oi_change_snapshot",
    }
    assert set(bundle["source_status"]) == set(bundle["frames"])
    assert bundle["decision_time"] == frozen["decision_time"] + 3
    assert bundle["decision_time"] == max(
        item["decision_time"] for item in bundle["collection_passes"]
    )
    assert bundle["decision_time_basis"] == "collection_complete"
    assert (bundle["run_id"], bundle["trace_id"]) == (
        "run-oi-live", "trace-oi-live"
    )
    assert bundle["snapshot_id"] not in {
        item["snapshot_id"] for item in bundle["collection_passes"]
    }
    assert bundle["warnings"] == [
        "fixture pass 1: universe,contract_meta",
        "fixture pass 2: open_interest_snapshot",
        "fixture pass 3: oi_change_snapshot",
    ]

    output = run_bundle(spec, bundle)
    assert output["signals"][0]["kind"] == "selection"
    assert output["signals"][0]["candidates"]


def test_candidate_trade_bars_are_planned_after_derivative_filters_and_replay(
    monkeypatch,
):
    frozen, calls = _fixture_collector(monkeypatch)
    spec = load_spec(str(SPEC_RUNUP))
    spec["selection"]["long_when"] = {
        "cond": "conditions.value_above", "args": ["gain_3m", 100.0]
    }
    spec["selection"]["candidate_trade"] = {"entry_type": "market"}
    spec["sizing"] = {"size": 0.05}
    spec["risk"] = {
        "exit": {
            "type": "pct_stop_tp", "stop_pct": 0.02, "tp_pct": 0.04,
            "max_bars": 48,
        }
    }
    errors, _warnings = validate_spec(spec)
    assert errors == []

    bundle = collect_live_bundle_for_spec(
        spec, run_id="run-candidate-live", trace_id="trace-candidate-live"
    )

    assert [[request[2] for request in call["requests"]] for call in calls] == [
        ["universe", "contract_meta"],
        ["open_interest_snapshot"],
        ["oi_change_snapshot"],
        ["universe_bars"],
    ]
    bars = _request(calls, "universe_bars")[1]
    assert len(bars["symbols"]) == 5
    assert bars["timeframes"] == ["1d"]
    # Bars are cut at the instant the prefix frames were complete, not at an
    # unrelated "now". The final decision is one pass later and records that.
    assert bars["end_ms"] == frozen["decision_time"] + 3
    assert bundle["decision_time"] == frozen["decision_time"] + 4
    assert all(
        int(row["available_time"]) <= bundle["decision_time"]
        for frame in bundle["frames"].values()
        for row in frame.get("rows") or ()
        if row.get("available_time") is not None
    )

    batch = run_bundle(spec, bundle)
    signal = batch["signals"][0]
    assert signal["kind"] == "selection"
    assert signal["candidates"]
    assert all(candidate["trade"]["kind"] == "trade"
               for candidate in signal["candidates"])
    assert all(candidate["trade"]["intent"] == "open_long"
               for candidate in signal["candidates"])
    assert all(candidate["trade"]["auto_trade_eligible"] is False
               for candidate in signal["candidates"])
    assert all(
        candidate["trade"]["provenance"]["snapshot_id"]
        == bundle["snapshot_id"]
        for candidate in signal["candidates"]
    )
    assert all(
        candidate["trade"]["provenance"]["run_id"] == "run-candidate-live"
        and candidate["trade"]["provenance"]["trace_id"]
        == "trace-candidate-live"
        for candidate in signal["candidates"]
    )

    # The serialized bundle is the complete replay boundary: same snapshot id,
    # same nested trade decisions, and no collector call on replay.
    frozen_calls = len(calls)
    replay = json.loads(json.dumps(bundle))
    assert run_bundle(spec, replay) == batch
    assert len(calls) == frozen_calls


def test_empty_selection_prefix_skips_paid_fan_out_but_stays_kind_selection(
    monkeypatch,
):
    _frozen, calls = _fixture_collector(monkeypatch)
    spec = load_spec(str(SPEC_OI))
    # This narrowing step is before both paid derivative sources and is known to
    # match nothing in the frozen universe.  The correct answer is an empty
    # selection, not a failed data capture and not a trade-shaped hold signal.
    spec["selection"]["universe"].insert(3, {
        "block": "universe.only_symbols",
        "params": {"symbols": ["THIS_SYMBOL_DOES_NOT_EXIST_USDT"]},
    })
    errors, _warnings = validate_spec(spec)
    assert errors == []

    bundle = collect_live_bundle_for_spec(spec)

    assert len(calls) == 1
    assert [request[2] for request in calls[0]["requests"]] == [
        "universe", "contract_meta",
    ]
    for key in ("open_interest_snapshot", "oi_change_snapshot"):
        assert bundle["frames"][key]["rows"] == []
        assert bundle["frames"][key]["collection_skipped"] is True
        assert bundle["source_status"][key] == "skipped: empty selection prefix"

    batch = run_bundle(spec, bundle)
    signal = batch["signals"][0]
    assert signal["kind"] == "selection"
    assert signal["candidates"] == []
    assert signal["reason_codes"] == [
        "empty_basket", "no_candidate_qualified",
    ]

    # The skip marker is evidence, not authority.  Reusing it with a spec whose
    # prefix still has survivors must fail closed instead of bypassing required
    # source checks.
    unfiltered = load_spec(str(SPEC_OI))
    with pytest.raises(BundleRunError, match="required input source"):
        run_bundle(unfiltered, bundle)


def test_mvp_yaml_entrypoint_uses_the_spec_aware_multi_pass_collector(
    monkeypatch, capsys,
):
    """The documented bundle CLI must not reintroduce one-pass fan-out.

    ``collect_live_bundle_for_spec`` already proves the dependency ordering, but
    this is the actual user-facing caller.  A regression here used to request
    every fan-out source before the YAML prefix had a surviving-symbol roster,
    then fail before making any network request.
    """
    _frozen, calls = _fixture_collector(monkeypatch)
    from cyqnt_trd.standard_bot.entrypoints import mvp_input_bundle

    code = mvp_input_bundle.main([
        "--strategy-yaml", str(SPEC_RUNUP), "--format", "json",
    ])

    assert code == 0
    assert [[request[2] for request in call["requests"]] for call in calls] == [
        ["universe", "contract_meta"],
        ["open_interest_snapshot"],
        ["oi_change_snapshot"],
        ["universe_bars"],
    ]
    output = json.loads(capsys.readouterr().out)
    assert output["signals"][0]["kind"] == "selection"


def test_mvp_yaml_entrypoint_refuses_sections_that_would_bypass_the_spec():
    from cyqnt_trd.standard_bot.entrypoints import mvp_input_bundle

    with pytest.raises(SystemExit, match="--sections cannot override --strategy-yaml"):
        mvp_input_bundle.main([
            "--strategy-yaml", str(SPEC_RUNUP), "--sections", "universe",
        ])
