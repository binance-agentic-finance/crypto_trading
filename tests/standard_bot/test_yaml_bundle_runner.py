"""Golden YAML + input bundle -> Blocks -> v2 output tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cyqnt_trd.standard_bot.bot import BotContext
from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import (
    BundleRunError, run_bundle)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import load_spec

ROOT = Path(__file__).parents[2]
BUNDLE = ROOT / "docs" / "standard_bot_io" / "samples" / "input_bundle_example.json"
TRADE_SPEC = ROOT / "docs" / "strategy_yaml_spec" / "example_multi_factor.yaml"
SELECTION_SPEC = ROOT / "docs" / "strategy_yaml_spec" / "example_selection.yaml"


def _bundle():
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def _always_long_spec():
    spec = load_spec(str(TRADE_SPEC))
    spec["strategy"]["id"] = "golden_yaml_trade"
    spec["signals"] = {
        "indicators": {},
        "entry": {
            "long": {"cond": "conditions.value_above", "args": ["close", 0]},
            "short": {"cond": "conditions.value_below", "args": ["close", 0]},
        },
    }
    spec["risk"]["exit"] = {"type": "time_only", "max_bars": 3}
    return spec


def test_trade_yaml_runs_blocks_and_emits_only_the_current_v2_decision():
    output = run_bundle(_always_long_spec(), _bundle())
    assert output["schema"] == "cyqnt.signal-batch/v1"
    assert output["signal_count"] == 1, "historical warm-up entries must not be republished"
    signal = output["signals"][0]
    assert signal["schema"] == "cyqnt.signal/v2"
    assert signal["kind"] == "trade"
    assert signal["intent"] == "open_long"
    assert len(signal) == 42
    assert signal["decision_time"] == output["decision_time"]

    import jsonschema

    schema = json.loads((ROOT / "strategies" / "_standard" /
                         "signal_batch.schema.v1.json").read_text(encoding="utf-8"))
    signal_schema = json.loads((ROOT / "strategies" / "_standard" /
                                "signal.schema.v2.json").read_text(encoding="utf-8"))
    schema["properties"]["signals"]["items"] = signal_schema
    jsonschema.validate(output, schema)


def test_changing_the_block_condition_changes_the_actual_output():
    """Counterfactual check: this proves the pass is not schema-only."""
    bundle = _bundle()
    spec = _always_long_spec()
    assert run_bundle(spec, bundle)["signal_count"] == 1
    latest_close = bundle["frames"]["klines"]["rows"][-1]["close"]
    changed = copy.deepcopy(spec)
    changed["signals"]["entry"]["long"]["args"][1] = latest_close + 1
    assert run_bundle(changed, bundle)["signal_count"] == 0


def test_selection_yaml_uses_the_same_batch_and_same_signal_contract():
    output = run_bundle(str(SELECTION_SPEC), _bundle())
    assert output["schema"] == "cyqnt.signal-batch/v1"
    assert output["signal_count"] == 1
    signal = output["signals"][0]
    assert signal["kind"] == "selection"
    assert signal["candidates"]
    assert len(signal) == 42


def test_a_required_source_error_stops_the_strategy_instead_of_lying():
    bundle = _bundle()
    spec = _always_long_spec()
    spec["data"]["derivatives"] = {"dir": "unused-for-bundle"}
    spec["signals"]["entry"]["long"] = {
        "cond": "conditions.value_above", "args": ["funding_rate_bps", -999]
    }
    bundle["source_status"]["funding"] = "error: colleague endpoint unavailable"
    with pytest.raises(BundleRunError, match="funding=error"):
        run_bundle(spec, bundle)


def test_run_bundle_rejects_a_tampered_event_clock_before_blocks_run():
    """The YAML runner cannot bypass the serialized input contract ingress."""
    bundle = _bundle()
    row = bundle["frames"]["universe"]["rows"][0]
    row["event_time"] = int(row["available_time"]) + 1

    with pytest.raises(
        ValueError,
        match=r"frame 'universe' has rows whose event_time is AFTER available_time",
    ):
        run_bundle(str(SELECTION_SPEC), bundle)


def test_required_error_with_a_reason_is_insufficient_not_merely_degraded():
    ctx = BotContext(
        decision_time=1,
        source_status={"funding": "error: timeout", "klines": "ok"},
    )
    assert ctx.data_quality(required=("funding",)).value == "insufficient"


def test_both_user_facing_clis_delegate_to_the_same_bundle_runner(tmp_path, capsys):
    import yaml

    from cyqnt_trd.standard_bot.entrypoints import mvp_input_bundle
    from cyqnt_trd.standard_bot.yaml_pipeline import cli as yaml_cli

    spec_path = tmp_path / "strategy.yaml"
    spec_path.write_text(yaml.safe_dump(_always_long_spec()), encoding="utf-8")
    expected = run_bundle(str(spec_path), str(BUNDLE))

    assert mvp_input_bundle.main([
        "--replay", str(BUNDLE), "--strategy-yaml", str(spec_path),
        "--format", "json",
    ]) == 0
    first = json.loads(capsys.readouterr().out)

    assert yaml_cli.main([
        "run", str(spec_path), "--input-json", str(BUNDLE),
    ]) == 0
    second = json.loads(capsys.readouterr().out)
    assert first == second == expected
