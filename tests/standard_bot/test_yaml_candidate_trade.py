"""Selection YAML can hand every qualified name a complete v2 trade plan."""

from __future__ import annotations

import copy
import pytest

import json
from pathlib import Path

from cyqnt_trd.standard_bot.core import StandardSignal
from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import run_bundle
from cyqnt_trd.standard_bot.yaml_pipeline.spec import validate_spec


FIXTURE = "tests/standard_bot/fixtures/universe_derivatives.json"
SYMBOLS = ["ALLOUSDT", "TAGUSDT", "UAIUSDT", "UBUSDT", "USUSDT"]
SHIPPED_SPEC = "docs/strategy_yaml_spec/example_selection_candidate_trade.yaml"


def _spec(*, candidate_trade: bool = True, long_below: float = 101) -> dict:
    selection = {
        "universe": [
            {
                "block": "universe.only_symbols",
                "params": {"symbols": SYMBOLS},
            },
            {
                "block": "universe.augment_with_indicator",
                "with": ["universe_bars"],
                "params": {
                    "indicator": "rsi",
                    "timeframe": "1h",
                    "period": 14,
                    "as": "rsi14",
                },
            },
        ],
        "score": "quoteVolume",
        "order": "desc",
        "top_k": 5,
        # The high threshold is deliberate: this contract test needs all five
        # frozen candidates to qualify; it is not claiming 101 is a trading rule.
        "long_when": {
            "cond": "conditions.value_below",
            "args": ["rsi14", long_below],
        },
    }
    spec = {
        "spec_version": "1.0",
        "target": "standard_bot",
        "strategy": {"id": "candidate_trade_probe"},
        "run": {"mode": "backtest"},
        "data": {
            "symbol": "BTCUSDT",
            "market_type": "futures",
            "primary": {"interval": "1h"},
        },
        "selection": selection,
    }
    if candidate_trade:
        selection["candidate_trade"] = {"entry_type": "market"}
        spec["sizing"] = {"size": 0.05}
        spec["risk"] = {
            "exit": {
                "type": "pct_stop_tp",
                "stop_pct": 0.02,
                "tp_pct": 0.04,
            }
        }
    return spec


def test_selection_candidate_trade_emits_nested_v2_signals():
    spec = _spec()
    errors, warnings = validate_spec(spec)
    assert errors == []
    assert warnings == []

    signal = run_bundle(spec, FIXTURE)["signals"][0]

    assert signal["kind"] == "selection"
    assert len(signal["candidates"]) == 5
    for candidate in signal["candidates"]:
        assert candidate["direction"] == "long"
        trade = candidate["trade"]
        assert trade["schema"] == "cyqnt.signal/v2"
        assert trade["kind"] == "trade"
        assert set(trade) == set(signal)
        assert trade["symbol"] == candidate["symbol"]
        assert trade["intent"] == "open_long"
        assert trade["entry"]["type"] == "market"
        assert trade["size"]["mode"] == "equity_pct"
        assert trade["size"]["value"] == 0.05
        assert trade["exit_plan"]["stop_loss"]["pct"] == 0.02
        assert trade["exit_plan"]["take_profit"][0]["pct"] == 0.04
        assert trade["auto_trade_eligible"] is False
        assert trade["requires_confirmation"] is True
        # The nested payload is not merely JSON-shaped; the canonical parser
        # must accept it without dropping its execution fields.
        rebuilt = StandardSignal.from_dict(trade)
        assert rebuilt.to_dict() == trade


def test_plain_selection_keeps_candidate_trade_null():
    signal = run_bundle(_spec(candidate_trade=False), FIXTURE)["signals"][0]
    assert signal["kind"] == "selection"
    assert signal["candidates"]
    assert all(candidate["trade"] is None for candidate in signal["candidates"])


def test_plain_selection_does_not_require_risk_or_sizing():
    spec = _spec(candidate_trade=False)

    errors, _ = validate_spec(spec)

    assert errors == []


def test_plain_selection_rejects_trade_fields_that_would_be_ignored():
    spec = _spec(candidate_trade=False)
    spec["sizing"] = {"size": 0.05}
    spec["risk"] = {"exit": {"type": "time_only", "max_bars": 10}}

    errors, _ = validate_spec(spec)

    assert any("pure selection does not consume sizing" in error for error in errors)
    assert any("pure selection does not consume risk" in error for error in errors)


def test_selection_rejects_backtest_fields_it_cannot_execute():
    spec = _spec()
    spec["backtest"] = {
        "initial_capital": 10_000,
        "execution_model": "next_bar_open",
    }

    errors, _ = validate_spec(spec)

    assert any("selection does not consume backtest" in error for error in errors)


@pytest.mark.parametrize("mutate,needle", [
    (lambda spec: spec["selection"].__setitem__("candidate_trade", "market"),
     "selection.candidate_trade must be a mapping"),
    (lambda spec: spec["selection"]["candidate_trade"].__setitem__(
        "entry_typo", "market"),
     "unknown selection.candidate_trade.entry_typo"),
    (lambda spec: spec["selection"]["candidate_trade"].pop("entry_type"),
     "entry_type is required and must be 'market'"),
    (lambda spec: spec["selection"]["candidate_trade"].__setitem__(
        "entry_type", "limit"),
     "entry_type is required and must be 'market'"),
    (lambda spec: spec["selection"].pop("long_when"),
     "requires at least one of selection.long_when / selection.short_when"),
    (lambda spec: spec.pop("sizing"),
     "requires explicit sizing.size"),
    (lambda spec: spec["sizing"].pop("size"),
     "requires explicit sizing.size"),
    (lambda spec: spec.pop("risk"),
     "requires risk.exit as a mapping"),
    (lambda spec: spec["risk"].pop("exit"),
     "requires risk.exit as a mapping"),
])
def test_candidate_trade_schema_fails_closed(mutate, needle):
    spec = _spec()
    mutate(spec)

    errors, _ = validate_spec(spec)

    assert any(needle in error for error in errors), errors


def test_spot_candidate_trade_cannot_emit_a_short_plan():
    spec = _spec()
    spec["data"]["market_type"] = "spot"
    spec["selection"]["short_when"] = spec["selection"].pop("long_when")

    errors, _ = validate_spec(spec)

    assert any("spot cannot open a short position" in error for error in errors), errors


def test_candidate_trade_keeps_selected_names_with_an_explicit_hold_signal():
    signal = run_bundle(_spec(long_below=-1), FIXTURE)["signals"][0]

    assert len(signal["candidates"]) == 5
    assert all(candidate["direction"] == "neutral"
               for candidate in signal["candidates"])
    assert all(candidate["trade"]["kind"] == "trade"
               for candidate in signal["candidates"])
    assert all(candidate["trade"]["intent"] == "hold"
               for candidate in signal["candidates"])
    assert all(candidate["trade"]["entry"] is None
               for candidate in signal["candidates"])


def test_overlapping_long_and_short_conditions_fail_closed():
    spec = _spec(long_below=101)
    spec["selection"]["short_when"] = {
        "cond": "conditions.value_above",
        "args": ["rsi14", -1],
    }

    with pytest.raises(
        ValueError,
        match="selection.long_when and selection.short_when are both true",
    ):
        run_bundle(spec, FIXTURE)


def test_batch_run_and_trace_ids_reach_nested_candidate_trades():
    bundle = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))
    bundle["run_id"] = "run-candidate-trade"
    bundle["trace_id"] = "trace-candidate-trade"

    signal = run_bundle(_spec(), bundle)["signals"][0]

    assert signal["provenance"]["run_id"] == "run-candidate-trade"
    assert signal["provenance"]["trace_id"] == "trace-candidate-trade"
    for candidate in signal["candidates"]:
        assert candidate["trade"]["provenance"]["run_id"] == "run-candidate-trade"
        assert candidate["trade"]["provenance"]["trace_id"] == "trace-candidate-trade"


def test_declared_assumptions_reach_each_nested_trade():
    spec = _spec()
    spec["strategy"]["assumptions"] = [{
        "cid": "candidate_rsi_thresholds",
        "reading": "未指定 RSI 門檻時採 30 / 70",
        "alternatives": ["由使用者另行指定門檻"],
    }]

    signal = run_bundle(spec, FIXTURE)["signals"][0]

    assert "assumption_declared" in signal["reason_codes"]
    for candidate in signal["candidates"]:
        trade = candidate["trade"]
        assert "assumption_declared" in trade["reason_codes"]
        assert any("candidate_rsi_thresholds" in warning
                   for warning in trade["warnings"])


@pytest.mark.parametrize("mutate,needle", [
    (lambda trade: trade.__setitem__("symbol", "WRONGUSDT"),
     "does not match candidate"),
    (lambda trade: trade.__setitem__("auto_trade_eligible", True),
     "not order authority"),
    (lambda trade: trade.__setitem__("requires_confirmation", False),
     "not order authority"),
])
def test_nested_trade_integrity_is_enforced_by_the_v2_contract(mutate, needle):
    signal = run_bundle(_spec(), FIXTURE)["signals"][0]
    mutate(signal["candidates"][0]["trade"])

    with pytest.raises(ValueError, match=needle):
        StandardSignal.from_dict(signal)


def test_rsi_counterfactual_controls_long_hold_and_short_nested_intents():
    import yaml

    spec = yaml.safe_load(Path(SHIPPED_SPEC).read_text(encoding="utf-8"))
    spec["selection"]["universe"][0] = {
        "block": "universe.only_symbols",
        "params": {"symbols": SYMBOLS},
    }
    bundle = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))

    baseline = run_bundle(spec, bundle)["signals"][0]
    baseline_intents = {
        item["symbol"]: item["trade"]["intent"]
        for item in baseline["candidates"]
    }
    assert baseline["kind"] == "selection"
    assert baseline_intents["UAIUSDT"] == "open_short"
    assert baseline_intents["ALLOUSDT"] == "hold"

    falling = copy.deepcopy(bundle)
    allo = [
        row for row in falling["frames"]["universe_bars"]["rows"]
        if row["instrument_id"] == "ALLOUSDT" and row["timeframe"] == "1h"
    ]
    for offset, row in enumerate(allo[-20:]):
        row["close"] = 0.30 - offset * 0.01

    changed = run_bundle(spec, falling)["signals"][0]
    changed_intents = {
        item["symbol"]: item["trade"]["intent"]
        for item in changed["candidates"]
    }
    assert changed["kind"] == "selection"
    assert changed_intents["ALLOUSDT"] == "open_long"
    assert changed_intents["UAIUSDT"] == "open_short"
