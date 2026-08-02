"""Funding selection must cross the real YAML -> bundle -> Blocks -> v2 path."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from cyqnt_trd.blocks import universe as universe_blocks
from cyqnt_trd.standard_bot.data.catalog import Availability, get_node
from cyqnt_trd.standard_bot.data.input_bundle import build_input_bundle
from cyqnt_trd.standard_bot.data.live_snapshot import requests_for_sections
from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import (
    BundleRunError,
    live_sections_for_spec,
    required_bundle_nodes,
    run_bundle,
)
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (
    SpecError,
    build_selection_fn,
)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import validate_spec

ROOT = Path(__file__).parents[2]
DT = 1_800_000_000_000


def _funding_spec():
    return {
        "spec_version": "1.0",
        "target": "standard_bot",
        "strategy": {
            "id": "yaml_funding_selector",
            "description": "rank the current universe by funding rate",
        },
        "run": {"mode": "backtest"},
        "data": {
            "symbol": "BTCUSDT",
            "market_type": "futures",
            "primary": {"interval": "1h"},
        },
        "selection": {
            "universe": [{
                "block": "universe.augment_with_funding",
                "with": ["funding"],
            }],
            "score": "fundingRatePct",
            "top_k": 3,
            "dedupe_by": "base_asset",
        },
    }


def _universe():
    return pd.DataFrame({
        "instrument_id": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "available_time": [DT] * 3,
        "quote_volume": [9e9, 8e9, 7e9],
    })


def _funding_frame(values=(0.0001, 0.0003, 0.0002)):
    rows = []
    # Older values deliberately imply the opposite ranking.  The block must
    # choose the latest PIT-safe row per instrument, not the first row it sees.
    for symbol, value in zip(("BTCUSDT", "ETHUSDT", "SOLUSDT"), reversed(values)):
        rows.append({
            "instrument_id": symbol,
            "metric": "funding_rate",
            "value": value,
            "event_time": DT - 10_000,
            "available_time": DT - 9_000,
            "unit": "ratio",
            "source_id": "fixture.funding",
        })
    for symbol, value in zip(("BTCUSDT", "ETHUSDT", "SOLUSDT"), values):
        rows.append({
            "instrument_id": symbol,
            "metric": "funding_rate",
            "value": value,
            "event_time": DT - 2_000,
            "available_time": DT - 1_000,
            "unit": "ratio",
            "source_id": "fixture.funding",
        })
    return pd.DataFrame(rows)


def _bundle(funding=None):
    return build_input_bundle(
        symbol="BTCUSDT",
        interval="1h",
        decision_time=DT,
        universe_frame=_universe(),
        extra_frames={} if funding is None else {"funding": funding},
    )


def _candidate_symbols(bundle, spec=None):
    output = run_bundle(spec or _funding_spec(), bundle)
    return [item["symbol"] for item in output["signals"][0]["candidates"]]


def test_validate_funding_selection_is_offline_and_runnable(monkeypatch):
    calls = []

    def _network_must_not_run(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("validation must use the supplied synthetic frame")

    monkeypatch.setattr(
        universe_blocks._data, "fetch_premium_index", _network_must_not_run
    )
    errors, warnings = validate_spec(_funding_spec())
    assert errors == []
    assert calls == []
    # Exactly one warning, and it is the annualisation notice: this spec declares
    # `with: [funding]` only, so fundingIntervalHours / fundingRateApr are NaN.
    # That is deliberate — the block will not assume an 8-hour settlement interval,
    # because the venue settles 443 of 743 perpetuals 4-hourly, and a spec ranking
    # the raw per-settlement rate is mixing units across rows. Asserted by content
    # rather than as `warnings == []`, which would go green again if the notice
    # were ever silently dropped.
    assert len(warnings) == 1, warnings
    assert "funding_info" in warnings[0]
    assert "NOT defaulted to 8 hours" in warnings[0]


def test_with_source_name_must_resolve_to_a_real_frame(monkeypatch):
    monkeypatch.setattr(
        universe_blocks._data,
        "fetch_premium_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a missing `with` source must not fall back to live")
        ),
    )
    with pytest.raises(SpecError, match="funding.*not provided"):
        build_selection_fn(_funding_spec())(_universe(), frames={})


@pytest.mark.parametrize("state", ["missing", "empty", "error"])
def test_missing_empty_or_failed_funding_fails_closed(state):
    bundle = _bundle(_funding_frame())
    if state == "missing":
        bundle["frames"].pop("funding")
        bundle["source_status"].pop("funding")
    elif state == "empty":
        bundle["frames"]["funding"]["rows"] = []
        bundle["source_status"]["funding"] = "empty"
    else:
        bundle["frames"]["funding"]["rows"] = []
        bundle["source_status"]["funding"] = "error: timeout"

    with pytest.raises(BundleRunError, match="funding"):
        run_bundle(_funding_spec(), bundle)


def test_funding_snapshot_normalizes_to_the_canonical_metric_frame():
    node = get_node("funding_snapshot")
    raw = pd.DataFrame({
        "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "lastFundingRate": [0.0001, 0.0003, 0.0002],
        "markPrice": [100.0, 200.0, 50.0],
        "time": [DT] * 3,
    })
    normalized, _warnings, _inferred = node.normalize(raw, available_time=DT)

    assert node.input_schema.name == "MetricFrame@1.0"
    assert node.availability is Availability.FORWARD_ONLY
    assert set(normalized["instrument_id"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert set(normalized["metric"]) == {"funding_rate"}
    assert set(normalized["unit"]) == {"ratio"}
    assert normalized[["event_time", "available_time", "value"]].notna().all().all()


def test_funding_bundle_runs_blocks_and_emits_v2():
    output = run_bundle(_funding_spec(), _bundle(_funding_frame()))
    assert output["schema"] == "cyqnt.signal-batch/v1"
    assert output["signal_count"] == 1
    signal = output["signals"][0]
    assert signal["schema"] == "cyqnt.signal/v2"
    assert signal["kind"] == "selection"
    assert len(signal) == 42
    assert [item["symbol"] for item in signal["candidates"]] == [
        "ETHUSDT", "SOLUSDT", "BTCUSDT",
    ]
    assert [item["features"]["fundingRatePct"] for item in signal["candidates"]] == [
        0.03, 0.02, 0.01,
    ]
    assert signal["source_status"]["funding"] == "ok"

    import jsonschema

    batch_schema = json.loads((ROOT / "strategies" / "_standard" /
                               "signal_batch.schema.v1.json").read_text())
    signal_schema = json.loads((ROOT / "strategies" / "_standard" /
                                "signal.schema.v2.json").read_text())
    batch_schema["properties"]["signals"]["items"] = signal_schema
    jsonschema.validate(output, batch_schema)


def test_swapping_only_funding_values_swaps_the_actual_ranking():
    original = _candidate_symbols(_bundle(_funding_frame()))
    swapped = _candidate_symbols(_bundle(_funding_frame((0.0003, 0.0001, 0.0002))))
    assert original == ["ETHUSDT", "SOLUSDT", "BTCUSDT"]
    assert swapped == ["BTCUSDT", "SOLUSDT", "ETHUSDT"]


def test_future_publication_cannot_change_a_past_selection():
    base_frame = _funding_frame()
    leaked = pd.concat([
        base_frame,
        pd.DataFrame([{
            "instrument_id": "BTCUSDT",
            "metric": "funding_rate",
            "value": 99.0,
            "event_time": DT - 30 * 60_000,
            "available_time": DT + 1,
            "unit": "ratio",
            "source_id": "fixture.future",
        }]),
    ], ignore_index=True)
    before = _bundle(base_frame)
    after = _bundle(leaked)

    assert len(after["frames"]["funding"]["rows"]) == len(
        before["frames"]["funding"]["rows"]
    )
    assert _candidate_symbols(before) == _candidate_symbols(after)


def test_single_symbol_history_cannot_masquerade_as_a_cross_section():
    one_symbol = _funding_frame()
    one_symbol = one_symbol[one_symbol["instrument_id"] == "BTCUSDT"]
    with pytest.raises(BundleRunError, match="coverage 1/3"):
        run_bundle(_funding_spec(), _bundle(one_symbol))


def test_live_plan_uses_snapshot_alias_for_selection_and_history_for_trade():
    spec = _funding_spec()
    assert required_bundle_nodes(spec) == {"universe", "funding"}
    selection_plan = requests_for_sections(live_sections_for_spec(spec))
    funding_requests = [request for request in selection_plan
                        if request[2] == "funding"]
    assert funding_requests == [("funding_snapshot", {}, "funding")]
    assert any(node == "universe" for node, _params, _alias in selection_plan)
    assert not any(node == "ticker_rank" for node, _params, _alias in selection_plan)

    trade = {
        "signals": {"entry": {"long": {
            "cond": "conditions.value_above",
            "args": ["funding_rate", 0],
        }}},
    }
    trade_plan = requests_for_sections(live_sections_for_spec(trade))
    assert ("funding_snapshot", {}, "funding") not in trade_plan
    assert any(node == "funding" and alias == "funding"
               for node, _params, alias in trade_plan)


def test_legacy_python_no_arg_funding_still_fetches_once(monkeypatch):
    calls = []

    def _premium():
        calls.append(True)
        return pd.DataFrame({
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "lastFundingRate": [0.0001, 0.0003],
        })

    monkeypatch.setattr(universe_blocks._data, "fetch_premium_index", _premium)
    selected = (
        universe_blocks.UniverseFilter(pd.DataFrame({
            "symbol": ["BTCUSDT", "ETHUSDT"],
        }))
        .with_funding()
        .filter_funding_rate(max_abs_pct=0.015)
        .symbols()
    )
    assert selected == ["BTCUSDT"]
    assert calls == [True]
