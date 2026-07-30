"""``cyqnt.input/v1`` — the typed input contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.core.input_contract import (
    BAR_FRAME,
    EVENT_FRAME,
    METRIC_FRAME,
    POSITION_FRAME,
    SCHEMAS,
    FrameKind,
    FrameValidationError,
    TypedFrame,
    normalize_frame,
)
from cyqnt_trd.standard_bot.data.catalog import get_node, list_nodes

NOW = 1_786_000_000_000
REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# the seven shapes
# --------------------------------------------------------------------------


def test_every_shape_states_what_one_row_means():
    for kind, schema in SCHEMAS.items():
        assert schema.row_grain, "%s does not say what a row is" % kind.value
        assert schema.description
        assert "available_time" in schema.required + schema.optional


def test_every_node_declares_a_shape():
    """No RAW left: an input with no declared shape is an input nobody can read."""
    raw = [node.name for node in list_nodes() if node.emits is FrameKind.RAW]
    assert raw == []


def test_shape_coverage_matches_the_catalog():
    from collections import Counter

    counts = Counter(node.emits.value for node in list_nodes())
    assert sum(counts.values()) == len(list_nodes())
    # the long-form metric shape carries the bulk of the numeric universe
    assert counts["metric"] >= counts["bar"]


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_missing_required_column_is_refused():
    frame = pd.DataFrame([{"instrument_id": "BTCUSDT", "value": 1.0}])
    with pytest.raises(FrameValidationError) as excinfo:
        METRIC_FRAME.validate(frame, node="unit")
    assert "missing required column" in str(excinfo.value)
    assert "metric" in str(excinfo.value)


def test_a_row_readable_before_it_happened_is_refused():
    frame = pd.DataFrame([{
        "event_time": pd.Timestamp("2026-08-06T08:00:00Z"),
        "available_time": pd.Timestamp("2026-08-06T07:00:00Z"),
        "instrument_id": "BTCUSDT", "metric": "x", "value": 1.0,
    }])
    with pytest.raises(FrameValidationError) as excinfo:
        METRIC_FRAME.validate(frame, node="unit")
    assert "AFTER available_time" in str(excinfo.value)


def test_empty_frames_pass_validation():
    METRIC_FRAME.validate(pd.DataFrame(columns=list(METRIC_FRAME.required)), node="unit")


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def test_wide_columns_melt_into_metric_rows():
    frame = pd.DataFrame({
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "timestamp": [NOW - 28_800_000, NOW],
        "rate": [0.0001, 0.0002],
        "mark_price": [100.0, 101.0],
    })
    out, _warnings, _inferred = normalize_frame(
        frame, kind=FrameKind.METRIC, node="funding",
        column_map={"symbol": "instrument_id", "timestamp": "event_time"},
        value_columns=("rate", "mark_price"), available_time=NOW,
    )
    assert set(out["metric"]) == {"rate", "mark_price"}
    assert len(out) == 4
    assert out.loc[out["metric"] == "rate", "value"].tolist() == [0.0001, 0.0002]


def test_epoch_ms_is_not_parsed_as_nanoseconds():
    """A bare int64 read as ns turns every ms stamp into 1970 and disables PIT."""
    frame = pd.DataFrame({"symbol": ["BTCUSDT"], "timestamp": [NOW], "rate": [0.0001]})
    out, _w, _i = normalize_frame(
        frame, kind=FrameKind.METRIC, node="funding",
        column_map={"symbol": "instrument_id", "timestamp": "event_time"},
        value_columns=("rate",), available_time=NOW,
    )
    assert out["event_time"].iloc[0].year >= 2026


def test_inferred_available_time_is_recorded_not_assumed():
    """A missing publication lag is an assumption, and must be stated."""
    frame = pd.DataFrame({"instrument_id": ["BTCUSDT"], "metric": ["x"], "value": [1.0],
                          "event_time": [pd.Timestamp("2026-08-06T07:00:00Z")]})
    out, warnings, inferred = normalize_frame(
        frame, kind=FrameKind.METRIC, node="unit", available_time=NOW,
    )
    assert inferred is True
    assert "available_time" in out.columns
    assert any("no publication lag" in warning for warning in warnings)


def test_available_time_is_derived_per_row_not_from_the_fetch_clock():
    """One constant on every row would defeat both the PIT gate and any window.

    A bar is knowable when it closes, so ``available_time`` must track
    ``close_time`` row by row. Stamping the fetch time instead claims the whole
    history arrived at once: every row then passes a point-in-time filter and no
    time window can trim anything.
    """
    closes = [pd.Timestamp("2026-08-06T05:00:00Z"), pd.Timestamp("2026-08-06T06:00:00Z"),
              pd.Timestamp("2026-08-06T07:00:00Z")]
    frame = pd.DataFrame({
        "instrument_id": ["BTCUSDT"] * 3, "timeframe": ["1h"] * 3,
        "open_time": [c - pd.Timedelta(hours=1) for c in closes], "close_time": closes,
        "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 3.0,
    })
    out, _, inferred = normalize_frame(
        frame, kind=FrameKind.BAR, node="klines", available_time=NOW,
    )
    assert list(out["available_time"]) == closes, "must equal close_time per row"
    assert out["available_time"].nunique() == 3, "a single constant defeats the gate"
    assert inferred is False, "close_time is exact, not an assumption"


@pytest.mark.parametrize("stamps", [
    [pd.Timestamp("2026-08-06T05:00:00Z"), pd.Timestamp("2026-08-06T06:00:00Z")],
    [pd.Timestamp("2026-08-06T05:00:00"), pd.Timestamp("2026-08-06T06:00:00")],
])
def test_columns_that_are_already_datetimes_survive_parsing(stamps):
    """A parsed datetime column must not be re-read as an epoch number.

    ``pd.to_numeric`` renders datetime64 as nanoseconds (~1.8e18); the epoch-unit
    heuristic then reads that as milliseconds and every value coerces to NaT.
    Sources that hand back parsed timestamps — parquet, most internal adapters —
    had their whole time axis silently erased, and the PIT check could not catch
    it because it only compares timestamps that are both present.
    """
    frame = pd.DataFrame({
        "instrument_id": ["BTCUSDT"] * 2, "metric": ["x"] * 2, "value": [1.0, 2.0],
        "event_time": stamps,
    })
    out, _, _ = normalize_frame(
        frame, kind=FrameKind.METRIC, node="unit", available_time=NOW,
    )
    assert out["event_time"].notna().all(), "timestamps must survive normalisation"
    assert list(out["event_time"].dt.year) == [2026, 2026]


def test_a_true_snapshot_still_falls_back_to_the_fetch_time():
    """An order book has no per-row event time; the fetch instant is the truth."""
    frame = pd.DataFrame({"instrument_id": ["BTCUSDT", "BTCUSDT"], "side": ["bid", "ask"],
                          "price": [1.0, 2.0], "quantity": [3.0, 4.0]})
    out, warnings, inferred = normalize_frame(
        frame, kind=FrameKind.BOOK, node="orderbook_depth", available_time=NOW,
    )
    assert inferred is True
    assert out["available_time"].nunique() == 1
    assert any("snapshot" in warning for warning in warnings)
    assert list(out["level"]) == [1, 1], "rung derived from row order within a side"


def test_normalisation_without_a_fetch_time_refuses_to_guess():
    frame = pd.DataFrame({"instrument_id": ["BTCUSDT"], "metric": ["x"], "value": [1.0]})
    with pytest.raises(FrameValidationError) as excinfo:
        normalize_frame(frame, kind=FrameKind.METRIC, node="unit")
    assert "requires available_time" in str(excinfo.value)


def test_clock_skew_is_clamped_and_reported():
    frame = pd.DataFrame({
        "instrument_id": ["BTCUSDT"], "metric": ["x"], "value": [1.0],
        "event_time": [NOW + 60_000], "available_time": [NOW],
    })
    out, warnings, _i = normalize_frame(
        frame, kind=FrameKind.METRIC, node="unit", available_time=NOW,
    )
    assert out["event_time"].iloc[0] == out["available_time"].iloc[0]
    assert any("clamped" in warning for warning in warnings)


def test_absent_declared_source_column_is_reported():
    frame = pd.DataFrame({"instrument_id": ["BTCUSDT"], "metric": ["x"], "value": [1.0],
                          "available_time": [NOW]})
    _out, warnings, _i = normalize_frame(
        frame, kind=FrameKind.METRIC, node="unit",
        column_map={"not_there": "value"}, available_time=NOW,
    )
    assert any("absent from the response" in warning for warning in warnings)


# --------------------------------------------------------------------------
# param-derived instrument
# --------------------------------------------------------------------------


def test_instrument_is_taken_from_the_request_when_the_response_omits_it():
    """open_interest / long_short_ratio / basis take symbol as a *parameter*
    and never echo it; the instrument is known because we asked for it."""
    spec = get_node("open_interest")
    frame = pd.DataFrame({"timestamp": [NOW], "oi_base": [10.0], "oi_value": [1000.0],
                          "oi_change_bps": [0.0]})
    out, _w, _i = spec.normalize(
        frame, available_time=NOW, params={"symbol": "btcusdt", "period": "1h"}
    )
    assert set(out["instrument_id"]) == {"BTCUSDT"}   # upper-cased
    assert set(out["timeframe"]) == {"1h"}


def test_an_explicit_node_constant_wins_over_the_request():
    spec = get_node("fear_greed")
    frame = pd.DataFrame({"date": ["2026-08-05"], "value": [55],
                          "value_classification": ["Greed"]})
    out, _w, _i = spec.normalize(
        frame, available_time=NOW, params={"symbol": "BTCUSDT"}
    )
    # the node pins instrument_id="MARKET"; a stray symbol param must not override it
    assert set(out["instrument_id"]) == {"MARKET"}


# --------------------------------------------------------------------------
# TypedFrame readers
# --------------------------------------------------------------------------


def _metric_view():
    frame = pd.DataFrame({
        "event_time": pd.to_datetime([NOW - 28_800_000, NOW], unit="ms", utc=True),
        "available_time": pd.to_datetime([NOW, NOW], unit="ms", utc=True),
        "instrument_id": ["BTCUSDT", "BTCUSDT"],
        "metric": ["rate", "rate"], "value": [0.0001, 0.0002],
    })
    return TypedFrame(node="funding", kind=FrameKind.METRIC, frame=frame, as_of=NOW)


def test_typed_frame_reads_the_latest_metric():
    assert _metric_view().metric("rate") == pytest.approx(0.0002)


def test_a_missing_metric_reads_as_none_not_zero():
    assert _metric_view().metric("open_interest") is None


def test_typed_frame_lists_its_instruments():
    assert _metric_view().instruments() == ["BTCUSDT"]


def test_typed_frame_exposes_the_replay_discipline():
    view = TypedFrame(node="ticker_rank", kind=FrameKind.RANK, frame=pd.DataFrame(),
                      availability="FORWARD_ONLY", pit_hazard="60s cache, no history")
    described = view.to_dict()
    assert described["availability"] == "FORWARD_ONLY"
    assert "no history" in described["pit_hazard"]
    assert view.empty is True


# --------------------------------------------------------------------------
# the shipped samples stay in step with the code
# --------------------------------------------------------------------------


def test_samples_regenerate_identically():
    """The design doc quotes these files; if the contract moves they must move."""
    samples = REPO / "docs" / "standard_bot_io" / "samples"
    before = {path.name: path.read_text(encoding="utf-8") for path in samples.glob("*.json")}
    assert before, "no samples committed"

    result = subprocess.run(
        [sys.executable, str(REPO / "docs" / "standard_bot_io" / "gen_samples.py")],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    after = {path.name: path.read_text(encoding="utf-8") for path in samples.glob("*.json")}
    stale = sorted(name for name in before if before[name] != after.get(name))
    assert stale == [], "stale samples: %s — rerun gen_samples.py" % ", ".join(stale)


def test_sample_context_shows_every_declared_input_with_a_status():
    payload = json.loads(
        (REPO / "docs" / "standard_bot_io" / "samples" / "bot_context.json").read_text()
    )
    assert payload["schema"] == "cyqnt.input/v1"
    # every declared input has a status, including ones that produced no frame
    assert set(payload["declared_inputs"]) <= set(payload["source_status"])
    # positions were adopted from the declared PositionFrame input
    assert payload["positions"]["BTCUSDT"] < 0


def test_sample_outputs_cover_the_position_lifecycle():
    folder = REPO / "docs" / "standard_bot_io" / "samples"
    opened = json.loads((folder / "output_open_long.json").read_text())
    closed = json.loads((folder / "output_close_short.json").read_text())
    advisory = json.loads((folder / "output_advisory.json").read_text())

    assert opened["intent"] == "open_long" and opened["closes_side"] is None
    assert closed["intent"] == "close_short"
    assert closed["closes_side"] == "short" and closed["order_side"] == "buy"
    assert closed["reduce_only"] is True
    assert advisory["auto_trade_eligible"] is False

    request = json.loads((folder / "output_open_long.execution_request.json").read_text())
    assert "idempotency_key" not in request
    assert request["position_intent"] == "open_long"
