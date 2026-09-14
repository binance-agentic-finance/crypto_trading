"""Funding acquisition metadata must survive the daily-panel integration layer."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / "eval/alpha101_crypto"
SPEC = importlib.util.spec_from_file_location("alpha101_v2_pipeline_test_target", DIRECTORY / "run_pipeline.py")
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC.loader.exec_module(PIPELINE)
finally:
    sys.path.pop(0)


def save_funding(directory, symbol, index, *, bad_mark_day=None,
                 invalid_row_day=None, gap_days=()):
    events = pd.date_range(index[0], index[-1] + pd.Timedelta(hours=16), freq="8h")
    frame = pd.DataFrame({
        "fundingTime": (events.asi8 // 1_000_000) + 4,
        "fundingRate": 0.0001,
        "markPrice": 100.0,
        "fundingIntervalHours": 8,
        "funding_input_valid": True,
    })
    if bad_mark_day is not None:
        pos = int(np.flatnonzero(events.normalize() == pd.Timestamp(bad_mark_day, tz="UTC"))[0])
        frame.loc[pos, "markPrice"] = np.nan
        frame.loc[pos, "funding_input_valid"] = False
    if invalid_row_day is not None:
        pos = int(np.flatnonzero(events.normalize() == pd.Timestamp(invalid_row_day, tz="UTC"))[0])
        frame.loc[pos, "funding_input_valid"] = False
    path = directory / "funding" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({
        "schema_version": 3, "status": "missing" if bad_mark_day or invalid_row_day or gap_days else "ok",
        "symbol": symbol, "pagination_complete": True,
        "queried_from": str(index[0]),
        "queried_through_exclusive": str(index[-1] + pd.Timedelta(days=1)),
        "first": str(events[0]), "last": str(events[-1]),
        "event_gap_diagnostics": {"gap_affected_utc_days": list(gap_days)},
    }))
    return frame


@pytest.fixture
def daily(tmp_path, monkeypatch):
    monkeypatch.setattr(PIPELINE, "DATA", tmp_path)
    index = pd.date_range("2025-01-01", periods=7, freq="D", tz="UTC")
    return tmp_path, index


def test_acquisition_gap_days_override_otherwise_complete_daily_cash(daily):
    directory, index = daily
    save_funding(directory, "A", index, gap_days=["2025-01-03"])
    panel, detail = PIPELINE.funding_panel(index, ["A"])
    assert np.isnan(panel.loc["2025-01-03", "A"])
    assert panel.loc["2025-01-02", "A"] == pytest.approx(0.03)
    assert panel.loc["2025-01-04", "A"] == pytest.approx(0.03)
    assert detail[0]["unknown_funding_days"] == 1


@pytest.mark.parametrize("verification", [False, None])
def test_unverified_pagination_keeps_entire_symbol_unknown(daily, verification):
    directory, index = daily
    save_funding(directory, "A", index)
    stamp = directory / "funding/A.json"
    metadata = json.loads(stamp.read_text())
    if verification is None:
        metadata.pop("pagination_complete")
    else:
        metadata["pagination_complete"] = verification
    stamp.write_text(json.dumps(metadata))
    panel, detail = PIPELINE.funding_panel(index, ["A"])
    assert panel.A.isna().all()
    assert detail[0]["evaluation_status"] == "unverified_pagination"


@pytest.mark.parametrize("kind", ["missing_mark", "invalid_other_input"])
def test_one_unknown_event_invalidates_whole_day_without_dropping_valid_days(daily, kind):
    directory, index = daily
    options = ({"bad_mark_day": "2025-01-03"} if kind == "missing_mark"
               else {"invalid_row_day": "2025-01-03"})
    save_funding(directory, "A", index, **options)
    panel, detail = PIPELINE.funding_panel(index, ["A"])
    assert np.isnan(panel.loc["2025-01-03", "A"])
    assert panel.A.notna().sum() == len(index) - 1
    assert detail[0]["status"] == "missing"


def test_midnight_plus_milliseconds_belongs_to_current_utc_day(daily):
    directory, index = daily
    frame = save_funding(directory, "A", index)
    stamp = pd.Timestamp("2025-01-03", tz="UTC").value // 1_000_000 + 4
    frame.loc[frame.fundingTime == stamp, "fundingRate"] = 0.001
    frame.to_parquet(directory / "funding/A.parquet", index=False)
    panel, _ = PIPELINE.funding_panel(index, ["A"])
    assert panel.loc["2025-01-02", "A"] == pytest.approx(0.03)
    assert panel.loc["2025-01-03", "A"] == pytest.approx(0.12)


def test_lifecycle_excludes_partial_listing_bar_and_keeps_only_live_exit_open():
    index = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    registry = {"symbols": [{"symbol": "A",
                 "onboardDate": pd.Timestamp("2025-01-01 12:00", tz="UTC").value // 1_000_000,
                 "deliveryDate": pd.Timestamp("2025-01-04 12:00", tz="UTC").value // 1_000_000}]}
    live_bar, live_open = PIPELINE.lifecycle_masks(index, ["A"], registry)
    assert live_bar.A.tolist() == [False, True, True, False, False]
    assert live_open.A.tolist() == [False, True, True, True, False]


def test_lifecycle_midnight_delivery_rejects_delivery_instant_open():
    index = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    registry = {"symbols": [{"symbol": "A",
                 "onboardDate": index[0].value // 1_000_000,
                 "deliveryDate": index[3].value // 1_000_000}]}
    live_bar, live_open = PIPELINE.lifecycle_masks(index, ["A"], registry)
    assert live_bar.A.tolist() == [True, True, True, False, False]
    assert live_open.A.tolist() == [True, True, True, False, False]


@pytest.mark.parametrize("unknown_is_held", [True, False])
def test_persisted_unknown_funding_reaches_engine_only_for_held_assets(daily, unknown_is_held):
    directory, index = daily
    save_funding(directory, "A", index)
    save_funding(directory, "B", index, bad_mark_day="2025-01-03" if unknown_is_held else None)
    save_funding(directory, "C", index, bad_mark_day=None if unknown_is_held else "2025-01-03")
    funding, _ = PIPELINE.funding_panel(index, ["A", "B", "C"])
    signal = pd.DataFrame({"A": -1.0, "B": 1.0, "C": 999.0}, index=index)
    opens = pd.DataFrame(100.0, index=index, columns=signal.columns)
    mask = pd.DataFrame({"A": True, "B": True, "C": False}, index=index)
    result = PIPELINE.evaluate_factor(signal, opens, opens, mask, funding,
                                     horizons=(1,), entry_lag=2, min_assets=2,
                                     splits={"dev": ("2025-01-01", "2025-01-07")})
    row = result["periods"].set_index("signal_time").loc[index[0]]
    assert row.entry_time == index[2]
    assert row.n_positions == 2
    if unknown_is_held:
        assert row.status == "missing_funding"
        assert np.isnan(row.net) and np.isfinite(row.net_ex_funding)
        assert np.isnan(result["metrics"].iloc[0].net_bp)
    else:
        assert row.status == "ok" and np.isfinite(row.net)
