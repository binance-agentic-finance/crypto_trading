"""Data identity and timing changes must invalidate a previously measured floor."""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from cyqnt_trd.eval import engine  # noqa: E402
from cyqnt_trd.eval.bundle import PRICE_FIELDS, Panel, load_bundle, to_long  # noqa: E402
from cyqnt_trd.eval.provenance import (calibration_contract, panel_fingerprint,
                                    validate_calibration)  # noqa: E402

FIELDS = (*PRICE_FIELDS, "funding", "mask")
SETTINGS = {"primary_h": 3, "cost_bps": 6.5, "entry_lag": 2,
            "splits": {"dev": ("2024-01-01", "2024-01-04"),
                       "val": ("2024-01-05", "2024-01-08"),
                       "oot": ("2024-01-09", "2024-01-12")}}


@pytest.fixture
def panel():
    index = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")

    def frame(value):
        return pd.DataFrame(value, index=index, columns=["A", "B", "C"])

    return Panel(open=frame(100.0), high=frame(110.0), low=frame(90.0), close=frame(100.0),
                 volume=frame(10.0), quote_volume=frame(1000.0), funding=frame(0.0),
                 mask=frame(True), meta={"pool": "example"})


def transform_frames(panel, transform):
    return replace(panel, **{name: transform(getattr(panel, name).copy()) for name in FIELDS})


@pytest.mark.parametrize("kind", ["duplicate_date", "missing_day", "descending", "midday", "local_midnight"])
def test_panel_rejects_ambiguous_or_irregular_daily_axis(panel, kind):
    def transform(frame):
        if kind == "missing_day":
            return frame.drop(frame.index[3])
        if kind == "descending":
            return frame.iloc[::-1]
        if kind == "duplicate_date":
            frame.index = frame.index[:1].append(frame.index[:-1])
        elif kind == "midday":
            frame.index += pd.Timedelta(hours=12)
        elif kind == "local_midnight":
            frame.index = frame.index.tz_localize(None).tz_localize("Asia/Shanghai")
        return frame

    with pytest.raises(ValueError, match="index|daily UTC"):
        transform_frames(panel, transform).validate()


def test_panel_rejects_empty_axes_and_duplicate_symbols(panel):
    for transform in (lambda f: f.iloc[:0], lambda f: f.iloc[:, :0],
                      lambda f: f.rename(columns={"B": "A"})):
        with pytest.raises(ValueError, match="nonempty|symbol"):
            transform_frames(panel, transform).validate()


@pytest.mark.parametrize("field,value", [("open", 0.0), ("close", -1.0), ("high", np.inf),
                                         ("funding", np.inf), ("volume", -1.0),
                                         ("quote_volume", -1.0), ("high", 95.0)])
def test_panel_rejects_invalid_observed_prices_and_flows(panel, field, value):
    frame = getattr(panel, field).copy()
    frame.iloc[0, 0] = value
    with pytest.raises(ValueError, match="positive|infinite|nonnegative|OHLC"):
        replace(panel, **{field: frame}).validate()


def test_missing_observations_remain_unknown_instead_of_becoming_zero(panel):
    missing = panel.close.copy()
    missing.iloc[0, 0] = np.nan
    funding = panel.funding.copy()
    funding.iloc[0, 0] = np.nan
    changed = replace(panel, close=missing, funding=funding)
    changed.validate()
    assert np.isnan(changed.close.iloc[0, 0]) and np.isnan(changed.funding.iloc[0, 0])


@pytest.mark.parametrize("kind", ["numeric", "string", "nullable_unknown"])
def test_panel_rejects_mask_truthiness_coercion(panel, kind):
    mask = panel.mask.astype(int if kind == "numeric" else str if kind == "string" else "boolean")
    if kind == "nullable_unknown":
        mask.iloc[0, 0] = pd.NA
    with pytest.raises(ValueError, match="mask must be boolean"):
        replace(panel, mask=mask).validate()


@pytest.mark.parametrize("value", [-1.0, 2.0, np.nan, np.inf])
def test_loader_rejects_invalid_eligibility_values_before_boolean_cast(panel, tmp_path, value):
    long = to_long(panel)
    cell = long.index[long.field == "eligible"][0]
    long.loc[cell, "value"] = value
    path = tmp_path / "bundle.parquet"
    long.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="eligible values must be explicit"):
        load_bundle(path)


def test_metadata_cannot_silently_drop_an_asset_from_the_panel(panel, tmp_path):
    path = tmp_path / "bundle.parquet"
    to_long(panel).to_parquet(path, index=False)
    path.with_suffix(".meta.json").write_text(json.dumps({"symbols": ["A", "B"]}))
    with pytest.raises(ValueError, match="symbols must match"):
        load_bundle(path)


def test_panel_fingerprint_is_stable_across_storage_and_descriptive_metadata(panel, tmp_path):
    path = tmp_path / "bundle.parquet"
    to_long(panel).to_parquet(path, index=False)
    restored = load_bundle(path)
    assert panel_fingerprint(panel) == panel_fingerprint(restored)
    assert panel_fingerprint(panel) == panel_fingerprint(replace(panel, meta={"pool": "renamed"}))
    as_int = replace(panel, volume=panel.volume.astype(int))
    assert panel_fingerprint(panel) == panel_fingerprint(as_int)


@pytest.mark.parametrize("field", FIELDS)
def test_every_input_field_is_bound_to_calibration_even_when_universe_label_is_unchanged(panel, field):
    calibration = {"contract": calibration_contract(panel, **SETTINGS)}
    changed = getattr(panel, field).copy()
    changed.iloc[0, 0] = False if field == "mask" else changed.iloc[0, 0] + 0.5
    mutated = replace(panel, **{field: changed})
    assert mutated.meta == panel.meta
    with pytest.raises(ValueError, match="panel_sha256.*calibrate"):
        validate_calibration(calibration, mutated, **SETTINGS)


def test_missing_funding_and_confirmed_zero_are_different_data_identities(panel):
    calibration = {"contract": calibration_contract(panel, **SETTINGS)}
    unknown = panel.funding.copy()
    unknown.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="panel_sha256"):
        validate_calibration(calibration, replace(panel, funding=unknown), **SETTINGS)


@pytest.mark.parametrize("setting,value", [("primary_h", 5), ("cost_bps", 12.0), ("entry_lag", 1),
                                          ("min_assets", 3), ("hac_lags", 8),
                                          ("annualization", 252.0)])
def test_all_numerical_settings_are_bound_to_calibration(panel, setting, value):
    calibration = {"contract": calibration_contract(panel, **SETTINGS)}
    settings = {**SETTINGS, setting: value}
    with pytest.raises(ValueError, match=setting):
        validate_calibration(calibration, panel, **settings)


def test_split_contract_uses_actual_engine_cutoff_semantics(panel):
    calibration = {"contract": calibration_contract(panel, **SETTINGS)}
    splits = copy.deepcopy(SETTINGS["splits"])
    splits["dev"] = ("2024-01-01T00:00:00Z", "2024-01-04T23:59:59.999999999Z")
    validate_calibration(calibration, panel, **{**SETTINGS, "splits": splits})
    # Midnight timestamps exclude most of the date that a date-only end includes.
    splits["dev"] = ("2024-01-01T00:00:00Z", "2024-01-04T00:00:00Z")
    with pytest.raises(ValueError, match="splits"):
        validate_calibration(calibration, panel, **{**SETTINGS, "splits": splits})


def test_engine_protocol_and_source_changes_invalidate_calibration(panel, monkeypatch, tmp_path):
    calibration = {"contract": calibration_contract(panel, **SETTINGS)}
    with pytest.raises(ValueError, match="engine_protocol"):
        validate_calibration(calibration, panel, **SETTINGS, engine_protocol="changed-protocol")
    source = tmp_path / "engine.py"
    source.write_text("changed numerical implementation\n")
    monkeypatch.setattr(engine, "__file__", str(source))
    with pytest.raises(ValueError, match="engine_sha256"):
        validate_calibration(calibration, panel, **SETTINGS)


def test_legacy_calibration_is_rejected_and_matching_contract_is_json_safe(panel):
    legacy = {"primary_h": 3, "cost_bps": 6.5, "entry_lag": 2}
    with pytest.raises(ValueError, match="lacks a verifiable contract.*calibrate"):
        validate_calibration(legacy, panel, **SETTINGS)
    calibration = {**legacy, "contract": calibration_contract(panel, **SETTINGS)}
    validate_calibration(json.loads(json.dumps(calibration, allow_nan=False)), panel, **SETTINGS)
    calibration["primary_h"] = 5
    with pytest.raises(ValueError, match="primary_h"):
        validate_calibration(calibration, panel, **SETTINGS)
