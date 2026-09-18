"""Counterexamples for missing-signal accounting and chronology validation."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[2] / "cyqnt_trd" / "eval" / "engine.py"
SPEC = importlib.util.spec_from_file_location("factor_engine_audit", PATH)
ENGINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENGINE)


@pytest.fixture
def panel():
    index = pd.date_range("2024-01-01", periods=16, freq="D", tz="UTC")
    signal = pd.DataFrame({"A": -1.0, "B": 1.0}, index=index)
    opens = pd.DataFrame({"A": 100 * 0.99 ** np.arange(len(index)),
                          "B": 100 * 1.01 ** np.arange(len(index))}, index=index)
    mask = pd.DataFrame(True, index=index, columns=signal.columns)
    funding = pd.DataFrame(0.0, index=index, columns=signal.columns)
    return signal, opens, mask, funding


def evaluate(panel, **kwargs):
    signal, opens, mask, funding = panel
    options = {"horizons": (1,), "min_assets": 2, "funding": funding,
               "splits": {"dev": ("2024-01-01", "2024-01-06"),
                          "val": ("2024-01-07", "2024-01-16")}}
    options.update(kwargs)
    return ENGINE.evaluate_factor(signal, opens, opens.copy(), mask, **options)


@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_insufficient_signals_do_not_become_free_flat_cycles(panel, missing):
    panel[0].loc["2024-01-08", "A"] = missing
    result = evaluate(panel)
    row = result["periods"].set_index("signal_time").loc[pd.Timestamp("2024-01-08", tz="UTC")]
    assert row.status == "missing_signal"
    assert not row.signal_complete
    assert np.isnan(row.net)
    assert np.isnan(row.net_ex_funding)
    assert np.isnan(row.gross)
    metric = result["metrics"].set_index("split").loc["val"]
    assert metric.n_missing_signal == 1
    assert metric.n_flat == 0
    assert metric.n_complete == metric.n_periods - 1
    assert np.isnan(metric.net_bp)
    assert np.isnan(metric.sharpe_net)
    assert np.isfinite(metric.net_available_bp)


def test_observed_constant_signal_remains_a_real_flat_cycle(panel):
    panel[0].loc["2024-01-08"] = 0
    result = evaluate(panel)
    row = result["periods"].set_index("signal_time").loc[pd.Timestamp("2024-01-08", tz="UTC")]
    assert row.status == "flat"
    assert row.signal_complete and row.net == 0
    metric = result["metrics"].set_index("split").loc["val"]
    assert metric.n_flat == 1
    assert metric.n_missing_signal == 0
    assert np.isfinite(metric.net_bp)


def test_insufficient_universe_is_reported_as_unknown(panel):
    panel[2].loc["2024-01-08", "A"] = False
    result = evaluate(panel)
    metric = result["metrics"].set_index("split").loc["val"]
    assert metric.n_missing_signal == 1
    assert np.isnan(metric.net_bp)


def test_future_development_split_cannot_choose_past_validation_direction(panel):
    with pytest.raises(ValueError, match="dev split must precede"):
        evaluate(panel, splits={"val": ("2024-01-01", "2024-01-06"),
                                "dev": ("2024-01-07", "2024-01-16")})


def test_out_of_time_split_must_follow_validation(panel):
    with pytest.raises(ValueError, match="val split must precede"):
        evaluate(panel, splits={"dev": ("2024-01-01", "2024-01-04"),
                                "oot": ("2024-01-05", "2024-01-08"),
                                "val": ("2024-01-09", "2024-01-16")})


@pytest.mark.parametrize("splits", [
    {"dev": (pd.NaT, "2024-01-06")},
    {"dev": ("2024-01-01", pd.NaT)},
    {"dev": "12"},
    {"dev": {"start": "2024-01-01", "end": "2024-01-06"}},
    {"dev": {"2024-01-01", "2024-01-06"}},
    {"dev": ("2024-01-01", "2024-01-06"), "": ("2024-01-07", "2024-01-16")},
    {"dev": ("2024-01-01", "2024-01-06"), 2: ("2024-01-07", "2024-01-16")},
])
def test_malformed_splits_are_rejected(panel, splits):
    with pytest.raises(ValueError):
        evaluate(panel, splits=splits)


@pytest.mark.parametrize("options", [
    {"entry_lag": True}, {"horizons": (True,)}, {"horizons": (1, 1)},
    {"horizons": ()}, {"min_assets": 2.5}, {"min_assets": np.nan},
    {"annualization": np.nan}, {"annualization": np.inf},
    {"annualization": True}, {"hac_lags": False}, {"cost_bps": True},
])
def test_ambiguous_or_nonfinite_parameters_are_rejected(panel, options):
    with pytest.raises(ValueError):
        evaluate(panel, **options)


@pytest.mark.parametrize("bad_mask", ["False", 2, -1, np.nan])
def test_mask_requires_explicit_boolean_membership(panel, bad_mask):
    signal, opens, mask, funding = panel
    mask = mask.astype(object)
    mask.loc["2024-01-08", "A"] = bad_mask
    with pytest.raises(ValueError, match="mask must explicitly"):
        evaluate((signal, opens, mask, funding))


def test_mask_cannot_silently_drop_a_signal_date(panel):
    signal, opens, mask, funding = panel
    with pytest.raises(ValueError, match="mask must explicitly"):
        evaluate((signal, opens, mask.drop(mask.index[7]), funding))


def test_zero_dev_mean_does_not_claim_to_have_selected_a_direction(panel):
    # Exactly two eligible development labels, with opposite rank relationships.
    panel[1].loc["2024-01-03"] = [100.0, 100.0]
    panel[1].loc["2024-01-04"] = [90.0, 110.0]
    panel[1].loc["2024-01-05"] = [100.0, 100.0]
    result = evaluate(panel, splits={"dev": ("2024-01-01", "2024-01-05"),
                                     "val": ("2024-01-06", "2024-01-16")})
    assert result["metrics"].dev_ic_raw_mean.eq(0).all()
    assert not result["metrics"].direction_frozen.any()
    assert result["metrics"].sign.eq(1).all()
    assert result["config"]["engine_protocol"] == ENGINE.ENGINE_PROTOCOL
