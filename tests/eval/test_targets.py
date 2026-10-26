"""Window alignment and no-future-information contracts for four-target diagnostics."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


from cyqnt_trd.eval.bundle import Panel, PRICE_FIELDS  # noqa: E402
from cyqnt_trd.eval.targets import build_targets, TARGET_NAMES  # noqa: E402


def _panel(n=140):
    rng = np.random.default_rng(194)
    index = pd.date_range("2023-01-01", periods=n, tz="UTC")
    market = rng.normal(0.0002, 0.015, n)
    increments = np.column_stack([market, 1.5 * market + rng.normal(0, 0.008, n),
                                  -0.6 * market + rng.normal(0, 0.012, n)])
    close = pd.DataFrame(100 * np.exp(increments.cumsum(axis=0)), index=index,
                         columns=["BTCUSDT", "ALTUSDT", "OTHERUSDT"])
    opens = close.shift(1).fillna(100) * 1.001
    high = pd.DataFrame(np.maximum(opens, close) * 1.01, index=index, columns=close.columns)
    low = pd.DataFrame(np.minimum(opens, close) * 0.99, index=index, columns=close.columns)
    ones = pd.DataFrame(1.0, index=index, columns=close.columns)
    return Panel(open=opens, high=high, low=low, close=close, volume=ones,
                 quote_volume=ones * 100, funding=ones * 0,
                 mask=ones.astype(bool), meta={})


def test_all_targets_use_the_same_entry_and_exit():
    panel = _panel()
    result = build_targets(panel, h=3, entry_lag=2)
    targets, estimates = result["targets"], result["estimates"]
    date, symbol = panel.index[90], "ALTUSDT"
    expected_raw = np.log(panel.open.iloc[95][symbol] / panel.open.iloc[92][symbol])
    expected_market = np.log(panel.open.iloc[95]["BTCUSDT"] / panel.open.iloc[92]["BTCUSDT"])
    beta = estimates["beta"].loc[date, symbol]
    assert targets["raw_rtf"].loc[date, symbol] == pytest.approx(expected_raw)
    assert targets["normed_rtf"].loc[date, symbol] == pytest.approx(expected_raw / estimates["sigma_raw"].loc[date, symbol])
    expected_res = expected_raw - beta * expected_market
    assert targets["res_rtf"].loc[date, symbol] == pytest.approx(expected_res)
    assert targets["normed_res_rtf"].loc[date, symbol] == pytest.approx(expected_res / estimates["sigma_res"].loc[date, symbol])
    assert all(target.iloc[-5:].isna().all().all() for target in targets.values())


def test_future_prices_cannot_change_historical_beta_or_volatility():
    panel = _panel()
    original = build_targets(panel, 3)
    cutoff = 90
    changed = {}
    for name in PRICE_FIELDS:
        frame = getattr(panel, name).copy()
        if name in ("open", "high", "low", "close"):
            frame.iloc[cutoff + 1:] *= 4.0
        changed[name] = frame
    altered = build_targets(replace(panel, **changed), 3)
    for field in ("beta", "sigma_raw", "sigma_res", "raw_observations", "paired_observations"):
        pd.testing.assert_frame_equal(original["estimates"][field].iloc[:cutoff + 1],
                                      altered["estimates"][field].iloc[:cutoff + 1])
    # The changed future price legitimately changes the forward label, proving
    # that the perturbation reached the target while leaving its estimates alone.
    assert original["targets"]["raw_rtf"].iloc[87, 1] != altered["targets"]["raw_rtf"].iloc[87, 1]


def test_risk_scales_use_empirical_same_horizon_history_not_sqrt_scaling():
    panel = _panel()
    result = build_targets(panel, 5)
    historical = np.log(panel.close).diff(5)
    window = historical.iloc[31:91]
    actual = result["estimates"]["sigma_raw"].iloc[90]
    pd.testing.assert_series_equal(actual, window.std(ddof=1), check_names=False)
    assert not np.allclose(actual, np.log(panel.close).diff().iloc[31:91].std() * np.sqrt(5))


def test_beta_and_residual_scale_use_only_matching_finite_pairs():
    panel = _panel()
    close = panel.close.copy()
    close.loc[close.index[40:48], "ALTUSDT"] = np.nan
    panel = replace(panel, close=close)
    result = build_targets(panel, 3, lookback=60, min_samples=30)
    history = np.log(panel.close).diff(3).iloc[31:91][["ALTUSDT", "BTCUSDT"]].dropna()
    y, market = history["ALTUSDT"], history["BTCUSDT"]
    covariance, variance_m = y.cov(market), market.var(ddof=1)
    assert result["estimates"]["beta"].iloc[90]["ALTUSDT"] == pytest.approx(covariance / variance_m)
    expected_sigma = np.sqrt(y.var(ddof=1) - covariance ** 2 / variance_m)
    assert result["estimates"]["sigma_res"].iloc[90]["ALTUSDT"] == pytest.approx(expected_sigma)
    assert result["estimates"]["paired_observations"].iloc[90]["ALTUSDT"] == len(history)


def test_common_mask_and_coverage_exclude_every_unavailable_target():
    panel = _panel()
    mask = panel.mask.copy()
    mask.loc[mask.index[80:85], "ALTUSDT"] = False
    result = build_targets(replace(panel, mask=mask), 3)
    expected = mask.copy()
    for name in TARGET_NAMES:
        target = result["targets"][name]
        expected &= np.isfinite(target)
        valid_cells = int((mask & np.isfinite(target)).to_numpy().sum())
        assert result["coverage"][name]["valid_cells"] == valid_cells
        assert result["coverage"][name]["fraction"] == pytest.approx(valid_cells / mask.to_numpy().sum())
        assert target.where(~mask).isna().all().all()
    pd.testing.assert_frame_equal(result["common_mask"], expected)
    assert not result["common_mask"]["BTCUSDT"].any()  # benchmark has zero residual scale
    assert result["common_mask"]["ALTUSDT"].any()


def test_missing_benchmark_keeps_raw_targets_without_choosing_a_substitute():
    result = build_targets(_panel(), 3, benchmark_symbol="ABSENTUSDT")
    assert result["targets"]["raw_rtf"].notna().any().any()
    assert result["targets"]["normed_rtf"].notna().any().any()
    for name in ("res_rtf", "normed_res_rtf"):
        assert result["targets"][name].isna().all().all()
        assert result["availability"][name]["status"] == "UNAVAILABLE"
    assert not result["common_mask"].any().any()
    assert result["metadata"]["benchmark_symbol"] == "ABSENTUSDT"


def test_constant_benchmark_does_not_invent_zero_beta_or_residual_targets():
    panel = _panel()
    frames = {name: getattr(panel, name).copy() for name in PRICE_FIELDS}
    for name in ("open", "high", "low", "close"):
        frames[name]["BTCUSDT"] = 100.0
    result = build_targets(replace(panel, **frames), 3)
    assert result["estimates"]["beta"].isna().all().all()
    assert result["estimates"]["sigma_res"].isna().all().all()
    assert result["targets"]["res_rtf"].isna().all().all()
    assert result["targets"]["normed_res_rtf"].isna().all().all()
    assert result["targets"]["normed_rtf"]["BTCUSDT"].isna().all()
    assert result["targets"]["normed_rtf"]["ALTUSDT"].notna().any()


def test_insufficient_history_remains_unavailable_and_metadata_is_strict_json():
    import json
    result = build_targets(_panel(35), 3)
    assert result["targets"]["raw_rtf"].notna().any().any()
    for name in ("normed_rtf", "res_rtf", "normed_res_rtf"):
        assert result["targets"][name].isna().all().all()
        assert result["availability"][name]["status"] == "UNAVAILABLE"
    assert result["metadata"]["min_samples"] == 40
    json.dumps({key: result[key] for key in ("coverage", "availability", "metadata")}, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"h": 0}, {"h": True}, {"h": 1.5},
                                  {"h": 3, "entry_lag": 0}, {"h": 3, "lookback": 20},
                                  {"h": 3, "min_samples": 1}])
def test_invalid_target_windows_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_targets(_panel(), **kwargs)
