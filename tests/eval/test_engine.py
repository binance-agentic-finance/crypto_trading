"""Accounting and timing invariants for the daily evaluation engine."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[2] / "cyqnt_trd" / "eval" / "engine.py"
SPEC = importlib.util.spec_from_file_location("alpha101_v2_engine", PATH)
ENGINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENGINE)


@pytest.fixture
def panel():
    idx = pd.date_range("2024-01-01", periods=12, tz="UTC", freq="D")
    signal = pd.DataFrame({"A": -1.0, "B": 1.0}, index=idx)
    opens = pd.DataFrame(100.0, index=idx, columns=signal.columns)
    # One complete dev cycle: short A (-10%), long B (+10%) => positive IC.
    opens.loc[idx[3], ["A", "B"]] = [90.0, 110.0]
    # Validation signal Jan 5 enters Jan 7, exits Jan 8.
    opens.loc[idx[6], ["A", "B"]] = [100.0, 200.0]
    opens.loc[idx[7], ["A", "B"]] = [120.0, 220.0]
    mask = pd.DataFrame(True, index=idx, columns=signal.columns)
    funding = pd.DataFrame(0.0, index=idx, columns=signal.columns)
    splits = {"dev": ("2024-01-01", "2024-01-04"),
              "val": ("2024-01-05", "2024-01-12")}
    return signal, opens, mask, funding, splits


def run(panel, **kwargs):
    signal, opens, mask, funding, splits = panel
    options = dict(horizons=(1,), min_assets=2, splits=splits, funding=funding)
    options.update(kwargs)
    return ENGINE.evaluate_factor(signal, opens, opens.copy(), mask, **options)


def cycle(result, date="2024-01-05"):
    table = result["periods"]
    return table.loc[table.signal_time == pd.Timestamp(date, tz="UTC")].iloc[0]


def metric(result, split="val"):
    return result["metrics"].set_index("split").loc[split]


def test_simple_pnl_fixed_units_drift_cost_and_signed_funding(panel):
    signal, opens, mask, funding, splits = panel
    funding.loc["2024-01-06"] = [1000.0, 1000.0]  # Before entry: excluded.
    funding.loc["2024-01-07"] = [1.0, 4.0]
    funding.loc["2024-01-08"] = [1000.0, 1000.0]  # At exit: excluded.
    result = run(panel)
    row = cycle(result)
    assert row.entry_time == pd.Timestamp("2024-01-07", tz="UTC")
    assert row.exit_time == pd.Timestamp("2024-01-08", tz="UTC")
    assert row.sign == 1
    assert row.long_gross == pytest.approx(0.05)
    assert row.short_gross == pytest.approx(-0.10)
    assert row.gross == pytest.approx(-0.05)
    assert row.entry_notional == pytest.approx(1.0)
    assert row.exit_notional == pytest.approx(1.15)
    assert row.turnover == pytest.approx(2.15)
    assert row.trading_cost == pytest.approx(2.15 * 6.5 / 10000)
    # Units are -0.5/100 and +0.5/200, so shorts receive 0.005, longs pay 0.01.
    assert row.funding == pytest.approx(0.005)
    assert row.net == pytest.approx(-0.05 - 2.15 * 6.5 / 10000 - 0.005)
    assert row.gross != pytest.approx(-0.5 * np.log(1.2) + 0.5 * np.log(1.1))


def test_dev_direction_is_frozen_when_validation_relation_reverses(panel):
    result = run(panel)
    assert result["metrics"].sign.tolist() == [1, 1]
    assert result["metrics"].direction_frozen.all()
    val_ic = result["ic"].set_index("signal_time").loc[pd.Timestamp("2024-01-05", tz="UTC")]
    assert val_ic.ic_raw == pytest.approx(-1.0)
    assert val_ic.ic == pytest.approx(-1.0)


def test_negative_dev_sign_is_used_for_returns_and_funding(panel):
    signal, opens, mask, funding, splits = panel
    opens.loc["2024-01-04", ["A", "B"]] = [110.0, 90.0]
    funding.loc["2024-01-07"] = [1.0, 4.0]
    result = run(panel)
    row = cycle(result)
    assert row.sign == -1
    assert row.gross == pytest.approx(0.05)
    assert row.funding == pytest.approx(-0.005)
    assert result["metrics"].sign.tolist() == [-1, -1]


def test_purge_requires_exit_within_split_not_only_signal_date(panel):
    result = run(panel)
    dev = metric(result, "dev")
    assert dev.n_ic == 1
    assert dev.n_periods == 1
    assert dev.n_purged == 3
    assert cycle(result, "2024-01-02").status == "purged_label"
    assert cycle(result, "2024-01-01").exit_time == pd.Timestamp("2024-01-04", tz="UTC")


def test_entry_lag_one_is_explicit_optimistic_boundary(panel):
    result = run(panel, entry_lag=1)
    row = cycle(result)
    assert row.entry_time == row.available_at
    assert result["config"]["additional_wait_days_after_bar_close"] == 0


def test_missing_held_target_invalidates_whole_cycle_and_official_mean(panel):
    signal, opens, mask, funding, splits = panel
    opens.loc["2024-01-08", "A"] = np.nan
    result = run(panel)
    row = cycle(result)
    assert row.status == "missing_price"
    assert np.isnan(row.gross) and np.isnan(row.net)
    assert metric(result).n_invalid_price >= 1
    assert np.isnan(metric(result).gross_bp)
    assert np.isfinite(metric(result).gross_available_bp)


def test_missing_funding_does_not_masquerade_as_zero_funding(panel):
    signal, opens, mask, funding, splits = panel
    funding.loc["2024-01-07", "A"] = np.nan
    result = run(panel)
    row = cycle(result)
    assert row.status == "missing_funding"
    assert np.isfinite(row.gross) and np.isfinite(row.net_ex_funding)
    assert np.isnan(row.funding) and np.isnan(row.net)
    assert np.isnan(metric(result).net_bp)


def test_no_funding_input_only_reports_cost_adjusted_pnl(panel):
    result = run(panel, funding=None)
    row = cycle(result)
    assert np.isfinite(row.net_ex_funding)
    assert np.isnan(row.net)
    assert not result["config"]["funding_supplied"]


def test_transition_to_flat_keeps_exit_charge_and_zero_cycle(panel):
    signal, opens, mask, funding, splits = panel
    signal.loc["2024-01-06"] = 0.0
    result = run(panel)
    before = cycle(result, "2024-01-05")
    flat = cycle(result, "2024-01-06")
    assert before.exit_time == flat.entry_time
    assert before.trading_cost == pytest.approx((1.0 + 1.15) * 6.5 / 10000)
    assert flat.status == "flat"
    assert flat.net == 0 and flat.trading_cost == 0
    assert metric(result).n_flat == 1
    assert metric(result).n_periods == metric(result).n_active + metric(result).n_flat


def test_masked_asset_does_not_change_ranks_or_require_target(panel):
    signal, opens, mask, funding, splits = panel
    signal["excluded"] = 999999.0
    opens["excluded"] = np.nan
    funding["excluded"] = np.nan
    mask["excluded"] = False
    row = cycle(run(panel))
    assert row.status == "ok"
    assert row.n_positions == 2
    assert row.gross == pytest.approx(-0.05)


def test_nonoverlapping_cycles_and_break_even_identity(panel):
    result = run(panel, horizons=(3,))
    periods = result["periods"]
    valid = periods.loc[periods.label_valid]
    if len(valid) > 1:
        assert (valid.entry_time.iloc[1:].to_numpy() >= valid.exit_time.iloc[:-1].to_numpy()).all()
    for _, row in result["metrics"].iterrows():
        if row.turnover > 0 and np.isfinite(row.net_bp):
            assert row.net_bp == pytest.approx((row.breakeven_cost_bps - 6.5) * row.turnover)


def test_hac_preserves_missing_calendar_gaps():
    x = np.array([1.0, np.nan, 3.0, 4.0])
    centered = np.array([-5 / 3, 0.0, 1 / 3, 4 / 3])
    variance_sum = centered @ centered + centered[1:] @ centered[:-1]
    expected = (8 / 3) / (np.sqrt(variance_sum) / 3)
    assert ENGINE._hac_mean_t(x, 1) == pytest.approx(expected)
    assert ENGINE._hac_mean_t(x, 1) != pytest.approx(ENGINE._hac_mean_t(x[np.isfinite(x)], 1))


def test_irregular_daily_grid_and_overlapping_splits_are_rejected(panel):
    signal, opens, mask, funding, splits = panel
    with pytest.raises(ValueError, match="daily UTC"):
        ENGINE.evaluate_factor(signal.drop(signal.index[2]), opens, opens, mask)
    with pytest.raises(ValueError, match="overlap"):
        run(panel, splits={"dev": ("2024-01-01", "2024-01-05"),
                           "val": ("2024-01-05", "2024-01-12")})
