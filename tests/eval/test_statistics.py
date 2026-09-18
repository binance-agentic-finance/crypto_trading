"""Date weighting, missing-calendar gaps, and whole-date bootstrap contracts."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest



from cyqnt_trd.eval.statistics import (  # noqa: E402
    daily_cross_sectional,
    hac_mean_ci,
    moving_block_indices,
    summarize_relation,
    summarize_time_mean,
    weighted_panel_relation,
)


def frames(n_dates=40, n_assets=6):
    rng = np.random.default_rng(71)
    index = pd.date_range("2024-01-01", periods=n_dates, freq="D", tz="UTC")
    columns = [f"asset_{i}" for i in range(n_assets)]
    x = pd.DataFrame(rng.normal(size=(n_dates, n_assets)), index=index, columns=columns)
    y = 0.5 * x + pd.DataFrame(rng.normal(size=x.shape), index=index, columns=columns)
    return x, y, pd.DataFrame(True, index=index, columns=columns)


def test_daily_pearson_and_tied_spearman_match_pandas():
    x, y, mask = frames()
    x.iloc[0] = [1, 1, 3, 3, 5, 6]
    y.iloc[0] = [4, 6, 5, 3, 2, 1]
    actual = daily_cross_sectional(x, y, mask)
    assert actual.loc[x.index[0], "rank_ic"] == pytest.approx(x.iloc[0].corr(y.iloc[0], method="spearman"))
    assert actual.loc[x.index[0], "pearson_ic"] == pytest.approx(x.iloc[0].corr(y.iloc[0]))


def test_ic_is_equal_date_not_pooled_asset_weighted():
    x, y, mask = frames(n_dates=2)
    x.iloc[:] = [1, 2, 3, 4, 5, 6]
    y.iloc[0] = x.iloc[0]
    y.iloc[1] = -x.iloc[1]
    mask.iloc[1, 2:] = False
    result = summarize_relation(x, y, mask, min_assets=2, block_length=1, n_bootstrap=0)
    assert result["rank_ic"] == pytest.approx(0)
    assert result["pearson_ic"] == pytest.approx(0)
    assert result["n_dates"] == 2
    assert result["n_pairs"] == 8
    assert result["n_assets_mean"] == 4


def test_pooled_moments_match_explicit_equal_date_pair_weights():
    x, y, mask = frames(n_dates=4)
    mask.iloc[0, 1:] = False
    mask.iloc[1, 3:] = False
    weights = mask.div(mask.sum(axis=1), axis=0).to_numpy()
    weights /= weights.sum()
    xx, yy = x.to_numpy(), y.to_numpy()
    mx, my = (weights * xx).sum(), (weights * yy).sum()
    cov = (weights * (xx - mx) * (yy - my)).sum()
    vx, vy = (weights * (xx - mx) ** 2).sum(), (weights * (yy - my) ** 2).sum()
    result = weighted_panel_relation(x, y, mask)
    assert result["corr_pooled"] == pytest.approx(cov / np.sqrt(vx * vy))
    assert result["slope"] == pytest.approx(cov / vx)
    assert "not IC" in result["method"]


def test_narrow_slice_keeps_ic_unknown_and_labels_pooled_relation():
    x, y, mask = frames(n_assets=2)
    result = summarize_relation(x, y, mask, min_assets=5, block_length=5, n_bootstrap=30)
    assert np.isnan(result["rank_ic"])
    assert np.isnan(result["pearson_ic"])
    assert result["n_dates"] == 0
    assert result["n_dates_pooled"] == 40
    assert np.isfinite(result["corr_pooled"])
    assert np.isnan(result["ci"]["rank_ic"]).all()
    assert np.isfinite(result["ci"]["corr_pooled"]).all()


def test_constant_cross_section_has_no_ic_even_with_many_assets():
    x, y, mask = frames()
    x.iloc[7] = 1.0
    result = daily_cross_sectional(x, y, mask)
    assert result.n_assets.iloc[7] == 6
    assert np.isnan(result.rank_ic.iloc[7])
    assert np.isnan(result.pearson_ic.iloc[7])


def test_frozen_sign_reverses_correlations_and_slope():
    x, y, mask = frames()
    a = summarize_relation(x, y, mask, sign=1, block_length=5, n_bootstrap=30)
    b = summarize_relation(x, y, mask, sign=-1, block_length=5, n_bootstrap=30)
    for field in ("rank_ic", "pearson_ic", "corr_pooled", "slope"):
        assert b[field] == pytest.approx(-a[field])
        assert b["ci"][field] == pytest.approx([-a["ci"][field][1], -a["ci"][field][0]])


def test_moving_blocks_preserve_order_and_share_dates_across_assets():
    indices = moving_block_indices(12, block_length=3, n_bootstrap=20, seed=5)
    assert indices.shape == (20, 12)
    assert np.array_equal(indices, moving_block_indices(12, 3, 20, 5))
    assert (np.diff(indices.reshape(20, 4, 3), axis=2) == 1).all()
    x, y, mask = frames(n_dates=12)
    summary = summarize_relation(x, y, mask, block_length=3, bootstrap_indices=indices)
    daily = daily_cross_sectional(x, y, mask)
    expected = np.quantile(daily.rank_ic.to_numpy()[indices].mean(axis=1), [0.025, 0.975])
    assert summary["ci"]["rank_ic"] == pytest.approx(expected)
    assert summary["n_bootstrap"] == 20


def test_pooled_bootstrap_preserves_date_weighting_with_varying_asset_counts():
    x, y, mask = frames(n_dates=12)
    mask.iloc[0, 1:] = False
    mask.iloc[1, 3:] = False
    indices = moving_block_indices(12, 3, 20, 5)
    summary = summarize_relation(x, y, mask, block_length=3, bootstrap_indices=indices)
    corr_draws, slope_draws = [], []
    for draw in indices:
        multiplicity = np.bincount(draw, minlength=len(x))
        weights = mask.div(mask.sum(axis=1), axis=0).to_numpy() * multiplicity[:, None]
        weights /= weights.sum()
        xx, yy = x.to_numpy(), y.to_numpy()
        dx, dy = xx - (weights * xx).sum(), yy - (weights * yy).sum()
        cov = (weights * dx * dy).sum()
        vx, vy = (weights * dx ** 2).sum(), (weights * dy ** 2).sum()
        corr_draws.append(cov / np.sqrt(vx * vy))
        slope_draws.append(cov / vx)
    assert summary["ci"]["corr_pooled"] == pytest.approx(np.quantile(corr_draws, [0.025, 0.975]))
    assert summary["ci"]["slope"] == pytest.approx(np.quantile(slope_draws, [0.025, 0.975]))


def test_missing_dates_stay_in_bootstrap_and_hac_grid():
    values = pd.Series([1.0, 3.0, 4.0], index=pd.to_datetime(
        ["2024-01-01", "2024-01-03", "2024-01-04"], utc=True))
    result = hac_mean_ci(values, lags=1)
    influence = np.array([-5 / 3, 0, 1 / 3, 4 / 3])
    se = np.sqrt(influence @ influence + influence[1:] @ influence[:-1]) / 3
    assert result["std_error"] == pytest.approx(se)
    assert result["n_dates"] == 3
    assert result["n_dates_total"] == 4
    assert result["std_error"] != pytest.approx(hac_mean_ci(values.to_numpy(), lags=1)["std_error"])
    indices = moving_block_indices(4, 1, 30, 7)
    draws = np.nanmean(np.array([1.0, np.nan, 3.0, 4.0])[indices], axis=1)
    expected = np.nanquantile(draws, [0.025, 0.975])
    summary = summarize_time_mean(values, block_length=1, bootstrap_indices=indices)
    assert [summary["ci_low"], summary["ci_high"]] == pytest.approx(expected)


def test_missing_panel_dates_are_restored_before_resampling():
    x, y, mask = frames(n_dates=12)
    dropped = x.index[5]
    daily = daily_cross_sectional(x.drop(dropped), y.drop(dropped), mask.drop(dropped))
    assert len(daily) == 12
    assert np.isnan(daily.loc[dropped, "rank_ic"])
    assert daily.loc[dropped, "n_assets"] == 0
    result = summarize_relation(x.drop(dropped), y.drop(dropped), mask.drop(dropped),
                                block_length=3, n_bootstrap=30)
    assert result["n_dates_total"] == 12
    assert result["n_dates"] == 11


def test_fewer_than_two_blocks_has_unknown_bootstrap_interval():
    result = summarize_time_mean(np.arange(9), block_length=5)
    assert np.isnan(result["ci_low"])
    assert np.isnan(result["ci_high"])
    assert "two complete calendar blocks" in result["reason"]
    with pytest.raises(ValueError, match="two complete calendar blocks"):
        moving_block_indices(9, 5)


def test_zero_activation_counts_as_observed_and_nan_does_not():
    result = summarize_time_mean([0, 0, 0, 1, np.nan, 0], block_length=2, n_bootstrap=50)
    assert result["mean"] == pytest.approx(0.2)
    assert result["n_dates"] == 5
    assert result["n_dates_total"] == 6
    assert np.isfinite(result["ci_low"])


def test_all_missing_is_unknown_in_every_relation():
    x, y, mask = frames()
    mask.iloc[:] = False
    result = summarize_relation(x, y, mask, block_length=5, n_bootstrap=30)
    assert result["n_dates"] == result["n_pairs"] == result["n_dates_pooled"] == 0
    for name in ("rank_ic", "pearson_ic", "corr_pooled", "slope"):
        assert np.isnan(result[name])
        assert np.isnan(result["ci"][name]).all()


def test_pooled_relation_resists_large_constant_offsets():
    x, y, mask = frames()
    original = weighted_panel_relation(x, y, mask)
    translated = weighted_panel_relation(x + 1e12, y + 1e12, mask)
    assert translated["corr_pooled"] == pytest.approx(original["corr_pooled"], abs=1e-5)
    assert translated["slope"] == pytest.approx(original["slope"], abs=1e-5)


@pytest.mark.parametrize("options", [{"block_length": 0}, {"block_length": True},
                                     {"n_bootstrap": -1}, {"confidence": 1}])
def test_invalid_statistics_options_are_rejected(options):
    with pytest.raises(ValueError):
        summarize_time_mean(np.arange(20), **options)


def test_misaligned_panels_are_rejected_instead_of_silently_joined():
    x, y, mask = frames()
    with pytest.raises(ValueError, match="identical"):
        summarize_relation(x, y.iloc[::-1], mask)
