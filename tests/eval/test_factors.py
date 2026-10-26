"""Numerical contracts that distinguish the v2 research object from legacy code."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cyqnt_trd.eval import alpha101 as factors


def raw_panel(periods=360, symbols=6, seed=18):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=periods, freq="D", tz="UTC")
    columns = [f"S{i}" for i in range(symbols)]

    def frame(x):
        return pd.DataFrame(x, index=index, columns=columns)

    close = np.exp(rng.normal(2, 0.4, size=(periods, symbols)))
    open_ = close * np.exp(rng.normal(0, 0.15, size=close.shape))
    high = np.maximum(close, open_) * (1 + rng.uniform(0.01, 0.2, close.shape))
    low = np.minimum(close, open_) / (1 + rng.uniform(0.01, 0.2, close.shape))
    base = np.exp(rng.normal(9, 1, close.shape))
    quote = base * (open_ + close) / 2
    raw = {"open": frame(open_), "high": frame(high), "low": frame(low),
           "close": frame(close), "volume": frame(base), "quote_volume": frame(quote)}
    eligible = frame(np.ones_like(close, dtype=bool))
    return raw, eligible


def test_rank_scale_and_neutralize_use_current_eligible_cross_section():
    x = pd.DataFrame([[1., 3., 10_000.], [-2., 2., 99.], [0., 0., 99.]])
    eligible = pd.DataFrame([[True, True, False]] * 3)
    op = factors.Operators(eligible)
    assert op.rank(x).iloc[0].tolist()[:2] == [0.5, 1.0]
    assert op.rank(x)[2].isna().all()
    np.testing.assert_allclose(op.scale(x, 2).iloc[:2].abs().sum(axis=1), 2)
    assert op.scale(x).iloc[2].isna().all()
    assert op.indneutralize(x).iloc[0].tolist()[:2] == [-1.0, 1.0]
    assert op.indneutralize(x)[2].isna().all()


def test_time_rank_extrema_and_decay_have_hand_calculated_values():
    x = pd.DataFrame({"A": [2., 1., 2.]})
    op = factors.Operators(x.notna())
    assert op.ts_rank(x, 3).iloc[-1, 0] == pytest.approx(2.5 / 3)
    assert op.ts_argmax(x, 3).iloc[-1, 0] == 1  # earliest tie, oldest=1
    assert op.ts_argmin(x, 3).iloc[-1, 0] == 2
    assert op.decay_linear(x, 3.9).iloc[-1, 0] == pytest.approx(10 / 6)
    assert op.ts_rank(x, 3).iloc[:2].isna().all().all()
    # The literal paper definition is power; the active formula bases are nonnegative.
    assert op.signedpower(pd.DataFrame([[-2.]]), 2).iloc[0, 0] == 4


def test_missing_windows_and_missing_conditions_never_become_false_defaults():
    x = pd.DataFrame({"A": [1., np.nan, 3.]})
    op = factors.Operators(pd.DataFrame(True, index=x.index, columns=x.columns))
    for function in (op.ts_sum, op.ts_mean, op.ts_rank, op.ts_argmax, op.decay_linear):
        assert function(x, 3).iloc[-1].isna().all()
    condition = op.lt(x, 2.)
    selected = op.choose(condition, 7., -1.)
    assert selected.iloc[0, 0] == 7.
    assert np.isnan(selected.iloc[1, 0])
    assert selected.iloc[2, 0] == -1.
    # Arithmetic identities must not turn an unavailable operand into a signal.
    assert op.power(pd.DataFrame([[1., np.nan]]),
                    pd.DataFrame([[np.nan, 0.]])).isna().all().all()
    const = pd.DataFrame({"A": [1., 1., 1.]})
    assert op.correlation(const, const, 3).iloc[-1].isna().all()


def test_paper_inputs_keep_base_volume_separate_from_dollar_adv():
    raw, eligible = raw_panel(90, 2)
    panel, meta = factors._prepare_panel(raw, eligible, "paper")
    a = factors.FormulaSet(panel, eligible)
    assert_frame_equal(a.v, raw["volume"])
    assert_frame_equal(a.adv(20), raw["quote_volume"].rolling(20).mean())
    assert_frame_equal(a.r, raw["close"] / raw["close"].shift(1) - 1)
    assert_frame_equal(a.vwap, raw["quote_volume"] / raw["volume"])
    assert_frame_equal(a.cap, raw["quote_volume"].rolling(30).mean().shift(1))
    assert meta["normalization"] is None


def test_normalization_is_lagged_and_has_separate_volume_denominators():
    raw, eligible = raw_panel(270, 2)
    panel, meta = factors._prepare_panel(raw, eligible, "normalized")
    row = 250
    np.testing.assert_allclose(panel["close"].iloc[row],
                               raw["close"].iloc[row] / raw["close"].iloc[:row].mean())
    np.testing.assert_allclose(panel["volume"].iloc[row],
                               raw["volume"].iloc[row] / raw["volume"].iloc[:row].mean())
    np.testing.assert_allclose(panel["adv_source"].iloc[row],
                               raw["quote_volume"].iloc[row] / raw["quote_volume"].iloc[:row].mean())
    assert panel["close"].iloc[:250].isna().all().all()
    assert_frame_equal(panel["returns"], raw["close"] / raw["close"].shift(1) - 1)
    assert meta["normalization"]["min_periods"] == 250


def test_eligibility_preserves_raw_history_but_masks_nested_ranks():
    raw, eligible = raw_panel(90, 3)
    eligible.iloc[:60] = False
    panel, _ = factors._prepare_panel(raw, eligible, "paper")
    a = factors.FormulaSet(panel, eligible)
    # Pure time-series correlation is available on first eligible day.
    assert a.a006().where(eligible).iloc[60].notna().all()
    # A time-series of cross-sectional ranks requires eligible ranked observations.
    assert a.a004().iloc[60].isna().all()
    assert a.a004().iloc[68].notna().all()


def test_hourly_normalization_uses_6000_bars_but_cap_proxy_stays_30_bars():
    raw, eligible = raw_panel(6002, 1)
    index = pd.date_range("2022-01-01", periods=6002, freq="h", tz="UTC")
    for frame in raw.values():
        frame.index = index
    eligible.index = index
    panel, meta = factors._prepare_panel(raw, eligible, "normalized")
    assert meta["bars_per_day"] == 24
    assert meta["normalization"]["bars"] == 6000
    assert panel["close"].iloc[5999].isna().all()
    assert panel["close"].iloc[6000].notna().all()
    assert_frame_equal(panel["cap_proxy"], raw["quote_volume"].rolling(30).mean().shift(1))


def test_piecewise_formulas_propagate_missing_condition_inputs():
    raw, eligible = raw_panel(90, 3)
    raw["volume"].iloc[-1, 0] = np.nan
    raw["high"].iloc[-1, 1] = np.nan
    panel, _ = factors._prepare_panel(raw, eligible, "paper")
    a = factors.FormulaSet(panel, eligible)
    assert np.isnan(a.a007().iloc[-1, 0])
    assert np.isnan(a.a023().iloc[-1, 1])


@pytest.fixture(scope="module")
def computed():
    raw, eligible = raw_panel()
    output, metadata = factors.compute_factors(raw, eligible)
    return raw, eligible, output, metadata


def test_all_formulas_return_metadata_and_alpha088_uses_both_branches(computed):
    raw, eligible, output, metadata = computed
    assert set(output) == {f"alpha{i:03d}" for i in range(1, 102)}
    assert sum(bool(v["substitutions"]) for v in metadata["factors"].values()) == 19
    assert metadata["inputs"]["volume_adv_unit_mismatch"] is True
    counts = metadata["alpha088_branch_diagnostic"]
    assert counts["left_strictly_smaller"] > 0
    assert counts["right_strictly_smaller"] > 0
    panel, _ = factors._prepare_panel(raw, eligible, "paper")
    p1, p2 = factors.FormulaSet(panel, eligible).alpha088_branches()
    assert_frame_equal(output["alpha088"], np.minimum(p1, p2))
    # Audit metadata can be saved without custom encoders or nonstandard NaN tokens.
    json.dumps(metadata, allow_nan=False)


def test_all_factor_histories_are_unchanged_when_future_rows_are_added(computed):
    raw, eligible, output, _ = computed
    end = 180
    prefix = {name: value.iloc[:end].copy() for name, value in raw.items()}
    shortened, _ = factors.compute_factors(prefix, eligible.iloc[:end])
    for name, frame in shortened.items():
        assert_frame_equal(frame, output[name].iloc[:end], atol=1e-12, rtol=1e-12)
