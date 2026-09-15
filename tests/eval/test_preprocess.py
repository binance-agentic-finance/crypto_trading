"""Factor-writing helpers and the persistence diagnostics behind the cost gate."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval import evaluate, load_bundle, signal_persistence  # noqa: E402
from factor_eval.examples.example_factors import low_volatility, volume_shock  # noqa: E402
from factor_eval.preprocess import (neutralize, winsorize_mad,  # noqa: E402
                                    winsorize_quantile, zscore)


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


@pytest.fixture(scope="module")
def static_signal(panel):
    """The same ranking every day — the low-turnover extreme."""
    order = np.arange(len(panel.symbols), dtype=float)
    return pd.DataFrame(np.tile(order, (len(panel.index), 1)),
                        index=panel.index, columns=panel.symbols)


@pytest.fixture(scope="module")
def noise_signal(panel):
    """Redrawn every day — the high-turnover extreme."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((len(panel.index), len(panel.symbols))),
                        index=panel.index, columns=panel.symbols)


def test_persistence_separates_a_static_ranking_from_noise(static_signal, noise_signal, panel):
    slow = signal_persistence(static_signal, panel.mask, (1, 3))
    fast = signal_persistence(noise_signal, panel.mask, (1, 3))
    assert slow["rank_autocorr"]["lag1"] > .95 and slow["rank_autocorr"]["lag3"] > .95
    assert abs(fast["rank_autocorr"]["lag1"]) < .15
    assert slow["quantile_turnover"]["top"] < .05
    assert fast["quantile_turnover"]["top"] > .3


def test_persistence_ignores_the_head_where_no_earlier_cycle_exists(static_signal, panel):
    out = signal_persistence(static_signal, panel.mask, (5,))
    assert np.isfinite(out["quantile_turnover"]["top"])
    assert out["quantile_turnover"]["top"] < .05


def test_persistence_is_sign_invariant(panel):
    a = signal_persistence(low_volatility(panel), panel.mask, (1, 3))
    b = signal_persistence(-low_volatility(panel), panel.mask, (1, 3))
    assert a["rank_autocorr"] == pytest.approx(b["rank_autocorr"])


def test_cost_gate_reports_persistence(panel):
    card = evaluate(low_volatility, panel, with_incremental=False, with_diagnostics=False)
    names = {c.name for c in card.gates["G3_cost"].checks}
    assert {"rank_autocorr_at_h", "top_bin_turnover"} <= names


def test_winsorize_mad_caps_an_outlier_without_dropping_anything():
    x = pd.DataFrame(np.random.default_rng(3).standard_normal((50, 8)))
    x.iloc[0, 0] = 1e6
    w = winsorize_mad(x, scale=3.)
    assert w.shape == x.shape and w.iloc[0, 0] < 1e6
    assert w.notna().sum().sum() == x.notna().sum().sum()


def test_winsorize_quantile_respects_its_bounds():
    x = pd.DataFrame(np.arange(100.).reshape(10, 10))
    w = winsorize_quantile(x, .1, .9)
    assert (w.max(axis=1) <= x.quantile(.9, axis=1)).all()


def test_zscore_cannot_change_a_rank(panel):
    """The matrix scores cross-sectional ranks, and z-score is a per-date positive
    affine map, so it must be a no-op for every verdict it feeds."""
    raw = low_volatility(panel)
    a = raw.where(panel.mask).rank(axis=1)
    b = zscore(raw, panel.mask).rank(axis=1)
    assert (a.fillna(-1) == b.fillna(-1)).to_numpy().mean() > .999


def test_neutralize_leaves_no_exposure_to_what_was_removed(panel):
    resid = neutralize(low_volatility(panel), volume_shock(panel), panel.mask)
    corr = resid.corrwith(volume_shock(panel).where(panel.mask), axis=1).mean()
    assert abs(corr) < .05


def test_neutralize_requires_a_factor_and_enough_names():
    with pytest.raises(ValueError):
        neutralize(pd.DataFrame(np.zeros((3, 3))), [])
    thin = pd.DataFrame(np.random.default_rng(1).standard_normal((4, 3)))
    out = neutralize(thin, pd.DataFrame(np.ones((4, 3))), min_assets=5)
    assert out.isna().all().all(), "too few names must stay unknown, not be fitted"
