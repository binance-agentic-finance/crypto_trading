"""Coefficients fitted on dev, frozen for everything after it."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval import load_bundle                                      # noqa: E402
from factor_eval.examples.example_factors import (low_volatility,        # noqa: E402
                                                  reversal_5d, volume_shock)
from factor_eval.forecast import (fit_betas, make_forecast,              # noqa: E402
                                  positions_from_forecast, quantile_positions,
                                  sign_positions)


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


@pytest.fixture(scope="module")
def factors(panel):
    return {"rev5": reversal_5d(panel), "lowvol": low_volatility(panel),
            "vshock": volume_shock(panel)}


def test_betas_only_ever_read_dev(panel, factors):
    """Changing val and oot must not move a dev-fitted coefficient."""
    tampered = {}
    cut = panel.index[int(len(panel.index) * 0.7)]
    for name, frame in factors.items():
        copy = frame.copy()
        copy.loc[copy.index > cut] *= -3.0
        tampered[name] = copy
    a = fit_betas(factors, panel, method="ic")
    b = fit_betas(tampered, panel, method="ic")
    assert a.weights.keys() == b.weights.keys()
    for name in a.weights:
        assert np.isclose(a.weights[name], b.weights[name], atol=1e-12)


def test_equal_method_keeps_only_the_sign(panel, factors):
    betas = fit_betas(factors, panel, method="equal")
    assert {round(abs(v) * len(betas.weights), 9) for v in betas.weights.values()} == {1.0}


def test_betas_are_normalised_and_shrink_moves_toward_equal(panel, factors):
    plain = fit_betas(factors, panel, method="ic")
    shrunk = fit_betas(factors, panel, method="ic", shrink=1.0)
    assert np.isclose(sum(abs(v) for v in plain.weights.values()), 1.0)
    assert {round(abs(v) * len(shrunk.weights), 9) for v in shrunk.weights.values()} == {1.0}


def test_ridge_runs_and_returns_one_coefficient_per_factor(panel, factors):
    betas = fit_betas(factors, panel, method="ridge")
    assert set(betas.weights) == set(factors)
    assert all(np.isfinite(v) for v in betas.weights.values())


def test_a_forecast_refuses_a_factor_set_it_was_not_fitted_on(panel, factors):
    forecast = make_forecast(fit_betas(factors, panel, method="ic"))
    with pytest.raises(KeyError, match="not in this factor set"):
        forecast({"rev5": factors["rev5"]}, panel)


def test_rank_standardisation_makes_the_forecast_scale_free(panel, factors):
    forecast = make_forecast(fit_betas(factors, panel, method="equal"))
    a = forecast(factors, panel)
    b = forecast({k: v * 1000.0 for k, v in factors.items()}, panel)
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


def test_a_missing_factor_does_not_vote_neutral(panel, factors):
    """Renormalise by the weight that participated, do not read the gap as 0."""
    holed = dict(factors)
    broken = factors["vshock"].copy()
    broken.iloc[:, 0] = np.nan
    holed["vshock"] = broken
    betas = fit_betas(factors, panel, method="equal")
    full = make_forecast(betas)(factors, panel)
    without = make_forecast(fit_betas({k: factors[k] for k in ("rev5", "lowvol")},
                                      panel, method="equal"))(
        {k: factors[k] for k in ("rev5", "lowvol")}, panel)
    partial = make_forecast(betas)(holed, panel)
    column = panel.symbols[0]
    shared = partial[column].notna() & without[column].notna()
    assert np.allclose(partial[column][shared], without[column][shared], atol=1e-9)
    assert not np.allclose(partial[column][shared], full[column][shared], atol=1e-9)


# ------------------------------------------------------------------- sizing
def test_weights_are_neutral_capped_and_normalised(panel, factors):
    """Neutral, capped and at target gross -- all three, exactly.

    Gross falls short only when the cap genuinely makes it unreachable (too few
    names on the thinner side), and the book stays balanced when it does.
    """
    mu = make_forecast(fit_betas(factors, panel, method="ic"))(factors, panel)
    for cap in (0.5, 0.35, 0.3):
        w = positions_from_forecast(mu, panel.mask, gross=1.0, max_weight=cap)
        live = w[w.abs().sum(axis=1) > 0]
        assert np.allclose(live.sum(axis=1).to_numpy(), 0.0, atol=1e-10)
        assert live.abs().max().max() <= cap + 1e-9
        assert np.allclose(live.abs().sum(axis=1).to_numpy(), 1.0, atol=1e-8)


def test_a_tight_cap_costs_gross_not_neutrality(panel, factors):
    """The documented priority: stay balanced, hold less.

    Sizing each side to its own capacity instead left a 0.15 net exposure on a
    supposedly market-neutral book.
    """
    mu = make_forecast(fit_betas(factors, panel, method="ic"))(factors, panel)
    w = positions_from_forecast(mu, panel.mask, gross=1.0, max_weight=0.12)
    live = w[w.abs().sum(axis=1) > 0]
    assert np.allclose(live.sum(axis=1).to_numpy(), 0.0, atol=1e-10)
    assert live.abs().max().max() <= 0.12 + 1e-9
    assert live.abs().sum(axis=1).min() < 1.0


def test_capping_happens_after_normalisation(panel, factors):
    """Capping the raw scores first squashes the spread while looking fine.

    Capping after normalisation keeps the book at its gross target; the weights
    concentrate less, but the money is still fully deployed.
    """
    mu = make_forecast(fit_betas(factors, panel, method="ic"))(factors, panel)
    tight = positions_from_forecast(mu, panel.mask, max_weight=0.3)
    loose = positions_from_forecast(mu, panel.mask, max_weight=None)
    live = loose.abs().sum(axis=1) > 0
    assert np.allclose(loose[live].abs().sum(axis=1).to_numpy(), 1.0, atol=1e-8)
    assert np.allclose(tight[live].abs().sum(axis=1).to_numpy(), 1.0, atol=1e-8)
    assert tight.abs().max().max() < loose.abs().max().max()


def test_a_thin_cross_section_is_flat_not_stale(panel, factors):
    """0 means do not hold. NaN would read as 'keep whatever you had'."""
    mu = make_forecast(fit_betas(factors, panel, method="ic"))(factors, panel)
    w = positions_from_forecast(mu, panel.mask, min_assets=99)
    assert not w.isna().to_numpy().any()
    assert np.allclose(w.to_numpy(), 0.0)


def test_sign_and_quantile_sizing_stay_neutral(panel, factors):
    mu = make_forecast(fit_betas(factors, panel, method="ic"))(factors, panel)
    for w in (sign_positions(mu, panel.mask), quantile_positions(mu, panel.mask)):
        live = w[w.abs().sum(axis=1) > 0]
        assert np.allclose(live.sum(axis=1).to_numpy(), 0.0, atol=1e-8)


def test_bad_arguments_are_refused(panel, factors):
    with pytest.raises(ValueError, match="unknown method"):
        fit_betas(factors, panel, method="magic")
    with pytest.raises(ValueError, match="shrink"):
        fit_betas(factors, panel, shrink=2.0)
    mu = make_forecast(fit_betas(factors, panel, method="ic"))(factors, panel)
    with pytest.raises(ValueError, match="gross"):
        positions_from_forecast(mu, panel.mask, gross=0.0)
    with pytest.raises(ValueError, match="quantile"):
        quantile_positions(mu, panel.mask, quantile=0.8)
