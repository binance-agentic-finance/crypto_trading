"""Stage boundaries must fail at the stage that broke them, and the leak check
must actually catch leaks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval import load_bundle                                      # noqa: E402
from factor_eval.examples.example_factors import (low_volatility,        # noqa: E402
                                                  reversal_5d)
from factor_eval.forecast import (fit_betas, make_forecast,              # noqa: E402
                                  positions_from_forecast)
from factor_eval.framework import (BacktestSpec, STAGES, backtest_weights,  # noqa: E402
                                   check_stateless, describe_stages, run,
                                   validate_factors, validate_mu, validate_weights)


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


@pytest.fixture(scope="module")
def factors(panel):
    return {"rev5": reversal_5d(panel), "lowvol": low_volatility(panel)}


@pytest.fixture(scope="module")
def spec(panel, factors):
    betas = fit_betas(factors, panel, method="ic", h=3)
    return BacktestSpec(panel=panel, factors=factors, forecast=make_forecast(betas),
                        sizing=lambda mu, mask: positions_from_forecast(mu, mask))


def test_the_stage_list_is_ordered_and_described():
    assert [s["n"] for s in STAGES] == list(range(len(STAGES)))
    assert all(s["input"] and s["output"] and s["impl"] for s in STAGES)
    assert "backtest" in describe_stages()


# ------------------------------------------------------------- boundary checks
def test_a_factor_off_the_grid_is_rejected(panel, factors):
    off = {"rev5": factors["rev5"].iloc[:-5]}
    with pytest.raises(ValueError, match="not on the panel's cell index"):
        validate_factors(off, panel)


def test_a_factor_with_reordered_symbols_is_rejected(panel, factors):
    shuffled = factors["rev5"][list(reversed(panel.symbols))]
    with pytest.raises(ValueError, match="columns must be the panel's symbols"):
        validate_factors({"rev5": shuffled}, panel)


def test_an_all_missing_factor_is_rejected(panel, factors):
    empty = factors["rev5"] * np.nan
    with pytest.raises(ValueError, match="no finite value inside the mask"):
        validate_factors({"empty": empty}, panel)


def test_mu_must_be_a_frame(panel):
    with pytest.raises(TypeError, match="must be a DataFrame"):
        validate_mu(pd.Series(1.0, index=panel.index), panel)


def test_weights_may_not_be_unknown(panel, factors):
    """NaN is not a position. It has to have become an explicit 0 by stage 5."""
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    w.iloc[10, 0] = np.nan
    with pytest.raises(ValueError, match="explicit 0"):
        validate_weights(w, panel)


def test_weights_may_not_hold_outside_the_mask(panel):
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    ineligible = np.argwhere(~panel.mask.to_numpy())
    if not len(ineligible):
        pytest.skip("this panel has no ineligible cell")
    row, col = ineligible[0]
    w.iloc[row, col] = 0.5
    with pytest.raises(ValueError, match="outside the eligibility mask"):
        validate_weights(w, panel)


# -------------------------------------------------------------------- engine
def test_a_flat_book_earns_nothing(panel):
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    result = backtest_weights(w, panel)
    assert np.allclose(result.pnl["net"].dropna(), 0.0)
    assert np.allclose(result.equity.to_numpy(), 1.0)


def test_periods_do_not_overlap(panel):
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    result = backtest_weights(w, panel, h=3, entry_lag=2)
    entries = pd.DatetimeIndex(result.pnl["entry"])
    exits = pd.DatetimeIndex(result.pnl["exit"])
    assert (entries[1:] >= exits[:-1]).all()


def test_cost_scales_with_turnover(panel, spec):
    cheap = run(spec).pnl
    dear = backtest_weights(run(spec).weights, panel, cost_bps=65.0).pnl
    assert np.allclose(dear["trading_cost"].to_numpy(),
                       cheap["trading_cost"].to_numpy() * 10.0)


def test_an_unknown_cost_is_not_a_zero_cost(panel):
    """A period holding a name with no price is NaN, never a free period."""
    broken = panel.open.copy()
    broken.iloc[:, 0] = np.nan
    holed = type(panel)(open=broken, high=panel.high, low=panel.low, close=panel.close,
                        volume=panel.volume, quote_volume=panel.quote_volume,
                        funding=panel.funding, mask=panel.mask, meta={})
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    w.iloc[:, 0] = 0.0
    w.iloc[:, 1] = 0.5
    w = w.where(panel.mask, 0.0)
    result = backtest_weights(w, holed)
    assert result.pnl["complete"].all()          # the missing name is not held
    w.iloc[:, 0] = 0.5
    w = w.where(panel.mask, 0.0)
    assert not backtest_weights(w, holed).pnl["complete"].all()


def test_run_produces_one_metric_row_per_split(spec):
    result = run(spec)
    assert list(result.metrics["split"]) == ["dev", "val", "oot"]
    assert result.config["stages"] == ["factors", "preprocess", "forecast",
                                       "sizing", "backtest"]


def test_run_rejects_a_stage_that_returns_the_wrong_shape(panel, spec):
    bad = BacktestSpec(panel=panel, factors=spec.factors,
                       forecast=lambda f, p: p.close.iloc[:, 0],
                       sizing=spec.sizing)
    with pytest.raises(TypeError, match="forecast mu"):
        run(bad)


# ------------------------------------------------------------ leak detection
@pytest.mark.parametrize("build,expected", [
    (lambda p: -(p.close / p.close.shift(5) - 1), True),
    (lambda p: -np.log(p.close / p.close.shift(1)).rolling(10).std(), True),
    (lambda p: p.close.shift(-1) / p.close - 1, False),
    (lambda p: p.close.shift(-3) / p.close - 1, False),
    (lambda p: (p.close - p.close.mean()) / p.close.std(), False),
    (lambda p: p.close.rolling(11, center=True).mean(), False),
])
def test_check_stateless_separates_causal_from_leaking(panel, build, expected):
    assert check_stateless(build, panel)["stateless"] is expected


def test_a_forward_shift_is_caught_by_the_tail_not_the_values(panel):
    """The reason the check needs `known_only_on_full`.

    A `shift(-k)` agrees on every cell both runs can compute; it differs only in
    the last k rows of the prefix, which a naive "compare where both are finite"
    rule discards. Without that counter this factor reads as clean.
    """
    report = check_stateless(lambda p: p.close.shift(-3) / p.close - 1, panel)
    assert all(c["mismatched"] == 0 for c in report["checks"])
    assert all(c["known_only_on_full"] > 0 for c in report["checks"])


def test_the_whole_chain_is_stateless(panel, factors, spec):
    forecast = spec.forecast
    chain = lambda p: positions_from_forecast(                    # noqa: E731
        forecast({k: v.reindex(p.index) for k, v in factors.items()}, p), p.mask)
    assert check_stateless(chain, panel)["stateless"]
