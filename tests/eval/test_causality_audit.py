"""Prefix checks catch panel look-ahead; they cannot certify external state."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


EVAL = Path(__file__).resolve().parents[2] / "eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from factor_eval.bundle import Panel  # noqa: E402
from factor_eval.causality import check_prefix_invariance  # noqa: E402
from factor_eval.matrix import FAIL, NA, PASS  # noqa: E402


@pytest.fixture
def panel():
    index = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC")
    t = np.arange(len(index))[:, None]
    asset = np.arange(8)[None, :]
    close = pd.DataFrame(100 + 0.3 * t + 3 * asset + np.sin(t / 3 + asset),
                         index=index, columns=[f"asset_{i}" for i in range(8)])
    volume = pd.DataFrame(1000.0, index=index, columns=close.columns)
    return Panel(open=close.copy(), high=close * 1.01, low=close * 0.99,
                 close=close, volume=volume, quote_volume=volume * close,
                 funding=close * 0, mask=close.notna())


@pytest.mark.parametrize("factor", [
    pytest.param(lambda p: p.close.shift(-1), id="future-shift"),
    pytest.param(lambda p: p.close.rolling(5, center=True).mean(), id="centered-window"),
    pytest.param(lambda p: (p.close - p.close.mean()) / p.close.std(),
                 id="full-sample-standardization"),
    pytest.param(lambda p: 1e12 + (p.close - p.close.mean()) / p.close.std(),
                 id="large-offset-must-not-hide-look-ahead"),
])
def test_panel_future_dependence_fails_prefix_invariance(panel, factor):
    check = check_prefix_invariance(factor, panel, factor(panel))
    assert check.status == FAIL
    assert check.value == 0.0
    assert "past signal changes" in check.note


@pytest.mark.parametrize("factor", [
    pytest.param(lambda p: p.close.rolling(7).mean(), id="causal-rolling"),
    pytest.param(lambda p: p.close.rank(axis=1, pct=True), id="cross-sectional-rank"),
    pytest.param(lambda p: p.close.shift(1).rolling(7).std().rank(axis=1),
                 id="lagged-rolling-and-rank"),
])
def test_causal_panel_computations_pass_prefix_invariance(panel, factor):
    check = check_prefix_invariance(factor, panel, factor(panel))
    assert check.status == PASS
    assert check.value == 1.0
    assert "sampled check only" in check.note


def test_precomputed_dataframe_has_unverified_causality(panel):
    signal = panel.close.shift(-1)
    check = check_prefix_invariance(signal, panel, signal)
    assert check.status == NA
    assert np.isnan(check.value)
    assert "no callable" in check.note


def test_prefix_exception_is_unknown_instead_of_pass(panel):
    full_length = len(panel.index)

    def factor(p):
        if len(p.index) < full_length:
            raise ValueError("this factor cannot evaluate shorter input")
        return p.close

    check = check_prefix_invariance(factor, panel, factor(panel))
    assert check.status == NA
    assert np.isnan(check.value)
    assert "prefix evaluation failed (ValueError)" in check.note


def test_no_finite_comparable_observations_is_unknown(panel):
    def factor(p):
        return pd.DataFrame(np.nan, index=p.index, columns=p.symbols)

    check = check_prefix_invariance(factor, panel, factor(panel))
    assert check.status == NA
    assert np.isnan(check.value)


def test_closed_over_external_state_is_outside_the_prefix_guarantee(panel):
    # An external cache already containing future information can stay identical
    # under every panel truncation. This PASS is deliberately a limitation test,
    # not evidence that the factor is causal or that upstream data was PIT-safe.
    external_signal = panel.close.shift(-1)

    def factor(p):
        return external_signal.reindex(p.index)

    check = check_prefix_invariance(factor, panel, factor(panel))
    assert check.status == PASS
    assert "sampled check only" in check.note
