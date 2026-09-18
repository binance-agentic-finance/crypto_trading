"""Sampling must be reproducible and de-duplication must only ever look at dev."""
import sys
from pathlib import Path

import numpy as np
import pytest

from cyqnt_trd.eval import load_bundle                                     # noqa: E402
from cyqnt_trd.eval.examples.example_factors import (low_volatility,       # noqa: E402
                                                  reversal_5d, volume_shock)
from cyqnt_trd.eval.library import (alpha101_factors, deduplicate,          # noqa: E402
                                 rank_corr, sample_factors)


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


@pytest.fixture(scope="module")
def factors(panel):
    return {"rev5": reversal_5d(panel), "lowvol": low_volatility(panel),
            "vshock": volume_shock(panel)}


def test_sampling_is_reproducible(factors):
    a = sample_factors(factors, 2, seed=7)
    b = sample_factors(factors, 2, seed=7)
    assert list(a) == list(b)
    assert list(sample_factors(factors, 2, seed=8)) != list(a) or len(factors) == 2


def test_sampling_does_not_depend_on_insertion_order(factors):
    shuffled = {k: factors[k] for k in reversed(list(factors))}
    assert list(sample_factors(factors, 2, seed=3)) == list(sample_factors(shuffled, 2, seed=3))


def test_asking_for_more_than_exists_raises(factors):
    with pytest.raises(ValueError, match="only"):
        sample_factors(factors, len(factors) + 1, seed=1)


def test_an_identical_copy_is_dropped(panel, factors):
    doubled = {**factors, "rev5_copy": factors["rev5"].copy()}
    kept, dropped = deduplicate(doubled, panel)
    assert len(kept) == len(factors)
    assert "rev5_copy" in dropped or "rev5" in dropped
    assert "rank corr" in next(iter(dropped.values()))


def test_a_rescaled_factor_is_a_duplicate(panel, factors):
    """Scale is removed by cross-sectional ranking, so x and 1000x are one bet."""
    doubled = {**factors, "rev5_scaled": factors["rev5"] * 1000.0}
    kept, _ = deduplicate(doubled, panel)
    assert len(kept) == len(factors)


def test_strength_decides_which_copy_survives(panel, factors):
    doubled = {**factors, "rev5_copy": factors["rev5"].copy()}
    kept, _ = deduplicate(doubled, panel,
                          strength={"rev5": 0.01, "rev5_copy": 0.5, "lowvol": 0.2,
                                    "vshock": 0.2})
    assert "rev5_copy" in kept and "rev5" not in kept


def test_correlation_is_measured_on_dev_only(panel, factors):
    """A val/oot-only difference must not change the dev correlation."""
    tampered = factors["rev5"].copy()
    late = tampered.index >= tampered.index[int(len(tampered) * 0.85)]
    tampered.loc[late] = -tampered.loc[late]
    dev = rank_corr({"a": factors["rev5"], "b": tampered}, panel, split="dev")
    assert np.isclose(dev.loc["a", "b"], 1.0, atol=1e-9)


def test_unrelated_factors_survive(panel, factors):
    kept, dropped = deduplicate(factors, panel, max_corr=0.99)
    assert len(kept) == len(factors) and not dropped


def test_alpha101_adapter_returns_the_full_set(panel):
    frames = alpha101_factors(panel, names=["alpha001", "alpha101"])
    assert set(frames) == {"alpha001", "alpha101"}
    for frame in frames.values():
        assert list(frame.columns) == panel.symbols
        assert frame.index.equals(panel.index)


def test_alpha101_rejects_an_unknown_name(panel):
    with pytest.raises(KeyError, match="unknown alpha101"):
        alpha101_factors(panel, names=["alpha999"])