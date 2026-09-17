"""The comparison is only worth reading if selection never touches val or oot."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval import load_bundle                                      # noqa: E402
from factor_eval.benchmarks import single_factor_naive                   # noqa: E402
from factor_eval.examples.example_factors import (low_volatility,        # noqa: E402
                                                  reversal_5d, volume_shock)
from factor_eval.experiment import (noise_floor, run_comparison,         # noqa: E402
                                    write_report)
from factor_eval.pipeline import build_strategy                          # noqa: E402
from factor_eval.portfolio import simulate                               # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


@pytest.fixture(scope="module")
def result(panel):
    return run_comparison(panel, source="alpha101", k=12, seed=1234,
                          variants=("neutral",))


def test_noise_floor_comes_from_the_calibration():
    floor = noise_floor(3)
    assert 0 < floor < 0.2


def test_the_run_is_reproducible(panel, result):
    again = run_comparison(panel, source="alpha101", k=12, seed=1234,
                           variants=("neutral",))
    assert again.sampled == result.sampled
    assert again.admitted == result.admitted
    assert np.allclose(again.table["ann_return"].to_numpy(dtype=float),
                       result.table["ann_return"].to_numpy(dtype=float), equal_nan=True)


def test_a_different_seed_draws_a_different_sample(panel, result):
    other = run_comparison(panel, source="alpha101", k=12, seed=20260917,
                           variants=("neutral",))
    assert other.sampled != result.sampled


def test_search_width_is_reported_as_the_pool_not_the_sample(result):
    """Drawing 6 of 99 is a 99-candidate search; saying 6 understates it 16x."""
    assert result.pool_size > len(result.sampled)


def test_every_baseline_is_present(result):
    books = set(result.books)
    assert {"constructed_neutral", "single_factor_naive", "btc_timing",
            "buy_hold_BTCUSDT", "equal_weight", "cash"} <= books


def test_the_comparison_covers_all_three_splits(result):
    assert set(result.table.index.get_level_values("split")) == {"dev", "val", "oot"}


def test_the_raw_factor_baseline_is_chosen_on_dev(result):
    """Picking it on oot would hand the baseline the answer."""
    best = result.config["naive_baseline_factor"]
    dev_ic = {n: abs(c.ic_dev) for n, c in result.cards.items() if n in result.admitted}
    assert best in dev_ic
    assert np.isclose(dev_ic[best], max(dev_ic.values()))


def test_one_factor_with_no_cap_reduces_to_the_naive_baseline(panel):
    """Construction must not quietly change a single-factor book.

    If this drifts, a comparison between 'constructed' and 'naive' is measuring an
    accounting difference rather than the weighting scheme.
    """
    frame = low_volatility(panel)
    built = build_strategy({"lowvol": frame}, panel, rebalance=3, cap=1.0,
                           neutral=True, fast=True)
    naive = simulate(single_factor_naive(frame, panel, rebalance=3), panel)
    assert np.allclose(built.book.equity.to_numpy(), naive.equity.to_numpy(),
                       equal_nan=True)


def test_switching_off_the_ic_floor_admits_more(panel):
    strict = run_comparison(panel, source="alpha101", k=12, seed=1234,
                            variants=("neutral",))
    loose = run_comparison(panel, source="alpha101", k=12, seed=1234,
                           admit_on_noise_floor=False, variants=("neutral",))
    assert len(loose.admitted) >= len(strict.admitted)


def test_report_is_written_and_reloadable(result, tmp_path):
    paths = write_report(result, tmp_path)
    text = Path(paths["markdown"]).read_text()
    assert "## Comparison" in text and "constructed_neutral" in text
    payload = json.loads(Path(paths["json"]).read_text())
    assert payload["source"] == "alpha101"
    assert payload["pool_size"] == result.pool_size
    assert payload["comparison"]


def test_an_unknown_source_is_refused(panel):
    with pytest.raises(ValueError, match="unknown source"):
        run_comparison(panel, source="nope")
