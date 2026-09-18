"""The comprehensive matrix keeps sample, timing and missing-evidence contracts."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.eval import evaluate, load_bundle  # noqa: E402
from cyqnt_trd.eval.diagnostics import _bucket, _edges, _weighted_quantiles  # noqa: E402
from cyqnt_trd.eval.examples.example_factors import low_volatility  # noqa: E402


@pytest.fixture(scope="module")
def card():
    return evaluate(low_volatility, diagnostic_horizons=(1, 3, 5), n_bootstrap=20)


def test_matrix_keeps_unavailable_inputs_visible(card):
    matrix = {row["dimension"]: row for row in card.diagnostics["matrix"]}
    assert len(matrix) == 14
    assert matrix["tradability"]["status"] == "NOT_EVALUATED"
    assert matrix["capacity"]["status"] == "NOT_EVALUATED"
    assert matrix["strategy_combination"]["status"] == "NOT_EVALUATED"
    assert matrix["four_target_comparison"]["status"] == "MEASURED"
    json.dumps(card.to_dict(), allow_nan=False)


def test_four_targets_compare_identical_samples_in_each_split(card):
    common = [r for r in card.diagnostics["target_comparison"]["rows"] if r["sample"] == "four_target_common"]
    assert len(common) == 12
    for split in ("dev", "val", "oot"):
        rows = [r for r in common if r["split"] == split]
        assert len({r["n_pairs"] for r in rows}) == 1
        assert len({r["n_dates_pooled"] for r in rows}) == 1


def test_curves_use_common_sample_and_fixed_holding_delay(card):
    rows = card.diagnostics["ftr"]["rows"]
    assert {r["h"] for r in rows} == {1, 3, 5}
    assert len({r["common_cells"] for r in rows}) == 1
    assert all(r["available_cells"] >= r["common_cells"] for r in rows)
    assert card.diagnostics["delay"]["holding_h"] == card.primary_h
    assert {r["lag"] for r in card.diagnostics["delay"]["rows"]} == {1, 2, 3, 4, 7}


def test_cost_sensitivity_keeps_funding_at_zero_trading_cost(card):
    rows = card.diagnostics["cost_stress"]["rows"]
    val = card.metrics[(card.metrics.h == 3) & (card.metrics.split == "val")].iloc[0]
    zero = next(r for r in rows if r["split"] == "val" and r["multiplier"] == 0)
    base = next(r for r in rows if r["split"] == "val" and r["multiplier"] == 1)
    assert zero["net_bp"] == pytest.approx(val.gross_bp - val.funding_bp)
    assert base["net_bp"] == pytest.approx(val.net_bp)
    assert all(np.isnan(r["net_bp"]) for r in rows if r["split"] == "oot")


def test_date_weights_prevent_wide_dates_from_dominating_quantiles():
    frame = pd.DataFrame([[0, np.nan, np.nan, np.nan], [10, 10, 10, 10]])
    q = _weighted_quantiles(frame, frame.notna(), [.25, .5, .75])
    assert q.tolist() == [0, 0, 10]


def test_frozen_edges_and_ties_are_preserved():
    frame = pd.DataFrame([[0, 0, 1, 1], [2, 2, 3, 3], [100, 100, 200, 200]])
    dev = frame.notna()
    dev.iloc[-1] = False
    edges = _edges(frame, dev, 25)
    changed = frame.copy()
    changed.iloc[-1] *= -1000
    assert np.array_equal(edges, _edges(changed, dev, 25))
    groups = _bucket(frame, edges)
    assert (groups.iloc[:, 0] == groups.iloc[:, 1]).all()
    assert (groups.iloc[:, 2] == groups.iloc[:, 3]).all()


def test_missing_benchmark_does_not_hide_available_raw_evidence():
    card = evaluate(low_volatility, benchmark_symbol="ABSENT", diagnostic_horizons=(1, 3), n_bootstrap=0)
    rows = card.diagnostics["target_comparison"]["rows"]
    raw = [r for r in rows if r["target"] == "raw_rtf" and r["sample"] == "target_available"]
    assert len(raw) == 3 and all(r["n_pairs"] > 0 for r in raw)
    assert card.diagnostics["target_comparison"]["availability"]["res_rtf"]["status"] == "UNAVAILABLE"
    assert all(r["target"] == "raw_rtf" for r in card.diagnostics["ftr"]["rows"])


def test_optional_spread_enables_n_diagnostics_without_inventing_capacity():
    panel = load_bundle()
    spread = panel.close * 0 + .001
    card = evaluate(low_volatility, panel, spread=spread, diagnostic_horizons=(1, 3), n_bootstrap=0)
    assert card.diagnostics["ic_vs_n"]["rows"]
    for row in card.diagnostics["ic_vs_n"]["rows"]:
        if not np.isfinite(row["rank_ic"]):
            assert not np.isfinite(row["n_median"])
    matrix = {r["dimension"]: r for r in card.diagnostics["matrix"]}
    assert matrix["capacity"]["status"] == "NOT_EVALUATED"


def test_sparse_dates_cannot_produce_a_block_confidence_claim():
    from cyqnt_trd.eval.statistics import summarize_time_mean
    x = pd.Series(np.nan, index=pd.date_range("2024-01-01", periods=100))
    x.iloc[:3] = [1, 2, 3]
    result = summarize_time_mean(x, block_length=10, n_bootstrap=20)
    assert np.isnan(result["ci_low"])
    assert "finite date" in result["reason"]