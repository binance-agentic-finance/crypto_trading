"""Adversarial regressions for missing evidence and manufactured factor signals."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EVAL = Path(__file__).resolve().parents[2] / "eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from factor_eval.baselines import cross_sectional_rank, residualise  # noqa: E402
from factor_eval.matrix import (Check, Gate, FAIL, NA, PASS, WARN,  # noqa: E402
                                gate_cost, gate_robustness, gate_structure, verdict_from)


GATE_KEYS = ("G0_data", "G1_information", "G2_structure", "G3_cost",
             "G4_robustness", "G5_incremental")


def _gates():
    return {key: Gate(key, key, "", [Check("measured", 1.0, 0.0, ">", PASS, "fixture")])
            for key in GATE_KEYS}


def test_failed_structure_cannot_be_cancelled_by_other_passes():
    gates = _gates()
    gates["G2_structure"].checks[0].status = FAIL
    verdict, blocking = verdict_from(gates)
    assert verdict == "HOLD_STRUCTURE"
    assert "G2_structure" in blocking


@pytest.mark.parametrize("missing_gate", GATE_KEYS)
def test_missing_required_gate_never_counts_as_a_pass(missing_gate):
    gates = _gates()
    del gates[missing_gate]
    verdict, blocking = verdict_from(gates)
    assert verdict == "HOLD_INCOMPLETE"
    assert missing_gate in blocking


def test_missing_required_measurement_is_not_hidden_by_other_passes():
    gates = _gates()
    gates["G3_cost"].checks.append(Check("net_bp_oot", np.nan, 0.0, ">", NA, "fixture"))
    assert gates["G3_cost"].status == NA
    assert verdict_from(gates)[0] == "HOLD_INCOMPLETE"


def test_reporting_only_na_does_not_block_a_complete_gate():
    gates = _gates()
    gates["G3_cost"].checks.append(
        Check("turnover", 2.0, None, ">", NA, "reported only", required=False))
    assert gates["G3_cost"].status == PASS
    assert verdict_from(gates)[0] == "PASS"


def test_advisory_warning_is_conditional_but_required_failure_is_held():
    gates = _gates()
    gates["G4_robustness"].checks[0].status = WARN
    assert verdict_from(gates)[0] == "PASS_CONDITIONAL"
    gates["G4_robustness"].checks[0].status = FAIL
    assert verdict_from(gates)[0] == "HOLD_CONDITIONAL"


def _cost_metrics():
    return pd.DataFrame([
        {"split": split, "h": 3, "net_bp": 100.0, "net_available_bp": 100.0,
         "breakeven_cost_bps": 100.0, "turnover": 2.0,
         "n_complete": 100, "n_periods": 100}
        for split in ("dev", "val", "oot")
    ])


@pytest.mark.parametrize("strict_net", [np.nan, 100.0])
def test_partial_cycle_profit_cannot_pass_the_cost_gate(strict_net):
    metrics = _cost_metrics()
    metrics.loc[metrics.split == "oot", ["n_complete", "net_bp"]] = [99, strict_net]
    cal = {"net_available_bp": {"val": {"p95": 1.0}}}
    gate = gate_cost(metrics, 3, 6.5, cal)
    assert gate.status == NA
    gates = _gates()
    gates["G3_cost"] = gate
    assert verdict_from(gates)[0] == "HOLD_INCOMPLETE"


def test_fully_accounted_profit_still_passes_cost_checks():
    cal = {"net_available_bp": {"val": {"p95": 1.0}}}
    assert gate_cost(_cost_metrics(), 3, 6.5, cal).status == PASS


def test_robustness_requires_all_three_splits_even_when_remaining_one_is_positive():
    metrics = pd.DataFrame([
        {"split": split, "h": 3, "ic_mean": 0.9 if split == "dev" else np.nan,
         "sharpe_net": 100.0}
        for split in ("dev", "val", "oot")
    ])
    gate = gate_robustness(metrics, None, 3, {"sharpe_net": {"val": {"p95": 1.0}}})
    assert gate.status == NA
    gates = _gates()
    gates["G4_robustness"] = gate
    assert verdict_from(gates)[0] == "HOLD_INCOMPLETE"


def _structure_metrics(raw_ics):
    return pd.DataFrame([
        {"split": "dev", "h": horizon, "ic_mean": abs(raw_ic),
         "ic_mean_raw": raw_ic, "sign": 1.0 if raw_ic >= 0 else -1.0}
        for horizon, raw_ic in zip((1, 3, 5), raw_ics)
    ])


def test_horizon_sign_check_cannot_use_independently_flipped_ics():
    structure = {"names_per_bin": 3.0, "spearman": 1.0, "spread_bp": 10.0}
    same = gate_structure(structure, _structure_metrics([-0.3, -0.2, -0.1]), 3, (1, 3, 5))
    flipped = gate_structure(structure, _structure_metrics([0.3, -0.2, -0.1]), 3, (1, 3, 5))
    assert same.status == PASS
    assert flipped.status == WARN


def test_missing_requested_horizon_is_not_evidence_of_sign_agreement():
    structure = {"names_per_bin": 3.0, "spearman": 1.0, "spread_bp": 10.0}
    gate = gate_structure(structure, _structure_metrics([0.3, 0.2, np.nan]), 3, (1, 3, 5))
    assert gate.status == NA


def _random_frames(n_assets, seed=11):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=4, tz="UTC")
    columns = [f"asset_{i}" for i in range(n_assets)]
    mask = pd.DataFrame(True, index=index, columns=columns)
    signal = pd.DataFrame(rng.normal(size=mask.shape), index=index, columns=columns)
    baselines = {str(k): pd.DataFrame(rng.normal(size=mask.shape), index=index, columns=columns)
                 for k in range(5)}
    return signal, baselines, mask


def test_saturated_baseline_fit_cannot_create_a_ranked_residual_signal():
    signal, baselines, mask = _random_frames(5)
    residual = residualise(signal, baselines, mask)
    assert residual.isna().all().all()
    assert cross_sectional_rank(residual, mask).isna().all().all()


def test_exact_baseline_clone_has_no_residual_signal_at_machine_precision():
    signal, baselines, mask = _random_frames(12)
    baselines["clone"] = signal.copy()
    residual = residualise(signal, baselines, mask)
    assert cross_sectional_rank(residual, mask).isna().all().all()


def test_independent_signal_retains_a_measurable_residual():
    signal, baselines, mask = _random_frames(12)
    residual = residualise(signal, baselines, mask)
    assert residual.notna().all().all()
    assert (residual.std(axis=1) > 0.1).all()


def test_duplicate_baselines_do_not_consume_nonexistent_degrees_of_freedom():
    signal, baselines, mask = _random_frames(5)
    duplicate_baselines = {name: baselines["0"] for name in baselines}
    residual = residualise(signal, duplicate_baselines, mask)
    assert residual.notna().all().all()
    assert (residual.std(axis=1) > 0.1).all()
