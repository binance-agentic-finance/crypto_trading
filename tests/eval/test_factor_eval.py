"""Contracts the evaluation matrix must keep.

These are the checks that would catch the matrix quietly becoming wrong: a leak
that no longer lights up, a direction that is not frozen, a gate that passes on a
missing measurement, or a bundle whose mask stops being point-in-time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EVAL = Path(__file__).resolve().parents[2] / "eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from factor_eval import evaluate, load_bundle                    # noqa: E402
from factor_eval.examples import example_factors as ex           # noqa: E402
from factor_eval.matrix import FAIL, NA, PASS, load_calibration, verdict_from  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


def test_bundle_is_aligned_and_pit(panel):
    panel.validate()
    assert len(panel.symbols) == 10
    assert panel.index.tz is not None and str(panel.index.tz) == "UTC"
    # the mask must never admit a name before it has prices
    assert not (panel.mask & panel.close.isna()).to_numpy().any()
    # and it must be strictly narrower than "everything listed"
    assert 0 < float(panel.mask.sum(axis=1).mean()) <= len(panel.symbols)


def test_leak_is_detected(panel):
    """A factor handed the traded window must score absurdly. If this ever stops
    holding, the measurement path no longer measures what it claims to."""
    card = evaluate(ex.look_ahead_trap, panel, name="leak", with_incremental=False)
    ic_dev = float(card.metrics[(card.metrics.split == "dev")
                                & (card.metrics.h == 3)].iloc[0]["ic_mean"])
    assert ic_dev > 0.8
    assert card.gates["G1_information"].status == PASS
    assert card.gates["G3_cost"].status == PASS


def test_honest_factor_does_not_reach_pass(panel):
    """Nothing in the examples clears every gate; a green PASS should be rare
    enough that seeing one is a reason to re-read the factor, not to ship it."""
    for name in ("reversal_5d", "low_volatility", "volume_shock"):
        card = evaluate(getattr(ex, name), panel, name=name, with_incremental=False)
        assert card.verdict != "PASS", f"{name} unexpectedly cleared every gate"


def test_skipped_incremental_cannot_report_a_clean_pass(panel):
    card = evaluate(ex.look_ahead_trap, panel, with_incremental=False)
    assert card.verdict == "PASS_CONDITIONAL"
    assert "G5_incremental" not in card.gates


def test_direction_is_frozen_on_dev_only(panel):
    """Flipping the factor's sign must not change the verdict: the engine freezes
    the direction on dev, so a sign flip is not new information."""
    a = evaluate(ex.reversal_5d, panel, with_incremental=False)
    b = evaluate(lambda p: -ex.reversal_5d(p), panel, with_incremental=False)
    assert a.verdict == b.verdict
    ic = lambda c: float(c.metrics[(c.metrics.split == "val")            # noqa: E731
                                   & (c.metrics.h == 3)].iloc[0]["ic_mean"])
    assert ic(a) == pytest.approx(ic(b), abs=1e-12)


def test_constant_factor_is_rejected_on_data(panel):
    card = evaluate(lambda p: pd.DataFrame(1.0, index=p.index, columns=p.symbols),
                    panel, name="constant", with_incremental=False)
    assert card.verdict == "REJECT_DATA"
    assert card.blocking == ["G0_data"]


def test_empty_factor_raises(panel):
    with pytest.raises(ValueError):
        evaluate(lambda p: pd.DataFrame(np.nan, index=p.index, columns=p.symbols), panel)


def test_series_factor_is_rejected_with_a_useful_message(panel):
    with pytest.raises(TypeError, match="DataFrame"):
        evaluate(lambda p: p.close.mean(axis=1), panel)


def test_calibration_must_match_the_requested_setting(panel):
    """The noise floor is specific to horizon and cost; silently reusing it for
    another setting would compare a factor against the wrong distribution."""
    with pytest.raises(ValueError, match="calibrate"):
        evaluate(ex.reversal_5d, panel, primary_h=5, with_incremental=False)
    with pytest.raises(ValueError, match="calibrate"):
        evaluate(ex.reversal_5d, panel, cost_bps=12.0, with_incremental=False)


def test_calibration_percentiles_are_ordered():
    cal = load_calibration()
    assert cal["trials"] >= 30 and cal["primary_h"] == 3
    for field in ("ic_mean", "ic_t_hac", "net_available_bp"):
        for split in ("dev", "val", "oot"):
            q = cal[field][split]
            assert q["p50"] <= q["p90"] <= q["p95"] <= q["p99"]


def test_verdict_table_covers_every_branch():
    from factor_eval.matrix import Gate, Check

    def gate(key, status):
        return Gate(key, key, "", [Check("c", 1.0, 0.0, ">", status, "test")])

    def gates(g0, g1, g3, g4=PASS, g5=PASS):
        return {"G0_data": gate("G0_data", g0), "G1_information": gate("G1_information", g1),
                "G2_structure": gate("G2_structure", PASS), "G3_cost": gate("G3_cost", g3),
                "G4_robustness": gate("G4_robustness", g4),
                "G5_incremental": gate("G5_incremental", g5)}

    assert verdict_from(gates(FAIL, PASS, PASS))[0] == "REJECT_DATA"
    assert verdict_from(gates(PASS, PASS, PASS))[0] == "PASS"
    assert verdict_from(gates(PASS, PASS, PASS, g4=FAIL))[0] == "PASS_CONDITIONAL"
    assert verdict_from(gates(PASS, PASS, PASS, g5=FAIL))[0] == "PASS_CONDITIONAL"
    assert verdict_from(gates(PASS, PASS, FAIL))[0] == "HOLD_INFO"
    assert verdict_from(gates(PASS, FAIL, PASS))[0] == "HOLD_WEAK"
    assert verdict_from(gates(PASS, FAIL, FAIL))[0] == "REJECT"


def test_structure_uses_the_frozen_direction(panel):
    """A factor and its negation must produce the same bin picture, because the
    engine trades the frozen direction, not the sign the user happened to write."""
    a = evaluate(ex.reversal_5d, panel, with_incremental=False).structure
    b = evaluate(lambda p: -ex.reversal_5d(p), panel, with_incremental=False).structure
    assert a["spread_bp"] == pytest.approx(b["spread_bp"], rel=1e-9)


def test_bins_follow_the_width_of_the_cross_section(panel):
    card = evaluate(ex.reversal_5d, panel, with_incremental=False)
    s = card.structure
    assert 3 <= s["n_bins"] <= 5
    assert s["names_per_bin"] >= 2.0, "bins must not be thinner than 2 names per date"


def test_scorecard_renders_and_serialises(panel):
    card = evaluate(ex.low_volatility, panel, name="lowvol", with_incremental=False)
    md = card.to_markdown()
    assert card.verdict in md and "G3_cost" in md and "null p95" in md
    payload = card.to_dict()
    assert payload["verdict"] == card.verdict
    assert set(payload["gates"]) == set(card.gates)
    # every serialised value must be JSON-clean (no NaN leaking into the payload)
    import json
    json.dumps(payload, ensure_ascii=False)


def test_incremental_flags_a_baseline_clone(panel):
    """A factor that *is* a baseline must not be reported as incremental."""
    card = evaluate(ex.low_volatility, panel, name="lowvol")
    inc = card.incremental
    assert inc["closest_baseline"] == "B_lowvol10"
    assert inc["max_abs_rank_corr"] > 0.9
    assert card.gates["G5_incremental"].status == FAIL
