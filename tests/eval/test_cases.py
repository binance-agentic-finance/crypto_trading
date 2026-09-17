"""An external spec must either run or say why not -- never look like it ran.

Every spec here is synthetic. The adapter is what is under test, not any
particular strategy.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval import load_bundle                                      # noqa: E402
from factor_eval.cases import (CaseSpec, build_blueprint, load_case,     # noqa: E402
                               resolve_signals, run_case)


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


def _spec(tiers, **verdicts):
    return {"scoring": {"tiers": tiers,
                        "verdicts": {"candidate_min": 5, **verdicts}}}


PRICE_ONLY = _spec({
    "rsi_zone": {"max_score": 3},
    "ema_trend": {"max_score": 3},
    "atr_regime": {"max_score": 2},
    "volume_confirmation": {"max_score": 2},
})


def test_a_price_and_volume_spec_runs_and_trades(panel):
    result = run_case(load_case(PRICE_ONLY, name="synthetic"), panel)
    assert result.ran, result.error
    assert result.coverage["n_unavailable"] == 0
    assert float(result.backtest.metrics["turnover"].max()) > 0


def test_tiers_the_panel_cannot_compute_are_named_not_scored(panel):
    spec = load_case(_spec({
        "rsi_zone": {"max_score": 3},
        "ema_trend": {"max_score": 3},
        "smart_money_inflow": {"max_score": 3},
        "hotrank_attention": {"max_score": 2},
    }), name="mixed")
    resolved, unavailable = resolve_signals(spec)
    assert set(resolved) == {"rsi_zone", "ema_trend"}
    assert set(unavailable) == {"smart_money_inflow", "hotrank_attention"}
    assert all(reason for reason in unavailable.values())


def test_an_uncomputable_hard_gate_stops_the_run(panel):
    """A gate nobody can evaluate is not a gate that passes."""
    spec = load_case(_spec({
        "rsi_zone": {"max_score": 3},
        "ema_trend": {"max_score": 3},
        "atr_regime": {"max_score": 2},
        "hotrank_attention": {"max_score": 2, "is_hard_gate": True},
    }), name="gated")
    blocked = run_case(spec, panel)
    assert not blocked.ran
    assert "hard gate" in blocked.error and "hotrank_attention" in blocked.error
    # Measuring the rest is allowed, but it is a different strategy.
    anyway = run_case(spec, panel, require_gates=False)
    assert anyway.ran
    assert anyway.coverage["gates_unavailable"] == ["hotrank_attention"]


def test_an_unreachable_threshold_is_refused_not_reported_as_flat(panel):
    """Dropped tiers can put the spec's own threshold out of reach.

    Backtesting it anyway produces a flat book and a 0.0 return, which reads as
    "this strategy makes no money" instead of "this panel cannot run it".
    """
    spec = load_case(_spec({
        "rsi_zone": {"max_score": 2},
        "smart_money_inflow": {"max_score": 5},
        "hotrank_attention": {"max_score": 3},
    }, candidate_min=8), name="short")
    result = run_case(spec, panel, min_resolved=1)
    assert not result.ran
    assert "unreachable" in result.error
    assert result.coverage["reachable_score"] < result.coverage["entry_score"]


def test_a_spec_that_never_opens_a_position_is_refused(panel):
    spec = load_case(_spec({
        "rsi_zone": {"max_score": 3},
        "ema_trend": {"max_score": 3},
        "atr_regime": {"max_score": 2},
    }, candidate_min=7.999), name="impossible")
    result = run_case(spec, panel)
    assert not result.ran
    assert "no position was ever opened" in result.error


def test_a_spec_without_tiers_is_not_a_tiered_strategy():
    with pytest.raises(ValueError, match="no scoring.tiers"):
        load_case({"scoring": {"verdicts": {"candidate_min": 5}}}, name="scanner")


def test_tier_weights_are_not_applied_on_top_of_max_score(panel):
    """`weight` is a normalised share; `max_score` already carries the magnitude.

    Applying both scales the total to about 1.0, the spec's threshold of 5 becomes
    unreachable, and the case backtests as a flat book that still reports "ran".
    """
    spec = load_case(_spec({
        "rsi_zone": {"max_score": 3, "weight": 0.4},
        "ema_trend": {"max_score": 3, "weight": 0.4},
        "atr_regime": {"max_score": 2, "weight": 0.2},
    }), name="weighted")
    resolved, _ = resolve_signals(spec)
    blueprint = build_blueprint(spec, resolved)
    assert {t.weight for t in blueprint.tiers} == {1.0}
    assert float(blueprint.score(panel).max().max()) >= spec.entry_score()


def test_funding_tiers_do_not_resolve_to_a_trend_signal():
    """`funding_bias` contains "bias"; a generic trend rule used to swallow it,
    which silently turned two different specs into the same strategy."""
    spec = load_case(_spec({"funding_bias": {"max_score": 2},
                            "trend_suitability": {"max_score": 3}}), name="order")
    resolved, _ = resolve_signals(spec)
    assert "funding" in resolved["funding_bias"].__name__
    assert "funding" not in resolved["trend_suitability"].__name__


def test_an_rsi_tier_is_not_mistaken_for_an_order_book_tier():
    """`rsi_depth` was rejected as needing an order book because of "depth"."""
    spec = load_case(_spec({"rsi_depth": {"max_score": 3},
                            "orderbook_imbalance": {"max_score": 2}}), name="depth")
    resolved, unavailable = resolve_signals(spec)
    assert "rsi" in resolved["rsi_depth"].__name__
    assert "orderbook_imbalance" in unavailable


def test_substituted_thresholds_are_recorded(panel):
    """Specs name their bands but not the boundaries; what was assumed must show."""
    result = run_case(load_case(PRICE_ONLY, name="synthetic"), panel)
    assert result.assumed
    assert any("bands" in key for key in result.assumed)


def test_coverage_is_reported_even_when_the_run_is_refused(panel):
    spec = load_case(_spec({"smart_money_inflow": {"max_score": 3},
                            "hotrank_attention": {"max_score": 3}}), name="none")
    result = run_case(spec, panel)
    assert not result.ran
    assert result.coverage["n_tiers"] == 2 and result.coverage["n_resolved"] == 0
    assert set(result.coverage["unavailable"]) == {"smart_money_inflow", "hotrank_attention"}


def test_the_backtest_uses_the_same_engine_as_everything_else(panel):
    result = run_case(load_case(PRICE_ONLY, name="synthetic"), panel)
    assert result.backtest.config["engine"] == "factor-eval.framework/weights-v1"
    assert list(result.backtest.metrics["split"]) == ["dev", "val", "oot"]
