"""`limits.liquidation_check` 的维持保证金率与 `sizing.grid_levels` 的等比网格(BIA-3342)。"""
from __future__ import annotations

import math

import pytest

from cyqnt_trd.blocks.limits import liquidation_check
from cyqnt_trd.blocks.sizing import grid_levels


# ── liquidation_check ───────────────────────────────────────────────────────

def test_default_mmr_keeps_the_old_estimate():
    out = liquidation_check(60000, 59000, "LONG", 10)
    assert out["liquidation_price_est"] == pytest.approx(54000.0)
    assert out["liq_distance_pct"] == pytest.approx(10.0)


@pytest.mark.parametrize("direction, expected", [
    # Binance isolated, first bracket: maintenance margin is charged on the notional at the
    # liquidation price, hence the division by (1 ∓ MMR) — not entry × (1 − 1/L + MMR) = 54240.
    ("LONG", 60000 * (1 - 1 / 10) / (1 - 0.004)),     # 54216.87
    ("SHORT", 60000 * (1 + 1 / 10) / (1 + 0.004)),    # 65737.05
])
def test_maintenance_margin_rate_moves_liquidation_toward_entry(direction, expected):
    out = liquidation_check(60000, 60000, direction, 10, maintenance_margin_rate=0.004)
    assert out["liquidation_price_est"] == pytest.approx(expected, rel=1e-9)
    assert out["liq_distance_pct"] < 10.0


def test_margin_balance_equals_maintenance_margin_at_the_estimate():
    """Counter-check from first principles, not from the formula under test."""
    entry, lev, mmr, qty = 60000.0, 10, 0.004, 1.0
    liq = liquidation_check(entry, entry, "LONG", lev, maintenance_margin_rate=mmr)["liquidation_price_est"]
    margin_balance = qty * entry / lev + qty * (liq - entry)
    assert margin_balance == pytest.approx(qty * liq * mmr, rel=1e-9)


def test_mmr_at_or_above_initial_margin_is_liquidatable_at_entry():
    out = liquidation_check(60000, 60000, "LONG", 100, maintenance_margin_rate=0.02)
    assert out["liq_distance_pct"] <= 0
    assert out["is_danger"] is True


@pytest.mark.parametrize("mmr", [-0.001, 1.0, 1.5])
def test_out_of_range_mmr_is_invalid(mmr):
    assert liquidation_check(60000, 60000, "LONG", 10, maintenance_margin_rate=mmr) == {
        "is_danger": False, "reason": "Invalid parameters"}


# ── grid_levels ─────────────────────────────────────────────────────────────

def test_legacy_positional_call_is_unchanged():
    assert [p for p, _ in grid_levels(100.0, 0.05, 5, 10.0)] == pytest.approx([95, 97.5, 100, 102.5, 105])


def test_geometric_step_pct_between_bounds():
    """单子里的用例:区间 0.6255–1.11、每格 0.88% 等比 → 66 条线,实际步长约 0.8863%。"""
    levels = grid_levels(lower=0.6255, upper=1.11, spacing="geometric", step_pct=0.0088,
                         per_grid_notional=237.29)
    prices = [p for p, _ in levels]
    assert len(prices) == 66
    assert prices[0] == pytest.approx(0.6255) and prices[-1] == pytest.approx(1.11)
    ratios = [b / a for a, b in zip(prices, prices[1:])]
    assert all(r == pytest.approx(ratios[0], rel=1e-12) for r in ratios)
    assert ratios[0] - 1 == pytest.approx(0.008863, abs=1e-6)
    assert ratios[0] - 1 >= 0.0088
    assert sum(p > 0.66 for p in prices) == 59
    assert {n for _, n in levels} == {237.29}


def test_geometric_n_grids_around_center():
    prices = [p for p, _ in grid_levels(100.0, 0.5, 3, 1.0, spacing="geometric")]
    assert prices == pytest.approx([50.0, math.sqrt(50.0 * 150.0), 150.0])


@pytest.mark.parametrize("kwargs, message", [
    (dict(lower=1, upper=2, n_grids=3, step_pct=0.01, spacing="geometric"), "n_grids or step_pct"),
    (dict(center_price=1, range_pct=0.1, lower=1, upper=2, n_grids=3), "center_price + range_pct or lower + upper"),
    (dict(lower=1, upper=2, step_pct=0.01), "spacing='geometric'"),
    (dict(lower=2, upper=1, n_grids=3), "0 < lower < upper"),
    (dict(lower=1, upper=1.005, step_pct=0.01, spacing="geometric"), "wider than the whole range"),
    (dict(lower=1, upper=2, n_grids=3, spacing="log"), "spacing must be"),
])
def test_rejects_ambiguous_or_impossible_grids(kwargs, message):
    with pytest.raises(ValueError, match=message.replace("+", r"\+").replace("(", r"\(")):
        grid_levels(**kwargs)
