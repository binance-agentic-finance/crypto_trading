"""策略蓝图的契约：三类策略必须能写出来，而且每个部件的语义不能漂。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.eval import load_bundle                                        # noqa: E402
from cyqnt_trd.eval.blueprint import (Blueprint, Exits, Gate, Tier,           # noqa: E402
                                   apply_exits, compile_targets, run_blueprint)


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


def const(value):
    return lambda p: pd.DataFrame(float(value), index=p.index, columns=p.symbols)


# ------------------------------------------------------------------ Gate
def test_gate_never_passes_a_missing_value(panel):
    """缺失必须挡住。放行 NaN 等于让一个还没有数据的标的进场。"""
    sparse = panel.close.copy()
    sparse.iloc[:, 0] = np.nan
    gate = Gate("has_data", lambda p: sparse, ">=", 0.0)
    assert not gate.mask(panel).iloc[:, 0].any()
    assert gate.mask(panel).iloc[:, 1].any()


def test_gate_is_hard_not_a_penalty(panel):
    """硬门不过 = 出局,不是扣分。分数可以被别的维度补回来,硬门不行。"""
    bp = Blueprint(name="t", tiers=[Tier("always", const(10), [(-1e9, 1e9, 10)])],
                   gates=[Gate("never", const(0), ">=", 1.0)])
    assert bp.score(panel).notna().to_numpy().sum() == 0


# ------------------------------------------------------------------ Tier
def test_tier_takes_the_first_matching_band(panel):
    tier = Tier("t", const(5), [(0, 10, 1), (0, 10, 99)])
    assert (tier.score(panel).dropna() == 1).all().all()


def test_tier_scores_zero_outside_every_band(panel):
    tier = Tier("t", const(50), [(0, 10, 3)])
    assert (tier.score(panel).dropna() == 0).all().all()


def test_raw_bands_break_when_the_factor_scale_changes(panel):
    """同一套 band 换个量纲就废 —— 这正是接"别人给的因子"时不能用 raw 的原因。"""
    from cyqnt_trd.eval.examples import example_factors as ex
    band = [(0.5, 9e9, 1)]
    counts = {}
    for name in ("reversal_5d", "low_volatility", "volume_shock"):
        f = getattr(ex, name)(panel)
        tier = Tier("f", lambda q, g=f: g, band)
        counts[name] = int((tier.score(panel).fillna(0) > 0).to_numpy().sum())
    assert counts["low_volatility"] == 0, "全负值的因子在这条 band 上一次都不会触发"
    assert max(counts.values()) > 1000, "另一些因子却几乎天天触发"


def test_rank_bands_mean_the_same_thing_for_every_factor(panel):
    """排名之后 band 的含义是"在截面里排多前",与因子量纲无关。"""
    from cyqnt_trd.eval.examples import example_factors as ex
    counts = []
    for name in ("reversal_5d", "low_volatility", "volume_shock"):
        f = getattr(ex, name)(panel)
        tier = Tier("f", lambda q, g=f: g, [(0.5, 9e9, 1)], normalize="rank")
        counts.append(int((tier.score(panel).fillna(0) > 0).to_numpy().sum()))
    assert min(counts) > 0
    assert max(counts) / min(counts) < 1.5, f"各因子入选量应当同量级: {counts}"


def test_rank_normalisation_does_not_peek(panel):
    """截面标准化只用当天同一批标的。截断未来不能改变此前任何一天的分数。"""
    from cyqnt_trd.eval.examples import example_factors as ex
    f = ex.volume_shock(panel)
    cut = panel.__class__(**{name: getattr(panel, name).iloc[:800] for name in
                             ("open", "high", "low", "close", "volume", "quote_volume",
                              "funding", "mask")}, meta=panel.meta)
    tier = Tier("f", lambda q, g=f: g.iloc[:len(q.index)], [(0.0, 9e9, 1)], normalize="rank")
    pd.testing.assert_frame_equal(tier.score(panel).iloc[:800], tier.score(cut),
                                  check_dtype=False)


def test_unknown_normalisation_is_rejected(panel):
    with pytest.raises(ValueError, match="normalize"):
        Tier("t", const(5), [(0, 10, 1)], normalize="minmax").score(panel)


def test_tier_weight_scales_the_points(panel):
    a = Tier("a", const(5), [(0, 10, 2)]).score(panel)
    b = Tier("b", const(5), [(0, 10, 2)], weight=3.0).score(panel)
    assert np.allclose(b.dropna(how="all").fillna(0), (a * 3).dropna(how="all").fillna(0))


# --------------------------------------------------------------- schedule
def test_non_evaluation_days_are_left_blank_to_hold(panel):
    bp = Blueprint(name="t", tiers=[Tier("s", const(5), [(0, 10, 5)])], schedule=5,
                   entry_score=1.0)
    targets = compile_targets(bp, panel)
    filled = targets.notna().any(axis=1)
    assert filled.iloc[::5].all() and not filled.iloc[1::5].any()


def test_event_trigger_fires_only_on_its_own_days(panel):
    fire = pd.Series(False, index=panel.index)
    fire.iloc[[10, 200, 900]] = True
    bp = Blueprint(name="t", tiers=[Tier("s", const(5), [(0, 10, 5)])], schedule=0,
                   trigger=lambda p: fire, entry_score=1.0)
    filled = compile_targets(bp, panel).notna().any(axis=1)
    assert filled.sum() == 3 and filled.iloc[[10, 200, 900]].all()


# ----------------------------------------------------------------- sizing
def test_risk_budget_sizes_from_the_stop_distance(panel):
    """现成策略最常见的写法:先定这笔最多亏多少,再由止损距离反推名义。"""
    bp = Blueprint(name="t", tiers=[Tier("s", const(5), [(0, 10, 5)])], schedule=1,
                   direction="long_only", sizing="risk_budget", risk_per_trade=0.02,
                   exits=Exits(stop_pct=0.05), gross=10.0, cap=1.0)
    w = compile_targets(bp, panel).dropna(how="all")
    assert np.allclose(w[w != 0].stack().unique(), 0.02 / 0.05)


def test_top_k_keeps_only_the_best_names(panel):
    score = panel.close.rank(axis=1)
    bp = Blueprint(name="t", tiers=[Tier("s", lambda p: score, [(-1e9, 1e9, 1)])],
                   schedule=1, top_k=2, entry_score=0.0, direction="long_only")
    held = (compile_targets(bp, panel).abs() > 0).sum(axis=1)
    assert held.max() <= 2


def test_neutral_makes_the_book_dollar_neutral(panel):
    bp = Blueprint(name="t", tiers=[Tier("s", lambda p: panel.close, [(-1e9, 1e9, 1)])],
                   schedule=1, entry_score=0.0, neutral=True, direction="score",
                   sizing="equal", cap=1.0)
    net = compile_targets(bp, panel).dropna(how="all").sum(axis=1).abs()
    assert float(net.max()) < 1e-9


# ------------------------------------------------------------------ exits
def test_stop_flattens_the_position(panel):
    first = panel.symbols[0]
    targets = pd.DataFrame(np.nan, index=panel.index, columns=panel.symbols)
    targets.iloc[100, targets.columns.get_loc(first)] = 1.0
    out = apply_exits(targets, panel, Exits(stop_pct=0.001))   # 几乎必然触发
    after = out[first].iloc[101:200]
    assert (after == 0).any(), "止损触发当天必须把目标置 0"


def test_max_holding_days_forces_an_exit(panel):
    first = panel.symbols[0]
    targets = pd.DataFrame(np.nan, index=panel.index, columns=panel.symbols)
    targets.iloc[100, targets.columns.get_loc(first)] = 1.0
    out = apply_exits(targets, panel, Exits(max_holding_days=5))
    assert out[first].iloc[105] == 0


def test_no_exit_rules_leaves_targets_untouched(panel):
    targets = pd.DataFrame(np.nan, index=panel.index, columns=panel.symbols)
    targets.iloc[100] = 0.1
    assert apply_exits(targets, panel, Exits()).equals(targets)


# ------------------------------------------------------- 三类策略都写得出来
def test_the_three_shapes_all_compile_and_run(panel):
    """规则型 / 事件驱动型 / 量价截面型 —— 同一副骨架,只是填的内容不同。
    这条测试就是框架的验收条件本身。"""
    close = panel.close
    momentum = close / close.shift(4) - 1
    rule = Blueprint(name="rule", schedule=1, entry_score=1.0, direction="long_only",
                     sizing="risk_budget", risk_per_trade=0.05, cap=1.0,
                     tiers=[Tier("mom", lambda p: momentum, [(0, 9, 2), (-9, 0, -1)])],
                     exits=Exits(stop_pct=0.08, take_profit_pct=0.25, max_holding_days=30))
    event = Blueprint(name="event", schedule=0, entry_score=1.0, sizing="equal", gross=0.5,
                      trigger=lambda p: (momentum.abs() >= 0.15),
                      tiers=[Tier("shock", lambda p: momentum, [(.15, 9, 2), (-9, -.15, -2)])],
                      exits=Exits(stop_pct=0.06, max_holding_days=5))
    cross = Blueprint(name="cross", schedule=5, entry_score=2.0, top_k=3,
                      direction="long_only", sizing="inverse_vol", gross=0.9,
                      gates=[Gate("liquid", lambda p: p.quote_volume, ">=", 5e7)],
                      tiers=[Tier("mom", lambda p: momentum, [(.02, 9, 3), (-9, .02, 0)])],
                      exits=Exits(stop_pct=0.10, max_holding_days=15))
    for bp in (rule, event, cross):
        out = run_blueprint(bp, panel)
        assert out["book"].equity.dropna().gt(0).all(), f"{bp.name}: 权益必须为正"
        assert np.isfinite(out["book"].metrics["ann_return"]), f"{bp.name}: 指标必须可算"
        assert out["targets"].abs().sum().sum() > 0, f"{bp.name}: 必须真的建过仓"


def test_blueprint_needs_at_least_one_tier(panel):
    with pytest.raises(ValueError, match="tier"):
        Blueprint(name="empty", tiers=[]).score(panel)


def test_warmup_is_not_traded(panel):
    bp = Blueprint(name="t", tiers=[Tier("s", const(5), [(0, 10, 5)])], schedule=1,
                   entry_score=1.0, direction="long_only")
    out = run_blueprint(bp, panel, start="2022-04-01")
    before = out["targets"].loc[:pd.Timestamp("2022-03-31", tz=panel.index.tz)]
    assert before.notna().to_numpy().sum() == 0, "预热期截面太窄,不能拿去交易"