"""调参这一步的契约。

这里钉住的每一条，错了都不会报错，只会让人报出一个更好看、更假的数字：漏记试验次数、
选择时读到未来、平台选择退化成 argmax、有效试验数按名义算、PBO 把"选得准"和"选得多"
搞混。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval import load_bundle                                          # noqa: E402
from factor_eval.blueprint import Blueprint, Exits, Tier                     # noqa: E402
from factor_eval.optimize import (Ledger, build_ledger, deflated_sharpe,     # noqa: E402
                                  effective_trials, ensemble_targets, expected_max_sharpe,
                                  grid_points, pbo, score_returns, select_trial,
                                  select_trials, sensitivity, smooth_scores, walk_forward)


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


def _ledger(shape, values, n_dates=600, seed=0):
    """按给定网格形状造一本账：每个候选的收益均值由 ``values`` 指定，噪声同分布。"""
    rng = np.random.default_rng(seed)
    axes = {f"a{i}": tuple(range(n)) for i, n in enumerate(shape)}
    points = grid_points(axes)
    index = pd.date_range("2022-01-01", periods=n_dates, freq="D", tz="UTC")
    data = {i: pd.Series(rng.normal(mu, 0.02, n_dates), index=index)
            for i, mu in enumerate(np.asarray(values, dtype=float).ravel())}
    return Ledger(axes=axes, points=points, returns=pd.DataFrame(data))


# ------------------------------------------------------------------ 账本
def test_a_failed_candidate_is_recorded_not_dropped(panel):
    """跑崩的候选必须留在账上。悄悄跳过会让试验计数偏小，DSR 就被高估。"""
    def build(lookback):
        if lookback == 2:
            raise RuntimeError("boom")
        close = panel.close
        return Blueprint(name="t", schedule=5, entry_score=1.0, direction="long_only",
                         tiers=[Tier("m", lambda p, n=lookback: close / close.shift(n) - 1,
                                     [(0.0, 9, 1)])])

    led = build_ledger(build, {"lookback": (1, 2, 3)}, panel)
    assert len(led) == 3, "网格大小不因失败而缩小"
    assert 1 in led.failures and "boom" in led.failures[1]
    assert led.returns[1].isna().all()


def test_warmup_is_not_in_the_ledger(panel):
    """预热期恒为空仓,留在账上会让训练窗里塞满没发生过的日子。"""
    close = panel.close
    led = build_ledger(lambda n: Blueprint(name="t", schedule=5, direction="long_only",
                                           tiers=[Tier("m", lambda p, k=n: close / close.shift(k) - 1,
                                                       [(0.0, 9, 1)])]),
                       {"n": (5, 10)}, panel, start="2023-01-01")
    assert led.returns.index[0] >= pd.Timestamp("2023-01-01", tz=panel.index.tz)


# --------------------------------------------------------------- 平台选择
def test_smoothing_demotes_an_isolated_spike():
    """孤峰 vs 平台：原始 argmax 选孤峰,平滑后必须改选平台。这是本模块的核心动作。"""
    surface = np.zeros((5, 5))
    surface[1:4, 1:4] = 1.0        # 一片平台
    surface[0, 4] = 2.0            # 一个孤峰,分更高
    led = _ledger((5, 5), surface)
    scores = pd.Series(surface.ravel(), index=range(25), dtype=float)
    assert select_trial(scores, led, rule="peak") == int(np.ravel_multi_index((0, 4), (5, 5)))
    plateau = select_trial(scores, led, rule="plateau")
    assert plateau != int(np.ravel_multi_index((0, 4), (5, 5)))
    assert surface.ravel()[plateau] == 1.0, "必须落在平台里"


def test_sensitivity_separates_a_peak_from_a_plateau():
    surface = np.zeros((5, 5))
    surface[1:4, 1:4] = 1.0
    surface[0, 4] = 2.0
    led = _ledger((5, 5), surface)
    scores = pd.Series(surface.ravel(), index=range(25), dtype=float)
    spike = sensitivity(scores, led, int(np.ravel_multi_index((0, 4), (5, 5))))
    plateau = sensitivity(scores, led, int(np.ravel_multi_index((2, 2), (5, 5))))
    assert spike["gap"] > plateau["gap"], "尖峰的点分远高于邻域均值,平台不会"
    assert plateau["neighbour_std"] < spike["neighbour_std"]


def test_a_flat_surface_makes_both_rules_agree():
    led = _ledger((3, 3), np.ones(9))
    scores = pd.Series(1.0, index=range(9))
    assert select_trial(scores, led, rule="plateau") is not None
    assert smooth_scores(scores, led).dropna().round(6).nunique() == 1


def test_corner_cells_are_not_favoured():
    """角落邻居少,不能因为"没人拉低它"就占便宜。"""
    led = _ledger((4, 4), np.arange(16))
    scores = pd.Series(np.nan, index=range(16))
    scores.iloc[0] = 10.0                       # 角落只有自己有分
    assert np.isnan(smooth_scores(scores, led).iloc[0])


def test_unknown_rule_is_rejected():
    led = _ledger((2, 2), np.ones(4))
    with pytest.raises(ValueError, match="rule"):
        select_trial(pd.Series(1.0, index=range(4)), led, rule="argmax")


# --------------------------------------------------------------- 滚动前推
def test_selection_never_sees_the_window_it_is_scored_on():
    """因果性：测试段必须整段落在训练段之后,并且隔开 embargo。"""
    led = _ledger((3, 3), np.linspace(0, 0.001, 9), n_dates=1200)
    wf = walk_forward(led, train=300, test=100, embargo=10)
    for _, row in wf.windows.iterrows():
        assert row["train_end"] < row["test_start"]
        gap = (row["test_start"] - row["train_end"]).days
        assert gap >= 10, f"embargo 不足: {gap} 天"


def test_out_of_sample_pieces_do_not_overlap():
    led = _ledger((3, 3), np.linspace(0, 0.001, 9), n_dates=1200)
    wf = walk_forward(led, train=300, test=100, embargo=10)
    assert wf.oos.index.is_monotonic_increasing and not wf.oos.index.duplicated().any()


def test_walk_forward_needs_enough_history():
    led = _ledger((2, 2), np.ones(4) * 0.001, n_dates=200)
    with pytest.raises(ValueError, match="训练"):
        walk_forward(led, train=300, test=100)


def test_ensembling_reduces_dispersion_of_the_outcome():
    """集成的本质是降方差：一堆同分布候选,平均之后的波动必须小于单个候选的中位波动。"""
    led = _ledger((4, 4), np.zeros(16), n_dates=900, seed=3)
    single = walk_forward(led, train=300, test=100, top=1).oos
    whole = walk_forward(led, train=300, test=100, top=16).oos
    assert whole.std() < single.std()


# ------------------------------------------------------------- 选择偏差统计
def test_effective_trials_collapses_duplicates():
    """同一条曲线复制 20 份不是 20 次独立试验。"""
    rng = np.random.default_rng(0)
    one = rng.normal(0, 0.02, 500)
    same = pd.DataFrame({i: one for i in range(20)})
    assert effective_trials(same) < 1.5


def test_effective_trials_counts_independent_columns():
    rng = np.random.default_rng(1)
    independent = pd.DataFrame(rng.normal(0, 0.02, (500, 20)))
    assert effective_trials(independent) > 15


def test_expected_max_sharpe_grows_with_the_search():
    assert expected_max_sharpe(1000, 0.05) > expected_max_sharpe(10, 0.05) > 0


def test_deflated_sharpe_punishes_the_best_of_many_noise_trials():
    """一堆纯噪声里挑最好的那条,DSR 必须低。这条不成立整个模块就没有意义。"""
    rng = np.random.default_rng(7)
    trials = pd.DataFrame(rng.normal(0, 0.02, (900, 200)))
    sharpes = trials.mean() / trials.std()
    best = trials[int(sharpes.idxmax())]
    lucky = deflated_sharpe(best, n_trials=200, sigma_trials=float(sharpes.std()))
    honest = deflated_sharpe(best, n_trials=1, sigma_trials=float(sharpes.std()))
    assert lucky["dsr"] < 0.5 < honest["dsr"], (lucky, honest)


def test_pbo_tells_a_real_edge_apart_from_a_lucky_winner():
    """全是噪声时"样本内冠军"在样本外就是抛硬币；有一个真候选时冠军稳定。

    单个种子上的噪声 PBO 方差很大（实测 0.21~0.71，因为 70 个组合复用同一批候选），
    所以这里钉的是**对比**而不是某个阈值 —— 对比才是 PBO 真正要回答的问题。
    """
    edge_values = np.zeros(16)
    edge_values[5] = 0.004                         # 一个真有边际的候选
    noise, edge = [], []
    for seed in range(5):
        noise.append(pbo(_ledger((4, 4), np.zeros(16), n_dates=800, seed=seed),
                         n_splits=8)["pbo"])
        edge.append(pbo(_ledger((4, 4), edge_values, n_dates=800, seed=seed),
                        n_splits=8)["pbo"])
    assert np.mean(edge) < 0.1 < np.mean(noise), (np.mean(edge), np.mean(noise))
    assert max(edge) < min(noise), "真候选的 PBO 必须整体低于纯噪声"


def test_pbo_rejects_an_odd_number_of_splits():
    led = _ledger((2, 2), np.zeros(4))
    with pytest.raises(ValueError, match="even"):
        pbo(led, n_splits=7)


# ------------------------------------------------------------------- 集成
def test_a_single_point_ensemble_is_that_blueprint(panel):
    """集成一个 = 它自己（摊平"留空=持有"之后）。差一点都说明加权口径错了。"""
    from factor_eval.blueprint import blueprint_targets
    close = panel.close

    def build(n):
        return Blueprint(name="t", schedule=5, entry_score=1.0, direction="long_only",
                         tiers=[Tier("m", lambda p, k=n: close / close.shift(k) - 1, [(0.0, 9, 1)])],
                         exits=Exits(stop_pct=0.1))

    one = ensemble_targets(build, [{"n": 20}], panel)
    direct = blueprint_targets(build(20), panel).ffill()
    pd.testing.assert_frame_equal(one, direct, check_dtype=False)


def test_ensemble_averages_the_two_members(panel):
    close = panel.close

    def build(n):
        return Blueprint(name="t", schedule=5, entry_score=1.0, direction="long_only",
                         tiers=[Tier("m", lambda p, k=n: close / close.shift(k) - 1, [(0.0, 9, 1)])])

    from factor_eval.blueprint import blueprint_targets
    both = ensemble_targets(build, [{"n": 10}, {"n": 40}], panel)
    mean = (blueprint_targets(build(10), panel).ffill()
            + blueprint_targets(build(40), panel).ffill()) / 2
    pd.testing.assert_frame_equal(both, mean, check_dtype=False)


def test_ensemble_holds_less_than_the_most_aggressive_member(panel):
    """平均之后的总敞口不可能超过成员里最大的那个 —— 超过了就是在悄悄加杠杆。"""
    close = panel.close

    def build(gross):
        return Blueprint(name="t", schedule=5, entry_score=1.0, direction="long_only",
                         gross=gross, cap=1.0,
                         tiers=[Tier("m", lambda p: close / close.shift(20) - 1, [(0.0, 9, 1)])])

    mixed = ensemble_targets(build, [{"gross": 0.5}, {"gross": 1.0}], panel)
    assert float(mixed.abs().sum(axis=1).max()) <= 1.0 + 1e-9


# ----------------------------------------------------------------- 目标函数
def test_short_samples_get_no_score():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 10))
    assert np.isnan(score_returns(r))


def test_calmar_needs_a_drawdown():
    r = pd.Series(0.001, index=pd.date_range("2022-01-01", periods=100, freq="D"))
    assert np.isnan(score_returns(r, "calmar")), "从没回撤过,Calmar 没有定义"


def test_unknown_objective_is_rejected():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    with pytest.raises(ValueError, match="objective"):
        score_returns(r, "profit")


def test_select_trials_returns_them_in_rank_order():
    led = _ledger((3, 3), np.arange(9))
    scores = pd.Series(np.arange(9, dtype=float), index=range(9))
    picks = select_trials(scores, led, rule="peak", top=3)
    assert picks == [8, 7, 6]
