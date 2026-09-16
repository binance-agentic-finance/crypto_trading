"""完整链路的契约：评测 → 准入 → 合成 → 权重 → 调仓信号。

这里最要紧的两条是**不变量**而不是数值：同一个因子换个正负号、换个量纲，构建出来的
权重必须一模一样。做不到这两条，"不管用户丢什么因子进来"就只是一句话。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval import load_bundle                                        # noqa: E402
from factor_eval.examples import example_factors as ex                     # noqa: E402
from factor_eval.pipeline import (admit, build_strategy, cards_frame,      # noqa: E402
                                  combine_scores, latest_orders, orders, screen)


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


@pytest.fixture(scope="module")
def factors(panel):
    close = panel.close
    return {"mom_20": close / close.shift(20) - 1,
            "vol_20": close.pct_change(fill_method=None).rolling(20).std(),
            "range_10": ((panel.high - panel.low) / close).rolling(10).mean()}


@pytest.fixture(scope="module")
def strategy(panel, factors):
    return build_strategy(factors, panel)


# ------------------------------------------------- 不管来的是什么因子
def test_the_authors_sign_does_not_matter(panel, factors):
    """同一个因子写成 f 或 -f，构建出来的权重必须相同 —— 方向由评测的 dev 段冻结，
    不由写因子的人顺手定的正负号决定。这条不成立，等于有一半因子在反着做。"""
    flipped = {name: -frame for name, frame in factors.items()}
    a = build_strategy(factors, panel).weights
    b = build_strategy(flipped, panel).weights
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_the_factor_scale_does_not_matter(panel, factors):
    """乘个 1000 倍不该改变任何仓位 —— 合成之前先做截面排名就是为了这个。"""
    scaled = {name: frame * 1000.0 for name, frame in factors.items()}
    pd.testing.assert_frame_equal(build_strategy(factors, panel).weights,
                                  build_strategy(scaled, panel).weights, check_dtype=False)


def test_a_recomputing_callable_matches_the_precomputed_frame(panel, factors):
    """真正按面板重算的 callable，与预先算好的 DataFrame，必须得到同一组权重。"""
    as_callable = {
        "mom_20": lambda p: p.close / p.close.shift(20) - 1,
        "vol_20": lambda p: p.close.pct_change(fill_method=None).rolling(20).std(),
        "range_10": lambda p: ((p.high - p.low) / p.close).rolling(10).mean(),
    }
    pd.testing.assert_frame_equal(build_strategy(factors, panel).weights,
                                  build_strategy(as_callable, panel).weights, check_dtype=False)


def test_a_constant_callable_fails_the_causality_check(panel, factors):
    """把一张算好的全历史表包进 ``lambda p: frame`` **应当**被判 FAIL —— 给它一段前缀，
    它照样吐出全历史，那就是真的读了未来。这与"直接传 DataFrame"（无法验证，记 N/A）
    是两回事，不能混。"""
    frame = factors["mom_20"]
    cards = admit(screen({"canned": lambda p, f=frame: f}, panel))
    assert cards["canned"].gate_status("G0_data") == "FAIL"
    assert cards["canned"].admitted is False

    passed_directly = admit(screen({"canned": frame}, panel))
    assert passed_directly["canned"].gate_status("G0_data") == "N/A"
    assert passed_directly["canned"].admitted is True
    assert passed_directly["canned"].unverified is True


# ------------------------------------------------------------ 评测驱动准入
def test_a_look_ahead_factor_is_blocked_by_the_data_gate(panel, factors):
    """偷看未来的因子必须被 G0 拦住,而且真的不参与合成。"""
    polluted = dict(factors, trap=ex.look_ahead_trap)
    out = build_strategy(polluted, panel)
    assert out.cards["trap"].admitted is False
    assert "G0_data" in out.cards["trap"].blocking
    pd.testing.assert_frame_equal(out.weights, build_strategy(factors, panel).weights,
                                  check_dtype=False)


def test_direction_is_frozen_on_dev_only(panel, factors):
    """方向必须来自 dev 段。与构建层既有的 frozen_sign 独立实现对齐。"""
    from factor_eval.strategy import frozen_sign
    cards = screen(factors, panel)
    for name, frame in factors.items():
        assert cards[name].direction == frozen_sign(frame, panel, h=3)


def test_an_ic_floor_can_reject_factors(panel, factors):
    cards = admit(screen(factors, panel), min_ic_dev=0.99)
    assert not any(c.admitted for c in cards.values())
    with pytest.raises(ValueError, match="准入"):
        combine_scores(factors, cards, panel)


def test_cards_frame_lists_every_factor(panel, factors):
    frame = cards_frame(screen(factors, panel))
    assert set(frame.index) == set(factors)
    assert {"verdict", "direction", "ic_dev", "admitted"} <= set(frame.columns)


# ----------------------------------------------------------------- 权重
def test_weights_respect_gross_cap_and_neutrality(strategy):
    w = strategy.weights.dropna(how="all")
    live = w[w.abs().sum(axis=1) > 0]
    assert np.allclose(live.abs().sum(axis=1), 1.0), "总敞口必须归一到 gross"
    assert float(w.abs().max().max()) <= 0.35 + 1e-9, "单标的不得超过 cap"
    assert float(live.sum(axis=1).abs().max()) < 1e-9, "neutral=True 时净敞口为 0"


def test_only_rebalance_days_carry_targets(panel, factors):
    w = build_strategy(factors, panel, rebalance=5).weights
    filled = w.notna().any(axis=1)
    gaps = np.diff(np.flatnonzero(filled.to_numpy()))
    assert set(gaps) <= {5}, "调仓日必须严格按节奏落点"


def test_warmup_is_not_traded(panel, factors):
    w = build_strategy(factors, panel, start="2023-01-01").weights
    before = w.loc[w.index < pd.Timestamp("2023-01-01", tz=panel.index.tz)]
    assert before.notna().to_numpy().sum() == 0


def test_ic_weighting_changes_the_book(panel, factors):
    """加权方案要真的起作用 —— 否则参数只是摆设。"""
    equal = build_strategy(factors, panel, scheme="equal").weights
    by_ic = build_strategy(factors, panel, scheme="ic").weights
    assert not equal.equals(by_ic)


def test_unknown_scheme_is_rejected(panel, factors):
    with pytest.raises(ValueError, match="scheme"):
        build_strategy(factors, panel, scheme="magic")


# ------------------------------------------------------------- 调仓信号
def test_orders_come_from_the_book_that_was_actually_traded(strategy, panel):
    """清单必须与模拟器那本账对得上。另算一遍就会出现"报表上的单子"与
    "回测真交的单子"不一致,而这种不一致只会在上线之后才暴露。"""
    table = orders(strategy.book, panel, min_weight=0.0)
    by_day = table.groupby("date")["delta_weight"].apply(lambda s: s.abs().sum())
    expected = strategy.book.trades.reindex(by_day.index)
    assert np.allclose(by_day.to_numpy(), expected.to_numpy(), atol=1e-9)


def test_every_order_has_a_side_and_a_reference_price(strategy):
    table = strategy.orders
    assert not table.empty
    assert set(table["side"]) <= {"BUY", "SELL"}
    assert (np.sign(table["delta_weight"]) == table["side"].map({"BUY": 1, "SELL": -1})).all()
    assert table["ref_price"].gt(0).all(), "参考成交价必须可对盘"


def test_small_orders_are_filtered(strategy, panel):
    coarse = orders(strategy.book, panel, min_weight=0.05)
    assert coarse["delta_weight"].abs().min() >= 0.05
    assert len(coarse) < len(orders(strategy.book, panel, min_weight=0.0))


def test_latest_orders_is_one_day(strategy, panel):
    last = latest_orders(strategy.book, panel)
    assert last["date"].nunique() == 1
    assert last["date"].iloc[0] == strategy.orders["date"].max()


def test_notional_scales_with_the_account(strategy, panel):
    one = orders(strategy.book, panel, equity=1.0)["notional"].abs().sum()
    hundred = orders(strategy.book, panel, equity=100.0)["notional"].abs().sum()
    assert np.isclose(hundred, one * 100.0)
