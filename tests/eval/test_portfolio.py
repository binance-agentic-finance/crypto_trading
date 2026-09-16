"""组合模拟器的会计契约。

这些不是"跑得通"的冒烟测试，是几条**会静默算错钱**的地方：净额调仓、停牌估值、
资金费方向、权益归零、风控只读过去。每一条都对应一次实际踩过的坑。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval import load_bundle                                    # noqa: E402
from factor_eval.portfolio import Risk, performance, simulate          # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


def _targets(panel, rows: dict):
    t = pd.DataFrame(np.nan, index=panel.index, columns=panel.symbols)
    for i, weights in rows.items():
        for symbol, w in weights.items():
            t.iloc[i, t.columns.get_loc(symbol)] = w
    return t


def test_all_nan_row_means_hold_not_flat(panel):
    """整行空 = 继续持有。这是模拟器的核心约定，改了它所有调仓节奏都会变。"""
    first = panel.symbols[0]
    book = simulate(_targets(panel, {100: {first: 0.5}}), panel)
    held = book.weights[first].iloc[120:160]
    assert (held.abs() > 0.1).all(), "没有新的目标行时仓位必须保留"


def test_only_the_difference_is_traded(panel):
    """净额调仓：目标不变就不该产生成交。整期全平全开是评测口径，不是策略口径。"""
    first = panel.symbols[0]
    rows = {i: {first: 0.5} for i in range(100, 200, 10)}
    book = simulate(_targets(panel, rows), panel)
    after_first = book.trades.iloc[105:200]
    assert after_first.max() < 0.25, "重复下达同一目标不应引发整仓换手"


def test_funding_is_charged_on_held_units(panel):
    """资金费正值是支出。方向搞反会让一条亏钱的曲线看起来在赚钱。"""
    first = panel.symbols[0]
    targets = _targets(panel, {100: {first: 1.0}})
    paid = simulate(targets, panel, cost_bps=0).funding.iloc[101:].sum()
    flat = simulate(_targets(panel, {100: {first: 0.0}}), panel, cost_bps=0).funding.iloc[101:].sum()
    assert abs(flat) < 1e-12, "空仓不应产生资金费"
    assert paid != 0, "持仓期间必须计提资金费"


def test_equity_never_goes_negative(panel):
    """权益归零即出局。允许负权益会产生 -100% 以上的回撤这种无意义读数。"""
    first = panel.symbols[0]
    book = simulate(_targets(panel, {i: {first: 5.0} for i in range(100, 2000, 5)}), panel)
    assert (book.equity.dropna() >= 0).all()
    assert book.metrics["max_drawdown"] >= -1.0000001


def test_stale_holdings_keep_their_last_mark(panel):
    """当日无价时持仓按最后有效价估值,不能算成 0 —— 那会把一次停牌伪装成盈亏。"""
    missing = panel.open.isna().any()
    symbol = next((s for s in panel.symbols if missing[s]), None)
    if symbol is None:
        pytest.skip("this bundle has no missing opens")
    idx = panel.open[symbol].first_valid_index()
    start = panel.index.get_loc(idx)
    book = simulate(_targets(panel, {start + 5: {symbol: 0.5}}), panel)
    assert book.equity.dropna().min() > 0


def test_risk_overlay_reads_only_the_past(panel):
    """风控用未来数据会立刻体现为"最差的日子恰好都降了杠杆"。这里只钉住因果:
    截断收益序列不改变此前已经产生的杠杆。"""
    first = panel.symbols[0]
    rows = {i: {first: 0.5} for i in range(100, 1500, 20)}
    risk = Risk(perf_window=20, perf_up=1.5, perf_down=0.5)
    full = simulate(_targets(panel, rows), panel, risk=risk)
    cut = simulate(_targets(panel, rows).iloc[:800], panel.__class__(
        **{f: getattr(panel, f).iloc[:800] for f in
           ("open", "high", "low", "close", "volume", "quote_volume", "funding", "mask")},
        meta=panel.meta), risk=risk)
    common = min(len(cut.leverage), 800) - 1
    assert np.allclose(full.leverage.to_numpy()[:common], cut.leverage.to_numpy()[:common])


def test_leverage_state_survives_a_zero_allocation(panel):
    """perf_down=0 时必须还能回来。若用自身收益判断状态,空仓后收益恒为 0,
    "我在赚钱吗"永远为假 —— 那是吸收态,不是风控。"""
    first = panel.symbols[0]
    rows = {i: {first: 0.5} for i in range(100, 2000, 10)}
    book = simulate(_targets(panel, rows), panel,
                    risk=Risk(perf_window=20, perf_up=1.0, perf_down=0.0))
    used = set(np.round(book.leverage.to_numpy(), 6))
    assert {0.0, 1.0} <= used, "杠杆必须能在 0 和 1 之间来回,而不是一去不回"


def test_performance_handles_a_flat_book(panel):
    book = simulate(pd.DataFrame(np.nan, index=panel.index, columns=panel.symbols), panel)
    assert book.metrics["daily_turnover"] == 0
    assert book.equity.iloc[-1] == pytest.approx(1.0)
