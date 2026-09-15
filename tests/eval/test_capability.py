"""把 capability 节点输出接进矩阵时必须守住的契约。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval import evaluate, load_bundle  # noqa: E402
from factor_eval.capability import capability_factor, read_port  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


# ------------------------------------------------------------------ read_port
def test_read_port_handles_every_manifest_shape():
    assert read_port(0.5) == .5                              # 裸值
    assert read_port({"sma": .5}) == .5                       # 单端口
    assert read_port({"value": .5, "reasons": None}) == .5    # 多端口默认取 value
    assert read_port({"value": .5, "series": [1, 2]}, "series") == [1, 2]


def test_read_port_refuses_to_guess():
    with pytest.raises(KeyError):
        read_port({"a": 1, "b": 2})              # 多端口且没有 value -> 不猜
    with pytest.raises(KeyError):
        read_port({"value": 1}, "series")        # 指名了不存在的端口


# --------------------------------------------------------------- value port
def test_value_port_sees_only_the_past(panel):
    """滚动窗口按构造不可能看到未来:节点拿到的最后一个值必须正好是当日收盘。"""
    factor = capability_factor(lambda series: {"value": series[-1]},
                               inputs={"series": "close"}, window=2, output="value")
    out = factor(panel)
    close = panel.close
    both = out.notna() & close.notna()
    assert both.to_numpy().sum() > 1000
    assert np.allclose(out.where(both).to_numpy()[both.to_numpy()],
                       close.where(both).to_numpy()[both.to_numpy()])


def test_value_port_window_length_is_respected(panel):
    seen = []
    capability_factor(lambda series: (seen.append(len(series)), 0.0)[1],
                      inputs={"series": "close"}, window=7)(panel)
    assert max(seen) == 7, "节点看到的窗口不能超过 window"


def test_value_port_rejects_a_sequence_with_a_useful_message(panel):
    factor = capability_factor(lambda series: {"value": [1.0, 2.0]},
                               inputs={"series": "close"}, window=3, output="value")
    with pytest.raises(RuntimeError, match="series form"):
        factor(panel)


# -------------------------------------------------------------- series port
def test_series_port_aligns_to_the_tail_not_the_head(panel):
    """短序列必须右对齐。左对齐会把整条曲线前挪,变成实打实的未来函数。"""
    n = len(panel.index)
    factor = capability_factor(lambda signals: {"series": [1.0] * (n - 10)},
                               inputs={"signals": "close"}, output="series")
    out = factor(panel)
    column = out[panel.symbols[0]]
    assert column.iloc[:10].isna().all(), "预热期缺的值应落在开头"
    assert (column.iloc[10:] == 1.0).all()


def test_series_port_refuses_more_values_than_bars(panel):
    factor = capability_factor(lambda signals: {"series": [1.0] * (len(panel.index) + 5)},
                               inputs={"signals": "close"}, output="series")
    with pytest.raises(ValueError, match="values for"):
        factor(panel)


def test_series_port_maps_none_to_missing(panel):
    n = len(panel.index)
    factor = capability_factor(lambda signals: {"series": [None] * 5 + [2.0] * (n - 5)},
                               inputs={"signals": "close"}, output="series")
    column = factor(panel)[panel.symbols[0]]
    assert column.iloc[:5].isna().all() and (column.iloc[5:] == 2.0).all()


# ------------------------------------------------------------------- inputs
def test_multiple_input_series_are_passed_by_name(panel):
    def node(high, low, close):
        # 适配器忠实地传 list[float | None]:标的还没上市的日期就是 None,
        # 节点必须自己处理,就像它在平台上收到的一样。
        h, l, c = high[-1], low[-1], close[-1]
        if None in (h, l, c) or h == l:
            return {"value": None}
        return {"value": (c - l) / (h - l)}

    out = capability_factor(node, inputs={"high": "high", "low": "low", "close": "close"},
                            window=2, output="value")(panel)
    assert out.notna().to_numpy().sum() > 1000
    finite = out.to_numpy()[np.isfinite(out.to_numpy())]
    assert finite.min() >= -1e-9 and finite.max() <= 1 + 1e-9


def test_unknown_panel_field_is_rejected():
    with pytest.raises(ValueError, match="unknown panel fields"):
        capability_factor(lambda series: 0.0, inputs={"series": "funding_rate"})


def test_a_name_cannot_be_both_series_and_fixed_param():
    with pytest.raises(ValueError, match="both"):
        capability_factor(lambda series: 0.0, inputs={"series": "close"},
                          params={"series": 3})


def test_window_must_be_a_sane_integer():
    for bad in (1, 0, -3, True):
        with pytest.raises(ValueError, match="window"):
            capability_factor(lambda series: 0.0, window=bad)


# ------------------------------------------------------------------ failures
def test_node_failure_surfaces_instead_of_becoming_a_silent_constant(panel):
    """节点抛错默认必须炸出来。悄悄填 0 会让一个坏因子看起来像常数因子。"""
    def broken(series):
        raise ZeroDivisionError("boom")

    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        capability_factor(broken, inputs={"series": "close"}, window=3)(panel)


def test_failures_can_be_skipped_explicitly(panel):
    def sometimes(series):
        if len(series) and series[-1] is not None and series[-1] > 0:
            raise ValueError("no")
        return 0.0

    out = capability_factor(sometimes, inputs={"series": "close"}, window=3,
                            skip_failures=True)(panel)
    assert out.isna().to_numpy().mean() > .5


# --------------------------------------------------------------- end to end
def test_a_capability_factor_scores_through_the_normal_path(panel):
    """适配之后走的仍是同一条 evaluate():没有第二套回测或权重口径。"""
    def ratio(series, period):
        values = [v for v in series if v is not None]
        if len(values) < period:
            return {"value": None}
        return {"value": values[-1] / (sum(values[-period:]) / period) - 1}

    card = evaluate(capability_factor(ratio, inputs={"series": "close"},
                                      params={"period": 20}, window=20, output="value"),
                    panel, name="cap", with_incremental=False, with_diagnostics=False)
    assert card.verdict in {"PASS", "PASS_CONDITIONAL", "HOLD_INFO", "HOLD_WEAK",
                            "HOLD_INCOMPLETE", "HOLD_SEARCH", "REJECT", "REJECT_DATA"}
    assert card.gates["G0_data"].status != "FAIL", "20 日均线偏离应当是可排序的"
