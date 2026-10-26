"""构建层的契约:合成、定仓、在线选因子。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.eval import load_bundle                                       # noqa: E402
from cyqnt_trd.eval.examples.example_factors import low_volatility, volume_shock  # noqa: E402
from cyqnt_trd.eval.strategy import (combine, factor_ic_series, frozen_sign,  # noqa: E402
                                  naive_weights, realised_vol, target_weights,
                                  walk_forward)


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


def test_cap_applies_to_weights_not_to_raw_scores(panel):
    """上限是"单个标的最多占权益多少",必须在归一化**之后**才有意义。
    先截再归一会把标准差单位的分数压平,信号凭空少掉一大截,而权重看着还正常。"""
    score = low_volatility(panel)
    loose = target_weights(score, panel, cap=1.0, rebalance=5)
    assert float(np.nanmax(np.abs(loose.to_numpy()))) < 0.35
    tight = target_weights(score, panel, cap=0.35, rebalance=5)
    assert np.allclose(loose.fillna(0), tight.fillna(0)), "上限不生效时必须与不设上限完全一致"


def test_cap_binds_when_it_should(panel):
    w = target_weights(low_volatility(panel), panel, cap=0.12, rebalance=5)
    assert float(np.nanmax(np.abs(w.to_numpy()))) <= 0.12 + 1e-9


def test_thin_cross_sections_go_flat_not_stale(panel):
    """名字不够的调仓日必须清仓。留空在模拟器里等于继续持有,
    会让一个早期仓位在截面塌掉后仍挂着,原样扛过后面的跳空。"""
    w = naive_weights(low_volatility(panel), panel, min_assets=99, rebalance=5)
    rebalanced = w.dropna(how="all")
    assert len(rebalanced) > 0 and (rebalanced.fillna(0) == 0).all().all()


def test_a_missing_factor_does_not_vote_neutral(panel):
    """某因子当日缺某标的时,其余因子按实际参与的权重重新归一 ——
    把缺失当 0 分等于替它投了一票中性票。"""
    a = low_volatility(panel)
    b = volume_shock(panel).copy()
    b.iloc[:, 0] = np.nan
    both = combine({"a": a, "b": b}, panel.mask)
    only_a = combine({"a": a}, panel.mask)
    col = both.columns[0]
    shared = both[col].notna() & only_a[col].notna()
    assert np.allclose(both[col][shared], only_a[col][shared])


def test_realised_vol_is_lagged(panel):
    """决策日当天的收盘收益在那一刻还没落定,直接用会把当日信息混进仓位。"""
    v = realised_vol(panel, 20)
    raw = np.log(panel.close / panel.close.shift(1)).rolling(20, min_periods=10).std()
    assert np.allclose(v.iloc[50].dropna(), raw.iloc[49].reindex(v.columns).dropna())


def test_frozen_sign_is_a_single_bit(panel):
    s = frozen_sign(low_volatility(panel), panel, h=5)
    assert s in (1.0, -1.0)
    assert frozen_sign(-low_volatility(panel), panel, h=5) == -s


def test_walk_forward_embargoes_unfinished_ic(panel):
    """用还没走完前瞻期的 IC 去挑因子,就是拿尚未发生的结果做决策。"""
    out = walk_forward({"a": low_volatility(panel), "b": volume_shock(panel)}, panel,
                       lookback=252, step=42, rebalance=5)
    assert out["config"]["embargo_days"] == 2 + 5
    assert out["selections"], "必须产生决策点"


def test_walk_forward_does_not_trade_the_warmup(panel):
    out = walk_forward({"a": low_volatility(panel)}, panel, lookback=252, step=42,
                       start="2022-04-01")
    traded = out["weights"].dropna(how="all")
    assert traded.index.min() >= pd.Timestamp("2022-04-01", tz=panel.index.tz)


def test_factor_ic_series_is_one_column_per_factor(panel):
    ic = factor_ic_series({"a": low_volatility(panel), "b": volume_shock(panel)}, panel, h=5)
    assert list(ic.columns) == ["a", "b"] and ic["a"].notna().sum() > 500