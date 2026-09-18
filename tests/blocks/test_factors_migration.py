"""JoinQuant 统计因子迁移到 cyqnt_trd.blocks.factors 后的契约。

- 新路径 cyqnt_trd.blocks.factors 暴露全部 4 类共 ~35 个因子；
- 旧路径 cyqnt_trd.trading_signal.factor 仍导出同名因子（同一对象）；
- 因子在真实形状数据上返回方向票。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXPECTED = {
    "risk": ["variance_factor", "skewness_factor", "kurtosis_factor", "sharpe_ratio_factor"],
    "technical": ["boll_factor", "emac_factor", "mac_factor", "macdc_factor", "mfi_factor"],
    "momentum": ["aroon_factor", "trix_factor", "roc_factor", "mass_factor", "cr_factor"],
    "sentiment": ["vroc_factor", "psy_factor", "wvad_factor", "arbr_factor", "money_flow_factor"],
}


def _pf(n=80, seed=4):
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(rng.normal(0, 0.02, n)) + 4)
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    vol = np.abs(rng.normal(1e3, 2e2, n))
    return pd.DataFrame({"open_price": open_, "high_price": high, "low_price": low,
                         "close_price": close, "volume": vol, "quote_volume": vol * close})


def test_blocks_factors_exposes_all_four_families():
    from cyqnt_trd.blocks import factors
    # risk+technical+momentum+sentiment (37) + 15 classic point-wise factors folded in
    # from the (now removed) trading_signal.factor.
    assert len(factors.__all__) == 4 + 5 + 14 + 14 + 15
    for fam, names in EXPECTED.items():
        for name in names:
            assert hasattr(factors, name), f"blocks.factors missing {name}"
    for classic in ("ma_factor", "rsi_factor", "stochastic_tsi_fast_factor"):
        assert hasattr(factors, classic), f"blocks.factors missing classic {classic}"


def test_legacy_trading_signal_factor_reexports_same_objects():
    from cyqnt_trd.blocks import factors
    from cyqnt_trd.blocks import factors as legacy
    for names in EXPECTED.values():
        for name in names:
            assert getattr(legacy, name) is getattr(factors, name)


def test_migrated_factors_return_direction_votes():
    from cyqnt_trd.blocks import factors
    pf = _pf()
    for names in EXPECTED.values():
        for name in names:
            v = getattr(factors, name)(pf)
            assert isinstance(v, float) and v in (-1.0, 0.0, 1.0), f"{name} -> {v}"
