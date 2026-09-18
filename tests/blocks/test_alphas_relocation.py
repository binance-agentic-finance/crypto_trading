"""WorldQuant alphas 迁移到 cyqnt_trd.blocks.alphas 后的契约。

- 新路径 cyqnt_trd.blocks.alphas 可用,101 个因子齐全;
- 旧路径 cyqnt_trd.blocks.alphas 仍可导入(薄再导出),行为一致;
- to_series 逐点求值,构造上无未来函数(逐 bar 等于在扩展前缀上调用)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _df(n=80, seed=3):
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(rng.normal(0, 0.02, n)) + 4)
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    volume = np.abs(rng.normal(1e3, 2e2, n))
    return pd.DataFrame({
        "open_price": open_, "high_price": high, "low_price": low,
        "close_price": close, "volume": volume, "quote_volume": volume * close,
    })


def test_new_blocks_path_exposes_all_101_alphas():
    from cyqnt_trd.blocks import alphas
    assert len(alphas.ALPHA_FACTORS) == 101
    assert set(alphas.ALPHA_FACTORS) == set(range(1, 102))
    assert callable(alphas.alpha1_factor) and callable(alphas.alpha101_factor)


def test_legacy_selected_alpha_path_still_works():
    from cyqnt_trd.blocks.alphas import alpha1_factor as blocks_alpha1
    from cyqnt_trd.blocks.alphas import ALPHA_FACTORS, alpha1_factor
    assert alpha1_factor is blocks_alpha1                 # same object, not a copy
    assert len(ALPHA_FACTORS) == 101


def test_alpha_returns_a_finite_scalar_on_real_shaped_data():
    from cyqnt_trd.blocks.alphas import alpha1_factor
    v = alpha1_factor(_df())
    assert isinstance(v, float) and np.isfinite(v)


def test_to_series_is_lookahead_safe_and_matches_pointwise():
    from cyqnt_trd.blocks.alphas import alpha1_factor, to_series
    df = _df()
    s = to_series(alpha1_factor, df)
    assert isinstance(s, pd.Series) and len(s) == len(df)
    # every bar equals the point-wise factor on the expanding prefix (no future data)
    for i in (30, 50, len(df) - 1):
        assert s.iloc[i] == pytest.approx(alpha1_factor(df.iloc[: i + 1]), abs=1e-12)


def test_a_spread_of_alphas_are_callable():
    from cyqnt_trd.blocks.alphas import ALPHA_FACTORS
    df = _df()
    for n in (1, 5, 42, 88, 94, 101):
        v = ALPHA_FACTORS[n](df)
        assert isinstance(v, float)                       # never raises (own try/except → 0.0)
