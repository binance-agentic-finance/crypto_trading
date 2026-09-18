"""blocks.factor_votes：向量化方向票 + trading_signal 经典 TA 因子委托后的契约。

- 票 ∈ {-1, 0, 1}（预热为 NaN）；
- 构造上无未来函数：截断未来 bar 不改变过去的票；
- trading_signal 经典 TA 因子委托后 == 对应 vote 序列的最后一根。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks import factor_votes as fv


def _df(n=120, seed=5):
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(rng.normal(0, 0.02, n)) + 4)
    high = np.maximum(close, close * (1 + np.abs(rng.normal(0, 0.01, n))))
    low = np.minimum(close, close * (1 - np.abs(rng.normal(0, 0.01, n))))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    vol = np.abs(rng.normal(1e3, 2e2, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol, "quote_volume": vol * close}, index=idx)


ALL_VOTES = [
    ("ma_vote", fv.ma_vote), ("ma_cross_vote", fv.ma_cross_vote),
    ("ema_vote", fv.ema_vote), ("ema_cross_vote", fv.ema_cross_vote),
    ("momentum_vote", fv.momentum_vote), ("rsi_vote", fv.rsi_vote),
    ("stochastic_k_vote", fv.stochastic_k_vote), ("cci_vote", fv.cci_vote),
    ("williams_r_vote", fv.williams_r_vote), ("ultimate_oscillator_vote", fv.ultimate_oscillator_vote),
    ("adx_vote", fv.adx_vote), ("awesome_oscillator_vote", fv.awesome_oscillator_vote),
    ("macd_level_vote", fv.macd_level_vote), ("bull_bear_power_vote", fv.bull_bear_power_vote),
]


@pytest.mark.parametrize("name,fn", ALL_VOTES)
def test_vote_is_a_series_in_the_three_valued_set(name, fn):
    df = _df()
    s = fn(df)
    assert isinstance(s, pd.Series) and len(s) == len(df)
    finite = s.dropna().unique()
    assert set(finite) <= {-1.0, 0.0, 1.0}, f"{name} produced {finite}"
    assert s.notna().any(), f"{name} is all-NaN"


@pytest.mark.parametrize("name,fn", ALL_VOTES)
def test_vote_is_lookahead_safe(name, fn):
    df = _df()
    cut = 90
    full = fn(df)
    truncated = fn(df.iloc[:cut])
    a = full.iloc[:cut].to_numpy()
    b = truncated.to_numpy()
    same = (np.isnan(a) & np.isnan(b)) | (a == b)
    assert same.all(), f"{name}: truncating future bars changed past votes"


def test_delegating_classic_factor_matches_its_vote_series():
    # trading_signal 因子委托后应等于 blocks vote 序列的最后一根
    from cyqnt_trd.blocks.factors.rsi_factor import rsi_factor
    df = _df()
    pf = df.rename(columns={"open": "open_price", "high": "high_price",
                            "low": "low_price", "close": "close_price"})
    got = rsi_factor(pf, period=14)
    want = fv.rsi_vote(df, 14).iloc[-1]
    assert got == pytest.approx(0.0 if pd.isna(want) else float(want))
    assert got in (-1.0, 0.0, 1.0)


def test_all_classic_factors_still_import_and_return_floats():
    from cyqnt_trd.blocks import factors as factor_mod
    df = _df()
    pf = df.rename(columns={"open": "open_price", "high": "high_price",
                            "low": "low_price", "close": "close_price"})
    for name in ("ma_factor", "ma_cross_factor", "rsi_factor", "cci_factor",
                 "adx_factor", "ao_factor", "momentum_factor", "macd_level_factor",
                 "williams_r_factor", "bbp_factor", "uo_factor", "ema_factor",
                 "ema_cross_factor", "stochastic_k_factor"):
        v = getattr(factor_mod, name)(pf)
        assert v in (-1.0, 0.0, 1.0), f"{name} -> {v}"
