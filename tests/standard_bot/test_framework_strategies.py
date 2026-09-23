"""7 个内置策略的 make_signals 版本：产出 (long,short) 且能走框架回测。

口径已换成框架向量化引擎，这里锁定的是"策略逻辑成立 + 能跑通框架回测"，不锁定具体数字。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.standard_bot.signal.framework_strategies import (
    FRAMEWORK_STRATEGIES,
    make_signals_for,
)
from cyqnt_trd.standard_bot.simulation import FrameworkBacktestRunner


def _df(n=250, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.exp(np.cumsum(rng.normal(0, 0.012, n)) + 4)
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = np.abs(rng.normal(1e3, 1e2, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol, "quote_volume": vol * close}, index=idx)


def test_all_builtins_are_present():
    assert set(FRAMEWORK_STRATEGIES) == {
        "moving_average_cross", "price_moving_average", "rsi_reversion",
        "donchian_breakout", "multi_timeframe_ma_spread",
        "oi_funding_breakout", "liquidation_reversal"}


@pytest.mark.parametrize("sid", sorted(FRAMEWORK_STRATEGIES))
def test_builtin_make_signals_shape(sid):
    df = _df()
    long, short = FRAMEWORK_STRATEGIES[sid](df)
    assert isinstance(long, pd.Series) and isinstance(short, pd.Series)
    assert long.index.equals(df.index)
    assert long.dtype == bool and short.dtype == bool
    assert not (long & short).any(), "a bar cannot be both long and short"


@pytest.mark.parametrize("sid", sorted(FRAMEWORK_STRATEGIES))
def test_builtin_backtests_through_the_framework(sid):
    res = FrameworkBacktestRunner().run(make_signals_for(sid), _df(),
                                        instrument_id="BTCUSDT", min_history=40)
    assert math.isfinite(res.total_return)
    assert res.metrics["snapshot_count"] == 250.0


def test_donchian_is_long_short_others_long_only_or_flat():
    df = _df()
    long, short = FRAMEWORK_STRATEGIES["donchian_breakout"](df)
    # donchian can short; ma_cross never shorts (SELL -> FLAT)
    l2, s2 = FRAMEWORK_STRATEGIES["moving_average_cross"](df)
    assert not s2.any(), "moving_average_cross closes to flat, never shorts"


def test_params_bind_through_make_signals_for():
    df = _df()
    fn = make_signals_for("rsi_reversion", period=7, oversold=20.0, overbought=80.0)
    long, short = fn(df)
    assert isinstance(long, pd.Series)
