"""make_signals 策略 → 新框架回测的桥接契约。

锁定"任意单标的 make_signals 策略都能走新框架回测"这条统一路径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cyqnt_trd.eval.adapters import (
    backtest_signals,
    backtest_weights_from_df,
    make_panel_from_df,
    signals_to_weights,
)


def _df(n=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 4)
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = np.abs(rng.normal(1e3, 2e2, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol, "quote_volume": vol * close}, index=idx)


def _ma_cross(d):
    fast = d["close"].rolling(5).mean()
    slow = d["close"].rolling(20).mean()
    return fast > slow, fast < slow


def test_signals_to_weights_is_three_valued_long_wins_conflict():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    w = signals_to_weights(pd.Series([True, False, True, False], index=idx),
                           pd.Series([False, True, True, False], index=idx), idx)
    assert list(w) == [1.0, -1.0, 1.0, 0.0]      # 冲突取多头


def test_make_panel_from_df_builds_a_one_symbol_regular_panel():
    panel = make_panel_from_df(_df(), symbol="BTC", min_history=20)
    panel.validate()
    assert panel.symbols == ["BTC"] and panel.cell_scheme == "regular"


def test_a_make_signals_strategy_backtests_through_portfolio_simulate():
    book = backtest_signals(_ma_cross, _df(), symbol="BTC", min_history=20)
    assert hasattr(book, "equity") and len(book.equity) == 200
    assert np.isfinite(float(book.equity.iloc[-1]))


def test_a_make_signals_strategy_backtests_through_backtest_weights():
    res = backtest_weights_from_df(_ma_cross, _df(), symbol="BTC", h=1, min_history=20)
    assert set(res.metrics.split) <= {"dev", "val", "oot"}
    assert hasattr(res, "weights") and res.weights.shape[1] == 1


def test_long_only_strategy_is_accepted():
    long_only = lambda d: (d["close"] > d["close"].rolling(10).mean(), None)  # noqa: E731
    book = backtest_signals(long_only, _df(), symbol="BTC", min_history=20)
    assert np.isfinite(float(book.equity.iloc[-1]))
