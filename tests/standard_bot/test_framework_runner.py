"""FrameworkBacktestRunner: 单标的 make_signals 策略 → 框架回测 → BacktestResult 契约。

这是把 standard_bot 回测收敛到 cyqnt_trd.eval 的运行器。此处只锁定**契约**（返回
BacktestResult、字段齐全、数值有限），不锁定具体数字（口径已换成框架的向量化引擎）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cyqnt_trd.standard_bot.core import BacktestResult, EquityPoint
from cyqnt_trd.standard_bot.simulation import FrameworkBacktestRunner


def _df(n=150, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 4)
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = np.abs(rng.normal(1e3, 1e2, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol, "quote_volume": vol * close}, index=idx)


def _ma_cross(d):
    fast, slow = d["close"].rolling(5).mean(), d["close"].rolling(20).mean()
    return fast > slow, fast < slow


def test_run_returns_a_wellformed_backtest_result():
    res = FrameworkBacktestRunner().run(_ma_cross, _df(), instrument_id="BTCUSDT",
                                        initial_capital=10000.0, min_history=20)
    assert isinstance(res, BacktestResult)
    assert isinstance(res.equity_curve, list) and len(res.equity_curve) == 150
    assert all(isinstance(p, EquityPoint) for p in res.equity_curve)
    assert math.isfinite(res.total_return)


def test_metrics_carry_the_keys_the_entrypoints_read():
    res = FrameworkBacktestRunner().run(_ma_cross, _df(), instrument_id="BTCUSDT",
                                        initial_capital=10000.0, min_history=20)
    for key in ("snapshot_count", "trade_count", "final_equity", "total_return"):
        assert key in res.metrics and math.isfinite(res.metrics[key])
    assert res.metrics["snapshot_count"] == 150.0
    # final_equity is total_return applied to initial capital
    assert res.metrics["final_equity"] == (1.0 + res.total_return) * 10000.0
    assert res.extras["engine"] == "framework"


def test_trades_are_discrete_position_changes():
    res = FrameworkBacktestRunner().run(_ma_cross, _df(), instrument_id="BTCUSDT", min_history=20)
    trades = res.extras["trades"]
    assert isinstance(trades, list)
    assert res.metrics["trade_count"] == float(len(trades))
    if trades:
        assert {"timestamp", "instrument_id", "side", "action"} <= set(trades[0])


def test_long_only_strategy_runs():
    long_only = lambda d: (d["close"] > d["close"].rolling(10).mean(), None)  # noqa: E731
    res = FrameworkBacktestRunner().run(long_only, _df(), instrument_id="BTCUSDT", min_history=20)
    assert math.isfinite(res.total_return)
