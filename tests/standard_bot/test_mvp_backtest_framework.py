"""mvp_backtest --engine framework: built-in strategies run on cyqnt_trd.eval.

Verifies the entrypoint's framework path (`_run_framework_engine`) produces a
well-formed BacktestResult from Bars, for every built-in strategy.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from cyqnt_trd.standard_bot.core import BacktestResult, Bar
from cyqnt_trd.standard_bot.entrypoints.mvp_backtest import _BUILTIN_PARAMS, _run_framework_engine
from cyqnt_trd.standard_bot.signal.framework_strategies import FRAMEWORK_STRATEGIES


def _bars(n=200, seed=3):
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(rng.normal(0, 0.012, n)) + 4)
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    bars = []
    for i in range(n):
        ts = (i + 1) * 3_600_000
        bars.append(Bar(open=float(open_[i]), high=float(high[i]), low=float(low[i]),
                        close=float(close[i]), volume=1000.0, timestamp=ts,
                        instrument_id="BTCUSDT", timeframe="1h", confirmed=True,
                        quote_volume=float(close[i] * 1000.0),
                        extras={"open_time": ts - 3_600_000, "close_time": ts}))
    return bars


def _args(strategy):
    return SimpleNamespace(
        strategy=strategy, symbol="btcusdt", initial_capital=10000.0, commission_bps=6.5,
        fast_window=5, slow_window=20, entry_threshold=0.0, ma_period=20,
        rsi_period=14, oversold=30.0, overbought=70.0, donchian_window=20,
        breakout_buffer_bps=0.0, primary_ma_period=20, reference_ma_period=50,
        spread_threshold_bps=0.0, oi_threshold_bps=0.0, max_funding_rate_bps=100.0,
        long_liquidation_threshold_usd=100_000.0, short_liquidation_threshold_usd=100_000.0,
        liquidation_imbalance_ratio=0.60)


@pytest.mark.parametrize("sid", sorted(FRAMEWORK_STRATEGIES))
def test_framework_engine_runs_every_builtin(sid):
    res = _run_framework_engine(_args(sid), _bars(), registry=None,
                                request=SimpleNamespace(request_id="test"))
    assert isinstance(res, BacktestResult)
    assert math.isfinite(res.total_return)
    assert res.metrics["snapshot_count"] == 200.0
    assert res.extras["engine"] == "framework"


def test_builtin_param_map_covers_all_builtins():
    assert set(_BUILTIN_PARAMS) == set(FRAMEWORK_STRATEGIES)


def test_block_strategy_uses_plugin_signal_fn():
    # a fake registry returning an object with signal_fn
    fn = lambda df: (df["close"] > df["close"].rolling(10).mean(), None)  # noqa: E731
    registry = SimpleNamespace(get=lambda sid: SimpleNamespace(signal_fn=fn))
    res = _run_framework_engine(_args("my_block_strategy"), _bars(), registry=registry,
                                request=SimpleNamespace(request_id="test"))
    assert math.isfinite(res.total_return)


def test_unknown_strategy_without_signal_fn_is_a_clear_error():
    registry = SimpleNamespace(get=lambda sid: SimpleNamespace())   # no signal_fn
    with pytest.raises(ValueError, match="no make_signals"):
        _run_framework_engine(_args("mystery"), _bars(), registry=registry,
                              request=SimpleNamespace(request_id="test"))
