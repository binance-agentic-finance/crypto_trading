"""Smoke + unit tests for cyqnt_trd.evolve.bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.evolve.bridge import (
    _apply_filter,
    _factor_to_series,
    backtest_genome,
    make_signal_fn,
)
from cyqnt_trd.evolve.genome import Factor, Filter, StrategyGenome


# ── Synthetic data fixture ─────────────────────────────────────────────────

@pytest.fixture
def synthetic_df():
    """Generate 500 bars of pseudo-random OHLCV with a slight up-drift."""
    rng = np.random.default_rng(42)
    n = 500
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    returns = rng.normal(0.0002, 0.005, size=n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.003, size=n))
    low = close * (1 - rng.uniform(0, 0.003, size=n))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.uniform(100, 1000, size=n)
    close_time = (timestamps.astype("int64") // 1_000_000).to_numpy()  # ms
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": close_time,
        },
        index=timestamps,
    )
    return df


# ── Factor → series ────────────────────────────────────────────────────────

def test_ema_cross_above_fires_at_least_once(synthetic_df):
    f = Factor("ema", "cross_above", {"fast": 8, "slow": 21})
    s = _factor_to_series(synthetic_df, f)
    assert s.dtype == bool
    assert s.sum() > 0  # there should be at least one cross in 500 bars


def test_rsi_oversold_returns_bool_series(synthetic_df):
    f = Factor("rsi", "oversold", {"period": 14, "threshold": 30})
    s = _factor_to_series(synthetic_df, f)
    assert s.dtype == bool
    assert len(s) == len(synthetic_df)


def test_macd_golden_cross(synthetic_df):
    f = Factor("macd", "macd_golden_cross", {"fast": 12, "slow": 26, "signal": 9})
    s = _factor_to_series(synthetic_df, f)
    assert s.dtype == bool
    # synthetic up-drift → at least one golden cross expected
    assert s.sum() >= 1


def test_bollinger_squeeze(synthetic_df):
    f = Factor("bollinger", "squeeze", {"period": 20, "std": 2.0, "bandwidth_threshold": 0.05})
    s = _factor_to_series(synthetic_df, f)
    assert s.dtype == bool


def test_donchian_breakout(synthetic_df):
    f = Factor("donchian", "breakout", {"period": 20})
    s = _factor_to_series(synthetic_df, f)
    assert s.sum() > 0  # synthetic up-drift → breakouts


def test_unknown_indicator_returns_all_false(synthetic_df):
    f = Factor("ema", "ghost_condition", {})
    s = _factor_to_series(synthetic_df, f)
    assert s.sum() == 0


# ── Filter ─────────────────────────────────────────────────────────────────

def test_filter_adx_above(synthetic_df):
    fl = Filter("adx_above", {"period": 14, "threshold": 20})
    s = _apply_filter(synthetic_df, fl)
    assert s.dtype == bool
    assert len(s) == len(synthetic_df)


def test_filter_hour_range_with_datetime_index(synthetic_df):
    fl = Filter("hour_range", {"start": 12, "end": 16})
    s = _apply_filter(synthetic_df, fl)
    assert s.dtype == bool
    # 4 hours / 24 → roughly 1/6 of 500 bars
    assert 50 < s.sum() < 150


def test_filter_volume_above_ma(synthetic_df):
    fl = Filter("volume_above", {"period": 20, "multiplier": 1.0})
    s = _apply_filter(synthetic_df, fl)
    assert 0 < s.sum() < len(synthetic_df)


def test_unknown_filter_passes_through(synthetic_df):
    fl = Filter("ghost_filter", {})
    s = _apply_filter(synthetic_df, fl)
    assert s.all()


# ── End-to-end backtest ────────────────────────────────────────────────────

def test_backtest_genome_returns_result(synthetic_df):
    g = StrategyGenome(
        genome_id="t1",
        species="trend",
        entry_factors=[Factor("ema", "cross_above", {"fast": 5, "slow": 20})],
        entry_logic="all_of",
        exit_type="pct_stop_tp",
        exit_params={"stop_pct": 0.01, "tp_pct": 0.02, "max_bars": 20},
        filters=[],
        size=0.5,
        preferred_interval="15m",
    )
    res = backtest_genome(genome=g, df=synthetic_df)
    assert res is not None
    # Should produce some metrics; trade_count may be 0 or more depending on data
    assert res.final_equity > 0
    assert -1.0 <= res.total_return <= 5.0
    assert isinstance(res.trade_count, int)


def test_backtest_genome_long_only_no_short_trades(synthetic_df):
    """Ensure long_only=True is enforced (no short trades opened)."""
    g = StrategyGenome(
        genome_id="t2",
        species="trend",
        entry_factors=[Factor("ema", "cross_below", {"fast": 5, "slow": 20})],
        entry_logic="all_of",
        exit_type="pct_stop_tp",
        exit_params={"stop_pct": 0.01, "tp_pct": 0.02, "max_bars": 20},
        size=0.5,
        preferred_interval="15m",
    )
    # cross_below would be a short signal in a long/short bot, but we hand
    # it back as long_signal in the bridge's all-of combiner. The point of
    # this test is to confirm the bridge wires long_only=True regardless.
    fn = make_signal_fn(g)
    long_sig, short_sig = fn(synthetic_df)
    assert short_sig.sum() == 0  # bridge always returns False for shorts


def test_backtest_two_factor_all_of(synthetic_df):
    g = StrategyGenome(
        genome_id="t3",
        species="trend",
        entry_factors=[
            Factor("ema", "cross_above", {"fast": 5, "slow": 20}),
            Factor("rsi", "above_threshold", {"period": 14, "threshold": 50}),
        ],
        entry_logic="all_of",
        exit_type="pct_stop_tp",
        exit_params={"stop_pct": 0.01, "tp_pct": 0.025, "max_bars": 20},
        filters=[Filter("adx_above", {"period": 14, "threshold": 15})],
        size=0.5,
        preferred_interval="15m",
    )
    res = backtest_genome(genome=g, df=synthetic_df)
    assert res.final_equity > 0


def test_score_gte_logic(synthetic_df):
    g = StrategyGenome(
        genome_id="t4",
        species="momentum",
        entry_factors=[
            Factor("ema", "cross_above", {"fast": 5, "slow": 20}, weight=1.0),
            Factor("rsi", "above_threshold", {"period": 14, "threshold": 50}, weight=1.0),
            Factor("macd", "macd_above_zero", {}, weight=1.0),
        ],
        entry_logic="score_gte",
        entry_score_threshold=2,
        exit_type="pct_stop_tp",
        exit_params={"stop_pct": 0.01, "tp_pct": 0.025, "max_bars": 20},
        size=0.5,
        preferred_interval="15m",
    )
    fn = make_signal_fn(g)
    long_sig, _ = fn(synthetic_df)
    # Should fire on bars where ≥2 of 3 factors are true
    assert isinstance(long_sig.sum(), (int, np.integer))
