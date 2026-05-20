"""
Tests for PythonLivePaperSession — paper trading powered by ``cyqnt_trd.blocks``
strategies (mirror of the existing test_live_paper_session.py for the Numba
session).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks import (
    indicators as ind,
    conditions as cond,
    strategy as strat,
)
from cyqnt_trd.blocks.strategy import (
    _KNOWN_BLOCK_STRATEGY_IDS,
    _PENDING_REGISTRATIONS,
)
from cyqnt_trd.standard_bot.simulation import (
    PaperFill,
    PaperPosition,
    PendingOrder,
    PythonLivePaperSession,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_pending_registrations():
    """Ensure each test starts with a clean blocks-strategy registry."""
    _PENDING_REGISTRATIONS.clear()
    _KNOWN_BLOCK_STRATEGY_IDS.clear()
    yield
    _PENDING_REGISTRATIONS.clear()
    _KNOWN_BLOCK_STRATEGY_IDS.clear()


@pytest.fixture
def synthetic_bars():
    """Generate 100 OHLCV bars with a clear uptrend → downtrend → rebound."""
    np.random.seed(42)
    n_bars = 100
    prices: list[float] = []
    # Uptrend
    for i in range(40):
        prices.append(100 + i * 0.5 + np.random.normal(0, 0.3))
    # Downtrend
    for i in range(40):
        prices.append(120 - i * 0.5 + np.random.normal(0, 0.3))
    # Rebound
    for i in range(20):
        prices.append(100 + i * 0.6 + np.random.normal(0, 0.3))

    ts_start = 1700000000000
    interval_ms = 15 * 60 * 1000
    bars = []
    for i, price in enumerate(prices):
        bars.append({
            "timestamp": ts_start + i * interval_ms,
            "open_time": ts_start + i * interval_ms - interval_ms + 1,
            "open": price - 0.1,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000.0,
            "quote_volume": price * 1000.0,
        })
    return bars


def _register_ma_cross():
    def make_signals(df):
        ma_fast = ind.sma(df["close"], 5)
        ma_slow = ind.sma(df["close"], 20)
        return cond.ma_cross_above(ma_fast, ma_slow), cond.ma_cross_below(ma_fast, ma_slow)

    strat.register("test_ma_cross", make_signals)


# ── Tests ──────────────────────────────────────────────────────────────────


def test_session_init_with_registered_strategy():
    """PythonLivePaperSession should resolve a registered block strategy."""
    _register_ma_cross()
    session = PythonLivePaperSession(
        strategy_id="test_ma_cross",
        symbol="BTCUSDT",
        config={"timeframe": "15m"},
        initial_capital=10_000.0,
    )
    assert session.strategy_id == "test_ma_cross"
    assert session.symbol == "BTCUSDT"
    assert session.cash == 10_000.0
    assert session.position is None
    assert session.position_qty == 0.0


def test_session_rejects_unregistered_strategy():
    """Constructor must raise if strategy_id was never registered."""
    with pytest.raises(ValueError, match="not registered"):
        PythonLivePaperSession(
            strategy_id="nonexistent_strategy",
            symbol="BTCUSDT",
            config={"timeframe": "15m"},
        )


def test_warm_up_does_not_create_pending_order(synthetic_bars):
    """warm_up() loads bars into history but never queues an order."""
    _register_ma_cross()
    session = PythonLivePaperSession(
        strategy_id="test_ma_cross",
        symbol="BTCUSDT",
        config={"timeframe": "15m"},
    )
    for bar in synthetic_bars[:30]:
        session.warm_up(bar)
    assert session._tick_count == 30
    assert len(session._timestamps) == 30
    assert session.has_pending_order() is False
    assert session.position is None


def test_tick_produces_fills_on_trend_reversal(synthetic_bars):
    """Feeding a full trend-reversal price series should produce fills."""
    _register_ma_cross()
    session = PythonLivePaperSession(
        strategy_id="test_ma_cross",
        symbol="BTCUSDT",
        config={"timeframe": "15m"},
        initial_capital=10_000.0,
        fee_bps=4.0,
        slippage_bps=2.0,
    )
    # Warm up
    for bar in synthetic_bars[:30]:
        session.warm_up(bar)
    # Tick remaining
    fills = []
    for bar in synthetic_bars[30:]:
        fill = session.tick(bar)
        if fill is not None:
            fills.append(fill)

    assert session._tick_count == len(synthetic_bars)
    assert len(fills) >= 1, "expected at least one fill on the trend reversal"
    # Verify all fills have valid action labels
    valid_actions = {
        "open_long", "open_short", "close_long", "close_short",
        "flip_to_long", "flip_to_short", "rebalance",
    }
    for fill in fills:
        assert fill.action in valid_actions
        assert fill.price > 0
        assert fill.quantity > 0


def test_state_snapshot_shape(synthetic_bars):
    """state_snapshot returns dict compatible with the watcher's state.json."""
    _register_ma_cross()
    session = PythonLivePaperSession(
        strategy_id="test_ma_cross",
        symbol="BTCUSDT",
        config={"timeframe": "15m"},
    )
    for bar in synthetic_bars[:50]:
        session.tick(bar)

    snap = session.state_snapshot()
    expected_keys = {
        "session_id", "strategy", "symbol", "market_type", "params",
        "initial_capital", "session_start_equity", "current_equity",
        "pnl_usd", "pnl", "position", "latest_signal", "has_pending_order",
        "tick_count", "bar_count", "trade_count", "trade_log",
        "last_bar_timestamp", "current_price", "last_update_ts",
    }
    assert expected_keys.issubset(snap.keys())
    assert snap["strategy"] == "test_ma_cross"
    assert snap["bar_count"] == 50


def test_checkpoint_round_trip(synthetic_bars):
    """checkpoint_state → from_checkpoint should preserve all state."""
    _register_ma_cross()
    session = PythonLivePaperSession(
        strategy_id="test_ma_cross",
        symbol="BTCUSDT",
        config={"timeframe": "15m"},
        initial_capital=10_000.0,
    )
    for bar in synthetic_bars:
        session.tick(bar)

    ckpt = session.checkpoint_state()
    assert ckpt["format_version"] == 1
    assert ckpt["engine"] == "python"

    # Re-register since `from_checkpoint` requires the plugin
    # to be in the pending list at construction time.
    _register_ma_cross()
    restored = PythonLivePaperSession.from_checkpoint(ckpt)

    assert restored._session_id == session._session_id
    assert restored.cash == pytest.approx(session.cash)
    assert restored.position_qty == pytest.approx(session.position_qty)
    assert len(restored._timestamps) == len(session._timestamps)
    assert len(restored.trade_log) == len(session.trade_log)


def test_pnl_properties(synthetic_bars):
    """equity / pnl_usd / pnl_pct properties should be consistent."""
    _register_ma_cross()
    session = PythonLivePaperSession(
        strategy_id="test_ma_cross",
        symbol="BTCUSDT",
        config={"timeframe": "15m"},
        initial_capital=10_000.0,
    )
    for bar in synthetic_bars:
        session.tick(bar)

    assert session.equity == session.cash + session.position_qty * session._closes[-1] * session.contract_multiplier
    assert session.pnl_usd == session.equity - session.initial_capital
    assert session.pnl_pct == session.pnl_usd / session.initial_capital


def test_tick_with_short_only_strategy(synthetic_bars):
    """A short-only strategy returns (long=False, short=signal). The session
    must handle ``short_signal=None`` semantics by reading the boolean correctly."""
    def short_only_signals(df):
        ma_fast = ind.sma(df["close"], 5)
        ma_slow = ind.sma(df["close"], 20)
        long_s = pd.Series(False, index=df.index)
        short_s = cond.ma_cross_below(ma_fast, ma_slow)
        return long_s, short_s

    strat.register("short_only", short_only_signals)
    session = PythonLivePaperSession(
        strategy_id="short_only",
        symbol="ETHUSDT",
        config={"timeframe": "15m"},
        initial_capital=10_000.0,
    )
    for bar in synthetic_bars:
        session.tick(bar)
    # Should not crash. Final position is either short or flat (but not long).
    direction = session._position_direction()
    assert direction in (0, -1), f"long-only result on a short-only strategy: {direction}"


def test_long_only_strategy_returns_long_series_and_none(synthetic_bars):
    """A long-only strategy returns (long_signal, None). The session must
    handle short_s being None correctly."""
    def long_only_signals(df):
        ma_fast = ind.sma(df["close"], 5)
        ma_slow = ind.sma(df["close"], 20)
        return cond.ma_cross_above(ma_fast, ma_slow), None

    strat.register("long_only", long_only_signals)
    session = PythonLivePaperSession(
        strategy_id="long_only",
        symbol="ETHUSDT",
        config={"timeframe": "15m"},
        initial_capital=10_000.0,
    )
    for bar in synthetic_bars:
        session.tick(bar)
    direction = session._position_direction()
    assert direction in (0, 1), f"short result on a long-only strategy: {direction}"


def test_empty_history_returns_no_signal():
    """When no bars have been added yet, _compute_latest_target should return
    TARGET_KEEP without crashing."""
    _register_ma_cross()
    session = PythonLivePaperSession(
        strategy_id="test_ma_cross",
        symbol="BTCUSDT",
        config={"timeframe": "15m"},
    )
    # No bars yet → safe defaults
    target, strength = session._compute_latest_target()
    from cyqnt_trd.standard_bot.signal.numba_kernels import TARGET_KEEP
    assert int(target) == int(TARGET_KEEP)
    assert strength == 0.0
