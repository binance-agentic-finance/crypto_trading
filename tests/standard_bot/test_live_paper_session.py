"""
Tests verifying signal consistency between NumbaBacktestRunner and NumbaLivePaperSession.

The core invariant: given the same bars fed to both systems, the target positions
at each bar must be IDENTICAL, and trades must happen at the same bars with the
same direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from cyqnt_trd.standard_bot.signal.numba_kernels import (
    TARGET_KEEP,
    TARGET_LONG,
    TARGET_SHORT,
    moving_average_cross_target_updates,
    rsi_reversion_target_updates,
)
from cyqnt_trd.standard_bot.simulation import NumbaBacktestRunner, NumbaLivePaperSession
from cyqnt_trd.standard_bot.simulation.execution_kernels import simulate_target_positions_next_open
from cyqnt_trd.standard_bot.simulation.numba_runner import EncodedSeries

# ─── Test fixtures ────────────────────────────────────────────────────────


def _generate_synthetic_bars(n: int, seed: int = 42) -> list:
    """Generate synthetic bar data with realistic price movements."""
    rng = np.random.default_rng(seed)
    base_price = 50000.0
    bars = []
    price = base_price

    for i in range(n):
        # Random walk with mean reversion
        change = rng.normal(0, 0.02) * price
        price = max(price + change, 100.0)

        open_price = price
        high = price * (1.0 + abs(rng.normal(0, 0.005)))
        low = price * (1.0 - abs(rng.normal(0, 0.005)))
        close = price * (1.0 + rng.normal(0, 0.003))
        volume = abs(rng.normal(1000, 200))
        ts = 1_700_000_000_000 + i * 3_600_000  # 1h intervals

        bars.append({
            "timestamp": ts,
            "open_time": ts,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": close * volume,
            "oi_change_bps": 0.0,
            "funding_rate_bps": 0.0,
            "long_liq_notional_usd": 0.0,
            "short_liq_notional_usd": 0.0,
        })
        price = close

    return bars


def _bars_to_encoded_series(bars: list) -> EncodedSeries:
    """Convert bar dicts to EncodedSeries (same as numba_runner uses)."""
    return EncodedSeries(
        timestamps=np.array([b["timestamp"] for b in bars], dtype=np.int64),
        open_times=np.array([b["open_time"] for b in bars], dtype=np.int64),
        opens=np.array([b["open"] for b in bars], dtype=np.float64),
        highs=np.array([b["high"] for b in bars], dtype=np.float64),
        lows=np.array([b["low"] for b in bars], dtype=np.float64),
        closes=np.array([b["close"] for b in bars], dtype=np.float64),
        volumes=np.array([b["volume"] for b in bars], dtype=np.float64),
        quote_volumes=np.array([b["quote_volume"] for b in bars], dtype=np.float64),
        oi_change_bps=np.array([b["oi_change_bps"] for b in bars], dtype=np.float64),
        funding_rate_bps=np.array([b["funding_rate_bps"] for b in bars], dtype=np.float64),
        long_liq_notional_usd=np.array([b["long_liq_notional_usd"] for b in bars], dtype=np.float64),
        short_liq_notional_usd=np.array([b["short_liq_notional_usd"] for b in bars], dtype=np.float64),
    )


def _find_direction_change_bars(target_updates: np.ndarray) -> list:
    """
    Find bars where a direction change occurs in the target array.

    A direction change is: the first non-KEEP target that differs from the
    previous active direction.  Repeated same-direction targets are NOT counted.

    Target values:
    - TARGET_KEEP (2): no change
    - TARGET_LONG (1): go long
    - TARGET_SHORT (-1): go short
    - 0: go flat (close position)
    """
    changes = []
    current_direction = 0  # flat

    for i, target in enumerate(target_updates):
        if int(target) == int(TARGET_KEEP):
            continue
        direction = int(target)
        if direction != current_direction:
            changes.append((i, direction))
            current_direction = direction

    return changes


# ─── Signal consistency tests ─────────────────────────────────────────────


class TestSignalConsistency:
    """Verify that live paper session produces the same signals as backtest kernel."""

    def test_moving_average_cross_signals_match(self):
        """
        Feed the same bars to:
        1. moving_average_cross kernel directly (batch mode, like backtest)
        2. NumbaLivePaperSession tick-by-tick (streaming mode)

        Assert target positions are identical at every bar.
        """
        bars = _generate_synthetic_bars(200, seed=7)
        series = _bars_to_encoded_series(bars)
        config = {
            "instrument_id": "BTCUSDT",
            "timeframe": "1h",
            "fast_window": 5,
            "slow_window": 20,
            "entry_threshold": 0.0,
        }

        # --- Batch (backtest) ---
        batch_targets, batch_strengths = moving_average_cross_target_updates(
            series.closes,
            int(config["fast_window"]),
            int(config["slow_window"]),
            float(config["entry_threshold"]),
        )

        # --- Streaming (live paper session) ---
        session = NumbaLivePaperSession(
            strategy_id="moving_average_cross",
            symbol="BTCUSDT",
            config=config,
            initial_capital=10_000.0,
            fee_bps=0.0,  # zero fees for pure signal comparison
            slippage_bps=0.0,
            max_bar_volume_fraction=0.0,  # disable volume cap
        )

        streaming_targets = []
        for bar in bars:
            session.tick(bar)
            streaming_targets.append(session._target_history[-1])

        # --- Compare ---
        batch_arr = np.array(batch_targets, dtype=np.int8)
        stream_arr = np.array(streaming_targets, dtype=np.int8)

        assert len(batch_arr) == len(stream_arr), (
            "Length mismatch: batch=%d stream=%d" % (len(batch_arr), len(stream_arr))
        )
        mismatches = np.where(batch_arr != stream_arr)[0]
        assert len(mismatches) == 0, (
            "Signal mismatch at bars: %s\n"
            "batch[mismatch]: %s\n"
            "stream[mismatch]: %s"
            % (mismatches[:10], batch_arr[mismatches[:10]], stream_arr[mismatches[:10]])
        )

    def test_rsi_reversion_signals_match(self):
        """RSI reversion kernel: batch vs streaming signals must match."""
        bars = _generate_synthetic_bars(150, seed=99)
        series = _bars_to_encoded_series(bars)
        config = {
            "instrument_id": "BTCUSDT",
            "timeframe": "1h",
            "period": 14,
            "oversold": 30.0,
            "overbought": 70.0,
        }

        # Batch
        batch_targets, _ = rsi_reversion_target_updates(
            series.closes,
            int(config["period"]),
            float(config["oversold"]),
            float(config["overbought"]),
        )

        # Streaming
        session = NumbaLivePaperSession(
            strategy_id="rsi_reversion",
            symbol="BTCUSDT",
            config=config,
            initial_capital=10_000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            max_bar_volume_fraction=0.0,
        )

        streaming_targets = []
        for bar in bars:
            session.tick(bar)
            streaming_targets.append(session._target_history[-1])

        batch_arr = np.array(batch_targets, dtype=np.int8)
        stream_arr = np.array(streaming_targets, dtype=np.int8)
        mismatches = np.where(batch_arr != stream_arr)[0]
        assert len(mismatches) == 0, "RSI reversion signal mismatch at bars: %s" % mismatches[:10]

    @pytest.mark.skipif(
        not Path(__file__).resolve().parents[2].joinpath(
            "mvp_strategy_lab", "__init__.py"
        ).exists(),
        reason="mvp_strategy_lab not a proper package (no __init__.py); run as: "
               "PYTHONPATH=. pytest ... for external strategy tests",
    )
    def test_custom_kernel_ema_rsi_cross_signals_match(self):
        """
        External ema_rsi_cross kernel: batch vs streaming signals must match.

        This imports the external strategy module and verifies it works
        identically in both batch (backtest) and streaming (daemon) modes.

        NOTE: Requires mvp_strategy_lab to be importable. Since it is an
        implicit namespace package without __init__.py, this test may be
        skipped in environments where the module path is not configured.
        """
        # Add project root to path so the external strategy module can be imported
        project_root = str(Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import importlib
        importlib.import_module("mvp_strategy_lab.external_strategies.ema_rsi_cross")

        bars = _generate_synthetic_bars(200, seed=13)
        series = _bars_to_encoded_series(bars)
        config = {
            "instrument_id": "BTCUSDT",
            "timeframe": "1h",
            "fast_window": 2,
            "slow_window": 5,
            "rsi_period": 3,
            "rsi_mid": 50.0,
            "session_start_h": 0,
            "session_end_h": 0,
        }

        # Batch (using runner's custom kernel path)
        runner = NumbaBacktestRunner()
        batch_targets, batch_strengths = runner._build_custom_signal_targets(
            strategy_id="ema_rsi_cross",
            raw_config=config,
            primary_series=series,
        )

        # Streaming
        session = NumbaLivePaperSession(
            strategy_id="ema_rsi_cross",
            symbol="BTCUSDT",
            config=config,
            initial_capital=10_000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            max_bar_volume_fraction=0.0,
        )

        streaming_targets = []
        for bar in bars:
            session.tick(bar)
            streaming_targets.append(session._target_history[-1])

        batch_arr = np.array(batch_targets, dtype=np.int8)
        stream_arr = np.array(streaming_targets, dtype=np.int8)
        mismatches = np.where(batch_arr != stream_arr)[0]
        assert len(mismatches) == 0, (
            "ema_rsi_cross signal mismatch at bars: %s" % mismatches[:10]
        )


class TestExecutionConsistency:
    """
    Verify that trade execution matches between backtest and live session.

    Note: The backtest execution kernel rebalances on EVERY non-KEEP bar (even
    repeated same-direction signals), while the live daemon only trades on
    direction changes.  This is intentional: in live trading, constant rebalancing
    on unchanged signals would generate excessive fees.

    Therefore, this test verifies that DIRECTION CHANGE trades are consistent
    between backtest and daemon.
    """

    def test_direction_change_trades_match(self):
        """
        For moving_average_cross, verify that:
        1. Direction changes detected by the daemon match the backtest
        2. The target position at each direction change is the same
        """
        bars = _generate_synthetic_bars(200, seed=7)
        series = _bars_to_encoded_series(bars)
        config = {
            "instrument_id": "BTCUSDT",
            "timeframe": "1h",
            "fast_window": 5,
            "slow_window": 20,
            "entry_threshold": 0.0,
        }

        # --- Backtest: find direction change bars from target_updates ---
        batch_targets, _ = moving_average_cross_target_updates(
            series.closes,
            int(config["fast_window"]),
            int(config["slow_window"]),
            float(config["entry_threshold"]),
        )
        backtest_direction_changes = _find_direction_change_bars(batch_targets)

        # --- Live session: collect fills and their triggering target ---
        session = NumbaLivePaperSession(
            strategy_id="moving_average_cross",
            symbol="BTCUSDT",
            config=config,
            initial_capital=10_000.0,
            fee_bps=10.0,
            slippage_bps=2.0,
            max_bar_volume_fraction=0.10,
        )

        # Track which bars generated pending orders and what target they had
        live_direction_changes = []
        prev_pending = None
        for i, bar in enumerate(bars):
            fill = session.tick(bar)
            # Check if a new pending order was just created this tick
            if session._pending_order is not None and session._pending_order is not prev_pending:
                live_direction_changes.append(
                    (i, session._pending_order.target_position)
                )
            prev_pending = session._pending_order

        # --- Compare direction changes ---
        assert len(backtest_direction_changes) == len(live_direction_changes), (
            "Direction change count mismatch.\n"
            "Backtest: %d changes: %s\n"
            "Live: %d changes: %s"
            % (
                len(backtest_direction_changes),
                backtest_direction_changes[:10],
                len(live_direction_changes),
                live_direction_changes[:10],
            )
        )

        for (bt_bar, bt_dir), (live_bar, live_dir) in zip(
            backtest_direction_changes, live_direction_changes
        ):
            assert bt_bar == live_bar, (
                "Direction change at different bars: backtest=%d live=%d" % (bt_bar, live_bar)
            )
            assert bt_dir == live_dir, (
                "Target mismatch at bar %d: backtest=%d live=%d" % (bt_bar, bt_dir, live_dir)
            )

    def test_live_session_does_not_rebalance_on_repeated_signals(self):
        """
        The daemon should NOT trade when the kernel emits the same direction
        as current position. Only direction changes trigger trades.
        """
        bars = _generate_synthetic_bars(200, seed=7)
        series = _bars_to_encoded_series(bars)
        config = {
            "instrument_id": "BTCUSDT",
            "timeframe": "1h",
            "fast_window": 5,
            "slow_window": 20,
            "entry_threshold": 0.0,
        }

        # Backtest trades (includes rebalances)
        batch_targets, _ = moving_average_cross_target_updates(
            series.closes, 5, 20, 0.0,
        )
        (_, _, _, trade_actions, _, _, _, _) = simulate_target_positions_next_open(
            series.opens, series.closes, series.volumes, series.quote_volumes,
            batch_targets, int(TARGET_KEEP), 10_000.0, 10.0, 2.0, 0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0,
        )
        backtest_total_trades = int(np.count_nonzero(trade_actions))

        # Live session trades (direction changes only)
        session = NumbaLivePaperSession(
            strategy_id="moving_average_cross",
            symbol="BTCUSDT",
            config=config,
            initial_capital=10_000.0,
            fee_bps=10.0,
            slippage_bps=2.0,
            max_bar_volume_fraction=0.10,
        )
        live_trade_count = 0
        for bar in bars:
            fill = session.tick(bar)
            if fill is not None:
                live_trade_count += 1

        # Live trades should be <= backtest trades (no rebalancing)
        assert live_trade_count <= backtest_total_trades, (
            "Live session has MORE trades (%d) than backtest (%d)"
            % (live_trade_count, backtest_total_trades)
        )
        # Live trades == number of direction changes
        direction_changes = _find_direction_change_bars(batch_targets)
        assert live_trade_count == len(direction_changes), (
            "Live trade count (%d) != direction changes (%d)"
            % (live_trade_count, len(direction_changes))
        )


class TestSessionLifecycle:
    """Test session state management and snapshot behavior."""

    def test_state_snapshot_is_readonly(self):
        """Calling state_snapshot() multiple times never changes state."""
        bars = _generate_synthetic_bars(50, seed=1)
        session = NumbaLivePaperSession(
            strategy_id="moving_average_cross",
            symbol="BTCUSDT",
            config={"fast_window": 5, "slow_window": 20, "entry_threshold": 0.0},
            initial_capital=10_000.0,
        )
        for bar in bars:
            session.tick(bar)

        snap1 = session.state_snapshot()
        snap2 = session.state_snapshot()
        snap3 = session.state_snapshot()

        assert snap1["current_equity"] == snap2["current_equity"] == snap3["current_equity"]
        assert snap1["tick_count"] == snap2["tick_count"] == snap3["tick_count"]
        assert snap1["trade_count"] == snap2["trade_count"] == snap3["trade_count"]

    def test_pending_order_executes_at_next_bar_open(self):
        """When a signal fires at bar[i], execution happens at bar[i+1] open."""
        bars = _generate_synthetic_bars(100, seed=42)
        session = NumbaLivePaperSession(
            strategy_id="moving_average_cross",
            symbol="BTCUSDT",
            config={"fast_window": 3, "slow_window": 10, "entry_threshold": 0.0},
            initial_capital=10_000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            max_bar_volume_fraction=0.0,
        )

        for i, bar in enumerate(bars):
            had_pending_before = session.has_pending_order()
            fill = session.tick(bar)

            if fill is not None:
                # The fill should have come from a pending order that was set
                # at a previous bar
                assert had_pending_before, (
                    "Fill at bar %d but no pending order existed before tick" % i
                )

    def test_warm_up_never_trades_or_carries_pending_order(self):
        """Historical warm-up bars should only build indicator context."""
        bars = _generate_synthetic_bars(120, seed=7)
        session = NumbaLivePaperSession(
            strategy_id="moving_average_cross",
            symbol="BTCUSDT",
            config={"fast_window": 3, "slow_window": 10, "entry_threshold": 0.0},
            initial_capital=10_000.0,
            fee_bps=10.0,
            slippage_bps=2.0,
            max_bar_volume_fraction=0.0,
        )

        for bar in bars:
            session.warm_up(bar)

        snapshot = session.state_snapshot()
        assert session.cash == pytest.approx(10_000.0)
        assert session.position is None
        assert session.position_qty == pytest.approx(0.0)
        assert session.trade_log == []
        assert snapshot["trade_count"] == 0
        assert snapshot["has_pending_order"] is False
        assert snapshot["bar_count"] == len(bars)

    def test_multi_timeframe_rejected(self):
        """multi_timeframe_ma_spread should raise ValueError."""
        with pytest.raises(ValueError, match="not supported in live paper mode"):
            NumbaLivePaperSession(
                strategy_id="multi_timeframe_ma_spread",
                symbol="BTCUSDT",
                config={},
                initial_capital=10_000.0,
            )

    def test_unregistered_strategy_rejected(self):
        """Unregistered strategy should raise ValueError."""
        with pytest.raises(ValueError, match="not registered"):
            NumbaLivePaperSession(
                strategy_id="nonexistent_strategy_xyz",
                symbol="BTCUSDT",
                config={},
                initial_capital=10_000.0,
            )

    def test_equity_tracks_position_value(self):
        """Equity should reflect cash + position mark-to-market."""
        bars = _generate_synthetic_bars(100, seed=55)
        session = NumbaLivePaperSession(
            strategy_id="moving_average_cross",
            symbol="BTCUSDT",
            config={"fast_window": 3, "slow_window": 10, "entry_threshold": 0.0},
            initial_capital=10_000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            max_bar_volume_fraction=0.0,
        )

        for bar in bars:
            session.tick(bar)

        # Verify equity = cash + position * last_price
        expected_equity = session.cash + session.position_qty * bars[-1]["close"]
        assert abs(session.equity - expected_equity) < 1e-6
