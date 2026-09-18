from __future__ import annotations

import uuid

from cyqnt_trd.standard_bot.core import (
    BacktestRequest,
    Bar,
    BundleMeta,
    MarketBundle,
    SignalPipelineSpec,
)
from cyqnt_trd.standard_bot.data import AlignmentPolicy, HistoricalSnapshotAssembler
from cyqnt_trd.standard_bot.signal import (
    MultiTimeframeMaSpreadConfig,
    MultiTimeframeMaSpreadPlugin,
    SignalPluginRegistry,
    register_builtin_plugins,
)
from cyqnt_trd.standard_bot.simulation import SnapshotBacktestRunner


def _mtf_market_bundle() -> MarketBundle:
    bars_5m = []
    bars_1h = []
    close_values_5m = (
        [100.0] * 12
        + [120.0] * 12
        + [150.0] * 12
        + [80.0] * 12
    )
    hour_closes = [100.0, 120.0, 150.0, 80.0]

    for index, close in enumerate(close_values_5m):
        close_ts = (index + 1) * 5 * 60_000
        open_ts = close_ts - 5 * 60_000
        bars_5m.append(
            Bar(
                open=close,
                high=close,
                low=close,
                close=close,
                volume=100.0,
                timestamp=close_ts,
                instrument_id="BTCUSDT",
                timeframe="5m",
                confirmed=True,
                extras={"open_time": open_ts, "close_time": close_ts},
            )
        )

    for index, close in enumerate(hour_closes):
        close_ts = (index + 1) * 60 * 60_000
        open_ts = close_ts - 60 * 60_000
        bars_1h.append(
            Bar(
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1_200.0,
                timestamp=close_ts,
                instrument_id="BTCUSDT",
                timeframe="1h",
                confirmed=True,
                extras={"open_time": open_ts, "close_time": close_ts},
            )
        )

    return MarketBundle(
        bars={
            MarketBundle.key("BTCUSDT", "5m"): bars_5m,
            MarketBundle.key("BTCUSDT", "1h"): bars_1h,
        },
        meta=BundleMeta(data_source="fixture", fetched_at=bars_5m[-1].timestamp),
    )


def _snapshots():
    return HistoricalSnapshotAssembler(
        policy=AlignmentPolicy(policy_id="bar_close_v1", primary_timeframe="5m"),
        tail_bars=60,
    ).build(_mtf_market_bundle())


def test_multi_timeframe_ma_spread_batch_matches_step() -> None:
    plugin = MultiTimeframeMaSpreadPlugin()
    config = MultiTimeframeMaSpreadConfig(
        instrument_id="BTCUSDT",
        primary_timeframe="5m",
        reference_timeframe="1h",
        primary_ma_period=3,
        reference_ma_period=2,
        spread_threshold_bps=0.0,
    )
    state = plugin.initialize_state()
    previous_batch_count = 0

    for snapshot in _snapshots():
        batch = plugin.run(snapshot, config)
        batch_emitted = None
        if len(batch.signals) > previous_batch_count:
            batch_emitted = batch.signals[-1]
        previous_batch_count = len(batch.signals)

        step_result = plugin.step(snapshot, state, config)
        state = step_result.state
        step_emitted = step_result.signals[-1] if step_result.signals else None

        batch_side = None if batch_emitted is None else batch_emitted.side.value
        step_side = None if step_emitted is None else step_emitted.side.value
        assert batch_side == step_side


def test_multi_timeframe_ma_spread_plugin_emits_short_signal_on_regime_break() -> None:
    plugin = MultiTimeframeMaSpreadPlugin()
    config = MultiTimeframeMaSpreadConfig(
        instrument_id="BTCUSDT",
        primary_timeframe="5m",
        reference_timeframe="1h",
        primary_ma_period=3,
        reference_ma_period=2,
        spread_threshold_bps=0.0,
    )

    batch = plugin.run(_snapshots()[-1], config)

    assert batch.signals
    assert batch.signals[-1].side.value == "sell"


def test_event_engine_supports_multi_timeframe_long_short_strategy() -> None:
    # Numba engine was removed; the multi-timeframe strategy is now backtested on
    # the event-driven SnapshotBacktestRunner via the same plugin.
    snapshots = _snapshots()
    request = BacktestRequest(
        request_id=str(uuid.uuid4()),
        instruments=["BTCUSDT"],
        primary_timeframe="5m",
        start_ts=snapshots[0].meta.decision_as_of or snapshots[0].meta.assembled_at,
        end_ts=snapshots[-1].meta.decision_as_of or snapshots[-1].meta.assembled_at,
        signal_pipeline=SignalPipelineSpec(
            plugin_chain=[
                {
                    "plugin_id": "multi_timeframe_ma_spread",
                    "config": {
                        "instrument_id": "BTCUSDT",
                        "primary_timeframe": "5m",
                        "reference_timeframe": "1h",
                        "primary_ma_period": 3,
                        "reference_ma_period": 2,
                        "spread_threshold_bps": 0.0,
                    },
                }
            ]
        ),
        initial_capital=10_000.0,
        fee_model={"taker_fee_bps": 0.0, "funding_bps_per_bar": 0.0},
        slippage_model={"slippage_bps": 0.0, "impact_slippage_bps": 0.0, "max_bar_volume_fraction": 1.0},
    )

    registry = SignalPluginRegistry()
    register_builtin_plugins(registry)
    result = SnapshotBacktestRunner(signal_registry=registry).run(
        request=request,
        snapshots=snapshots,
    )

    total_signals = sum(len(batch.signals) for batch in result.signal_batches)
    assert total_signals >= 1
    trades = result.extras["trades"]
    assert any(
        t.get("action") == "entry" and t.get("position_side") == "short"
        for t in trades
    )
