"""
Signal plugins backed by shared pure-NumPy kernels.

The public plugin interfaces remain Python-friendly, while all numerical
decision logic is routed through the same array kernels used by the fast
backtest path.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core import (
    DataSnapshot,
    SignalBatch,
    SignalContext,
    SignalEnvelope,
    SignalKind,
    SignalProvenance,
    TradeSide,
)
from .encoders import encode_close_series, series_for
from .interfaces import SignalState, StepSignalResult
from .kernels import (
    SIGNAL_BUY,
    SIGNAL_NONE,
    adx_trend_strength_signal_rows,
    atr_breakout_signal_rows,
    bollinger_mean_reversion_signal_rows,
    donchian_breakout_signal_rows,
    liquidation_reversal_signal_rows,
    macd_trend_follow_signal_rows,
    moving_average_cross_signal_rows,
    multi_timeframe_ma_spread_signal_rows,
    oi_funding_breakout_signal_rows,
    price_moving_average_signal_rows,
    rsi_reversion_signal_rows,
)

SIGNAL_NAMESPACE = uuid.UUID("6d011acb-2b89-5dc5-bd38-fa9f903e6495")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _signal_side(direction: int) -> TradeSide:
    return TradeSide.BUY if int(direction) == int(SIGNAL_BUY) else TradeSide.SELL


def _blocked_instruments(context: Optional[SignalContext]) -> List[str]:
    return [] if context is None else context.blacklist


def _config_hash(*parts: object) -> str:
    return "|".join(str(part) for part in parts)


def _build_batch(plugin_id: str, snapshot: DataSnapshot, signals: Sequence[SignalEnvelope]) -> SignalBatch:
    batch_id = str(
        uuid.uuid5(
            SIGNAL_NAMESPACE,
            "%s|%s|%s" % (plugin_id, snapshot.meta.snapshot_id, len(signals)),
        )
    )
    return SignalBatch(signals=list(signals), batch_id=batch_id, created_at=_now_ms())


def _build_signal(
    *,
    snapshot: DataSnapshot,
    plugin_id: str,
    plugin_version: str,
    instrument_id: str,
    timeframe: str,
    time_horizon: str,
    timestamp: int,
    direction: int,
    strength: float,
    payload: Dict[str, object],
    config_hash: str,
) -> SignalEnvelope:
    side = _signal_side(direction)
    signal_id = str(
        uuid.uuid5(
            SIGNAL_NAMESPACE,
            "%s|%s|%s|%s|%s"
            % (
                snapshot.meta.snapshot_id,
                instrument_id,
                timeframe,
                timestamp,
                side.value,
            ),
        )
    )
    return SignalEnvelope(
        version="numba/v1",
        signal_id=signal_id,
        kind=SignalKind.TRADE,
        instrument_id=instrument_id,
        side=side,
        strength=float(strength),
        time_horizon=time_horizon,
        valid_until=snapshot.meta.decision_as_of,
        payload=payload,
        provenance=SignalProvenance(
            plugin_id=plugin_id,
            plugin_version=plugin_version,
            config_hash=config_hash,
            input_fingerprint=snapshot.meta.snapshot_id,
        ),
    )


def _trim_window(values: List[float], max_length: int) -> List[float]:
    if len(values) <= max_length:
        return values
    return values[-max_length:]


def _feature_array_from_bars(bars: Sequence, feature_name: str) -> np.ndarray:
    values = np.full(len(bars), np.nan, dtype=np.float64)
    for index, bar in enumerate(bars):
        raw_value = bar.extras.get(feature_name)
        if raw_value is None:
            continue
        try:
            values[index] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return values


def _with_target_position(payload: Dict[str, object], *, target_position: int) -> Dict[str, object]:
    normalized = dict(payload)
    normalized["target_position"] = int(target_position)
    risk_hints = dict(normalized.get("risk_hints", {}))
    risk_hints["target_position"] = int(target_position)
    normalized["risk_hints"] = risk_hints
    return normalized


def _long_only_target_position(direction: int) -> int:
    return 1 if int(direction) == int(SIGNAL_BUY) else 0


@dataclass
class MovingAverageCrossConfig:
    instrument_id: str
    timeframe: str
    fast_window: int = 5
    slow_window: int = 20
    entry_threshold: float = 0.0
    time_horizon: str = "swing"

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.upper()
        if self.fast_window < 1:
            raise ValueError("fast_window must be >= 1")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window")
        if self.entry_threshold < 0:
            raise ValueError("entry_threshold must be >= 0")


@dataclass
class PriceMovingAverageConfig:
    instrument_id: str
    timeframe: str
    period: int = 5
    entry_threshold: float = 0.0
    time_horizon: str = "swing"

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.upper()
        if self.period < 2:
            raise ValueError("period must be >= 2")
        if self.entry_threshold < 0:
            raise ValueError("entry_threshold must be >= 0")


@dataclass
class RsiReversionConfig:
    instrument_id: str
    timeframe: str
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    time_horizon: str = "mean_reversion"

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.upper()
        if self.period < 2:
            raise ValueError("period must be >= 2")
        if self.oversold <= 0 or self.oversold >= 100:
            raise ValueError("oversold must be within (0, 100)")
        if self.overbought <= 0 or self.overbought >= 100:
            raise ValueError("overbought must be within (0, 100)")
        if self.oversold >= self.overbought:
            raise ValueError("oversold must be less than overbought")


@dataclass
class DonchianBreakoutConfig:
    instrument_id: str
    timeframe: str
    lookback_window: int = 20
    breakout_buffer_bps: float = 0.0
    time_horizon: str = "trend"

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.upper()
        if self.lookback_window < 2:
            raise ValueError("lookback_window must be >= 2")
        if self.breakout_buffer_bps < 0:
            raise ValueError("breakout_buffer_bps must be >= 0")


@dataclass
class AdxTrendStrengthConfig:
    instrument_id: str
    timeframe: str
    period: int = 14
    adx_threshold: float = 25.0
    time_horizon: str = "trend"

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.upper()
        if self.period < 2:
            raise ValueError("period must be >= 2")
        if self.adx_threshold < 0.0:
            raise ValueError("adx_threshold must be >= 0")


@dataclass
class AtrBreakoutConfig:
    instrument_id: str
    timeframe: str
    ma_period: int = 20
    atr_period: int = 14
    atr_multiplier: float = 2.0
    time_horizon: str = "trend"

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.upper()
        if self.ma_period < 1:
            raise ValueError("ma_period must be >= 1")
        if self.atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if self.atr_multiplier <= 0.0:
            raise ValueError("atr_multiplier must be > 0")


@dataclass
class BollingerMeanReversionConfig:
    instrument_id: str
    timeframe: str
    period: int = 20
    stddev_multiplier: float = 2.0
    time_horizon: str = "mean_reversion"

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.upper()
        if self.period < 2:
            raise ValueError("period must be >= 2")
        if self.stddev_multiplier <= 0.0:
            raise ValueError("stddev_multiplier must be > 0")


@dataclass
class MacdTrendFollowConfig:
    instrument_id: str
    timeframe: str
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    histogram_threshold: float = 0.0
    time_horizon: str = "trend"

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.upper()
        if self.fast_period < 1:
            raise ValueError("fast_period must be >= 1")
        if self.slow_period <= self.fast_period:
            raise ValueError("slow_period must be greater than fast_period")
        if self.signal_period < 1:
            raise ValueError("signal_period must be >= 1")
        if self.histogram_threshold < 0.0:
            raise ValueError("histogram_threshold must be >= 0")


@dataclass
class OiFundingBreakoutConfig:
    instrument_id: str
    timeframe: str
    lookback_window: int = 20
    breakout_buffer_bps: float = 0.0
    oi_threshold_bps: float = 0.0
    max_funding_rate_bps: float = 100.0
    time_horizon: str = "trend"

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.upper()
        if self.lookback_window < 2:
            raise ValueError("lookback_window must be >= 2")
        if self.breakout_buffer_bps < 0.0:
            raise ValueError("breakout_buffer_bps must be >= 0")
        if self.max_funding_rate_bps < 0.0:
            raise ValueError("max_funding_rate_bps must be >= 0")


@dataclass
class LiquidationReversalConfig:
    instrument_id: str
    timeframe: str
    long_liquidation_threshold_usd: float = 100_000.0
    short_liquidation_threshold_usd: float = 100_000.0
    liquidation_imbalance_ratio: float = 0.60
    time_horizon: str = "reversal"

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.upper()
        if self.long_liquidation_threshold_usd < 0.0:
            raise ValueError("long_liquidation_threshold_usd must be >= 0")
        if self.short_liquidation_threshold_usd < 0.0:
            raise ValueError("short_liquidation_threshold_usd must be >= 0")
        if self.liquidation_imbalance_ratio <= 0.0 or self.liquidation_imbalance_ratio > 1.0:
            raise ValueError("liquidation_imbalance_ratio must be within (0, 1]")


@dataclass
class MultiTimeframeMaSpreadConfig:
    instrument_id: str
    primary_timeframe: str
    reference_timeframe: str
    primary_ma_period: int = 20
    reference_ma_period: int = 20
    spread_threshold_bps: float = 0.0
    time_horizon: str = "intraday"

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.upper()
        if self.primary_ma_period < 1:
            raise ValueError("primary_ma_period must be >= 1")
        if self.reference_ma_period < 1:
            raise ValueError("reference_ma_period must be >= 1")
        if self.spread_threshold_bps < 0:
            raise ValueError("spread_threshold_bps must be >= 0")


class MovingAverageCrossPlugin:
    plugin_id = "moving_average_cross"
    plugin_version = "numba/v1"

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def run(
        self,
        snapshot: DataSnapshot,
        config: MovingAverageCrossConfig,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        encoded = encode_close_series(snapshot, config.instrument_id, config.timeframe)
        if encoded.closes.size == 0:
            return _build_batch(self.plugin_id, snapshot, [])
        directions, strengths, fast_values, slow_values, spreads = moving_average_cross_signal_rows(
            encoded.closes,
            config.fast_window,
            config.slow_window,
            config.entry_threshold,
        )
        signals = []
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.fast_window,
            config.slow_window,
        )
        blocked = _blocked_instruments(context)
        for index, direction in enumerate(directions):
            if int(direction) == int(SIGNAL_NONE):
                continue
            signals.append(
                _build_signal(
                    snapshot=snapshot,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    instrument_id=config.instrument_id,
                    timeframe=config.timeframe,
                    time_horizon=config.time_horizon,
                    timestamp=int(encoded.timestamps[index]),
                    direction=int(direction),
                    strength=float(strengths[index]),
                    payload=_with_target_position(
                        {
                            "bar_timestamp": int(encoded.timestamps[index]),
                            "fast_sma": float(fast_values[index]),
                            "slow_sma": float(slow_values[index]),
                            "spread": float(spreads[index]),
                            "blocked_instruments": blocked,
                        },
                        target_position=_long_only_target_position(int(direction)),
                    ),
                    config_hash=config_hash,
                )
            )
        return _build_batch(self.plugin_id, snapshot, signals)

    def initialize_state(self) -> SignalState:
        return SignalState(plugin_id=self.plugin_id, values={"cursor": None, "close_window": []})

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: MovingAverageCrossConfig,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return StepSignalResult(state=state, signals=[])
        cursor = state.values.get("cursor")
        close_window = [float(value) for value in state.values.get("close_window", [])]
        emitted: List[SignalEnvelope] = []
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.fast_window,
            config.slow_window,
        )

        for bar in bars:
            if cursor is not None and bar.timestamp <= cursor:
                continue
            close_window.append(float(bar.close))
            close_window = _trim_window(close_window, config.slow_window)
            closes = np.asarray(close_window, dtype=np.float64)
            directions, strengths, fast_values, slow_values, spreads = moving_average_cross_signal_rows(
                closes,
                config.fast_window,
                config.slow_window,
                config.entry_threshold,
            )
            last_index = closes.shape[0] - 1
            if last_index >= 0 and int(directions[last_index]) != int(SIGNAL_NONE):
                emitted.append(
                    _build_signal(
                        snapshot=snapshot,
                        plugin_id=self.plugin_id,
                        plugin_version=self.plugin_version,
                        instrument_id=config.instrument_id,
                        timeframe=config.timeframe,
                        time_horizon=config.time_horizon,
                        timestamp=int(bar.timestamp),
                        direction=int(directions[last_index]),
                        strength=float(strengths[last_index]),
                        payload=_with_target_position(
                            {
                                "bar_timestamp": int(bar.timestamp),
                                "fast_sma": float(fast_values[last_index]),
                                "slow_sma": float(slow_values[last_index]),
                                "spread": float(spreads[last_index]),
                                "blocked_instruments": blocked,
                            },
                            target_position=_long_only_target_position(int(directions[last_index])),
                        ),
                        config_hash=config_hash,
                    )
                )
            cursor = bar.timestamp
        return StepSignalResult(
            state=SignalState(
                plugin_id=self.plugin_id,
                values={"cursor": cursor, "close_window": close_window},
            ),
            signals=emitted,
        )


class PriceMovingAveragePlugin:
    plugin_id = "price_moving_average"
    plugin_version = "numba/v1"

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def run(
        self,
        snapshot: DataSnapshot,
        config: PriceMovingAverageConfig,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        encoded = encode_close_series(snapshot, config.instrument_id, config.timeframe)
        if encoded.closes.size == 0:
            return _build_batch(self.plugin_id, snapshot, [])
        directions, strengths, current_ma_values, prev_ma_values, spreads = price_moving_average_signal_rows(
            encoded.closes,
            config.period,
            config.entry_threshold,
        )
        signals = []
        config_hash = _config_hash(config.instrument_id, config.timeframe, config.period)
        blocked = _blocked_instruments(context)
        for index, direction in enumerate(directions):
            if int(direction) == int(SIGNAL_NONE):
                continue
            current_price = float(encoded.closes[index])
            prev_price = float(encoded.closes[index - 1]) if index > 0 else current_price
            signals.append(
                _build_signal(
                    snapshot=snapshot,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    instrument_id=config.instrument_id,
                    timeframe=config.timeframe,
                    time_horizon=config.time_horizon,
                    timestamp=int(encoded.timestamps[index]),
                    direction=int(direction),
                    strength=float(strengths[index]),
                    payload=_with_target_position(
                        {
                            "bar_timestamp": int(encoded.timestamps[index]),
                            "current_price": current_price,
                            "current_ma": float(current_ma_values[index]),
                            "prev_price": prev_price,
                            "prev_ma": float(prev_ma_values[index]),
                            "spread": float(spreads[index]),
                            "blocked_instruments": blocked,
                        },
                        target_position=_long_only_target_position(int(direction)),
                    ),
                    config_hash=config_hash,
                )
            )
        return _build_batch(self.plugin_id, snapshot, signals)

    def initialize_state(self) -> SignalState:
        return SignalState(plugin_id=self.plugin_id, values={"cursor": None, "close_window": []})

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: PriceMovingAverageConfig,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return StepSignalResult(state=state, signals=[])
        cursor = state.values.get("cursor")
        close_window = [float(value) for value in state.values.get("close_window", [])]
        emitted: List[SignalEnvelope] = []
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(config.instrument_id, config.timeframe, config.period)

        for bar in bars:
            if cursor is not None and bar.timestamp <= cursor:
                continue
            close_window.append(float(bar.close))
            close_window = _trim_window(close_window, config.period + 2)
            closes = np.asarray(close_window, dtype=np.float64)
            directions, strengths, current_ma_values, prev_ma_values, spreads = price_moving_average_signal_rows(
                closes,
                config.period,
                config.entry_threshold,
            )
            last_index = closes.shape[0] - 1
            if last_index >= 0 and int(directions[last_index]) != int(SIGNAL_NONE):
                current_price = float(closes[last_index])
                prev_price = float(closes[last_index - 1]) if last_index > 0 else current_price
                emitted.append(
                    _build_signal(
                        snapshot=snapshot,
                        plugin_id=self.plugin_id,
                        plugin_version=self.plugin_version,
                        instrument_id=config.instrument_id,
                        timeframe=config.timeframe,
                        time_horizon=config.time_horizon,
                        timestamp=int(bar.timestamp),
                        direction=int(directions[last_index]),
                        strength=float(strengths[last_index]),
                        payload=_with_target_position(
                            {
                                "bar_timestamp": int(bar.timestamp),
                                "current_price": current_price,
                                "current_ma": float(current_ma_values[last_index]),
                                "prev_price": prev_price,
                                "prev_ma": float(prev_ma_values[last_index]),
                                "spread": float(spreads[last_index]),
                                "blocked_instruments": blocked,
                            },
                            target_position=_long_only_target_position(int(directions[last_index])),
                        ),
                        config_hash=config_hash,
                    )
                )
            cursor = bar.timestamp
        return StepSignalResult(
            state=SignalState(
                plugin_id=self.plugin_id,
                values={"cursor": cursor, "close_window": close_window},
            ),
            signals=emitted,
        )


class RsiReversionPlugin:
    plugin_id = "rsi_reversion"
    plugin_version = "numba/v1"

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def run(
        self,
        snapshot: DataSnapshot,
        config: RsiReversionConfig,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        encoded = encode_close_series(snapshot, config.instrument_id, config.timeframe)
        if encoded.closes.size == 0:
            return _build_batch(self.plugin_id, snapshot, [])
        directions, strengths, rsi_values = rsi_reversion_signal_rows(
            encoded.closes,
            config.period,
            config.oversold,
            config.overbought,
        )
        signals = []
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.period,
            config.oversold,
            config.overbought,
        )
        for index, direction in enumerate(directions):
            if int(direction) == int(SIGNAL_NONE):
                continue
            signals.append(
                _build_signal(
                    snapshot=snapshot,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    instrument_id=config.instrument_id,
                    timeframe=config.timeframe,
                    time_horizon=config.time_horizon,
                    timestamp=int(encoded.timestamps[index]),
                    direction=int(direction),
                    strength=float(strengths[index]),
                    payload=_with_target_position(
                        {
                            "bar_timestamp": int(encoded.timestamps[index]),
                            "rsi": float(rsi_values[index]),
                            "oversold": float(config.oversold),
                            "overbought": float(config.overbought),
                            "blocked_instruments": blocked,
                        },
                        target_position=_long_only_target_position(int(direction)),
                    ),
                    config_hash=config_hash,
                )
            )
        return _build_batch(self.plugin_id, snapshot, signals)

    def initialize_state(self) -> SignalState:
        return SignalState(plugin_id=self.plugin_id, values={"cursor": None, "close_window": []})

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: RsiReversionConfig,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return StepSignalResult(state=state, signals=[])
        cursor = state.values.get("cursor")
        close_window = [float(value) for value in state.values.get("close_window", [])]
        emitted: List[SignalEnvelope] = []
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.period,
            config.oversold,
            config.overbought,
        )

        for bar in bars:
            if cursor is not None and bar.timestamp <= cursor:
                continue
            close_window.append(float(bar.close))
            close_window = _trim_window(close_window, config.period + 1)
            closes = np.asarray(close_window, dtype=np.float64)
            directions, strengths, rsi_values = rsi_reversion_signal_rows(
                closes,
                config.period,
                config.oversold,
                config.overbought,
            )
            last_index = closes.shape[0] - 1
            if last_index >= 0 and int(directions[last_index]) != int(SIGNAL_NONE):
                emitted.append(
                    _build_signal(
                        snapshot=snapshot,
                        plugin_id=self.plugin_id,
                        plugin_version=self.plugin_version,
                        instrument_id=config.instrument_id,
                        timeframe=config.timeframe,
                        time_horizon=config.time_horizon,
                        timestamp=int(bar.timestamp),
                        direction=int(directions[last_index]),
                        strength=float(strengths[last_index]),
                        payload=_with_target_position(
                            {
                                "bar_timestamp": int(bar.timestamp),
                                "rsi": float(rsi_values[last_index]),
                                "oversold": float(config.oversold),
                                "overbought": float(config.overbought),
                                "blocked_instruments": blocked,
                            },
                            target_position=_long_only_target_position(int(directions[last_index])),
                        ),
                        config_hash=config_hash,
                    )
                )
            cursor = bar.timestamp

        return StepSignalResult(
            state=SignalState(
                plugin_id=self.plugin_id,
                values={"cursor": cursor, "close_window": close_window},
            ),
            signals=emitted,
        )


class MultiTimeframeMaSpreadPlugin:
    plugin_id = "multi_timeframe_ma_spread"
    plugin_version = "numba/v1"

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def run(
        self,
        snapshot: DataSnapshot,
        config: MultiTimeframeMaSpreadConfig,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        primary_encoded = encode_close_series(snapshot, config.instrument_id, config.primary_timeframe)
        secondary_encoded = encode_close_series(snapshot, config.instrument_id, config.reference_timeframe)
        if primary_encoded.closes.size == 0 or secondary_encoded.closes.size == 0:
            return _build_batch(self.plugin_id, snapshot, [])

        directions, strengths, primary_ma_values, secondary_ma_values, spread_bps_values = multi_timeframe_ma_spread_signal_rows(
            primary_encoded.timestamps,
            primary_encoded.closes,
            secondary_encoded.timestamps,
            secondary_encoded.closes,
            config.primary_ma_period,
            config.reference_ma_period,
            config.spread_threshold_bps,
        )
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.primary_timeframe,
            config.reference_timeframe,
            config.primary_ma_period,
            config.reference_ma_period,
            config.spread_threshold_bps,
        )
        signals = []
        previous_direction = SIGNAL_NONE
        for index, direction in enumerate(directions):
            if int(direction) == int(SIGNAL_NONE):
                previous_direction = SIGNAL_NONE
                continue
            if int(direction) == int(previous_direction):
                previous_direction = direction
                continue
            signals.append(
                _build_signal(
                    snapshot=snapshot,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    instrument_id=config.instrument_id,
                    timeframe=config.primary_timeframe,
                    time_horizon=config.time_horizon,
                    timestamp=int(primary_encoded.timestamps[index]),
                    direction=int(direction),
                    strength=float(strengths[index]),
                    payload=_with_target_position(
                        {
                            "bar_timestamp": int(primary_encoded.timestamps[index]),
                            "primary_timeframe": config.primary_timeframe,
                            "reference_timeframe": config.reference_timeframe,
                            "primary_ma": float(primary_ma_values[index]),
                            "reference_ma": float(secondary_ma_values[index]),
                            "spread_bps": float(spread_bps_values[index]),
                            "blocked_instruments": blocked,
                        },
                        target_position=1 if int(direction) == int(SIGNAL_BUY) else -1,
                    ),
                    config_hash=config_hash,
                )
            )
            previous_direction = direction
        return _build_batch(self.plugin_id, snapshot, signals)

    def initialize_state(self) -> SignalState:
        return SignalState(
            plugin_id=self.plugin_id,
            values={
                "primary_cursor": None,
                "reference_cursor": None,
                "last_direction": int(SIGNAL_NONE),
                "primary_timestamp_window": [],
                "primary_close_window": [],
                "reference_timestamp_window": [],
                "reference_close_window": [],
            },
        )

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: MultiTimeframeMaSpreadConfig,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        primary_bars = series_for(snapshot, config.instrument_id, config.primary_timeframe)
        reference_bars = series_for(snapshot, config.instrument_id, config.reference_timeframe)
        if not primary_bars or not reference_bars:
            return StepSignalResult(state=state, signals=[])

        primary_cursor = state.values.get("primary_cursor")
        reference_cursor = state.values.get("reference_cursor")
        last_direction = int(state.values.get("last_direction", int(SIGNAL_NONE)))
        primary_timestamp_window = [int(value) for value in state.values.get("primary_timestamp_window", [])]
        primary_close_window = [float(value) for value in state.values.get("primary_close_window", [])]
        reference_timestamp_window = [int(value) for value in state.values.get("reference_timestamp_window", [])]
        reference_close_window = [float(value) for value in state.values.get("reference_close_window", [])]
        emitted: List[SignalEnvelope] = []
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.primary_timeframe,
            config.reference_timeframe,
            config.primary_ma_period,
            config.reference_ma_period,
            config.spread_threshold_bps,
        )

        for primary_bar in primary_bars:
            if primary_cursor is not None and primary_bar.timestamp <= primary_cursor:
                continue
            for reference_bar in reference_bars:
                if reference_bar.timestamp > primary_bar.timestamp:
                    break
                if reference_cursor is not None and reference_bar.timestamp <= reference_cursor:
                    continue
                reference_timestamp_window.append(int(reference_bar.timestamp))
                reference_close_window.append(float(reference_bar.close))
                if len(reference_timestamp_window) > config.reference_ma_period + 1:
                    reference_timestamp_window = reference_timestamp_window[-(config.reference_ma_period + 1) :]
                reference_close_window = _trim_window(reference_close_window, config.reference_ma_period + 1)
                reference_cursor = reference_bar.timestamp

            primary_timestamp_window.append(int(primary_bar.timestamp))
            primary_close_window.append(float(primary_bar.close))
            if len(primary_timestamp_window) > config.primary_ma_period + 1:
                primary_timestamp_window = primary_timestamp_window[-(config.primary_ma_period + 1) :]
            primary_close_window = _trim_window(primary_close_window, config.primary_ma_period + 1)
            full_primary_timestamps = np.asarray(primary_timestamp_window, dtype=np.int64)
            full_primary_closes = np.asarray(primary_close_window, dtype=np.float64)
            full_reference_timestamps = np.asarray(reference_timestamp_window, dtype=np.int64)
            full_reference_closes = np.asarray(reference_close_window, dtype=np.float64)

            directions, strengths, primary_ma_values, secondary_ma_values, spread_bps_values = multi_timeframe_ma_spread_signal_rows(
                full_primary_timestamps,
                full_primary_closes,
                full_reference_timestamps,
                full_reference_closes,
                config.primary_ma_period,
                config.reference_ma_period,
                config.spread_threshold_bps,
            )
            last_index = full_primary_closes.shape[0] - 1
            if last_index >= 0 and int(directions[last_index]) != int(SIGNAL_NONE):
                if int(directions[last_index]) != int(last_direction):
                    emitted.append(
                        _build_signal(
                            snapshot=snapshot,
                            plugin_id=self.plugin_id,
                            plugin_version=self.plugin_version,
                            instrument_id=config.instrument_id,
                            timeframe=config.primary_timeframe,
                            time_horizon=config.time_horizon,
                            timestamp=int(primary_bar.timestamp),
                            direction=int(directions[last_index]),
                            strength=float(strengths[last_index]),
                            payload=_with_target_position(
                                {
                                    "bar_timestamp": int(primary_bar.timestamp),
                                    "primary_timeframe": config.primary_timeframe,
                                    "reference_timeframe": config.reference_timeframe,
                                    "primary_ma": float(primary_ma_values[last_index]),
                                    "reference_ma": float(secondary_ma_values[last_index]),
                                    "spread_bps": float(spread_bps_values[last_index]),
                                    "blocked_instruments": blocked,
                                },
                                target_position=1 if int(directions[last_index]) == int(SIGNAL_BUY) else -1,
                            ),
                            config_hash=config_hash,
                        )
                    )
                    last_direction = int(directions[last_index])
            elif last_index >= 0:
                last_direction = int(SIGNAL_NONE)
            primary_cursor = primary_bar.timestamp

        return StepSignalResult(
            state=SignalState(
                plugin_id=self.plugin_id,
                values={
                    "primary_cursor": primary_cursor,
                    "reference_cursor": reference_cursor,
                    "last_direction": last_direction,
                    "primary_timestamp_window": primary_timestamp_window,
                    "primary_close_window": primary_close_window,
                    "reference_timestamp_window": reference_timestamp_window,
                    "reference_close_window": reference_close_window,
                },
            ),
            signals=emitted,
        )


class DonchianBreakoutPlugin:
    plugin_id = "donchian_breakout"
    plugin_version = "numba/v1"

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def run(
        self,
        snapshot: DataSnapshot,
        config: DonchianBreakoutConfig,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return _build_batch(self.plugin_id, snapshot, [])

        timestamps = np.asarray([int(bar.timestamp) for bar in bars], dtype=np.int64)
        closes = np.asarray([float(bar.close) for bar in bars], dtype=np.float64)
        highs = np.asarray([float(bar.high) for bar in bars], dtype=np.float64)
        lows = np.asarray([float(bar.low) for bar in bars], dtype=np.float64)
        directions, strengths, upper_band_values, lower_band_values, breakout_bps_values = donchian_breakout_signal_rows(
            closes,
            highs,
            lows,
            config.lookback_window,
            config.breakout_buffer_bps,
        )
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.lookback_window,
            config.breakout_buffer_bps,
        )
        signals = []
        previous_direction = SIGNAL_NONE
        for index, direction in enumerate(directions):
            if int(direction) == int(SIGNAL_NONE):
                previous_direction = SIGNAL_NONE
                continue
            if int(direction) == int(previous_direction):
                previous_direction = direction
                continue
            target_position = 1 if int(direction) == int(SIGNAL_BUY) else -1
            signals.append(
                _build_signal(
                    snapshot=snapshot,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    instrument_id=config.instrument_id,
                    timeframe=config.timeframe,
                    time_horizon=config.time_horizon,
                    timestamp=int(timestamps[index]),
                    direction=int(direction),
                    strength=float(strengths[index]),
                    payload=_with_target_position(
                        {
                            "bar_timestamp": int(timestamps[index]),
                            "upper_band": float(upper_band_values[index]),
                            "lower_band": float(lower_band_values[index]),
                            "breakout_bps": float(breakout_bps_values[index]),
                            "breakout_buffer_bps": float(config.breakout_buffer_bps),
                            "lookback_window": int(config.lookback_window),
                            "blocked_instruments": blocked,
                        },
                        target_position=target_position,
                    ),
                    config_hash=config_hash,
                )
            )
            previous_direction = direction
        return _build_batch(self.plugin_id, snapshot, signals)

    def initialize_state(self) -> SignalState:
        return SignalState(
            plugin_id=self.plugin_id,
            values={
                "cursor": None,
                "last_direction": int(SIGNAL_NONE),
                "timestamp_window": [],
                "close_window": [],
                "high_window": [],
                "low_window": [],
            },
        )

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: DonchianBreakoutConfig,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return StepSignalResult(state=state, signals=[])

        cursor = state.values.get("cursor")
        last_direction = int(state.values.get("last_direction", int(SIGNAL_NONE)))
        timestamp_window = [int(value) for value in state.values.get("timestamp_window", [])]
        close_window = [float(value) for value in state.values.get("close_window", [])]
        high_window = [float(value) for value in state.values.get("high_window", [])]
        low_window = [float(value) for value in state.values.get("low_window", [])]
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.lookback_window,
            config.breakout_buffer_bps,
        )
        emitted: List[SignalEnvelope] = []

        for bar in bars:
            if cursor is not None and bar.timestamp <= cursor:
                continue
            timestamp_window.append(int(bar.timestamp))
            close_window.append(float(bar.close))
            high_window.append(float(bar.high))
            low_window.append(float(bar.low))
            max_length = config.lookback_window + 1
            if len(timestamp_window) > max_length:
                timestamp_window = timestamp_window[-max_length:]
            close_window = _trim_window(close_window, max_length)
            high_window = _trim_window(high_window, max_length)
            low_window = _trim_window(low_window, max_length)

            directions, strengths, upper_band_values, lower_band_values, breakout_bps_values = donchian_breakout_signal_rows(
                np.asarray(close_window, dtype=np.float64),
                np.asarray(high_window, dtype=np.float64),
                np.asarray(low_window, dtype=np.float64),
                config.lookback_window,
                config.breakout_buffer_bps,
            )
            last_index = len(close_window) - 1
            if last_index >= 0 and int(directions[last_index]) != int(SIGNAL_NONE):
                if int(directions[last_index]) != int(last_direction):
                    target_position = 1 if int(directions[last_index]) == int(SIGNAL_BUY) else -1
                    emitted.append(
                        _build_signal(
                            snapshot=snapshot,
                            plugin_id=self.plugin_id,
                            plugin_version=self.plugin_version,
                            instrument_id=config.instrument_id,
                            timeframe=config.timeframe,
                            time_horizon=config.time_horizon,
                            timestamp=int(bar.timestamp),
                            direction=int(directions[last_index]),
                            strength=float(strengths[last_index]),
                            payload=_with_target_position(
                                {
                                    "bar_timestamp": int(bar.timestamp),
                                    "upper_band": float(upper_band_values[last_index]),
                                    "lower_band": float(lower_band_values[last_index]),
                                    "breakout_bps": float(breakout_bps_values[last_index]),
                                    "breakout_buffer_bps": float(config.breakout_buffer_bps),
                                    "lookback_window": int(config.lookback_window),
                                    "blocked_instruments": blocked,
                                },
                                target_position=target_position,
                            ),
                            config_hash=config_hash,
                        )
                    )
                    last_direction = int(directions[last_index])
            elif last_index >= 0:
                last_direction = int(SIGNAL_NONE)
            cursor = bar.timestamp

        return StepSignalResult(
            state=SignalState(
                plugin_id=self.plugin_id,
                values={
                    "cursor": cursor,
                    "last_direction": last_direction,
                    "timestamp_window": timestamp_window,
                    "close_window": close_window,
                    "high_window": high_window,
                    "low_window": low_window,
                },
            ),
            signals=emitted,
        )


class OiFundingBreakoutPlugin:
    plugin_id = "oi_funding_breakout"
    plugin_version = "numba/v1"

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def run(
        self,
        snapshot: DataSnapshot,
        config: OiFundingBreakoutConfig,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return _build_batch(self.plugin_id, snapshot, [])

        timestamps = np.asarray([int(bar.timestamp) for bar in bars], dtype=np.int64)
        closes = np.asarray([float(bar.close) for bar in bars], dtype=np.float64)
        highs = np.asarray([float(bar.high) for bar in bars], dtype=np.float64)
        lows = np.asarray([float(bar.low) for bar in bars], dtype=np.float64)
        oi_change_bps = _feature_array_from_bars(bars, "oi_change_bps")
        funding_rate_bps = _feature_array_from_bars(bars, "funding_rate_bps")
        (
            directions,
            strengths,
            upper_band_values,
            lower_band_values,
            breakout_bps_values,
            oi_values,
            funding_values,
        ) = oi_funding_breakout_signal_rows(
            closes,
            highs,
            lows,
            oi_change_bps,
            funding_rate_bps,
            config.lookback_window,
            config.breakout_buffer_bps,
            config.oi_threshold_bps,
            config.max_funding_rate_bps,
        )
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.lookback_window,
            config.breakout_buffer_bps,
            config.oi_threshold_bps,
            config.max_funding_rate_bps,
        )
        signals = []
        previous_direction = SIGNAL_NONE
        for index, direction in enumerate(directions):
            if int(direction) == int(SIGNAL_NONE):
                previous_direction = SIGNAL_NONE
                continue
            if int(direction) == int(previous_direction):
                previous_direction = direction
                continue
            target_position = 1 if int(direction) == int(SIGNAL_BUY) else -1
            signals.append(
                _build_signal(
                    snapshot=snapshot,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    instrument_id=config.instrument_id,
                    timeframe=config.timeframe,
                    time_horizon=config.time_horizon,
                    timestamp=int(timestamps[index]),
                    direction=int(direction),
                    strength=float(strengths[index]),
                    payload=_with_target_position(
                        {
                            "bar_timestamp": int(timestamps[index]),
                            "upper_band": float(upper_band_values[index]),
                            "lower_band": float(lower_band_values[index]),
                            "breakout_bps": float(breakout_bps_values[index]),
                            "oi_change_bps": float(oi_values[index]),
                            "funding_rate_bps": float(funding_values[index]),
                            "oi_threshold_bps": float(config.oi_threshold_bps),
                            "max_funding_rate_bps": float(config.max_funding_rate_bps),
                            "blocked_instruments": blocked,
                        },
                        target_position=target_position,
                    ),
                    config_hash=config_hash,
                )
            )
            previous_direction = direction
        return _build_batch(self.plugin_id, snapshot, signals)

    def initialize_state(self) -> SignalState:
        return SignalState(
            plugin_id=self.plugin_id,
            values={
                "cursor": None,
                "last_direction": int(SIGNAL_NONE),
                "timestamp_window": [],
                "close_window": [],
                "high_window": [],
                "low_window": [],
                "oi_window": [],
                "funding_window": [],
            },
        )

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: OiFundingBreakoutConfig,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return StepSignalResult(state=state, signals=[])

        cursor = state.values.get("cursor")
        last_direction = int(state.values.get("last_direction", int(SIGNAL_NONE)))
        timestamp_window = [int(value) for value in state.values.get("timestamp_window", [])]
        close_window = [float(value) for value in state.values.get("close_window", [])]
        high_window = [float(value) for value in state.values.get("high_window", [])]
        low_window = [float(value) for value in state.values.get("low_window", [])]
        oi_window = [float(value) for value in state.values.get("oi_window", [])]
        funding_window = [float(value) for value in state.values.get("funding_window", [])]
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.lookback_window,
            config.breakout_buffer_bps,
            config.oi_threshold_bps,
            config.max_funding_rate_bps,
        )
        emitted: List[SignalEnvelope] = []

        for bar in bars:
            if cursor is not None and bar.timestamp <= cursor:
                continue
            timestamp_window.append(int(bar.timestamp))
            close_window.append(float(bar.close))
            high_window.append(float(bar.high))
            low_window.append(float(bar.low))
            raw_oi = bar.extras.get("oi_change_bps")
            raw_funding = bar.extras.get("funding_rate_bps")
            oi_window.append(float(raw_oi) if raw_oi is not None else np.nan)
            funding_window.append(float(raw_funding) if raw_funding is not None else np.nan)
            max_length = config.lookback_window + 1
            if len(timestamp_window) > max_length:
                timestamp_window = timestamp_window[-max_length:]
            close_window = _trim_window(close_window, max_length)
            high_window = _trim_window(high_window, max_length)
            low_window = _trim_window(low_window, max_length)
            oi_window = _trim_window(oi_window, max_length)
            funding_window = _trim_window(funding_window, max_length)

            (
                directions,
                strengths,
                upper_band_values,
                lower_band_values,
                breakout_bps_values,
                oi_values,
                funding_values,
            ) = oi_funding_breakout_signal_rows(
                np.asarray(close_window, dtype=np.float64),
                np.asarray(high_window, dtype=np.float64),
                np.asarray(low_window, dtype=np.float64),
                np.asarray(oi_window, dtype=np.float64),
                np.asarray(funding_window, dtype=np.float64),
                config.lookback_window,
                config.breakout_buffer_bps,
                config.oi_threshold_bps,
                config.max_funding_rate_bps,
            )
            last_index = len(close_window) - 1
            if last_index >= 0 and int(directions[last_index]) != int(SIGNAL_NONE):
                if int(directions[last_index]) != int(last_direction):
                    target_position = 1 if int(directions[last_index]) == int(SIGNAL_BUY) else -1
                    emitted.append(
                        _build_signal(
                            snapshot=snapshot,
                            plugin_id=self.plugin_id,
                            plugin_version=self.plugin_version,
                            instrument_id=config.instrument_id,
                            timeframe=config.timeframe,
                            time_horizon=config.time_horizon,
                            timestamp=int(bar.timestamp),
                            direction=int(directions[last_index]),
                            strength=float(strengths[last_index]),
                            payload=_with_target_position(
                                {
                                    "bar_timestamp": int(bar.timestamp),
                                    "upper_band": float(upper_band_values[last_index]),
                                    "lower_band": float(lower_band_values[last_index]),
                                    "breakout_bps": float(breakout_bps_values[last_index]),
                                    "oi_change_bps": float(oi_values[last_index]),
                                    "funding_rate_bps": float(funding_values[last_index]),
                                    "oi_threshold_bps": float(config.oi_threshold_bps),
                                    "max_funding_rate_bps": float(config.max_funding_rate_bps),
                                    "blocked_instruments": blocked,
                                },
                                target_position=target_position,
                            ),
                            config_hash=config_hash,
                        )
                    )
                    last_direction = int(directions[last_index])
            elif last_index >= 0:
                last_direction = int(SIGNAL_NONE)
            cursor = bar.timestamp

        return StepSignalResult(
            state=SignalState(
                plugin_id=self.plugin_id,
                values={
                    "cursor": cursor,
                    "last_direction": last_direction,
                    "timestamp_window": timestamp_window,
                    "close_window": close_window,
                    "high_window": high_window,
                    "low_window": low_window,
                    "oi_window": oi_window,
                    "funding_window": funding_window,
                },
            ),
            signals=emitted,
        )


class LiquidationReversalPlugin:
    plugin_id = "liquidation_reversal"
    plugin_version = "numba/v1"

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def run(
        self,
        snapshot: DataSnapshot,
        config: LiquidationReversalConfig,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return _build_batch(self.plugin_id, snapshot, [])

        timestamps = np.asarray([int(bar.timestamp) for bar in bars], dtype=np.int64)
        long_liq_notional_usd = _feature_array_from_bars(bars, "long_liq_notional_usd")
        short_liq_notional_usd = _feature_array_from_bars(bars, "short_liq_notional_usd")
        directions, strengths, total_notional_values, imbalance_values = liquidation_reversal_signal_rows(
            long_liq_notional_usd,
            short_liq_notional_usd,
            config.long_liquidation_threshold_usd,
            config.short_liquidation_threshold_usd,
            config.liquidation_imbalance_ratio,
        )
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.long_liquidation_threshold_usd,
            config.short_liquidation_threshold_usd,
            config.liquidation_imbalance_ratio,
        )
        signals = []
        previous_direction = SIGNAL_NONE
        for index, direction in enumerate(directions):
            if int(direction) == int(SIGNAL_NONE):
                previous_direction = SIGNAL_NONE
                continue
            if int(direction) == int(previous_direction):
                previous_direction = direction
                continue
            target_position = 1 if int(direction) == int(SIGNAL_BUY) else -1
            signals.append(
                _build_signal(
                    snapshot=snapshot,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    instrument_id=config.instrument_id,
                    timeframe=config.timeframe,
                    time_horizon=config.time_horizon,
                    timestamp=int(timestamps[index]),
                    direction=int(direction),
                    strength=float(strengths[index]),
                    payload=_with_target_position(
                        {
                            "bar_timestamp": int(timestamps[index]),
                            "long_liq_notional_usd": float(long_liq_notional_usd[index]),
                            "short_liq_notional_usd": float(short_liq_notional_usd[index]),
                            "total_liq_notional_usd": float(total_notional_values[index]),
                            "liq_imbalance_ratio": float(imbalance_values[index]),
                            "long_liquidation_threshold_usd": float(config.long_liquidation_threshold_usd),
                            "short_liquidation_threshold_usd": float(config.short_liquidation_threshold_usd),
                            "blocked_instruments": blocked,
                        },
                        target_position=target_position,
                    ),
                    config_hash=config_hash,
                )
            )
            previous_direction = direction
        return _build_batch(self.plugin_id, snapshot, signals)

    def initialize_state(self) -> SignalState:
        return SignalState(
            plugin_id=self.plugin_id,
            values={
                "cursor": None,
                "last_direction": int(SIGNAL_NONE),
                "timestamp_window": [],
                "long_window": [],
                "short_window": [],
            },
        )

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: LiquidationReversalConfig,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return StepSignalResult(state=state, signals=[])

        cursor = state.values.get("cursor")
        last_direction = int(state.values.get("last_direction", int(SIGNAL_NONE)))
        timestamp_window = [int(value) for value in state.values.get("timestamp_window", [])]
        long_window = [float(value) for value in state.values.get("long_window", [])]
        short_window = [float(value) for value in state.values.get("short_window", [])]
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.long_liquidation_threshold_usd,
            config.short_liquidation_threshold_usd,
            config.liquidation_imbalance_ratio,
        )
        emitted: List[SignalEnvelope] = []

        for bar in bars:
            if cursor is not None and bar.timestamp <= cursor:
                continue
            timestamp_window.append(int(bar.timestamp))
            long_window.append(float(bar.extras.get("long_liq_notional_usd") or 0.0))
            short_window.append(float(bar.extras.get("short_liq_notional_usd") or 0.0))
            timestamp_window = _trim_window(timestamp_window, 1)
            long_window = _trim_window(long_window, 1)
            short_window = _trim_window(short_window, 1)

            directions, strengths, total_notional_values, imbalance_values = liquidation_reversal_signal_rows(
                np.asarray(long_window, dtype=np.float64),
                np.asarray(short_window, dtype=np.float64),
                config.long_liquidation_threshold_usd,
                config.short_liquidation_threshold_usd,
                config.liquidation_imbalance_ratio,
            )
            last_index = len(long_window) - 1
            if last_index >= 0 and int(directions[last_index]) != int(SIGNAL_NONE):
                if int(directions[last_index]) != int(last_direction):
                    target_position = 1 if int(directions[last_index]) == int(SIGNAL_BUY) else -1
                    emitted.append(
                        _build_signal(
                            snapshot=snapshot,
                            plugin_id=self.plugin_id,
                            plugin_version=self.plugin_version,
                            instrument_id=config.instrument_id,
                            timeframe=config.timeframe,
                            time_horizon=config.time_horizon,
                            timestamp=int(bar.timestamp),
                            direction=int(directions[last_index]),
                            strength=float(strengths[last_index]),
                            payload=_with_target_position(
                                {
                                    "bar_timestamp": int(bar.timestamp),
                                    "long_liq_notional_usd": float(long_window[last_index]),
                                    "short_liq_notional_usd": float(short_window[last_index]),
                                    "total_liq_notional_usd": float(total_notional_values[last_index]),
                                    "liq_imbalance_ratio": float(imbalance_values[last_index]),
                                    "long_liquidation_threshold_usd": float(config.long_liquidation_threshold_usd),
                                    "short_liquidation_threshold_usd": float(config.short_liquidation_threshold_usd),
                                    "blocked_instruments": blocked,
                                },
                                target_position=target_position,
                            ),
                            config_hash=config_hash,
                        )
                    )
                    last_direction = int(directions[last_index])
            elif last_index >= 0:
                last_direction = int(SIGNAL_NONE)
            cursor = bar.timestamp

        return StepSignalResult(
            state=SignalState(
                plugin_id=self.plugin_id,
                values={
                    "cursor": cursor,
                    "last_direction": last_direction,
                    "timestamp_window": timestamp_window,
                    "long_window": long_window,
                    "short_window": short_window,
                },
            ),
            signals=emitted,
        )


class AdxTrendStrengthPlugin:
    plugin_id = "adx_trend_strength"
    plugin_version = "numba/v1"

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def run(
        self,
        snapshot: DataSnapshot,
        config: AdxTrendStrengthConfig,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return _build_batch(self.plugin_id, snapshot, [])

        timestamps = np.asarray([int(bar.timestamp) for bar in bars], dtype=np.int64)
        closes = np.asarray([float(bar.close) for bar in bars], dtype=np.float64)
        highs = np.asarray([float(bar.high) for bar in bars], dtype=np.float64)
        lows = np.asarray([float(bar.low) for bar in bars], dtype=np.float64)
        directions, strengths, adx_values, plus_di_values, minus_di_values = adx_trend_strength_signal_rows(
            closes,
            highs,
            lows,
            config.period,
            config.adx_threshold,
        )
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.period,
            config.adx_threshold,
        )
        signals = []
        previous_direction = SIGNAL_NONE
        for index, direction in enumerate(directions):
            if int(direction) == int(SIGNAL_NONE):
                previous_direction = SIGNAL_NONE
                continue
            if int(direction) == int(previous_direction):
                previous_direction = direction
                continue
            signals.append(
                _build_signal(
                    snapshot=snapshot,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    instrument_id=config.instrument_id,
                    timeframe=config.timeframe,
                    time_horizon=config.time_horizon,
                    timestamp=int(timestamps[index]),
                    direction=int(direction),
                    strength=float(strengths[index]),
                    payload=_with_target_position(
                        {
                            "bar_timestamp": int(timestamps[index]),
                            "adx": float(adx_values[index]),
                            "plus_di": float(plus_di_values[index]),
                            "minus_di": float(minus_di_values[index]),
                            "adx_threshold": float(config.adx_threshold),
                            "blocked_instruments": blocked,
                        },
                        target_position=1 if int(direction) == int(SIGNAL_BUY) else -1,
                    ),
                    config_hash=config_hash,
                )
            )
            previous_direction = direction
        return _build_batch(self.plugin_id, snapshot, signals)

    def initialize_state(self) -> SignalState:
        return SignalState(
            plugin_id=self.plugin_id,
            values={
                "cursor": None,
                "last_direction": int(SIGNAL_NONE),
                "close_window": [],
                "high_window": [],
                "low_window": [],
            },
        )

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: AdxTrendStrengthConfig,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return StepSignalResult(state=state, signals=[])

        cursor = state.values.get("cursor")
        last_direction = int(state.values.get("last_direction", int(SIGNAL_NONE)))
        close_window = [float(value) for value in state.values.get("close_window", [])]
        high_window = [float(value) for value in state.values.get("high_window", [])]
        low_window = [float(value) for value in state.values.get("low_window", [])]
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.period,
            config.adx_threshold,
        )
        emitted: List[SignalEnvelope] = []

        for bar in bars:
            if cursor is not None and bar.timestamp <= cursor:
                continue
            close_window.append(float(bar.close))
            high_window.append(float(bar.high))
            low_window.append(float(bar.low))
            directions, strengths, adx_values, plus_di_values, minus_di_values = adx_trend_strength_signal_rows(
                np.asarray(close_window, dtype=np.float64),
                np.asarray(high_window, dtype=np.float64),
                np.asarray(low_window, dtype=np.float64),
                config.period,
                config.adx_threshold,
            )
            last_index = len(close_window) - 1
            if last_index >= 0 and int(directions[last_index]) != int(SIGNAL_NONE):
                if int(directions[last_index]) != int(last_direction):
                    emitted.append(
                        _build_signal(
                            snapshot=snapshot,
                            plugin_id=self.plugin_id,
                            plugin_version=self.plugin_version,
                            instrument_id=config.instrument_id,
                            timeframe=config.timeframe,
                            time_horizon=config.time_horizon,
                            timestamp=int(bar.timestamp),
                            direction=int(directions[last_index]),
                            strength=float(strengths[last_index]),
                            payload=_with_target_position(
                                {
                                    "bar_timestamp": int(bar.timestamp),
                                    "adx": float(adx_values[last_index]),
                                    "plus_di": float(plus_di_values[last_index]),
                                    "minus_di": float(minus_di_values[last_index]),
                                    "adx_threshold": float(config.adx_threshold),
                                    "blocked_instruments": blocked,
                                },
                                target_position=1 if int(directions[last_index]) == int(SIGNAL_BUY) else -1,
                            ),
                            config_hash=config_hash,
                        )
                    )
                    last_direction = int(directions[last_index])
            elif last_index >= 0:
                last_direction = int(SIGNAL_NONE)
            cursor = bar.timestamp

        return StepSignalResult(
            state=SignalState(
                plugin_id=self.plugin_id,
                values={
                    "cursor": cursor,
                    "last_direction": last_direction,
                    "close_window": close_window,
                    "high_window": high_window,
                    "low_window": low_window,
                },
            ),
            signals=emitted,
        )


class AtrBreakoutPlugin:
    plugin_id = "atr_breakout"
    plugin_version = "numba/v1"

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def run(
        self,
        snapshot: DataSnapshot,
        config: AtrBreakoutConfig,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return _build_batch(self.plugin_id, snapshot, [])

        timestamps = np.asarray([int(bar.timestamp) for bar in bars], dtype=np.int64)
        closes = np.asarray([float(bar.close) for bar in bars], dtype=np.float64)
        highs = np.asarray([float(bar.high) for bar in bars], dtype=np.float64)
        lows = np.asarray([float(bar.low) for bar in bars], dtype=np.float64)
        directions, strengths, ma_values, atr_values, upper_channel, lower_channel = atr_breakout_signal_rows(
            closes,
            highs,
            lows,
            config.ma_period,
            config.atr_period,
            config.atr_multiplier,
        )
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.ma_period,
            config.atr_period,
            config.atr_multiplier,
        )
        signals = []
        previous_direction = SIGNAL_NONE
        for index, direction in enumerate(directions):
            if int(direction) == int(SIGNAL_NONE):
                previous_direction = SIGNAL_NONE
                continue
            if int(direction) == int(previous_direction):
                previous_direction = direction
                continue
            signals.append(
                _build_signal(
                    snapshot=snapshot,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    instrument_id=config.instrument_id,
                    timeframe=config.timeframe,
                    time_horizon=config.time_horizon,
                    timestamp=int(timestamps[index]),
                    direction=int(direction),
                    strength=float(strengths[index]),
                    payload=_with_target_position(
                        {
                            "bar_timestamp": int(timestamps[index]),
                            "ma": float(ma_values[index]),
                            "atr": float(atr_values[index]),
                            "upper_channel": float(upper_channel[index]),
                            "lower_channel": float(lower_channel[index]),
                            "blocked_instruments": blocked,
                        },
                        target_position=1 if int(direction) == int(SIGNAL_BUY) else -1,
                    ),
                    config_hash=config_hash,
                )
            )
            previous_direction = direction
        return _build_batch(self.plugin_id, snapshot, signals)

    def initialize_state(self) -> SignalState:
        return SignalState(
            plugin_id=self.plugin_id,
            values={
                "cursor": None,
                "last_direction": int(SIGNAL_NONE),
                "close_window": [],
                "high_window": [],
                "low_window": [],
            },
        )

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: AtrBreakoutConfig,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return StepSignalResult(state=state, signals=[])

        cursor = state.values.get("cursor")
        last_direction = int(state.values.get("last_direction", int(SIGNAL_NONE)))
        close_window = [float(value) for value in state.values.get("close_window", [])]
        high_window = [float(value) for value in state.values.get("high_window", [])]
        low_window = [float(value) for value in state.values.get("low_window", [])]
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.ma_period,
            config.atr_period,
            config.atr_multiplier,
        )
        emitted: List[SignalEnvelope] = []

        for bar in bars:
            if cursor is not None and bar.timestamp <= cursor:
                continue
            close_window.append(float(bar.close))
            high_window.append(float(bar.high))
            low_window.append(float(bar.low))
            directions, strengths, ma_values, atr_values, upper_channel, lower_channel = atr_breakout_signal_rows(
                np.asarray(close_window, dtype=np.float64),
                np.asarray(high_window, dtype=np.float64),
                np.asarray(low_window, dtype=np.float64),
                config.ma_period,
                config.atr_period,
                config.atr_multiplier,
            )
            last_index = len(close_window) - 1
            if last_index >= 0 and int(directions[last_index]) != int(SIGNAL_NONE):
                if int(directions[last_index]) != int(last_direction):
                    emitted.append(
                        _build_signal(
                            snapshot=snapshot,
                            plugin_id=self.plugin_id,
                            plugin_version=self.plugin_version,
                            instrument_id=config.instrument_id,
                            timeframe=config.timeframe,
                            time_horizon=config.time_horizon,
                            timestamp=int(bar.timestamp),
                            direction=int(directions[last_index]),
                            strength=float(strengths[last_index]),
                            payload=_with_target_position(
                                {
                                    "bar_timestamp": int(bar.timestamp),
                                    "ma": float(ma_values[last_index]),
                                    "atr": float(atr_values[last_index]),
                                    "upper_channel": float(upper_channel[last_index]),
                                    "lower_channel": float(lower_channel[last_index]),
                                    "blocked_instruments": blocked,
                                },
                                target_position=1 if int(directions[last_index]) == int(SIGNAL_BUY) else -1,
                            ),
                            config_hash=config_hash,
                        )
                    )
                    last_direction = int(directions[last_index])
            elif last_index >= 0:
                last_direction = int(SIGNAL_NONE)
            cursor = bar.timestamp

        return StepSignalResult(
            state=SignalState(
                plugin_id=self.plugin_id,
                values={
                    "cursor": cursor,
                    "last_direction": last_direction,
                    "close_window": close_window,
                    "high_window": high_window,
                    "low_window": low_window,
                },
            ),
            signals=emitted,
        )


class BollingerMeanReversionPlugin:
    plugin_id = "bollinger_mean_reversion"
    plugin_version = "numba/v1"

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def run(
        self,
        snapshot: DataSnapshot,
        config: BollingerMeanReversionConfig,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        encoded = encode_close_series(snapshot, config.instrument_id, config.timeframe)
        if encoded.closes.size == 0:
            return _build_batch(self.plugin_id, snapshot, [])
        directions, strengths, mid_values, upper_values, lower_values = bollinger_mean_reversion_signal_rows(
            encoded.closes,
            config.period,
            config.stddev_multiplier,
        )
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.period,
            config.stddev_multiplier,
        )
        signals = []
        previous_direction = SIGNAL_NONE
        for index, direction in enumerate(directions):
            if int(direction) == int(SIGNAL_NONE):
                previous_direction = SIGNAL_NONE
                continue
            if int(direction) == int(previous_direction):
                previous_direction = direction
                continue
            signals.append(
                _build_signal(
                    snapshot=snapshot,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    instrument_id=config.instrument_id,
                    timeframe=config.timeframe,
                    time_horizon=config.time_horizon,
                    timestamp=int(encoded.timestamps[index]),
                    direction=int(direction),
                    strength=float(strengths[index]),
                    payload=_with_target_position(
                        {
                            "bar_timestamp": int(encoded.timestamps[index]),
                            "mid_band": float(mid_values[index]),
                            "upper_band": float(upper_values[index]),
                            "lower_band": float(lower_values[index]),
                            "stddev_multiplier": float(config.stddev_multiplier),
                            "blocked_instruments": blocked,
                        },
                        target_position=1 if int(direction) == int(SIGNAL_BUY) else -1,
                    ),
                    config_hash=config_hash,
                )
            )
            previous_direction = direction
        return _build_batch(self.plugin_id, snapshot, signals)

    def initialize_state(self) -> SignalState:
        return SignalState(
            plugin_id=self.plugin_id,
            values={"cursor": None, "last_direction": int(SIGNAL_NONE), "close_window": []},
        )

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: BollingerMeanReversionConfig,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return StepSignalResult(state=state, signals=[])

        cursor = state.values.get("cursor")
        last_direction = int(state.values.get("last_direction", int(SIGNAL_NONE)))
        close_window = [float(value) for value in state.values.get("close_window", [])]
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.period,
            config.stddev_multiplier,
        )
        emitted: List[SignalEnvelope] = []

        for bar in bars:
            if cursor is not None and bar.timestamp <= cursor:
                continue
            close_window.append(float(bar.close))
            directions, strengths, mid_values, upper_values, lower_values = bollinger_mean_reversion_signal_rows(
                np.asarray(close_window, dtype=np.float64),
                config.period,
                config.stddev_multiplier,
            )
            last_index = len(close_window) - 1
            if last_index >= 0 and int(directions[last_index]) != int(SIGNAL_NONE):
                if int(directions[last_index]) != int(last_direction):
                    emitted.append(
                        _build_signal(
                            snapshot=snapshot,
                            plugin_id=self.plugin_id,
                            plugin_version=self.plugin_version,
                            instrument_id=config.instrument_id,
                            timeframe=config.timeframe,
                            time_horizon=config.time_horizon,
                            timestamp=int(bar.timestamp),
                            direction=int(directions[last_index]),
                            strength=float(strengths[last_index]),
                            payload=_with_target_position(
                                {
                                    "bar_timestamp": int(bar.timestamp),
                                    "mid_band": float(mid_values[last_index]),
                                    "upper_band": float(upper_values[last_index]),
                                    "lower_band": float(lower_values[last_index]),
                                    "stddev_multiplier": float(config.stddev_multiplier),
                                    "blocked_instruments": blocked,
                                },
                                target_position=1 if int(directions[last_index]) == int(SIGNAL_BUY) else -1,
                            ),
                            config_hash=config_hash,
                        )
                    )
                    last_direction = int(directions[last_index])
            elif last_index >= 0:
                last_direction = int(SIGNAL_NONE)
            cursor = bar.timestamp

        return StepSignalResult(
            state=SignalState(
                plugin_id=self.plugin_id,
                values={
                    "cursor": cursor,
                    "last_direction": last_direction,
                    "close_window": close_window,
                },
            ),
            signals=emitted,
        )


class MacdTrendFollowPlugin:
    plugin_id = "macd_trend_follow"
    plugin_version = "numba/v1"

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def run(
        self,
        snapshot: DataSnapshot,
        config: MacdTrendFollowConfig,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        encoded = encode_close_series(snapshot, config.instrument_id, config.timeframe)
        if encoded.closes.size == 0:
            return _build_batch(self.plugin_id, snapshot, [])
        directions, strengths, macd_line, signal_line, histogram = macd_trend_follow_signal_rows(
            encoded.closes,
            config.fast_period,
            config.slow_period,
            config.signal_period,
            config.histogram_threshold,
        )
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.fast_period,
            config.slow_period,
            config.signal_period,
            config.histogram_threshold,
        )
        signals = []
        previous_direction = SIGNAL_NONE
        for index, direction in enumerate(directions):
            if int(direction) == int(SIGNAL_NONE):
                previous_direction = SIGNAL_NONE
                continue
            if int(direction) == int(previous_direction):
                previous_direction = direction
                continue
            signals.append(
                _build_signal(
                    snapshot=snapshot,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    instrument_id=config.instrument_id,
                    timeframe=config.timeframe,
                    time_horizon=config.time_horizon,
                    timestamp=int(encoded.timestamps[index]),
                    direction=int(direction),
                    strength=float(strengths[index]),
                    payload=_with_target_position(
                        {
                            "bar_timestamp": int(encoded.timestamps[index]),
                            "macd_line": float(macd_line[index]),
                            "signal_line": float(signal_line[index]),
                            "histogram": float(histogram[index]),
                            "histogram_threshold": float(config.histogram_threshold),
                            "blocked_instruments": blocked,
                        },
                        target_position=1 if int(direction) == int(SIGNAL_BUY) else -1,
                    ),
                    config_hash=config_hash,
                )
            )
            previous_direction = direction
        return _build_batch(self.plugin_id, snapshot, signals)

    def initialize_state(self) -> SignalState:
        return SignalState(
            plugin_id=self.plugin_id,
            values={"cursor": None, "last_direction": int(SIGNAL_NONE), "close_window": []},
        )

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: MacdTrendFollowConfig,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        bars = series_for(snapshot, config.instrument_id, config.timeframe)
        if not bars:
            return StepSignalResult(state=state, signals=[])

        cursor = state.values.get("cursor")
        last_direction = int(state.values.get("last_direction", int(SIGNAL_NONE)))
        close_window = [float(value) for value in state.values.get("close_window", [])]
        blocked = _blocked_instruments(context)
        config_hash = _config_hash(
            config.instrument_id,
            config.timeframe,
            config.fast_period,
            config.slow_period,
            config.signal_period,
            config.histogram_threshold,
        )
        emitted: List[SignalEnvelope] = []

        for bar in bars:
            if cursor is not None and bar.timestamp <= cursor:
                continue
            close_window.append(float(bar.close))
            directions, strengths, macd_line, signal_line, histogram = macd_trend_follow_signal_rows(
                np.asarray(close_window, dtype=np.float64),
                config.fast_period,
                config.slow_period,
                config.signal_period,
                config.histogram_threshold,
            )
            last_index = len(close_window) - 1
            if last_index >= 0 and int(directions[last_index]) != int(SIGNAL_NONE):
                if int(directions[last_index]) != int(last_direction):
                    emitted.append(
                        _build_signal(
                            snapshot=snapshot,
                            plugin_id=self.plugin_id,
                            plugin_version=self.plugin_version,
                            instrument_id=config.instrument_id,
                            timeframe=config.timeframe,
                            time_horizon=config.time_horizon,
                            timestamp=int(bar.timestamp),
                            direction=int(directions[last_index]),
                            strength=float(strengths[last_index]),
                            payload=_with_target_position(
                                {
                                    "bar_timestamp": int(bar.timestamp),
                                    "macd_line": float(macd_line[last_index]),
                                    "signal_line": float(signal_line[last_index]),
                                    "histogram": float(histogram[last_index]),
                                    "histogram_threshold": float(config.histogram_threshold),
                                    "blocked_instruments": blocked,
                                },
                                target_position=1 if int(directions[last_index]) == int(SIGNAL_BUY) else -1,
                            ),
                            config_hash=config_hash,
                        )
                    )
                    last_direction = int(directions[last_index])
            elif last_index >= 0:
                last_direction = int(SIGNAL_NONE)
            cursor = bar.timestamp

        return StepSignalResult(
            state=SignalState(
                plugin_id=self.plugin_id,
                values={
                    "cursor": cursor,
                    "last_direction": last_direction,
                    "close_window": close_window,
                },
            ),
            signals=emitted,
        )


#: Canonical ids of the built-in market SignalPlugins registered by
#: :func:`register_builtin_plugins`. Callers that used to probe
#: ``NumbaBacktestRunner.registered_kernels()`` (now removed) use this instead.
BUILTIN_PLUGIN_IDS = frozenset({
    "moving_average_cross",
    "price_moving_average",
    "rsi_reversion",
    "donchian_breakout",
    "oi_funding_breakout",
    "liquidation_reversal",
    "adx_trend_strength",
    "atr_breakout",
    "bollinger_mean_reversion",
    "macd_trend_follow",
    "multi_timeframe_ma_spread",
})


def register_builtin_plugins(registry) -> None:
    registry.register(MovingAverageCrossPlugin(), lambda raw: MovingAverageCrossConfig(**raw))
    registry.register(PriceMovingAveragePlugin(), lambda raw: PriceMovingAverageConfig(**raw))
    registry.register(RsiReversionPlugin(), lambda raw: RsiReversionConfig(**raw))
    registry.register(DonchianBreakoutPlugin(), lambda raw: DonchianBreakoutConfig(**raw))
    registry.register(OiFundingBreakoutPlugin(), lambda raw: OiFundingBreakoutConfig(**raw))
    registry.register(LiquidationReversalPlugin(), lambda raw: LiquidationReversalConfig(**raw))
    registry.register(AdxTrendStrengthPlugin(), lambda raw: AdxTrendStrengthConfig(**raw))
    registry.register(AtrBreakoutPlugin(), lambda raw: AtrBreakoutConfig(**raw))
    registry.register(BollingerMeanReversionPlugin(), lambda raw: BollingerMeanReversionConfig(**raw))
    registry.register(MacdTrendFollowPlugin(), lambda raw: MacdTrendFollowConfig(**raw))
    registry.register(MultiTimeframeMaSpreadPlugin(), lambda raw: MultiTimeframeMaSpreadConfig(**raw))
