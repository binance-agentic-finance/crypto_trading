"""Built-in standard_bot strategy plugins ported as pure-pandas signal factories.

Per `docs/migration/overlap-policy.md` Class C (intentional-divergence):
standard_bot ships these as numba-kernel-backed `SignalPlugin` classes
(`MovingAverageCrossPlugin`, `RsiReversionPlugin`, `BollingerMeanReversionPlugin`,
…). The new repo keeps the same **plugin_id** strings and the same
**Config dataclass shapes** (`field-set parity vs standard_bot.signal.plugins`)
but implements the signal computation in pure pandas using
`library.features.indicators` and `library.conditions.atomic`.

This trades raw kernel speed for portability: no numba dependency,
identical signal semantics, and the canonical `library.strategy.default_registry`
gets each plugin registered automatically when this module is imported.

Each plugin exposes:
- `<Name>Config` dataclass with `__post_init__` validation (parity vs standard_bot)
- `build_signal_<plugin_id>(spec: StrategySpec, data: pd.DataFrame) -> pd.Series`

The signal factories are registered with `default_registry` under the
plugin_id used by `standard_bot.signal.plugins.register_builtin_plugins`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_pro_trading_library.library.conditions.atomic import (
    breakout_high,
    breakout_low,
    cross_above,
    cross_below,
)
from ai_pro_trading_library.library.core.protocols import StrategySpec
from ai_pro_trading_library.library.features.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
)
from ai_pro_trading_library.library.strategy.registry import default_registry


# ---------------------------------------------------------------------------
# Config dataclasses (field-set parity vs standard_bot.signal.plugins)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Signal factories — pure pandas implementations
# ---------------------------------------------------------------------------


def _close(data: pd.DataFrame) -> pd.Series:
    if "close" not in data.columns:
        raise ValueError("data must include a 'close' column")
    return data["close"].astype(float)


def build_signal_moving_average_cross(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    """Long when fast SMA crosses above slow SMA (golden cross)."""
    p = spec.parameters
    fast_w = int(p.get("fast_window", 5))
    slow_w = int(p.get("slow_window", 20))
    threshold = float(p.get("entry_threshold", 0.0))
    close = _close(data)
    f = sma(close, fast_w)
    s = sma(close, slow_w)
    crossed = cross_above(f, s)
    if threshold > 0:
        spread_pct = (f - s) / s.replace(0.0, np.nan)
        crossed = crossed & (spread_pct.abs() >= threshold)
    return crossed.fillna(False)


def build_signal_price_moving_average(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    """Long when close crosses above its SMA(period)."""
    p = spec.parameters
    period = int(p.get("period", 5))
    threshold = float(p.get("entry_threshold", 0.0))
    close = _close(data)
    ma = sma(close, period)
    crossed = cross_above(close, ma)
    if threshold > 0:
        deviation = (close - ma) / ma.replace(0.0, np.nan)
        crossed = crossed & (deviation.abs() >= threshold)
    return crossed.fillna(False)


def build_signal_rsi_reversion(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    """Long when RSI(period) <= oversold (mean-reversion entry)."""
    p = spec.parameters
    period = int(p.get("period", 14))
    oversold = float(p.get("oversold", 30.0))
    rsi_s = rsi(_close(data), period)
    return (rsi_s <= oversold).fillna(False)


def build_signal_donchian_breakout(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    """Long when close breaks above the prior `lookback_window` high (Donchian upper)."""
    p = spec.parameters
    lookback = int(p.get("lookback_window", 20))
    buffer_bps = float(p.get("breakout_buffer_bps", 0.0))
    if "high" not in data.columns:
        return pd.Series(False, index=data.index)
    base = breakout_high(data, lookback=lookback)
    if buffer_bps > 0:
        prior_high = data["high"].shift(1).rolling(window=lookback, min_periods=lookback).max()
        threshold = prior_high * (1.0 + buffer_bps / 10_000.0)
        base = base & (data["close"] > threshold)
    return base.fillna(False)


def build_signal_adx_trend_strength(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    """Long when ADX(period) >= adx_threshold AND +DI > -DI."""
    p = spec.parameters
    period = int(p.get("period", 14))
    threshold = float(p.get("adx_threshold", 25.0))
    if not {"high", "low", "close"}.issubset(data.columns):
        return pd.Series(False, index=data.index)
    adx_s, plus_di, minus_di = adx(data, period)
    return ((adx_s >= threshold) & (plus_di > minus_di)).fillna(False)


def build_signal_atr_breakout(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    """Long when close > MA(ma_period) + ATR(atr_period) * multiplier."""
    p = spec.parameters
    ma_period = int(p.get("ma_period", 20))
    atr_period = int(p.get("atr_period", 14))
    mult = float(p.get("atr_multiplier", 2.0))
    if not {"high", "low", "close"}.issubset(data.columns):
        return pd.Series(False, index=data.index)
    close = _close(data)
    ma = sma(close, ma_period)
    atr_s = atr(data, atr_period)
    band = ma + mult * atr_s
    return (close > band).fillna(False)


def build_signal_bollinger_mean_reversion(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    """Long when close < lower Bollinger Band (mean-reversion long entry)."""
    p = spec.parameters
    period = int(p.get("period", 20))
    stddev = float(p.get("stddev_multiplier", 2.0))
    close = _close(data)
    lower, _mid, _upper = bollinger_bands(close, period, stddev)
    return (close <= lower).fillna(False)


def build_signal_macd_trend_follow(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    """Long when MACD line crosses above signal AND histogram >= threshold."""
    p = spec.parameters
    fast_p = int(p.get("fast_period", 12))
    slow_p = int(p.get("slow_period", 26))
    sig_p = int(p.get("signal_period", 9))
    hist_threshold = float(p.get("histogram_threshold", 0.0))
    line, sig_line, hist = macd(_close(data), fast_p, slow_p, sig_p)
    crossed = cross_above(line, sig_line)
    if hist_threshold > 0:
        crossed = crossed & (hist.abs() >= hist_threshold)
    return crossed.fillna(False)


def build_signal_oi_funding_breakout(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    """Donchian breakout filtered by OI delta and funding rate cap.

    Requires optional `oi` and `funding_rate` columns on `data`. If absent,
    falls back to plain Donchian breakout (graceful degradation).
    """
    p = spec.parameters
    lookback = int(p.get("lookback_window", 20))
    buffer_bps = float(p.get("breakout_buffer_bps", 0.0))
    oi_threshold_bps = float(p.get("oi_threshold_bps", 0.0))
    max_funding_bps = float(p.get("max_funding_rate_bps", 100.0))
    base = build_signal_donchian_breakout(
        StrategySpec(
            strategy_id="oi_funding_breakout",
            symbols=spec.symbols,
            timeframe=spec.timeframe,
            parameters={
                "lookback_window": lookback,
                "breakout_buffer_bps": buffer_bps,
            },
        ),
        data,
    )
    if "oi" in data.columns and oi_threshold_bps > 0:
        oi_pct = data["oi"].pct_change(periods=lookback).abs() * 10_000.0
        base = base & (oi_pct >= oi_threshold_bps)
    if "funding_rate" in data.columns:
        funding_bps = data["funding_rate"].abs() * 10_000.0
        base = base & (funding_bps <= max_funding_bps)
    return base.fillna(False)


def build_signal_liquidation_reversal(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    """Long when long-side liquidation imbalance exceeds the configured ratio.

    Requires optional `long_liq_usd` and `short_liq_usd` columns. If absent,
    returns all-False.
    """
    p = spec.parameters
    long_threshold = float(p.get("long_liquidation_threshold_usd", 100_000.0))
    short_threshold = float(p.get("short_liquidation_threshold_usd", 100_000.0))
    ratio = float(p.get("liquidation_imbalance_ratio", 0.60))
    if not {"long_liq_usd", "short_liq_usd"}.issubset(data.columns):
        return pd.Series(False, index=data.index)
    L = data["long_liq_usd"].astype(float)
    S = data["short_liq_usd"].astype(float)
    total = L + S
    long_share = (L / total).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fired = (long_share >= ratio) & (L >= long_threshold) & (S < short_threshold)
    return fired.fillna(False)


def build_signal_multi_timeframe_ma_spread(
    spec: StrategySpec, data: pd.DataFrame
) -> pd.Series:
    """Long when (primary_ma - reference_ma) / reference_ma * 10000 >= spread_threshold_bps.

    Single-timeframe simplification: we use one DataFrame; the "primary" and
    "reference" MAs are both computed on `data["close"]` with their own
    periods. True multi-TF resampling lives in the `MarketBundleClient` /
    `klines_multi_tf` layer and feeds the runtime, not this signal.
    """
    p = spec.parameters
    primary_p = int(p.get("primary_ma_period", 20))
    reference_p = int(p.get("reference_ma_period", 20))
    threshold_bps = float(p.get("spread_threshold_bps", 0.0))
    close = _close(data)
    primary_ma = sma(close, primary_p)
    reference_ma = sma(close, reference_p)
    spread_bps = (primary_ma - reference_ma) / reference_ma.replace(0.0, np.nan) * 10_000.0
    return (spread_bps >= threshold_bps).fillna(False)


# ---------------------------------------------------------------------------
# Registration with default_registry
# ---------------------------------------------------------------------------


_BUILTIN_PLUGINS: dict[str, callable] = {
    "moving_average_cross": build_signal_moving_average_cross,
    "price_moving_average": build_signal_price_moving_average,
    "rsi_reversion": build_signal_rsi_reversion,
    "donchian_breakout": build_signal_donchian_breakout,
    "adx_trend_strength": build_signal_adx_trend_strength,
    "atr_breakout": build_signal_atr_breakout,
    "bollinger_mean_reversion": build_signal_bollinger_mean_reversion,
    "macd_trend_follow": build_signal_macd_trend_follow,
    "oi_funding_breakout": build_signal_oi_funding_breakout,
    "liquidation_reversal": build_signal_liquidation_reversal,
    "multi_timeframe_ma_spread": build_signal_multi_timeframe_ma_spread,
}


def register_builtin_plugins() -> tuple[str, ...]:
    """Register all 11 built-in plugin signal factories with `default_registry`.

    Idempotent: skips IDs that are already present (e.g. a case has registered
    a more specific override).
    """
    registered: list[str] = []
    for plugin_id, factory in _BUILTIN_PLUGINS.items():
        if not default_registry.has(plugin_id):
            default_registry.register(plugin_id, factory)
            registered.append(plugin_id)
    return tuple(registered)


# Register on import — same idempotency as cases.migrated does
register_builtin_plugins()


__all__ = [
    "AdxTrendStrengthConfig",
    "AtrBreakoutConfig",
    "BollingerMeanReversionConfig",
    "DonchianBreakoutConfig",
    "LiquidationReversalConfig",
    "MacdTrendFollowConfig",
    "MovingAverageCrossConfig",
    "MultiTimeframeMaSpreadConfig",
    "OiFundingBreakoutConfig",
    "PriceMovingAverageConfig",
    "RsiReversionConfig",
    "build_signal_adx_trend_strength",
    "build_signal_atr_breakout",
    "build_signal_bollinger_mean_reversion",
    "build_signal_donchian_breakout",
    "build_signal_liquidation_reversal",
    "build_signal_macd_trend_follow",
    "build_signal_moving_average_cross",
    "build_signal_multi_timeframe_ma_spread",
    "build_signal_oi_funding_breakout",
    "build_signal_price_moving_average",
    "build_signal_rsi_reversion",
    "register_builtin_plugins",
]
