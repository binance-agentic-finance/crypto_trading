"""Vector backtest contract.

Runtime consumes `StrategySpec`, looks up the registered signal builder, and
returns a `BacktestResult` with sharpe / max_drawdown / trade_count populated.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ai_pro_trading_library.library.core.protocols import BacktestResult, StrategySpec
from ai_pro_trading_library.library.features.indicators import bollinger_bands, ema, rsi, sma
from ai_pro_trading_library.library.strategy.registry import StrategyRegistry, default_registry


def _annualization(timeframe: str) -> float:
    bars_per_year = {
        "1m": 365 * 24 * 60,
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "1h": 365 * 24,
        "4h": 365 * 6,
        "8h": 365 * 3,
        "1d": 365,
    }
    return float(bars_per_year.get(timeframe, 365 * 24))


def _sharpe(strategy_returns: pd.Series, annualization_factor: float) -> float | None:
    arr = strategy_returns.dropna().astype(float).values
    if len(arr) < 2:
        return None
    std = float(arr.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float(arr.mean() / std * math.sqrt(annualization_factor))


def run_vector_backtest(
    close: pd.Series,
    long_signal: pd.Series,
    fee_rate: float = 0.0,
    timeframe: str = "1h",
) -> BacktestResult:
    if len(close) != len(long_signal):
        raise ValueError("close and long_signal lengths must match")
    position = long_signal.fillna(False).astype(int).shift(1).fillna(0)
    returns = close.astype(float).pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_returns = position * returns - turnover * float(fee_rate)
    equity = (1.0 + strategy_returns).cumprod()
    entries = int(((position == 1) & (position.shift(1).fillna(0) == 0)).sum())
    drawdown = equity / equity.cummax() - 1.0
    return BacktestResult(
        bars=len(close),
        trades=entries,
        total_return=float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        equity_curve=equity,
        sharpe=_sharpe(strategy_returns, _annualization(timeframe)),
        max_drawdown=float(drawdown.min()) if len(drawdown) else None,
    )


class BacktestRunner:
    """Run a `StrategySpec` against `data` using the registered signal builder.

    Signal builders register at import time via `default_registry`. If the
    spec's strategy_id is not registered, the runner falls back to the
    bootstrap inline builder for the 6 priority cases.
    """

    def __init__(self, registry: StrategyRegistry | None = None, fee_rate: float = 0.0) -> None:
        self.registry = registry if registry is not None else default_registry
        self.fee_rate = float(fee_rate)

    def run(self, spec: StrategySpec, data: pd.DataFrame) -> BacktestResult:
        spec.require_symbols()
        if "close" not in data.columns:
            raise ValueError("backtest data must include a close column")
        try:
            signal = self.registry.build_signal(spec, data)
        except KeyError:
            signal = build_signal_from_spec(spec, data)
        return run_vector_backtest(data["close"], signal, fee_rate=self.fee_rate, timeframe=spec.timeframe)


def build_signal_from_spec(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    """Bootstrap signal builder for the 6 priority cases.

    Used as fallback when no factory is registered. Concrete cases register
    themselves via `cases.migrated.<case>.signal`.
    """
    close = data["close"].astype(float)
    strategy_id = spec.strategy_id
    if strategy_id == "rsi-mean-reversion":
        period = int(spec.parameters.get("rsi_period", 14))
        oversold = float(spec.parameters.get("oversold", 30))
        return rsi(close, period) <= oversold
    if strategy_id == "bb-squeeze-momentum":
        period = int(spec.parameters.get("bb_period", 20))
        lower, _middle, upper = bollinger_bands(
            close, period, float(spec.parameters.get("bb_stddev", 2.0))
        )
        bandwidth = (upper - lower) / close
        squeeze = bandwidth <= bandwidth.rolling(window=period, min_periods=period).quantile(0.35)
        return (squeeze & (close > upper.shift(1))).fillna(False)
    if strategy_id == "btc-multi-factor-trend":
        fast = ema(close, int(spec.parameters.get("fast_ema", 20)))
        slow = ema(close, int(spec.parameters.get("slow_ema", 50)))
        return ((close > fast) & (fast > slow)).fillna(False)
    if strategy_id == "structure-breakout-priceaction":
        lookback = int(spec.parameters.get("breakout_lookback", 20))
        if "high" not in data.columns:
            return pd.Series(False, index=data.index)
        prior_high = data["high"].shift(1).rolling(window=lookback, min_periods=lookback).max()
        return (close > prior_high).fillna(False)
    if strategy_id == "stochrsi-mdi-trend-v1":
        fast = sma(close, 10)
        slow = sma(close, 20)
        return (fast > slow).fillna(False)
    if strategy_id == "funding-rate-scanner":
        return pd.Series(False, index=data.index)
    return pd.Series(True, index=data.index)
