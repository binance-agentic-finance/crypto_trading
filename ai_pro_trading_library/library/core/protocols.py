"""Library and runtime boundary protocols.

Runtime modules should depend on these dataclasses instead of concrete strategy,
broker, or data-adapter implementations. The goal is a narrow contract:
library code produces specs, intents, plans, and risk assessments; runtime code
returns account, backtest, and state projections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    symbols: tuple[str, ...]
    timeframe: str
    entry_rules: tuple[str, ...] = ()
    exit_rules: tuple[str, ...] = ()
    risk_profile: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    source_case_id: str | None = None

    def require_symbols(self) -> None:
        if not self.symbols:
            raise ValueError("StrategySpec.symbols must not be empty")


@dataclass(frozen=True)
class ExecutionIntent:
    symbol: str
    side: str
    notional: float
    leverage: float = 1.0
    order_type: str = "market"
    reduce_only: bool = False
    client_order_id: str | None = None

    def validate(self) -> None:
        if self.notional <= 0:
            raise ValueError("ExecutionIntent.notional must be > 0")
        if self.leverage <= 0:
            raise ValueError("ExecutionIntent.leverage must be > 0")


@dataclass(frozen=True)
class BacktestResult:
    bars: int
    trades: int
    total_return: float
    equity_curve: pd.Series
    sharpe: float | None = None
    max_drawdown: float | None = None
    artifacts: tuple[Path, ...] = ()

    @property
    def trade_count(self) -> int:
        return self.trades


@dataclass(frozen=True)
class Bar:
    """Canonical OHLCV bar.

    Field set is the union of `cyqnt_trd.blocks.data` DataFrame columns,
    `atomic_strategy_lib.core.types.Candle` fields, and
    `cyqnt_trd.standard_bot.core.contracts.Bar`. Adapters fill what their
    endpoint provides; consumers read what they need.

    Per `docs/architecture/data-layer-design.md`, this is the L1 canonical
    shape that all `library.data.adapters.*` return. Conversion helpers in
    `library.data.conversions` bridge to pandas DataFrame and atomic
    `Candle` shapes.
    """

    instrument_id: str
    timeframe: str
    timestamp: int  # close_time in ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None = None
    trades: int | None = None
    taker_buy_base: float | None = None
    taker_buy_quote: float | None = None
    open_time: int | None = None
    confirmed: bool = True


@dataclass(frozen=True)
class AccountPosition:
    symbol: str
    side: str
    notional: float
    unrealized_pnl: float = 0.0


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    available_balance: float
    positions: tuple[AccountPosition, ...] = ()
    ts: str | None = None


@dataclass(frozen=True)
class Signal:
    feature_name: str
    label: str
    value: float | str | bool
    weight: float = 1.0
    passed: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalEnvelope:
    strategy_id: str
    signals: tuple[Signal, ...]
    ts: str | None = None


@dataclass(frozen=True)
class TradePlan:
    strategy_id: str
    intents: tuple[ExecutionIntent, ...]
    rationale: str = ""
    stop_loss: float | None = None
    take_profit: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskAssessment:
    accepted: bool
    reason: str = ""
    violations: tuple[str, ...] = ()
    max_loss_quote: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketBundle:
    frames: Mapping[str, Mapping[str, pd.DataFrame]]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def frame(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self.frames[symbol][timeframe]
