"""Shared schemas for library, cases, and runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from ai_pro_trading_library.library.core.protocols import (
    AccountSnapshot,
    BacktestResult,
    ExecutionIntent,
    MarketBundle,
    RiskAssessment,
    Signal,
    SignalEnvelope,
    StrategySpec,
    TradePlan,
)


class RuntimeMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    DAEMON = "daemon"


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    kind: str
    source: str | None = None


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    display_name: str
    category: str
    status: str
    source: str
    research_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeRequest:
    mode: RuntimeMode
    strategy: StrategySpec
    data_uri: str | None = None
    output_dir: Path | None = None
    paper_account: str | None = None
    dry_run: bool = True
