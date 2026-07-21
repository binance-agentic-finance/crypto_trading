"""StrategyContext — single-symbol strategy input.

UI card: "template input" (single-symbol) — shows which dataclass a
template reads on every tick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .market_data  import MarketData
from .account_data import AccountData


@dataclass(frozen=True)
class StrategyContext:
    market:     MarketData
    account:    AccountData
    config:     dict[str, Any]        # validated against TEMPLATE_META.config_schema
    bar_index:  int = 0               # index of current bar in market.bars
    close_time: int = 0               # ms epoch of current bar's close
