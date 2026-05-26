"""Decision-layer dataclasses independent from execution adapters.

`TradePlan` lives in `library.core.protocols` (the library/runtime contract).
This module re-exports it so existing decision-layer imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ai_pro_trading_library.library.core.protocols import TradePlan

__all__ = ["Direction", "PositionIntent", "TradePlan"]


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class PositionIntent:
    symbol: str
    direction: Direction
    target_notional: float
    leverage: float = 1.0

    def validate(self) -> None:
        if self.target_notional < 0:
            raise ValueError("target_notional must be >= 0")
        if self.leverage <= 0:
            raise ValueError("leverage must be > 0")
