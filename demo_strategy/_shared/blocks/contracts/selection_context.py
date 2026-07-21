"""SelectionContext — cross-sectional (universe-level) strategy input.

UI card: "template input" (selection type).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SelectionContext:
    universe:       list[str]
    bars_by_symbol: dict[str, pd.DataFrame]
    timeframe:      str
    config:         dict[str, Any]
    close_time:     int = 0
    metadata:       dict[str, Any] = field(default_factory=dict)
