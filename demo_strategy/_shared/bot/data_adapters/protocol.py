"""DataAdapter interface — every backend implements this shape.

UI card: "data source" (top of the settings drawer) — lists which
backends are wired up in this deployment.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataAdapter(Protocol):
    """Minimal contract every data backend must satisfy."""

    def fetch_bars(self, symbol: str, timeframe: str,
                   limit: int = 200) -> pd.DataFrame: ...
