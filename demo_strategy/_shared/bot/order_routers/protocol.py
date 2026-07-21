"""OrderRouter interface — every execution backend implements this shape.

UI card: "execution · protocol" — read-only, shows the contract.
"""
from __future__ import annotations

from typing import Any, Protocol


class OrderRouter(Protocol):
    """All backends support submit_single (single-symbol trade) and
    submit_portfolio (basket target)."""

    def submit_single(self, *, symbol: str, side: str, qty: float,
                      entry: float, stop: float, mode: str,
                      metadata: dict[str, Any]) -> dict: ...

    def submit_portfolio(self, *, weights: dict[str, float],
                         notional_usdt: float, mode: str,
                         metadata: dict[str, Any]) -> dict: ...
