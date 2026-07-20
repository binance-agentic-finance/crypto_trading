"""Formatting helpers for terminal + JSON report output.

None-safe: helpers accept None and return "n/a" instead of raising, which
matters because upstream data (funding rate, ATR, etc.) can be missing.
"""
from __future__ import annotations

from typing import Optional


def fmt_pct(v: Optional[float], digits: int = 2) -> str:
    """Format `v` as a percentage. `v=0.0125` → `'1.25%'`."""
    if v is None:
        return "n/a"
    return f"{v * 100:.{digits}f}%"


def fmt_price(v: Optional[float]) -> str:
    """Adaptive price formatting: 5 digits under $1, 2 above."""
    if v is None:
        return "n/a"
    if abs(v) < 1:
        return f"{v:.5f}"
    return f"{v:,.2f}"


def fmt_scalar(v: Optional[float], digits: int = 2) -> str:
    """Generic number format, None-safe."""
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def banner(title: str, width: int = 100, char: str = "=") -> str:
    """Terminal banner line for section headers."""
    return f"{char * width}\n{title}\n{char * width}"
