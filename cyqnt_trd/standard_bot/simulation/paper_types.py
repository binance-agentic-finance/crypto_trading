"""
Numba-free shared value types for the paper/live execution layer.

These were previously defined in the (now removed) ``live_paper_session`` module
alongside ``NumbaLivePaperSession``. They carry no numba dependency and are used
by :class:`~cyqnt_trd.standard_bot.simulation.python_live_paper_session.PythonLivePaperSession`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

#: Stable namespace for deterministic fill/signal UUIDs across restarts.
SESSION_NAMESPACE = uuid.UUID("a7e8d1c4-5b3f-4e2a-9f1d-6c0e8b7a2d54")

# Target encoding shared with the signal kernels (kept numba-free here so the
# live session never has to import the signal-kernel module).
TARGET_SHORT = np.int8(-1)
TARGET_FLAT = np.int8(0)
TARGET_LONG = np.int8(1)
TARGET_KEEP = np.int8(2)


@dataclass
class PaperFill:
    """One simulated fill."""

    fill_id: str
    timestamp_ms: int
    side: str  # "buy" or "sell"
    price: float
    quantity: float
    fee: float
    signal_bar_timestamp_ms: int
    action: str  # "open_long", "close_long", "open_short", "close_short", ...


@dataclass
class PaperPosition:
    """Current open position."""

    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    opened_at_ms: int


@dataclass
class PendingOrder:
    """A trade pending execution at the next bar's open."""

    target_position: int  # TARGET_LONG=1, TARGET_SHORT=-1, 0=flat
    signal_bar_index: int
    signal_bar_timestamp_ms: int
    signal_strength: float


__all__ = [
    "SESSION_NAMESPACE",
    "TARGET_SHORT",
    "TARGET_FLAT",
    "TARGET_LONG",
    "TARGET_KEEP",
    "PaperFill",
    "PaperPosition",
    "PendingOrder",
]
