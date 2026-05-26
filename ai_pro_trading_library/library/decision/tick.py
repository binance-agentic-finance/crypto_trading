"""Lot-size / tick-size quantization helpers — L1 parity vs `atomic.execution.position`.

These are pure scalar utilities consumed by both decision-time stop/TP
placement and runtime broker pre-submit normalisation.
"""

from __future__ import annotations

import math
from decimal import Decimal


def quantize(value: float, step: float) -> float:
    """Floor `value` to the nearest multiple of `step` using Decimal arithmetic.

    Used for Binance LOT_SIZE filter (`stepSize`). Decimal avoids the
    `0.1 + 0.2 != 0.3` IEEE-754 surprises common in `value % step`
    formulations.
    """
    if step <= 0:
        return value
    d = Decimal(str(value))
    q = Decimal(str(step))
    return float((d // q) * q)


def round_to_tick(value: float, tick_size: float) -> float:
    """Floor `value` to the nearest `tick_size` (Binance PRICE_FILTER)."""
    if tick_size <= 0:
        return value
    steps = math.floor(value / tick_size)
    return steps * tick_size


__all__ = ["quantize", "round_to_tick"]
