"""Atomic-style stop-loss helpers (dict-out).

Sourced from ``ai_pro_trading_library.library.risk.stop_loss`` to fill the gap
identified in the atomic→cyqnt_trd audit. These return dicts (not pandas
Series) — point-in-time computations suitable for one-off use during
order planning or rescan loops.

For stateful pandas-Series exit rules used inside backtest loops, see
:mod:`cyqnt_trd.blocks.exit` (``FixedStopLoss`` / ``AtrTrailingStop`` etc.).
"""

from __future__ import annotations

from typing import Sequence


def atr_dynamic_stop(
    entry_price: float,
    direction: str,
    atr_value: float,
    multiplier: float = 2.0,
) -> dict:
    """Stop at ``multiplier * atr_value`` from entry."""
    distance = float(atr_value) * float(multiplier)
    if direction.upper() == "LONG":
        stop_price = float(entry_price) - distance
    else:
        stop_price = float(entry_price) + distance
    distance_pct = distance / float(entry_price) * 100 if entry_price > 0 else 0.0
    return {
        "stop_price": round(stop_price, 8),
        "distance": round(distance, 8),
        "distance_pct": round(distance_pct, 2),
        "atr_value": float(atr_value),
        "multiplier": float(multiplier),
    }


def fixed_pct_stop(
    entry_price: float,
    direction: str,
    stop_pct: float = 5.0,
) -> dict:
    """Stop at ``stop_pct`` (percent) from entry."""
    distance = float(stop_pct) / 100.0
    if direction.upper() == "LONG":
        stop_price = float(entry_price) * (1.0 - distance)
    else:
        stop_price = float(entry_price) * (1.0 + distance)
    return {
        "stop_price": round(stop_price, 8),
        "stop_pct": float(stop_pct),
    }


def trailing_stop(
    current_price: float,
    direction: str,
    trailing_pct: float,
    existing_stop: float = 0.0,
) -> dict:
    """Trailing stop — only ratchets in favourable direction."""
    if direction.upper() == "LONG":
        new_stop = float(current_price) * (1.0 - float(trailing_pct) / 100.0)
        final_stop = max(new_stop, float(existing_stop)) if existing_stop > 0 else new_stop
    else:
        new_stop = float(current_price) * (1.0 + float(trailing_pct) / 100.0)
        final_stop = min(new_stop, float(existing_stop)) if existing_stop > 0 else new_stop
    return {
        "stop_price": round(final_stop, 8),
        "computed_stop": round(new_stop, 8),
        "previous_stop": float(existing_stop),
        "moved": final_stop != float(existing_stop),
        "trailing_pct": float(trailing_pct),
    }


def nbar_trailing_tp(
    candles: Sequence,
    direction: str,
    n_bars: int = 3,
    existing_stop: float = 0.0,
) -> dict:
    """Trail stop to lowest-low (long) / highest-high (short) of last *n_bars*.

    *candles* is a sequence with ``.low`` and ``.high`` attributes (atomic
    :class:`Candle` or any duck-typed equivalent).
    """
    if len(candles) < n_bars:
        return {"stop_price": float(existing_stop), "moved": False}
    recent = list(candles)[-int(n_bars):]
    if direction.upper() == "LONG":
        new_stop = min(c.low for c in recent)
        final_stop = max(new_stop, float(existing_stop)) if existing_stop > 0 else new_stop
    else:
        new_stop = max(c.high for c in recent)
        final_stop = min(new_stop, float(existing_stop)) if existing_stop > 0 else new_stop
    return {
        "stop_price": round(final_stop, 8),
        "n_bar_level": round(new_stop, 8),
        "moved": final_stop != float(existing_stop),
        "n_bars": int(n_bars),
    }


__all__ = [
    "atr_dynamic_stop",
    "fixed_pct_stop",
    "nbar_trailing_tp",
    "trailing_stop",
]
