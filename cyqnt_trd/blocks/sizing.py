"""Position-sizing helpers.

All functions return a notional USD size given the chosen sizing rule
and account state. They never apply leverage themselves — the caller
should multiply by ``leverage`` if needed and convert to base-asset
quantity using the latest mark price.

Examples
--------
>>> from cyqnt_trd.blocks import sizing
>>> notional = sizing.fixed_pct_of_equity(equity=10_000, pct=0.15)
>>> qty = notional * leverage / mark_price
"""

from __future__ import annotations

from typing import List, Tuple

from ._utils import non_negative_float, positive_int

__all__ = [
    "fixed_pct_of_equity",
    "fixed_amount",
    "atr_position_size",
    "risk_based_size",
    "kelly_fraction",
    "grid_levels",
    "pyramid_add",
    "round_step_size",
]


def fixed_pct_of_equity(equity: float, pct: float) -> float:
    """Notional = ``equity * pct``."""
    pct = non_negative_float(pct, "pct")
    return float(equity) * pct


def fixed_amount(amount_usd: float) -> float:
    """Notional = constant ``amount_usd``."""
    return non_negative_float(amount_usd, "amount_usd")


def atr_position_size(
    equity: float,
    atr_value: float,
    mark_price: float,
    risk_pct: float = 0.01,
    stop_distance_atr_mult: float = 2.0,
) -> float:
    """Compute position size targeting ``risk_pct * equity`` of dollar risk.

    A common ATR-based rule: ``stop = mark_price - atr * mult``.
    Position size in base-asset is ``risk_dollars / stop_distance``.
    Returns the notional value (size * mark_price).
    """
    risk_pct = non_negative_float(risk_pct, "risk_pct")
    stop_distance_atr_mult = non_negative_float(stop_distance_atr_mult, "stop_distance_atr_mult")
    if mark_price <= 0:
        raise ValueError(f"mark_price must be > 0, got {mark_price}")
    if atr_value <= 0:
        return 0.0
    risk_dollars = float(equity) * risk_pct
    stop_distance = atr_value * stop_distance_atr_mult
    qty = risk_dollars / stop_distance
    return qty * mark_price


def risk_based_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.01,
) -> float:
    """Compute position size where loss-at-stop = ``risk_pct * equity``.

    Returns notional (qty * entry_price).
    """
    risk_pct = non_negative_float(risk_pct, "risk_pct")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be > 0, got {entry_price}")
    distance = abs(float(entry_price) - float(stop_price))
    if distance == 0:
        return 0.0
    qty = (float(equity) * risk_pct) / distance
    return qty * float(entry_price)


def kelly_fraction(
    win_rate: float, avg_win: float, avg_loss: float, fractional: float = 0.5
) -> float:
    """Fractional-Kelly position fraction. Returns a value in ``[0, 1]``.

    *fractional* in ``(0, 1]`` is the user's risk-adjustment multiplier
    (defaults to half-Kelly = 0.5, the standard "safety" choice).
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError(f"win_rate must be within [0, 1], got {win_rate}")
    if avg_loss <= 0:
        raise ValueError(f"avg_loss must be > 0, got {avg_loss}")
    if avg_win <= 0:
        return 0.0
    if not 0.0 < fractional <= 1.0:
        raise ValueError(f"fractional must be within (0, 1], got {fractional}")
    b = avg_win / avg_loss
    f = win_rate - (1.0 - win_rate) / b
    return max(0.0, min(1.0, f * fractional))


def grid_levels(
    center_price: float,
    range_pct: float,
    n_grids: int,
    per_grid_notional: float,
) -> List[Tuple[float, float]]:
    """Build an evenly-spaced limit-buy grid around *center_price*.

    Returns ``[(price, notional), ...]`` from lowest to highest.
    """
    if range_pct <= 0:
        raise ValueError(f"range_pct must be > 0, got {range_pct}")
    n_grids = positive_int(n_grids, "n_grids")
    low = float(center_price) * (1.0 - range_pct)
    high = float(center_price) * (1.0 + range_pct)
    if n_grids == 1:
        return [(float(center_price), float(per_grid_notional))]
    step = (high - low) / (n_grids - 1)
    return [(low + i * step, float(per_grid_notional)) for i in range(n_grids)]


def pyramid_add(
    initial_notional: float,
    add_count: int,
    add_ratio: float = 0.5,
    max_adds: int = 2,
) -> float:
    """Compute the additional notional for the *add_count*-th pyramid add.

    Returns 0 if *add_count* exceeds *max_adds*.
    """
    if add_count < 0:
        raise ValueError(f"add_count must be >= 0, got {add_count}")
    if add_count == 0 or add_count > max_adds:
        return 0.0
    return float(initial_notional) * float(add_ratio)


def round_step_size(qty: float, step_size: float) -> float:
    """Round *qty* down to the nearest *step_size* — matches Binance LOT_SIZE filter."""
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    n_steps = int(qty / step_size)
    return n_steps * step_size
