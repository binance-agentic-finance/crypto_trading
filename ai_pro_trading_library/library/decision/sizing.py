"""Position sizing helpers — merged blocks + atomic surface.

Blocks-style helpers (`fixed_pct_of_equity`, `fixed_amount`,
`atr_position_size`, `risk_based_size`, `kelly_fraction`, `grid_levels`,
`pyramid_add`, `round_step_size`) carry L1 parity vs `cyqnt_trd.blocks.sizing`.

Atomic-style helpers (`fixed_dollar_loss`, `fixed_risk_pct`,
`atr_inverse_size`, `leverage_cap`, `compute_stop_price`,
`adaptive_stop_pct`) carry L2 parity vs
`atomic_strategy_lib.decision.sizing` (return-shape kept; rounding
preserved).

`kelly_size` is the canonical half-Kelly-with-correlation-discount entry
point first added in Phase 4.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Blocks-style sizing (L1 parity vs cyqnt_trd.blocks.sizing)
# ---------------------------------------------------------------------------


def fixed_pct_of_equity(equity: float, pct: float) -> float:
    if pct < 0:
        raise ValueError(f"pct must be >= 0, got {pct}")
    return float(equity) * float(pct)


def fixed_amount(amount_usd: float) -> float:
    if amount_usd < 0:
        raise ValueError(f"amount_usd must be >= 0, got {amount_usd}")
    return float(amount_usd)


def atr_position_size(
    equity: float,
    atr_value: float,
    mark_price: float,
    risk_pct: float = 0.01,
    stop_distance_atr_mult: float = 2.0,
) -> float:
    """Notional sized so dollar-risk-at-stop == risk_pct * equity."""
    if risk_pct < 0:
        raise ValueError(f"risk_pct must be >= 0, got {risk_pct}")
    if stop_distance_atr_mult < 0:
        raise ValueError(f"stop_distance_atr_mult must be >= 0, got {stop_distance_atr_mult}")
    if mark_price <= 0:
        raise ValueError(f"mark_price must be > 0, got {mark_price}")
    if atr_value <= 0:
        return 0.0
    risk_dollars = float(equity) * float(risk_pct)
    stop_distance = float(atr_value) * float(stop_distance_atr_mult)
    qty = risk_dollars / stop_distance
    return qty * float(mark_price)


def risk_based_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.01,
) -> float:
    """Notional where loss-at-stop == risk_pct * equity."""
    if risk_pct < 0:
        raise ValueError(f"risk_pct must be >= 0, got {risk_pct}")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be > 0, got {entry_price}")
    distance = abs(float(entry_price) - float(stop_price))
    if distance == 0:
        return 0.0
    qty = (float(equity) * float(risk_pct)) / distance
    return qty * float(entry_price)


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fractional: float = 0.5,
) -> float:
    """Fractional-Kelly fraction in [0, 1]. L1 parity vs blocks.sizing."""
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError(f"win_rate must be in [0,1], got {win_rate}")
    if avg_loss <= 0:
        raise ValueError(f"avg_loss must be > 0, got {avg_loss}")
    if avg_win <= 0:
        return 0.0
    if not 0.0 < fractional <= 1.0:
        raise ValueError(f"fractional must be in (0,1], got {fractional}")
    b = float(avg_win) / float(avg_loss)
    f = float(win_rate) - (1.0 - float(win_rate)) / b
    return max(0.0, min(1.0, f * float(fractional)))


def grid_levels(
    center_price: float,
    range_pct: float,
    n_grids: int,
    per_grid_notional: float,
) -> list[tuple[float, float]]:
    """Evenly-spaced limit-buy grid around center_price."""
    if range_pct <= 0:
        raise ValueError(f"range_pct must be > 0, got {range_pct}")
    if not isinstance(n_grids, int) or n_grids <= 0:
        raise ValueError(f"n_grids must be a positive integer, got {n_grids!r}")
    low = float(center_price) * (1.0 - float(range_pct))
    high = float(center_price) * (1.0 + float(range_pct))
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
    if add_count < 0:
        raise ValueError(f"add_count must be >= 0, got {add_count}")
    if add_count == 0 or add_count > max_adds:
        return 0.0
    return float(initial_notional) * float(add_ratio)


def round_step_size(qty: float, step_size: float) -> float:
    """Round qty down to the nearest step_size (Binance LOT_SIZE filter)."""
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    n_steps = int(qty / step_size)
    return n_steps * step_size


# ---------------------------------------------------------------------------
# Atomic-style sizing (L2 parity vs atomic_strategy_lib.decision.sizing)
# ---------------------------------------------------------------------------


def fixed_dollar_loss(
    balance: float,
    fixed_loss_usd: float,
    entry_price: float,
    stop_price: float,
) -> dict:
    """Notional sized so loss-at-stop == fixed_loss_usd."""
    distance = abs(float(entry_price) - float(stop_price))
    if distance == 0 or entry_price <= 0:
        return {"notional": 0.0, "qty": 0.0, "reason": "invalid distance"}
    qty = float(fixed_loss_usd) / distance
    notional = qty * float(entry_price)
    return {
        "notional": round(notional, 8),
        "qty": round(qty, 8),
        "fixed_loss_usd": float(fixed_loss_usd),
        "distance": round(distance, 8),
    }


def fixed_risk_pct(
    balance: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
) -> dict:
    """Notional sized so loss-at-stop == risk_pct (percent) * balance."""
    risk_usd = float(balance) * (float(risk_pct) / 100.0)
    distance = abs(float(entry_price) - float(stop_price))
    if distance == 0 or entry_price <= 0:
        return {"notional": 0.0, "qty": 0.0, "reason": "invalid distance"}
    qty = risk_usd / distance
    notional = qty * float(entry_price)
    return {
        "notional": round(notional, 8),
        "qty": round(qty, 8),
        "risk_usd": round(risk_usd, 4),
        "risk_pct": float(risk_pct),
    }


def kelly_size(
    equity: float,
    win_prob: float,
    win_loss_ratio: float,
    *,
    half_kelly: bool = True,
    correlation_discount: float = 0.0,
    max_fraction: float = 0.5,
) -> float:
    """Half-Kelly notional with correlation discount.

    `correlation_discount` ∈ [0, 1] reduces the fraction multiplicatively to
    account for capital already in correlated symbols. Result clamped to
    `max_fraction * equity`. Intentional divergence from blocks/atomic.
    """
    if equity < 0:
        raise ValueError("equity must be >= 0")
    if not 0.0 <= correlation_discount <= 1.0:
        raise ValueError("correlation_discount must be in [0,1]")
    if max_fraction <= 0:
        raise ValueError("max_fraction must be > 0")
    if not 0.0 <= win_prob <= 1.0:
        raise ValueError("win_prob must be in [0,1]")
    if win_loss_ratio <= 0:
        raise ValueError("win_loss_ratio must be > 0")
    raw = (float(win_prob) * float(win_loss_ratio) - (1.0 - float(win_prob))) / float(win_loss_ratio)
    f = max(0.0, raw)
    if half_kelly:
        f *= 0.5
    f *= 1.0 - float(correlation_discount)
    f = min(f, float(max_fraction))
    return float(f * float(equity))


def atr_inverse_size(
    balance: float,
    base_pct: float,
    atr_pct: float,
    target_atr_pct: float = 1.5,
) -> dict:
    """Volatility-inverse sizing: shrink notional when realised ATR rises."""
    if atr_pct <= 0:
        return {"notional": 0.0, "reason": "invalid atr"}
    scale = float(target_atr_pct) / float(atr_pct)
    scale = min(scale, 2.0)  # cap upscale at 2x
    scale = max(scale, 0.25)  # floor at 0.25x
    notional = float(balance) * (float(base_pct) / 100.0) * scale
    return {
        "notional": round(notional, 8),
        "scale": round(scale, 3),
        "atr_pct": float(atr_pct),
        "target_atr_pct": float(target_atr_pct),
    }


def leverage_cap(
    requested_leverage: float,
    max_leverage: float = 10.0,
) -> dict:
    """Clamp requested leverage to max_leverage."""
    if requested_leverage < 0:
        raise ValueError("requested_leverage must be >= 0")
    capped = min(float(requested_leverage), float(max_leverage))
    return {
        "leverage": round(capped, 2),
        "requested": float(requested_leverage),
        "max": float(max_leverage),
        "was_capped": capped < float(requested_leverage),
    }


def compute_stop_price(
    entry_price: float,
    direction: str,
    stop_distance_pct: float,
) -> float:
    """Stop price at stop_distance_pct (percent) from entry."""
    distance = float(stop_distance_pct) / 100.0
    if direction.upper() == "LONG":
        return float(entry_price) * (1.0 - distance)
    return float(entry_price) * (1.0 + distance)


def adaptive_stop_pct(
    atr_pct: float,
    base_stop_pct: float = 2.0,
    atr_floor: float = 0.5,
    atr_ceiling: float = 5.0,
) -> float:
    """Scale stop distance by realised ATR, clamped to [atr_floor, atr_ceiling]."""
    capped_atr = min(max(float(atr_pct), float(atr_floor)), float(atr_ceiling))
    return float(base_stop_pct) * (capped_atr / float(atr_floor))


__all__ = [
    "adaptive_stop_pct",
    "atr_inverse_size",
    "atr_position_size",
    "compute_stop_price",
    "fixed_amount",
    "fixed_dollar_loss",
    "fixed_pct_of_equity",
    "fixed_risk_pct",
    "grid_levels",
    "kelly_fraction",
    "kelly_size",
    "leverage_cap",
    "pyramid_add",
    "risk_based_size",
    "round_step_size",
]
