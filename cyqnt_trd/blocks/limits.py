"""Atomic-style risk-limit checks.

Sourced from ``ai_pro_trading_library.library.risk.limits`` to fill the gap
identified in the atomic→cyqnt_trd audit. Pure-function dict-out check
helpers; complement the stateful :class:`cyqnt_trd.blocks.risk.RiskGuard`.

Each function returns a dict so the policy layer can compose results into
:func:`circuit_breaker_check`.
"""

from __future__ import annotations

from typing import Iterable, List, Optional


def is_funding_window(
    now_ms: int,
    settle_hours_utc: Iterable[int] = (0, 8, 16),
    buffer_min: int = 15,
) -> bool:
    """True if *now_ms* falls within ``buffer_min`` of any settlement hour."""
    if buffer_min < 0:
        raise ValueError(f"buffer_min must be >= 0, got {buffer_min}")
    minute_of_day = (int(now_ms) // 60_000) % (24 * 60)
    settle_minutes = [int(h) * 60 for h in settle_hours_utc]
    return any(
        min(abs(minute_of_day - s), 24 * 60 - abs(minute_of_day - s)) <= int(buffer_min)
        for s in settle_minutes
    )


def liquidation_check(
    entry_price: float,
    mark_price: float,
    direction: str,
    leverage: int,
    alert_pct: float = 80.0,
) -> dict:
    """Liquidation proximity given approximate ``liq_distance = 100 / leverage`` (%)."""
    if leverage <= 0 or entry_price <= 0:
        return {"is_danger": False, "reason": "Invalid parameters"}
    liq_distance_pct = 100.0 / float(leverage)
    if direction.upper() == "LONG":
        liq_price_est = float(entry_price) * (1.0 - liq_distance_pct / 100.0)
        current_distance = (float(mark_price) - liq_price_est) / float(entry_price) * 100.0
    else:
        liq_price_est = float(entry_price) * (1.0 + liq_distance_pct / 100.0)
        current_distance = (liq_price_est - float(mark_price)) / float(entry_price) * 100.0
    used_pct = (
        (liq_distance_pct - current_distance) / liq_distance_pct * 100.0
        if liq_distance_pct > 0
        else 0.0
    )
    return {
        "liquidation_price_est": round(liq_price_est, 8),
        "liq_distance_pct": round(liq_distance_pct, 2),
        "current_distance_pct": round(current_distance, 2),
        "margin_used_pct": round(used_pct, 1),
        "is_danger": used_pct >= float(alert_pct),
    }


def max_positions_check(open_count: int, max_count: int = 3) -> dict:
    """Stateless counterpart of :class:`RiskGuard` open-position cap."""
    can_open = int(open_count) < int(max_count)
    return {
        "can_open": can_open,
        "current": int(open_count),
        "max": int(max_count),
        "reason": "OK" if can_open else f"Max {max_count} positions reached",
    }


def max_exposure_check(
    current_notional: float,
    proposed_notional: float,
    balance: float,
    max_exposure_pct: float = 300.0,
) -> dict:
    """Block if total notional / balance would exceed *max_exposure_pct*."""
    total = float(current_notional) + float(proposed_notional)
    exposure_pct = total / float(balance) * 100.0 if balance > 0 else float("inf")
    can_open = exposure_pct <= float(max_exposure_pct)
    return {
        "can_open": can_open,
        "current_notional": float(current_notional),
        "proposed_notional": float(proposed_notional),
        "total_notional": total,
        "exposure_pct": round(exposure_pct, 1),
        "max_exposure_pct": float(max_exposure_pct),
        "reason": "OK"
        if can_open
        else f"Exposure {exposure_pct:.1f}% exceeds max {max_exposure_pct:.1f}%",
    }


def daily_loss_check(
    total_unrealized_pnl: float,
    balance: float,
    daily_loss_limit_pct: float = 10.0,
    daily_loss_limit_abs: Optional[float] = None,
) -> dict:
    """Stateless counterpart of :class:`RiskGuard` daily-loss halt."""
    if total_unrealized_pnl >= 0:
        return {"breached": False, "loss_pct": 0.0, "loss_abs": 0.0, "reason": "No loss"}
    loss_abs = abs(float(total_unrealized_pnl))
    loss_pct = loss_abs / float(balance) * 100.0 if balance > 0 else 0.0
    breached = False
    reasons: List[str] = []
    if daily_loss_limit_abs is not None and loss_abs >= float(daily_loss_limit_abs):
        breached = True
        reasons.append(f"Abs loss ${loss_abs:.2f} >= limit ${daily_loss_limit_abs:.2f}")
    if loss_pct >= float(daily_loss_limit_pct):
        breached = True
        reasons.append(f"Loss {loss_pct:.1f}% >= limit {daily_loss_limit_pct:.1f}%")
    return {
        "breached": breached,
        "loss_pct": round(loss_pct, 2),
        "loss_abs": round(loss_abs, 2),
        "reason": "; ".join(reasons) if reasons else "Within limits",
    }


def price_deviation_check(
    estimated_price: float,
    live_price: float,
    max_deviation_pct: float = 2.0,
) -> dict:
    """Reject stale entries when live price has drifted too far from the estimate."""
    if estimated_price <= 0:
        return {"can_enter": True, "deviation_pct": 0.0, "reason": "No estimated price"}
    deviation = abs(float(live_price) - float(estimated_price)) / float(estimated_price) * 100.0
    can_enter = deviation <= float(max_deviation_pct)
    return {
        "can_enter": can_enter,
        "deviation_pct": round(deviation, 2),
        "max_deviation_pct": float(max_deviation_pct),
        "estimated_price": float(estimated_price),
        "live_price": float(live_price),
        "reason": "OK"
        if can_enter
        else f"Deviation {deviation:.1f}% exceeds max {max_deviation_pct:.1f}%",
    }


def circuit_breaker_check(checks: List[dict]) -> dict:
    """Aggregate risk checks; halt if any fired.

    Recognizes the standard fire flags emitted by the other checks in this
    module: ``breached``, ``is_danger``, ``can_open=False``, ``can_enter=False``.
    """
    failed: List[str] = []
    for check in checks:
        if check.get("breached", False):
            failed.append(check.get("reason", "Unknown breach"))
        elif check.get("is_danger", False):
            failed.append(check.get("reason", "Danger detected"))
        elif "can_open" in check and not check["can_open"]:
            failed.append(check.get("reason", "Cannot open"))
        elif "can_enter" in check and not check["can_enter"]:
            failed.append(check.get("reason", "Cannot enter"))
    return {
        "halt": len(failed) > 0,
        "failed_count": len(failed),
        "total_checks": len(checks),
        "failed_checks": failed,
    }


__all__ = [
    "circuit_breaker_check",
    "daily_loss_check",
    "is_funding_window",
    "liquidation_check",
    "max_exposure_check",
    "max_positions_check",
    "price_deviation_check",
]
