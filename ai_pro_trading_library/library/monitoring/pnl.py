"""PnL helpers — L2 parity vs `atomic_strategy_lib.monitoring.pnl`."""

from __future__ import annotations


def unrealized_pnl_compute(
    entry_price: float,
    current_price: float,
    direction: str,
    leverage: int = 1,
    notional: float = 0.0,
) -> dict:
    """Unrealized PnL for a single position. Returns dict with rounded values."""
    if entry_price <= 0 or current_price <= 0:
        return {"pnl_pct": 0, "pnl_leveraged_pct": 0, "pnl_abs": 0, "in_profit": False}
    if direction == "LONG":
        pnl_pct = (current_price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - current_price) / entry_price * 100
    pnl_leveraged_pct = pnl_pct * leverage
    pnl_abs = notional * (pnl_pct / 100) if notional > 0 else 0
    return {
        "pnl_pct": round(pnl_pct, 2),
        "pnl_leveraged_pct": round(pnl_leveraged_pct, 2),
        "pnl_abs": round(pnl_abs, 2),
        "in_profit": pnl_pct > 0,
    }


def portfolio_pnl_aggregate(positions: list[dict]) -> dict:
    """Aggregate PnL across position dicts; returns totals + best/worst."""
    total_unrealized = 0.0
    total_notional = 0.0
    details: list[dict] = []
    for pos in positions:
        entry = pos.get("entry_price", 0)
        current = pos.get("current_price", 0)
        direction = pos.get("direction", "LONG")
        leverage = pos.get("leverage", 1) or 1
        notional = pos.get("notional", 0)
        pnl = unrealized_pnl_compute(entry, current, direction, leverage, notional)
        total_unrealized += pnl["pnl_abs"]
        total_notional += notional
        details.append(
            {
                "symbol": pos.get("symbol", "?"),
                "direction": direction,
                "entry": entry,
                "current": current,
                **pnl,
            }
        )
    sorted_details = sorted(details, key=lambda d: d.get("pnl_pct", 0), reverse=True)
    return {
        "total_unrealized": round(total_unrealized, 2),
        "total_notional": round(total_notional, 2),
        "positions_count": len(details),
        "details": details,
        "best": sorted_details[0] if sorted_details else None,
        "worst": sorted_details[-1] if sorted_details else None,
    }


__all__ = ["portfolio_pnl_aggregate", "unrealized_pnl_compute"]
