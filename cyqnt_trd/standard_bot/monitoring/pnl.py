"""
cyqnt_trd.standard_bot.monitoring.pnl
=======================================

PnL computation — unrealized PnL for a single position and portfolio-level
aggregation.

Ported from atomic_strategy_lib.monitoring.pnl (L7-01, L7-02).
Input shapes are identical; output dicts are identical so existing callers
work without changes.

The functions accept plain dicts (or PaperPosition/PaperFill dataclasses via
dict conversion) so they integrate naturally with the daemon's state.json
format and with NumbaLivePaperSession.state_snapshot().
"""

from __future__ import annotations

__all__ = ["unrealized_pnl_compute", "portfolio_pnl_aggregate"]


# ---------------------------------------------------------------------------
# L7-01  Unrealized PnL — single position
# ---------------------------------------------------------------------------

def unrealized_pnl_compute(
    entry_price: float,
    current_price: float,
    direction: str,
    leverage: int = 1,
    notional: float = 0.0,
) -> dict:
    """Compute unrealized PnL for a single position.

    Parameters
    ----------
    entry_price:
        Price at which the position was opened.
    current_price:
        Current mark price.
    direction:
        ``"LONG"`` or ``"SHORT"`` (case-insensitive for robustness).
    leverage:
        Leverage multiplier (1 = spot / unlevered).
    notional:
        Position size in quote currency; used to compute ``pnl_abs``.
        Pass 0 to skip absolute PnL.

    Returns
    -------
    dict with keys:
        ``pnl_pct``           – unleveraged % PnL
        ``pnl_leveraged_pct`` – leveraged % PnL
        ``pnl_abs``           – absolute PnL in quote currency
        ``in_profit``         – bool
    """
    if entry_price <= 0 or current_price <= 0:
        return {
            "pnl_pct": 0.0,
            "pnl_leveraged_pct": 0.0,
            "pnl_abs": 0.0,
            "in_profit": False,
        }

    direction_upper = direction.upper()
    if direction_upper == "LONG":
        pnl_pct = (current_price - entry_price) / entry_price * 100.0
    else:
        pnl_pct = (entry_price - current_price) / entry_price * 100.0

    leverage = leverage or 1  # guard against 0 leverage
    pnl_leveraged_pct = pnl_pct * leverage
    pnl_abs = notional * (pnl_pct / 100.0) if notional > 0 else 0.0

    return {
        "pnl_pct": round(pnl_pct, 2),
        "pnl_leveraged_pct": round(pnl_leveraged_pct, 2),
        "pnl_abs": round(pnl_abs, 2),
        "in_profit": pnl_pct > 0,
    }


# ---------------------------------------------------------------------------
# L7-02  Portfolio PnL aggregation
# ---------------------------------------------------------------------------

def portfolio_pnl_aggregate(positions: list[dict]) -> dict:
    """Aggregate unrealized PnL across a list of position dicts.

    Each position dict must contain:
        ``entry_price``, ``current_price``, ``direction``

    Optional fields (with sane defaults):
        ``leverage`` (int, default 1), ``notional`` (float, default 0),
        ``symbol`` (str, default ``"?"``).

    Returns
    -------
    dict with keys:
        ``total_unrealized``   – sum of abs PnL across all positions
        ``total_notional``     – sum of notional values
        ``positions_count``    – number of positions
        ``details``            – list of per-position PnL dicts
        ``best``               – highest ``pnl_pct`` entry (or None)
        ``worst``              – lowest ``pnl_pct`` entry (or None)
    """
    total_unrealized = 0.0
    total_notional = 0.0
    details: list[dict] = []

    for pos in positions:
        entry = pos.get("entry_price", 0)
        current = pos.get("current_price", 0)
        direction = pos.get("direction", "LONG")
        leverage = pos.get("leverage", 1) or 1
        notional = pos.get("notional", 0)

        pnl = unrealized_pnl_compute(
            entry_price=entry,
            current_price=current,
            direction=direction,
            leverage=leverage,
            notional=notional,
        )

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

    sorted_details = sorted(details, key=lambda d: d.get("pnl_pct", 0.0), reverse=True)

    return {
        "total_unrealized": round(total_unrealized, 2),
        "total_notional": round(total_notional, 2),
        "positions_count": len(details),
        "details": details,
        "best": sorted_details[0] if sorted_details else None,
        "worst": sorted_details[-1] if sorted_details else None,
    }
