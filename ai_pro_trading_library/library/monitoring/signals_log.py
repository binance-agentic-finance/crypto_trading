"""Signal-tracking helpers — L2 parity vs `atomic_strategy_lib.monitoring.signals`.

`signal_record` writes to `signals_file`; tests rely on a temp dir.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def signal_record(
    symbol: str,
    direction: str,
    verdict: str,
    score: float,
    entry_price: float,
    stop_price: float = 0.0,
    stop_pct: float = 0.0,
    leverage: float = 0.0,
    notional: float = 0.0,
    max_loss: float = 0.0,
    metadata: dict | None = None,
    signals_file: str = "tmp/signals_log.jsonl",
) -> dict:
    """Append a recorded signal to JSONL; return the dict."""
    signal = {
        "symbol": symbol,
        "direction": direction,
        "verdict": verdict,
        "score": score,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "stop_pct": stop_pct,
        "leverage": leverage,
        "notional": notional,
        "max_loss": max_loss,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "outcomes": {},
    }
    if metadata:
        signal["metadata"] = metadata
    path = Path(signals_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(signal, ensure_ascii=False) + "\n")
    return signal


def signal_outcome_check(
    entry_price: float,
    current_price: float,
    direction: str,
    leverage: float = 1.0,
    stop_price: float = 0.0,
) -> dict:
    """Hypothetical PnL for a recorded signal at a given price."""
    if entry_price <= 0 or current_price <= 0:
        return {"pnl_pct": None, "pnl_leveraged_pct": None, "stop_hit": None}
    if direction == "LONG":
        pnl_pct = (current_price - entry_price) / entry_price * 100
        stop_hit = current_price <= stop_price if stop_price > 0 else None
    else:
        pnl_pct = (entry_price - current_price) / entry_price * 100
        stop_hit = current_price >= stop_price if stop_price > 0 else None
    return {
        "price": current_price,
        "pnl_pct": round(pnl_pct, 2),
        "pnl_leveraged_pct": round(pnl_pct * leverage, 2),
        "stop_hit": stop_hit,
    }


def performance_report(signals: list[dict]) -> dict:
    """Aggregate win-rate / expectancy / breakdowns from a list of signal dicts."""
    if not signals:
        return {"total": 0, "evaluated": 0}
    total = len(signals)
    by_verdict: dict[str, dict] = {}
    by_direction: dict[str, dict] = {}
    wins = 0
    losses = 0
    win_pcts: list[float] = []
    loss_pcts: list[float] = []
    for sig in signals:
        verdict = sig.get("verdict", "?")
        direction = sig.get("direction", "?")
        outcomes = sig.get("outcomes", {})
        best_outcome = None
        for key in ["24h", "4h", "1h", "current"]:
            if key in outcomes and outcomes[key].get("pnl_pct") is not None:
                best_outcome = outcomes[key]
                break
        if best_outcome is None:
            continue
        pnl = best_outcome["pnl_pct"]
        if verdict not in by_verdict:
            by_verdict[verdict] = {"count": 0, "wins": 0, "pnls": []}
        by_verdict[verdict]["count"] += 1
        by_verdict[verdict]["pnls"].append(pnl)
        if direction not in by_direction:
            by_direction[direction] = {"count": 0, "wins": 0, "pnls": []}
        by_direction[direction]["count"] += 1
        by_direction[direction]["pnls"].append(pnl)
        if pnl > 0:
            wins += 1
            win_pcts.append(pnl)
            by_verdict[verdict]["wins"] += 1
            by_direction[direction]["wins"] += 1
        else:
            losses += 1
            loss_pcts.append(pnl)
    evaluated = wins + losses
    win_rate = round(wins / evaluated * 100, 1) if evaluated > 0 else 0
    avg_win = round(sum(win_pcts) / len(win_pcts), 2) if win_pcts else 0
    avg_loss = round(sum(loss_pcts) / len(loss_pcts), 2) if loss_pcts else 0
    expectancy = (
        round((win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss), 2)
        if evaluated > 0
        else 0
    )
    return {
        "total": total,
        "evaluated": evaluated,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "by_verdict": {
            k: {
                "count": v["count"],
                "wins": v["wins"],
                "win_rate": round(v["wins"] / v["count"] * 100, 1) if v["count"] > 0 else 0,
            }
            for k, v in by_verdict.items()
        },
        "by_direction": {
            k: {
                "count": v["count"],
                "wins": v["wins"],
                "win_rate": round(v["wins"] / v["count"] * 100, 1) if v["count"] > 0 else 0,
            }
            for k, v in by_direction.items()
        },
    }


_STOP_TYPES = {
    "STOP_MARKET",
    "STOP_LOSS_LIMIT",
    "TAKE_PROFIT_LIMIT",
    "TAKE_PROFIT_MARKET",
    "TRAILING_STOP_MARKET",
}


def stale_stop_detect(open_orders: list[dict]) -> dict:
    """Pick stop-type orders out of an open-orders list."""
    stops = [order for order in open_orders if order.get("type", "") in _STOP_TYPES]
    return {
        "has_stop": len(stops) > 0,
        "stop_count": len(stops),
        "stop_orders": [
            {
                "orderId": order.get("orderId"),
                "type": order.get("type"),
                "stopPrice": order.get("stopPrice"),
                "side": order.get("side"),
            }
            for order in stops
        ],
    }


__all__ = [
    "performance_report",
    "signal_outcome_check",
    "signal_record",
    "stale_stop_detect",
]
