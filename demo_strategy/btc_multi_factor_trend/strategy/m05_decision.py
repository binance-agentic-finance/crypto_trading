"""Module ⑤ DECISION — turn score + signals into a trade plan.

Reads:
    - signals.ema_trend.direction  (bullish / bearish / neutral)
    - signals.macd.histogram
    - signals.resonance.dominant
    to compute a final `direction` (LONG / SHORT / WATCH).

    - cfg._active_mode.{leverage_range, stop_pct, risk_pct}
    - cfg.trade.account_balance_usdt
    to compute size and stop price.

Output shape matches `atomic_strategy_lib.core.types.TradePlan` (dict form).
"""
from __future__ import annotations

from atomic_strategy_lib.decision.direction import direction_from_multi_factor
from atomic_strategy_lib.decision.sizing import (
    fixed_risk_pct, compute_stop_price, leverage_cap,
)


def decide_one(symbol: str, signals: dict, score: dict, cfg: dict) -> dict:
    mode = cfg.get("_active_mode", {})
    balance = cfg.get("trade", {}).get("account_balance_usdt", 1000)

    # ── Direction — vote among the strongest per-tier direction indicators ──
    votes = _collect_direction_votes(signals)
    direction = direction_from_multi_factor(votes) or "WATCH"

    # ── If the verdict is SKIP or WATCHLIST, don't size an order ──
    if score["verdict"] in ("SKIP", "WATCHLIST"):
        return {
            "direction": "WATCH",
            "reason":    f"verdict={score['verdict']} — no trade",
            "size_usdt": 0.0,
            "leverage":  1,
        }

    # ── Sizing ──
    stop_pct  = mode.get("stop_pct", 5.0) / 100.0
    risk_pct  = mode.get("risk_pct", 3.0) / 100.0
    lev_max   = mode.get("leverage_range", [2, 5])[-1]

    size_usdt = fixed_risk_pct(balance, risk_pct, stop_pct)
    leverage  = leverage_cap(size_usdt / balance if balance else 1, lev_max)

    stop_price = compute_stop_price(
        entry=signals.get("_entry_price") or 0,
        direction=direction,
        stop_pct=stop_pct,
    ) if signals.get("_entry_price") else None

    return {
        "direction":  direction,
        "size_usdt":  round(size_usdt, 2),
        "leverage":   leverage,
        "stop_pct":   mode.get("stop_pct"),
        "stop_price": stop_price,
        "risk_usdt":  round(balance * risk_pct, 2),
        "mode":       mode.get("name", "balanced"),
    }


# ─────────────────────────────────────────────────────────────────────
def _collect_direction_votes(signals: dict) -> list[str]:
    """Bag of signals that each vote LONG / SHORT / NEUTRAL."""
    votes = []

    ema_dir = signals["ema_trend"]["direction"]
    if ema_dir == "BULLISH": votes.append("LONG")
    elif ema_dir == "BEARISH": votes.append("SHORT")

    hist = signals["macd"].get("histogram") or 0
    if hist > 0: votes.append("LONG")
    elif hist < 0: votes.append("SHORT")

    reso = signals["resonance"]["dominant"]
    if reso == "BULLISH": votes.append("LONG")
    elif reso == "BEARISH": votes.append("SHORT")

    funding_signal = signals["derivatives"].get("funding_signal", "NEUTRAL")
    if funding_signal in ("LONG", "SHORT"):
        votes.append(funding_signal)

    return votes
