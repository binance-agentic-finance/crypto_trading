"""Module ⑤ DECISION — verdict-gated direction + light sizing.

For the buzz screener, sizing is intentionally simpler than the BTC
strategy (only one mode) — the main output is verdict + direction bias.
`AVOID` explicitly bumps to `WATCH` on the direction to prevent trades.
"""
from __future__ import annotations

from atomic_strategy_lib.decision.direction import direction_from_multi_factor
from atomic_strategy_lib.decision.sizing import fixed_risk_pct


def decide_one(symbol: str, signals: dict, score: dict, cfg: dict) -> dict:
    mode = cfg.get("_active_mode", {})

    # ── Direction: vote across market signals ──
    votes = []
    if signals["ema_trend"]["direction"] == "BULLISH": votes.append("LONG")
    elif signals["ema_trend"]["direction"] == "BEARISH": votes.append("SHORT")

    hist = signals["macd"].get("histogram") or 0
    if hist > 0: votes.append("LONG")
    elif hist < 0: votes.append("SHORT")

    reso = signals["resonance"]["dominant"]
    if reso == "BULLISH": votes.append("LONG")
    elif reso == "BEARISH": votes.append("SHORT")

    funding_signal = signals["derivatives"].get("funding_signal", "NEUTRAL")
    if funding_signal in ("LONG", "SHORT"):
        votes.append(funding_signal)

    direction = direction_from_multi_factor(votes) or "WATCH"

    # ── Verdict gating: AVOID / SKIP / WATCHLIST → don't trade ──
    verdict = score["verdict"]
    if verdict in ("SKIP", "WATCHLIST", "AVOID"):
        return {
            "direction": "WATCH",
            "reason":    f"verdict={verdict}",
            "size_usdt": 0.0,
        }

    # ── Sizing: fixed risk % ──
    balance = cfg.get("trade", {}).get("account_balance_usdt", 1000)
    stop_pct = mode.get("stop_pct", 5.0) / 100.0
    risk_pct = mode.get("risk_pct", 2.0) / 100.0
    size_usdt = fixed_risk_pct(balance, risk_pct, stop_pct)

    return {
        "direction":  direction,
        "size_usdt":  round(size_usdt, 2),
        "leverage":   mode.get("leverage_range", [1, 3])[-1],
        "stop_pct":   mode.get("stop_pct"),
        "risk_usdt":  round(balance * risk_pct, 2),
    }
