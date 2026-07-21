"""vote_ema_direction — vote long/short based on EMA alignment.

UI card: "vote · EMA".
"""
from __future__ import annotations


def vote_ema_direction(ema: dict) -> str | None:
    d = ema.get("direction")
    if d == "BULLISH": return "long"
    if d == "BEARISH": return "short"
    return None
