"""vote_macd_histogram — vote by MACD histogram sign.

UI card: "vote · MACD hist sign".
"""
from __future__ import annotations


def vote_macd_histogram(macd: dict) -> str | None:
    h = macd.get("histogram") or 0.0
    if h > 0: return "long"
    if h < 0: return "short"
    return None
