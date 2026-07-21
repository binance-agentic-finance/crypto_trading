"""vote_resonance_direction — vote from multi-TF resonance dominant.

UI card: "vote · Multi-TF direction".
"""
from __future__ import annotations


def vote_resonance_direction(reso: dict) -> str | None:
    d = reso.get("dominant")
    if d == "BULLISH": return "long"
    if d == "BEARISH": return "short"
    return None
