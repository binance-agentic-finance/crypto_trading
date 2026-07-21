"""verdict_five_tier — SKIP / WATCHLIST / CANDIDATE / STRONG_CANDIDATE
                     plus AVOID (high attention + weak technicals).

UI card: "verdict · Square 5-tier" — where AVOID rule lives.
"""
from __future__ import annotations


def verdict_five_tier(total_dict: dict, tiers: list[dict], gate_ov: dict) -> str:
    total = total_dict["total"]
    att_score  = sum(t["score"] for t in tiers
                     if t["name"].startswith("attention"))
    tech_score = total - att_score

    thr = gate_ov.get("thresholds",
                      {"STRONG_CANDIDATE": 10, "CANDIDATE": 6, "WATCHLIST": 2})

    if att_score >= 3 and tech_score < 0:
        return "AVOID"
    if total >= thr.get("STRONG_CANDIDATE", 10): return "STRONG_CANDIDATE"
    if total >= thr.get("CANDIDATE",         6): return "CANDIDATE"
    if total >= thr.get("WATCHLIST",         2): return "WATCHLIST"
    return "SKIP"
