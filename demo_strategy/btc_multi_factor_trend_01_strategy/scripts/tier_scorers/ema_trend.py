"""score_ema_trend — map EmaBlock output → per-tier score.

UI card: "tier · EMA alignment" — inline sliders for
all_aligned / two_aligned / one_aligned / none scores.
"""
from __future__ import annotations


def score_ema_trend(ema: dict, thr: dict) -> dict:
    n = ema.get("aligned_count", 0)
    direction = ema.get("direction", "NEUTRAL")
    if n == 3:
        return {"name": "ema_trend",
                "score": thr.get("all_aligned", 3),
                "reason": f"EMA aligned ({direction})"}
    if n == 2:
        return {"name": "ema_trend",
                "score": thr.get("two_aligned", 2), "reason": "EMA 2/3"}
    if n == 1:
        return {"name": "ema_trend",
                "score": thr.get("one_aligned", 1), "reason": "EMA 1/3"}
    return {"name": "ema_trend",
            "score": thr.get("none", -2), "reason": "EMAs misaligned"}
