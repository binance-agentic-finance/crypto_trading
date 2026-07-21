"""score_ema_trend — same math as BTC template (kept per-strategy so
tuning here doesn't affect BTC)."""
from __future__ import annotations


def score_ema_trend(ema: dict, thr: dict) -> dict:
    n = ema.get("aligned_count", 0)
    d = ema.get("direction", "NEUTRAL")
    if n == 3:  return {"name": "ema_trend", "score": thr.get("all_aligned", 3), "reason": f"EMA aligned ({d})"}
    if n == 2:  return {"name": "ema_trend", "score": thr.get("two_aligned", 2), "reason": "EMA 2/3"}
    if n == 1:  return {"name": "ema_trend", "score": thr.get("one_aligned", 1), "reason": "EMA 1/3"}
    return          {"name": "ema_trend", "score": thr.get("none", -2),       "reason": "EMAs misaligned"}
