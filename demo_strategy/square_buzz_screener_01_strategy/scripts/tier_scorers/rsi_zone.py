from __future__ import annotations


def score_rsi_zone(rsi: dict, thr: dict) -> dict:
    z = rsi.get("zone", "UNKNOWN")
    if z in ("BULLISH_NEUTRAL", "BEARISH_NEUTRAL"):
        return {"name": "rsi_zone", "score": thr.get("favorable", 2), "reason": f"RSI {z}"}
    if z == "NEUTRAL":
        return {"name": "rsi_zone", "score": thr.get("neutral", 1), "reason": "RSI neutral"}
    if z in ("OVERBOUGHT", "OVERSOLD"):
        return {"name": "rsi_zone", "score": thr.get("extreme", -1), "reason": f"RSI {z}"}
    return {"name": "rsi_zone", "score": 0, "reason": "RSI unknown"}
