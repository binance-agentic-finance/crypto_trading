from __future__ import annotations


def score_macd_momentum(macd: dict, thr: dict) -> dict:
    hist = macd.get("histogram") or 0.0
    inc  = macd.get("hist_increasing", False)
    if hist > 0 and inc:
        return {"name": "macd_momentum", "score": thr.get("strong", 2), "reason": "MACD +ve rising"}
    if hist > 0:
        return {"name": "macd_momentum", "score": thr.get("positive", 1), "reason": "MACD +ve"}
    if hist < 0 and inc:
        return {"name": "macd_momentum", "score": thr.get("divergent", -2), "reason": "MACD divergent"}
    return {"name": "macd_momentum", "score": thr.get("negative", -1), "reason": "MACD -ve"}
