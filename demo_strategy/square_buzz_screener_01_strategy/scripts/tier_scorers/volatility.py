from __future__ import annotations


def score_volatility(atr: dict, thr: dict) -> dict:
    pct = atr.get("atr_pct") or 0
    if pct > 8:
        return {"name": "volatility", "score": thr.get("extreme", -1), "reason": f"ATR% {pct:.1f}"}
    if atr.get("expanding"):
        return {"name": "volatility", "score": thr.get("expanding", 1), "reason": "ATR expanding"}
    return {"name": "volatility", "score": thr.get("contracting", 0), "reason": "ATR flat"}
