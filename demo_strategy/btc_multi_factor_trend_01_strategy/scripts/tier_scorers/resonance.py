"""score_resonance — map ResonanceBlock output → per-tier score.

UI card: "tier · Multi-TF resonance".
"""
from __future__ import annotations


def score_resonance(reso: dict, thr: dict) -> dict:
    if reso.get("aligned"):
        return {"name": "resonance",
                "score": thr.get("all_aligned", 2),
                "reason": f"TFs aligned ({reso['dominant']})"}
    if reso.get("dominant") in ("BULLISH", "BEARISH"):
        return {"name": "resonance",
                "score": thr.get("partial", 1),
                "reason": f"Partial ({reso['dominant']})"}
    return {"name": "resonance",
            "score": thr.get("conflicting", -1), "reason": "TFs conflict"}
