"""score_attention_deep — hashtag deep-fetch mention count → tier."""
from __future__ import annotations


def score_attention_deep(att_deep_out: dict, thr: dict) -> dict:
    return {
        "name":   "attention_deep",
        "score":  int(att_deep_out.get("score", 0)),
        "reason": att_deep_out.get("reason", ""),
    }
