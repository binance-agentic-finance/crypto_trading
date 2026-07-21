"""score_attention_frequency — Square locale/section overlap → tier."""
from __future__ import annotations


def score_attention_frequency(att_freq_out: dict, thr: dict) -> dict:
    """`att_freq_out` = output of signals/attention_frequency block."""
    # The block already emits `score`, `tier`, `reason` — we forward.
    return {
        "name":   "attention_frequency",
        "score":  int(att_freq_out.get("score", 0)),
        "reason": att_freq_out.get("reason", ""),
    }
