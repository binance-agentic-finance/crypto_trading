"""score_volume — VolumeSurge block output → tier score."""
from __future__ import annotations


def score_volume(vol: dict, thr: dict) -> dict:
    if vol.get("surge"):
        return {"name": "volume",
                "score":  thr.get("surge", 2),
                "reason": f"Vol surge ×{vol.get('ratio'):.1f}"}
    r = vol.get("ratio") or 0.0
    if r > 1.5:
        return {"name": "volume",
                "score": thr.get("elevated", 1),
                "reason": f"Vol ×{r:.1f}"}
    return {"name": "volume",
            "score": thr.get("flat", 0), "reason": "Vol flat"}
