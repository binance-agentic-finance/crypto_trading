"""Module ④ SCORING — 6-tier hierarchical scoring + verdict gate.

Each tier reads a signal group + its threshold table from cfg.scoring.tiers,
emits `(tier_name, score, reason)`, and the combined score maps to a
4-level verdict via `cfg.scoring.verdict_thresholds`.

This module never fetches data or computes signals. It only turns
"already computed" signal values into a numeric score + a categorical
verdict — which makes it the sole "config-driven knob" for tuning.
"""
from __future__ import annotations

from atomic_strategy_lib.scoring.combinators import hierarchical_combine
from atomic_strategy_lib.scoring.gates import verdict_with_gate


def score_one(signals: dict, cfg: dict) -> dict:
    """Return {total, tiers[{name, score, reason}], verdict}."""
    tiers_cfg  = cfg.get("scoring", {}).get("tiers", {})
    thresholds = cfg.get("scoring", {}).get("verdict_thresholds", {})

    tiers = [
        _score_ema(signals["ema_trend"],       tiers_cfg.get("ema_trend", {})),
        _score_rsi(signals["rsi"],             tiers_cfg.get("rsi_zone", {})),
        _score_macd(signals["macd"],           tiers_cfg.get("macd_momentum", {})),
        _score_derivatives(signals["derivatives"], tiers_cfg.get("derivatives", {})),
        _score_volatility(signals["volatility"],   tiers_cfg.get("volatility", {})),
        _score_resonance(signals["resonance"],     tiers_cfg.get("resonance", {})),
    ]

    total = hierarchical_combine([t["score"] for t in tiers])
    verdict = verdict_with_gate(total, thresholds) or "SKIP"

    return {"total": total, "tiers": tiers, "verdict": verdict}


# ─────────────────────────────────────────────────────────────────────
# per-tier scorers — each is a pure function of (signal_group_dict, threshold_cfg)
# ─────────────────────────────────────────────────────────────────────
def _tier(name, score, reason):
    return {"name": name, "score": score, "reason": reason}


def _score_ema(sig, thr):
    aligned = sig["aligned_count"]
    if aligned == 3:
        return _tier("ema_trend", thr.get("all_aligned", 3),
                     f"EMA all aligned ({sig['direction']})")
    if aligned == 2:
        return _tier("ema_trend", thr.get("two_aligned", 2), "EMA 2/3 aligned")
    if aligned == 1:
        return _tier("ema_trend", thr.get("one_aligned", 1), "EMA 1/3 aligned")
    return _tier("ema_trend", thr.get("none", -2), "EMAs not aligned")


def _score_rsi(sig, thr):
    z = sig["zone"]
    if z in ("BULLISH_NEUTRAL", "BEARISH_NEUTRAL"):
        return _tier("rsi_zone", thr.get("favorable", 2), f"RSI in {z}")
    if z == "NEUTRAL":
        return _tier("rsi_zone", thr.get("neutral", 1), "RSI neutral")
    if z in ("OVERBOUGHT", "OVERSOLD"):
        return _tier("rsi_zone", thr.get("extreme", -1), f"RSI {z}")
    return _tier("rsi_zone", 0, f"RSI zone unknown ({z})")


def _score_macd(sig, thr):
    hist = sig.get("histogram") or 0.0
    if hist > 0 and sig.get("hist_increasing"):
        return _tier("macd_momentum", thr.get("strong", 2),
                     "MACD histogram +ve and rising")
    if hist > 0:
        return _tier("macd_momentum", thr.get("positive", 1),
                     "MACD histogram +ve")
    if hist < 0 and sig.get("hist_increasing"):
        # bearish weakening / potential reversal
        return _tier("macd_momentum", thr.get("divergent", -2),
                     "MACD histogram -ve but rising")
    return _tier("macd_momentum", thr.get("negative", -1),
                 "MACD histogram -ve")


def _score_derivatives(sig, thr):
    z = sig["funding_zone"]
    if z in ("SQUEEZE", "EXTREME_NEGATIVE"):
        return _tier("derivatives", thr.get("funding_squeeze", 2),
                     f"Funding {z} — short squeeze setup")
    if z in ("BUILDING_LONG", "BUILDING_SHORT"):
        return _tier("derivatives", thr.get("oi_building", 1),
                     f"OI {z}")
    if z in ("CROWDED_LONG", "CROWDED_SHORT"):
        return _tier("derivatives", thr.get("crowded", -2),
                     f"Funding {z} — crowded trade")
    return _tier("derivatives", thr.get("neutral", 0), "Derivatives neutral")


def _score_volatility(sig, thr):
    ratio = sig.get("dual_atr_ratio")
    pct   = sig.get("atr_pct") or 0.0
    if pct > 8.0:
        return _tier("volatility", thr.get("extreme", -1),
                     f"ATR% {pct:.1f} — extreme volatility")
    if sig.get("expanding"):
        return _tier("volatility", thr.get("expanding", 1),
                     f"ATR expanding (ratio {ratio})")
    return _tier("volatility", thr.get("contracting", 0),
                 "ATR contracting / flat")


def _score_resonance(sig, thr):
    if sig.get("aligned"):
        return _tier("resonance", thr.get("all_aligned", 2),
                     f"TFs aligned ({sig['dominant']})")
    if sig.get("dominant") in ("BULLISH", "BEARISH"):
        return _tier("resonance", thr.get("partial", 1),
                     f"Partial resonance ({sig['dominant']})")
    return _tier("resonance", thr.get("conflicting", -1),
                 "TFs conflicting")
