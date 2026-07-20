"""Module ④ SCORING — 9-tier scoring with attention + market fusion.

Adds two attention tiers (T1/T2) on top of the 7 market tiers, and adds
a fifth verdict `AVOID` for cases where attention is high but the
technical picture is deteriorating.
"""
from __future__ import annotations

from atomic_strategy_lib.scoring.combinators import hierarchical_combine


def score_one(signals: dict, cfg: dict) -> dict:
    thr_all = cfg.get("scoring", {}).get("tiers", {})
    thresholds = cfg.get("scoring", {}).get("verdict_thresholds", {})

    tiers = [
        _score_att_freq(signals["attention_frequency"], thr_all.get("attention_frequency", {})),
        _score_att_deep(signals["attention_deep"],      thr_all.get("attention_deep", {})),
        _score_ema(signals["ema_trend"],                thr_all.get("ema_trend", {})),
        _score_rsi(signals["rsi"],                      thr_all.get("rsi_zone", {})),
        _score_macd(signals["macd"],                    thr_all.get("macd_momentum", {})),
        _score_derivatives(signals["derivatives"],       thr_all.get("derivatives", {})),
        _score_volatility(signals["volatility"],         thr_all.get("volatility", {})),
        _score_resonance(signals["resonance"],           thr_all.get("resonance", {})),
        _score_volume(signals["volume"],                 thr_all.get("volume", {})),
    ]

    total = hierarchical_combine([t["score"] for t in tiers])
    verdict = _verdict_5tier(total, tiers, thresholds)
    return {"total": total, "tiers": tiers, "verdict": verdict}


# ─────────────────────────────────────────────────────────────────────
# Verdict logic — attention high + technicals negative = AVOID
# ─────────────────────────────────────────────────────────────────────
def _verdict_5tier(total, tiers, thr):
    strong = thr.get("STRONG_CANDIDATE", 12)
    cand   = thr.get("CANDIDATE",         7)
    watch  = thr.get("WATCHLIST",         3)
    avoid_below = thr.get("AVOID_below",  0)

    att_score = sum(t["score"] for t in tiers if t["name"].startswith("attention"))
    tech_score = total - att_score

    if att_score >= 3 and tech_score < avoid_below:
        return "AVOID"    # hot but structurally weakening
    if total >= strong: return "STRONG_CANDIDATE"
    if total >= cand:   return "CANDIDATE"
    if total >= watch:  return "WATCHLIST"
    return "SKIP"


# ─────────────────────────────────────────────────────────────────────
# per-tier scorers
# ─────────────────────────────────────────────────────────────────────
def _tier(name, score, reason):
    return {"name": name, "score": score, "reason": reason}


def _score_att_freq(sig, thr):
    if sig["en_cn_overlap"]:
        return _tier("attention_frequency", thr.get("en_cn_overlap", 3),
                     "Cross-locale (EN + CN) mention")
    if sig["section_overlap"]:
        return _tier("attention_frequency", thr.get("section_overlap", 2),
                     "Multi-section (Most Searched + Rapid Riser)")
    if sig["locale_count"] > 0:
        return _tier("attention_frequency", thr.get("single_source", 1),
                     f"Mention in {sig['locale_count']} locale(s)")
    return _tier("attention_frequency", thr.get("isolated", 0),
                 "No Square mention")


def _score_att_deep(sig, thr):
    s = sig["sentiment"]
    if s == "STRONG":
        return _tier("attention_deep", thr.get("positive_sentiment", 2),
                     f"Deep mentions × {sig['deep_mentions']} (strong)")
    if s == "MIXED":
        return _tier("attention_deep", thr.get("mixed", 1),
                     f"Deep mentions × {sig['deep_mentions']} (mixed)")
    return _tier("attention_deep", thr.get("negative", -1),
                 "No hashtag deep signal")


def _score_ema(sig, thr):
    a = sig["aligned_count"]
    if a == 3: return _tier("ema_trend", thr.get("all_aligned", 3), f"EMA all aligned ({sig['direction']})")
    if a == 2: return _tier("ema_trend", thr.get("two_aligned", 2), "EMA 2/3 aligned")
    if a == 1: return _tier("ema_trend", thr.get("one_aligned", 1), "EMA 1/3 aligned")
    return _tier("ema_trend", thr.get("none", -2), "EMAs misaligned")


def _score_rsi(sig, thr):
    z = sig["zone"]
    if z in ("BULLISH_NEUTRAL", "BEARISH_NEUTRAL"):
        return _tier("rsi_zone", thr.get("favorable", 2), f"RSI {z}")
    if z == "NEUTRAL":
        return _tier("rsi_zone", thr.get("neutral", 1), "RSI neutral")
    if z in ("OVERBOUGHT", "OVERSOLD"):
        return _tier("rsi_zone", thr.get("extreme", -1), f"RSI {z}")
    return _tier("rsi_zone", 0, f"RSI zone {z}")


def _score_macd(sig, thr):
    hist = sig.get("histogram") or 0.0
    if hist > 0 and sig.get("hist_increasing"):
        return _tier("macd_momentum", thr.get("strong", 2), "MACD +ve and rising")
    if hist > 0:
        return _tier("macd_momentum", thr.get("positive", 1), "MACD +ve")
    if hist < 0 and sig.get("hist_increasing"):
        return _tier("macd_momentum", thr.get("divergent", -2), "MACD -ve but rising")
    return _tier("macd_momentum", thr.get("negative", -1), "MACD -ve")


def _score_derivatives(sig, thr):
    z = sig["funding_zone"]
    if z in ("SQUEEZE", "EXTREME_NEGATIVE"):
        return _tier("derivatives", thr.get("funding_squeeze", 2), f"Funding {z}")
    if z in ("BUILDING_LONG", "BUILDING_SHORT"):
        return _tier("derivatives", thr.get("oi_building", 1), f"OI {z}")
    if z in ("CROWDED_LONG", "CROWDED_SHORT"):
        return _tier("derivatives", thr.get("crowded", -2), f"Funding {z} — crowded")
    return _tier("derivatives", thr.get("neutral", 0), "Derivatives neutral")


def _score_volatility(sig, thr):
    pct = sig.get("atr_pct") or 0.0
    if pct > 8.0:
        return _tier("volatility", thr.get("extreme", -1), f"ATR% {pct:.1f}")
    if sig.get("expanding"):
        return _tier("volatility", thr.get("expanding", 1), "ATR expanding")
    return _tier("volatility", thr.get("contracting", 0), "ATR contracting/flat")


def _score_resonance(sig, thr):
    if sig.get("aligned"):
        return _tier("resonance", thr.get("all_aligned", 2), f"TFs aligned ({sig['dominant']})")
    if sig.get("dominant") in ("BULLISH", "BEARISH"):
        return _tier("resonance", thr.get("partial", 1), f"Partial ({sig['dominant']})")
    return _tier("resonance", thr.get("conflicting", -1), "TFs conflicting")


def _score_volume(sig, thr):
    if sig.get("surge"):
        return _tier("volume", thr.get("surge", 2), f"Volume surge (×{sig.get('ratio')})")
    ratio = sig.get("ratio") or 0.0
    if ratio > 1.5:
        return _tier("volume", thr.get("elevated", 1), f"Volume elevated (×{ratio:.1f})")
    return _tier("volume", thr.get("flat", 0), "Volume flat")
