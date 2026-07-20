"""Module ③ SIGNALS — 9 groups: 2 attention groups + 7 market groups.

Attention groups (from m01_universe's `cand` dict):
    - attention_frequency  — how many locales × sections mention this token
    - attention_deep       — hashtag deep-fetch mention count / sentiment

Market groups (all delegated to atomic_strategy_lib):
    - ema_trend / rsi / macd / derivatives / volatility / resonance / volume
"""
from __future__ import annotations

from atomic_strategy_lib.signals.trend import (
    ema_current, multi_tf_resonance, trend_classify,
)
from atomic_strategy_lib.signals.momentum import (
    rsi_current, rsi_zone_detect, macd_compute,
)
from atomic_strategy_lib.signals.derivatives import funding_extreme_detect
from atomic_strategy_lib.signals.volatility import (
    atr_current, atr_ratio, dual_speed_atr,
)
from atomic_strategy_lib.signals.volume import volume_surge_detect


def compute_signals(bundle: dict, cand: dict, cfg: dict) -> dict:
    """Note: this signature is different from btc_*/m03 — we take the
    candidate dict too because it carries the attention metadata."""
    sig_cfg = cfg.get("signals", {})
    ema_cfg = sig_cfg.get("ema", {})
    rsi_cfg = sig_cfg.get("rsi", {})
    macd_cfg = sig_cfg.get("macd", {})
    atr_cfg = sig_cfg.get("atr", {})
    vol_cfg = sig_cfg.get("volume", {})
    tf_cfg  = sig_cfg.get("timeframes", {})

    primary_tf = bundle["primary_tf"]
    tf_candles = bundle["tf_candles"]
    candles = tf_candles.get(primary_tf, [])
    closes  = [c.close for c in candles]
    price   = bundle["price"]

    return {
        "attention_frequency": _attention_frequency_sig(cand),
        "attention_deep":      _attention_deep_sig(cand),
        "ema_trend":  _ema_signal(closes, price, ema_cfg),
        "rsi":        _rsi_signal(closes, rsi_cfg),
        "macd":       _macd_signal(closes, macd_cfg),
        "derivatives": _derivatives_signal(bundle),
        "volatility": _volatility_signal(candles, atr_cfg),
        "resonance":  _resonance_signal(tf_candles, primary_tf,
                                        tf_cfg.get("confirmation", [])),
        "volume":     _volume_signal(candles, vol_cfg),
    }


# ── attention (from m01) ─────────────────────────────────────────────
def _attention_frequency_sig(cand):
    a = cand.get("attention", {})
    return {
        "locale_count":     len(a.get("locales", [])),
        "section_count":    len(a.get("sections", [])),
        "en_cn_overlap":    a.get("en_cn_overlap", False),
        "section_overlap":  a.get("section_overlap", False),
    }


def _attention_deep_sig(cand):
    a = cand.get("attention", {})
    n = a.get("deep_mentions", 0)
    if n >= 3:   sentiment = "STRONG"
    elif n >= 1: sentiment = "MIXED"
    else:        sentiment = "NONE"
    return {"deep_mentions": n, "sentiment": sentiment}


# ── market (same idea as btc_*) ──────────────────────────────────────
def _ema_signal(closes, price, cfg):
    if not closes:
        return {"aligned_count": 0, "direction": "NEUTRAL"}
    f = ema_current(closes, period=cfg.get("fast", 20))
    m = ema_current(closes, period=cfg.get("mid",  60))
    s = ema_current(closes, period=cfg.get("slow", 200))
    aligned, direction = 0, "NEUTRAL"
    if f and m and s and price:
        above = [price > f, price > m, price > s]
        aligned = sum(above)
        if aligned >= 2:     direction = "BULLISH"
        elif not any(above): direction = "BEARISH"
    return {"ema_fast": f, "ema_mid": m, "ema_slow": s,
            "aligned_count": aligned, "direction": direction}


def _rsi_signal(closes, cfg):
    v = rsi_current(closes, period=cfg.get("period", 14))
    z = rsi_zone_detect(v, overbought=cfg.get("overbought", 70),
                           oversold=cfg.get("oversold", 30)) \
        if v is not None else "UNKNOWN"
    return {"value": v, "zone": z}


def _macd_signal(closes, cfg):
    m = macd_compute(closes,
                     fast=cfg.get("fast", 12), slow=cfg.get("slow", 26),
                     signal=cfg.get("signal", 9)) or {}
    return {
        "histogram":      m.get("histogram_current"),
        "hist_increasing": m.get("histogram_increasing", False),
    }


def _derivatives_signal(bundle):
    fr = bundle["funding"].rate if bundle.get("funding") else 0.0
    info = funding_extreme_detect(fr) or {}
    oi = bundle.get("oi") or {}
    return {
        "funding_rate": fr,
        "funding_zone": info.get("direction", "NEUTRAL"),
        "funding_signal": info.get("signal_direction", "NEUTRAL"),
        "oi_value":     oi.get("oi_value", 0) if isinstance(oi, dict) else 0,
    }


def _volatility_signal(candles, cfg):
    atr_val = atr_current(candles, period=cfg.get("period", 14))
    atr_pct = atr_ratio(candles, period=cfg.get("period", 14))
    dual = dual_speed_atr(candles) if len(candles) > 21 else {}
    return {"atr": atr_val, "atr_pct": atr_pct,
            "expanding": dual.get("expanding", False)}


def _resonance_signal(tf_candles, primary_tf, confirm_tfs):
    dirs = {}
    for tf in [primary_tf, *confirm_tfs]:
        c = tf_candles.get(tf, [])
        if len(c) >= 3:
            dirs[tf] = trend_classify(c[-5:])
    r = multi_tf_resonance(dirs) if dirs else {}
    return {"tf_directions": dirs, "aligned": r.get("aligned", False),
            "dominant": r.get("dominant", "UNKNOWN")}


def _volume_signal(candles, cfg):
    if not candles:
        return {"surge": False, "ratio": None}
    surge = volume_surge_detect(candles,
                                multiplier=cfg.get("surge_multiplier", 2.0),
                                lookback=cfg.get("lookback", 24)) or {}
    return {
        "surge": surge.get("surge", False),
        "ratio": surge.get("ratio"),
    }
