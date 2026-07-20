"""Module ③ SIGNALS — compute 6 groups of numerical indicators.

Key discipline: THIS MODULE DOES NOT MAKE VERDICT CALLS. It returns raw
numeric or state values (`ema_fast=…`, `rsi_value=…`, `funding_zone="EXTREME"`
where "EXTREME" is a *state name*, not a score). Scoring — the mapping from
signals → tier scores — happens in `m04_scoring.py`.

The 6 groups:
    1. ema_trend       — EMA20/60/200 alignment vs price + fast/slow cross
    2. rsi             — RSI(14) value + zone label
    3. macd            — MACD(12,26,9) hist direction + strength
    4. derivatives     — funding rate zone + OI value
    5. volatility      — ATR value + expansion regime
    6. resonance       — 1h/4h/1d trend alignment
"""
from __future__ import annotations

from atomic_strategy_lib.signals.trend import (
    ema_current, ema_cross_detect, multi_tf_resonance, trend_classify,
)
from atomic_strategy_lib.signals.momentum import (
    rsi_current, rsi_zone_detect, macd_compute,
)
from atomic_strategy_lib.signals.derivatives import funding_extreme_detect
from atomic_strategy_lib.signals.volatility import (
    atr_current, atr_ratio, dual_speed_atr,
)


def compute_signals(bundle: dict, cfg: dict) -> dict:
    sig_cfg = cfg.get("signals", {})
    ema_cfg  = sig_cfg.get("ema", {})
    rsi_cfg  = sig_cfg.get("rsi", {})
    macd_cfg = sig_cfg.get("macd", {})
    atr_cfg  = sig_cfg.get("atr", {})
    tf_cfg   = sig_cfg.get("timeframes", {})

    primary_tf = bundle["primary_tf"]
    tf_candles = bundle["tf_candles"]
    candles    = tf_candles.get(primary_tf, [])
    closes     = [c.close for c in candles]
    price      = bundle["price"]

    return {
        "ema_trend":   _ema_signals(closes, price, ema_cfg),
        "rsi":         _rsi_signal(closes, rsi_cfg),
        "macd":        _macd_signal(closes, macd_cfg),
        "derivatives": _derivatives_signal(bundle),
        "volatility":  _volatility_signal(candles, atr_cfg),
        "resonance":   _resonance_signal(tf_candles, primary_tf,
                                         tf_cfg.get("confirmation", [])),
    }


# ─────────────────────────────────────────────────────────────────────
# per-group helpers — kept tiny so each is easy to swap out
# ─────────────────────────────────────────────────────────────────────
def _ema_signals(closes, price, cfg):
    ema_fast = ema_current(closes, period=cfg.get("fast", 20))
    ema_mid  = ema_current(closes, period=cfg.get("mid",  60))
    ema_slow = ema_current(closes, period=cfg.get("slow", 200))
    ema_cross = ema_cross_detect(
        closes,
        fast_period=cfg.get("cross_fast", 20),
        slow_period=cfg.get("cross_slow", 60),
    ) or {}

    aligned = 0
    direction = "NEUTRAL"
    if ema_fast and ema_mid and ema_slow and price:
        above = [price > ema_fast, price > ema_mid, price > ema_slow]
        aligned = sum(above)
        if aligned >= 2:                 direction = "BULLISH"
        elif not any(above):             direction = "BEARISH"

    return {
        "ema_fast":      ema_fast,
        "ema_mid":       ema_mid,
        "ema_slow":      ema_slow,
        "aligned_count": aligned,
        "direction":     direction,
        "cross":         ema_cross.get("cross", "NONE"),
        "spread_pct":    ema_cross.get("spread_pct", 0.0),
    }


def _rsi_signal(closes, cfg):
    rsi_val = rsi_current(closes, period=cfg.get("period", 14))
    zone = "UNKNOWN"
    if rsi_val is not None:
        zone = rsi_zone_detect(
            rsi_val,
            overbought=cfg.get("overbought", 70),
            oversold=cfg.get("oversold", 30),
        )
    return {"value": rsi_val, "zone": zone}


def _macd_signal(closes, cfg):
    m = macd_compute(closes,
                     fast=cfg.get("fast", 12),
                     slow=cfg.get("slow", 26),
                     signal=cfg.get("signal", 9)) or {}
    return {
        "macd":           m.get("macd_current"),
        "signal":         m.get("signal_current"),
        "histogram":      m.get("histogram_current"),
        "hist_increasing": m.get("histogram_increasing", False),
    }


def _derivatives_signal(bundle):
    fr = bundle["funding"].rate if bundle.get("funding") else 0.0
    info = funding_extreme_detect(fr) or {}
    oi   = bundle.get("oi") or {}
    return {
        "funding_rate":   fr,
        "funding_zone":   info.get("direction", "NEUTRAL"),
        "funding_signal": info.get("signal_direction", "NEUTRAL"),
        "oi_value":       oi.get("oi_value", 0) if isinstance(oi, dict) else 0,
    }


def _volatility_signal(candles, cfg):
    atr_val = atr_current(candles, period=cfg.get("period", 14))
    atr_pct = atr_ratio(candles, period=cfg.get("period", 14))
    dual    = dual_speed_atr(candles) if len(candles) > 21 else {}
    return {
        "atr":              atr_val,
        "atr_pct":          atr_pct,
        "dual_atr_ratio":   dual.get("ratio"),
        "expanding":        dual.get("expanding", False),
    }


def _resonance_signal(tf_candles, primary_tf, confirm_tfs):
    directions = {}
    for tf in [primary_tf, *confirm_tfs]:
        c = tf_candles.get(tf, [])
        if len(c) >= 3:
            directions[tf] = trend_classify(c[-5:])
    reso = multi_tf_resonance(directions) if directions else {}
    return {
        "tf_directions": directions,
        "aligned":       reso.get("aligned", False),
        "dominant":      reso.get("dominant", "UNKNOWN"),
        "score":         reso.get("score", 0),
    }
