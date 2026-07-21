"""BTC Multi-Factor Trend — orchestration file.

Only assembly here. Each responsibility lives in its own module so a
future UI can edit them independently:

    tier_scorers/    — how each signal maps to a tier score (5 files)
    direction_votes/ — how each signal votes long/short (3 files)
    template_meta.py — declared metadata (moved out of this file)

This file is intentionally small: read it top-to-bottom and you see
"which blocks I use → which scorers I use → which voters I use".
"""
from __future__ import annotations

from demo_strategy._shared.blocks import (
    StrategyContext, StrategyDecision, flat_decision, load_block,
)
from scripts.template_meta import TEMPLATE_META  # noqa: F401  (re-exported)
from scripts.tier_scorers import (
    score_ema_trend, score_rsi_zone, score_macd_momentum,
    score_volatility, score_resonance,
)
from scripts.direction_votes import (
    vote_ema_direction, vote_macd_histogram, vote_resonance_direction,
)


def calculate_signal(ctx: StrategyContext) -> StrategyDecision:
    bars = ctx.market.bars
    if bars is None or len(bars) < 60:
        n = 0 if bars is None else len(bars)
        return flat_decision(f"insufficient bars ({n})")

    # ctx.config is the whole strategy config (6 business sections)
    ind_cfg = ctx.config.get("indicators", {})
    sig_ov  = ind_cfg.get("params", {})
    sco_ov  = {
        "hierarchical": {"mode": ind_cfg.get("scoring", {}).get("combinator", "additive")},
        "verdict_gate": {"thresholds": ind_cfg.get("scoring", {}).get("verdict_thresholds", {})},
    }
    tier_thr = ind_cfg.get("tier_scores", {})
    # decision.direction_vote block override — read tie_breaker from sizing.direction
    dec_ov = {
        "direction_vote": {
            "tie_breaker": ctx.config.get("sizing", {}).get("direction", {}).get("tie_breaker", "flat"),
        },
    }

    # ── 1. signals ────────────────────────────────────────────────────
    ema  = load_block("signals", "ema",       overrides=sig_ov.get("ema"))       .compute(bars)
    rsi  = load_block("signals", "rsi",       overrides=sig_ov.get("rsi"))       .compute(bars)
    macd = load_block("signals", "macd",      overrides=sig_ov.get("macd"))      .compute(bars)
    atr  = load_block("signals", "atr",       overrides=sig_ov.get("atr"))       .compute(bars)
    reso = load_block("signals", "resonance", overrides=sig_ov.get("resonance")) .compute(bars)

    # ── 2. per-tier scoring (each in its own file) ────────────────────
    tiers = [
        score_ema_trend(     ema,  tier_thr.get("ema_trend",     {})),
        score_rsi_zone(      rsi,  tier_thr.get("rsi_zone",      {})),
        score_macd_momentum( macd, tier_thr.get("macd_momentum", {})),
        score_volatility(    atr,  tier_thr.get("volatility",    {})),
        score_resonance(     reso, tier_thr.get("resonance",     {})),
    ]

    # ── 3. score aggregation + verdict ────────────────────────────────
    total_dict = load_block("scoring", "hierarchical",
                            overrides=sco_ov.get("hierarchical")).compute(tiers)
    verdict_dict = load_block("scoring", "verdict_gate",
                              overrides=sco_ov.get("verdict_gate")).compute(total_dict)
    verdict = verdict_dict["verdict"]
    total = total_dict["total"]

    if verdict in ("SKIP", "WATCHLIST"):
        return StrategyDecision(side="flat", strength=0.0,
                                reason=f"{verdict} · score={total}")

    # ── 4. direction voting (each vote in its own file) ──────────────
    votes = [
        vote_ema_direction(ema),
        vote_macd_histogram(macd),
        vote_resonance_direction(reso),
    ]
    votes = [v for v in votes if v]           # drop abstains

    side = load_block("decision", "direction_vote",
                      overrides=dec_ov.get("direction_vote")).compute(votes)["side"]
    if side == "flat":
        return StrategyDecision(side="flat", strength=0.0,
                                reason=f"votes tied · score={total}")

    strength = min(1.0, max(0.0, total / 10.0))
    reason = f"{verdict}(score={total}) · " + " / ".join(
        f"{t['name']}:{t['score']:+d}" for t in tiers
    )
    return StrategyDecision(side=side, strength=strength, reason=reason)
