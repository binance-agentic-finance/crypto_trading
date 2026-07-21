"""Square Buzz Screener — orchestration only.

Assembly reads: block calls → tier scorers (7 files) → 5-tier verdict
gate (1 file) → basket weighting. Metadata is in template_meta.py.
"""
from __future__ import annotations

from demo_strategy._shared.blocks import (
    SelectionContext, SelectionDecision, load_block,
)
from scripts.template_meta import TEMPLATE_META  # noqa: F401
from scripts.tier_scorers import (
    score_attention_frequency, score_attention_deep,
    score_ema_trend, score_rsi_zone, score_macd_momentum,
    score_volatility, score_volume,
)
from scripts.verdict_gate import verdict_five_tier


def calculate_selection(ctx: SelectionContext) -> SelectionDecision:
    per_symbol_att = ctx.metadata.get("attention", {})
    max_basket     = int(ctx.metadata.get("max_basket_size", 5))
    ind_cfg = ctx.config.get("indicators", {})
    sig_ov  = ind_cfg.get("params", {})
    sco_ov  = {
        "hierarchical": {"mode": ind_cfg.get("scoring", {}).get("combinator", "additive")},
        "verdict_gate": {"thresholds": ind_cfg.get("scoring", {}).get("verdict_thresholds", {})},
    }
    tier_thr = ind_cfg.get("tier_scores", {})

    scored = []
    for sym, bars in ctx.bars_by_symbol.items():
        if bars is None or len(bars) < 60:
            continue
        att = per_symbol_att.get(sym, {})
        tiers = _score_symbol(bars, att, sig_ov, tier_thr)
        total = load_block("scoring", "hierarchical",
                           overrides=sco_ov.get("hierarchical")).compute(tiers)
        verdict = verdict_five_tier(total, tiers, sco_ov.get("verdict_gate", {}))
        scored.append((sym, total["total"], verdict, tiers))

    survivors = [(s, t) for s, t, v, _ in scored
                 if v in ("STRONG_CANDIDATE", "CANDIDATE")]
    survivors.sort(key=lambda x: -x[1])
    survivors = survivors[:max_basket]

    if not survivors:
        return SelectionDecision(
            weights={}, reason=f"no CANDIDATE (scanned {len(scored)})",
            metadata={"scored": len(scored)},
        )

    denom = sum(t for _, t in survivors) or 1
    weights = {s: (t / denom) for s, t in survivors}
    return SelectionDecision(
        weights=weights,
        reason=f"basket={len(weights)} / scanned={len(scored)}",
        metadata={"scored": len(scored)},
    )


def _score_symbol(bars, att: dict, sig_ov: dict, tier_thr: dict) -> list[dict]:
    """Compute all blocks for ONE symbol → list of tier dicts.

    Kept inline here (not extracted to a file) because it's just the
    N-line assembly that binds block outputs to their tier scorers.
    """
    # attention blocks (Square-specific)
    fr = load_block("signals", "attention_frequency",
                    overrides=sig_ov.get("attention_frequency")).compute(att)
    dp = load_block("signals", "attention_deep",
                    overrides=sig_ov.get("attention_deep")).compute(att)

    # technical blocks
    ema  = load_block("signals", "ema",   overrides=sig_ov.get("ema")) .compute(bars)
    rsi  = load_block("signals", "rsi",   overrides=sig_ov.get("rsi")) .compute(bars)
    macd = load_block("signals", "macd",  overrides=sig_ov.get("macd")).compute(bars)
    atr  = load_block("signals", "atr",   overrides=sig_ov.get("atr")) .compute(bars)
    vol  = load_block("signals", "volume_surge",
                      overrides=sig_ov.get("volume_surge")).compute(bars)

    return [
        score_attention_frequency(fr,  tier_thr.get("attention_frequency", {})),
        score_attention_deep(     dp,  tier_thr.get("attention_deep",      {})),
        score_ema_trend(          ema, tier_thr.get("ema_trend",           {})),
        score_rsi_zone(           rsi, tier_thr.get("rsi_zone",            {})),
        score_macd_momentum(      macd, tier_thr.get("macd_momentum",      {})),
        score_volatility(         atr, tier_thr.get("volatility",          {})),
        score_volume(             vol, tier_thr.get("volume",              {})),
    ]
