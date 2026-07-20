"""square_buzz_screener · strategy facade.

Same 7-module orchestration shape as btc_multi_factor_trend, but each
module has a different job: universe comes from a Square scrape, and
scoring adds attention factors on top of the technical stack.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .m01_universe import discover_universe
from .m02_data import fetch_market_bundle
from .m03_signals import compute_signals
from .m04_scoring import score_one
from .m05_decision import decide_one
from .m06_execution import execute_all
from .m07_report import render_report


def run(cfg: dict) -> dict:
    """End-to-end pipeline: attention discovery → per-token full validation."""
    candidates = discover_universe(cfg)

    results = []
    for cand in candidates:
        symbol = cand["symbol"]
        bundle  = fetch_market_bundle(symbol, cfg)
        signals = compute_signals(bundle, cand, cfg)  # NB: cand carries attention data
        score   = score_one(signals, cfg)
        decision = decide_one(symbol, signals, score, cfg)
        results.append({
            "symbol":   symbol,
            "price":    bundle["price"],
            "attention": cand.get("attention", {}),
            "signals":  signals,
            "score":    score,
            "decision": decision,
        })

    execution = execute_all(results, cfg)

    return {
        "strategy":     "square_buzz_screener",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidates":   len(candidates),
        "results":      results,
        "execution":    execution,
    }


__all__ = ["run", "render_report"]
