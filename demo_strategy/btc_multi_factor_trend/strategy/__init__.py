"""btc_multi_factor_trend · strategy facade.

Pipeline entry point — orchestrates the 7 numbered modules in order.
`run.py` only needs to call `run(cfg)` and hand the result to the report.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .m01_universe import pick_universe
from .m02_data import fetch_market_bundle
from .m03_signals import compute_signals
from .m04_scoring import score_one
from .m05_decision import decide_one
from .m06_execution import execute_all
from .m07_report import render_report


def run(cfg: dict) -> dict:
    """End-to-end: universe → data → signals → score → decide → execute.

    Returns the pipeline_result dict (also fit for JSON dump).
    """
    universe = pick_universe(cfg)
    results = []
    for symbol in universe:
        bundle  = fetch_market_bundle(symbol, cfg)
        signals = compute_signals(bundle, cfg)
        score   = score_one(signals, cfg)
        decision = decide_one(symbol, signals, score, cfg)
        results.append({
            "symbol":   symbol,
            "price":    bundle["price"],
            "signals":  signals,
            "score":    score,
            "decision": decision,
        })

    execution = execute_all(results, cfg)

    return {
        "strategy":     "btc_multi_factor_trend",
        "mode":         cfg.get("_active_mode", {}).get("name", "balanced"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results":      results,
        "execution":    execution,
    }


__all__ = ["run", "render_report"]
