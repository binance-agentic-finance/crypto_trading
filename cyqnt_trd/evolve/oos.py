"""OOS validation — runs only at the end on the held-out 30-day window.

Anti-overfit gate per SPEC §11:
    OOS Sharpe < IS Sharpe × 0.4  → reject (degradation_check fails)

Only the top 5 IS performers (or whatever subset is requested) are evaluated
on OOS. Every other genome is dropped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from .bridge import backtest_genome
from .engine import EvolutionConfig
from .fitness import fitness_score
from .genome import StrategyGenome
from .population import Population


DEGRADATION_THRESHOLD = 0.4  # OOS Sharpe must be ≥ this × IS Sharpe


def run_oos_validation(
    *,
    candidates: List[StrategyGenome],
    df_is: pd.DataFrame,
    df_oos: pd.DataFrame,
    config: EvolutionConfig,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Validate top-N candidates on OOS data.

    Returns a dict with structure:
        {
          "top_n": int,
          "candidates": [
            {
              "genome_id": ...,
              "genome": {...},
              "is_metrics": {...},
              "oos_metrics": {...},
              "is_fitness": float,
              "oos_fitness": float,
              "passes_degradation_check": bool,
              "degradation_reason": str (if fail),
            }, ...
          ],
          "champions": [genome_id, ...]   # those that passed
        }
    """
    out_candidates: List[Dict[str, Any]] = []

    for g in candidates[:top_n]:
        # IS re-run for clean comparison
        is_res = backtest_genome(
            genome=g, df=df_is,
            fee_bps=config.fee_bps, slippage_bps=config.slippage_bps,
            initial_capital=config.initial_capital, record_trades=False,
        )
        oos_res = backtest_genome(
            genome=g, df=df_oos,
            fee_bps=config.fee_bps, slippage_bps=config.slippage_bps,
            initial_capital=config.initial_capital, record_trades=False,
        )

        is_metrics = is_res.to_dict()
        oos_metrics = oos_res.to_dict()
        is_fit = fitness_score(is_metrics, n_factors=len(g.entry_factors))
        oos_fit = fitness_score(oos_metrics, n_factors=len(g.entry_factors))

        # Degradation check: OOS Sharpe ≥ IS Sharpe × threshold (and same sign)
        is_sharpe = float(is_metrics.get("sharpe_ratio") or 0.0)
        oos_sharpe = float(oos_metrics.get("sharpe_ratio") or 0.0)
        passes = True
        reason = ""

        if is_sharpe <= 0:
            passes = oos_sharpe > 0  # IS already bad → OOS must be good to pass
            if not passes:
                reason = f"IS sharpe non-positive ({is_sharpe:.2f}) and OOS not improving ({oos_sharpe:.2f})"
        else:
            min_oos_sharpe = is_sharpe * DEGRADATION_THRESHOLD
            if oos_sharpe < min_oos_sharpe:
                passes = False
                reason = (
                    f"OOS sharpe {oos_sharpe:.2f} < {DEGRADATION_THRESHOLD} × IS sharpe "
                    f"({is_sharpe:.2f}) = {min_oos_sharpe:.2f}"
                )

        # Additional check: OOS trade count ≥ 5 (statistical significance)
        oos_trades = int(oos_metrics.get("trade_count") or 0)
        if oos_trades < 5:
            passes = False
            reason = (reason + "; " if reason else "") + f"OOS trades too few ({oos_trades})"

        out_candidates.append(
            {
                "genome_id": g.genome_id,
                "genome": g.to_dict(),
                "is_metrics": is_metrics,
                "oos_metrics": oos_metrics,
                "is_fitness": is_fit,
                "oos_fitness": oos_fit,
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "passes_degradation_check": passes,
                "degradation_reason": reason,
            }
        )

    champions = [c["genome_id"] for c in out_candidates if c["passes_degradation_check"]]
    return {
        "top_n": top_n,
        "candidates": out_candidates,
        "champions": champions,
        "degradation_threshold": DEGRADATION_THRESHOLD,
    }


def pick_top_n_by_fitness(
    population: Population, scored_fitness: Dict[str, float], n: int = 5
) -> List[StrategyGenome]:
    """Return the top-N genomes by their last-generation fitness."""
    pairs = []
    for g in population.genomes:
        f = scored_fitness.get(g.genome_id)
        if f is None and g.fitness_history:
            f = g.fitness_history[-1]
        if f is None:
            f = float("-inf")
        pairs.append((g, f))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in pairs[:n]]
