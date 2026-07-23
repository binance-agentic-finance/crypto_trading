"""Evolution engine — orchestrates one generation.

A generation has the following sub-steps (called by the CLI):

    Step A — backtest all genomes on the IS DataFrame, compute fitness.
    Step B — select survivors (elitism + tournament).
    Step C — fill empty slots with crossover children.
    Step D — apply random mutation to a fraction of the new pool.
    Step E — emit a results JSON describing top/bottom performers
             (the LLM consumes this and produces ``mutations.json``).
    Step F — apply the LLM mutations on top of the new pool (next gen).

Steps A-D run together in :func:`run_generation_step`. The CLI then writes
``gen_NNN_results.json``. Step F runs in :func:`apply_llm_mutations` when
the LLM hands back mutations.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .bridge import backtest_genome
from .crossover import crossover, random_mutation
from .fitness import fitness_score
from .genome import (
    ALLOWED_SPECIES,
    StrategyGenome,
    cap_max_bars,
    infer_species,
)
from .llm_mutation import apply_mutations
from .population import Population
from .selection import select_survivors

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────

@dataclass
class EvolutionConfig:
    """Per-session evolution config."""

    population_size: int = 50
    elite_pct: float = 0.30
    tournament_pct: float = 0.40
    tournament_k: int = 3
    random_mutation_pct: float = 0.20  # fraction of new gen pool randomly mutated
    llm_mutation_count: int = 15        # how many LLM mutations expected per gen
    species_floor: int = 5              # min individuals per species (if total enough)
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    initial_capital: float = 10_000.0
    seed: int = 42


# ── Result containers ──────────────────────────────────────────────────────

@dataclass
class GenomeEvalResult:
    genome: StrategyGenome
    fitness: float
    metrics: Dict[str, Any]


@dataclass
class GenerationResult:
    generation: int
    symbol: str
    interval: str
    population_size: int
    best_fitness: float
    avg_fitness: float
    species_distribution: Dict[str, int]
    top_5: List[Dict[str, Any]]
    bottom_5: List[Dict[str, Any]]
    eliminated: List[str]
    survivors: List[str]
    new_via_crossover: List[str]
    new_via_random_mutation: List[str]
    need_mutations: int
    elapsed_seconds: float
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "symbol": self.symbol,
            "interval": self.interval,
            "population_size": self.population_size,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "species_distribution": self.species_distribution,
            "top_5": self.top_5,
            "bottom_5": self.bottom_5,
            "eliminated": self.eliminated,
            "survivors": self.survivors,
            "new_via_crossover": self.new_via_crossover,
            "new_via_random_mutation": self.new_via_random_mutation,
            "need_mutations": self.need_mutations,
            "elapsed_seconds": self.elapsed_seconds,
            "errors": self.errors,
        }


# ── Step A: evaluate ───────────────────────────────────────────────────────

def evaluate_population(
    population: Population,
    df_is: pd.DataFrame,
    config: EvolutionConfig,
) -> List[GenomeEvalResult]:
    """Backtest every genome on IS data and compute fitness."""
    results: List[GenomeEvalResult] = []
    for g in population.genomes:
        try:
            res = backtest_genome(
                genome=g,
                df=df_is,
                fee_bps=config.fee_bps,
                slippage_bps=config.slippage_bps,
                initial_capital=config.initial_capital,
                record_trades=False,
            )
            metrics = res.to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.warning("backtest failed for %s: %s", g.genome_id, exc)
            metrics = {
                "total_return": -1.0,
                "sharpe_ratio": -5.0,
                "max_drawdown": -1.0,
                "trade_count": 0,
                "win_rate": 0.0,
            }
        f = fitness_score(metrics, n_factors=len(g.entry_factors))
        g.fitness_history.append(float(f))
        results.append(GenomeEvalResult(genome=g, fitness=f, metrics=metrics))
    return results


# ── Step B-D: produce next-generation pool ─────────────────────────────────

def build_next_generation(
    *,
    scored: List[GenomeEvalResult],
    config: EvolutionConfig,
    generation: int,
    rng: Optional[random.Random] = None,
) -> Tuple[Population, List[str], List[str], List[str]]:
    """Apply selection + crossover + random mutation. Returns next-gen pool.

    Returns
    -------
    (next_population, culled_ids, crossover_ids, mutation_ids)
    """
    if rng is None:
        rng = random.Random(config.seed + generation)

    pairs = [(r.genome, r.fitness) for r in scored]
    survivors, culled = select_survivors(
        scored=pairs,
        elite_pct=config.elite_pct,
        tournament_pct=config.tournament_pct,
        tournament_k=config.tournament_k,
        rng=rng,
    )

    next_pop = Population(generation=generation)
    for g in survivors:
        next_pop.add(g)

    # ── Fill empty slots ──
    n_target = config.population_size
    n_to_fill = n_target - len(next_pop)
    crossover_ids: List[str] = []
    mutation_ids: List[str] = []

    if n_to_fill <= 0 or not survivors:
        return next_pop, culled, crossover_ids, mutation_ids

    # Reserve a fraction for random mutation, the rest for crossover
    n_random_mut = max(1, int(n_to_fill * config.random_mutation_pct))
    n_crossover = n_to_fill - n_random_mut

    # crossover children
    next_idx = 0
    for i in range(n_crossover):
        a = rng.choice(survivors)
        b = rng.choice(survivors)
        if a.genome_id == b.genome_id and len(survivors) > 1:
            b = rng.choice([g for g in survivors if g.genome_id != a.genome_id])
        new_id = f"x_g{generation}_{i:03d}"
        # ensure unique
        while next_pop.get(new_id) is not None:
            next_idx += 1
            new_id = f"x_g{generation}_{next_idx:03d}"
            next_idx += 1
        child = crossover(a, b, new_id=new_id, generation=generation, rng=rng)
        # Validate; if invalid skip
        if child.is_valid():
            next_pop.add(child)
            crossover_ids.append(child.genome_id)

    # random mutation children
    for i in range(n_random_mut):
        parent = rng.choice(survivors)
        new_id = f"rm_g{generation}_{i:03d}"
        while next_pop.get(new_id) is not None:
            new_id = f"rm_g{generation}_{i:03d}_{rng.randint(0, 9999)}"
        child = random_mutation(parent, new_id=new_id, generation=generation, rng=rng)
        if child.is_valid():
            next_pop.add(child)
            mutation_ids.append(child.genome_id)

    return next_pop, culled, crossover_ids, mutation_ids


# ── Apply LLM mutations on top of next-gen pool ────────────────────────────

def apply_llm_mutations(
    population: Population,
    mutations: List[Dict[str, Any]],
    *,
    generation: int,
    config: EvolutionConfig,
) -> Tuple[List[str], List[str]]:
    """Apply LLM mutations (replaces some random ones if pool is over capacity).

    Returns (added_ids, errors).
    """
    new_genomes, errors = apply_mutations(population, mutations, generation=generation)
    added_ids = [g.genome_id for g in new_genomes]

    # Trim back to target population size if exceeded.
    # Order of preference for dropping (keep best, drop newest random/crossover):
    #   1. random_mutation entries that were NOT just added by LLM
    #   2. crossover entries (x_) that were NOT just added
    #   3. any entry that has never been evaluated (no fitness history)
    target = config.population_size
    while len(population) > target:
        # Build the candidate-to-drop list freshly each iteration so removals
        # are reflected.
        droppable_rm = [g.genome_id for g in population.genomes
                        if g.genome_id.startswith("rm_") and g.genome_id not in added_ids]
        droppable_x = [g.genome_id for g in population.genomes
                       if g.genome_id.startswith("x_") and g.genome_id not in added_ids]
        droppable_unscored = [g.genome_id for g in population.genomes
                              if not g.fitness_history and g.genome_id not in added_ids]
        if droppable_rm:
            population.remove(droppable_rm[-1])
        elif droppable_x:
            population.remove(droppable_x[-1])
        elif droppable_unscored:
            population.remove(droppable_unscored[-1])
        else:
            # Final fallback: drop the lowest-fitness genome that isn't a
            # newly-injected LLM mutation. We never drop LLM-added genomes
            # (they haven't had a chance to be evaluated yet).
            scored_pool = [
                (g, g.fitness_history[-1] if g.fitness_history else float("-inf"))
                for g in population.genomes if g.genome_id not in added_ids
            ]
            if not scored_pool:
                break
            scored_pool.sort(key=lambda x: x[1])  # ascending: worst first
            population.remove(scored_pool[0][0].genome_id)

    return added_ids, errors


# ── Reporting helpers ──────────────────────────────────────────────────────

def build_generation_result(
    *,
    generation: int,
    symbol: str,
    interval: str,
    scored: List[GenomeEvalResult],
    next_pop: Population,
    culled: List[str],
    crossover_ids: List[str],
    mutation_ids: List[str],
    elapsed: float,
    config: EvolutionConfig,
) -> GenerationResult:
    """Build the per-generation result JSON for the LLM to read."""
    sorted_scored = sorted(scored, key=lambda r: r.fitness, reverse=True)
    fitnesses = [r.fitness for r in scored]
    best = max(fitnesses) if fitnesses else 0.0
    avg = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

    species_dist = next_pop.species_distribution()

    top5 = [_eval_to_summary(r) for r in sorted_scored[:5]]
    bottom5 = [_eval_to_summary(r, with_failure=True) for r in sorted_scored[-5:]]

    survivor_ids = [g.genome_id for g in next_pop.genomes
                    if g.genome_id not in crossover_ids and g.genome_id not in mutation_ids]

    return GenerationResult(
        generation=generation,
        symbol=symbol,
        interval=interval,
        population_size=len(next_pop),
        best_fitness=best,
        avg_fitness=avg,
        species_distribution=species_dist,
        top_5=top5,
        bottom_5=bottom5,
        eliminated=culled,
        survivors=survivor_ids,
        new_via_crossover=crossover_ids,
        new_via_random_mutation=mutation_ids,
        need_mutations=config.llm_mutation_count,
        elapsed_seconds=elapsed,
    )


def _eval_to_summary(r: GenomeEvalResult, *, with_failure: bool = False) -> Dict[str, Any]:
    summary = {
        "genome_id": r.genome.genome_id,
        "species": r.genome.species,
        "generation": r.genome.generation,
        "fitness": r.fitness,
        "metrics": _slim_metrics(r.metrics),
        "genome": r.genome.to_dict(),
    }
    if with_failure:
        summary["failure_analysis"] = _failure_analysis(r)
    return summary


def _slim_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    keep = ["total_return", "sharpe_ratio", "max_drawdown", "trade_count",
            "win_rate", "avg_trade_pnl", "exposure"]
    return {k: m.get(k) for k in keep if k in m}


def _failure_analysis(r: GenomeEvalResult) -> str:
    m = r.metrics
    trades = int(m.get("trade_count", 0))
    win = float(m.get("win_rate", 0.0))
    ret = float(m.get("total_return", 0.0))
    dd = float(m.get("max_drawdown", 0.0))
    reasons = []
    if trades < 10:
        reasons.append(f"too few trades ({trades})")
    if trades > 500:
        reasons.append(f"overtrading ({trades})")
    if win < 0.35 and trades >= 10:
        reasons.append(f"low win rate ({win:.2f})")
    if ret < -0.1:
        reasons.append(f"negative return ({ret:.2%})")
    if dd < -0.2:
        reasons.append(f"large drawdown ({dd:.2%})")
    if not reasons:
        reasons.append("low fitness — likely poor signal quality or no edge")
    return "; ".join(reasons)
