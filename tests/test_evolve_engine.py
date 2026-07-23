"""Tests for fitness/selection/crossover/llm_mutation/engine."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.evolve.crossover import crossover, random_mutation
from cyqnt_trd.evolve.engine import (
    EvolutionConfig,
    apply_llm_mutations,
    build_generation_result,
    build_next_generation,
    evaluate_population,
)
from cyqnt_trd.evolve.fitness import fitness_score
from cyqnt_trd.evolve.genome import Factor, Filter, StrategyGenome
from cyqnt_trd.evolve.llm_mutation import apply_mutations
from cyqnt_trd.evolve.population import Population
from cyqnt_trd.evolve.selection import select_survivors


# ── Fitness ────────────────────────────────────────────────────────────────

def test_fitness_high_sharpe_high_return_positive():
    m = {"sharpe_ratio": 2.0, "total_return": 0.3, "max_drawdown": -0.05,
         "trade_count": 100, "win_rate": 0.55}
    f = fitness_score(m, n_factors=2)
    assert f > 0.5


def test_fitness_too_few_trades_penalised():
    m_few = {"sharpe_ratio": 2.0, "total_return": 0.3, "max_drawdown": -0.05,
             "trade_count": 5, "win_rate": 0.6}
    m_norm = {"sharpe_ratio": 2.0, "total_return": 0.3, "max_drawdown": -0.05,
              "trade_count": 100, "win_rate": 0.6}
    assert fitness_score(m_few, n_factors=1) < fitness_score(m_norm, n_factors=1)


def test_fitness_handles_nan_safely():
    m = {"sharpe_ratio": float("nan"), "total_return": float("nan"), "max_drawdown": 0,
         "trade_count": 0, "win_rate": 0}
    f = fitness_score(m)
    assert isinstance(f, float)


# ── Selection ──────────────────────────────────────────────────────────────

def _g(gid: str) -> StrategyGenome:
    return StrategyGenome(
        genome_id=gid, species="trend",
        entry_factors=[Factor("ema", "cross_above", {"fast": 8, "slow": 21})],
        exit_type="pct_stop_tp",
        exit_params={"stop_pct": 0.01, "tp_pct": 0.02, "max_bars": 20},
    )


def test_selection_elitism_keeps_top_genomes():
    scored = [(_g(f"a{i}"), float(i)) for i in range(10)]  # a9 highest
    rng = random.Random(0)
    survivors, culled = select_survivors(
        scored=scored, elite_pct=0.3, tournament_pct=0.4, rng=rng,
    )
    survivor_ids = {g.genome_id for g in survivors}
    # Top 3 (a9, a8, a7) should be elites
    assert "a9" in survivor_ids and "a8" in survivor_ids and "a7" in survivor_ids
    # Bottom 30% should be culled
    assert "a0" in set(culled)


def test_selection_culled_count_correct():
    scored = [(_g(f"a{i}"), float(i)) for i in range(10)]
    rng = random.Random(0)
    survivors, culled = select_survivors(
        scored=scored, elite_pct=0.3, tournament_pct=0.4, rng=rng,
    )
    assert len(survivors) + len(culled) == 10


# ── Crossover ──────────────────────────────────────────────────────────────

def test_crossover_produces_valid_child():
    a = _g("pa")
    a.exit_params = {"stop_pct": 0.01, "tp_pct": 0.02, "max_bars": 20}
    b = _g("pb")
    b.entry_factors = [Factor("rsi", "oversold", {"period": 14, "threshold": 30})]
    b.species = "mean_reversion"
    rng = random.Random(0)
    child = crossover(a, b, new_id="child1", generation=1, rng=rng)
    assert child.genome_id == "child1"
    assert child.parent_ids == ["pa", "pb"]
    assert child.is_valid(), child.validate()


def test_random_mutation_changes_something():
    g = _g("g0")
    rng = random.Random(123)
    mutated = random_mutation(g, new_id="m1", generation=1, rng=rng)
    assert mutated.genome_id == "m1"
    assert mutated.parent_ids == ["g0"]
    assert mutated.mutation_log  # at least one log entry
    assert mutated.is_valid()


# ── LLM mutations ──────────────────────────────────────────────────────────

def test_apply_mutations_add_filter():
    pop = Population(generation=0, genomes=[_g("p1")])
    mutations = [
        {
            "target_genome_id": "p1",
            "action": "add_filter",
            "detail": {"filter_type": "adx_above", "params": {"period": 14, "threshold": 25}},
            "reasoning": "filter low-trend regime",
        }
    ]
    new, errs = apply_mutations(pop, mutations, generation=1)
    assert errs == []
    assert len(new) == 1
    assert len(new[0].filters) == 1
    assert new[0].filters[0].filter_type == "adx_above"


def test_apply_mutations_new_genome():
    pop = Population(generation=0, genomes=[_g("p1")])
    mutations = [
        {
            "action": "new_genome",
            "detail": {
                "genome_id": "fresh_g",
                "species": "momentum",
                "entry_factors": [
                    {"indicator": "macd", "condition": "macd_golden_cross", "params": {}},
                ],
                "entry_logic": "all_of",
                "exit_type": "pct_stop_tp",
                "exit_params": {"stop_pct": 0.01, "tp_pct": 0.02, "max_bars": 20},
                "size": 0.5,
                "preferred_interval": "15m",
            },
            "reasoning": "inject MACD-momentum strategy",
        }
    ]
    new, errs = apply_mutations(pop, mutations, generation=1)
    assert errs == []
    assert pop.get("fresh_g") is not None


def test_apply_mutations_invalid_target():
    pop = Population(generation=0, genomes=[_g("p1")])
    mutations = [{"target_genome_id": "nonexistent", "action": "add_filter",
                  "detail": {"filter_type": "adx_above", "params": {}}}]
    new, errs = apply_mutations(pop, mutations, generation=1)
    assert len(new) == 0
    assert any("not found" in e for e in errs)


# ── End-to-end engine: 1 gen on synthetic data ─────────────────────────────

@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(42)
    n = 500
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    returns = rng.normal(0.0002, 0.005, size=n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.003, size=n))
    low = close * (1 - rng.uniform(0, 0.003, size=n))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.uniform(100, 1000, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=timestamps,
    )


def test_full_generation_step_end_to_end(synthetic_df):
    config = EvolutionConfig(population_size=10, seed=7)
    pop = Population(generation=0)
    # Build 10 simple genomes
    for i in range(10):
        g = _g(f"init_{i}")
        g.exit_params = {"stop_pct": 0.005 + i * 0.001, "tp_pct": 0.015,
                         "max_bars": 15 + i}
        pop.add(g)

    # Evaluate
    scored = evaluate_population(pop, synthetic_df, config)
    assert len(scored) == 10

    # Build next gen
    next_pop, culled, xover, rmut = build_next_generation(
        scored=scored, config=config, generation=1,
    )
    assert len(next_pop) <= config.population_size
    assert len(next_pop) >= int(config.population_size * 0.7)  # roughly target

    # Build result
    result = build_generation_result(
        generation=1, symbol="TEST", interval="15m",
        scored=scored, next_pop=next_pop, culled=culled,
        crossover_ids=xover, mutation_ids=rmut,
        elapsed=0.5, config=config,
    )
    assert result.generation == 1
    assert result.population_size == len(next_pop)
    assert len(result.top_5) <= 5
    assert isinstance(result.species_distribution, dict)
