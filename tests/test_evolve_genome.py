"""Unit tests for cyqnt_trd.evolve.genome and population."""

from __future__ import annotations

import json

import pytest

from cyqnt_trd.evolve.genome import (
    Factor,
    Filter,
    StrategyGenome,
    cap_max_bars,
    infer_species,
)
from cyqnt_trd.evolve.population import Population


# ── Helpers ────────────────────────────────────────────────────────────────

def _good_genome(gid: str = "g1", interval: str = "15m") -> StrategyGenome:
    return StrategyGenome(
        genome_id=gid,
        species="trend",
        entry_factors=[
            Factor("ema", "cross_above", {"fast": 8, "slow": 21}),
        ],
        entry_logic="all_of",
        exit_type="pct_stop_tp",
        exit_params={"stop_pct": 0.01, "tp_pct": 0.025, "max_bars": 20},
        filters=[
            Filter("adx_above", {"period": 14, "threshold": 20}),
        ],
        size=0.5,
        preferred_interval=interval,
        generation=0,
    )


# ── Factor / Filter round-trip ─────────────────────────────────────────────

def test_factor_round_trip():
    f = Factor("rsi", "oversold", {"period": 14, "threshold": 30})
    d = f.to_dict()
    s = json.dumps(d)
    f2 = Factor.from_dict(json.loads(s))
    assert f2.indicator == "rsi"
    assert f2.condition == "oversold"
    assert f2.params == {"period": 14, "threshold": 30}
    assert f2.weight == 1.0


def test_filter_round_trip():
    fl = Filter("adx_above", {"period": 14, "threshold": 25})
    fl2 = Filter.from_dict(fl.to_dict())
    assert fl2.filter_type == "adx_above"
    assert fl2.params == {"period": 14, "threshold": 25}


# ── Genome round-trip ──────────────────────────────────────────────────────

def test_genome_round_trip_preserves_everything():
    g = _good_genome("xyz")
    g.fitness_history = [0.3, 0.5]
    g.mutation_log = ["add_filter:adx_above"]
    g.notes = "trend follower"

    raw = json.dumps(g.to_dict())
    g2 = StrategyGenome.from_dict(json.loads(raw))

    assert g2.genome_id == "xyz"
    assert g2.species == "trend"
    assert len(g2.entry_factors) == 1
    assert g2.entry_factors[0].params == {"fast": 8, "slow": 21}
    assert g2.exit_params["max_bars"] == 20
    assert g2.fitness_history == [0.3, 0.5]
    assert g2.notes == "trend follower"


def test_genome_clone_with_new_id_resets_history():
    g = _good_genome("parent")
    g.fitness_history = [1.2]
    g.mutation_log = ["x"]
    c = g.clone(new_id="child")
    assert c.genome_id == "child"
    assert c.parent_ids == ["parent"]
    assert c.fitness_history == []
    assert c.mutation_log == []
    # original untouched
    assert g.fitness_history == [1.2]


# ── Validation ─────────────────────────────────────────────────────────────

def test_validate_passes_on_good_genome():
    g = _good_genome()
    assert g.is_valid(), g.validate()


def test_validate_catches_unknown_species():
    g = _good_genome()
    g.species = "ghost"
    errs = g.validate()
    assert any("species" in e for e in errs)


def test_validate_catches_too_many_factors():
    g = _good_genome()
    g.entry_factors = [Factor("ema", "cross_above", {"fast": 5, "slow": 10})] * 5
    errs = g.validate()
    assert any("<= 4" in e for e in errs)


def test_validate_catches_max_bars_over_intraday_cap():
    # 15m cap = 24
    g = _good_genome(interval="15m")
    g.exit_params["max_bars"] = 100
    errs = g.validate()
    assert any("max_bars" in e for e in errs)


def test_validate_catches_unknown_indicator_in_factor():
    g = _good_genome()
    g.entry_factors[0] = Factor("foobar", "cross_above", {})
    errs = g.validate()
    assert any("unknown indicator" in e for e in errs)


# ── Helpers ────────────────────────────────────────────────────────────────

def test_cap_max_bars_clamps():
    g = _good_genome(interval="5m")
    g.exit_params["max_bars"] = 999
    cap_max_bars(g)
    assert g.exit_params["max_bars"] == 60  # 5m cap


def test_infer_species_macd_is_momentum():
    g = StrategyGenome(
        genome_id="m1",
        species="trend",  # wrong on purpose
        entry_factors=[Factor("macd", "macd_golden_cross", {})],
    )
    assert infer_species(g) == "momentum"


def test_infer_species_rsi_oversold_is_mean_reversion():
    g = StrategyGenome(
        genome_id="m1",
        species="trend",
        entry_factors=[Factor("rsi", "oversold", {"period": 14, "threshold": 30})],
    )
    assert infer_species(g) == "mean_reversion"


def test_infer_species_breakout():
    g = StrategyGenome(
        genome_id="m1",
        species="trend",
        entry_factors=[Factor("donchian", "breakout", {"period": 20})],
    )
    assert infer_species(g) == "breakout"


# ── Population ─────────────────────────────────────────────────────────────

def test_population_round_trip():
    p = Population(generation=3, genomes=[_good_genome("a"), _good_genome("b")])
    p2 = Population.from_dict(json.loads(json.dumps(p.to_dict())))
    assert p2.generation == 3
    assert len(p2) == 2
    assert {g.genome_id for g in p2} == {"a", "b"}


def test_population_species_distribution():
    p = Population()
    a = _good_genome("a"); a.species = "trend"
    b = _good_genome("b"); b.species = "momentum"
    c = _good_genome("c"); c.species = "trend"
    p.add(a); p.add(b); p.add(c)
    dist = p.species_distribution()
    assert dist["trend"] == 2
    assert dist["momentum"] == 1
    assert dist["mean_reversion"] == 0


def test_population_validation_detects_duplicate_ids():
    p = Population()
    p.add(_good_genome("dup"))
    p.add(_good_genome("dup"))
    errs = p.validate()
    assert "dup" in errs
    assert any("duplicate" in e for e in errs["dup"])


def test_population_get_remove_replace():
    p = Population()
    p.add(_good_genome("a"))
    p.add(_good_genome("b"))
    assert p.get("a") is not None
    assert p.get("zzz") is None

    new_b = _good_genome("b")
    new_b.size = 0.7
    assert p.replace("b", new_b)
    assert p.get("b").size == 0.7

    assert p.remove("a")
    assert p.get("a") is None
    assert len(p) == 1
