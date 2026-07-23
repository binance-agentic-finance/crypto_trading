"""Crossover + random mutation (SPEC §3.3 + §3.4).

These produce new individuals from two parents (crossover) or perturb a
single individual (random mutation). The 15-per-gen LLM mutations are
handled separately in ``engine.py`` via the ``mutations.json`` channel.
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional

from .genome import (
    ALLOWED_FILTERS,
    ALLOWED_INDICATORS,
    Factor,
    Filter,
    MAX_BARS_BY_INTERVAL,
    StrategyGenome,
    cap_max_bars,
    infer_species,
)


# ── Crossover ──────────────────────────────────────────────────────────────

def crossover(
    parent_a: StrategyGenome,
    parent_b: StrategyGenome,
    *,
    new_id: str,
    generation: int,
    rng: Optional[random.Random] = None,
) -> StrategyGenome:
    """Produce a child genome from two parents.

    - entry_factors: union, then random subset of size in [1, min(4, |union|)]
    - entry_logic: random pick from {a, b}
    - exit_type: random pick from {a, b}
    - exit_params: blend (mean for numerics, random pick otherwise)
    - filters: random subset (≤ 3) of union
    - size: mean of parent sizes
    - preferred_interval: random pick from {a, b}
    - species: inferred from resulting factors
    """
    if rng is None:
        rng = random.Random()

    # ── entry factors ──
    union_factors = list(parent_a.entry_factors) + list(parent_b.entry_factors)
    rng.shuffle(union_factors)
    n_factors = rng.randint(1, min(4, len(union_factors))) if union_factors else 1
    child_factors = union_factors[:n_factors] if union_factors else []
    # deep copy each factor so child mutations don't bleed into parents
    child_factors = [copy.deepcopy(f) for f in child_factors]

    # ── entry logic ──
    entry_logic = rng.choice([parent_a.entry_logic, parent_b.entry_logic])
    score_thr = rng.choice([parent_a.entry_score_threshold, parent_b.entry_score_threshold])

    # ── exit ──
    exit_type = rng.choice([parent_a.exit_type, parent_b.exit_type])
    exit_params = _blend_exit_params(
        parent_a.exit_params, parent_b.exit_params, exit_type, rng
    )

    # ── filters ──
    union_filters = list(parent_a.filters) + list(parent_b.filters)
    rng.shuffle(union_filters)
    n_filters = rng.randint(0, min(3, len(union_filters))) if union_filters else 0
    child_filters = [copy.deepcopy(f) for f in union_filters[:n_filters]]

    # ── size ──
    size = (parent_a.size + parent_b.size) / 2.0
    size = max(0.2, min(0.8, size))

    # ── interval ──
    interval = rng.choice([parent_a.preferred_interval, parent_b.preferred_interval])

    child = StrategyGenome(
        genome_id=new_id,
        species=parent_a.species,  # tentative; infer below
        entry_factors=child_factors,
        entry_logic=entry_logic,
        entry_score_threshold=int(score_thr),
        exit_type=exit_type,
        exit_params=exit_params,
        filters=child_filters,
        size=float(size),
        preferred_interval=interval,
        generation=generation,
        parent_ids=[parent_a.genome_id, parent_b.genome_id],
    )
    child.species = infer_species(child)
    cap_max_bars(child)
    return child


def _blend_exit_params(
    a: dict, b: dict, exit_type: str, rng: random.Random
) -> dict:
    """Blend numeric exit params; for discrete keys pick from {a, b}."""
    out = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        va = a.get(k); vb = b.get(k)
        if va is None:
            out[k] = vb
        elif vb is None:
            out[k] = va
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            avg = (float(va) + float(vb)) / 2.0
            out[k] = int(round(avg)) if isinstance(va, int) and isinstance(vb, int) else avg
        else:
            out[k] = rng.choice([va, vb])
    # ensure required keys per exit_type exist
    if exit_type == "pct_stop_tp":
        out.setdefault("stop_pct", 0.01)
        out.setdefault("tp_pct", 0.02)
        out.setdefault("max_bars", 20)
    elif exit_type == "atr_stop_tp":
        out.setdefault("stop_mult", 2.0)
        out.setdefault("tp_mult", 3.0)
        out.setdefault("max_bars", 20)
    elif exit_type == "time_only":
        out.setdefault("max_bars", 12)
    elif exit_type == "ma_cross_exit":
        out.setdefault("period", 21)
        out.setdefault("max_bars", 24)
    return out


# ── Random mutation ────────────────────────────────────────────────────────

NUMERIC_PARAM_KEYS = {
    "fast", "slow", "period", "k", "d", "k_period", "d_period", "smooth_k",
    "signal", "threshold", "stop_pct", "tp_pct", "stop_mult", "tp_mult",
    "max_bars", "multiplier", "lookback", "low", "high", "std",
    "bandwidth_threshold", "ma_period", "atr_period", "short", "mid", "long",
    "window", "percentile",
}


def random_mutation(
    g: StrategyGenome,
    *,
    new_id: Optional[str] = None,
    generation: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> StrategyGenome:
    """Produce a randomly-mutated copy of a genome.

    Possible operations (one per call):
      - perturb a numeric param ±20%
      - add a random filter (if < 3 filters)
      - drop a filter
      - replace one entry factor with a fresh one of the same indicator
    """
    if rng is None:
        rng = random.Random()

    child = g.clone(new_id=new_id or f"{g.genome_id}_m{rng.randint(0, 9999)}")
    if generation is not None:
        child.generation = generation

    op_pool = ["perturb_param"]
    if len(child.filters) < 3:
        op_pool.append("add_filter")
    if child.filters:
        op_pool.append("drop_filter")
    if child.entry_factors:
        op_pool.append("perturb_factor_param")

    op = rng.choice(op_pool)
    if op == "perturb_param":
        _perturb_random_numeric(child, rng)
        child.mutation_log.append("perturb_random_param")
    elif op == "add_filter":
        new_f = _random_filter(rng)
        child.filters.append(new_f)
        child.mutation_log.append(f"add_filter:{new_f.filter_type}")
    elif op == "drop_filter":
        idx = rng.randrange(len(child.filters))
        dropped = child.filters.pop(idx)
        child.mutation_log.append(f"drop_filter:{dropped.filter_type}")
    elif op == "perturb_factor_param":
        idx = rng.randrange(len(child.entry_factors))
        f = child.entry_factors[idx]
        _perturb_factor_params(f, rng)
        child.mutation_log.append(f"perturb_factor:{idx}")

    cap_max_bars(child)
    return child


def _perturb_random_numeric(g: StrategyGenome, rng: random.Random) -> None:
    """Pick one numeric param somewhere in the genome and ±20% it."""
    # Targets: exit_params, then per-factor params, then per-filter params
    targets = []
    for k, v in g.exit_params.items():
        if isinstance(v, (int, float)) and k in NUMERIC_PARAM_KEYS:
            targets.append(("exit", k, v))
    for i, f in enumerate(g.entry_factors):
        for k, v in f.params.items():
            if isinstance(v, (int, float)) and k in NUMERIC_PARAM_KEYS:
                targets.append(("factor", i, k, v))
    for i, fl in enumerate(g.filters):
        for k, v in fl.params.items():
            if isinstance(v, (int, float)) and k in NUMERIC_PARAM_KEYS:
                targets.append(("filter", i, k, v))

    if not targets:
        return

    target = rng.choice(targets)
    factor = 1.0 + rng.uniform(-0.2, 0.2)
    if target[0] == "exit":
        _, k, v = target
        new_v = v * factor
        g.exit_params[k] = (
            max(1, int(round(new_v))) if isinstance(v, int) else max(0.0001, float(new_v))
        )
    elif target[0] == "factor":
        _, i, k, v = target
        new_v = v * factor
        g.entry_factors[i].params[k] = (
            max(1, int(round(new_v))) if isinstance(v, int) else max(0.0001, float(new_v))
        )
    elif target[0] == "filter":
        _, i, k, v = target
        new_v = v * factor
        g.filters[i].params[k] = (
            max(1, int(round(new_v))) if isinstance(v, int) else max(0.0001, float(new_v))
        )


def _perturb_factor_params(f: Factor, rng: random.Random) -> None:
    for k, v in list(f.params.items()):
        if isinstance(v, (int, float)) and k in NUMERIC_PARAM_KEYS:
            factor = 1.0 + rng.uniform(-0.2, 0.2)
            new_v = v * factor
            f.params[k] = (
                max(1, int(round(new_v))) if isinstance(v, int) else max(0.0001, float(new_v))
            )


# ── Random filter generator ────────────────────────────────────────────────

def _random_filter(rng: random.Random) -> Filter:
    """Generate a random filter from the allowed vocabulary."""
    ftype = rng.choice(list(ALLOWED_FILTERS))
    params: dict
    if ftype == "adx_above":
        params = {"period": rng.choice([10, 14, 20]), "threshold": rng.choice([15, 20, 25, 30])}
    elif ftype == "atr_above_pct":
        params = {"period": 14, "threshold": rng.choice([0.002, 0.003, 0.005, 0.008])}
    elif ftype == "atr_above_percentile":
        params = {"period": 14, "window": 100, "percentile": rng.choice([0.3, 0.4, 0.5])}
    elif ftype in ("volume_above", "volume_above_ma"):
        params = {"period": rng.choice([10, 20, 50]), "multiplier": rng.choice([1.0, 1.2, 1.5])}
    elif ftype == "hour_range":
        starts = [(0, 8), (8, 16), (12, 20), (16, 24), (20, 4)]
        s, e = rng.choice(starts)
        params = {"start": s, "end": e}
    elif ftype == "ma_slope_positive":
        params = {"period": rng.choice([20, 50, 100]), "lookback": rng.choice([3, 5, 10])}
    elif ftype == "bbw_above":
        params = {"period": 20, "std": 2.0, "threshold": rng.choice([0.015, 0.02, 0.03])}
    elif ftype == "rsi_in_range":
        params = {"period": 14, "low": 40, "high": 60}
    elif ftype == "ema_alignment":
        params = {"short": 8, "mid": 21, "long": 55}
    elif ftype == "price_above_ma":
        params = {"period": rng.choice([50, 100, 200]), "ma_type": "ema"}
    elif ftype == "price_below_ma":
        params = {"period": rng.choice([50, 100, 200]), "ma_type": "ema"}
    else:
        params = {}
    return Filter(filter_type=ftype, params=params)
