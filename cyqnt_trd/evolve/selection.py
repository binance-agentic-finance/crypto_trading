"""Selection: elitism + tournament (SPEC §3.2).

Input: list of (genome, fitness) tuples already evaluated.
Output: (survivors, eliminated_ids).

Default: top 30% elitism + middle 40% tournament-survives + bottom 30% culled.
The 30% slots vacated by culling are filled by crossover + mutation in
``engine.py`` — this module only decides who lives.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from .genome import StrategyGenome


def select_survivors(
    *,
    scored: List[Tuple[StrategyGenome, float]],
    elite_pct: float = 0.30,
    tournament_pct: float = 0.40,
    tournament_k: int = 3,
    rng: random.Random | None = None,
) -> Tuple[List[StrategyGenome], List[str]]:
    """Pick survivors via elitism + tournament. Return (survivors, culled_ids).

    Parameters
    ----------
    scored : list[(genome, fitness)]
    elite_pct : float
        Top N% auto-survive.
    tournament_pct : float
        Middle N% picked via k-tournament. The bottom (1 - elite - tournament)%
        are culled.
    tournament_k : int
        Tournament group size.
    rng : random.Random | None
        Use a deterministic RNG in tests.
    """
    if rng is None:
        rng = random.Random()

    if not scored:
        return [], []

    # Sort descending by fitness (NaN goes last)
    sorted_scored = sorted(
        scored, key=lambda x: (x[1] if x[1] == x[1] else float("-inf")), reverse=True
    )
    n = len(sorted_scored)
    n_elite = max(1, int(round(n * elite_pct)))
    n_tournament = int(round(n * tournament_pct))
    n_cull = n - n_elite - n_tournament
    if n_cull < 0:
        n_cull = 0
        n_tournament = n - n_elite

    # ── Elites ──
    elites = [g for g, _ in sorted_scored[:n_elite]]
    elite_ids = {g.genome_id for g in elites}

    # ── Tournament from non-elites ──
    candidates = [(g, f) for g, f in sorted_scored[n_elite:]]
    survivors_t: List[StrategyGenome] = []
    chosen_ids = set(elite_ids)

    # Run tournaments until we have n_tournament survivors or run out of candidates
    pool = list(candidates)
    rng.shuffle(pool)
    while len(survivors_t) < n_tournament and pool:
        # Sample without replacement up to k
        k = min(tournament_k, len(pool))
        bucket = [pool.pop() for _ in range(k)]
        # Pick the highest fitness from bucket
        bucket.sort(key=lambda x: x[1], reverse=True)
        winner_genome, _ = bucket[0]
        if winner_genome.genome_id not in chosen_ids:
            survivors_t.append(winner_genome)
            chosen_ids.add(winner_genome.genome_id)
        # Put losers back into pool? No — they got their shot. (Standard tournament.)

    # If pool exhausted but quota not met, top up with best remaining non-survivor
    if len(survivors_t) < n_tournament:
        for g, _ in candidates:
            if g.genome_id not in chosen_ids:
                survivors_t.append(g)
                chosen_ids.add(g.genome_id)
                if len(survivors_t) >= n_tournament:
                    break

    survivors = elites + survivors_t

    # Culled = everything else
    survivor_ids = {g.genome_id for g in survivors}
    culled = [g.genome_id for g, _ in sorted_scored if g.genome_id not in survivor_ids]

    return survivors, culled
