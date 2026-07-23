"""LLM mutation applicator.

The LLM (the AI in the CodeMax window) emits a JSON list of mutations after
reading each generation's results. This module applies those mutations
deterministically to the current population.

Mutation actions (matching SPEC §3.4 / PROMPT.md):

    add_factor         — append a new Factor to entry_factors (max 4)
    replace_factor     — replace entry_factors[replace_index]
    drop_factor        — drop entry_factors[drop_index]
    add_filter         — append a new Filter (max 3)
    drop_filter        — drop filters[drop_index]
    change_exit        — replace exit_type + exit_params
    change_size        — set new size (clamped to [0.1, 1.0])
    change_entry_logic — set new entry_logic
    new_genome         — inject a brand-new genome (ignores target_genome_id)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from .genome import (
    ALLOWED_ENTRY_LOGIC,
    ALLOWED_EXIT_TYPES,
    Factor,
    Filter,
    StrategyGenome,
    cap_max_bars,
    infer_species,
)
from .population import Population


def apply_mutations(
    population: Population,
    mutations: List[Dict[str, Any]],
    *,
    generation: int,
) -> Tuple[List[StrategyGenome], List[str]]:
    """Apply LLM-emitted mutations to the population.

    Returns
    -------
    (new_or_modified, errors)
        new_or_modified : list of new/replaced StrategyGenome objects
                          (already inserted into the population)
        errors          : human-readable error strings (skipped mutations)
    """
    new_genomes: List[StrategyGenome] = []
    errors: List[str] = []

    for idx, m in enumerate(mutations):
        action = str(m.get("action", "")).lower()
        target_id = m.get("target_genome_id")
        detail = m.get("detail", {}) or {}
        reasoning = m.get("reasoning", "")

        try:
            if action == "new_genome":
                # Build genome from detail
                if "genome_id" not in detail:
                    detail = {**detail, "genome_id": f"llm_g{generation}_{idx}"}
                detail.setdefault("generation", generation)
                g = StrategyGenome.from_dict(detail)
                cap_max_bars(g)
                errs = g.validate()
                if errs:
                    errors.append(f"mutation[{idx}] new_genome invalid: {errs}")
                    continue
                # Avoid id collision
                while population.get(g.genome_id) is not None:
                    g.genome_id = f"{g.genome_id}_dup"
                population.add(g)
                new_genomes.append(g)
                continue

            # All other actions need an existing target
            if not target_id:
                errors.append(f"mutation[{idx}] action={action} missing target_genome_id")
                continue
            parent = population.get(target_id)
            if parent is None:
                errors.append(f"mutation[{idx}] target {target_id} not found")
                continue

            child = parent.clone(new_id=f"{target_id}_g{generation}m{idx}")
            child.generation = generation
            child.mutation_log.append(f"llm:{action}:{reasoning[:80]}")

            if action == "add_factor":
                if len(child.entry_factors) >= 4:
                    errors.append(f"mutation[{idx}] add_factor: already at 4")
                    continue
                child.entry_factors.append(Factor.from_dict(detail))
            elif action == "replace_factor":
                replace_idx = int(detail.get("replace_index", 0))
                new_f_data = detail.get("new_factor", detail)
                if replace_idx >= len(child.entry_factors):
                    errors.append(f"mutation[{idx}] replace_factor index OOB")
                    continue
                child.entry_factors[replace_idx] = Factor.from_dict(new_f_data)
            elif action == "drop_factor":
                drop_idx = int(detail.get("drop_index", 0))
                if drop_idx >= len(child.entry_factors):
                    errors.append(f"mutation[{idx}] drop_factor index OOB")
                    continue
                if len(child.entry_factors) <= 1:
                    errors.append(f"mutation[{idx}] drop_factor: would empty factor list")
                    continue
                child.entry_factors.pop(drop_idx)
            elif action == "add_filter":
                if len(child.filters) >= 3:
                    errors.append(f"mutation[{idx}] add_filter: already at 3")
                    continue
                child.filters.append(Filter.from_dict(detail))
            elif action == "drop_filter":
                drop_idx = int(detail.get("drop_index", 0))
                if drop_idx >= len(child.filters):
                    errors.append(f"mutation[{idx}] drop_filter index OOB")
                    continue
                child.filters.pop(drop_idx)
            elif action == "change_exit":
                new_type = str(detail.get("exit_type", child.exit_type)).lower()
                if new_type not in ALLOWED_EXIT_TYPES:
                    errors.append(f"mutation[{idx}] change_exit unknown type: {new_type}")
                    continue
                child.exit_type = new_type
                child.exit_params = dict(detail.get("exit_params", child.exit_params))
            elif action == "change_size":
                new_size = float(detail.get("size", child.size))
                child.size = max(0.1, min(1.0, new_size))
            elif action == "change_entry_logic":
                new_logic = str(detail.get("entry_logic", child.entry_logic)).lower()
                if new_logic not in ALLOWED_ENTRY_LOGIC:
                    errors.append(f"mutation[{idx}] unknown entry_logic: {new_logic}")
                    continue
                child.entry_logic = new_logic
                if "entry_score_threshold" in detail:
                    child.entry_score_threshold = int(detail["entry_score_threshold"])
            else:
                errors.append(f"mutation[{idx}] unknown action: {action}")
                continue

            cap_max_bars(child)
            child.species = infer_species(child)
            errs = child.validate()
            if errs:
                errors.append(f"mutation[{idx}] resulting genome invalid: {errs}")
                continue

            population.add(child)
            new_genomes.append(child)

        except Exception as exc:  # noqa: BLE001
            errors.append(f"mutation[{idx}] threw: {type(exc).__name__}: {exc}")

    return new_genomes, errors
