"""Population container — manages a list of genomes + species bookkeeping."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .genome import ALLOWED_SPECIES, StrategyGenome


@dataclass
class Population:
    """A collection of genomes for a single generation."""

    generation: int = 0
    genomes: List[StrategyGenome] = field(default_factory=list)

    # ── Basic ─────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.genomes)

    def __iter__(self):
        return iter(self.genomes)

    def add(self, g: StrategyGenome) -> None:
        self.genomes.append(g)

    def get(self, genome_id: str) -> Optional[StrategyGenome]:
        for g in self.genomes:
            if g.genome_id == genome_id:
                return g
        return None

    def replace(self, genome_id: str, new: StrategyGenome) -> bool:
        for i, g in enumerate(self.genomes):
            if g.genome_id == genome_id:
                self.genomes[i] = new
                return True
        return False

    def remove(self, genome_id: str) -> bool:
        for i, g in enumerate(self.genomes):
            if g.genome_id == genome_id:
                self.genomes.pop(i)
                return True
        return False

    # ── Species bookkeeping ───────────────────────────────────────────
    def species_distribution(self) -> Dict[str, int]:
        counter = Counter(g.species for g in self.genomes)
        # Always include all known species (zeros are informative)
        return {s: counter.get(s, 0) for s in ALLOWED_SPECIES}

    def by_species(self) -> Dict[str, List[StrategyGenome]]:
        out: Dict[str, List[StrategyGenome]] = {s: [] for s in ALLOWED_SPECIES}
        for g in self.genomes:
            out.setdefault(g.species, []).append(g)
        return out

    # ── Validation ────────────────────────────────────────────────────
    def validate(self) -> Dict[str, List[str]]:
        """Return {genome_id: [error, ...]} for invalid genomes."""
        out: Dict[str, List[str]] = {}
        seen: Dict[str, int] = {}
        for g in self.genomes:
            errs = g.validate()
            if g.genome_id in seen:
                errs.append(f"duplicate genome_id (also at index {seen[g.genome_id]})")
            seen[g.genome_id] = len(seen)
            if errs:
                out[g.genome_id] = errs
        return out

    # ── Serialization ─────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "generation": int(self.generation),
            "genomes": [g.to_dict() for g in self.genomes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Population":
        return cls(
            generation=int(data.get("generation", 0)),
            genomes=[StrategyGenome.from_dict(g) for g in data.get("genomes", [])],
        )
