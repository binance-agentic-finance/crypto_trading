"""Session state — manages a single evolution run on disk.

A session looks like::

    sessions/<session_id>/
        config.json                  ← symbol, interval, dates, EvolutionConfig
        data/<SYMBOL>_<int>.parquet  ← cached OHLCV (90 days)
        populations/
            gen_000.json             ← initial population (LLM-injected)
            gen_001.json             ← post-step population (next gen)
            ...
        gen_000_results.json         ← per-gen result for LLM
        gen_001_results.json
        ...
        mutations_001.json           ← LLM-emitted mutations
        ...
        oos_results.json             ← final OOS validation
        champions/                   ← exported .py strategy files
        REPORT.md
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .engine import EvolutionConfig
from .population import Population

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    session_id: str
    symbol: str
    interval: str
    days: int = 90
    is_days: int = 60
    oos_days: int = 30
    created_at: str = ""
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    current_generation: int = 0
    completed_generations: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "symbol": self.symbol,
            "interval": self.interval,
            "days": self.days,
            "is_days": self.is_days,
            "oos_days": self.oos_days,
            "created_at": self.created_at,
            "evolution": asdict(self.evolution),
            "current_generation": self.current_generation,
            "completed_generations": list(self.completed_generations),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionConfig":
        ev = EvolutionConfig(**d.get("evolution", {}))
        return cls(
            session_id=d["session_id"],
            symbol=d["symbol"],
            interval=d["interval"],
            days=int(d.get("days", 90)),
            is_days=int(d.get("is_days", 60)),
            oos_days=int(d.get("oos_days", 30)),
            created_at=d.get("created_at", ""),
            evolution=ev,
            current_generation=int(d.get("current_generation", 0)),
            completed_generations=list(d.get("completed_generations", [])),
        )


# ── Path helpers ───────────────────────────────────────────────────────────

def session_dir(root: Path, session_id: str) -> Path:
    return root / session_id


def init_session(
    *,
    root: Path,
    session_id: str,
    symbol: str,
    interval: str,
    days: int = 90,
    is_days: int = 60,
    oos_days: int = 30,
    population_size: int = 50,
    seed: int = 42,
) -> SessionConfig:
    """Create the session directory + write config.json."""
    sdir = session_dir(root, session_id)
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "data").mkdir(exist_ok=True)
    (sdir / "populations").mkdir(exist_ok=True)
    (sdir / "champions").mkdir(exist_ok=True)

    cfg = SessionConfig(
        session_id=session_id,
        symbol=symbol.upper(),
        interval=interval,
        days=days,
        is_days=is_days,
        oos_days=oos_days,
        created_at=datetime.now(timezone.utc).isoformat(),
        evolution=EvolutionConfig(population_size=population_size, seed=seed),
    )
    write_config(sdir, cfg)
    return cfg


def write_config(sdir: Path, cfg: SessionConfig) -> None:
    (sdir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))


def load_config(sdir: Path) -> SessionConfig:
    raw = json.loads((sdir / "config.json").read_text())
    return SessionConfig.from_dict(raw)


def save_population(sdir: Path, pop: Population) -> Path:
    p = sdir / "populations" / f"gen_{pop.generation:03d}.json"
    p.write_text(json.dumps(pop.to_dict(), indent=2))
    return p


def load_population(sdir: Path, generation: int) -> Population:
    p = sdir / "populations" / f"gen_{generation:03d}.json"
    return Population.from_dict(json.loads(p.read_text()))


def save_generation_result(sdir: Path, generation: int, result: Dict[str, Any]) -> Path:
    p = sdir / f"gen_{generation:03d}_results.json"
    p.write_text(json.dumps(result, indent=2))
    return p


def save_mutations(sdir: Path, generation: int, mutations: List[Dict[str, Any]]) -> Path:
    p = sdir / f"mutations_{generation:03d}.json"
    p.write_text(json.dumps({"mutations": mutations}, indent=2))
    return p


def load_mutations(sdir: Path, generation: int) -> List[Dict[str, Any]]:
    p = sdir / f"mutations_{generation:03d}.json"
    raw = json.loads(p.read_text())
    return raw.get("mutations", [])


def save_oos_results(sdir: Path, results: Dict[str, Any]) -> Path:
    p = sdir / "oos_results.json"
    p.write_text(json.dumps(results, indent=2))
    return p
