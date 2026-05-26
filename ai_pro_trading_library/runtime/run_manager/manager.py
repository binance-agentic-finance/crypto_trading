"""Runtime run registry with optional disk persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ai_pro_trading_library.library.core.schema import (
    RuntimeMode,
    RuntimeRequest,
    StrategySpec,
)


@dataclass
class RunManager:
    runs: dict[str, RuntimeRequest] = field(default_factory=dict)
    root: Path | None = None

    def register(self, run_id: str, request: RuntimeRequest) -> None:
        if run_id in self.runs:
            raise ValueError(f"run already exists: {run_id}")
        self.runs[run_id] = request

    def get(self, run_id: str) -> RuntimeRequest:
        return self.runs[run_id]

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.runs.keys()))

    def save(self) -> Path:
        if self.root is None:
            raise ValueError("RunManager.root must be set to call save()")
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "runs": {
                run_id: {
                    "mode": req.mode.value,
                    "strategy": {
                        "strategy_id": req.strategy.strategy_id,
                        "symbols": list(req.strategy.symbols),
                        "timeframe": req.strategy.timeframe,
                        "entry_rules": list(req.strategy.entry_rules),
                        "exit_rules": list(req.strategy.exit_rules),
                        "risk_profile": dict(req.strategy.risk_profile),
                        "parameters": dict(req.strategy.parameters),
                        "source_case_id": req.strategy.source_case_id,
                    },
                    "data_uri": req.data_uri,
                    "output_dir": str(req.output_dir) if req.output_dir else None,
                    "paper_account": req.paper_account,
                    "dry_run": req.dry_run,
                }
                for run_id, req in self.runs.items()
            }
        }
        target = self.root / "runs.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, root: str | Path) -> "RunManager":
        root_path = Path(root)
        target = root_path / "runs.json"
        if not target.exists():
            return cls(root=root_path)
        payload = json.loads(target.read_text(encoding="utf-8"))
        runs: dict[str, RuntimeRequest] = {}
        for run_id, raw in payload.get("runs", {}).items():
            spec_raw = raw["strategy"]
            spec = StrategySpec(
                strategy_id=spec_raw["strategy_id"],
                symbols=tuple(spec_raw["symbols"]),
                timeframe=spec_raw["timeframe"],
                entry_rules=tuple(spec_raw.get("entry_rules", [])),
                exit_rules=tuple(spec_raw.get("exit_rules", [])),
                risk_profile=spec_raw.get("risk_profile", {}),
                parameters=spec_raw.get("parameters", {}),
                source_case_id=spec_raw.get("source_case_id"),
            )
            runs[run_id] = RuntimeRequest(
                mode=RuntimeMode(raw["mode"]),
                strategy=spec,
                data_uri=raw.get("data_uri"),
                output_dir=Path(raw["output_dir"]) if raw.get("output_dir") else None,
                paper_account=raw.get("paper_account"),
                dry_run=raw.get("dry_run", True),
            )
        return cls(runs=runs, root=root_path)
