# Unified Trading Skill

Use this skill for trading case discovery, strategy generation, backtest, paper,
live routing, monitoring, stop/resume, and lifecycle explanation.

## Source Of Truth

- New code and docs live in `ai_pro_trading_library`.
- `crypto_trading`, `cyqnt_trd`, `Vibe-Trading` are migration source only.
- Convert user intent into a case or `StrategySpec`, then dispatch to runtime.
- Do not run per-case `run_pipeline.py` as the primary path.

## Executable Entry Points

```python
from ai_pro_trading_library.skills.trading import (
    select_case_by_intent,    # NL string -> case_id
    build_strategy_spec,      # case_id (+ overrides) -> StrategySpec
    dispatch_runtime,         # StrategySpec + mode -> dict (run artifacts)
    summarize_run_status,     # run_dir -> watcher snapshot dict
)
```

## Flow

1. `select_case_by_intent(text)` returns a case_id from the catalog.
2. `build_strategy_spec(case_id, overrides)` returns `StrategySpec`.
3. `dispatch_runtime(spec, mode="backtest"|"paper", output_dir, **kwargs)` runs.
4. Paper mode writes `state.json`, `trades.jsonl`, `events.jsonl`.
5. `summarize_run_status(run_dir)` reads `state.json` into a watcher snapshot.

## Safety

- Live execution requires explicit confirmation (mode="live" not yet enabled).
- Legacy subprocess live execution is not a recommended path.
- Research-only cases must not be upgraded to executable until required data
  adapters and runtime tests exist.
- Modules marked `verify` in the migration matrix are not canonical.
