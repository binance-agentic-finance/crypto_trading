"""Case strategy loading helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ai_pro_trading_library.library.core.protocols import StrategySpec


def cases_root() -> Path:
    """Default case root — bootstrap-migrated strategy cases live here.

    The 6 priority cases that bootstrapped the library migration moved
    to `migrated/` (see `docs/migration/cases-relocation.md`). New
    end-to-end pipeline cases ship as a separate skill outside this
    package.
    """
    return Path(__file__).resolve().parent / "migrated"


def load_case_strategy(case_id: str) -> StrategySpec:
    strategy_file = cases_root() / case_id / "strategy.py"
    if not strategy_file.exists():
        raise KeyError(f"unknown or unmigrated case: {case_id}")
    spec = importlib.util.spec_from_file_location(f"{case_id.replace('-', '_')}_strategy", strategy_file)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load strategy module for case: {case_id}")
    spec.loader.exec_module(module)
    strategy = module.build_spec()
    if not isinstance(strategy, StrategySpec):
        raise TypeError(f"{strategy_file} build_spec() did not return StrategySpec")
    return strategy

