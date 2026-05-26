"""Strategy registry — strategy_id → signal factory.

`default_registry` is the module-level singleton. Cases register their signal
builder by calling `default_registry.register(...)` at import time.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from ai_pro_trading_library.library.core.schema import StrategySpec

StrategyFactory = Callable[[StrategySpec, pd.DataFrame], pd.Series]


class StrategyRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, StrategyFactory] = {}

    def register(self, strategy_id: str, factory: StrategyFactory, replace: bool = False) -> None:
        if strategy_id in self._factories and not replace:
            raise ValueError(f"strategy already registered: {strategy_id}")
        self._factories[strategy_id] = factory

    def has(self, strategy_id: str) -> bool:
        return strategy_id in self._factories

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories.keys()))

    def build_signal(self, spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
        try:
            factory = self._factories[spec.strategy_id]
        except KeyError as exc:
            raise KeyError(f"unknown strategy: {spec.strategy_id}") from exc
        return factory(spec, data).fillna(False).astype(bool)


default_registry = StrategyRegistry()
