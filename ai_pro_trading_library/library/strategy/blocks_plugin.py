"""Block-strategy plugin adapter — `cyqnt_trd.blocks.strategy` migration.

Intentional divergence: the new repo does not import a global
`SignalPluginRegistry` — block plugins register into the canonical
`library.strategy.default_registry` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import pandas as pd

from ai_pro_trading_library.library.core.protocols import StrategySpec
from ai_pro_trading_library.library.strategy.registry import default_registry


SignalFnOutput = pd.Series | Tuple[pd.Series, pd.Series | None]
SignalFn = Callable[[pd.DataFrame], SignalFnOutput]


@dataclass
class BlockStrategyPlugin:
    """Adapter wrapping a user-supplied signal function."""

    plugin_id: str
    plugin_version: str
    signal_fn: SignalFn

    def build_signal(self, spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
        out = self.signal_fn(data)
        if isinstance(out, tuple):
            return out[0]
        return out


_PENDING: list[Tuple[BlockStrategyPlugin, Callable]] = []
_KNOWN_BLOCK_STRATEGY_IDS: set[str] = set()


def build_plugin(
    strategy_id: str,
    signal_fn: SignalFn,
    *,
    version: str = "block/v1",
) -> BlockStrategyPlugin:
    if not strategy_id or not isinstance(strategy_id, str):
        raise ValueError("strategy_id must be a non-empty string")
    if not callable(signal_fn):
        raise TypeError("signal_fn must be callable")
    return BlockStrategyPlugin(
        plugin_id=strategy_id,
        plugin_version=version,
        signal_fn=signal_fn,
    )


def register(
    strategy_id: str,
    signal_fn: SignalFn,
    *,
    version: str = "block/v1",
) -> None:
    """Register a block-style signal_fn into the canonical default_registry."""
    plugin = build_plugin(strategy_id, signal_fn, version=version)
    factory = lambda spec, data: plugin.build_signal(spec, data)  # noqa: E731
    if not default_registry.has(strategy_id):
        default_registry.register(strategy_id, factory)
    _PENDING.append((plugin, factory))
    _KNOWN_BLOCK_STRATEGY_IDS.add(strategy_id)


def flush_pending_into(registry) -> int:
    """Drain pending block strategies into a registry. Returns count installed."""
    count = 0
    while _PENDING:
        plugin, factory = _PENDING.pop(0)
        try:
            registry.register(plugin.plugin_id, factory)
            count += 1
        except ValueError:
            # already registered — silently skip
            pass
    return count


def is_known_block_strategy(strategy_id: str) -> bool:
    return strategy_id in _KNOWN_BLOCK_STRATEGY_IDS


def registered_block_strategies() -> list:
    return list(_PENDING)


__all__ = [
    "BlockStrategyPlugin",
    "build_plugin",
    "flush_pending_into",
    "is_known_block_strategy",
    "register",
    "registered_block_strategies",
]
