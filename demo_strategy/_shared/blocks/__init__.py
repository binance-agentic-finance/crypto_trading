"""Template-layer building blocks — modular, per-block, 3-tier config.

## Layout

    _shared/blocks/
    ├── base.py          — Block base class
    ├── registry.py      — discovery + load_block + load_layer
    ├── contracts.py     — StrategyContext / StrategyDecision / TemplateMeta
    ├── signals/         — ema / rsi / macd / atr / resonance / volume_surge
    ├── scoring/         — hierarchical / verdict_gate
    └── decision/        — direction_vote / position_size

Each `<layer>/<name>.py` defines a Block subclass; each `<layer>/<name>.yaml`
holds its default params. `<layer>/layer.yaml` declares which blocks the
layer enables + per-layer overrides. Strategy `config.yaml` provides the
final override.

## Typical usage from a template

    from demo_strategy._shared.blocks import load_block, load_layer

    # Individual block
    ema = load_block("signals", "ema",
                     overrides={"periods": {"fast": 10, "mid": 30, "slow": 100}})
    ema_out = ema.compute(bars)   # → {ema_fast, ema_mid, ema_slow, direction, ...}

    # All enabled blocks in a layer at once
    signals_layer = load_layer("signals",
                               strategy_overrides={"rsi": {"period": 21}})
    for name, block in signals_layer.items():
        print(name, block.compute(bars))
"""
from .contracts import (
    MarketData, AccountData,
    StrategyContext, StrategyDecision,
    SelectionContext, SelectionDecision,
    TemplateMeta, flat_decision,
)
from .base     import Block
from .registry import load_block, load_layer, registered_blocks

__all__ = [
    # contracts (unchanged)
    "MarketData", "AccountData",
    "StrategyContext", "StrategyDecision",
    "SelectionContext", "SelectionDecision",
    "TemplateMeta", "flat_decision",
    # block plumbing
    "Block", "load_block", "load_layer", "registered_blocks",
]
