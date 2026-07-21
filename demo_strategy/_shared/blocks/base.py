"""Block base class + config loading + registry.

## 一入一出的 Block

一个 block = 一个可独立调用的计算单元：
    - 声明自己的 DEFAULT_PARAMS
    - 有一个 compute(inputs, **params) 方法
    - 每次调用返回一个 dict（不带状态，重复调用等价）

## 三层 config 覆盖顺序（后一层覆盖前一层）

    ① block 自带的 `<block>.yaml`     — 每个 block 目录里 default 参数
    ② layer.yaml                      — 层级默认 (signals/scoring/decision)
    ③ strategy config.yaml            — 策略最终覆盖

`load_block("signals", "ema", overrides={...})` 走完三层 overlay 后返回一个
已配置好的 block 实例；template 直接 `block.compute(bars)` 即可。

## 目录约定

    _shared/blocks/
    ├── base.py            ← 本文
    ├── registry.py        ← Discovery + load_block
    ├── signals/
    │   ├── layer.yaml     ← ② 层级默认
    │   ├── ema.py         ← 类 EmaBlock
    │   ├── ema.yaml       ← ① block 默认
    │   ├── rsi.py / rsi.yaml
    │   ├── macd.py / macd.yaml
    │   └── ...
    ├── scoring/
    │   ├── layer.yaml
    │   ├── hierarchical.py + hierarchical.yaml
    │   └── verdict_gate.py + verdict_gate.yaml
    └── decision/
        ├── layer.yaml
        ├── direction_vote.py + direction_vote.yaml
        └── position_size.py + position_size.yaml
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class Block:
    """Base class every block subclasses.

    Subclass contract:
        class MyBlock(Block):
            NAME  = "my_block"          # required; matches yaml filename
            LAYER = "signals"           # signals | scoring | decision

            DEFAULT_PARAMS: ClassVar[dict] = {"period": 14}

            def compute(self, inputs, **kwargs):
                params = self._merge(kwargs)
                # ... use inputs + params, return dict
    """
    NAME:  ClassVar[str] = ""
    LAYER: ClassVar[str] = ""
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {}

    params: dict[str, Any] = field(default_factory=dict)

    def _merge(self, per_call: dict[str, Any] | None = None) -> dict[str, Any]:
        """Merge class defaults ← instance params ← per-call kwargs."""
        merged = copy.deepcopy(self.DEFAULT_PARAMS)
        merged.update(self.params or {})
        if per_call:
            merged.update(per_call)
        return merged

    def compute(self, inputs: Any, **kwargs) -> dict:
        raise NotImplementedError

    # For introspection (docs / control plane)
    def describe(self) -> dict:
        return {
            "name":            self.NAME,
            "layer":           self.LAYER,
            "default_params":  dict(self.DEFAULT_PARAMS),
            "current_params":  dict(self.params),
        }
