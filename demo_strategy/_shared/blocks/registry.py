"""Block discovery + three-tier YAML overlay + factory.

Discovery: walk `_shared/blocks/<layer>/`, pick up every `.py` module
that defines a `Block` subclass. Register by `(layer, NAME)`.

YAML overlay order (final wins):
    ① block-level default:  `<layer>/<name>.yaml`
    ② layer-level default:  `<layer>/layer.yaml` (keys under `blocks.<name>`)
    ③ user overrides:       `overrides` dict passed to `load_block`

Usage from a template:

    from demo_strategy._shared.blocks import load_block

    ema  = load_block("signals", "ema",  overrides=cfg.get("ema"))
    rsi  = load_block("signals", "rsi",  overrides=cfg.get("rsi"))
    hs   = load_block("scoring", "hierarchical")
    vg   = load_block("scoring", "verdict_gate",
                      overrides={"thresholds": {"STRONG_CANDIDATE": 8}})

    ema_out = ema.compute(bars)
    verdict = vg.compute(hs.compute([t["score"] for t in tiers]))
"""
from __future__ import annotations

import importlib
import inspect
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .base import Block

log = logging.getLogger(__name__)

_BLOCKS_ROOT = Path(__file__).resolve().parent

# In-process cache: (layer, name) → Block class
_REGISTRY: dict[tuple[str, str], type[Block]] = {}


# ─────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────
def _discover_once() -> None:
    """Populate _REGISTRY by importing every layer/*.py that isn't
    a support module. Idempotent — safe to call multiple times."""
    if _REGISTRY:
        return
    for layer_dir in sorted(_BLOCKS_ROOT.iterdir()):
        if not layer_dir.is_dir() or layer_dir.name.startswith("_"):
            continue
        if layer_dir.name not in (
            "signals", "scoring", "decision", "risk", "backtest",
        ):
            continue
        # Recurse: risk/ has sub-layers (risk/stop_loss/*.py etc.)
        for py in sorted(layer_dir.rglob("*.py")):
            if py.name.startswith("_") or py.stem == "layer":
                continue
            rel = py.relative_to(_BLOCKS_ROOT).with_suffix("")
            mod_name = "demo_strategy._shared.blocks." + ".".join(rel.parts)
            try:
                mod = importlib.import_module(mod_name)
            except Exception as e:  # noqa: BLE001
                log.warning("block discovery: failed to import %s: %s", mod_name, e)
                continue
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, Block) and obj is not Block and obj.NAME:
                    key = (obj.LAYER, obj.NAME)
                    if key in _REGISTRY and _REGISTRY[key] is not obj:
                        log.warning("block registry: %s duplicated by %s",
                                    key, obj.__module__)
                        continue
                    _REGISTRY[key] = obj


def registered_blocks() -> dict[tuple[str, str], type[Block]]:
    """Return a shallow copy of the (layer, name) → class map."""
    _discover_once()
    return dict(_REGISTRY)


# ─────────────────────────────────────────────────────────────────────
# YAML overlay
# ─────────────────────────────────────────────────────────────────────
def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        log.warning("bad yaml %s: %s", path, e)
        return {}


def _resolve_params(layer: str, name: str,
                    overrides: dict | None = None) -> dict:
    """Apply the three-tier overlay for one block.

    `name` may be a sub-layer path, e.g. `stop_loss/fixed_pct` for
    blocks under `risk/stop_loss/`. In that case we look up
    `risk/stop_loss/fixed_pct.yaml` and `risk/layer.yaml::blocks.stop_loss.fixed_pct`.
    """
    parts = name.split("/")
    block_yaml = _BLOCKS_ROOT.joinpath(layer, *parts[:-1], f"{parts[-1]}.yaml")
    layer_yaml = _BLOCKS_ROOT / layer / "layer.yaml"

    params: dict[str, Any] = {}

    # ① block default
    params.update(_read_yaml(block_yaml))

    # ② layer default — nested lookup for sub-layer blocks
    layer_blocks = _read_yaml(layer_yaml).get("blocks") or {}
    cur = layer_blocks
    for p in parts:
        cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            break
    _deep_update(params, cur or {})

    # ③ user overrides (from strategy config.yaml)
    if overrides:
        _deep_update(params, overrides)

    return params


def _deep_update(dst: dict, src: dict) -> None:
    """Recursive dict update: {a:{b:1}} + {a:{c:2}} → {a:{b:1,c:2}}."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = deepcopy(v)


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────
def load_block(layer: str, name: str, *,
               overrides: dict | None = None) -> Block:
    """Instantiate `_shared/blocks/<layer>/<name>.py::<class>` with the
    three-tier merged params.

    Raises KeyError if the block isn't registered.
    """
    _discover_once()
    key = (layer, name)
    cls = _REGISTRY.get(key)
    if cls is None:
        avail = sorted(f"{l}/{n}" for (l, n) in _REGISTRY)
        raise KeyError(f"block {layer}/{name} not registered; "
                       f"available: {avail}")
    params = _resolve_params(layer, name, overrides)
    return cls(params=params)


def load_layer(layer: str, *,
               strategy_overrides: dict | None = None) -> dict[str, Block]:
    """Load every block declared enabled in `<layer>/layer.yaml::enabled`
    and return `{block_name: instance}`.

    Strategy overrides shape:
        {block_name: {…params…}, other_block_name: {…}}
    Merged on top of layer defaults for each named block.

    Useful when a template wants "everything my layer enables" instead
    of picking one at a time.
    """
    _discover_once()
    layer_yaml = _BLOCKS_ROOT / layer / "layer.yaml"
    layer_cfg = _read_yaml(layer_yaml)
    enabled = layer_cfg.get("enabled") or []
    strategy_overrides = strategy_overrides or {}
    out: dict[str, Block] = {}
    for name in enabled:
        out[name] = load_block(layer, name,
                               overrides=strategy_overrides.get(name))
    return out
