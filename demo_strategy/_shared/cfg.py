"""Config loader + CLI overlay helper.

All strategies use the same pattern:
    1. Load `config/config.json` (declarative defaults).
    2. Overlay CLI-provided kwargs on top (e.g. --mode aggressive, --delay).
    3. Activate a named `mode` by copying `modes.<mode_name>` into
       `_active_mode` — decision/execution modules read `_active_mode`.

Keeps run.py trivial and lets the config file speak for itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_cfg(cfg_path: str | Path) -> dict:
    """Parse a JSON config file. Raise a friendly error if malformed."""
    p = Path(cfg_path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON in {p}: {e}") from e


def apply_cli_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Overlay a flat dict of dotted keys onto cfg. Non-destructive
    (returns a new dict tree, doesn't mutate `cfg` in place).

    Example:
        apply_cli_overrides(cfg, {
            "execution.enabled": True,
            "execution.live":    False,
            "execution.mode":    "futures",
        })
    """
    import copy
    out = copy.deepcopy(cfg)
    for dotted, val in overrides.items():
        if val is None:
            continue
        parts = dotted.split(".")
        cur = out
        for k in parts[:-1]:
            cur = cur.setdefault(k, {})
        cur[parts[-1]] = val
    return out


def activate_mode(cfg: dict, mode_name: str) -> dict:
    """Copy `cfg.modes.<mode_name>` into `cfg._active_mode` so downstream
    modules can read one canonical location. Returns cfg for chaining.

    Raise KeyError with a helpful message if the mode doesn't exist.
    """
    modes = cfg.get("modes", {})
    if mode_name not in modes:
        available = list(modes.keys())
        raise KeyError(f"unknown mode {mode_name!r}; available: {available}")
    cfg["_active_mode"] = dict(modes[mode_name])
    cfg["_active_mode"]["name"] = mode_name
    return cfg
