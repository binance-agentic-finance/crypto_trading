"""Bot-dir config loader.

Reads `<bot_dir>/config.yaml` and validates the fields bdp-ai-trading-bot's
Store.add_from_dir requires:  `name`, `sid`, `symbol`, `interval`,
`runtime`, `template_id`. Any missing → raise so registration fails
loudly, not silently mid-run.

`load_bot_config(bot_dir)` returns a plain dict, no dataclass — same
mental model as bdp-bot's Store.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Supervisor-contract fields that must live at the root of config.yaml.
# NB: symbol/interval moved into `data:` block per the 6-section reorg;
# for backward-compat we also accept them at the top level.
REQUIRED_TOP    = ("name", "sid", "runtime", "template_id")
REQUIRED_IN_DATA = ("symbol", "interval")


class ConfigError(ValueError):
    """Raised when config.yaml is missing / malformed / incomplete."""


def load_bot_config(bot_dir: str | Path) -> dict[str, Any]:
    """Load + validate `<bot_dir>/config.yaml`."""
    bot_dir = Path(bot_dir).resolve()
    cfg_path = bot_dir / "config.yaml"
    if not cfg_path.is_file():
        raise ConfigError(f"config.yaml missing at {cfg_path}")

    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"malformed yaml in {cfg_path}: {e}") from e

    # ── validate top-level fields ────────────────────────────────────
    missing_top = [f for f in REQUIRED_TOP if not cfg.get(f)]
    if missing_top:
        raise ConfigError(f"config.yaml missing top-level fields: {missing_top}")

    # symbol/interval either at top (legacy) or under data.* (new)
    data_cfg = cfg.get("data") or {}
    missing_data = [f for f in REQUIRED_IN_DATA
                    if not (cfg.get(f) or data_cfg.get(f))]
    if missing_data:
        raise ConfigError(
            f"config.yaml missing fields (either top-level or under data:): "
            f"{missing_data}")

    # For legacy readers, mirror data.symbol/interval into root when only new.
    for f in REQUIRED_IN_DATA:
        if not cfg.get(f) and data_cfg.get(f):
            cfg[f] = data_cfg[f]

    # Materialize convenience paths derived from bot_dir
    cfg["_bot_dir"]    = str(bot_dir)
    cfg["_state_path"] = str(bot_dir / "state" / "state.json")
    cfg["_logs_dir"]   = str(bot_dir / "logs")

    # Env override — BDP_BOT_<CFG_KEY>__<SUBKEY>=value overrides cfg[key][subkey]
    _apply_env_overlay(cfg)

    return cfg


def _apply_env_overlay(cfg: dict[str, Any]) -> None:
    """Env vars of the form BDP_BOT_<UPPER_DOTTED>=value overlay cfg."""
    prefix = "BDP_BOT_"
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        dotted = key[len(prefix):].replace("__", ".").lower()
        parts = dotted.split(".")
        cur = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
