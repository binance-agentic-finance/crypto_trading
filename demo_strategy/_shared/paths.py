"""sys.path bootstrap + workspace path helpers.

The two demo strategies both import from `atomic_strategy_lib` (sibling of
`demo_strategy/`). Rather than requiring a pip install, each strategy's
`run.py` calls `bootstrap_sys_path()` once at startup — that prepends the
atomic_compat directory to `sys.path` so `from atomic_strategy_lib.…` works
without any environment setup.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap_sys_path() -> Path:
    """Prepend the parent atomic_compat/ dir to sys.path, return its path.

    Layout expected:
        crypto_trading/
            atomic_compat/
                atomic_strategy_lib/       ← the library
            demo_strategy/
                _shared/paths.py           ← this file
                <strategy>/run.py          ← the caller

    Walks up from this file until it finds a sibling `atomic_compat/` and
    prepends it to sys.path. If not found, raises RuntimeError.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        cand = parent.parent / "atomic_compat"
        if cand.is_dir() and (cand / "atomic_strategy_lib").is_dir():
            p = str(cand)
            if p not in sys.path:
                sys.path.insert(0, p)
            return cand
    raise RuntimeError(
        "atomic_compat/ not found — expected sibling of demo_strategy/"
    )


def workspace_dir(strategy_name: str) -> Path:
    """Return `~/.openclaw/workspace/<strategy_name>/`, creating it if
    needed. All strategies write their pipeline_result.json here so a
    downstream tool (or the LLM template) can find output in a known
    location.

    Honors `OPENCLAW_WORKSPACE` env var if the user overrides.
    """
    base = Path(os.environ.get("OPENCLAW_WORKSPACE",
                               str(Path.home() / ".openclaw" / "workspace")))
    out = base / strategy_name
    out.mkdir(parents=True, exist_ok=True)
    return out
