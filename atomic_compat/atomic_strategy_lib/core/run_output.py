"""shim — atomic.core.run_output.

This is a verbatim copy of atomic_strategy_lib.core.run_output. It uses
only stdlib (no atomic / cyqnt_trd dependencies), so we host it inline
in the shim package to avoid placing IO helpers inside cyqnt_trd's
main namespace.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text to a file atomically via tmp-then-rename.

    Why: pipeline artifacts and state files (positions, signal logs) hold
    workflow-critical data. A crash mid-write with plain ``write_text`` leaves
    a truncated file that breaks the next load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


@dataclass(frozen=True)
class RunOutput:
    """Resolved output paths for one pipeline run."""

    path: Path
    latest_path: Optional[Path]
    runs_dir: Optional[Path]
    run_id: Optional[str]


def prepare_run_output(
    requested_out: str,
    *,
    default_out: str = "tmp/pipeline_result.json",
    resolve_path: Optional[Callable[[str], Path]] = None,
) -> RunOutput:
    """Resolve output paths.

    If the caller uses the default output path, write the durable artifact
    into a unique run directory and mirror it to ``tmp/latest/...``. If the
    caller explicitly supplies ``--out``, honor that path exactly and skip
    mirroring.
    """
    resolver = resolve_path or Path
    requested = resolver(str(requested_out))
    default = resolver(default_out)

    if str(requested_out) != default_out and requested != default:
        return RunOutput(path=requested, latest_path=None, runs_dir=None, run_id=None)

    run_id = _new_run_id()
    runs_dir = default.parent / "runs"
    run_path = runs_dir / run_id / default.name
    latest_path = default.parent / "latest" / default.name
    return RunOutput(
        path=run_path,
        latest_path=latest_path,
        runs_dir=runs_dir,
        run_id=run_id,
    )


def write_json_output(
    output,
    data: Any,
    *,
    keep_runs: int = 20,
    **json_kwargs: Any,
) -> None:
    """Write JSON to the run artifact and update the latest mirror."""
    if isinstance(output, RunOutput):
        run_output = output
        target = run_output.path
    else:
        run_output = None
        target = Path(output)

    text = json.dumps(data, **json_kwargs)
    atomic_write_text(target, text)

    if run_output and run_output.latest_path:
        atomic_write_text(run_output.latest_path, text)

    if run_output and run_output.runs_dir and keep_runs > 0:
        prune_run_dirs(run_output.runs_dir, keep=keep_runs)


def prune_run_dirs(runs_dir: Path, *, keep: int = 20) -> None:
    """Keep the newest run directories by lexical run id, deleting older runs."""
    if keep <= 0 or not runs_dir.exists():
        return

    run_dirs = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    stale = run_dirs[:-keep]
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{os.getpid()}"


__all__ = [
    "RunOutput",
    "atomic_write_text",
    "prepare_run_output",
    "prune_run_dirs",
    "write_json_output",
]
