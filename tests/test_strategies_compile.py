"""Compile-smoke for every strategy module.

A syntax error in a strategy file (e.g. the ``r19_scenario_classifier.py``
``< <`` typo) previously slipped through because nothing imported or compiled
these files during CI. This test byte-compiles every ``*.py`` under
``strategies/`` and ``cyqnt_trd/strategies/`` so such errors fail pytest.

It intentionally uses ``py_compile`` (not ``import``) to avoid executing
module-level side effects or requiring optional runtime dependencies — it only
asserts the source parses/compiles.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STRATEGY_DIRS = ["strategies", "cyqnt_trd/strategies"]


def _collect() -> list[Path]:
    files: list[Path] = []
    for rel in _STRATEGY_DIRS:
        base = _REPO_ROOT / rel
        if base.is_dir():
            files.extend(sorted(base.rglob("*.py")))
    return files


_FILES = _collect()


def test_strategy_dirs_are_non_empty():
    # Guards against a silently-empty glob making the smoke test vacuous.
    assert _FILES, f"no strategy *.py found under {_STRATEGY_DIRS}"


@pytest.mark.parametrize("path", _FILES, ids=[str(p.relative_to(_REPO_ROOT)) for p in _FILES])
def test_strategy_file_compiles(path: Path):
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        pytest.fail(f"{path.relative_to(_REPO_ROOT)} failed to compile:\n{exc}")
