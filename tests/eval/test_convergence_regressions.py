"""Regressions from moving `eval/factor_eval` to `cyqnt_trd/eval` by text replacement.

A blind `factor_eval` -> `cyqnt_trd.eval` rewrite also hit identifiers and paths
that merely contained the substring. None of them raised; each changed an
output contract or pointed a user at something that does not exist.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import cyqnt_trd.eval as ev
from cyqnt_trd.eval import __main__ as cli
from cyqnt_trd.eval import load_bundle
from cyqnt_trd.eval.examples.example_factors import reversal_5d

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "cyqnt_trd" / "eval"


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


@pytest.mark.parametrize("with_diagnostics, scope", [
    (True, "single_factor_evaluation_matrix"),
    (False, "six_gate_research_screen"),
])
def test_evaluation_scope_keeps_its_contract_value(panel, with_diagnostics, scope):
    # was "single_cyqnt_trd.evaluation_matrix": the rename matched inside
    # "factor_evaluation". Downstream readers switch on this string.
    card = ev.evaluate(reversal_5d, panel, name="rev5", with_incremental=False,
                       with_diagnostics=with_diagnostics, n_bootstrap=0,
                       diagnostic_horizons=(1, 3))
    assert card.config["evaluation_scope"] == scope


def test_missing_bundle_error_points_at_files_that_exist(tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        load_bundle(tmp_path / "nope.parquet")
    message = str(info.value)
    assert "cyqnt_trd.eval/" not in message       # a module name used as a directory
    paths = re.findall(r"cyqnt_trd/[\w/]+\.(?:parquet|md)", message)
    assert paths, message
    for rel in paths:
        assert (REPO / rel).exists(), f"error message names a missing file: {rel}"
    assert "cyqnt_trd.eval.snapshot.build_bundle" in message


def test_cli_usage_examples_resolve():
    doc = cli.__doc__
    assert "cd eval" not in doc                     # there is no eval/ directory any more
    specs = re.findall(r"--factor (\S+:\w+)", doc)
    shipped = [s for s in specs if "my_ideas" not in s]
    assert len(shipped) >= 2, doc
    for spec in shipped:
        where = spec.rsplit(":", 1)[0]
        if where.endswith(".py"):
            assert (REPO / where).exists(), f"usage names a missing file: {where}"
            spec = str(REPO / spec)
        assert callable(cli._load_callable(spec))


def test_cli_bootstrap_puts_repo_root_not_the_package_dir_on_sys_path():
    # `HERE.parent` used to be `eval/`; after the move it is `cyqnt_trd/`, which
    # would expose `utils`, `strategies`, ... as top-level modules.
    assert cli.REPO_ROOT == REPO
    assert str(REPO / "cyqnt_trd") not in sys.path


def test_no_dangling_references_to_the_old_layout():
    stale = re.compile(r"cyqnt_trd\.eval/|single_cyqnt_trd|\bfactor_eval\b|cd eval\b")
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if stale.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")
    for doc in ("README.md", "STRUCTURE.md"):
        for n, line in enumerate((PACKAGE / doc).read_text().splitlines(), 1):
            if stale.search(line):
                offenders.append(f"cyqnt_trd/eval/{doc}:{n}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
