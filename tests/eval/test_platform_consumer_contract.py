"""The exact surface binance-ai-platform's `factor_evaluate` capability calls (BIA-3341).

`capability-sdk/.../analysis/factor_evaluate/impl.py` does, verbatim:

    from cyqnt_trd.eval import evaluate, capability_factor, load_bundle
    panel = load_bundle()
    card = evaluate(capability_factor(fn, inputs=..., params=..., window=..., output=...),
                    panel, name=..., with_incremental=..., with_diagnostics=False)
    card.verdict / card.blocking / {k: g.status for k, g in card.gates.items()}

cyqnt-trd 0.1.0 through 0.1.15 shipped without `cyqnt_trd.eval`, so that import raised
in the sandbox probe and stopped the strategy host. 434 tests covered eval itself;
none covered what the installed package has to provide to its one external caller.
"""
from __future__ import annotations

import builtins
import fnmatch
import sys
from pathlib import Path

import pytest

from cyqnt_trd.eval import VERDICTS, bundle, matrix

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "cyqnt_trd" / "eval"

# nl2factor emits this shape: one function, keyword-only inputs, returns a dict literal.
# The platform execs it with a whitelist of builtins, so the operator never sees numpy.
OPERATOR_SRC = '''
def momentum_5(*, close, period):
    vals = [v for v in close if v is not None]
    if len(vals) <= period or vals[-period - 1] == 0:
        return {"value": None}
    return {"value": vals[-1] / vals[-period - 1] - 1.0}
'''
SAFE_BUILTINS = ("len", "float", "int", "min", "max", "abs", "sum", "range", "list")


def _operator():
    ns: dict = {}
    safe = {n: getattr(builtins, n) for n in SAFE_BUILTINS}
    exec(compile(OPERATOR_SRC, "<nl2factor_operator>", "exec"), {"__builtins__": safe}, ns)
    return ns["momentum_5"]


def test_platform_call_returns_a_scorecard_it_can_serialise():
    from cyqnt_trd.eval import capability_factor, evaluate, load_bundle

    panel = load_bundle()                       # no argument: the platform has no repo checkout
    card = evaluate(
        capability_factor(_operator(), inputs={"close": "close"}, params={"period": 5},
                          window=10, output="value"),
        panel, name="mom5", with_incremental=False, with_diagnostics=False,
    )
    assert card.verdict in VERDICTS
    gates = {name: g.status for name, g in card.gates.items()}
    assert gates["G0_data"] == "PASS"           # the operator produced a usable panel
    assert all(isinstance(s, str) for s in gates.values())
    assert isinstance(card.blocking, list)


def _package_data_globs() -> list[str]:
    # Only the packaging checks need a TOML reader; the platform call above must
    # still run on the 3.8-3.10 floor where tomllib does not exist.
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover
        tomllib = pytest.importorskip("tomli")
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    return config["tool"]["setuptools"]["package-data"].get("cyqnt_trd.eval", [])


@pytest.mark.parametrize("path", [bundle.DEFAULT_BUNDLE, matrix.CALIBRATION],
                         ids=["default_bundle", "calibration"])
def test_default_inputs_are_declared_as_package_data(path):
    # `data/` and `calibration/` are not packages, so the `"*"` globs do not reach
    # them, and MANIFEST.in only feeds the sdist (a clean wheel build with the
    # MANIFEST lines but without this entry carries 0 of the 4 files): the wheel
    # imports fine and then `load_bundle()` raises FileNotFoundError on the platform.
    rel = path.relative_to(PACKAGE).as_posix()
    assert path.exists()
    assert any(fnmatch.fnmatch(rel, g) for g in _package_data_globs()), (
        f"{rel} is read by default but not shipped; add it to "
        "[tool.setuptools.package-data] \"cyqnt_trd.eval\"")


def test_every_bundled_data_file_is_shipped():
    globs = _package_data_globs()
    for sub in ("data", "calibration"):
        for path in (PACKAGE / sub).iterdir():
            rel = path.relative_to(PACKAGE).as_posix()
            assert any(fnmatch.fnmatch(rel, g) for g in globs), f"{rel} would be left out of the wheel"
