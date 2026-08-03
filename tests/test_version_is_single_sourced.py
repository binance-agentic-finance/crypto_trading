# =============================================================================
# 版本只能有一個真相來源
# =============================================================================
# 上架 PyPI 前清點時發現:
#
#   cyqnt_trd/__init__.py   __version__ = "0.1.9.dev2"
#   pyproject.toml          version     = "0.1.11"
#
# 四個發布版的漂移,而沒有任何東西抓到,因為**沒有任何東西比對它們**。
#
# 這比「沒有版本號」更糟:`cyqnt_trd.__version__` 會回一個好幾個月前就不存在於 PyPI 的
# 數字,而它看起來很權威。下游拿它去判斷「我裝的是哪一版、有沒有那個 bug fix」會得到
# 錯的答案,而且不會有任何錯誤訊息。
#
# 同一次清點還發現兩個被全域版本取代誤傷的地方:
#
#   [tool.black] target-version = "0.1.11"
#   [tool.ruff]  target-version = "0.1.11"
#
# 那兩個欄位是 **Python** 版本(py38 / py311),不是套件版本。填錯的後果是靜默的 ——
# 兩個工具會退回從看到的語法去猜目標版本,而那跟 requires-python 宣告的下限不是同一件事。
# 於是 lint 可能放行一段 3.8 跑不起來的語法,而 CI 全綠。
# =============================================================================
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
PYPROJECT = REPO / "pyproject.toml"

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the repo's own venv is 3.11
    tomllib = pytest.importorskip("tomli")


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _dunder_version() -> str:
    text = (REPO / "cyqnt_trd" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    assert match, "cyqnt_trd/__init__.py has no __version__"
    return match.group(1)


def test_the_package_and_the_metadata_agree():
    """The check that was missing. Four releases of drift went unnoticed."""
    assert _dunder_version() == _pyproject()["project"]["version"]


def test_the_importable_value_agrees_too():
    """Read through the import, not just the source text — a lazy ``__getattr__``
    or a re-export could make the file and the module disagree."""
    import cyqnt_trd

    assert cyqnt_trd.__version__ == _pyproject()["project"]["version"]


def test_the_version_is_a_release_not_a_dev_marker():
    """``0.1.9.dev2`` sat in ``__init__`` for four releases. A dev marker is fine
    while iterating and wrong the moment it ships, so it is checked here rather
    than remembered."""
    version = _dunder_version()
    for marker in ("dev", "rc", "a", "b", "+"):
        assert marker not in version.replace(".", "") or marker in ("a", "b"), version
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), (
        "shipping version must be plain MAJOR.MINOR.PATCH, got %r" % version)


# ---------------------------------------------------------------------------
# the two fields a version find-replace damaged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", ["black", "ruff"])
def test_tool_target_version_is_a_python_version(tool):
    """``target-version`` means the PYTHON target for both tools.

    Holding it as the package version is silent: neither tool errors, both fall
    back to inferring the target from the syntax they see, and a lint run can
    then bless syntax the declared ``requires-python`` floor rejects.
    """
    value = _pyproject()["tool"][tool]["target-version"]
    values = value if isinstance(value, list) else [value]
    for item in values:
        assert re.fullmatch(r"py3\d+", item), (
            "%s target-version must look like py38, got %r" % (tool, item))


def test_the_lint_target_matches_the_declared_python_floor():
    """py38 here, ``>=3.8`` there. If they drift, lint is checking a different
    language than the package claims to support."""
    data = _pyproject()
    floor = re.search(r">=\s*(\d+)\.(\d+)", data["project"]["requires-python"])
    assert floor, data["project"]["requires-python"]
    expected = "py%s%s" % (floor.group(1), floor.group(2))

    ruff = data["tool"]["ruff"]["target-version"]
    black = data["tool"]["black"]["target-version"]
    black = black[0] if isinstance(black, list) else black
    assert ruff == expected, (ruff, expected)
    assert black == expected, (black, expected)
