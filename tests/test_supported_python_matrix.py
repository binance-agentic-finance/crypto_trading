# =============================================================================
# 宣告支援的每一個 Python 版本,都要真的裝得起來
# =============================================================================
# 加 3.13 支援時發現的坑:光把 `requires-python` 的上界從 `<3.13` 改成 `<3.14`
# **是不夠的**,因為依賴裡有 `numpy>=1.24.0,<2.0`,而
#
#     numpy 1.x 在 3.13 上沒有官方 wheel(1.26.4 是 1.x 最後一版,只到 cp312)
#
# 實測 `pip install --only-binary=:all: "numpy<2.0"` 在 3.13 上是
# `No matching distribution found`,能裝的最早是 2.1.0。所以只改 requires-python 會做出
# 一個「classifier 寫著支援 3.13、`pip install` 卻解不出依賴」的包 —— 跟宣告一個
# 從未發布的版本是同一類錯:**宣告的東西不存在**,而且本機只要有快取就照樣是綠的
#(這台機器的 pip 快取裡就有一個自己編出來的 numpy-1.26.4-cp313 wheel,差點看漏)。
#
# 所以這裡把支援矩陣本身變成可執行的斷言:宣告的區間、classifier、以及每個版本實際
# 適用的 numpy 約束,三者必須自洽,而且不能出現「這個 Python 版本沒有任何能裝的 numpy」。
# =============================================================================
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the repo's own venv is 3.11+
    tomllib = pytest.importorskip("tomli")

packaging_specifiers = pytest.importorskip("packaging.specifiers")
packaging_requirements = pytest.importorskip("packaging.requirements")
SpecifierSet = packaging_specifiers.SpecifierSet
Requirement = packaging_requirements.Requirement

REPO = Path(__file__).parents[1]
PYPROJECT = REPO / "pyproject.toml"

#: numpy 1.x 的最後一版,以及它的 wheel 最高覆蓋到哪個 CPython。
#: 這兩個數字是 2026-08 從 PyPI 實際查到的:numpy 1.26.4 的 cp tag 只有
#: cp39 / cp310 / cp311 / cp312。往後 numpy 1.x 不會再出新版,所以這是穩定事實。
LAST_NUMPY_1 = "1.26.4"
FIRST_PYTHON_WITHOUT_NUMPY_1 = (3, 13)
#: 3.13 上最早有 wheel 的 numpy。
FIRST_NUMPY_FOR_313 = "2.1.0"


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _declared_minors() -> list[tuple[int, int]]:
    """`requires-python` 宣告涵蓋的 3.x 版本清單。"""
    spec = _project()["requires-python"]
    floor = re.search(r">=\s*(\d+)\.(\d+)", spec)
    cap = re.search(r"<\s*(\d+)\.(\d+)", spec)
    assert floor and cap, f"requires-python 需要同時有下界與上界,拿到 {spec!r}"
    assert floor.group(1) == cap.group(1) == "3", spec
    return [(3, minor) for minor in range(int(floor.group(2)), int(cap.group(2)))]


def _numpy_requirements() -> list:
    return [Requirement(text) for text in _project()["dependencies"]
            if Requirement(text).name == "numpy"]


def test_the_classifiers_list_exactly_the_declared_versions():
    """classifier 與 `requires-python` 不能各自漂。

    classifier 是使用者在 PyPI 頁面上看到的東西,`requires-python` 才是 pip 真正照著解的。
    兩者不一致時沒有任何錯誤訊息:頁面上寫著支援,`pip install` 卻說版本不符,反之亦然。
    """
    declared = {f"{major}.{minor}" for major, minor in _declared_minors()}
    listed = {match.group(1)
              for classifier in _project()["classifiers"]
              if (match := re.fullmatch(r"Programming Language :: Python :: (3\.\d+)",
                                        classifier))}
    assert listed == declared, (
        f"classifier 寫著 {sorted(listed)},requires-python 宣告的是 {sorted(declared)}；"
        "改了其中一個就要改另一個")


@pytest.mark.parametrize("minor", [minor for _, minor in _declared_minors()])
def test_every_declared_version_has_exactly_one_applicable_numpy_constraint(minor):
    """每個宣告支援的 Python 都要**剛好**適用一條 numpy 約束。

    零條 = 那個版本沒有 numpy 可裝(pip 會裝一個任意版本,或在別處炸);
    兩條 = 兩個區間同時生效,pip 要取交集,而交集可能是空的。
    兩種都不會在 `pip install` 之外的地方留下線索。
    """
    environment = {"python_version": f"3.{minor}"}
    applicable = [r for r in _numpy_requirements()
                  if r.marker is None or r.marker.evaluate(environment)]
    assert len(applicable) == 1, (
        f"python 3.{minor} 適用 {len(applicable)} 條 numpy 約束:"
        f"{[str(r) for r in applicable]}")


@pytest.mark.parametrize("minor", [minor for _, minor in _declared_minors()])
def test_no_declared_version_is_left_without_an_installable_numpy(minor):
    """3.13 及以上不得允許 numpy 1.x —— 那個組合裝不起來。

    這條就是「只改 requires-python 就宣稱支援 3.13」會踩的那個坑的可執行版本。
    """
    environment = {"python_version": f"3.{minor}"}
    applicable = [r for r in _numpy_requirements()
                  if r.marker is None or r.marker.evaluate(environment)]
    specifier = applicable[0].specifier

    if (3, minor) >= FIRST_PYTHON_WITHOUT_NUMPY_1:
        assert not specifier.contains(LAST_NUMPY_1), (
            f"python 3.{minor} 的 numpy 約束是 {specifier},它允許 numpy {LAST_NUMPY_1} —— "
            f"而 numpy 1.x 在 3.{minor} 上沒有 wheel,`pip install` 會解不出來。"
            f"3.13+ 的下界至少要 {FIRST_NUMPY_FOR_313}")
        assert specifier.contains(FIRST_NUMPY_FOR_313), (
            f"python 3.{minor} 的 numpy 約束是 {specifier},卻連 {FIRST_NUMPY_FOR_313} 都不允許,"
            "那這個版本沒有任何 numpy 可裝")
    else:
        # 舊版 Python 維持 1.x:numpy 2.x 改了純量型別晉升與 repr,對這個庫會動到數字,
        # 不該在使用者只是 `pip install -U` 時靜默跨過去。
        assert specifier.contains(LAST_NUMPY_1), (
            f"python 3.{minor} 的 numpy 約束是 {specifier},它排除了 numpy {LAST_NUMPY_1};"
            "舊版 Python 應該維持在 1.x,跨大版本要是明確的決定")


def test_the_running_interpreter_is_inside_the_declared_range():
    """跑測試的這個解釋器本身要在宣告範圍內 —— 否則這份測試結果不代表任何被支援的組合。"""
    current = sys.version_info[:2]
    assert current in _declared_minors(), (
        f"正在用 python {current[0]}.{current[1]} 跑測試,而 requires-python 宣告的是 "
        f"{_project()['requires-python']}")


def test_the_installed_numpy_satisfies_the_constraint_for_this_interpreter():
    """這個環境裡裝著的 numpy,必須落在本解釋器適用的那條約束裡。

    宣告與實際不一致時,測試結果就不能拿來支撐「這個組合可以跑」——
    在 3.13 上裝著一個從原始碼編出來的 numpy 1.x(本機快取就有)正是這種情況。
    """
    numpy = pytest.importorskip("numpy")
    environment = {"python_version": "%d.%d" % sys.version_info[:2]}
    applicable = [r for r in _numpy_requirements()
                  if r.marker is None or r.marker.evaluate(environment)]
    specifier = SpecifierSet(str(applicable[0].specifier))
    assert specifier.contains(numpy.__version__), (
        f"裝著的是 numpy {numpy.__version__},但本解釋器適用的約束是 {specifier}")
