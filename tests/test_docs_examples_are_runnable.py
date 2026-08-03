# =============================================================================
# 文件裡的指令與範例必須真的能跑
# =============================================================================
# 這兩份文件之後要給別人和給 AI 讀,而**壞掉的範例比沒有範例更糟**:讀者會照著做、
# 失敗、然後不知道是自己錯還是文件錯,而多半會歸咎自己。
#
# 上架前清點時,兩份文件裡宣稱的數字有一半是我當下量出來寫進去的 —— 那些數字會隨
# blocks 增減而漂,而漂掉的那次不會有任何東西報錯。所以改成在這裡重算。
#
# 同一次清點也發現版本字串漂了四個發布版而沒人發現(見
# tests/test_version_is_single_sourced.py),原因一樣:沒有任何東西比對文件與程式。
# =============================================================================
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
README = REPO / "README.md"
AGENTS = REPO / "AGENTS.md"


def _own_callables(module) -> list[str]:
    return [name for name in dir(module)
            if not name.startswith("_") and callable(getattr(module, name))
            and getattr(getattr(module, name), "__module__", "") == module.__name__]


# ---------------------------------------------------------------------------
# the counts the docs quote
# ---------------------------------------------------------------------------

def test_the_universe_block_counts_in_the_readme_are_current():
    """README quotes "29 universe.* blocks" and an 8 / 13 split by prefix.

    Recomputed rather than trusted: these move whenever a block lands, and a
    stale count in a shipped README reads as authoritative.
    """
    from cyqnt_trd.blocks import universe

    names = _own_callables(universe)
    augment = [n for n in names if n.startswith("augment_with_")]
    filters = [n for n in names if n.startswith("filter_")]

    text = README.read_text(encoding="utf-8")
    assert "**%d `universe.*` blocks**" % len(names) in text, (
        "README says a different total; actual = %d" % len(names))
    assert "`augment_with_*` (%d)" % len(augment) in text, len(augment)
    assert "`filter_*` (%d)" % len(filters) in text, len(filters)


def test_every_documented_entrypoint_module_exists():
    """A module path in a code block is a promise the reader will type it."""
    text = README.read_text(encoding="utf-8") + AGENTS.read_text(encoding="utf-8")
    modules = set(re.findall(r"python -m (cyqnt_trd[\w.]+)", text))
    assert modules, "no entrypoints documented — did the code blocks move?"

    import importlib.util
    for module in sorted(modules):
        assert importlib.util.find_spec(module) is not None, module


# ---------------------------------------------------------------------------
# the YAML the docs show
# ---------------------------------------------------------------------------

_SPEC_HEADER = '''spec_version: "1.0"
target: standard_bot
strategy: { id: docs_fragment_probe, description: "fragment lifted from the docs" }
run: { mode: backtest }
data: { symbol: BTCUSDT, market_type: futures, primary: { interval: "1h" } }
'''


@pytest.mark.parametrize("doc", [README, AGENTS], ids=["README", "AGENTS"])
def test_the_selection_fragment_validates(doc, tmp_path):
    """The ``selection:`` block each doc shows is lifted verbatim and validated.

    Only the boilerplate header is added, because that is exactly what a reader
    has to supply — so if this passes, following the doc works.
    """
    text = doc.read_text(encoding="utf-8")
    match = re.search(r"```yaml\n(selection:\n.*?)```", text, re.S)
    if match is None:
        pytest.skip("%s shows no selection fragment" % doc.name)

    spec = tmp_path / "fragment.yaml"
    spec.write_text(_SPEC_HEADER + match.group(1), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "cyqnt_trd.standard_bot.yaml_pipeline", "validate", str(spec)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, (
        "the selection fragment in %s does not validate:\n%s\n%s"
        % (doc.name, result.stdout, result.stderr))


def test_the_assumptions_fragment_uses_only_the_closed_key_set():
    """Both docs show a ``strategy.assumptions[]`` example. Its keys are closed,
    and a doc that shows a key the validator rejects teaches the reader a
    mistake."""
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import ASSUMPTION_KEYS

    for doc in (README, AGENTS):
        text = doc.read_text(encoding="utf-8")
        block = re.search(r"assumptions:\n(.*?)\n\n", text, re.S)
        if block is None:
            continue
        shown = set(re.findall(r"^\s*-?\s*(\w+):", block.group(1), re.M))
        stray = shown - set(ASSUMPTION_KEYS)
        assert not stray, "%s shows non-existent assumption key(s): %s" % (doc.name, stray)


# ---------------------------------------------------------------------------
# the invariants the docs assert
# ---------------------------------------------------------------------------

def test_the_fan_out_list_in_the_docs_matches_the_code():
    """Both docs name the four blocks whose step order is load-bearing. A block
    added to the set without a doc update leaves a reader thinking three are
    safe to place anywhere."""
    from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import FAN_OUT_AUGMENTS

    for doc in (README, AGENTS):
        text = doc.read_text(encoding="utf-8")
        for ref in FAN_OUT_AUGMENTS:
            short = ref.split(".")[-1]
            assert short in text, "%s never mentions %s" % (doc.name, ref)


def test_the_naming_contract_the_docs_describe_actually_holds():
    """Docs claim ``augment_*`` cannot narrow and ``filter_*``/``top_*``/
    ``only_*``/``exclude_*`` can. That claim is what the static ordering check
    rests on, so it is asserted against the predicate itself."""
    from cyqnt_trd.blocks import universe
    from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import narrows_the_universe

    for name in _own_callables(universe):
        ref = "universe." + name
        if name.startswith("augment_with_"):
            assert not narrows_the_universe(ref), ref
        elif name.startswith(("filter_", "top_", "only_", "exclude_")):
            assert narrows_the_universe(ref), ref
