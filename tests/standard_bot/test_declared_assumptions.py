# =============================================================================
# strategy.assumptions[] —— spec 承認自己猜了哪一邊
# =============================================================================
# 為什麼需要這個 section
# -----------------------------------------------------------------------------
# 有些需求本身就有兩種讀法,而且沒有任何 block 能替它決定。「成交量一千萬」是
# 1e7 美元的成交額,還是 1e7 顆幣?兩種讀法會篩出不同的名單,差別就是夾在兩個
# 門檻之間的那些幣。系統的預設是**宣告而不是阻斷** —— 取比較可能的讀法、然後
# 說出來 —— 但宣告只有在跟著產出物一起走的時候才算數。
#
# 第一輪端到端示範裡,三個案子的招認全部寫在 YAML 註解與報告裡,而註解在
# load 的時候就被丟掉了:實際發出去的 signal JSON 對此**零揭露**。拿到籃子的
# 人看不到任何一句「這是猜的」,那跟安靜選一邊沒有分別。
#
# 這支測試釘住三件事:
#   1. 這個 section 是封閉形狀 —— 拼錯的 key 是錯誤,不是被忽略的欄位
#   2. 兩條執行路徑都會把它帶到輸出(register 路徑與 bundle_runner 路徑)
#   3. 沒有宣告的 spec 不會憑空長出 assumption(否則這個訊號會失去意義)
#
# 順帶釘住 summary 的評分池:它原本說「top 5 of 727」,而 727 是交給選幣函式的
# 宇宙大小,不是通過篩選、真正參與排序的池子。示範當天真正被評分的是 8 檔,
# 也就是把篩選的廣度誇大了 90 倍,而且就寫在讀者最會掃的那一行。
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyqnt_trd.blocks import strategy as blocks_strategy
from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import run_bundle
from cyqnt_trd.standard_bot.yaml_pipeline.spec import (
    ASSUMPTION_KEYS,
    assumption_warnings,
    load_spec,
    validate_spec,
)

REPO = Path(__file__).parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "universe_derivatives.json"
SPEC_OI = REPO / "docs" / "strategy_yaml_spec" / "example_open_interest_screen.yaml"

ASSUMPTION = {
    "cid": "c3",
    "reading": "「合約持倉 500 萬」讀作 OI 名目價值 (USD)",
    "alternatives": ["合約張數 / 幣的數量 (oi_base)"],
    "basis": "各幣單價差 5 個數量級,張數之間不可比較",
}


def _written(spec: dict, tmp_path) -> Path:
    import yaml

    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    return path


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _spec(*assumptions: dict) -> dict:
    spec = load_spec(str(SPEC_OI))
    if assumptions:
        spec["strategy"]["assumptions"] = [dict(a) for a in assumptions]
    return spec


# ---------------------------------------------------------------------------
# The shape is closed
# ---------------------------------------------------------------------------

def test_a_well_formed_assumption_validates():
    assert validate_spec(_spec(ASSUMPTION))[0] == []


def test_the_minimum_is_a_cid_and_a_reading():
    """``alternatives`` and ``basis`` are optional; the other two are not."""
    assert validate_spec(_spec({"cid": "c1", "reading": "讀作 USD"}))[0] == []


@pytest.mark.parametrize("mutation, fragment", [
    ({"assumption": "讀作 USD"}, "unknown strategy.assumptions[0].assumption"),
    ({"cid": "c1"}, "reading is required"),
    ({"reading": "讀作 USD"}, "cid is required"),
    ({"cid": "c1", "reading": "  "}, "reading is required"),
    ({"cid": "c1", "reading": "x", "alternatives": "一個字串"}, "alternatives must be a list"),
    ({"cid": "c1", "reading": "x", "alternatives": ["", "y"]}, "alternatives must be a list"),
    ({"cid": "c1", "reading": "x", "basis": ""}, "basis must be a non-empty string"),
])
def test_a_malformed_assumption_is_an_error_not_a_shrug(mutation, fragment):
    """A misspelled key is the failure mode this whole section exists to stop.

    Recording the chosen reading under ``assumption:`` instead of ``reading:``
    and having the loader ignore it puts the spec back where it started — a
    silent choice — while looking like a declaration in the source. So the key
    set is closed and every field is checked.
    """
    errors, _warnings = validate_spec(_spec(mutation))
    assert any(fragment in error for error in errors), errors


def test_an_empty_list_is_an_error_rather_than_a_no_op():
    errors, _warnings = validate_spec({**_spec(), "strategy": {
        **_spec()["strategy"], "assumptions": []}})
    assert any("non-empty list" in error for error in errors), errors


def test_two_readings_of_one_condition_is_a_conflict_not_an_assumption():
    errors, _warnings = validate_spec(_spec(
        {"cid": "c3", "reading": "讀作 USD"},
        {"cid": "c3", "reading": "讀作張數"}))
    assert any("repeats cid" in error for error in errors), errors


def test_the_key_set_is_the_one_the_dataset_schema_records():
    """Drift here silently changes what a gold spec is allowed to say."""
    assert ASSUMPTION_KEYS == {"cid", "reading", "alternatives", "basis"}


# ---------------------------------------------------------------------------
# It reaches the artifact, on every path
# ---------------------------------------------------------------------------

def test_the_declaration_rides_out_in_the_signal():
    output = run_bundle(_spec(ASSUMPTION), _fixture())
    signal = output["signals"][0]

    declared = [w for w in signal["warnings"] if w.startswith("assumption")]
    assert len(declared) == 1, signal["warnings"]
    assert ASSUMPTION["reading"] in declared[0]
    assert ASSUMPTION["alternatives"][0] in declared[0], (
        "the reading NOT taken has to be visible too, or the reader cannot tell "
        "what was given up")
    assert ASSUMPTION["basis"] in declared[0]
    assert "assumption_declared" in signal["reason_codes"], (
        "a consumer has to be able to branch on this without parsing prose")


def test_a_spec_that_guessed_nothing_declares_nothing():
    """The signal only means something if its absence means something."""
    signal = run_bundle(_spec(), _fixture())["signals"][0]

    assert not [w for w in signal["warnings"] if w.startswith("assumption")]
    assert "assumption_declared" not in signal["reason_codes"]


def test_both_construction_sites_carry_it(tmp_path):
    """``register_and_run`` and ``bundle_runner`` build the plugin separately.

    Two sites, one spec: a declaration that survives one path and not the other
    is worse than none, because the reader cannot tell which they were handed.
    This is the same divergence that let ``exclude_symbols`` accept a frame on
    one path and raise on the other, so it is asserted rather than assumed.
    """
    spec = _spec(ASSUMPTION)
    expected = assumption_warnings(spec)
    assert expected and expected[0].startswith("assumption (c3)")

    from cyqnt_trd.standard_bot.yaml_pipeline import bundle_runner

    plugin = bundle_runner._build_plugin(spec)
    assert plugin.assumptions == expected

    # The other site is register_from_yaml, which only takes a path, so the
    # spec is written out with the section added rather than passed in.
    captured = {}
    original = blocks_strategy.register_selection

    def spy(strategy_id, selection_fn, **kwargs):
        captured.update(kwargs)
        return original(strategy_id, selection_fn, **kwargs)

    blocks_strategy.register_selection = spy
    try:
        from cyqnt_trd.standard_bot.yaml_pipeline.spec import register_from_yaml

        register_from_yaml(str(_written(spec, tmp_path)))
    finally:
        blocks_strategy.register_selection = original
    assert captured.get("assumptions") == expected


# ---------------------------------------------------------------------------
# The summary counts the pool it ranked, not the market it was handed
# ---------------------------------------------------------------------------

def test_the_summary_names_the_pool_that_was_actually_scored():
    signal = run_bundle(_spec(), _fixture())["signals"][0]

    assert "that passed the screen" in signal["summary"], signal["summary"]
    assert "universe %d" % signal["universe_size"] in signal["summary"]
    assert signal["universe_size"] == 727, (
        "the field keeps its old meaning for consumers that key on it; only the "
        "sentence changed")

    pools = {c["features"].get("scored_pool") for c in signal["candidates"]}
    assert pools == {None}, (
        "scored_pool is plumbing between the selection function and the "
        "summary, not a feature of any symbol; leaking it into features would "
        "put a constant in every candidate's feature vector")


def test_the_pool_is_smaller_than_the_universe_and_that_is_the_point():
    """727 was the sentence's number before this fix; the screen ranked far fewer."""
    signal = run_bundle(_spec(), _fixture())["signals"][0]
    scored = int(signal["summary"].split(" of ")[1].split(" ")[0])
    assert scored < signal["universe_size"]
