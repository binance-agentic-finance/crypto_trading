# =============================================================================
# 資料集的可重derive性 —— 重跑一次不可以改變任何東西
# =============================================================================
# 這份資料集存在的理由是「之後要訓練跟檢驗」。要能檢驗,就必須能重新產製並且
# 跟舊的對得起來。實測發現兩件事都不成立:
#
#   1. case_id 是 ULID,每呼叫一次就是一個新值。同一批對話重新跑一次標註規則,
#      五個案子拿到五個新 id,而 attempts.case_id 與 gaps.example_case_ids 都
#      指著舊的 —— 全部變成孤兒。
#
#   2. _drop_existing 只看 text_sha256,而**內部紀錄根本沒有這個欄位**
#      (它存的是 user_text_raw)。所以內部檔從來沒去重過:每跑一次追加五列,
#      清點時已經是 5 個對話 40 列,而且長得就像一份 40 案的資料集。
#
# 兩個都是安靜的。這支測試把它們變成吵的。
# =============================================================================
from __future__ import annotations

import json

from tools.nl2yaml import ingest_demo, schema as S


def test_the_case_id_is_decided_by_the_text_not_by_the_clock():
    text_sha = S.sha256_hex("幫我選五個 funding 為負的幣")
    assert S.case_id_for(text_sha) == S.case_id_for(text_sha)
    assert S.case_id_for(text_sha) != S.case_id_for(S.sha256_hex("另一段話"))
    assert S.CASE_ID_RE.match(S.case_id_for(text_sha))


def test_a_minted_id_is_still_accepted_for_records_with_no_source_text():
    """``new_case_id`` keeps its place; only text-derived records changed."""
    assert S.ULID_RE.match(S.new_case_id())


def test_the_shipped_records_use_the_content_addressed_form():
    """And their ids really are the hash of their own text, not just the shape."""
    from pathlib import Path

    path = Path(__file__).parents[2] / "tools" / "nl2yaml" / "dataset" / "cases.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()
            if line.strip()]
    assert rows, "cases.jsonl is empty"
    for row in rows:
        assert row["case_id"] == S.case_id_for(row["text_sha256"]), row["case_id"]


def test_one_conversation_is_one_row_however_many_times_ingest_runs(tmp_path):
    """The dedup key has to be found in BOTH record shapes.

    Written against the real failure: the internal record has no
    ``text_sha256``, so a lookup of that key alone matched nothing there and the
    file grew by five rows a run while the public one stayed at five. Both
    shapes are fed here, which is the only version of this test that would have
    caught it.
    """
    text = "找三個月內漲了一倍的幣"
    sha = S.sha256_hex(text)
    public = tmp_path / "cases.jsonl"
    internal = tmp_path / "cases_internal.jsonl"
    public.write_text(json.dumps({"case_id": "case_" + sha[:16],
                                  "text_sha256": sha}) + "\n", "utf-8")
    internal.write_text(json.dumps({"case_id": "case_" + sha[:16],
                                    "user_text_raw": text}) + "\n", "utf-8")

    dropped_public = ingest_demo._drop_existing(public, {sha})
    dropped_internal = ingest_demo._drop_existing(internal, {sha})

    assert dropped_public == 1
    assert dropped_internal == 1, (
        "the internal record spells the text as user_text_raw; a dedup that "
        "only knows text_sha256 never removes anything from it")
    assert public.read_text("utf-8").strip() == ""
    assert internal.read_text("utf-8").strip() == ""


def test_a_row_for_another_conversation_survives(tmp_path):
    """The mutation side: dedup must replace, not truncate."""
    keep = S.sha256_hex("留下來的那一段")
    drop = S.sha256_hex("要被換掉的那一段")
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"case_id": "case_" + keep[:16], "text_sha256": keep}) + "\n"
        + json.dumps({"case_id": "case_" + drop[:16], "text_sha256": drop}) + "\n",
        "utf-8")

    assert ingest_demo._drop_existing(path, {drop}) == 1

    left = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    assert [r["text_sha256"] for r in left] == [keep]


def test_reporting_asks_are_not_labelled_ambiguous():
    """"列出幣名跟現價" has nothing ambiguous about it.

    The blanket ``vague -> UNDEFINED`` rule could not tell "no computable
    criterion" from "not a criterion at all", and labelled ten of the demo
    corpus's output-format asks as undeclared readings. Under the G1e rule that
    demands a ``strategy.assumptions[]`` entry per ambiguous condition, that
    turns the section into boilerplate — one pasted assumption per spec, which
    is how a real declaration stops being read.
    """
    conds = ingest_demo._conditions({"conditions": [
        {"cid": "c1", "subject": "output_field_price", "measurability": "vague",
         "scope": "universe"},
        {"cid": "c2", "subject": "report_structure_sequence", "measurability": "vague",
         "scope": "universe"},
        {"cid": "c3", "subject": "market_sentiment_overall", "measurability": "vague",
         "scope": "universe"},
    ]})
    by_cid = {c.cid: c for c in conds}

    assert by_cid["c1"].scope is S.Scope.REPORTING
    assert by_cid["c1"].ambiguity_type is None
    assert by_cid["c2"].ambiguity_type is None
    assert by_cid["c3"].ambiguity_type is S.AmbiguityType.UNDEFINED, (
        "a genuinely undefinable criterion keeps its label — the reclassification "
        "is meant to be narrow, because stripping the label off a real ambiguity "
        "switches the gate off for it in silence")
