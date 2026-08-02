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


def _reviewed_conditions():
    """One wholly synthetic legacy condition plus its capability ruling."""
    conditions = ingest_demo._conditions({"conditions": [
        {
            "cid": "c1",
            "polarity": "require",
            "subject": "quote_volume_24h",
            "operator": ">",
            "value": 1,
            "unit": "usd",
            "scope": "universe",
            "measurability": "quantified",
        },
    ]})
    capabilities = [S.CapabilityEntry(
        cid="c1", verdict=S.CapabilityVerdict.SUPPORTED,
    )]
    return conditions, capabilities


def _frozen_conditions():
    """The safe evaluator input, intentionally distinct from reviewer quotes."""
    return [{
        "id": "c1",
        "subject": "quote_volume_24h",
        "operator": "compare",
        "scope": "cross_section",
        "value": {"op": ">", "threshold": 1},
        "quantified": True,
    }]


def _safe_receipt(*, yaml_text, user_text, conditions_sha256, bundle_sha256):
    """Minimal fully synthetic v1 receipt accepted by the migration boundary."""
    return {
        "schema": ingest_demo.frozen_evaluate.RECEIPT_SCHEMA,
        "nl_sha256": S.sha256_hex(user_text),
        "yaml_sha256": S.sha256_hex(yaml_text),
        "conditions_sha256": conditions_sha256,
        "bundle_sha256": bundle_sha256,
        "gate_status": "passed",
        "failed_gate": None,
        "gate_statuses": {gate: "passed" for gate in ("G1a", "G1b", "G1c", "G1d", "G1e")},
        "failure_signature": "passed",
        "draft_valid": True,
        "runtime_valid": True,
        "gold_clean": True,
        "conditions": {
            "requested_count": 1,
            "evaluated_count": 1,
            "verdict_counts": {"satisfied": 1, "violated": 0,
                               "unverifiable": 0, "not_expressed": 0},
            "silently_proxied_count": 0,
            "undisclosed_assumption_count": 0,
        },
        "signal_batch_sha256": "a" * 64,
        "signal_count": 1,
        "selection_candidate_count": 1,
    }


def _gold_verdict():
    return {"condition_verdicts": [{"cid": "c1", "verdict": "satisfied"}]}


def _receipt_context(tmp_path):
    cid = "synthetic_case"
    yaml_text = "strategy:\n  id: synthetic\n"
    user_text = "invented request for a synthetic liquidity screen"
    conditions, capabilities = _reviewed_conditions()
    frozen_path = ingest_demo._frozen_conditions_path(tmp_path, cid)
    frozen_path.write_text(json.dumps(_frozen_conditions()), encoding="utf-8")
    evidence = ingest_demo._frozen_conditions_evidence(frozen_path)
    assert evidence is not None
    conditions_sha256, _ = evidence
    return {
        "cid": cid,
        "yaml_text": yaml_text,
        "user_text": user_text,
        "conditions": conditions,
        "capabilities": capabilities,
        "frozen_path": frozen_path,
        # It is an injected synthetic fixture: this test never opens a real
        # market bundle or a real conversation export.
        "bundle_sha256": "b" * 64,
        "conditions_sha256": conditions_sha256,
    }


def _state(tmp_path, context):
    return ingest_demo._receipt_state(
        ingest_demo._receipt_path(tmp_path, context["cid"]),
        frozen_conditions_path=context["frozen_path"],
        yaml_text=context["yaml_text"],
        user_text=context["user_text"],
        conditions=context["conditions"],
        pinned_bundle_sha256=context["bundle_sha256"],
    )


def test_yaml_and_review_without_a_receipt_cap_gold_at_intent_reconciled(tmp_path):
    context = _receipt_context(tmp_path)

    assert _state(tmp_path, context) == "missing"
    gold = ingest_demo._gold(
        _gold_verdict(), context["yaml_text"], context["conditions"],
        context["capabilities"], receipt_verified=False,
    )

    assert gold["gold_verification_level"] is S.GoldVerificationLevel.INTENT_RECONCILED
    assert gold["gold_verification_level"] < S.GoldVerificationLevel.EXECUTED_ON_PINNED_BUNDLE
    assert [item.checked_by for item in gold["gold_condition_verdicts"]] == [
        S.CheckedBy.HUMAN,
    ]


def test_exact_safe_receipt_promotes_only_to_pinned_execution_level(tmp_path):
    context = _receipt_context(tmp_path)
    receipt = _safe_receipt(
        yaml_text=context["yaml_text"],
        user_text=context["user_text"],
        conditions_sha256=context["conditions_sha256"],
        bundle_sha256=context["bundle_sha256"],
    )
    receipt_path = ingest_demo._receipt_path(tmp_path, context["cid"])
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert receipt_path.name == "synthetic_case_receipt.json"
    assert context["frozen_path"].name == "synthetic_case_frozen_conditions.json"
    assert _state(tmp_path, context) == "verified"
    gold = ingest_demo._gold(
        _gold_verdict(), context["yaml_text"], context["conditions"],
        context["capabilities"], receipt_verified=True,
    )

    assert gold["gold_verification_level"] is S.GoldVerificationLevel.EXECUTED_ON_PINNED_BUNDLE
    assert gold["gold_verification_level"] < S.GoldVerificationLevel.ALL_CHECKABLE_CONDITIONS_SATISFIED
    assert [item.checked_by for item in gold["gold_condition_verdicts"]] == [
        S.CheckedBy.HUMAN,
    ]


def test_receipt_requires_exact_legacy_condition_semantics(tmp_path):
    """A matching count/hash cannot certify a different reviewed predicate."""
    context = _receipt_context(tmp_path)
    base = _frozen_conditions()[0]
    # Each frozen file is valid evaluator input and its receipt hash is updated
    # to match it.  The only failing boundary is therefore the legacy <-> frozen
    # semantic link, which must reject ids, subjects, polarity, scope,
    # thresholds, and operators that differ even though every variant still has
    # one condition.
    variants = (
        {**base, "id": "c2", "subject": "market_cap"},
        {**base, "subject": "market_cap"},
        {**base, "polarity": "exclude"},
        {**base, "scope": "per_candidate_series"},
        {**base, "value": {"op": ">", "threshold": 2}},
        {**base, "value": {"op": ">=", "threshold": 1}},
    )
    receipt_path = ingest_demo._receipt_path(tmp_path, context["cid"])
    for frozen in variants:
        context["frozen_path"].write_text(json.dumps([frozen]), encoding="utf-8")
        evidence = ingest_demo._frozen_conditions_evidence(context["frozen_path"])
        assert evidence is not None
        conditions_sha256, _ = evidence
        receipt_path.write_text(json.dumps(_safe_receipt(
            yaml_text=context["yaml_text"],
            user_text=context["user_text"],
            conditions_sha256=conditions_sha256,
            bundle_sha256=context["bundle_sha256"],
        )), encoding="utf-8")

        assert _state(tmp_path, context) == "invalid"


def test_g1e_failed_receipt_does_not_relabel_legacy_verdict_as_executed(tmp_path):
    """Level 4 local provenance is not a per-condition execution certificate."""
    context = _receipt_context(tmp_path)
    receipt = _safe_receipt(
        yaml_text=context["yaml_text"],
        user_text=context["user_text"],
        conditions_sha256=context["conditions_sha256"],
        bundle_sha256=context["bundle_sha256"],
    )
    # G1a--G1d completed, so this still records an honest local execution claim
    # at level 4; G1e then failed.  The legacy human label must not become an
    # executed satisfied condition merely because an adjacent receipt exists.
    receipt.update({
        "gate_status": "condition_violated",
        "failed_gate": "G1e",
        "failure_signature": "condition_violated",
        "gold_clean": False,
    })
    receipt["gate_statuses"]["G1e"] = "failed"
    ingest_demo._receipt_path(tmp_path, context["cid"]).write_text(
        json.dumps(receipt), encoding="utf-8")

    assert _state(tmp_path, context) == "verified"
    gold = ingest_demo._gold(
        _gold_verdict(), context["yaml_text"], context["conditions"],
        context["capabilities"], receipt_verified=True,
    )

    assert gold["gold_verification_level"] is S.GoldVerificationLevel.EXECUTED_ON_PINNED_BUNDLE
    assert gold["gold_condition_verdicts"] == [S.ConditionVerdict(
        cid="c1", status=S.ConditionCheckStatus.SATISFIED,
        checked_by=S.CheckedBy.HUMAN,
    )]


def test_explicit_not_expressed_verdict_remains_fail_closed(tmp_path):
    """A legacy reviewer must not turn an omitted condition into a pass."""
    context = _receipt_context(tmp_path)

    gold = ingest_demo._gold(
        {"condition_verdicts": [{"cid": "c1", "verdict": "not_expressed"}]},
        context["yaml_text"], context["conditions"], context["capabilities"],
        receipt_verified=False,
    )

    assert gold["gold_condition_verdicts"] == [S.ConditionVerdict(
        cid="c1", status=S.ConditionCheckStatus.NOT_EXPRESSED,
        checked_by=S.CheckedBy.HUMAN,
    )]


def test_mismatched_yaml_or_condition_hash_cannot_promote_a_legacy_case(tmp_path):
    context = _receipt_context(tmp_path)
    receipt_path = ingest_demo._receipt_path(tmp_path, context["cid"])

    for field, bad_hash in (("yaml_sha256", "0" * 64),
                            ("conditions_sha256", "1" * 64)):
        receipt = _safe_receipt(
            yaml_text=context["yaml_text"],
            user_text=context["user_text"],
            conditions_sha256=context["conditions_sha256"],
            bundle_sha256=context["bundle_sha256"],
        )
        assert receipt[field] != bad_hash
        receipt[field] = bad_hash
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        assert _state(tmp_path, context) == "invalid"
        gold = ingest_demo._gold(
            _gold_verdict(), context["yaml_text"], context["conditions"],
            context["capabilities"], receipt_verified=False,
        )
        assert gold["gold_verification_level"] is S.GoldVerificationLevel.INTENT_RECONCILED


def test_ingest_uses_the_verified_receipt_to_set_the_persisted_level(tmp_path, monkeypatch):
    """Exercise the migration wiring with invented input only.

    The tiny fake repository contains only the canonical bundle header needed
    to verify its hash; no real market capture or user export is opened here.
    """
    repo = tmp_path / "synthetic-repo"
    bundle_path = repo / ingest_demo.BUNDLE_REL
    bundle_path.parent.mkdir(parents=True)
    bundle_path.write_text(json.dumps({"schema": "cyqnt.input/v1"}), encoding="utf-8")
    bundle_sha256 = ingest_demo._pinned_bundle_sha256(repo)
    assert bundle_sha256 is not None

    internal_root = tmp_path / "private"
    internal_root.mkdir()
    salt = internal_root / "salt"
    salt.write_bytes(b"0123456789abcdef0123456789abcdef")
    salt.chmod(0o600)
    monkeypatch.setenv(S.INTERNAL_ROOT_ENV, str(internal_root))

    src = tmp_path / "synthetic-source"
    src.mkdir()
    cid = "case_synthetic"
    first_query = "invented request alpha"
    user_text = "invented follow up beta"
    combined_user_text = f"{first_query}\n{user_text}"
    yaml_text = "strategy:\n  id: q1\n"
    raw_conditions = [{
        "cid": "c1", "polarity": "require", "subject": "quote_volume_24h",
        "operator": ">", "value": 1, "unit": "usd", "scope": "universe",
        "measurability": "quantified", "quote": "invented condition wording",
    }]
    frozen_conditions = _frozen_conditions()
    (src / "cases.json").write_text(json.dumps({cid: {
        "first_query": first_query, "user_text": user_text,
        "lang": "en", "day": "2026-01-01",
    }}), encoding="utf-8")
    (src / f"{cid}_conditions.json").write_text(
        json.dumps({"conditions": raw_conditions}), encoding="utf-8")
    (src / f"{cid}_report.json").write_text(json.dumps({
        "condition_map": [{"cid": "c1", "verdict": "expressible"}],
    }), encoding="utf-8")
    (src / f"{cid}_verdict.json").write_text(json.dumps(_gold_verdict()), encoding="utf-8")
    (src / f"{cid}.yaml").write_text(yaml_text, encoding="utf-8")
    frozen_path = ingest_demo._frozen_conditions_path(src, cid)
    frozen_path.write_text(json.dumps(frozen_conditions), encoding="utf-8")
    evidence = ingest_demo._frozen_conditions_evidence(frozen_path)
    assert evidence is not None
    conditions_sha256, _ = evidence
    ingest_demo._receipt_path(src, cid).write_text(json.dumps(_safe_receipt(
        yaml_text=yaml_text,
        user_text=combined_user_text,
        conditions_sha256=conditions_sha256,
        bundle_sha256=bundle_sha256,
    )), encoding="utf-8")

    public_out = tmp_path / "public" / "cases.jsonl"
    internal_out = internal_root / "cases_demo_internal.jsonl"
    summary = ingest_demo.ingest(src, public_out, internal_out, repo)
    payload = json.loads(public_out.read_text(encoding="utf-8"))

    assert summary["written"][0]["receipt"] == "verified"
    assert summary["written"][0]["level"] == int(
        S.GoldVerificationLevel.EXECUTED_ON_PINNED_BUNDLE
    )
    assert payload["gold_verification_level"] == int(
        S.GoldVerificationLevel.EXECUTED_ON_PINNED_BUNDLE
    )


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
