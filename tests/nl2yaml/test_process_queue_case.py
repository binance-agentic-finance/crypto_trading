"""Private queue -> reviewed YAML -> frozen replay -> ledger bridge tests.

Every queue/request/review fixture in this file is invented.  Do not point this
test suite at the real ignored queue or chat export.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import re
from pathlib import Path

import pytest

from tools.nl2yaml import process_queue_case as worker
from tools.nl2yaml import evaluate
from tools.nl2yaml import schema as S
from tools.nl2yaml import sol_review


REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "standard_bot" / "fixtures" / "universe_cross_section.json"
SPEC = REPO / "docs" / "strategy_yaml_spec" / "example_selection.yaml"
ROW_ID = "r_0123456789abcdef"
RAW_REQUEST = (
    "Select 5 coins with daily volume above 100000000 USD, rank them by Square "
    "mentions descending, use both long and short "
    "synthetic-private-marker-queue-cedar"
)

GATE_CONDITIONS = [
    {"id": "liq", "subject": "quote_volume_24h", "scope": "cross_section",
     "operator": "compare", "value": {"op": ">", "threshold": 100_000_000},
     "quantified": True},
    {"id": "buzz", "subject": "social_mentions", "scope": "cross_section",
     "operator": "rank", "value": None},
    {"id": "five", "subject": "basket_size", "scope": "cross_section",
     "operator": "exact_top_k", "value": 5, "quantified": True},
    {"id": "order", "subject": "score_order", "scope": "cross_section",
     "operator": "rank", "value": "desc", "quantified": True},
]

CASE_CONDITIONS = [
    {"cid": "liq", "polarity": "include", "subject": "quote_volume_24h",
     "operator": "gt", "value": 100_000_000, "unit": "usd", "scope": "universe"},
    {"cid": "buzz", "polarity": "include", "subject": "social_mentions",
     "operator": "exists", "scope": "universe", "is_ranking": True,
     "rank_direction": "desc"},
    {"cid": "five", "polarity": "include", "subject": "basket_size",
     "operator": "exact_n", "value": 5, "unit": "count", "scope": "universe"},
    {"cid": "order", "polarity": "include", "subject": "score_order",
     "operator": "exists", "scope": "universe", "is_ranking": True,
     "rank_direction": "desc"},
]


def _private_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.chmod(path, 0o600)


def _review(yaml_text: str, *, request: str = RAW_REQUEST,
            promote_to_gold: bool = False) -> dict:
    review = {
        "schema": worker.REVIEW_SCHEMA,
        "queue_row_id": ROW_ID,
        "user_text_sha256": S.sha256_hex(request),
        "yaml_sha256": S.sha256_hex(yaml_text),
        "bundle_sha256": worker._canonical_json_sha256(
            json.loads(FIXTURE.read_text(encoding="utf-8"))
        ),
        "prompt_sha256": S.sha256_hex("synthetic frozen prompt for queue bridge"),
        "sol_review_prompt_sha256": S.sha256_hex("synthetic frozen Sol review prompt"),
        "gate_conditions": copy.deepcopy(GATE_CONDITIONS),
        "case_conditions": copy.deepcopy(CASE_CONDITIONS),
        "capability_map": [
            {"cid": cid, "verdict": "supported"}
            for cid in ("liq", "buzz", "five", "order")
        ],
        "intent_slots": {
            "intent": "COIN_SELECTION", "market_type": "futures",
            "direction": "both", "universe_scope": "all_futures",
            "wants_ranking": True,
        },
        "tier": "t1_expressible",
        "source_snapshot_id": "synthetic_queue_2026_08_02",
        "converter_model": "gpt-5.6-sol",
        "temperature": 0.0,
        "seed": 7,
        "resolution_path": "big_model",
        "conditions_with_quotes": [
            {"cid": "liq", "quote": "above 100000000 USD"},
            {"cid": "buzz", "quote": "Square mentions descending"},
            {"cid": "five", "quote": "Select 5"},
            {"cid": "order", "quote": "descending"},
        ],
        "promote_to_gold": promote_to_gold,
    }
    if promote_to_gold:
        review.update({
            "human_reviewed_by": "reviewer_alpha",
            "human_reviewed_at": "2026-08-02T12:00:00Z",
        })
    return review


def _write_approved_sol_review(inputs: dict, *, review: dict | None = None,
                               yaml_text: str | None = None) -> None:
    """Issue synthetic local audit evidence after replaying the fixture."""
    if review is None:
        review = json.loads(inputs["review"].read_text(encoding="utf-8"))
    if yaml_text is None:
        yaml_text = inputs["yaml"].read_text(encoding="utf-8")
    row = worker._queue_row(inputs["queue"], ROW_ID)
    request = worker._request_from_row(row)
    case_id = S.case_id_for(S.sha256_hex(request))
    bundle = json.loads(inputs["bundle"].read_text(encoding="utf-8"))
    prepared = evaluate.prepare_ledger_evaluation(
        case_id=case_id,
        attempt_index=1,
        prompt_sha256=review["prompt_sha256"],
        source_text=request,
        nl=request,
        yaml_answer=yaml_text,
        conditions=review["gate_conditions"],
        bundle=bundle,
        repo_git_sha="a" * 40,
        sampling_purpose=S.SamplingPurpose.CONVERT,
    )
    artifacts = worker._sol_artifact_hashes(
        review=review,
        evaluation=prepared,
        verified_repo_git_sha="a" * 40,
    )
    receipt = sol_review.issue_receipt(
        artifact_hashes=artifacts,
        cid_verdicts=[
            sol_review.CidVerdict(cid=cid, verdict="satisfied")
            for cid in sorted(condition["cid"] for condition in review["case_conditions"])
        ],
        decision=sol_review.APPROVED,
        provider_id="synthetic-test-provider",
        response_sha256=S.sha256_hex("synthetic Sol review response"),
        issued_at="2026-08-02T11:00:00Z",
    )
    sol_review.write_receipt(inputs["sol_review"], receipt)


@pytest.fixture
def queue_inputs(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    monkeypatch.setenv(S.INTERNAL_ROOT_ENV, str(private))
    monkeypatch.setattr(worker, "_verified_repo_git_sha", lambda claimed: claimed)
    salt = private / "salt"
    salt.write_bytes(b"synthetic queue bridge salt bytes")
    os.chmod(salt, 0o600)
    sol_key = private / sol_review.HMAC_KEY_FILENAME
    sol_key.write_bytes(b"synthetic queue bridge Sol review HMAC key material")
    os.chmod(sol_key, 0o600)

    queue = private / "strategy_test_queue.csv"
    fields = ["queue_rank", "cluster_id", "row_id", "dup_count", "month", "lang",
              "user_id", "first_query", "user_text_excerpt"]
    with queue.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "queue_rank": 1,
            "cluster_id": "dup_0123456789abcdef",
            "row_id": ROW_ID,
            "dup_count": 3,
            "month": "2026-08",
            "lang": "en",
            "user_id": "synthetic-user-queue-alpha",
            "first_query": "Select 5 coins with daily volume above 100000000 USD, rank them by Square mentions descending, use both long and short",
            "user_text_excerpt": RAW_REQUEST,
        })
    os.chmod(queue, 0o600)

    yaml_text = SPEC.read_text(encoding="utf-8")
    yaml_path = private / "reviewed.yaml"
    review_path = private / "review.json"
    _private_file(yaml_path, yaml_text)
    _private_file(review_path, json.dumps(_review(yaml_text), ensure_ascii=False))
    inputs = {
        "private": private,
        "queue": queue,
        "review": review_path,
        "yaml": yaml_path,
        "bundle": FIXTURE,
        "public": tmp_path / "public-ledger",
        "sol_review": private / "sol_review.json",
    }
    _write_approved_sol_review(inputs)
    return inputs


def _process(inputs: dict, **overrides):
    kwargs = {
        "queue_path": inputs["queue"],
        "row_id": ROW_ID,
        "review_path": inputs["review"],
        "yaml_path": inputs["yaml"],
        "bundle_path": inputs["bundle"],
        "repo_git_sha": "a" * 40,
        "public_root": inputs["public"],
        "sol_review_path": inputs["sol_review"],
    }
    kwargs.update(overrides)
    return worker.process_queue_case(**kwargs)


def _write_review(inputs: dict, review: dict) -> None:
    _private_file(inputs["review"], json.dumps(review, ensure_ascii=False))


def _rewrite_queue_request(inputs: dict, request: str, *, user_id: str | None = None) -> None:
    with inputs["queue"].open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None and len(rows) == 1
    rows[0]["first_query"] = request
    rows[0]["user_text_excerpt"] = request
    if user_id is not None:
        rows[0]["user_id"] = user_id
    with inputs["queue"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(inputs["queue"], 0o600)


def test_queue_case_writes_private_pair_without_claiming_gold(queue_inputs):
    result = _process(queue_inputs)

    assert result.gold_promoted is False
    assert result.run_written is True
    cases_path = queue_inputs["public"] / "cases.jsonl"
    attempts_path = queue_inputs["public"] / "attempts.jsonl"
    runs_path = queue_inputs["public"] / "runs.jsonl"
    internal_path = queue_inputs["private"] / worker.DEFAULT_INTERNAL_CASES
    receipt_path = queue_inputs["private"] / worker.DEFAULT_PRIVATE_RECEIPT
    case = list(S.read_cases(cases_path))[0]
    attempt = list(S.read_attempts(attempts_path))[0]
    run = list(S.read_runs(runs_path))[0]

    assert case.case_id == result.case_id == attempt.case_id == run.case_id
    assert case.gold_eligible_for_sft is False
    assert case.human_reviewed_by is None
    assert attempt.stopped_at_gate is S.Gate.PASSED
    assert run.signal_count == 1
    assert S.audit_public_private_case_links(cases_path, [internal_path])[case.case_id] == RAW_REQUEST
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # A local receipt may exist for private replay integrity, but a non-gold
    # row must not present it as proof of an external Sol review.
    assert receipt["sol_reviewer_model"] is None
    assert receipt["sol_reasoning_effort"] is None
    assert receipt["sol_review_receipt_sha256"] is None
    assert receipt_path.stat().st_mode & 0o777 == 0o600

    public_rendered = "\n".join(path.read_text(encoding="utf-8")
                                  for path in (cases_path, attempts_path, runs_path))
    assert RAW_REQUEST not in public_rendered
    assert "synthetic-private-marker-queue-cedar" not in public_rendered
    assert '"sol_reviewer_model"' not in cases_path.read_text(encoding="utf-8"), (
        "Sol review is private; converter metadata is separate from review identity"
    )


def test_gold_promotion_requires_a_content_bound_sol_receipt(queue_inputs):
    review = _review(SPEC.read_text(encoding="utf-8"), promote_to_gold=True)
    _write_review(queue_inputs, review)
    queue_inputs["sol_review"].unlink()

    with pytest.raises(ValueError, match="Sol review receipt is missing"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()
    assert not (queue_inputs["private"] / worker.DEFAULT_INTERNAL_CASES).exists()


def test_local_hmac_audit_receipt_can_never_promote_strict_gold(queue_inputs):
    review = _review(SPEC.read_text(encoding="utf-8"), promote_to_gold=True)
    _write_review(queue_inputs, review)
    _write_approved_sol_review(queue_inputs, review=review)

    with pytest.raises(ValueError, match="local HMAC review receipt is private integrity evidence"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()
    assert not (queue_inputs["private"] / worker.DEFAULT_INTERNAL_CASES).exists()


def test_free_model_review_metadata_cannot_substitute_for_sol_receipt(queue_inputs):
    queue_inputs["sol_review"].unlink()
    review = _review(SPEC.read_text(encoding="utf-8"), promote_to_gold=True)
    review["model_reviewed_by"] = "gpt-5.6-sol"
    review["model_reviewed_at"] = "2026-08-02T11:00:00Z"
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="unknown or missing field"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()


def test_proxy_or_unsupported_review_refuses_before_any_public_write(queue_inputs):
    review = _review(SPEC.read_text(encoding="utf-8"))
    review["tier"] = "t2_proxy_needed"
    review["capability_map"][0] = {
        "cid": "liq", "verdict": "proxy", "gap_id": "social_sentiment",
    }
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="t1_expressible"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()
    assert not (queue_inputs["private"] / worker.DEFAULT_INTERNAL_CASES).exists()

    outcome_path = queue_inputs["private"] / worker.DEFAULT_PRIVATE_OUTCOMES
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["outcome"] == "unsupported"
    assert outcome["queue_row_id"] == ROW_ID
    assert RAW_REQUEST not in outcome_path.read_text(encoding="utf-8")
    assert "synthetic-user-queue-alpha" not in outcome_path.read_text(encoding="utf-8")
    assert outcome_path.stat().st_mode & 0o777 == 0o600


def test_public_condition_must_losslessly_match_the_executed_gate(queue_inputs):
    review = _review(SPEC.read_text(encoding="utf-8"))
    for condition in review["case_conditions"]:
        if condition["cid"] == "liq":
            condition["value"] = 10 ** 20
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="does not match its executed comparison"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()
    assert not (queue_inputs["private"] / worker.DEFAULT_PRIVATE_DIAGNOSTICS).exists()


def test_exact_basket_count_rejects_at_most_source_wording(queue_inputs):
    raw_request = RAW_REQUEST.replace("Select 5", "Select at most 5")
    _rewrite_queue_request(queue_inputs, raw_request)
    review = _review(SPEC.read_text(encoding="utf-8"), request=raw_request)
    for quote in review["conditions_with_quotes"]:
        if quote["cid"] == "five":
            quote["quote"] = "Select at most 5"
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="exact basket count conflicts"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()


def test_legacy_at_most_basket_count_needs_and_accepts_ceiling_wording(queue_inputs):
    raw_request = RAW_REQUEST.replace("Select 5", "Select at most 5")
    _rewrite_queue_request(queue_inputs, raw_request)
    review = _review(SPEC.read_text(encoding="utf-8"), request=raw_request)
    for condition in review["gate_conditions"]:
        if condition["id"] == "five":
            condition["operator"] = "top_k"
    for condition in review["case_conditions"]:
        if condition["cid"] == "five":
            condition["operator"] = "top_n"
    for quote in review["conditions_with_quotes"]:
        if quote["cid"] == "five":
            quote["quote"] = "Select at most 5"
    _write_review(queue_inputs, review)

    result = _process(queue_inputs)
    assert result.gold_promoted is False


def test_public_condition_unit_must_match_the_executed_predicate(queue_inputs):
    review = _review(SPEC.read_text(encoding="utf-8"))
    for condition in review["case_conditions"]:
        if condition["cid"] == "liq":
            condition["unit"] = "pct"
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="does not match its executed comparison"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()


def test_review_must_bind_the_exact_frozen_bundle(queue_inputs):
    review = _review(SPEC.read_text(encoding="utf-8"))
    review["bundle_sha256"] = "0" * 64
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="different frozen bundle"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()


def test_unquantified_liquid_word_cannot_be_reviewed_into_numeric_gold(queue_inputs):
    raw_request = (
        "Select 5 liquid coins ranked by Square mentions descending, use both long and short "
        "synthetic-private-marker-unquantified"
    )
    _rewrite_queue_request(queue_inputs, raw_request)
    review = _review(SPEC.read_text(encoding="utf-8"), request=raw_request)
    for quote in review["conditions_with_quotes"]:
        if quote["cid"] == "liq":
            quote["quote"] = "liquid"
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="direct source provenance for its value"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()
    assert not (queue_inputs["private"] / worker.DEFAULT_PRIVATE_DIAGNOSTICS).exists()


def test_numeric_source_provenance_cannot_accept_a_prefix_of_a_larger_literal(queue_inputs):
    review = _review(SPEC.read_text(encoding="utf-8"))
    for condition in review["gate_conditions"]:
        if condition["id"] == "liq":
            condition["value"] = {"op": ">", "threshold": 10_000_000}
    for condition in review["case_conditions"]:
        if condition["cid"] == "liq":
            condition["value"] = 10_000_000
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="direct source provenance for its value"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()


def test_gate_and_public_large_integers_must_match_without_float_rounding(queue_inputs):
    raw_request = RAW_REQUEST.replace("100000000", "9007199254740993")
    _rewrite_queue_request(queue_inputs, raw_request)
    review = _review(SPEC.read_text(encoding="utf-8"), request=raw_request)
    for condition in review["gate_conditions"]:
        if condition["id"] == "liq":
            condition["value"] = {"op": ">", "threshold": 9_007_199_254_740_992}
    for condition in review["case_conditions"]:
        if condition["cid"] == "liq":
            condition["value"] = 9_007_199_254_740_993
    for quote in review["conditions_with_quotes"]:
        if quote["cid"] == "liq":
            quote["quote"] = "above 9007199254740993 USD"
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="does not match its executed comparison"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()


def test_integer_not_exactly_representable_by_g1e_float_is_refused(queue_inputs):
    raw_request = RAW_REQUEST.replace("100000000", "9007199254740993")
    _rewrite_queue_request(queue_inputs, raw_request)
    review = _review(SPEC.read_text(encoding="utf-8"), request=raw_request)
    for condition in review["gate_conditions"]:
        if condition["id"] == "liq":
            condition["value"] = {"op": ">", "threshold": 9_007_199_254_740_993}
    for condition in review["case_conditions"]:
        if condition["cid"] == "liq":
            condition["value"] = 9_007_199_254_740_993
    for quote in review["conditions_with_quotes"]:
        if quote["cid"] == "liq":
            quote["quote"] = "above 9007199254740993 USD"
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="does not match its executed comparison"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()


def test_numeric_unit_must_be_adjacent_to_the_same_source_literal(queue_inputs):
    raw_request = RAW_REQUEST.replace(
        "above 100000000 USD", "above 100000000 coins; balance is $1"
    )
    _rewrite_queue_request(queue_inputs, raw_request)
    review = _review(SPEC.read_text(encoding="utf-8"), request=raw_request)
    for quote in review["conditions_with_quotes"]:
        if quote["cid"] == "liq":
            quote["quote"] = "above 100000000 coins; balance is $1"
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="direct source provenance for its unit"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()


def test_nonclean_frozen_replay_cannot_promote_gold_or_write_pair(queue_inputs):
    raw_request = RAW_REQUEST.replace("100000000", "9000000000000000")
    _rewrite_queue_request(queue_inputs, raw_request)
    review = _review(
        SPEC.read_text(encoding="utf-8"), request=raw_request, promote_to_gold=True
    )
    for condition in review["gate_conditions"]:
        if condition["id"] == "liq":
            condition["value"] = {"op": ">", "threshold": 9_000_000_000_000_000}
    for condition in review["case_conditions"]:
        if condition["cid"] == "liq":
            condition["value"] = 9_000_000_000_000_000
    for quote in review["conditions_with_quotes"]:
        if quote["cid"] == "liq":
            quote["quote"] = "above 9000000000000000 USD"
    _write_review(queue_inputs, review)

    with pytest.raises(ValueError, match="clean frozen replay"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()
    assert not (queue_inputs["public"] / "attempts.jsonl").exists()
    assert not (queue_inputs["private"] / worker.DEFAULT_INTERNAL_CASES).exists()
    diagnostics = queue_inputs["private"] / worker.DEFAULT_PRIVATE_DIAGNOSTICS
    diagnostic = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert diagnostic["gate_status"] == "condition_violated"
    assert diagnostic["diagnostics"]["condition_verdicts"]


def test_private_metadata_and_user_id_are_not_copied_to_public_records(queue_inputs):
    _rewrite_queue_request(queue_inputs, RAW_REQUEST, user_id="123456789")
    review = _review(SPEC.read_text(encoding="utf-8"))
    review["source_snapshot_id"] = "snapshot_123456789"
    _write_review(queue_inputs, review)

    _process(queue_inputs)
    public_text = "\n".join(
        (queue_inputs["public"] / name).read_text(encoding="utf-8")
        for name in ("cases.jsonl", "attempts.jsonl", "runs.jsonl")
    )
    assert not re.search(r"(?<![A-Za-z0-9])123456789(?![A-Za-z0-9])", public_text)
    assert "snapshot_123456789" not in public_text
    case = list(S.read_cases(queue_inputs["public"] / "cases.jsonl"))[0]
    assert case.source_snapshot_id == worker._public_metadata_label(
        "snapshot", "snapshot_123456789"
    )
    assert case.source_snapshot_id != "snapshot_" + S.sha256_hex(
        "snapshot_123456789"
    )[:16]
    run = list(S.read_runs(queue_inputs["public"] / "runs.jsonl"))[0]
    assert run.bundle_snapshot_id == "snapshot_" + run.bundle_sha256[:16]


def test_short_private_user_id_is_held_before_any_public_artifact(queue_inputs):
    yaml_text = SPEC.read_text(encoding="utf-8") + "\n# operator account 1234567\n"
    _private_file(queue_inputs["yaml"], yaml_text)
    _rewrite_queue_request(queue_inputs, RAW_REQUEST, user_id="1234567")
    _write_review(queue_inputs, _review(yaml_text))

    with pytest.raises(ValueError, match="identity is too short"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()


def test_private_user_id_in_yaml_is_rejected_even_when_not_in_request(queue_inputs):
    yaml_text = SPEC.read_text(encoding="utf-8") + "\n# operator account 123456789\n"
    _private_file(queue_inputs["yaml"], yaml_text)
    _rewrite_queue_request(queue_inputs, RAW_REQUEST, user_id="123456789")
    _write_review(queue_inputs, _review(yaml_text))

    with pytest.raises(S.PrivacyError, match="private user identifier"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()


def test_nonpromoted_yaml_with_copied_request_is_rejected_before_case_pair(queue_inputs):
    yaml_text = SPEC.read_text(encoding="utf-8") + "\n# " + RAW_REQUEST + "\n"
    _private_file(queue_inputs["yaml"], yaml_text)
    review = _review(yaml_text)
    review["promote_to_gold"] = False
    _write_review(queue_inputs, review)

    with pytest.raises(S.PrivacyError, match=r"reproduces (entire )?user wording"):
        _process(queue_inputs)
    assert not (queue_inputs["public"] / "cases.jsonl").exists()
    assert not (queue_inputs["public"] / "attempts.jsonl").exists()
    assert not (queue_inputs["private"] / worker.DEFAULT_INTERNAL_CASES).exists()


def test_public_preflight_rejects_an_entire_short_request_without_8gram_overlap():
    """The broad worker boundary closes short copied values outside YAML writes."""
    request = "Buy BTC now"  # invented fixture; never corpus text
    public_value = {"metadata": "review note: BUY btc NOW"}
    assert S.verbatim_overlap(public_value["metadata"], request, 8) == set()

    with pytest.raises(S.PrivacyError, match="entire user wording"):
        worker._assert_no_verbatim_public_strings(
            public_value, request=request, user_id="synthetic-user-queue-alpha"
        )


def test_repeated_or_partial_queue_recovery_is_idempotent(queue_inputs):
    first = _process(queue_inputs)
    second = _process(queue_inputs)
    assert second == first
    for name, reader in (("cases.jsonl", S.read_cases),
                         ("attempts.jsonl", S.read_attempts),
                         ("runs.jsonl", S.read_runs)):
        assert len(list(reader(queue_inputs["public"] / name))) == 1
    for name in (worker.DEFAULT_PRIVATE_RECEIPT, worker.DEFAULT_PRIVATE_DIAGNOSTICS):
        assert len((queue_inputs["private"] / name).read_text(encoding="utf-8").splitlines()) == 1

    # Simulate a process dying after the paired case append.  The next exact
    # replay must fill only the missing public links, not duplicate the case or
    # claim that recovery is impossible because the case is already present.
    (queue_inputs["public"] / "attempts.jsonl").unlink()
    (queue_inputs["public"] / "runs.jsonl").unlink()
    recovered = _process(queue_inputs)
    assert recovered == first
    assert len(list(S.read_cases(queue_inputs["public"] / "cases.jsonl"))) == 1
    assert len(list(S.read_attempts(queue_inputs["public"] / "attempts.jsonl"))) == 1
    assert len(list(S.read_runs(queue_inputs["public"] / "runs.jsonl"))) == 1


def test_case_claim_refuses_a_second_worker_for_the_same_case(queue_inputs):
    case_id = S.case_id_for(S.sha256_hex(RAW_REQUEST))
    with worker._CaseClaim(case_id) as claim:
        with pytest.raises(ValueError, match="already being curated"):
            with worker._CaseClaim(case_id):
                pass
        claim.complete()


def test_repo_sha_is_bound_to_actual_clean_local_checkout(monkeypatch):
    def clean_run(command, **_kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return type("Result", (), {"stdout": "a" * 40 + "\n"})()
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr(worker.subprocess, "run", clean_run)
    assert worker._verified_repo_git_sha("a" * 40) == "a" * 40

    def dirty_run(command, **_kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return type("Result", (), {"stdout": "a" * 40 + "\n"})()
        return type("Result", (), {"stdout": "?? local-runtime-config.py\n"})()

    monkeypatch.setattr(worker.subprocess, "run", dirty_run)
    with pytest.raises(ValueError, match="dirty checkout"):
        worker._verified_repo_git_sha("a" * 40)


def test_private_queue_symlink_escape_is_rejected_before_opening_other_inputs(
        tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    monkeypatch.setenv(S.INTERNAL_ROOT_ENV, str(private))
    (private / "escape").symlink_to(REPO, target_is_directory=True)

    with pytest.raises(S.PrivacyError, match="private queue"):
        worker.process_queue_case(
            queue_path=private / "escape" / "queue.csv",
            row_id=ROW_ID,
            review_path=private / "missing-review.json",
            yaml_path=private / "missing.yaml",
            bundle_path=FIXTURE,
            repo_git_sha="a" * 40,
            public_root=tmp_path / "public-ledger",
        )
