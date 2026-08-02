"""Frozen evaluator -> public ledger linkage, with no raw-input spillover."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.nl2yaml import evaluate
from tools.nl2yaml import gates
from tools.nl2yaml import schema as S


REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "standard_bot" / "fixtures" / "universe_cross_section.json"
SPEC_NEWS = REPO / "docs" / "strategy_yaml_spec" / "example_selection.yaml"
NL_NEWS = ("幫我選 5 個幣:先過濾流動性,再依 Square 提及量熱度排名,"
           "情緒偏多做多、偏空做空")
# This is intentionally not copied from the shipped YAML/comments.  The public
# writer must reject copied request wording, so a happy ledger test needs a
# semantically equivalent synthetic request whose text cannot be mistaken for a
# safe YAML description.
SAFE_NL_NEWS = ("Select 5 liquid coins ranked by Square mentions, "
                "use both long and short")
NEWS_CONDITIONS = [
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


def _bundle() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _yaml() -> str:
    return SPEC_NEWS.read_text(encoding="utf-8")


def _run(tmp_path, *, raw_request: str, conditions=NEWS_CONDITIONS, source_text=None,
         case_id=None):
    return evaluate.evaluate_and_write_ledger(
        case_id=S.case_id_for(S.sha256_hex(raw_request)) if case_id is None else case_id,
        attempt_index=1,
        prompt_sha256=S.sha256_hex("synthetic frozen prompt only"),
        source_text=raw_request if source_text is None else source_text,
        nl=raw_request,
        yaml_answer=_yaml(),
        conditions=copy.deepcopy(conditions),
        bundle=_bundle(),
        attempt_path=tmp_path / "attempts.jsonl",
        run_path=tmp_path / "runs.jsonl",
        repo_git_sha="a" * 40,
        python_version="3.11.0",
        pandas_version="2.2.3",
    )


def test_happy_selection_writes_one_signal_many_candidates_with_linked_provenance(tmp_path):
    raw_request = SAFE_NL_NEWS + " synthetic-private-marker-lemongrass"
    result = _run(tmp_path, raw_request=raw_request)

    assert result.receipt.gate_status == gates.STATUS_PASSED
    assert result.attempt.stopped_at_gate is S.Gate.PASSED
    assert result.run is not None
    assert result.run.signal_count == 1
    assert len(result.run.candidates) > 1
    assert all(candidate.attributes == {} for candidate in result.run.candidates)
    assert result.run.case_id == result.attempt.case_id
    assert result.run.attempt_index == result.attempt.attempt_index
    assert result.run.yaml_sha256 == result.attempt.yaml_sha256 == result.receipt.yaml_sha256
    assert result.run.bundle_sha256 == result.receipt.bundle_sha256
    assert result.run.signal_batch_sha256 == result.receipt.signal_batch_sha256

    attempts = list(S.read_attempts(tmp_path / "attempts.jsonl"))
    runs = list(S.read_runs(tmp_path / "runs.jsonl"))
    assert attempts == [result.attempt]
    assert runs == [result.run]

    # The immutable hashes bind the real output, while the public records omit
    # every free-prose candidate explanation and the request itself.
    rendered = "\n".join((
        json.dumps(result.receipt.to_dict(), ensure_ascii=False, sort_keys=True),
        (tmp_path / "attempts.jsonl").read_text(encoding="utf-8"),
        (tmp_path / "runs.jsonl").read_text(encoding="utf-8"),
    ))
    assert raw_request not in rendered
    assert "synthetic-private-marker-lemongrass" not in rendered
    assert '"reason"' not in (tmp_path / "runs.jsonl").read_text(encoding="utf-8")


def test_prepare_then_write_runs_the_frozen_evaluation_once_without_early_append(tmp_path):
    raw_request = SAFE_NL_NEWS + " synthetic-private-marker-prepared"
    prepared = evaluate.prepare_ledger_evaluation(
        case_id=S.case_id_for(S.sha256_hex(raw_request)),
        attempt_index=1,
        prompt_sha256=S.sha256_hex("synthetic prepared prompt"),
        source_text=raw_request,
        nl=raw_request,
        yaml_answer=_yaml(),
        conditions=copy.deepcopy(NEWS_CONDITIONS),
        bundle=_bundle(),
        repo_git_sha="a" * 40,
        python_version="3.11.0",
        pandas_version="2.2.3",
    )

    assert prepared.receipt.gate_status == gates.STATUS_PASSED
    assert not (tmp_path / "attempts.jsonl").exists()
    assert not (tmp_path / "runs.jsonl").exists()
    with pytest.raises(ValueError, match="prepared frozen NL request"):
        evaluate.write_ledger_evaluation(
            prepared,
            attempt_path=tmp_path / "attempts.jsonl",
            run_path=tmp_path / "runs.jsonl",
            source_text="different synthetic source",
        )
    assert not (tmp_path / "attempts.jsonl").exists()
    assert not (tmp_path / "runs.jsonl").exists()

    written = evaluate.write_ledger_evaluation(
        prepared,
        attempt_path=tmp_path / "attempts.jsonl",
        run_path=tmp_path / "runs.jsonl",
        source_text=raw_request,
    )
    assert written == prepared
    assert list(S.read_attempts(tmp_path / "attempts.jsonl")) == [prepared.attempt]
    assert list(S.read_runs(tmp_path / "runs.jsonl")) == [prepared.run]


def test_prepared_mutable_attempt_cannot_be_changed_before_write(tmp_path):
    raw_request = SAFE_NL_NEWS + " synthetic-private-marker-altered-prepared"
    prepared = evaluate.prepare_ledger_evaluation(
        case_id=S.case_id_for(S.sha256_hex(raw_request)),
        attempt_index=1,
        prompt_sha256=S.sha256_hex("synthetic altered prepared prompt"),
        source_text=raw_request,
        nl=raw_request,
        yaml_answer=_yaml(),
        conditions=copy.deepcopy(NEWS_CONDITIONS),
        bundle=_bundle(),
        repo_git_sha="a" * 40,
        python_version="3.11.0",
        pandas_version="2.2.3",
    )

    # LedgerEvaluation is frozen only at its outer layer; prove the writer
    # rejects mutation of a nested record instead of pairing new YAML with an
    # old yaml_sha256/receipt.
    prepared.attempt.yaml_text += "\n# changed-after-frozen-replay\n"
    with pytest.raises(ValueError, match="prepared attempt record was altered"):
        evaluate.write_ledger_evaluation(
            prepared,
            attempt_path=tmp_path / "attempts.jsonl",
            run_path=tmp_path / "runs.jsonl",
            source_text=raw_request,
        )
    assert not (tmp_path / "attempts.jsonl").exists()
    assert not (tmp_path / "runs.jsonl").exists()


def test_g1e_failure_writes_a_nonpassed_attempt_and_its_completed_run(tmp_path):
    raw_request = SAFE_NL_NEWS + " synthetic-private-marker-plum"
    conditions = copy.deepcopy(NEWS_CONDITIONS)
    for condition in conditions:
        if condition["id"] == "liq":
            # The emitted candidates are real, but none can meet this reviewed
            # floor.  G1e therefore has an objective output violation after G1d.
            condition["value"] = {"op": ">", "threshold": 10 ** 20}
    conditions.append({
        "id": "cap", "subject": "market_cap", "scope": "cross_section",
        "operator": "compare", "value": {"op": ">", "threshold": 1_000_000},
        "quantified": True,
    })

    result = _run(tmp_path, raw_request=raw_request, conditions=conditions)

    assert result.receipt.gate_status == "condition_violated"
    assert result.receipt.failed_gate == "G1e"
    assert result.attempt.stopped_at_gate is S.Gate.CONDITION_VIOLATED
    assert len(result.attempt.gate_errors) == 1
    assert result.attempt.gate_errors[0].startswith(
        "frozen-evaluation-gate-condition_violated-")
    assert result.attempt.error_signature == S.error_signature_for(result.attempt.gate_errors)
    assert {verdict.status for verdict in result.attempt.condition_verdicts} >= {
        S.ConditionCheckStatus.VIOLATED,
        S.ConditionCheckStatus.PROXIED,
    }
    assert dict(result.receipt.conditions.verdict_counts)[gates.NOT_EXPRESSED] == 1
    assert result.run is not None, "G1d ran before G1e rejected the semantics"
    assert result.run.signal_count == 1
    assert len(result.run.candidates) > 1
    assert list(S.read_attempts(tmp_path / "attempts.jsonl"))[0].stopped_at_gate is S.Gate.CONDITION_VIOLATED
    assert list(S.read_runs(tmp_path / "runs.jsonl"))[0].yaml_sha256 == result.attempt.yaml_sha256


def test_public_write_requires_the_same_in_memory_source_text(tmp_path):
    with pytest.raises(ValueError, match="source_text must exactly match"):
        _run(tmp_path, raw_request=SAFE_NL_NEWS + " synthetic-private-marker-mismatch",
             conditions=NEWS_CONDITIONS, source_text="different synthetic source")
    assert not (tmp_path / "attempts.jsonl").exists()
    assert not (tmp_path / "runs.jsonl").exists()


@pytest.mark.parametrize("case_id", [
    "case_0000000000000000",
    "01KYZ4WCWMDA7N8WWE6JZBY9X1",
    "123e4567-e89b-42d3-a456-426614174000",
])
def test_ledger_bridge_rejects_case_ids_not_bound_to_the_frozen_request(
        tmp_path, case_id):
    raw_request = SAFE_NL_NEWS + " synthetic-private-marker-case-identity"
    expected = S.case_id_for(S.sha256_hex(raw_request))
    # The fixed ``case_`` literal is overwhelmingly likely to differ.  Make
    # the mismatch deterministic so this test cannot rely on a hash collision.
    if case_id == expected:
        case_id = expected[:-1] + ("0" if expected[-1] != "0" else "1")

    with pytest.raises(ValueError, match="content-addressed case_id") as exc_info:
        _run(tmp_path, raw_request=raw_request, case_id=case_id)

    # A ledger helper has no CaseRecord/InternalCaseRecord proof for a ULID or
    # UUID, and must not write an unbound replay under either form.  Its fixed
    # error also must not echo request text.
    assert raw_request not in str(exc_info.value)
    assert not (tmp_path / "attempts.jsonl").exists()
    assert not (tmp_path / "runs.jsonl").exists()


def test_run_identity_includes_case_and_attempt_not_only_the_batch_hash(tmp_path):
    # Each raw request gets its own valid content-addressed case ID.  The same
    # YAML and frozen bundle still yield the same output batch, so this proves
    # run identity is linked to the case/attempt rather than only that batch.
    first = _run(
        tmp_path,
        raw_request=SAFE_NL_NEWS + " synthetic-private-marker-run-id-first",
    )
    second = _run(
        tmp_path,
        raw_request=SAFE_NL_NEWS + " synthetic-private-marker-run-id-second",
    )

    assert first.run is not None and second.run is not None
    assert first.run.case_id != second.run.case_id
    assert first.run.signal_batch_sha256 == second.run.signal_batch_sha256
    assert first.run.run_id != second.run.run_id
    assert len(list(S.read_runs(tmp_path / "runs.jsonl"))) == 2
