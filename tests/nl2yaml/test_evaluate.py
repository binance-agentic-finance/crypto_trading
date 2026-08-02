"""Tests for the privacy-safe frozen evaluator front door."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from tools.nl2yaml import evaluate
from tools.nl2yaml import gates


REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "standard_bot" / "fixtures" / "universe_cross_section.json"
SPEC_NEWS = REPO / "docs" / "strategy_yaml_spec" / "example_selection.yaml"
NL_NEWS = ("幫我選 5 個幣:先過濾流動性,再依 Square 提及量熱度排名,"
           "情緒偏多做多、偏空做空")
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


def _three_eligible_bundle() -> dict:
    bundle = _bundle()
    keep = {"BNBUSDT", "BTCUSDT", "GIGGLEUSDT"}
    for row in bundle["frames"]["universe"]["rows"]:
        if row.get("instrument_id") not in keep:
            row["quoteVolume"] = 0
    return bundle


def _yaml() -> str:
    return SPEC_NEWS.read_text(encoding="utf-8")


def _evaluate(*, conditions=NEWS_CONDITIONS, nl=NL_NEWS, bundle=None):
    return evaluate.evaluate_frozen(
        nl=nl,
        yaml_answer=_yaml(),
        conditions=copy.deepcopy(conditions),
        bundle=_bundle() if bundle is None else bundle,
    )


def test_frozen_evaluation_is_repeatable_and_does_not_mutate_inputs():
    bundle = _bundle()
    before = copy.deepcopy(bundle)

    first = evaluate.evaluate_frozen(
        nl=NL_NEWS,
        yaml_answer=_yaml(),
        conditions=copy.deepcopy(NEWS_CONDITIONS),
        bundle=bundle,
    )
    second = _evaluate()

    assert first == second
    assert bundle == before
    assert first.bundle_sha256 == hashlib.sha256(
        json.dumps(before, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def test_frozen_happy_path_is_gold_clean_with_safe_batch_receipt():
    receipt = _evaluate()

    assert receipt.schema == evaluate.RECEIPT_SCHEMA
    assert receipt.gate_status == gates.STATUS_PASSED
    assert receipt.failed_gate is None
    assert receipt.draft_valid is True
    assert receipt.runtime_valid is True
    assert receipt.gold_clean is True
    assert receipt.conditions.requested_count == len(NEWS_CONDITIONS)
    assert receipt.conditions.evaluated_count == len(NEWS_CONDITIONS)
    assert receipt.conditions.verdict_counts == (
        (gates.SATISFIED, len(NEWS_CONDITIONS)),
        (gates.VIOLATED, 0),
        (gates.UNVERIFIABLE, 0),
        (gates.NOT_EXPRESSED, 0),
    )
    assert receipt.signal_batch_sha256 is not None
    assert len(receipt.signal_batch_sha256) == 64
    assert receipt.signal_count == 1
    assert receipt.selection_candidate_count and receipt.selection_candidate_count > 0


def test_exact_count_shortage_is_runtime_evidence_but_never_gold_clean():
    """A real G1d run with only three eligible rows is data-limited, not green."""
    receipt = _evaluate(bundle=_three_eligible_bundle())

    assert receipt.gate_status == "bundle_insufficient"
    assert receipt.failed_gate == "G1e"
    assert receipt.runtime_valid is True, "G1d did execute the frozen bundle"
    assert receipt.gold_clean is False
    assert receipt.selection_candidate_count == 3
    assert dict(receipt.conditions.verdict_counts)[gates.UNVERIFIABLE] == 1


def test_empty_conditions_are_only_a_smoke_test_not_runtime_or_gold_valid():
    receipt = _evaluate(conditions=())

    # G1a--G1c still establish a useful draft, but a request with no reviewed
    # predicates cannot be treated as an executed, semantically checked result.
    assert receipt.draft_valid is True
    assert receipt.runtime_valid is False
    assert receipt.gold_clean is False
    assert receipt.gate_status == gates.STATUS_CONDITION_UNRESOLVED
    assert receipt.failed_gate == "G1e"
    assert receipt.conditions.requested_count == 0
    assert receipt.conditions.evaluated_count == 0
    assert receipt.signal_batch_sha256 is not None


def test_serialized_receipt_never_contains_the_raw_request_text():
    raw_request = "PRIVATE-RAW-REQUEST-8af48eaf: 請不要把這句話寫入收據"
    receipt = _evaluate(nl=raw_request)
    rendered = json.dumps(receipt.to_dict(), ensure_ascii=False, sort_keys=True)

    assert raw_request not in rendered
    assert "請不要把這句話" not in rendered
    assert receipt.nl_sha256 == hashlib.sha256(raw_request.encode("utf-8")).hexdigest()
    assert raw_request != receipt.nl_sha256


def test_cli_reads_local_files_and_emits_only_a_safe_json_receipt(tmp_path, capsys):
    raw_request = "PRIVATE-CLI-RAW-3ddcea79: 絕不能印出這段"
    nl_path = tmp_path / "request.txt"
    yaml_path = tmp_path / "answer.yaml"
    conditions_path = tmp_path / "conditions.json"
    bundle_path = tmp_path / "bundle.json"
    nl_path.write_text(raw_request, encoding="utf-8")
    # A compact parse failure is enough to prove the CLI's I/O boundary without
    # turning this test into a second expensive strategy replay.
    yaml_path.write_text("- not a YAML mapping\n", encoding="utf-8")
    conditions_path.write_text("[]", encoding="utf-8")
    bundle_path.write_text(json.dumps(_bundle()), encoding="utf-8")

    exit_code = evaluate.main([
        "--nl-file", str(nl_path),
        "--yaml-file", str(yaml_path),
        "--conditions-file", str(conditions_path),
        "--bundle-file", str(bundle_path),
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["schema"] == evaluate.RECEIPT_SCHEMA
    assert payload["gate_status"] == "parse_error"
    assert raw_request not in captured.out
    assert "絕不能印出" not in captured.out
