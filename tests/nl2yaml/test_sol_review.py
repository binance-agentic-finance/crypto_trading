"""Synthetic tests for the private, HMAC-attested Sol review receipt boundary."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tools.nl2yaml import schema
from tools.nl2yaml import sol_review as sol


def _digest(label: str) -> str:
    """A digest of invented fixture material, never a user request or response."""
    return hashlib.sha256(("synthetic-sol-review:" + label).encode("utf-8")).hexdigest()


def _artifacts() -> sol.ArtifactHashes:
    return sol.ArtifactHashes(
        review_prompt_sha256=_digest("review-prompt"),
        request_sha256=_digest("request"),
        yaml_sha256=_digest("yaml"),
        conditions_sha256=_digest("conditions"),
        bundle_sha256=_digest("bundle"),
        attempt_sha256=_digest("attempt"),
        run_sha256=_digest("run"),
        signal_sha256=_digest("signal"),
        capability_sha256=_digest("capability"),
        repo_sha256=_digest("repo-manifest"),
    )


def _satisfied_cids() -> list[sol.CidVerdict]:
    return [
        sol.CidVerdict(cid="liquidity", verdict="satisfied"),
        sol.CidVerdict(cid="ranking", verdict="satisfied"),
    ]


def _issue(**overrides) -> sol.SolReviewReceipt:
    kwargs = {
        "artifact_hashes": _artifacts(),
        "cid_verdicts": _satisfied_cids(),
        "decision": sol.APPROVED,
        "provider_id": "resp_synthetic_20260802_01",
        "response_sha256": _digest("provider-response"),
        "issued_at": "2026-08-02T12:34:56Z",
    }
    kwargs.update(overrides)
    return sol.issue_receipt(**kwargs)


@pytest.fixture()
def private_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "private-sol-receipts"
    root.mkdir(mode=0o700)
    monkeypatch.setenv(schema.INTERNAL_ROOT_ENV, str(root))
    return root


@pytest.fixture()
def provisioned_key(private_root: Path) -> Path:
    key = sol.attestation_key_path()
    # Fixed synthetic bytes are sufficient for a unit test; production key
    # provisioning remains explicitly out of band and is never auto-generated.
    key.write_bytes(b"synthetic-sol-review-key-material-32b")
    os.chmod(key, 0o600)
    return key


def test_good_path_issues_writes_reads_and_binds_all_artifacts(
    private_root: Path, provisioned_key: Path,
) -> None:
    raw_provider_response = "SYNTHETIC RAW PROVIDER RESPONSE -- DO NOT PERSIST"
    receipt = _issue(response_sha256=hashlib.sha256(
        raw_provider_response.encode("utf-8")
    ).hexdigest())

    assert receipt.reviewer_model == "gpt-5.6-sol"
    assert receipt.reasoning_effort == "xhigh"
    assert receipt.is_clean_approval is True
    verified = sol.require_approved_receipt(
        receipt, expected_artifact_hashes=_artifacts(),
    )
    assert verified == receipt

    target = private_root / "receipts" / "case_synthetic.json"
    assert sol.write_receipt(target, receipt) == target
    assert target.stat().st_mode & 0o777 == 0o600
    loaded = sol.read_receipt(
        target, expected_artifact_hashes=_artifacts(), require_approved=True,
    )
    assert loaded == receipt

    persisted = target.read_text(encoding="utf-8")
    assert raw_provider_response not in persisted
    assert receipt.response_sha256 in persisted
    assert sol.write_receipt(target, receipt) == target, "identical retry is idempotent"


def test_missing_or_loose_private_key_fails_closed(private_root: Path) -> None:
    with pytest.raises(sol.SolReviewKeyError, match="missing"):
        _issue()

    key = sol.attestation_key_path()
    key.write_bytes(b"synthetic-sol-review-key-material-32b")
    os.chmod(key, 0o644)
    with pytest.raises(sol.SolReviewKeyError, match="mode 600"):
        _issue()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reviewer_model", "gpt-5.6-mini", "reviewer_model"),
        ("reasoning_effort", "high", "reasoning_effort"),
    ],
)
def test_wrong_model_or_effort_is_rejected_before_attestation(
    provisioned_key: Path, field: str, value: str, message: str,
) -> None:
    with pytest.raises(sol.SolReviewError, match=message):
        _issue(**{field: value})


def test_tamper_and_expected_hash_mismatch_fail_closed(provisioned_key: Path) -> None:
    receipt = _issue()
    tampered = receipt.to_dict()
    tampered["artifact_hashes"]["yaml_sha256"] = _digest("other-yaml")
    with pytest.raises(sol.SolReviewIntegrityError, match="HMAC"):
        sol.verify_receipt(tampered)

    wrong_expected = replace(_artifacts(), bundle_sha256=_digest("other-bundle"))
    with pytest.raises(sol.SolReviewIntegrityError, match="frozen artifacts"):
        sol.require_approved_receipt(receipt, expected_artifact_hashes=wrong_expected)


def test_non_satisfied_cid_can_be_a_signed_rejection_but_not_gold(
    provisioned_key: Path,
) -> None:
    rejected = _issue(
        decision=sol.REJECTED,
        cid_verdicts=[sol.CidVerdict(cid="liquidity", verdict="violated")],
    )
    assert sol.verify_receipt(rejected) == rejected
    with pytest.raises(sol.SolReviewIntegrityError, match="not a clean approved"):
        sol.require_approved_receipt(rejected, expected_artifact_hashes=_artifacts())

    with pytest.raises(sol.SolReviewError, match="every CID"):
        _issue(cid_verdicts=[sol.CidVerdict(cid="liquidity", verdict="violated")])


def test_receipt_parser_rejects_extra_fields_and_loose_receipt_file(
    private_root: Path, provisioned_key: Path,
) -> None:
    receipt = _issue()
    payload = receipt.to_dict()
    payload["raw_response"] = "synthetic-only-but-forbidden"
    with pytest.raises(sol.SolReviewError, match="missing or unexpected"):
        sol.verify_receipt(payload)

    target = private_root / "loose.json"
    target.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
    os.chmod(target, 0o644)
    with pytest.raises(sol.SolReviewKeyError, match="mode 600"):
        sol.read_receipt(target)
