"""Private, HMAC-bound *claimed-review* receipts.

This module is deliberately a small local integrity boundary, not an LLM client
or an external-provider attestation service.  A caller may submit only
fixed-format metadata and SHA-256 digests of the prompt, request, YAML, replay
artifacts, and a claimed provider response.  It cannot submit a raw provider
response, and this module neither makes network calls nor persists one.

Receipts are private: both their HMAC key and any receipt file must live beneath
``schema.internal_root()``.  The key is provisioned out of band at
``<internal-root>/sol_review_hmac.key``.  It is never auto-created: a missing,
non-regular, short, or non-``0600`` key is a fail-closed error.  Because a
worker which can read that key can also issue a receipt, this HMAC proves only
local record integrity and artifact binding.  It does **not** prove that
``gpt-5.6-sol`` (or any external provider) was invoked.

The distinction between cryptographic validity and promotion eligibility is
intentional.  A signed ``rejected`` receipt (or one with a non-satisfied CID)
is useful private audit evidence.  Even a locally ``approved`` receipt is not
an SFT/gold-release prerequisite; that needs an independently controlled,
provider-attesting adapter that this module does not implement.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from . import schema


__all__ = [
    "ArtifactHashes",
    "CidVerdict",
    "SolReviewError",
    "SolReviewIntegrityError",
    "SolReviewKeyError",
    "SolReviewReceipt",
    "APPROVED",
    "REJECTED",
    "EXPECTED_REASONING_EFFORT",
    "EXPECTED_REVIEWER_MODEL",
    "HMAC_KEY_FILENAME",
    "RECEIPT_SCHEMA",
    "attestation_key_path",
    "issue_receipt",
    "read_receipt",
    "require_approved_receipt",
    "verify_receipt",
    "write_receipt",
]


RECEIPT_SCHEMA = "cyqnt.nl2yaml.sol-review/v1"
EXPECTED_REVIEWER_MODEL = "gpt-5.6-sol"
EXPECTED_REASONING_EFFORT = "xhigh"
APPROVED = "approved"
REJECTED = "rejected"
HMAC_KEY_FILENAME = "sol_review_hmac.key"

_ATTESTATION_DOMAIN = b"cyqnt.nl2yaml.sol-review/v1\x00"
_HMAC_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_CID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,191}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_MAX_KEY_BYTES = 4096
_MIN_KEY_BYTES = 32
_MAX_RECEIPT_BYTES = 128 * 1024
_VALID_DECISIONS = frozenset((APPROVED, REJECTED))
_VALID_CID_VERDICTS = frozenset(status.value for status in schema.ConditionCheckStatus)


class SolReviewError(schema.RecordError):
    """A Sol review receipt is malformed or outside the trusted boundary."""


class SolReviewKeyError(SolReviewError):
    """The private HMAC key is absent or does not meet its security contract."""


class SolReviewIntegrityError(SolReviewError):
    """A receipt fails HMAC verification or disagrees with expected artifacts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SolReviewError(message)


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HMAC_HEX_RE.fullmatch(value) is None:
        raise SolReviewError("%s must be a lowercase SHA-256 hex digest" % label)
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    # Do not echo unknown values or keys: callers can receive a raw, untrusted
    # mapping from an adapter and exception strings frequently end up in logs.
    if set(value) != expected:
        raise SolReviewError("%s has missing or unexpected fields" % label)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value), ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SolReviewError("receipt payload must be finite JSON") from exc


def _canonical_timestamp(value: Any) -> str:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise SolReviewError("issued_at must be canonical UTC RFC3339 seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SolReviewError("issued_at must be a valid UTC timestamp") from exc
    canonical = parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
    if value != canonical:
        raise SolReviewError("issued_at must be canonical UTC RFC3339 seconds")
    return value


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ArtifactHashes:
    """All replay artifacts a reviewer must bind before signing a receipt.

    Each value is a SHA-256 digest, never an artifact body.  ``repo_sha256`` is
    the SHA-256 of the repository/replay manifest chosen by the caller rather
    than a loose branch name or mutable Git ref.
    """

    review_prompt_sha256: str
    request_sha256: str
    yaml_sha256: str
    conditions_sha256: str
    bundle_sha256: str
    attempt_sha256: str
    run_sha256: str
    signal_sha256: str
    capability_sha256: str
    repo_sha256: str

    _FIELDS = frozenset({
        "review_prompt_sha256", "request_sha256", "yaml_sha256", "conditions_sha256",
        "bundle_sha256", "attempt_sha256", "run_sha256", "signal_sha256",
        "capability_sha256", "repo_sha256",
    })

    def __post_init__(self) -> None:
        for field_name in self._FIELDS:
            _require_sha256(getattr(self, field_name), label=field_name)

    def to_dict(self) -> Dict[str, str]:
        return {
            "review_prompt_sha256": self.review_prompt_sha256,
            "request_sha256": self.request_sha256,
            "yaml_sha256": self.yaml_sha256,
            "conditions_sha256": self.conditions_sha256,
            "bundle_sha256": self.bundle_sha256,
            "attempt_sha256": self.attempt_sha256,
            "run_sha256": self.run_sha256,
            "signal_sha256": self.signal_sha256,
            "capability_sha256": self.capability_sha256,
            "repo_sha256": self.repo_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactHashes":
        if not isinstance(value, Mapping):
            raise SolReviewError("artifact_hashes must be an object")
        _require_exact_keys(value, cls._FIELDS, label="artifact_hashes")
        return cls(**dict(value))


@dataclass(frozen=True)
class CidVerdict:
    """Closed, non-verbatim verdict for one reviewed condition identifier."""

    cid: str
    verdict: str
    proxy_used: bool = False
    assumption_used: bool = False

    _FIELDS = frozenset({"cid", "verdict", "proxy_used", "assumption_used"})

    def __post_init__(self) -> None:
        if not isinstance(self.cid, str) or _CID_RE.fullmatch(self.cid) is None:
            raise SolReviewError("cid must be a compact identifier")
        if self.verdict not in _VALID_CID_VERDICTS:
            raise SolReviewError("cid verdict is not a supported closed status")
        if type(self.proxy_used) is not bool or type(self.assumption_used) is not bool:
            raise SolReviewError("cid proxy_used and assumption_used must be booleans")

    @property
    def satisfied_without_proxy_or_assumption(self) -> bool:
        return (
            self.verdict == schema.ConditionCheckStatus.SATISFIED.value
            and not self.proxy_used
            and not self.assumption_used
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid,
            "verdict": self.verdict,
            "proxy_used": self.proxy_used,
            "assumption_used": self.assumption_used,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CidVerdict":
        if not isinstance(value, Mapping):
            raise SolReviewError("cid verdict must be an object")
        _require_exact_keys(value, cls._FIELDS, label="cid verdict")
        return cls(**dict(value))


def _coerce_artifact_hashes(value: Union[ArtifactHashes, Mapping[str, Any]]) -> ArtifactHashes:
    if isinstance(value, ArtifactHashes):
        return value
    if isinstance(value, Mapping):
        return ArtifactHashes.from_dict(value)
    raise SolReviewError("artifact_hashes must be an ArtifactHashes or object")


def _coerce_cid_verdicts(
    values: Sequence[Union[CidVerdict, Mapping[str, Any]]],
) -> Tuple[CidVerdict, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SolReviewError("cid_verdicts must be a non-empty sequence")
    out = []
    for value in values:
        if isinstance(value, CidVerdict):
            out.append(value)
        elif isinstance(value, Mapping):
            out.append(CidVerdict.from_dict(value))
        else:
            raise SolReviewError("each cid verdict must be an object")
    if not out:
        raise SolReviewError("cid_verdicts must not be empty")
    out.sort(key=lambda item: item.cid)
    if len({item.cid for item in out}) != len(out):
        raise SolReviewError("cid_verdicts contains duplicate identifiers")
    return tuple(out)


@dataclass(frozen=True)
class SolReviewReceipt:
    """A private signed review outcome, with no request or response body."""

    schema: str
    reviewer_model: str
    reasoning_effort: str
    decision: str
    cid_verdicts: Tuple[CidVerdict, ...]
    artifact_hashes: ArtifactHashes
    provider_id: str
    response_sha256: str
    issued_at: str
    attestation_hmac_sha256: str

    _FIELDS = frozenset({
        "schema", "reviewer_model", "reasoning_effort", "decision", "cid_verdicts",
        "artifact_hashes", "provider_id", "response_sha256", "issued_at",
        "attestation_hmac_sha256",
    })

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise SolReviewError("receipt schema is unsupported")
        if self.reviewer_model != EXPECTED_REVIEWER_MODEL:
            raise SolReviewError("reviewer_model must be exactly gpt-5.6-sol")
        if self.reasoning_effort != EXPECTED_REASONING_EFFORT:
            raise SolReviewError("reasoning_effort must be exactly xhigh")
        if self.decision not in _VALID_DECISIONS:
            raise SolReviewError("decision must be approved or rejected")
        verdicts = _coerce_cid_verdicts(self.cid_verdicts)
        object.__setattr__(self, "cid_verdicts", verdicts)
        artifacts = _coerce_artifact_hashes(self.artifact_hashes)
        object.__setattr__(self, "artifact_hashes", artifacts)
        if not isinstance(self.provider_id, str) or _PROVIDER_ID_RE.fullmatch(self.provider_id) is None:
            raise SolReviewError("provider_id must be a compact opaque identifier")
        _require_sha256(self.response_sha256, label="response_sha256")
        object.__setattr__(self, "issued_at", _canonical_timestamp(self.issued_at))
        _require_sha256(self.attestation_hmac_sha256, label="attestation_hmac_sha256")
        if self.decision == APPROVED and not self.is_clean_approval:
            raise SolReviewError(
                "approved receipt requires every CID to be satisfied without proxy or assumption"
            )

    @property
    def is_clean_approval(self) -> bool:
        return self.decision == APPROVED and all(
            verdict.satisfied_without_proxy_or_assumption for verdict in self.cid_verdicts
        )

    def unsigned_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "reviewer_model": self.reviewer_model,
            "reasoning_effort": self.reasoning_effort,
            "decision": self.decision,
            "cid_verdicts": [item.to_dict() for item in self.cid_verdicts],
            "artifact_hashes": self.artifact_hashes.to_dict(),
            "provider_id": self.provider_id,
            "response_sha256": self.response_sha256,
            "issued_at": self.issued_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self.unsigned_dict()
        payload["attestation_hmac_sha256"] = self.attestation_hmac_sha256
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SolReviewReceipt":
        if not isinstance(value, Mapping):
            raise SolReviewError("receipt must be an object")
        _require_exact_keys(value, cls._FIELDS, label="receipt")
        raw_verdicts = value["cid_verdicts"]
        if not isinstance(raw_verdicts, list):
            raise SolReviewError("cid_verdicts must be a JSON array")
        return cls(
            schema=value["schema"],
            reviewer_model=value["reviewer_model"],
            reasoning_effort=value["reasoning_effort"],
            decision=value["decision"],
            cid_verdicts=_coerce_cid_verdicts(raw_verdicts),
            artifact_hashes=_coerce_artifact_hashes(value["artifact_hashes"]),
            provider_id=value["provider_id"],
            response_sha256=value["response_sha256"],
            issued_at=value["issued_at"],
            attestation_hmac_sha256=value["attestation_hmac_sha256"],
        )


def attestation_key_path() -> Path:
    """Return the only permitted key location; this function never creates it."""
    return schema.internal_root() / HMAC_KEY_FILENAME


def _open_regular_file(path: Path, *, label: str, max_bytes: int) -> bytes:
    """Read a non-symlink regular file with a bounded payload.

    ``O_NOFOLLOW`` plus the lstat/fstat identity check avoids accepting a key or
    receipt redirected through a symlink between validation and open.
    """
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise SolReviewKeyError("%s is missing" % label) from exc
    except OSError as exc:
        raise SolReviewKeyError("%s is unavailable" % label) from exc
    if not stat.S_ISREG(before.st_mode):
        raise SolReviewKeyError("%s must be a regular file" % label)
    if stat.S_IMODE(before.st_mode) != 0o600:
        raise SolReviewKeyError("%s must have mode 600" % label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise SolReviewKeyError("%s cannot be opened safely" % label) from exc
    try:
        after = os.fstat(descriptor)
        if (not stat.S_ISREG(after.st_mode)
                or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
                or stat.S_IMODE(after.st_mode) != 0o600):
            raise SolReviewKeyError("%s changed while opening" % label)
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(8192, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise SolReviewKeyError("%s exceeds its size limit" % label)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_hmac_key() -> bytes:
    key = _open_regular_file(
        attestation_key_path(), label="Sol review HMAC key", max_bytes=_MAX_KEY_BYTES
    )
    if len(key) < _MIN_KEY_BYTES:
        raise SolReviewKeyError("Sol review HMAC key is too short")
    return key


def _attestation_for(key: bytes, receipt: SolReviewReceipt) -> str:
    payload = _ATTESTATION_DOMAIN + _canonical_json_bytes(receipt.unsigned_dict())
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def issue_receipt(
    *,
    artifact_hashes: Union[ArtifactHashes, Mapping[str, Any]],
    cid_verdicts: Sequence[Union[CidVerdict, Mapping[str, Any]]],
    decision: str,
    provider_id: str,
    response_sha256: str,
    issued_at: Optional[str] = None,
    reviewer_model: str = EXPECTED_REVIEWER_MODEL,
    reasoning_effort: str = EXPECTED_REASONING_EFFORT,
) -> SolReviewReceipt:
    """Issue an in-memory local audit receipt from already-hashed data.

    This function intentionally has no ``response`` or ``raw_response``
    argument.  The adapter must hash a provider response in memory and pass only
    ``response_sha256`` here.  It also cannot choose a model or effort outside
    the two fixed values enforced by :class:`SolReviewReceipt`.  This is *not*
    evidence that a provider made the claimed response: any local worker with
    the HMAC key can issue it.  Keep it for private tamper detection and replay
    binding only.
    """
    draft = SolReviewReceipt(
        schema=RECEIPT_SCHEMA,
        reviewer_model=reviewer_model,
        reasoning_effort=reasoning_effort,
        decision=decision,
        cid_verdicts=_coerce_cid_verdicts(cid_verdicts),
        artifact_hashes=_coerce_artifact_hashes(artifact_hashes),
        provider_id=provider_id,
        response_sha256=response_sha256,
        issued_at=_now_utc() if issued_at is None else issued_at,
        # Construction validates the entire unsigned payload before the key is
        # touched.  This placeholder is replaced immediately below.
        attestation_hmac_sha256="0" * 64,
    )
    return replace(draft, attestation_hmac_sha256=_attestation_for(_read_hmac_key(), draft))


def _coerce_receipt(value: Union[SolReviewReceipt, Mapping[str, Any]]) -> SolReviewReceipt:
    if isinstance(value, SolReviewReceipt):
        return value
    if isinstance(value, Mapping):
        return SolReviewReceipt.from_dict(value)
    raise SolReviewError("receipt must be a SolReviewReceipt or object")


def verify_receipt(
    receipt: Union[SolReviewReceipt, Mapping[str, Any]],
    *,
    expected_artifact_hashes: Optional[Union[ArtifactHashes, Mapping[str, Any]]] = None,
    require_approved: bool = False,
) -> SolReviewReceipt:
    """Verify the private HMAC and, optionally, exact replay-artifact binding.

    ``require_approved`` is intentionally opt-in: signed rejected receipts are
    still valid audit facts.  An approved result remains local audit evidence,
    not external model provenance or a strict SFT/gold gate.
    """
    parsed = _coerce_receipt(receipt)
    actual = _attestation_for(_read_hmac_key(), parsed)
    if not hmac.compare_digest(actual, parsed.attestation_hmac_sha256):
        raise SolReviewIntegrityError("Sol review receipt HMAC verification failed")
    if expected_artifact_hashes is not None:
        expected = _coerce_artifact_hashes(expected_artifact_hashes)
        if parsed.artifact_hashes != expected:
            raise SolReviewIntegrityError("Sol review receipt does not match frozen artifacts")
    if require_approved and not parsed.is_clean_approval:
        raise SolReviewIntegrityError("Sol review receipt is not a clean approved review")
    return parsed


def require_approved_receipt(
    receipt: Union[SolReviewReceipt, Mapping[str, Any]],
    *,
    expected_artifact_hashes: Union[ArtifactHashes, Mapping[str, Any]],
) -> SolReviewReceipt:
    """Return a verified local clean-approval audit receipt.

    This checks record integrity and artifact binding only.  It cannot satisfy
    a strict SFT/gold gate because the signing key is available to local
    workers; callers need a separately controlled provider-attesting boundary
    for that purpose.
    """
    return verify_receipt(
        receipt,
        expected_artifact_hashes=expected_artifact_hashes,
        require_approved=True,
    )


def _private_receipt_path(path: Union[str, Path], *, must_exist: bool) -> Path:
    root = schema.internal_root().resolve()
    raw = Path(path).expanduser()
    target = raw.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise schema.PrivacyError("Sol review receipt must be beneath the internal root") from exc
    if target == root:
        raise schema.PrivacyError("Sol review receipt path must name a file")
    if must_exist:
        if not target.exists():
            raise SolReviewError("Sol review receipt is missing")
        # `_open_regular_file` enforces regular-file and 0600 semantics.
    else:
        if target.exists() and not target.is_file():
            raise SolReviewError("Sol review receipt target must be a regular file")
        current = root
        for part in target.parent.relative_to(root).parts:
            current /= part
            current.mkdir(exist_ok=True)
            os.chmod(current, 0o700)
    return target


def _strict_json_object(data: bytes) -> Dict[str, Any]:
    def no_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SolReviewError("receipt JSON contains duplicate fields")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SolReviewError("Sol review receipt must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SolReviewError("Sol review receipt must be one JSON object")
    return value


def read_receipt(
    path: Union[str, Path],
    *,
    expected_artifact_hashes: Optional[Union[ArtifactHashes, Mapping[str, Any]]] = None,
    require_approved: bool = False,
) -> SolReviewReceipt:
    """Read a private mode-600 receipt and verify it before returning it."""
    target = _private_receipt_path(path, must_exist=True)
    payload = _strict_json_object(
        _open_regular_file(target, label="Sol review receipt", max_bytes=_MAX_RECEIPT_BYTES)
    )
    return verify_receipt(
        SolReviewReceipt.from_dict(payload),
        expected_artifact_hashes=expected_artifact_hashes,
        require_approved=require_approved,
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        offset += os.write(descriptor, payload[offset:])


def write_receipt(path: Union[str, Path], receipt: Union[SolReviewReceipt, Mapping[str, Any]]) -> Path:
    """Persist a verified receipt privately, refusing overwrite conflicts.

    New files use ``O_EXCL`` and mode ``0600`` before their first byte is
    written.  Retrying an identical receipt is idempotent; a different receipt
    at the same path is rejected instead of silently overwriting audit history.
    """
    parsed = verify_receipt(receipt)
    target = _private_receipt_path(path, must_exist=False)
    encoded = _canonical_json_bytes(parsed.to_dict()) + b"\n"
    try:
        descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = read_receipt(target)
        if existing.to_dict() != parsed.to_dict():
            raise SolReviewIntegrityError("refusing to overwrite a different Sol review receipt")
        return target
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        try:
            target.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
        os.chmod(target, 0o600)
    return target
