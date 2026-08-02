#!/usr/bin/env python3
"""A deterministic, privacy-preserving front door for frozen NL -> YAML checks.

This module deliberately does *not* convert natural language, call an LLM, or
collect market data.  It accepts exactly the evidence that can be replayed
locally:

* a request string, kept in memory only;
* a generated YAML answer plus reviewed, structured conditions; and
* one serialized ``cyqnt.input/v1`` bundle.

It then delegates the actual five-gate evaluation to :func:`gates.run_gates`.
The returned receipt is intentionally smaller than the gate report: it carries
only deterministic hashes and aggregate verdicts, never raw request text, YAML,
bundle content, candidate symbols, warnings, or gate error text.  That makes it
safe to persist beside a public dataset while the inputs remain in the private
store.

The CLI is equally bounded: it reads four local files, makes no network or LLM
calls, and writes one non-verbatim JSON receipt to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import gates
from . import schema as ledger_schema
from .capability import normalize_condition

__all__ = [
    "ConditionSummary",
    "FrozenEvaluationReceipt",
    "LedgerEvaluation",
    "RECEIPT_SCHEMA",
    "evaluate_and_write_ledger",
    "evaluate_frozen",
    "main",
]


RECEIPT_SCHEMA = "cyqnt.nl2yaml.frozen-evaluation/v1"
# The evaluator only needs compact YAML, JSON conditions, and one serialized
# input bundle.  A fixed limit prevents a CLI typo from treating an unbounded
# local export as an evaluator input while comfortably covering normal bundles.
MAX_LOCAL_INPUT_BYTES = 64 * 1024 * 1024


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any, *, label: str) -> bytes:
    """Encode JSON deterministically, rejecting values that cannot be frozen."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "%s must be JSON-serializable and contain only finite numbers" % label
        ) from exc
    return text.encode("utf-8")


def _json_clone(value: Any, *, label: str) -> Tuple[Any, bytes]:
    """Return a JSON-only copy plus the exact canonical bytes used for its hash."""
    payload = _canonical_json_bytes(value, label=label)
    return json.loads(payload.decode("utf-8")), payload


def _require_frozen_bundle(bundle: Mapping[str, Any]) -> Tuple[Dict[str, Any], bytes]:
    """Detach and minimally identify a persisted ``cyqnt.input/v1`` snapshot."""
    if not isinstance(bundle, Mapping):
        raise TypeError("bundle must be a mapping decoded from cyqnt.input/v1 JSON")
    copied, payload = _json_clone(dict(bundle), label="bundle")
    if copied.get("schema") != "cyqnt.input/v1":
        raise ValueError("bundle must have schema='cyqnt.input/v1'")
    if not isinstance(copied.get("frames"), dict) or not copied["frames"]:
        raise ValueError("frozen cyqnt.input/v1 bundle needs at least one frame")
    for field in ("snapshot_id", "decision_time"):
        if not str(copied.get(field) or "").strip():
            raise ValueError("frozen cyqnt.input/v1 bundle needs %s" % field)
    return copied, payload


def _require_structured_conditions(
    conditions: Sequence[Mapping[str, Any]],
) -> Tuple[Tuple[Dict[str, Any], ...], bytes]:
    """Reject non-JSON or leaky predicates before the strategy is executed."""
    if not isinstance(conditions, (list, tuple)):
        raise TypeError("conditions must be a JSON list or tuple of mappings")
    copied, payload = _json_clone(list(conditions), label="conditions")
    if not all(isinstance(item, dict) for item in copied):
        raise TypeError("each condition must be a JSON mapping")
    # ``normalize_condition`` is the existing privacy/schema guard.  Do this
    # before G1d so a forbidden `quote`/`user_id` key cannot trigger execution
    # before it is refused by G1e.
    for condition in copied:
        normalize_condition(condition)
    return tuple(copied), payload


def _answer_and_hash(answer: Any) -> Tuple[Any, str]:
    """Keep YAML text exact; canonicalize a caller-supplied parsed mapping."""
    if isinstance(answer, str):
        return answer, _sha256(answer.encode("utf-8"))
    if isinstance(answer, Mapping):
        copied, payload = _json_clone(dict(answer), label="YAML mapping")
        return copied, _sha256(payload)
    raise TypeError("yaml_answer must be YAML text or a parsed mapping")


def _passed_through(report: gates.GateReport, required: Sequence[str]) -> bool:
    """Whether every named gate was reached and passed (not merely skipped)."""
    by_gate = {result.gate: result.ok for result in report.results}
    return all(by_gate.get(gate) is True for gate in required)


@dataclass(frozen=True)
class ConditionSummary:
    """Aggregate G1e evidence with no raw condition values or user text."""

    requested_count: int
    evaluated_count: int
    verdict_counts: Tuple[Tuple[str, int], ...]
    silently_proxied_count: int
    undisclosed_assumption_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_count": self.requested_count,
            "evaluated_count": self.evaluated_count,
            "verdict_counts": dict(self.verdict_counts),
            "silently_proxied_count": self.silently_proxied_count,
            "undisclosed_assumption_count": self.undisclosed_assumption_count,
        }


@dataclass(frozen=True)
class FrozenEvaluationReceipt:
    """Serializable result of one local replay, containing no verbatim inputs."""

    schema: str
    nl_sha256: str
    yaml_sha256: str
    conditions_sha256: str
    bundle_sha256: str
    gate_status: str
    failed_gate: Optional[str]
    gate_statuses: Tuple[Tuple[str, str], ...]
    failure_signature: str
    # G1a--G1c only: the answer parses, compiles, and matches the requested
    # intent.  It deliberately says nothing about whether it ran.
    draft_valid: bool
    # G1a--G1d *and* a nonempty reviewed condition set.  The latter is ingress
    # evidence rather than a sixth gate: without it, a successful replay is only
    # a smoke test and must not be promoted to runtime-valid.
    runtime_valid: bool
    # Exact canonical acceptance invariant from GateReport, including G1e.
    gold_clean: bool
    conditions: ConditionSummary
    signal_batch_sha256: Optional[str]
    signal_count: Optional[int]
    selection_candidate_count: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe aggregate evidence; intentionally omit raw inputs."""
        return {
            "schema": self.schema,
            "nl_sha256": self.nl_sha256,
            "yaml_sha256": self.yaml_sha256,
            "conditions_sha256": self.conditions_sha256,
            "bundle_sha256": self.bundle_sha256,
            "gate_status": self.gate_status,
            "failed_gate": self.failed_gate,
            "gate_statuses": dict(self.gate_statuses),
            "failure_signature": self.failure_signature,
            "draft_valid": self.draft_valid,
            "runtime_valid": self.runtime_valid,
            "gold_clean": self.gold_clean,
            "conditions": self.conditions.to_dict(),
            "signal_batch_sha256": self.signal_batch_sha256,
            "signal_count": self.signal_count,
            "selection_candidate_count": self.selection_candidate_count,
        }


@dataclass(frozen=True)
class LedgerEvaluation:
    """One frozen replay plus the public, linked records it wrote.

    ``receipt`` is safe to publish; ``attempt`` deliberately retains YAML so it
    can go through the existing public-record privacy writer.  It never carries
    raw request text, a rendered prompt, gate error prose, warnings, or
    candidate explanation prose.  ``run`` is present exactly when G1d produced
    a frozen signal batch, including when G1e subsequently found a semantic
    failure.
    """

    receipt: FrozenEvaluationReceipt
    attempt: ledger_schema.AttemptRecord
    run: Optional[ledger_schema.RunRecord]


@dataclass(frozen=True)
class _FrozenEvaluation:
    """Private in-memory result used to avoid re-running G1d for the ledger."""

    receipt: FrozenEvaluationReceipt
    report: gates.GateReport
    bundle: Dict[str, Any]


def _evaluate_frozen(
    *,
    nl: str,
    yaml_answer: Any,
    conditions: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
) -> _FrozenEvaluation:
    """Run the frozen replay once and retain only in-memory execution evidence.

    The public receipt is constructed here, alongside the in-memory gate report
    needed by :func:`evaluate_and_write_ledger`.  Keeping the report private
    matters: its error strings and batch warnings can contain YAML-derived or
    source-derived prose and are therefore never passed to a public writer.
    """
    if not isinstance(nl, str) or not nl.strip():
        raise ValueError("nl must be non-empty request text")

    answer_copy, yaml_sha256 = _answer_and_hash(yaml_answer)
    conditions_copy, conditions_payload = _require_structured_conditions(conditions)
    bundle_copy, bundle_payload = _require_frozen_bundle(bundle)

    report = gates.run_gates(
        answer_copy,
        nl=nl,
        bundle=bundle_copy,
        conditions=conditions_copy,
    )

    counts = Counter(item.verdict for item in report.condition_verdicts)
    summary = ConditionSummary(
        requested_count=len(conditions_copy),
        evaluated_count=len(report.condition_verdicts),
        verdict_counts=tuple((name, counts.get(name, 0))
                             for name in gates.CONDITION_VERDICTS),
        silently_proxied_count=sum(
            item.silently_proxied for item in report.condition_verdicts
        ),
        undisclosed_assumption_count=sum(
            item.undisclosed_assumption for item in report.condition_verdicts
        ),
    )

    if report.batch is None:
        signal_batch_sha256 = None
        signal_count = None
        selection_candidate_count = None
    else:
        batch_payload = _canonical_json_bytes(report.batch, label="signal batch")
        signal_batch_sha256 = _sha256(batch_payload)
        signal_count = len(report.batch.get("signals") or ())
        selection_candidate_count = len(gates.selection_candidates(report.batch))

    receipt = FrozenEvaluationReceipt(
        schema=RECEIPT_SCHEMA,
        nl_sha256=_sha256(nl.encode("utf-8")),
        yaml_sha256=yaml_sha256,
        conditions_sha256=_sha256(conditions_payload),
        bundle_sha256=_sha256(bundle_payload),
        gate_status=report.status,
        failed_gate=report.failed_gate,
        gate_statuses=tuple((result.gate, result.status) for result in report.results),
        failure_signature=report.signature,
        draft_valid=_passed_through(report, ("G1a", "G1b", "G1c")),
        runtime_valid=(bool(conditions_copy)
                       and _passed_through(report, ("G1a", "G1b", "G1c", "G1d"))),
        gold_clean=report.clean,
        conditions=summary,
        signal_batch_sha256=signal_batch_sha256,
        signal_count=signal_count,
        selection_candidate_count=selection_candidate_count,
    )
    return _FrozenEvaluation(receipt=receipt, report=report, bundle=bundle_copy)


def evaluate_frozen(
    *,
    nl: str,
    yaml_answer: Any,
    conditions: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
) -> FrozenEvaluationReceipt:
    """Replay one NL/YAML conversion against one frozen ``cyqnt.input/v1`` input.

    This is the canonical public receipt entry point.  It accepts no LLM client,
    endpoint, live-data fallback, retry callback, or output path.  Callers that
    need conversion must do that separately, then hand this function the exact
    generated answer and a reviewed structured condition list.
    """
    return _evaluate_frozen(
        nl=nl,
        yaml_answer=yaml_answer,
        conditions=conditions,
        bundle=bundle,
    ).receipt


# ---------------------------------------------------------------------------
# Public attempt/run ledger bridge
# ---------------------------------------------------------------------------

# ``gates`` deliberately uses runtime-oriented labels (``run_error``), while
# the public dataset has a closed historical vocabulary (``execution_error``).
# Keep this table total for every status this one-shot evaluator can emit; do
# not turn an unknown string into a plausible record by guessing.
_GATE_STATUS_TO_LEDGER_GATE = {
    gates.STATUS_PASSED: ledger_schema.Gate.PASSED,
    "parse_error": ledger_schema.Gate.PARSE_ERROR,
    "static_invalid": ledger_schema.Gate.SCHEMA_INVALID,
    "dryrun_failed": ledger_schema.Gate.DRY_RUN_FAILED,
    "intent_mismatch": ledger_schema.Gate.INTENT_MISMATCH,
    "run_error": ledger_schema.Gate.EXECUTION_ERROR,
    "bundle_insufficient": ledger_schema.Gate.BUNDLE_INSUFFICIENT,
    "condition_violated": ledger_schema.Gate.CONDITION_VIOLATED,
    gates.STATUS_CONDITION_UNRESOLVED: ledger_schema.Gate.CONDITION_UNRESOLVED,
    "undisclosed_assumption": ledger_schema.Gate.UNDISCLOSED_ASSUMPTION,
}

_CONDITION_STATUS_TO_LEDGER_STATUS = {
    gates.SATISFIED: ledger_schema.ConditionCheckStatus.SATISFIED,
    gates.VIOLATED: ledger_schema.ConditionCheckStatus.VIOLATED,
    gates.UNVERIFIABLE: ledger_schema.ConditionCheckStatus.UNVERIFIABLE,
    gates.NOT_EXPRESSED: ledger_schema.ConditionCheckStatus.NOT_EXPRESSED,
}

_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_SYMBOL = re.compile(r"^[A-Z0-9_]{2,32}$")
_SAFE_GIT_SHA = re.compile(r"^[0-9a-f]{7,64}$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,63}$")


def _ledger_gate_for(report: gates.GateReport) -> ledger_schema.Gate:
    try:
        return _GATE_STATUS_TO_LEDGER_GATE[report.status]
    except KeyError as exc:
        # Never concatenate the unexpected status into this error: the bridge
        # is used at a public-write boundary and an unreviewed future status is
        # not a reason to echo source-derived text.
        raise RuntimeError("frozen evaluator received an unmapped gate status") from exc


def _safe_failure_token(report: gates.GateReport) -> Tuple[str, ...]:
    """A stable, deliberately uninformative replacement for gate error prose."""
    if report.status == gates.STATUS_PASSED:
        return ()
    # ``GateResult.errors`` can contain a YAML parser excerpt, a block error,
    # a candidate explanation, or a source error.  Persisting any of those in
    # attempts.jsonl would defeat the boundary even if the main receipt is safe.
    # The structured stopped_at_gate plus this fixed token are enough to group
    # failures through AttemptRecord.error_signature without exporting prose.
    # GateReport.signature already hashes error text; hash it once more so this
    # public field remains a fixed compact token even if the report format gains
    # a diagnostic component later.  That keeps distinct failures at the same
    # gate distinct for bookkeeping without serialising either error message.
    signature = _sha256(report.signature.encode("utf-8"))[:16]
    return ("frozen-evaluation-gate-%s-%s"
            % (_ledger_gate_for(report).value, signature),)


def _ledger_condition_verdicts(
    report: gates.GateReport,
) -> list[ledger_schema.ConditionVerdict]:
    """Convert G1e results without copying its explanations or offenders."""
    out: list[ledger_schema.ConditionVerdict] = []
    for item in report.condition_verdicts:
        cid = str(item.condition.id or "")
        if not _SAFE_IDENTIFIER.fullmatch(cid):
            raise ValueError("condition ids must be structured safe identifiers")

        # A silent proxy must remain visible in the ledger even if the proxy's
        # numeric predicate happened to pass.  ``not_expressed`` is similarly
        # first-class rather than being flattened into satisfied/unverifiable.
        if item.silently_proxied:
            # A proxy is the stronger downstream safety invariant: it forces
            # both strict and loose CaseRecord eligibility false.  The receipt
            # retains the independent G1e ``not_expressed`` aggregate, while
            # this one-status public ledger cannot represent both facts.
            status = ledger_schema.ConditionCheckStatus.PROXIED
        else:
            try:
                status = _CONDITION_STATUS_TO_LEDGER_STATUS[item.verdict]
            except KeyError as exc:
                raise RuntimeError("frozen evaluator received an unmapped condition status") from exc
        checked_by = (ledger_schema.CheckedBy.STATIC
                      if status is ledger_schema.ConditionCheckStatus.NOT_EXPRESSED
                      else ledger_schema.CheckedBy.EXECUTED)
        out.append(ledger_schema.ConditionVerdict(
            cid=cid,
            status=status,
            checked_by=checked_by,
        ))
    return out


def _decision_time_iso8601(value: Any) -> str:
    """Convert the input contract's epoch-ms decision clock to an audit instant."""
    if isinstance(value, bool):
        raise ValueError("frozen bundle decision_time is not a serialised epoch-ms value")
    try:
        epoch_ms = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("frozen bundle decision_time is not a serialised epoch-ms value") from exc
    if isinstance(value, float) and (not math.isfinite(value) or value != float(epoch_ms)):
        raise ValueError("frozen bundle decision_time is not a serialised epoch-ms value")
    try:
        instant = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("frozen bundle decision_time is outside the audit range") from exc
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _public_bundle_nodes(bundle: Mapping[str, Any]) -> list[ledger_schema.BundleNode]:
    """Summarise frame contents by hash without exporting arbitrary frame keys."""
    frames = bundle.get("frames")
    if not isinstance(frames, Mapping):
        raise ValueError("frozen execution bundle has no serialised frame map")

    # The contract's known frame names are safe protocol labels.  An unknown
    # frame name is represented by a positional label instead of being copied
    # into a public receipt; its content remains bound by the node and bundle
    # hashes below.
    from cyqnt_trd.standard_bot.data.input_bundle import FRAME_SHAPES

    known_names = frozenset(FRAME_SHAPES)
    nodes: list[ledger_schema.BundleNode] = []
    for position, name in enumerate(sorted(frames), start=1):
        frame = frames[name]
        if not isinstance(frame, Mapping):
            raise ValueError("frozen execution bundle has a non-object frame")
        rows = frame.get("rows")
        if not isinstance(rows, list):
            raise ValueError("frozen execution bundle has a non-array frame rows value")
        node_label = name if isinstance(name, str) and name in known_names else "frame_%d" % position
        node_payload = _canonical_json_bytes(dict(frame), label="bundle frame")
        nodes.append(ledger_schema.BundleNode(
            node=node_label,
            rows=len(rows),
            sha256=_sha256(node_payload),
        ))
    return nodes


def _public_candidates(batch: Mapping[str, Any]) -> list[ledger_schema.Candidate]:
    """Retain only structured basket fields; never candidate reasons/features."""
    candidates: list[ledger_schema.Candidate] = []
    for rank, raw in enumerate(gates.selection_candidates(batch), start=1):
        if not isinstance(raw, Mapping):
            raise ValueError("signal batch contains a non-object candidate")
        symbol = raw.get("symbol")
        direction = raw.get("direction")
        if not isinstance(symbol, str) or not _SAFE_SYMBOL.fullmatch(symbol):
            raise ValueError("signal batch contains a non-ledger-safe candidate symbol")
        if not isinstance(direction, str):
            raise ValueError("signal batch contains a non-ledger-safe candidate direction")
        try:
            side = ledger_schema.CandidateSide(direction.lower())
        except ValueError as exc:
            raise ValueError("signal batch contains an unsupported candidate direction") from exc

        score = raw.get("score")
        if score is not None:
            if isinstance(score, bool):
                raise ValueError("signal batch contains a non-numeric candidate score")
            try:
                score = float(score)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("signal batch contains a non-numeric candidate score") from exc
            if not math.isfinite(score):
                raise ValueError("signal batch contains a non-finite candidate score")
        # ``reason``, ``summary``, ``warnings``, nested trade proposals and
        # arbitrary ``features`` can all be prose.  The immutable batch hash
        # binds the full output privately; the public ledger keeps just the
        # machine-readable basket identity.
        candidates.append(ledger_schema.Candidate(
            rank=rank,
            symbol=symbol,
            side=side,
            score=score,
        ))
    return candidates


def _runtime_versions(
    python_version: Optional[str], pandas_version: Optional[str],
) -> Tuple[str, str]:
    if python_version is None:
        python_version = "%d.%d.%d" % sys.version_info[:3]
    if pandas_version is None:
        import pandas as pd
        pandas_version = str(pd.__version__)
    if (not isinstance(python_version, str) or not _SAFE_VERSION.fullmatch(python_version)
            or not isinstance(pandas_version, str) or not _SAFE_VERSION.fullmatch(pandas_version)):
        raise ValueError("runtime version fields must be compact safe identifiers")
    return python_version, pandas_version


def _linked_run(
    *,
    report: gates.GateReport,
    receipt: FrozenEvaluationReceipt,
    bundle: Mapping[str, Any],
    case_id: str,
    attempt_index: int,
    repo_git_sha: str,
    python_version: Optional[str],
    pandas_version: Optional[str],
) -> ledger_schema.RunRecord:
    """Build the public run half for an execution G1d has already completed."""
    if report.batch is None or receipt.signal_batch_sha256 is None or receipt.signal_count is None:
        raise ValueError("a linked run requires a completed frozen G1d batch")
    if not isinstance(repo_git_sha, str) or not _SAFE_GIT_SHA.fullmatch(repo_git_sha):
        raise ValueError("repo_git_sha must be a lowercase git SHA")
    py_version, pd_version = _runtime_versions(python_version, pandas_version)

    # The external snapshot identifier is deliberately not copied: it is part
    # of untrusted input and may be an operator's private label.  Both derived
    # IDs are deterministic handles to the immutable public hashes instead.
    snapshot_digest = _sha256(str(bundle.get("snapshot_id")).encode("utf-8"))
    run_identity = "\x00".join((case_id, str(attempt_index), receipt.signal_batch_sha256))
    return ledger_schema.RunRecord(
        # The same frozen batch can be used to evaluate two cases.  Its hash is
        # provenance, not a run primary key; fold the case/attempt link in so
        # both audit records can coexist without a collision.
        run_id="run_" + _sha256(run_identity.encode("utf-8"))[:32],
        bundle_id="bundle_" + receipt.bundle_sha256[:16],
        bundle_sha256=receipt.bundle_sha256,
        bundle_decision_time=_decision_time_iso8601(bundle.get("decision_time")),
        bundle_snapshot_id="snapshot_" + snapshot_digest[:16],
        repo_git_sha=repo_git_sha,
        python_version=py_version,
        pandas_version=pd_version,
        signal_count=receipt.signal_count,
        case_id=case_id,
        attempt_index=attempt_index,
        yaml_sha256=receipt.yaml_sha256,
        bundle_nodes=_public_bundle_nodes(bundle),
        signal_batch_sha256=receipt.signal_batch_sha256,
        candidates=_public_candidates(report.batch),
    )


def evaluate_and_write_ledger(
    *,
    case_id: str,
    attempt_index: int,
    prompt_sha256: str,
    source_text: str,
    nl: str,
    yaml_answer: str,
    conditions: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    attempt_path: Path | str,
    run_path: Path | str,
    repo_git_sha: str,
    sampling_purpose: ledger_schema.SamplingPurpose | str = ledger_schema.SamplingPurpose.CONVERT,
    python_version: Optional[str] = None,
    pandas_version: Optional[str] = None,
) -> LedgerEvaluation:
    """Evaluate frozen evidence once, then append privacy-safe attempt/run links.

    ``source_text`` is required to be the exact in-memory ``nl`` string.  The
    existing :func:`schema.write_attempt` writer receives it for its verbatim
    scan but never serialises it.  A G1e failure is still paired with a
    :class:`RunRecord` when G1d emitted a batch: the failed semantic judgement
    must stay auditable against the exact output rather than erasing the run.

    This bridge accepts only the content-addressed ``case_<16 hex>`` identity
    derived from the exact in-memory request.  It has no :class:`CaseRecord`
    or private-record evidence with which to bind a ULID/UUID to that request,
    so it deliberately rejects those otherwise-valid dataset identities rather
    than creating an unprovable attempt/run link.  A workflow that uses a
    non-content-addressed case id must establish that link before calling this
    helper (or use a separate, explicitly evidenced writer).

    This helper has no LLM client, endpoint, network fallback, retry callback,
    or live-data collection path.
    """
    if not isinstance(source_text, str) or source_text != nl:
        raise ValueError("source_text must exactly match the in-memory NL request")
    if not isinstance(yaml_answer, str):
        raise TypeError("ledger evaluation requires the exact YAML text answer")

    evaluated = _evaluate_frozen(
        nl=nl,
        yaml_answer=yaml_answer,
        conditions=conditions,
        bundle=bundle,
    )
    report = evaluated.report
    receipt = evaluated.receipt
    expected_case_id = ledger_schema.case_id_for(receipt.nl_sha256)
    if not isinstance(case_id, str) or case_id != expected_case_id:
        # Do not include the supplied or expected identifier in the error: this
        # public-write boundary must not become an accidental reflection path
        # for untrusted caller input.  (The identifier itself is safe today,
        # but keeping errors fixed preserves that property if the contract
        # changes.)
        raise ValueError(
            "ledger bridge requires the content-addressed case_id bound to the "
            "in-memory NL request"
        )
    attempt = ledger_schema.AttemptRecord(
        case_id=case_id,
        attempt_index=attempt_index,
        sampling_purpose=sampling_purpose,
        prompt_sha256=prompt_sha256,
        yaml_text=yaml_answer,
        stopped_at_gate=_ledger_gate_for(report),
        gate_errors=list(_safe_failure_token(report)),
        condition_verdicts=_ledger_condition_verdicts(report),
    )
    run = (_linked_run(
        report=report,
        receipt=receipt,
        bundle=evaluated.bundle,
        case_id=case_id,
        attempt_index=attempt_index,
        repo_git_sha=repo_git_sha,
        python_version=python_version,
        pandas_version=pandas_version,
    ) if report.batch is not None else None)

    # Validate all content before either append.  Attempts legitimately exist
    # without G1d runs, so if an OS failure occurs during the second append an
    # orphaned attempt is still honest; the reverse ordering would publish a
    # run that points at an attempt that never existed.
    ledger_schema.validate_record(attempt, "attempt")
    if run is not None:
        ledger_schema.validate_record(run, "run")
    ledger_schema.write_attempt(attempt_path, attempt, source_text=source_text)
    if run is not None:
        ledger_schema.write_run(run_path, run)
    return LedgerEvaluation(receipt=receipt, attempt=attempt, run=run)


def _read_local_file(path_text: str, *, label: str) -> bytes:
    """Read one regular local file under a fixed size bound."""
    path = Path(path_text).expanduser()
    try:
        if not path.is_file():
            raise ValueError("%s must be a regular file" % label)
        # Read at most one byte beyond the bound.  Checking the content read,
        # rather than only ``stat().st_size``, also holds if a file grows
        # between the metadata check and this open.
        with path.open("rb") as handle:
            payload = handle.read(MAX_LOCAL_INPUT_BYTES + 1)
    except OSError as exc:
        raise ValueError("%s file is unavailable" % label) from exc
    if len(payload) > MAX_LOCAL_INPUT_BYTES:
        raise ValueError("%s exceeds the local evaluator size limit" % label)
    return payload


def _read_utf8(path_text: str, *, label: str) -> str:
    try:
        return _read_local_file(path_text, label=label).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("%s must be UTF-8" % label) from exc


def _parse_json_file(path_text: str, *, label: str) -> Any:
    try:
        return json.loads(_read_utf8(path_text, label=label))
    except json.JSONDecodeError as exc:
        raise ValueError("%s must be valid JSON" % label) from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the local evaluator and emit exactly one safe JSON object on stdout."""
    parser = argparse.ArgumentParser(
        description="Evaluate local NL/YAML evidence against a frozen cyqnt.input/v1 bundle."
    )
    parser.add_argument("--nl-file", required=True,
                        help="UTF-8 request text file; its contents are never emitted")
    parser.add_argument("--yaml-file", required=True,
                        help="UTF-8 generated YAML file")
    parser.add_argument("--conditions-file", required=True,
                        help="JSON list of reviewed structured conditions")
    parser.add_argument("--bundle-file", required=True,
                        help="JSON cyqnt.input/v1 bundle")
    args = parser.parse_args(argv)

    # Do not print exception text: parser/gate errors can quote YAML and a
    # caller may have put private wording there.  The exit code distinguishes a
    # malformed local input from a complete receipt whose gate status is a
    # conversion failure.
    try:
        receipt = evaluate_frozen(
            nl=_read_utf8(args.nl_file, label="NL"),
            yaml_answer=_read_utf8(args.yaml_file, label="YAML"),
            conditions=_parse_json_file(args.conditions_file, label="conditions"),
            bundle=_parse_json_file(args.bundle_file, label="bundle"),
        )
    except Exception:  # noqa: BLE001 - never expose input-derived exception text
        print(json.dumps({"schema": RECEIPT_SCHEMA, "gate_status": "invalid_input"},
                         ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return 2

    print(json.dumps(receipt.to_dict(), ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
