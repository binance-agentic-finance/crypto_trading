"""Curate one private strategy-test queue row into an auditable ledger case.

This is intentionally the bridge *after* conversion, not another LLM client.
An operator supplies a private reviewed YAML artifact, a private review JSON,
and (for gold promotion) a private, content-bound Sol review receipt.  The
worker binds them to exactly one private queue row, replays the YAML against one
frozen ``cyqnt.input/v1`` bundle, and writes only privacy-safe ledger records.
It never sends a chat excerpt, prompt, YAML, or bundle to a network endpoint.

The review JSON is private and has this compact contract::

    {
      "schema": "cyqnt.nl2yaml.queue-review/v2",
      "queue_row_id": "r_...",
      "user_text_sha256": "...",
      "yaml_sha256": "...",
      "bundle_sha256": "...",
      "prompt_sha256": "...",
      "sol_review_prompt_sha256": "...",
      "gate_conditions": [{"id": "...", ...}],
      "case_conditions": [{"cid": "...", ...}],
      "capability_map": [{"cid": "...", "verdict": "supported"}],
      "intent_slots": {"intent": "COIN_SELECTION", ...},
      "tier": "t1_expressible",
      "source_snapshot_id": "...",
      "converter_model": "...", "temperature": 0.0,
      "resolution_path": "big_model",
      "conditions_with_quotes": [{"cid": "...", "quote": "..."}],
      "promote_to_gold": true,
      "human_reviewed_by": "reviewer_alias",
      "human_reviewed_at": "2026-08-02T12:00:00Z"
    }

All condition IDs must agree across the three structured condition views, and
each quote must be an exact substring of the private request.  Only a clear,
fully ``supported`` T1 case may pass this conversion bridge.  A model review is
not a free-text claim in this JSON: ``--sol-review`` may point to a private,
HMAC-bound audit receipt for the exact frozen artifacts.  Since this worker can
read the local signing key, that receipt is *not* proof that ``gpt-5.6-sol`` (or
any external provider) was called and it cannot unlock strict SFT/gold
promotion.  It never replaces an independent human signoff; a separately
controlled provider-attesting adapter is also required before this bridge can
promote a gold.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from . import evaluate
from . import gates
from . import mine
from . import schema
from . import sol_review


REPO = Path(__file__).resolve().parents[2]
PUBLIC_DATASET_ROOT = REPO / "tools" / "nl2yaml" / "dataset"
REVIEW_SCHEMA = "cyqnt.nl2yaml.queue-review/v2"
CURATION_RECEIPT_SCHEMA = "cyqnt.nl2yaml.queue-curation/v2"
DEFAULT_INTERNAL_CASES = "cases_queue_internal.jsonl"
DEFAULT_PRIVATE_RECEIPT = "queue_curation.jsonl"
DEFAULT_PRIVATE_DIAGNOSTICS = "queue_evaluation_diagnostics.jsonl"
DEFAULT_PRIVATE_OUTCOMES = "queue_curation_outcomes.jsonl"
_CLAIM_DIRECTORY = "queue_claims"
MAX_INPUT_BYTES = evaluate.MAX_LOCAL_INPUT_BYTES

# ``sol_review`` deliberately stores its HMAC key beneath the same configured
# private root that this worker can access.  It is therefore useful for local
# tamper detection and artifact binding, but cannot attest that an external
# model actually performed a review.  Keep strict promotion closed until a
# separate provider-attesting adapter/signing boundary is introduced.
_LOCAL_HMAC_GOLD_BLOCK = (
    "strict SFT/gold promotion is unavailable: a local HMAC review receipt is "
    "private integrity evidence, not independent external gpt-5.6-sol "
    "attestation; require a provider-attesting adapter with a signing key "
    "unavailable to this worker"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROW_ID = re.compile(r"^r_[0-9a-f]{16}$")
_SAFE_METADATA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,127}$")
_SAFE_REVIEWER = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
_SAFE_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_TIMEFRAME = re.compile(r"^\d{1,5}(?:m|h|d|w)$")
_SAFE_SYMBOL = re.compile(r"^(?=.*[A-Z])[A-Z0-9]{2,20}$")
_MODEL_MARKER = re.compile(r"(?:gpt|claude|gemini|llm|model|openai|anthropic)", re.I)
_SENSITIVE_TEXT = re.compile(
    r"(?ix)(?:\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|"
    r"\b(?:api[_-]?key|access[_-]?token|auth(?:orization)?|bearer|secret|"
    r"password|cookie)\s*(?:=|:)\s*\S+)"
)

_QUEUE_COLUMNS = frozenset({
    "queue_rank", "cluster_id", "row_id", "dup_count", "month", "lang",
    "user_id", "first_query", "user_text_excerpt",
})
_REVIEW_REQUIRED = frozenset({
    "schema", "queue_row_id", "user_text_sha256", "yaml_sha256", "bundle_sha256", "prompt_sha256",
    "sol_review_prompt_sha256",
    "gate_conditions", "case_conditions", "capability_map", "intent_slots", "tier",
    "source_snapshot_id", "converter_model", "temperature", "resolution_path",
    "conditions_with_quotes", "promote_to_gold",
})
_REVIEW_OPTIONAL = frozenset({
    "seed", "zh_variant", "human_reviewed_by", "human_reviewed_at",
})


@dataclass(frozen=True)
class QueueCurationResult:
    """Non-verbatim summary safe to print or retain beside the public ledger."""

    case_id: str
    queue_row_id: str
    gate_status: str
    gold_promoted: bool
    run_written: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": CURATION_RECEIPT_SCHEMA,
            "case_id": self.case_id,
            "queue_row_id": self.queue_row_id,
            "gate_status": self.gate_status,
            "gold_promoted": self.gold_promoted,
            "run_written": self.run_written,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("private review must be finite JSON") from exc
    return _sha256_bytes(encoded)


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _private_root() -> Path:
    root = schema.internal_root().resolve()
    repo_root = REPO.resolve()
    if _is_beneath(root, repo_root):
        raise schema.PrivacyError("private queue root resolves inside the repository")
    return root


def _prepare_private_parent(target: Path, root: Path) -> None:
    """Create only proven-contained private parents with restrictive modes."""
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    current = root
    for part in target.parent.relative_to(root).parts:
        current /= part
        current.mkdir(exist_ok=True)
        os.chmod(current, 0o700)


def _private_path(path: Path | str, *, label: str, must_exist: bool) -> Path:
    """Resolve a private input/output without accepting a repo or symlink escape."""
    root = _private_root()
    raw = Path(path).expanduser()
    target = raw.resolve()
    repo_root = REPO.resolve()
    if target == root or not _is_beneath(target, root) or _is_beneath(target, repo_root):
        raise schema.PrivacyError("private %s must be beneath the configured internal root" % label)
    if must_exist:
        if not target.is_file():
            raise schema.PrivacyError("private %s must be a regular file" % label)
        if target.stat().st_mode & 0o777 != 0o600:
            raise schema.PrivacyError("private %s must have mode 600" % label)
    else:
        if target.exists() and not target.is_file():
            raise schema.PrivacyError("private %s target must be a regular file" % label)
        _prepare_private_parent(target, root)
        if target.exists():
            os.chmod(target, 0o600)
    return target


def _public_path(root: Path | str, name: str) -> Path:
    """Keep every public append in one explicitly public output directory."""
    public_root = Path(root).expanduser().resolve()
    private_root = _private_root()
    if _is_beneath(public_root, private_root):
        raise schema.PrivacyError("public ledger directory cannot be inside the private root")
    public_root.mkdir(parents=True, exist_ok=True)
    target = (public_root / name).resolve()
    if not _is_beneath(target, public_root) or _is_beneath(target, private_root):
        raise schema.PrivacyError("public ledger output escapes its configured directory")
    if target.exists() and not target.is_file():
        raise ValueError("public ledger output must be a regular file")
    return target


def _read_limited(path: Path, *, label: str) -> bytes:
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise ValueError("%s is unavailable" % label) from exc
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError("%s exceeds the local input limit" % label)
    return payload


def _read_private_text(path: Path | str, *, label: str) -> tuple[Path, str]:
    target = _private_path(path, label=label, must_exist=True)
    try:
        return target, _read_limited(target, label=label).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("private %s must be UTF-8" % label) from exc


def _read_private_json(path: Path | str, *, label: str) -> tuple[Path, Dict[str, Any]]:
    target, text = _read_private_text(path, label=label)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("private %s must be valid JSON" % label) from exc
    if not isinstance(value, dict):
        raise ValueError("private %s must be one JSON object" % label)
    return target, value


def _read_bundle(path: Path | str) -> Dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise ValueError("frozen bundle must be a regular file")
    try:
        value = json.loads(_read_limited(target, label="frozen bundle").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen bundle must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("frozen bundle must be a JSON object")
    return value


def _queue_row(queue_path: Path, row_id: str) -> Dict[str, str]:
    if not isinstance(row_id, str) or not _ROW_ID.fullmatch(row_id):
        raise ValueError("private queue row id is invalid")
    try:
        with queue_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = frozenset(reader.fieldnames or ())
            if not _QUEUE_COLUMNS <= fields:
                raise ValueError("private queue lacks its required schema")
            matches = [dict(row) for row in reader if row.get("row_id") == row_id]
    except csv.Error as exc:
        raise ValueError("private queue is malformed") from exc
    if len(matches) != 1:
        raise ValueError("private queue row must resolve exactly once")
    return matches[0]


def _request_from_row(row: Mapping[str, str]) -> str:
    """Use the exact excerpt once; never silently concatenate a duplicate turn."""
    excerpt = str(row.get("user_text_excerpt") or "").strip()
    first = str(row.get("first_query") or "").strip()
    if not excerpt:
        raise ValueError("private queue row has no request excerpt")
    if first:
        canonical_first = mine.canonicalise(first)
        canonical_excerpt = mine.canonicalise(excerpt)
        if not canonical_first or not canonical_excerpt.startswith(canonical_first):
            raise ValueError("private queue first query and excerpt do not share the same source")
    return excerpt


def _require_safe_metadata(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_METADATA.fullmatch(value):
        raise ValueError("%s must be a compact safe identifier" % label)
    return value


def _private_metadata(value: Any, *, label: str) -> str:
    """Accept bounded private metadata only; public records receive a digest."""
    if (not isinstance(value, str) or not value.strip() or len(value) > 512
            or "\x00" in value or "\n" in value or "\r" in value):
        raise ValueError("%s must be bounded one-line private metadata" % label)
    return value


def _public_metadata_label(prefix: str, private_value: str) -> str:
    """Prevent a review-side label from becoming a dictionary-reversible leak."""
    # A plain hash of a small reviewer alias or snapshot label is trivially
    # reversible.  Reuse the internal HMAC salt that already protects public
    # user pseudonyms, but keep a distinct public field prefix for each domain.
    domain_value = "metadata:%s\x00%s" % (prefix, private_value)
    return "%s_%s" % (prefix, schema.hmac_pseudonym(domain_value)[4:])


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError("%s must be a SHA-256 digest" % label)
    return value


def _verified_repo_git_sha(claimed_sha: str) -> str:
    """Bind a run record to the actual clean checkout that executed it."""
    if not isinstance(claimed_sha, str) or not _SAFE_GIT_SHA.fullmatch(claimed_sha):
        raise ValueError("repo_git_sha must be the full lowercase local HEAD SHA")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("local repository provenance is unavailable") from exc
    if head != claimed_sha:
        raise ValueError("repo_git_sha does not match the local HEAD")
    if dirty:
        raise ValueError("refusing to publish a run from a dirty checkout")
    return head


def _require_timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("%s is required" % label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("%s must be an ISO8601 instant" % label) from exc
    if parsed.tzinfo is None:
        raise ValueError("%s must include a timezone" % label)
    return value


def _human_reviewer_pair(review: Mapping[str, Any], *,
                         required: bool = False) -> tuple[Optional[str], Optional[str]]:
    """Validate the independent human signoff; model claims use a typed receipt."""
    reviewer = review.get("human_reviewed_by")
    reviewed_at = review.get("human_reviewed_at")
    if bool(reviewer) != bool(reviewed_at):
        raise ValueError("human reviewer identity and timestamp must appear together")
    if required and not reviewer:
        raise ValueError("gold promotion requires an actual human reviewer")
    if not reviewer:
        return None, None
    if not isinstance(reviewer, str) or not _SAFE_REVIEWER.fullmatch(reviewer):
        raise ValueError("human reviewer must be a compact reviewer alias")
    if _MODEL_MARKER.search(reviewer):
        raise ValueError("a model reviewer cannot be recorded as a human signer")
    return str(reviewer), _require_timestamp(reviewed_at, label="human reviewed_at")


_COMPARE_OPERATORS = {
    ">": schema.Operator.GT,
    ">=": schema.Operator.GTE,
    "<": schema.Operator.LT,
    "<=": schema.Operator.LTE,
    "==": schema.Operator.EQ,
}
_GATE_GRANULARITY = {
    "cross_section": schema.EvaluationGranularity.UNIVERSE,
    "per_symbol_series": schema.EvaluationGranularity.PER_SYMBOL,
    "per_candidate_series": schema.EvaluationGranularity.PER_SYMBOL,
}
_DESC_WORDS = ("desc", "descending", "highest", "most", "由高到低", "從高到低", "最高", "最多")
_ASC_WORDS = ("asc", "ascending", "lowest", "least", "由低到高", "從低到高", "最低", "最少")
_NUMERIC_LITERAL = re.compile(
    r"(?<![A-Za-z0-9_.])(?:\d{1,3}(?:[,_]\d{3})+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![A-Za-z0-9_.])"
)
_AT_MOST_COUNT_TERMS = re.compile(
    r"(?:\bat\s+most\b|\bup\s+to\b|\bno\s+more\s+than\b|\bmaximum\b|\bmax\b|"
    r"最多|至多|不超過|不超过)",
    re.IGNORECASE,
)
_EXACT_SELECTION_COUNT_TERMS = re.compile(
    r"(?:\bselect\b|\bpick\b|\bchoose\b|\btop\b|選(?:出|擇)?|挑(?:選)?|取(?:出)?)",
    re.IGNORECASE,
)
_MAX_EXACT_FLOAT_INTEGER = 2 ** 53
_NUMERIC_SUBJECT_UNITS = {
    "quote_volume_24h": schema.Unit.USD,
    "price_change_24h": schema.Unit.PCT,
    "funding_rate": schema.Unit.PCT,
    "social_mentions": schema.Unit.COUNT,
    "social_sentiment": schema.Unit.RATIO,
}
_UNIT_TERMS = {
    schema.Unit.USD: ("usd", "usdt", "dollar", "dollars", "美元", "$"),
    schema.Unit.PCT: ("%", "pct", "percent", "percentage", "百分"),
    schema.Unit.RATIO: ("ratio", "比例", "比率"),
}


def _decimal_number(value: Any) -> Optional[Decimal]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _same_number(left: Any, right: Any) -> bool:
    left_decimal = _decimal_number(left)
    right_decimal = _decimal_number(right)
    return left_decimal is not None and left_decimal == right_decimal


def _is_finite_number(value: Any) -> bool:
    if _decimal_number(value) is None:
        return False
    # G1e predicates ultimately use Python floats.  Large JSON integers would
    # otherwise be rounded before the predicate executes, so the public decimal
    # could truthfully match the review yet name a different mathematical gate.
    return not (isinstance(value, int) and abs(value) > _MAX_EXACT_FLOAT_INTEGER)


def _assert_condition_semantics(gate_conditions: Sequence[Mapping[str, Any]],
                                conditions: Sequence[schema.Condition]) -> None:
    """Require a lossless projection from the executed G1e condition to CaseRecord.

    CID equality alone proves nothing: a public case can say ``>1e20`` while G1e
    checked ``>1e8``.  This bridge intentionally supports only gate forms whose
    equivalent schema condition is mechanical; unfamiliar forms are held for a
    reviewer/schema extension rather than guessed into an SFT target.
    """
    by_cid = {condition.cid: condition for condition in conditions}
    for gate_condition in gate_conditions:
        cid = gate_condition.get("id")
        if not isinstance(cid, str) or cid not in by_cid:
            raise ValueError("review condition identities must agree across all three views")
        condition = by_cid[cid]
        subject = gate_condition.get("subject")
        scope = gate_condition.get("scope")
        operator = gate_condition.get("operator")
        if (not isinstance(subject, str) or subject not in gates.PREDICATES
                or subject != condition.subject or scope not in _GATE_GRANULARITY):
            raise ValueError("review condition forms cannot be losslessly projected")
        if condition.granularity is not _GATE_GRANULARITY[scope]:
            raise ValueError("review condition granularity does not match its executed scope")
        if scope == "cross_section" and condition.scope is not schema.Scope.UNIVERSE:
            raise ValueError("cross-section condition must use public universe scope")
        if gate_condition.get("quantified") is True and (
                condition.measurability is not schema.Measurability.QUANTIFIED):
            raise ValueError("quantified gate condition lost its public measurability")

        if operator == "compare":
            expected_unit = _NUMERIC_SUBJECT_UNITS.get(subject)
            if expected_unit is None:
                raise ValueError("review comparison subject has no canonical public unit")
            value = gate_condition.get("value")
            if not isinstance(value, Mapping):
                raise ValueError("unsupported gate comparison form")
            if "op" in value and "threshold" in value:
                expected_operator = _COMPARE_OPERATORS.get(value.get("op"))
                if (expected_operator is None or condition.polarity is not schema.Polarity.INCLUDE
                        or condition.operator is not expected_operator
                        or not _is_finite_number(value.get("threshold"))
                        or not _is_finite_number(condition.value)
                        or not _same_number(condition.value, value.get("threshold"))
                        or condition.unit is not expected_unit):
                    raise ValueError("public condition does not match its executed comparison")
            elif "min" in value or "max" in value:
                expected = [value.get("min"), value.get("max")]
                if (condition.polarity is not schema.Polarity.INCLUDE
                        or condition.operator is not schema.Operator.BETWEEN
                        or not isinstance(condition.value, list)
                        or len(condition.value) != 2
                        or not all(_is_finite_number(item) for item in expected)
                        or not all(_is_finite_number(item) for item in condition.value)
                        or not all(_same_number(left, right)
                                   for left, right in zip(condition.value, expected))
                        or condition.unit is not expected_unit):
                    raise ValueError("public condition does not match its executed range")
            else:
                raise ValueError("unsupported gate comparison form")
        elif operator == "rank":
            direction = gate_condition.get("value")
            if (condition.polarity is not schema.Polarity.INCLUDE
                    or condition.operator is not schema.Operator.EXISTS
                    or condition.value is not None or not condition.is_ranking):
                raise ValueError("public condition does not match its executed ranking")
            if direction not in (None, "asc", "desc"):
                raise ValueError("review ranking direction is invalid")
            if direction in ("asc", "desc") and condition.rank_direction.value != direction:
                raise ValueError("public condition ranking direction does not match G1e")
        elif operator in ("top_k", "exact_top_k"):
            expected_operator = (
                schema.Operator.EXACT_N if operator == "exact_top_k"
                else schema.Operator.TOP_N
            )
            if (condition.polarity is not schema.Polarity.INCLUDE
                    or condition.operator is not expected_operator
                    or not _is_finite_number(gate_condition.get("value"))
                    or not _is_finite_number(condition.value)
                    or not _same_number(condition.value, gate_condition.get("value"))
                    or condition.unit is not schema.Unit.COUNT):
                raise ValueError("public condition does not match its executed basket count")
        elif operator in ("exclude", "require", "equals"):
            # These gate forms can carry arbitrary reviewer strings (sector
            # labels, aliases, or opaque IDs).  Until each has a closed typed
            # vocabulary and a separate public projection, this bridge refuses
            # them rather than making a private scalar a training label.
            raise ValueError("review gate operator has no privacy-safe public projection")
        else:
            raise ValueError("review gate operator has no lossless public projection")


def _quoted_number_spans(quote: str) -> list[tuple[Decimal, int, int]]:
    """Parse whole numeric literals; never accept a prefix of a larger value."""
    values: list[tuple[Decimal, int, int]] = []
    for match in _NUMERIC_LITERAL.finditer(quote):
        try:
            values.append((
                Decimal(match.group(0).replace(",", "").replace("_", "")),
                match.start(), match.end(),
            ))
        except InvalidOperation:
            continue
    return values


def _unit_is_adjacent_to_number(
    unit: Optional[schema.Unit], quote: str, *, start: int, end: int,
) -> bool:
    """Require the unit token beside the same literal, not elsewhere in a quote."""
    terms = _UNIT_TERMS.get(unit)
    if terms is None:  # count/top-k is self-evident from the named basket size.
        return True
    before, after = quote[:start], quote[end:]
    for term in terms:
        escaped = re.escape(term)
        word = term.isascii() and term.isalnum()
        left = r"(?<![A-Za-z0-9_])" if word else ""
        right = r"(?![A-Za-z0-9_])" if word else ""
        if (re.search(left + escaped + r"[ \t]{0,3}$", before, flags=re.IGNORECASE)
                or re.match(r"^[ \t]{0,3}" + escaped + right, after,
                            flags=re.IGNORECASE)):
            return True
    return False


def _assert_explicit_value_provenance(conditions: Sequence[schema.Condition],
                                      quotes: Mapping[str, str]) -> None:
    """Do not turn words such as 'liquid' into a reviewer-invented threshold."""
    for condition in conditions:
        quote = quotes[condition.cid]
        numeric_values = ([condition.value] if isinstance(condition.value, (int, float))
                          and not isinstance(condition.value, bool)
                          else [value for value in condition.value
                                if isinstance(value, (int, float)) and not isinstance(value, bool)]
                          if isinstance(condition.value, list) else [])
        for value in numeric_values:
            number = _decimal_number(value)
            matching_spans = [] if number is None else [
                (start, end) for quoted, start, end in _quoted_number_spans(quote)
                if quoted == number
            ]
            if not matching_spans:
                raise ValueError("quantified condition needs direct source provenance for its value")
            if not any(_unit_is_adjacent_to_number(
                    condition.unit, quote, start=start, end=end)
                   for start, end in matching_spans):
                raise ValueError("quantified condition needs direct source provenance for its unit")
        if condition.is_ranking:
            words = _DESC_WORDS if condition.rank_direction is schema.RankDirection.DESC else _ASC_WORDS
            if not any(word.casefold() in quote.casefold() for word in words):
                raise ValueError("ranking direction needs direct source provenance")
        if condition.operator in (schema.Operator.IN, schema.Operator.NOT_IN):
            values = condition.value if isinstance(condition.value, list) else [condition.value]
            if any(str(value).casefold() not in quote.casefold() for value in values):
                raise ValueError("membership condition needs direct source provenance")
        if condition.operator is schema.Operator.EXACT_N:
            if _AT_MOST_COUNT_TERMS.search(quote):
                raise ValueError("exact basket count conflicts with at-most source wording")
            if not _EXACT_SELECTION_COUNT_TERMS.search(quote):
                raise ValueError("exact basket count needs direct selection source provenance")
        if condition.operator is schema.Operator.TOP_N:
            if not _AT_MOST_COUNT_TERMS.search(quote):
                raise ValueError("at-most basket count needs direct ceiling source provenance")


def _assert_safe_intent_slots(intent_slots: schema.IntentSlots) -> None:
    if intent_slots.primary_timeframe and not _SAFE_TIMEFRAME.fullmatch(intent_slots.primary_timeframe):
        raise schema.PrivacyError("public intent timeframe is not protocol-shaped")
    if any(not _SAFE_SYMBOL.fullmatch(symbol) for symbol in intent_slots.symbols):
        raise schema.PrivacyError("public intent symbol is not protocol-shaped")


def _review_contract(review: Mapping[str, Any], *, row_id: str, request: str,
                     yaml_text: str, bundle: Mapping[str, Any]) -> tuple[Any, ...]:
    keys = set(review)
    if not _REVIEW_REQUIRED <= keys or not keys <= (_REVIEW_REQUIRED | _REVIEW_OPTIONAL):
        raise ValueError("private review has an unknown or missing field")
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("private review has an unsupported schema")
    if review.get("queue_row_id") != row_id:
        raise ValueError("private review is bound to another queue row")
    if _require_sha256(review.get("user_text_sha256"), label="review user_text_sha256") != schema.sha256_hex(request):
        raise ValueError("private review is bound to different request text")
    if _require_sha256(review.get("yaml_sha256"), label="review yaml_sha256") != schema.sha256_hex(yaml_text):
        raise ValueError("private review is bound to different YAML")
    if (_require_sha256(review.get("bundle_sha256"), label="review bundle_sha256")
            != _canonical_json_sha256(bundle)):
        raise ValueError("private review is bound to a different frozen bundle")
    _require_sha256(review.get("prompt_sha256"), label="review prompt_sha256")
    _require_sha256(
        review.get("sol_review_prompt_sha256"), label="review sol_review_prompt_sha256"
    )
    _private_metadata(review.get("source_snapshot_id"), label="source_snapshot_id")
    _private_metadata(review.get("converter_model"), label="converter_model")
    if not isinstance(review.get("temperature"), (int, float)) or isinstance(review.get("temperature"), bool):
        raise ValueError("review temperature must be numeric")
    if not math.isfinite(float(review["temperature"])) or not 0 <= float(review["temperature"]) <= 2:
        raise ValueError("review temperature must be between 0 and 2")
    if review.get("seed") is not None and (not isinstance(review["seed"], int)
                                             or isinstance(review["seed"], bool)
                                             or not 0 <= review["seed"] <= 2_147_483_647):
        raise ValueError("review seed must be a non-negative 32-bit integer")
    if not isinstance(review.get("promote_to_gold"), bool):
        raise ValueError("review promote_to_gold must be boolean")

    try:
        tier = schema.CaseTier(review.get("tier"))
        resolution_path = schema.ResolutionPath(review.get("resolution_path"))
        intent_slots = schema.IntentSlots.from_dict(review.get("intent_slots"))
    except (schema.RecordError, TypeError, ValueError) as exc:
        raise ValueError("private review has invalid structured adjudication") from exc
    if tier is not schema.CaseTier.T1_EXPRESSIBLE:
        raise ValueError("only clear t1_expressible cases enter the conversion bridge")
    _assert_safe_intent_slots(intent_slots)

    raw_gate_conditions = review.get("gate_conditions")
    raw_case_conditions = review.get("case_conditions")
    raw_capabilities = review.get("capability_map")
    if not all(isinstance(value, list) for value in
               (raw_gate_conditions, raw_case_conditions, raw_capabilities)):
        raise ValueError("private review conditions must be JSON lists")
    try:
        gate_conditions = [dict(item) for item in raw_gate_conditions]
        conditions = [schema.Condition.from_dict(item) for item in raw_case_conditions]
        capabilities = [schema.CapabilityEntry.from_dict(item) for item in raw_capabilities]
    except (TypeError, ValueError, schema.RecordError) as exc:
        raise ValueError("private review has invalid condition structure") from exc
    if not conditions:
        raise ValueError("a conversion case requires at least one condition")
    try:
        gate_cids = [str(item["id"]) for item in gate_conditions]
    except (KeyError, TypeError) as exc:
        raise ValueError("gate conditions need safe ids") from exc
    case_cids = [condition.cid for condition in conditions]
    capability_cids = [entry.cid for entry in capabilities]
    if (len(gate_cids) != len(set(gate_cids))
            or len(case_cids) != len(set(case_cids))
            or len(capability_cids) != len(set(capability_cids))
            or set(gate_cids) != set(case_cids)
            or set(case_cids) != set(capability_cids)):
        raise ValueError("review condition identities must agree across all three views")
    if any(entry.verdict is not schema.CapabilityVerdict.SUPPORTED for entry in capabilities):
        raise ValueError("proxy, unsupported, or undecidable conditions cannot enter conversion")
    if any(condition.ambiguity_type is not None for condition in conditions):
        raise ValueError("ambiguous conditions require clarification, not conversion")
    _assert_condition_semantics(gate_conditions, conditions)

    quotes = review.get("conditions_with_quotes")
    if not isinstance(quotes, list):
        raise ValueError("private review quotes must be a JSON list")
    quote_cids: list[str] = []
    validated_quotes: list[Dict[str, Any]] = []
    for item in quotes:
        if not isinstance(item, dict) or set(item) != {"cid", "quote"}:
            raise ValueError("private review quotes need only cid and quote")
        cid, quote = item.get("cid"), item.get("quote")
        if not isinstance(cid, str) or not isinstance(quote, str) or not quote or quote not in request:
            raise ValueError("private review quote does not bind to the queue request")
        quote_cids.append(cid)
        validated_quotes.append({"cid": cid, "quote": quote})
    if len(quote_cids) != len(set(quote_cids)) or set(quote_cids) != set(case_cids):
        raise ValueError("private review needs one source quote for every condition")
    _assert_explicit_value_provenance(
        conditions, {item["cid"]: item["quote"] for item in validated_quotes}
    )

    zh_variant = review.get("zh_variant")
    if zh_variant is not None:
        try:
            zh_variant = schema.ZhVariant(zh_variant)
        except ValueError as exc:
            raise ValueError("review zh_variant is invalid") from exc
    human_by, human_at = _human_reviewer_pair(
        review, required=bool(review["promote_to_gold"])
    )
    return (gate_conditions, conditions, capabilities, intent_slots, zh_variant,
            human_by, human_at, resolution_path,
            validated_quotes)


def _existing_public_records(path: Path, *, kind: str) -> list[Any]:
    if not path.exists():
        return []
    try:
        readers = {"case": schema.read_cases, "attempt": schema.read_attempts,
                   "run": schema.read_runs}
        return list(readers[kind](path))
    except Exception as exc:  # noqa: BLE001 - keep private-like diagnostics out of callers
        raise ValueError("existing public ledger is unreadable") from exc


def _existing_internal_records(path: Path) -> list[schema.InternalCaseRecord]:
    if not path.exists():
        return []
    try:
        return list(schema.read_cases_internal(path))
    except Exception as exc:  # noqa: BLE001 - private data must not reach callers
        raise ValueError("existing private ledger is unreadable") from exc


def _record_payload(record: Any, kind: str) -> Dict[str, Any]:
    try:
        return schema.validate_record(record, kind)
    except Exception as exc:  # noqa: BLE001 - avoid surfacing a record's contents
        raise ValueError("ledger record cannot be validated for recovery") from exc


def _is_exact_existing(
    records: Sequence[Any],
    *,
    predicate: Any,
    expected: Any,
    kind: str,
    label: str,
) -> bool:
    """Return true only for one byte-equivalent prior append.

    A crashed worker can be retried safely, but a new review may never silently
    reuse a case/attempt/run key whose prior payload disagrees.  Re-serialising
    schema-validated records gives an exact, non-verbatim comparison for both
    public and private halves.
    """
    matches = [item for item in records if predicate(item)]
    if len(matches) > 1:
        raise ValueError("existing %s ledger has duplicate recovery records" % label)
    if not matches:
        return False
    if (_canonical_json_sha256(_record_payload(matches[0], kind))
            != _canonical_json_sha256(_record_payload(expected, kind))):
        raise ValueError("existing %s ledger record conflicts with this frozen replay" % label)
    return True


def _write_or_verify_case_pair(
    *,
    case: schema.CaseRecord,
    internal: schema.InternalCaseRecord,
    cases_path: Path,
    internal_path: Path,
    source_text: str,
) -> None:
    public_exists = _is_exact_existing(
        _existing_public_records(cases_path, kind="case"),
        predicate=lambda item: item.case_id == case.case_id,
        expected=case, kind="case", label="public case",
    )
    internal_exists = _is_exact_existing(
        _existing_internal_records(internal_path),
        predicate=lambda item: item.case_id == internal.case_id,
        expected=internal, kind="case_internal", label="private case",
    )
    if not public_exists and not internal_exists:
        schema.write_case_pair(cases_path, internal_path, case, internal)
    elif not internal_exists:
        # Recover a crash after the public append.  It should be impossible with
        # the normal pair order, but recovery must fail closed rather than making
        # a later retry permanently duplicate the public record.
        schema.write_case_internal(internal_path, internal)
    elif not public_exists:
        # Recover the normal partial state: internal append succeeded, public
        # append was interrupted.  Repeat the public writer's source scan.
        schema.write_case(cases_path, case, source_text=source_text)


def _write_or_verify_evaluation(
    *,
    attempt: schema.AttemptRecord,
    run: Optional[schema.RunRecord],
    attempts_path: Path,
    runs_path: Path,
    source_text: str,
) -> None:
    attempt_exists = _is_exact_existing(
        _existing_public_records(attempts_path, kind="attempt"),
        predicate=lambda item: (item.case_id, item.attempt_index)
        == (attempt.case_id, attempt.attempt_index),
        expected=attempt, kind="attempt", label="public attempt",
    )
    if not attempt_exists:
        schema.write_attempt(attempts_path, attempt, source_text=source_text)

    existing_runs = _existing_public_records(runs_path, kind="run")
    linked_runs = [item for item in existing_runs
                   if (item.case_id, item.attempt_index)
                   == (attempt.case_id, attempt.attempt_index)]
    if run is None:
        if linked_runs:
            raise ValueError("existing public run conflicts with an absent frozen run")
        return
    run_exists = _is_exact_existing(
        existing_runs,
        predicate=lambda item: item.run_id == run.run_id,
        expected=run, kind="run", label="public run",
    )
    if any(item.run_id != run.run_id for item in linked_runs):
        raise ValueError("existing public run conflicts with this frozen replay")
    if not run_exists:
        schema.write_run(runs_path, run)


def _assert_promotable(evaluation: evaluate.LedgerEvaluation, *, cids: set[str],
                       human_reviewer: Optional[str]) -> None:
    if not human_reviewer:
        raise ValueError("gold promotion requires an actual human reviewer")
    if (not evaluation.receipt.gold_clean or evaluation.run is None
            or evaluation.attempt.stopped_at_gate is not schema.Gate.PASSED):
        raise ValueError("gold promotion requires a clean frozen replay")
    verdicts = evaluation.attempt.condition_verdicts
    if (len(verdicts) != len(cids) or {item.cid for item in verdicts} != cids
            or any(item.status is not schema.ConditionCheckStatus.SATISFIED
                   for item in verdicts)):
        raise ValueError("gold promotion requires every reviewed condition to be satisfied")


def _sol_artifact_hashes(
    *,
    review: Mapping[str, Any],
    evaluation: evaluate.LedgerEvaluation,
    verified_repo_git_sha: str,
) -> sol_review.ArtifactHashes:
    """Return the exact, non-verbatim replay manifest a Sol receipt must bind.

    This is intentionally built only after ``materialize_ledger_evaluation``:
    a review of an equivalent-looking YAML or a pre-materialization draft is not
    a review of the record that promotion would append.  The capability map and
    repository provenance are canonical manifests, not mutable paths or branch
    names.
    """
    if (evaluation.run is None or evaluation.run_record_sha256 is None
            or evaluation.receipt.signal_batch_sha256 is None):
        raise ValueError("gold promotion requires replay artifacts for Sol review")
    return sol_review.ArtifactHashes(
        review_prompt_sha256=_require_sha256(
            review.get("sol_review_prompt_sha256"), label="Sol review prompt_sha256"
        ),
        request_sha256=evaluation.receipt.nl_sha256,
        yaml_sha256=evaluation.receipt.yaml_sha256,
        conditions_sha256=evaluation.receipt.conditions_sha256,
        bundle_sha256=evaluation.receipt.bundle_sha256,
        attempt_sha256=evaluation.attempt_record_sha256,
        run_sha256=evaluation.run_record_sha256,
        signal_sha256=evaluation.receipt.signal_batch_sha256,
        capability_sha256=_canonical_json_sha256({
            "capability_map": review["capability_map"],
        }),
        repo_sha256=_canonical_json_sha256({"repo_git_sha": verified_repo_git_sha}),
    )


def _require_content_bound_sol_review(
    *,
    sol_review_path: Path | str | None,
    expected_artifacts: sol_review.ArtifactHashes,
    cids: set[str],
) -> sol_review.SolReviewReceipt:
    """Verify local audit evidence, then keep strict promotion fail-closed.

    ``sol_review`` HMAC records still protect the private audit trail and bind
    one claimed review to the frozen artifacts.  They cannot establish external
    provider provenance because this worker reads the same local key that signs
    them.  Do the structural verification first so malformed/tampered evidence
    is still detected, then refuse promotion until an independent adapter is
    available.
    """
    if sol_review_path is None:
        raise ValueError(
            "gold promotion requires a local content-bound audit receipt and "
            "independent external provider attestation"
        )
    receipt = sol_review.read_receipt(
        sol_review_path,
        expected_artifact_hashes=expected_artifacts,
        require_approved=True,
    )
    if {item.cid for item in receipt.cid_verdicts} != cids:
        raise ValueError("Sol review receipt must cover exactly every reviewed condition")
    raise ValueError(_LOCAL_HMAC_GOLD_BLOCK)


def _assert_no_verbatim_public_strings(value: Any, *, request: str, user_id: str) -> None:
    """Preflight every string that is about to cross into a public record.

    ``write_case_pair`` and ``write_attempt`` repeat the YAML-specific scan at
    their own boundary.  This broader preflight closes the non-gold path too:
    a reviewer must not be able to put a copied user phrase into a structured
    condition subject, metadata field, or a YAML comment and leave an otherwise
    valid public CaseRecord behind when the later attempt write refuses it.
    """
    if isinstance(value, Mapping):
        for nested in value.values():
            _assert_no_verbatim_public_strings(nested, request=request, user_id=user_id)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_verbatim_public_strings(nested, request=request, user_id=user_id)
    elif isinstance(value, str):
        # Eight-gram overlap intentionally ignores short ordinary vocabulary,
        # but a short *entire* request is still private verbatim text.  Scan a
        # canonical whole-source containment first so a copied ``buy BTC now``
        # cannot slip into a public YAML description, condition, or metadata
        # field just because it contains fewer than eight English tokens/CJK
        # characters.  The exception contains no source/value text.
        canonical_request = schema.canonicalize_text(request)
        if canonical_request and canonical_request in schema.canonicalize_text(value):
            raise schema.PrivacyError("public ledger value reproduces entire user wording")
        if schema.verbatim_overlap(value, request):
            raise schema.PrivacyError("public ledger value reproduces user wording")
        # A queue user id is private even when the user did not type it in the
        # request.  Only compare meaningful IDs: a one-character account id
        # would spuriously appear in ordinary YAML numbers and identifiers.
        user_id_pattern = (r"(?<![A-Za-z0-9])" + re.escape(user_id)
                           + r"(?![A-Za-z0-9])")
        if len(user_id) >= 8 and re.search(user_id_pattern, value):
            raise schema.PrivacyError("public ledger value reproduces a private user identifier")
        if _SENSITIVE_TEXT.search(value):
            raise schema.PrivacyError("public ledger value resembles a credential or contact identifier")


def _append_private_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one private audit record with mode 600 from its first byte."""
    encoded = (json.dumps(dict(payload), ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        os.chmod(path, 0o600)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)


def _read_private_jsonl(path: Path, *, label: str) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        text = _read_limited(path, label=label).decode("utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("existing private %s is unreadable" % label) from exc
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("existing private %s is malformed" % label)
    return [dict(record) for record in records]


def _write_or_verify_private_jsonl(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
    identity: Mapping[str, str],
) -> None:
    """Append once, or accept only an exact private record on retry."""
    expected = dict(payload)
    matches = [record for record in _read_private_jsonl(path, label=label)
               if all(record.get(key) == value for key, value in identity.items())]
    if len(matches) > 1:
        raise ValueError("existing private %s has duplicate recovery records" % label)
    if matches:
        if _canonical_json_sha256(matches[0]) != _canonical_json_sha256(expected):
            raise ValueError("existing private %s conflicts with this frozen replay" % label)
        return
    _append_private_receipt(path, expected)


_OUTCOME_CID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _outcome_code(exc: Exception) -> str:
    """Classify a refusal without retaining the exception text.

    Exceptions below may include an untrusted YAML key, source value, or a
    provider error.  This private audit stream must preserve the fact and broad
    class of a failure without becoming a second copy of those inputs.
    """
    message = str(exc).casefold()
    if "clarification" in message or "ambiguous" in message:
        return "needs_clarification"
    if any(token in message for token in (
            "unsupported", "proxy", "capability", "not expressible", "gap")):
        return "unsupported"
    if any(token in message for token in ("yaml", "parse", "schema", "spec")):
        return "yaml_rejected"
    if any(token in message for token in (
            "bundle", "runtime", "frozen replay", "condition", "gate")):
        return "replay_rejected"
    if "review" in message or "promotion" in message:
        return "review_rejected"
    return "curation_refused"


def _review_outcome_fields(review_path: Path | str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract only closed/safe IDs from an optional private review artifact."""
    try:
        _, review = _read_private_json(review_path, label="review")
    except Exception:  # noqa: BLE001 - audit must never mask the original refusal
        return (), ()
    cids: set[str] = set()
    gaps: set[str] = set()
    for item in review.get("case_conditions") or []:
        if isinstance(item, Mapping) and _OUTCOME_CID.fullmatch(str(item.get("cid") or "")):
            cids.add(str(item["cid"]))
    for item in review.get("capability_map") or []:
        if not isinstance(item, Mapping):
            continue
        try:
            gap = schema.GapId(item.get("gap_id")) if item.get("gap_id") else None
        except ValueError:
            gap = None
        if gap is not None:
            gaps.add(gap.value)
    return tuple(sorted(cids)), tuple(sorted(gaps))


def _outcome_code_from_review(review_path: Path | str, *, fallback: str) -> str:
    """Recover a closed outcome type when the raised error is deliberately terse."""
    try:
        _, review = _read_private_json(review_path, label="review")
    except Exception:  # noqa: BLE001 - no review means no safe refinement
        return fallback
    if review.get("tier") != schema.CaseTier.T1_EXPRESSIBLE.value:
        return "unsupported"
    for item in review.get("capability_map") or []:
        if isinstance(item, Mapping) and item.get("verdict") != schema.CapabilityVerdict.SUPPORTED.value:
            return "unsupported"
    for item in review.get("case_conditions") or []:
        if isinstance(item, Mapping) and item.get("ambiguity_type"):
            return "needs_clarification"
    return fallback


def _record_private_refusal(
    *,
    queue_path: Path | str,
    row_id: str,
    review_path: Path | str,
    exc: Exception,
    private_outcome_path: Path | str | None,
) -> None:
    """Best-effort, non-verbatim failure history for a queue-row curation.

    This runs only after a refusal and never changes the error visible to the
    caller.  If the queue itself cannot be safely opened there is no reliable
    private case identity to record, so it intentionally does nothing.
    """
    try:
        queue = _private_path(queue_path, label="queue", must_exist=True)
        row = _queue_row(queue, row_id)
        request = _request_from_row(row)
        request_sha256 = schema.sha256_hex(request)
        case_id = schema.case_id_for(request_sha256)
        cids, gaps = _review_outcome_fields(review_path)
        code = _outcome_code_from_review(
            review_path, fallback=_outcome_code(exc),
        )
        target = _private_path(
            private_outcome_path or (_private_root() / DEFAULT_PRIVATE_OUTCOMES),
            label="curation outcome", must_exist=False,
        )
        payload = {
            "schema": "cyqnt.nl2yaml.queue-curation-outcome/v1",
            "case_id": case_id,
            "queue_row_id": row_id,
            "request_sha256": request_sha256,
            "outcome": code,
            "condition_ids": list(cids),
            "gap_ids": list(gaps),
        }
        _write_or_verify_private_jsonl(
            target, payload, label="curation outcome",
            identity={"case_id": case_id, "outcome": code},
        )
    except Exception:
        # Failure evidence is valuable, but failure to write it must not turn a
        # correct fail-closed refusal into a different, potentially noisy error.
        return


class _CaseClaim:
    """One non-blocking private file lock and tiny durable recovery journal."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.path = _private_path(
            _private_root() / _CLAIM_DIRECTORY / (case_id + ".lock"),
            label="case claim", must_exist=False,
        )
        self._descriptor: Optional[int] = None

    def _state(self, state: str) -> None:
        if self._descriptor is None:
            return
        encoded = (json.dumps({
            "schema": "cyqnt.nl2yaml.queue-claim/v1",
            "case_id": self.case_id,
            "state": state,
        }, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        os.ftruncate(self._descriptor, 0)
        written = 0
        while written < len(encoded):
            written += os.write(self._descriptor, encoded[written:])
        os.fsync(self._descriptor)

    def __enter__(self) -> "_CaseClaim":
        descriptor = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        os.chmod(self.path, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise ValueError("queue case is already being curated") from exc
        self._descriptor = descriptor
        self._state("claimed")
        return self

    def complete(self) -> None:
        self._state("complete")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is not None:
                self._state("incomplete")
        finally:
            if self._descriptor is not None:
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
                os.close(self._descriptor)
                self._descriptor = None


def _process_queue_case(
    *,
    queue_path: Path | str,
    row_id: str,
    review_path: Path | str,
    yaml_path: Path | str,
    bundle_path: Path | str,
    repo_git_sha: str,
    public_root: Path | str = PUBLIC_DATASET_ROOT,
    internal_cases_path: Path | str | None = None,
    private_receipt_path: Path | str | None = None,
    sol_review_path: Path | str | None = None,
) -> QueueCurationResult:
    """Process exactly one manually reviewed private queue row.

    Public and paired-case mutations are intentionally last.  A completed
    frozen replay first gets one mode-600 private diagnostic record, so a G1e
    failure remains inspectable; source mismatches and unsupported/ambiguous
    conditions still leave every ledger untouched.  Promotion failures never
    append a public case, attempt, or run.
    """
    queue = _private_path(queue_path, label="queue", must_exist=True)
    review_file, review = _read_private_json(review_path, label="review")
    _, yaml_text = _read_private_text(yaml_path, label="YAML")
    bundle = _read_bundle(bundle_path)
    internal_cases = _private_path(
        internal_cases_path or (_private_root() / DEFAULT_INTERNAL_CASES),
        label="case ledger", must_exist=False,
    )
    private_receipt = _private_path(
        private_receipt_path or (_private_root() / DEFAULT_PRIVATE_RECEIPT),
        label="curation receipt", must_exist=False,
    )
    private_diagnostics = _private_path(
        _private_root() / DEFAULT_PRIVATE_DIAGNOSTICS,
        label="evaluation diagnostics", must_exist=False,
    )
    cases_path = _public_path(public_root, "cases.jsonl")
    attempts_path = _public_path(public_root, "attempts.jsonl")
    runs_path = _public_path(public_root, "runs.jsonl")

    row = _queue_row(queue, row_id)
    request = _request_from_row(row)
    (gate_conditions, case_conditions, capabilities, intent_slots, zh_variant,
     human_by, human_at, resolution_path,
     validated_quotes) = _review_contract(
         review, row_id=row_id, request=request, yaml_text=yaml_text, bundle=bundle)

    try:
        lang = schema.Lang(str(row.get("lang") or ""))
    except ValueError as exc:
        raise ValueError("queue language is outside the dataset scope") from exc
    if (lang is schema.Lang.ZH) != (zh_variant is not None):
        raise ValueError("review zh_variant must agree with the queue language")
    try:
        dup_count = int(str(row.get("dup_count") or ""))
        queue_rank = int(str(row.get("queue_rank") or ""))
    except ValueError as exc:
        raise ValueError("queue duplicate count or rank is invalid") from exc
    if dup_count < 1 or queue_rank < 1:
        raise ValueError("queue duplicate count and rank must be positive")
    cluster_id = str(row.get("cluster_id") or "")
    if not re.fullmatch(r"dup_[0-9a-f]{16}", cluster_id):
        raise ValueError("queue cluster id is invalid")
    user_id = str(row.get("user_id") or "")
    if not user_id:
        raise ValueError("queue row has no user identity")
    if len(user_id) < 8:
        # A short opaque account label is indistinguishable from harmless YAML
        # vocabulary, so a generic post-hoc scanner cannot prove it was not
        # copied into a public artifact.  Hold such rows for a stronger private
        # identity contract instead of pretending the pseudonym alone protects
        # their source identifier.
        raise ValueError("queue user identity is too short for this privacy-preserving bridge")

    # Refuse a copied request or private identity before replay or any external
    # review receipt is considered.  The later structured-record scan remains
    # necessary for review metadata, but it must not be the first time a raw
    # YAML artifact is checked.
    _assert_no_verbatim_public_strings(yaml_text, request=request, user_id=user_id)

    request_sha256 = schema.sha256_hex(request)
    case_id = schema.case_id_for(request_sha256)
    verified_repo_git_sha = _verified_repo_git_sha(repo_git_sha)
    prepared = evaluate.prepare_ledger_evaluation(
        case_id=case_id,
        attempt_index=1,
        prompt_sha256=review["prompt_sha256"],
        source_text=request,
        nl=request,
        yaml_answer=yaml_text,
        conditions=gate_conditions,
        bundle=bundle,
        repo_git_sha=verified_repo_git_sha,
        sampling_purpose=schema.SamplingPurpose.CONVERT,
    )

    # This checks the immutable fingerprints before any recovery decision.  A
    # prepared replay is not safe to append merely because its outer dataclass
    # is frozen: nested AttemptRecord/RunRecord objects can otherwise be edited.
    attempt, run = evaluate.materialize_ledger_evaluation(prepared, source_text=request)
    review_sha256 = _canonical_json_sha256(review)
    promoted = bool(review["promote_to_gold"])
    diagnostics_payload = {
        "schema": "cyqnt.nl2yaml.queue-evaluation-diagnostics/v1",
        "case_id": case_id,
        "queue_row_id": row_id,
        "review_sha256": review_sha256,
        "yaml_sha256": prepared.receipt.yaml_sha256,
        "bundle_sha256": prepared.receipt.bundle_sha256,
        "gate_status": prepared.receipt.gate_status,
        "diagnostics": prepared.diagnostics.to_dict(),
    }

    # The claim is deliberately acquired only after all private inputs have
    # bound to one frozen replay.  It serialises the check/write sequence, and
    # exact-record recovery makes a retry safe if any append was interrupted.
    with _CaseClaim(case_id) as claim:
        _write_or_verify_private_jsonl(
            private_diagnostics, diagnostics_payload, label="evaluation diagnostics",
            identity={"case_id": case_id, "review_sha256": review_sha256},
        )

        cids = {condition.cid for condition in case_conditions}
        if promoted:
            # A failed promotion still leaves the private diagnostic record
            # above, but never a public case/attempt/run half.
            _assert_promotable(prepared, cids=cids, human_reviewer=human_by)
            expected_sol_artifacts = _sol_artifact_hashes(
                review=review,
                evaluation=prepared,
                verified_repo_git_sha=verified_repo_git_sha,
            )
            approved_sol_review = _require_content_bound_sol_review(
                sol_review_path=sol_review_path,
                expected_artifacts=expected_sol_artifacts,
                cids=cids,
            )
            gold_specs = [schema.GoldSpec(role=schema.GoldRole.PRIMARY, yaml_text=yaml_text)]
            gold_source = schema.GoldSource.ATTEMPT_K
            gold_level = schema.GoldVerificationLevel.ALL_CHECKABLE_CONDITIONS_SATISFIED
            gold_verdicts = attempt.condition_verdicts
        else:
            approved_sol_review = None
            gold_specs = []
            gold_source = schema.GoldSource.NONE
            gold_level = schema.GoldVerificationLevel.UNPARSED
            gold_verdicts = []

        case = schema.CaseRecord(
            case_id=case_id,
            text_sha256=request_sha256,
            canon_sha256=schema.canon_sha256_of(request),
            dup_cluster_id=cluster_id,
            dup_count=dup_count,
            preset_case=False,
            pseudonym_id=schema.hmac_pseudonym(user_id),
            lang=lang,
            zh_variant=zh_variant,
            month=str(row.get("month") or ""),
            mining_source=schema.MiningSource.HUMAN_CURATED,
            source_snapshot_id=_public_metadata_label("snapshot", review["source_snapshot_id"]),
            text_provenance=schema.TextProvenance.VERBATIM_INTERNAL,
            conditions=case_conditions,
            intent_slots=intent_slots,
            capability_map=capabilities,
            tier=schema.CaseTier.T1_EXPRESSIBLE,
            text_len_chars=len(request),
            converter_model=_public_metadata_label("model", review["converter_model"]),
            temperature=float(review["temperature"]),
            seed=review.get("seed"),
            repo_git_sha=verified_repo_git_sha,
            gold_specs=gold_specs,
            gold_source=gold_source,
            gold_verification_level=gold_level,
            gold_condition_verdicts=gold_verdicts,
            human_reviewed_by=(_public_metadata_label("reviewer", human_by)
                               if human_by is not None else None),
            human_reviewed_at=human_at,
            resolution_path=resolution_path,
        )
        if promoted and not case.gold_eligible_for_sft:
            raise ValueError("promoted case did not satisfy strict SFT eligibility")
        internal = schema.InternalCaseRecord(
            case_id=case_id,
            user_text_raw=request,
            user_text_redacted=request,
            pseudonym_id=case.pseudonym_id or "",
            user_id=user_id,
            conditions_with_quotes=validated_quotes,
        )
        # Do every validation/privacy check before either side of the pair can
        # append.  The wider scan prevents review metadata or a copied YAML from
        # leaking through the non-gold path as well.
        public_payloads = [
            _record_payload(case, "case"),
            _record_payload(attempt, "attempt"),
        ]
        if run is not None:
            public_payloads.append(_record_payload(run, "run"))
        _record_payload(internal, "case_internal")
        for payload in public_payloads:
            _assert_no_verbatim_public_strings(payload, request=request, user_id=user_id)

        _write_or_verify_case_pair(
            case=case, internal=internal, cases_path=cases_path,
            internal_path=internal_cases, source_text=request,
        )
        _write_or_verify_evaluation(
            attempt=attempt, run=run, attempts_path=attempts_path,
            runs_path=runs_path, source_text=request,
        )
        curation_payload = {
            "schema": CURATION_RECEIPT_SCHEMA,
            "case_id": case_id,
            "queue_row_id": row_id,
            "queue_rank": queue_rank,
            "review_sha256": review_sha256,
            "yaml_sha256": prepared.receipt.yaml_sha256,
            "bundle_sha256": prepared.receipt.bundle_sha256,
            "gate_status": prepared.receipt.gate_status,
            "gold_promoted": promoted,
            "human_reviewed_by": human_by,
            "human_reviewed_at": human_at,
            "sol_review_receipt_sha256": (
                _canonical_json_sha256(approved_sol_review.to_dict())
                if approved_sol_review is not None else None
            ),
            "sol_reviewer_model": (
                approved_sol_review.reviewer_model if approved_sol_review is not None else None
            ),
            "sol_reasoning_effort": (
                approved_sol_review.reasoning_effort if approved_sol_review is not None else None
            ),
            "sol_reviewed_at": (
                approved_sol_review.issued_at if approved_sol_review is not None else None
            ),
            "review_path_sha256": _sha256_bytes(str(review_file).encode("utf-8")),
            "diagnostics": prepared.diagnostics.to_dict(),
        }
        _write_or_verify_private_jsonl(
            private_receipt, curation_payload, label="curation receipt",
            identity={"case_id": case_id, "review_sha256": review_sha256},
        )
        claim.complete()

    return QueueCurationResult(
        case_id=case_id,
        queue_row_id=row_id,
        gate_status=prepared.receipt.gate_status,
        gold_promoted=promoted,
        run_written=run is not None,
    )


def process_queue_case(
    *,
    queue_path: Path | str,
    row_id: str,
    review_path: Path | str,
    yaml_path: Path | str,
    bundle_path: Path | str,
    repo_git_sha: str,
    public_root: Path | str = PUBLIC_DATASET_ROOT,
    internal_cases_path: Path | str | None = None,
    private_receipt_path: Path | str | None = None,
    sol_review_path: Path | str | None = None,
    private_outcome_path: Path | str | None = None,
) -> QueueCurationResult:
    """Curate one row and retain a private, non-verbatim refusal outcome.

    The inner worker still raises every refusal and still writes no public
    record unless all gates pass.  The wrapper merely makes ambiguity,
    capability gaps and bad replays visible to the private audit trail, rather
    than losing their history behind the CLI's generic ``refused`` response.
    """
    try:
        return _process_queue_case(
            queue_path=queue_path,
            row_id=row_id,
            review_path=review_path,
            yaml_path=yaml_path,
            bundle_path=bundle_path,
            repo_git_sha=repo_git_sha,
            public_root=public_root,
            internal_cases_path=internal_cases_path,
            private_receipt_path=private_receipt_path,
            sol_review_path=sol_review_path,
        )
    except Exception as exc:
        _record_private_refusal(
            queue_path=queue_path,
            row_id=row_id,
            review_path=review_path,
            exc=exc,
            private_outcome_path=private_outcome_path,
        )
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the bounded local queue bridge and print only a safe summary."""
    parser = argparse.ArgumentParser(
        description="Curate one private queue row with reviewed YAML and a frozen bundle."
    )
    parser.add_argument("--queue", required=True,
                        help="mode-600 queue CSV below $NL2YAML_INTERNAL_ROOT")
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--review", required=True,
                        help="mode-600 reviewed JSON below $NL2YAML_INTERNAL_ROOT")
    parser.add_argument("--yaml", required=True,
                        help="mode-600 reviewed YAML below $NL2YAML_INTERNAL_ROOT")
    parser.add_argument("--bundle", required=True, help="frozen cyqnt.input/v1 JSON")
    parser.add_argument("--repo-git-sha", required=True)
    parser.add_argument("--public-dir", default=str(PUBLIC_DATASET_ROOT))
    parser.add_argument("--internal-cases")
    parser.add_argument("--private-receipt")
    parser.add_argument(
        "--sol-review",
        help=(
            "mode-600 local HMAC audit receipt beneath $NL2YAML_INTERNAL_ROOT; "
            "it verifies private artifact binding only and cannot unlock strict "
            "SFT/gold promotion"
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = process_queue_case(
            queue_path=args.queue,
            row_id=args.row_id,
            review_path=args.review,
            yaml_path=args.yaml,
            bundle_path=args.bundle,
            repo_git_sha=args.repo_git_sha,
            public_root=args.public_dir,
            internal_cases_path=args.internal_cases,
            private_receipt_path=args.private_receipt,
            sol_review_path=args.sol_review,
        )
    except Exception:  # noqa: BLE001 - CLI must never echo private input diagnostics
        print(json.dumps({"schema": CURATION_RECEIPT_SCHEMA, "status": "refused"},
                         sort_keys=True))
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI
    sys.exit(main())
