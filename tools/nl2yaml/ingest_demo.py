#!/usr/bin/env python3
"""Turn the 2026-08-02 five-case end-to-end demo into dataset records.

Why this exists: the demo preserved five user conversations, their extracted
conditions, converted YAML, and independent per-condition review.  A legacy
review is enough to retain an intent-reconciled training example, but it is not
enough to assert an execution fact: only an adjacent safe frozen receipt can
make a locally checkable claim tying the exact YAML to a pinned bundle and
emitted signal batch.  That receipt is local evidence, not a signature, remote
attestation, or non-forgeable proof.  This migration does not re-run the
evaluator, so even a linked receipt must never be described as cryptographic
proof that the recorded execution happened.  The source was written as loose
JSON in a session scratchpad, which is wiped. Nothing downstream can train on
or re-check a directory that no longer exists.

So: read the archived demo artifacts, map them onto :mod:`tools.nl2yaml.schema`
records, and split them by privacy — user text and per-condition quotes to the
internal store (outside the repo; both git remotes are PUBLIC), everything else
to the committable dataset.

The verdicts are the reason these records are worth keeping rather than
regenerating: each condition carries a machine-checked status *with cited
evidence from the run output*, and three of the five cases are honest negatives
(one basket violated a condition, two were correctly refused). A dataset of only
successes teaches a model that success is the only shape.

Usage:
    python -m tools.nl2yaml.ingest_demo --src ~/nl2yaml_internal/raw_demo_2026-08-02
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import evaluate as frozen_evaluate
from . import schema as S
from .capability import normalize_condition

# The demo ranked a frozen cross-section; a gold verified against it is only
# meaningful while that bundle is pinned, so the id travels with the record.
BUNDLE_REL = "tests/standard_bot/fixtures/universe_cross_section.json"
DEMO_RUN = "demo_2026-08-02_five_case"

# A legacy source directory may optionally contain one safe frozen-evaluation
# receipt per case.  Keep the convention next to the YAML it attests, rather
# than in a shared log whose entries have to be joined by a hand-maintained id:
#
#     <case>.yaml
#     <case>_frozen_conditions.json
#     <case>_receipt.json
#
# ``<case>_frozen_conditions.json`` is the exact privacy-safe JSON list passed
# to ``evaluate_frozen``; it is deliberately distinct from the legacy
# ``<case>_conditions.json``, which may contain internal reviewer quotes.  The
# receipt is a local *evidence/claim*, not a replacement for either input and
# not a signed or otherwise non-forgeable attestation.  It contains only
# hashes/aggregate statuses in ``cyqnt.nl2yaml.frozen-evaluation/v1`` and is
# never copied into the public dataset.  This importer checks its linkage but
# does not replay it, so it cannot claim cryptographic authenticity.  A missing
# (or malformed) pair is intentionally non-fatal for migration: it preserves
# the reviewed YAML at level 3, but can never promote it to level 4/5.
RECEIPT_SUFFIX = "_receipt.json"
FROZEN_CONDITIONS_SUFFIX = "_frozen_conditions.json"
_REQUIRED_RUNTIME_GATES = ("G1a", "G1b", "G1c", "G1d")

#: Verdict vocabulary used by the demo's reviewer -> schema vocabulary.
#: Legacy imports intentionally omit a condition that was never expressed;
#: explicit ``not_expressed`` verdicts are mapped to the fail-closed schema
#: status so "we did not try" can never be read as "we tried and it held".
_STATUS = {
    "satisfied": S.ConditionCheckStatus.SATISFIED,
    "violated": S.ConditionCheckStatus.VIOLATED,
    "unverifiable": S.ConditionCheckStatus.UNVERIFIABLE,
    "not_expressed": S.ConditionCheckStatus.NOT_EXPRESSED,
}

# The demo's A1 agents were not given the closed vocabulary (my omission), so
# they invented one. Mapping it here rather than loosening the enums keeps the
# schema strict — and the mapping is where the ambiguity surfaces: every unit the
# extractor could not name is a condition whose threshold has two readings.
_POLARITY = {"require": S.Polarity.INCLUDE, "exclude": S.Polarity.EXCLUDE,
             "prefer": S.Polarity.PREFER}

_OPERATOR = {
    ">": S.Operator.GT, ">=": S.Operator.GTE, "<": S.Operator.LT, "<=": S.Operator.LTE,
    "equals": S.Operator.EQ, "eq": S.Operator.EQ, "neq": S.Operator.NEQ,
    "in": S.Operator.IN, "not_in": S.Operator.NOT_IN, "between": S.Operator.BETWEEN,
    "top_n": S.Operator.TOP_N, "bottom_n": S.Operator.BOTTOM_N,
    # "rank" alone never said which end; TOP_N would invent a direction.
    "rank": S.Operator.UNSPECIFIED,
    "exists": S.Operator.EXISTS, "absent": S.Operator.ABSENT,
    None: S.Operator.UNSPECIFIED, "": S.Operator.UNSPECIFIED,
}

_GRANULARITY = {"universe": S.EvaluationGranularity.UNIVERSE,
                "per_symbol": S.EvaluationGranularity.PER_SYMBOL,
                "portfolio": S.EvaluationGranularity.PORTFOLIO}

#: demo granularity -> which part of the strategy the condition governs
_SCOPE = {"universe": S.Scope.UNIVERSE, "per_symbol": S.Scope.UNIVERSE,
          "portfolio": S.Scope.RISK}

#: Units the extractor wrote in free text. ``None`` means "no unit survived the
#: mapping" and the caller must treat the condition as UNIT-ambiguous rather than
#: guessing — the demo's own reviewer showed one candidate passing under
#: "1e7 USD turnover" and failing under "1e7 coins".
_UNIT = {
    "percent": S.Unit.PCT, "%": S.Unit.PCT, "pct": S.Unit.PCT,
    "percent_of_capital": S.Unit.PCT,
    "usd": S.Unit.USD, "USD": S.Unit.USD, "usdt": S.Unit.USD,
    "quote_currency": S.Unit.USD, "price": S.Unit.USD,
    "symbols": S.Unit.SYMBOL, "symbol": S.Unit.SYMBOL, "label": S.Unit.LABEL,
    "count": S.Unit.COUNT, "hours": S.Unit.HOURS, "days": S.Unit.DAYS,
    "bars": S.Unit.BARS, "ratio": S.Unit.RATIO, "bps": S.Unit.BPS,
    "x": S.Unit.RATIO, "x_baseline_volume": S.Unit.RATIO,
    "leverage_x": S.Unit.LEVERAGE_X, "coin_qty": S.Unit.COIN_QTY,
    "indicator_value": S.Unit.INDICATOR_VALUE,
}

_MEASURABILITY = {"quantified": S.Measurability.QUANTIFIED,
                  "vague": S.Measurability.VAGUE,
                  "unmeasurable": S.Measurability.UNMEASURABLE}

#: Subject substring -> the capability that is missing, checked in order.
#: ``UNNAMED_CAPABILITY`` is the deliberate fallback: forcing a plausible-looking
#: gap onto an unrecognised subject corrupts the gap ranking that decides what
#: gets built next, which is worse than admitting the subject was not classified.
_GAP_PATTERNS: List[Tuple[Tuple[str, ...], "S.GapId"]] = [
    (("onchain", "holder_balance", "holder_concentration", "chip"), S.GapId.ONCHAIN_DATA),
    (("market_cap", "circulating"), S.GapId.UNIVERSE_MARKET_CAP),
    (("sector_label", "concept_label", "symbol_category", "asset_class",
      "contract_type", "alpha", "listing_venue", "tokenized"), S.GapId.UNIVERSE_CONTRACT_META),
    (("open_interest",), S.GapId.UNIVERSE_OPEN_INTEREST),
    (("long_short", "ls_ratio"), S.GapId.UNIVERSE_LONG_SHORT_RATIO),
    (("multi_timeframe", "timeframe_confluence"), S.GapId.MULTI_TIMEFRAME_CONFLUENCE),
    # Anything asking about a window longer than the 24h cross-section needs bars.
    (("90d", "since_listing", "low_to_high", "ath", "volume_spike", "kline_volume",
      "stats_window"), S.GapId.BACKTEST_HISTORY_MISSING),
    (("technical_pattern", "accumulation", "bottoming", "momentum", "volatility",
      "market_structure", "indicator"), S.GapId.UNIVERSE_PER_CANDIDATE_INDICATOR),
    (("entry_exit", "trailing_stop", "order_execution", "position_sizing",
      "leverage", "automated_position_close"), S.GapId.ENTRY_EXIT_PER_CANDIDATE),
    (("notification",), S.GapId.NOTIFICATION_CHANNEL),
    (("scheduled_jobs",), S.GapId.EXECUTION_VENUE_FEATURE),
    (("risk_reward", "position_details"), S.GapId.PORTFOLIO_LEVEL_RISK),
    (("social", "sentiment"), S.GapId.SOCIAL_SENTIMENT),
    # "satisfy as many as possible", "setup_quality", "signal_precision",
    # "confidence_score_rank", "expected_payoff" are all soft multi-factor asks.
    (("condition_satisfaction_policy", "setup_quality", "setup_probability",
      "signal_precision", "confidence_score", "expected_payoff",
      "tradable_actionable", "allowed_signal_basis"), S.GapId.WEIGHTED_MULTI_FACTOR_SCORE),
    (("per_symbol_deep_analysis", "market_sentiment_overall", "sentiment_logic",
      "report_structure", "plan_option_confirmation"), S.GapId.MANUAL_DISCRETION),
]


def _gap_for(subject: str) -> "S.GapId":
    low = (subject or "").lower()
    for keys, gap in _GAP_PATTERNS:
        if any(k in low for k in keys):
            return gap
    return S.GapId.UNNAMED_CAPABILITY



_HANT = re.compile(r"[個們這麼說時間為對開實現關點學問題國東證後產經體發覺變適當價業車輪]")


def _zh_variant(text: str) -> str:
    """Traditional vs simplified, decided by characters that differ between them."""
    return "zh-Hant" if _HANT.search(text or "") else "zh-Hans"


def _read(path: Path) -> Optional[Any]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _receipt_path(src: Path, cid: str) -> Path:
    """Return the one deterministic safe-receipt location for ``cid``."""
    return src / f"{cid}{RECEIPT_SUFFIX}"


def _frozen_conditions_path(src: Path, cid: str) -> Path:
    """Return the exact reviewed-condition input paired with a receipt."""
    return src / f"{cid}{FROZEN_CONDITIONS_SUFFIX}"


def _canonical_json_sha256(value: Any) -> Optional[str]:
    """Match the frozen evaluator's canonical JSON hash without emitting input."""
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _pinned_bundle_sha256(repo: Path) -> Optional[str]:
    """Hash the frozen legacy fixture using the evaluator's canonical encoding.

    The receipt must attest to the same bundle the migration names.  A hash
    which merely looks like a SHA-256 is not enough: otherwise a receipt from a
    different market snapshot could promote a legacy case to ``level=4``.
    """
    bundle = _read(repo / BUNDLE_REL)
    if not isinstance(bundle, dict) or bundle.get("schema") != "cyqnt.input/v1":
        return None
    return _canonical_json_sha256(bundle)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _normalized_frozen_conditions(path: Path) -> Optional[Tuple[str, Tuple[Any, ...]]]:
    """Return canonical hash plus closed-vocabulary conditions for one receipt.

    The legacy ``*_conditions.json`` is an extraction/reviewer artifact and can
    contain quote fields.  It must never be treated as the evaluator input just
    because the filename is convenient.  The separately named frozen file has
    to pass the same closed, privacy-safe condition boundary as
    :func:`tools.nl2yaml.evaluate.evaluate_frozen` before its hash can attest to
    a runtime receipt.
    """
    if not path.is_file():
        return None
    conditions = _read(path)
    if not isinstance(conditions, list):
        return None
    try:
        # The evaluator first canonicalizes JSON, then normalizes every mapping.
        # Do the same work here before using a hash as promotion evidence.
        canonical = json.loads(json.dumps(
            conditions,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ))
        if not all(isinstance(condition, dict) for condition in canonical):
            return None
        normalized = tuple(normalize_condition(condition) for condition in canonical)
    except (TypeError, ValueError):
        return None
    digest = _canonical_json_sha256(canonical)
    if digest is None:
        return None
    return digest, normalized


def _frozen_conditions_evidence(path: Path) -> Optional[Tuple[str, int]]:
    """Return ``(canonical_sha256, count)`` for one evaluator condition input."""
    frozen = _normalized_frozen_conditions(path)
    if frozen is None:
        return None
    digest, normalized = frozen
    return digest, len(normalized)


# A v1 frozen condition does not carry a unit, timeframe, conflict set, or the
# whole legacy operator vocabulary.  Promotion is therefore intentionally
# limited to the one lossless legacy spelling that the frozen gate can express.
# Add a row only after it has a similarly exact, reviewed mapping; a broad
# "same subject and count" fallback would turn an unrelated receipt into a
# claimed execution of a different user condition.
_RECEIPT_COMPARISON_UNITS = {
    "quote_volume_24h": S.Unit.USD,
}
_LEGACY_COMPARISON_OPS = {
    S.Operator.GT: ">",
    S.Operator.GTE: ">=",
    S.Operator.LT: "<",
    S.Operator.LTE: "<=",
    S.Operator.EQ: "==",
}


def _same_canonical_json(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""
    left_hash = _canonical_json_sha256(left)
    right_hash = _canonical_json_sha256(right)
    return left_hash is not None and left_hash == right_hash


def _legacy_condition_matches_frozen(legacy: "S.Condition", frozen: Any) -> bool:
    """Return whether one frozen v1 predicate is a lossless legacy counterpart.

    This is deliberately smaller than either source vocabulary.  The frozen
    input cannot safely represent a legacy timeframe, unit ambiguity, ranking,
    or conflict relation, so those shapes fail closed rather than being
    approximately matched.  Current approved mapping: a universe-wide USD
    quote-volume comparison becomes the frozen cross-section ``compare`` form.
    """
    expected_unit = _RECEIPT_COMPARISON_UNITS.get(legacy.subject)
    expected_op = _LEGACY_COMPARISON_OPS.get(legacy.operator)
    if expected_unit is None or expected_op is None:
        return False
    if (
        legacy.polarity is not S.Polarity.INCLUDE
        or legacy.scope is not S.Scope.UNIVERSE
        or legacy.granularity is not S.EvaluationGranularity.UNIVERSE
        or legacy.measurability is not S.Measurability.QUANTIFIED
        or legacy.unit is not expected_unit
        or legacy.timeframe is not None
        or legacy.is_ranking
        or legacy.rank_direction is not None
        or legacy.ambiguity_type is not None
        or legacy.conflicts_with
    ):
        return False
    if not isinstance(legacy.value, (int, float)) or isinstance(legacy.value, bool):
        return False

    frozen_value = getattr(frozen, "value", None)
    if (
        getattr(frozen, "id", None) != legacy.cid
        or getattr(frozen, "subject", None) != legacy.subject
        or getattr(frozen, "operator", None) != "compare"
        or getattr(frozen, "polarity", None) != "require"
        or getattr(frozen, "scope", None) != "cross_section"
        or getattr(frozen, "quantified", None) is not True
        or getattr(frozen, "ambiguity_type", None) is not None
        or not isinstance(frozen_value, dict)
        or set(frozen_value) != {"op", "threshold"}
        or frozen_value.get("op") != expected_op
    ):
        return False
    return _same_canonical_json(frozen_value.get("threshold"), legacy.value)


def _legacy_conditions_match_frozen(
    legacy_conditions: List["S.Condition"], frozen_conditions: Tuple[Any, ...],
) -> bool:
    """Require an exact id set and a lossless semantic pair for every condition."""
    if not legacy_conditions or not frozen_conditions:
        return False
    legacy_by_id = {condition.cid: condition for condition in legacy_conditions}
    frozen_by_id = {getattr(condition, "id", None): condition
                    for condition in frozen_conditions}
    # Dict construction must not erase a duplicate or unnamed condition.  A
    # count-only match was the original bug: it let c1's receipt certify a
    # different one-condition request.
    if (
        len(legacy_by_id) != len(legacy_conditions)
        or len(frozen_by_id) != len(frozen_conditions)
        or any(not isinstance(cid, str) or not cid for cid in legacy_by_id)
        or any(not isinstance(cid, str) or not cid for cid in frozen_by_id)
        or set(legacy_by_id) != set(frozen_by_id)
    ):
        return False
    return all(
        _legacy_condition_matches_frozen(legacy, frozen_by_id[cid])
        for cid, legacy in legacy_by_id.items()
    )


def _receipt_state(
    path: Path,
    *,
    frozen_conditions_path: Path,
    yaml_text: str,
    user_text: str,
    conditions: List["S.Condition"],
    pinned_bundle_sha256: Optional[str],
) -> str:
    """Classify one optional frozen-evaluation receipt without trusting it.

    ``verified`` is deliberately a narrow *local-linkage* state.  Besides
    matching the public evaluator schema and the exact YAML text that will be
    stored as gold, it requires a receipt claiming a runtime-valid replay on
    this repository's pinned bundle, an exact adjacent privacy-safe condition
    input with lossless legacy-condition correspondence, and an emitted
    signal-batch hash.  The importer does not rerun the evaluator or verify a
    signature, so ``verified`` is never cryptographic proof that execution
    happened.  The caller only promotes a case when this returns ``verified``;
    all other states are safe, explainable caps at intent-reconciled level 3.

    This function returns compact state names only.  It must not surface a
    malformed receipt's text or values because source directories may be
    internal/private.
    """
    if not path.exists():
        return "missing"
    if not path.is_file():
        return "invalid"
    receipt = _read(path)
    if not isinstance(receipt, dict):
        return "invalid"

    frozen_conditions = _normalized_frozen_conditions(frozen_conditions_path)
    if frozen_conditions is None:
        return "invalid"
    conditions_sha256, normalized_frozen_conditions = frozen_conditions
    frozen_condition_count = len(normalized_frozen_conditions)
    if not _legacy_conditions_match_frozen(conditions, normalized_frozen_conditions):
        return "invalid"

    expected_yaml_sha = S.sha256_hex(yaml_text)
    expected_nl_sha = S.sha256_hex(user_text)
    if (
        receipt.get("schema") != frozen_evaluate.RECEIPT_SCHEMA
        or receipt.get("yaml_sha256") != expected_yaml_sha
        or receipt.get("nl_sha256") != expected_nl_sha
        or receipt.get("conditions_sha256") != conditions_sha256
        or receipt.get("runtime_valid") is not True
        or receipt.get("draft_valid") is not True
        or not _is_sha256(receipt.get("bundle_sha256"))
        or receipt.get("bundle_sha256") != pinned_bundle_sha256
        or not _is_sha256(receipt.get("signal_batch_sha256"))
    ):
        return "invalid"

    signal_count = receipt.get("signal_count")
    if (not isinstance(signal_count, int) or isinstance(signal_count, bool)
            or signal_count < 0):
        return "invalid"
    candidate_count = receipt.get("selection_candidate_count")
    if candidate_count is not None and (
            not isinstance(candidate_count, int) or isinstance(candidate_count, bool)
            or candidate_count < 0):
        return "invalid"

    gate_statuses = receipt.get("gate_statuses")
    if (not isinstance(gate_statuses, dict)
            or any(gate_statuses.get(gate) != "passed"
                   for gate in _REQUIRED_RUNTIME_GATES)):
        return "invalid"

    summary = receipt.get("conditions")
    if not isinstance(summary, dict):
        return "invalid"
    requested = summary.get("requested_count")
    evaluated = summary.get("evaluated_count")
    if (not isinstance(requested, int) or isinstance(requested, bool)
            or not isinstance(evaluated, int) or isinstance(evaluated, bool)
            or requested <= 0 or evaluated != requested
            or requested != len(conditions)
            or frozen_condition_count != len(conditions)):
        return "invalid"
    return "verified"


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


#: Subject prefixes the extractor uses for asks about the OUTPUT rather than
#: about which symbols come back — "列出幣名跟現價", "報告照這個順序寫". They are
#: unmeasurable because there is nothing to measure, not because the criterion is
#: vague, and the blanket ``vague -> UNDEFINED`` rule below could not tell the
#: difference: it labelled "show me the price" as an undeclared reading, which
#: would then demand a strategy.assumptions[] entry for it. Ten of the demo
#: corpus's 48 UNDEFINED labels were this.
#:
#: Deliberately tight — two prefixes, no fuzzy matching. Stripping the label off
#: a genuinely ambiguous condition turns the gate off for it silently, which is
#: the worse of the two errors, so a deliverable that does not match a prefix
#: (``sentiment_logic_writeup``) keeps its UNDEFINED and stays visible. The
#: reclassification is recorded as ``scope=reporting`` rather than applied
#: invisibly, so a reviewer can list exactly what moved.
_REPORTING_PREFIXES = ("output_", "report_")


def _conditions(raw: Dict[str, Any]) -> List[S.Condition]:
    """Structured conditions only — ``quote`` stays in the internal record.

    Also labels ambiguity, which is the point of running the mapping at all:
    - a numeric threshold whose unit did not survive the mapping is ``UNIT``
      ambiguous (the demo proved the two readings select different names);
    - a ``vague``/``unmeasurable`` condition is ``UNDEFINED``;
    - ``CONFLICT`` cannot be inferred from one condition in isolation, so it is
      filled afterwards by :func:`_mark_conflicts`.
    """
    out: List[S.Condition] = []
    for c in raw.get("conditions") or []:
        raw_unit = c.get("unit")
        unit = _UNIT.get(str(raw_unit).strip()) if raw_unit else None
        meas = _MEASURABILITY.get(str(c.get("measurability") or "quantified"),
                                  S.Measurability.QUANTIFIED)
        op = _OPERATOR.get(c.get("operator"), S.Operator.UNSPECIFIED)
        value = c.get("value")
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)

        ambiguity: Optional[S.AmbiguityType] = None
        if numeric and unit is None:
            # The extractor saw a number but could not name its unit — e.g. it
            # wrote "计价货币金额(一千万,币别未明说)". That is the ambiguity, not a
            # gap to be filled in with a plausible default.
            ambiguity = S.AmbiguityType.UNIT
            unit = S.Unit.COUNT          # carry the number without asserting USD
        elif meas is not S.Measurability.QUANTIFIED:
            ambiguity = S.AmbiguityType.UNDEFINED

        # The schema forbids QUANTIFIED with operator=unspecified (that pairing is
        # how a vague ask becomes an invented threshold). An extractor that gave a
        # number but no comparator has not quantified anything yet.
        if meas is S.Measurability.QUANTIFIED and op is S.Operator.UNSPECIFIED:
            meas = S.Measurability.VAGUE
            ambiguity = ambiguity or S.AmbiguityType.UNDEFINED
        if meas is S.Measurability.QUANTIFIED and value is None and \
                op not in (S.Operator.EXISTS, S.Operator.ABSENT):
            meas = S.Measurability.VAGUE
            op = S.Operator.UNSPECIFIED
            ambiguity = ambiguity or S.AmbiguityType.UNDEFINED
        if meas is S.Measurability.UNMEASURABLE:
            op = S.Operator.UNSPECIFIED

        scope = _SCOPE.get(str(c.get("scope") or "universe"), S.Scope.UNIVERSE)
        if str(c.get("subject") or "").startswith(_REPORTING_PREFIXES):
            scope = S.Scope.REPORTING
            ambiguity = None

        gran = _GRANULARITY.get(str(c.get("scope") or "universe"),
                                S.EvaluationGranularity.UNIVERSE)
        is_rank = bool(c.get("is_ranking") or False)
        rank_dir = c.get("rank_direction") if is_rank else None
        if is_rank and not rank_dir:
            # "排名" with no end named: recording desc here would be inventing it.
            is_rank, rank_dir = False, None
            ambiguity = ambiguity or S.AmbiguityType.UNDEFINED

        out.append(S.Condition(
            cid=str(c.get("cid")),
            polarity=_POLARITY.get(str(c.get("polarity") or "require"), S.Polarity.INCLUDE),
            subject=str(c.get("subject") or "unknown"),
            operator=op,
            value=value,
            unit=unit,
            scope=scope,
            timeframe=c.get("timeframe"),
            measurability=meas,
            is_ranking=is_rank,
            rank_direction=rank_dir,
            granularity=gran,
            ambiguity_type=ambiguity,
        ))
    return out


#: Subject pairs that cannot both be honoured. The demo's E1 asked for both and
#: got a basket from the wrong end of the market with nothing admitting a choice.
_CONFLICTING_SUBJECTS = [
    ({"price_change_24h", "top_gainers", "momentum"},
     {"accumulation", "basing", "consolidation", "bottoming", "absorption"}),
]


def _mark_conflicts(conds: List[S.Condition]) -> int:
    """Label mutually exclusive conditions on *both* sides.

    One-sided labelling is worse than none: the converter satisfies the unlabelled
    side and reports no ambiguity at all.
    """
    n = 0
    for left, right in _CONFLICTING_SUBJECTS:
        a = [c for c in conds if any(k in c.subject.lower() for k in left)]
        b = [c for c in conds if any(k in c.subject.lower() for k in right)]
        for x in a:
            for y in b:
                if x.cid == y.cid:
                    continue
                for src, dst in ((x, y), (y, x)):
                    if dst.cid not in src.conflicts_with:
                        src.conflicts_with.append(dst.cid)
                        src.ambiguity_type = S.AmbiguityType.CONFLICT
                        n += 1
    return n



_CID_REF = re.compile(r"\b(c\d{1,3})\b")


def sanitize_yaml_for_public(yaml_text: str, user_text: str) -> Tuple[str, int]:
    """Strip the user's wording out of a generated spec, keeping its traceability.

    The converter agents did the *right* engineering thing — they annotated each
    block with the user condition it serves — and that is precisely what leaked:
    two of three specs carried verbatim request text in comments (one line was
    literally labelled "原始需求(逐字)"). Both remotes are public and this file is
    the training target, so a model taught from it would learn to echo the user's
    words into every spec it writes.

    Traceability survives by referring to the ``cid`` instead of quoting: the words
    stay in :class:`InternalCaseRecord.conditions_with_quotes`, keyed by that cid.
    A comment carrying no cid and overlapping the request is dropped outright —
    rewriting it would be guessing at what it meant to say.

    Returns the sanitized text and the number of lines changed.
    """
    out: List[str] = []
    changed = 0
    for line in yaml_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#") or len(stripped) < 8:
            out.append(line)
            continue
        if not S.verbatim_overlap(stripped, user_text):
            out.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        cids = _CID_REF.findall(stripped)
        if cids:
            out.append(f"{indent}# {', '.join(dict.fromkeys(cids))}"
                       f" (wording in the internal record, not repeated here)")
        changed += 1
    text = "\n".join(out) + ("\n" if yaml_text.endswith("\n") else "")

    # description is the other place a request gets paraphrased into the artifact,
    # and unlike a comment it ships inside every emitted signal.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "description:" in line and S.verbatim_overlap(line, user_text):
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f'{indent}description: "generated from extracted conditions"'
            changed += 1
    text = "\n".join(lines) + ("\n" if yaml_text.endswith("\n") else "")
    text, n_num = _scientific_notation(text)
    return text, changed + n_num


#: A plain integer of 8+ digits is indistinguishable from a Telegram user id, so
#: the privacy sweep flags it and cannot be taught otherwise — 10000000 (a "一千萬"
#: turnover floor) and a user id are the same shape. Writing the threshold as 1.0e7
#: removes the ambiguity at the source instead of weakening the guard, and reads
#: better besides. Only bare scalars after ``key:`` are touched; anything inside a
#: string stays as the author wrote it.
# Matches ``key: 12345678`` both as a block scalar and inside a flow mapping
# (``params: { min_quote_volume: 10000000 }`` is how the converter writes it).
# The trailing lookahead keeps it off anything that is part of a longer token.
_BIG_INT_SCALAR = re.compile(r"([A-Za-z_][\w.-]*:\s*)(\d{8,})(?=\s*[,}\]]|\s*(?:#|$))")


def _scientific_notation(text: str) -> Tuple[str, int]:
    n = 0

    def repl(m: "re.Match[str]") -> str:
        nonlocal n
        value = int(m.group(2))
        mantissa, exp = f"{value:e}".split("e")
        # Both the decimal point and the explicit exponent sign are load-bearing:
        # PyYAML resolves floats with YAML 1.1's pattern, where the sign is
        # mandatory, so "1e7" AND "1.0e7" both parse as *strings*.
        # filter_quote_volume happens to float() its argument and so survives
        # either — which is exactly how a string would have shipped unnoticed
        # until some block that does not coerce reads it.
        pretty = f"{float(mantissa):.1f}e{int(exp):+d}"
        if float(pretty) != float(value):        # never trade precision for shape
            return m.group(0)
        n += 1
        return f"{m.group(1)}{pretty}"

    return _BIG_INT_SCALAR.sub(repl, text), n


def _tier(caps: List["S.CapabilityEntry"], conds: List["S.Condition"]) -> "S.CaseTier":
    """Tier by what blocks conversion, not by how many conditions there are.

    Ordered by which answer changes what we do next: a gap gets a block built, a
    proxy gets a human decision, an underspecified ask gets a question back. Any
    unsupported condition dominates, because shipping a basket that silently drops
    it is the failure this whole pipeline exists to stop.
    """
    verdicts = [c.verdict for c in caps]
    if any(v is S.CapabilityVerdict.UNSUPPORTED for v in verdicts):
        return S.CaseTier.T3_BLOCKED_BY_GAP
    if any(v is S.CapabilityVerdict.PROXY for v in verdicts):
        return S.CaseTier.T2_PROXY_NEEDED
    ambiguous = sum(1 for c in conds if c.ambiguity_type is not None)
    if conds and ambiguous * 2 >= len(conds):
        return S.CaseTier.T0_UNDERSPECIFIED
    return S.CaseTier.T1_EXPRESSIBLE


def _capability(report: Optional[Dict[str, Any]],
                conds: List[S.Condition]) -> List[S.CapabilityEntry]:
    """Map the converter's per-condition expressibility onto the closed gap set.

    A missing entry defaults to ``unknown`` rather than ``expressible``: the
    optimistic default is what lets an unexpressed condition look handled.
    """
    by_cid = {e.get("cid"): e for e in ((report or {}).get("condition_map") or [])}
    unexp = {u.get("cid"): u for u in ((report or {}).get("unexpressible") or [])}
    subj = {c.cid: c.subject for c in conds}
    out: List[S.CapabilityEntry] = []
    for c in conds:
        entry = by_cid.get(c.cid) or {}
        verdict = str(entry.get("verdict") or "").lower()
        if c.cid in unexp or verdict == "unexpressible":
            out.append(S.CapabilityEntry(
                cid=c.cid, verdict=S.CapabilityVerdict.UNSUPPORTED,
                gap_id=_gap_for(subj.get(c.cid, ""))))
        elif verdict == "expressible":
            out.append(S.CapabilityEntry(cid=c.cid, verdict=S.CapabilityVerdict.SUPPORTED))
        else:
            out.append(S.CapabilityEntry(cid=c.cid, verdict=S.CapabilityVerdict.UNDECIDABLE))
    return out


def _gold(verdict: Optional[Dict[str, Any]], yaml_text: Optional[str],
          conds: List[S.Condition], caps: List[S.CapabilityEntry], *,
          receipt_verified: bool = False) -> Dict[str, Any]:
    """Derive gold fields from the independent reviewer's verdict.

    A reviewer plus YAML establishes intent reconciliation (level 3), not an
    execution fact.  Level 4 requires :func:`_receipt_state` to link the exact
    public YAML to a locally claimed runtime-valid frozen receipt; this
    migration does not rerun that receipt and therefore makes no cryptographic
    authenticity claim.  The receipt is execution provenance only: it never
    upgrades the legacy reviewer's per-condition verdict from ``HUMAN`` to
    ``EXECUTED`` (in particular, a G1e-failed run must not certify a human
    ``satisfied`` label).  Level 5 remains unavailable to this migration: it
    means *all* checkable conditions were satisfied, while the legacy review
    records omissions/violations.
    """
    v = verdict or {}
    cvs = v.get("condition_verdicts") or []
    checked = {c.get("cid"): c for c in cvs}
    expressed = {c.cid for c, cap in zip(conds, caps)
                 if cap.verdict is S.CapabilityVerdict.SUPPORTED}

    verdicts: List[S.ConditionVerdict] = []
    proxied: List[str] = []
    # The migration reads a receipt but does not replay it.  It may therefore
    # establish level-4 execution *provenance*, never per-condition execution
    # evidence; every status below remains the independent human review.
    checked_by = S.CheckedBy.HUMAN
    for c in conds:
        raw = checked.get(c.cid) or {}
        status_raw = str(raw.get("verdict") or "").lower()
        if raw.get("silently_proxied"):
            proxied.append(c.cid)
            verdicts.append(S.ConditionVerdict(
                cid=c.cid, status=S.ConditionCheckStatus.PROXIED,
                checked_by=checked_by))
            continue
        status = _STATUS.get(status_raw)
        if status is None:                     # not_expressed / unknown
            if c.cid in expressed:
                # Expressible, yet the spec never asked and the output cannot say.
                # Recorded as an explicit UNVERIFIABLE verdict rather than a side
                # list, so "we could have checked this and did not" is visible in
                # the same place every other check lives.
                verdicts.append(S.ConditionVerdict(
                    cid=c.cid, status=S.ConditionCheckStatus.UNVERIFIABLE,
                    checked_by=checked_by))
            continue
        verdicts.append(S.ConditionVerdict(
            cid=c.cid, status=status, checked_by=checked_by))

    refused = not yaml_text
    if refused:
        level = S.GoldVerificationLevel.UNPARSED
        source = S.GoldSource.NONE
        specs: List[S.GoldSpec] = []
    else:
        # YAML+review reaches at most intent reconciliation.  Only a safe
        # receipt ties this exact YAML to the pinned input and emitted batch.
        level = (S.GoldVerificationLevel.EXECUTED_ON_PINNED_BUNDLE
                 if receipt_verified
                 else S.GoldVerificationLevel.INTENT_RECONCILED)
        source = S.GoldSource.ATTEMPT_K
        specs = [S.GoldSpec(role=S.GoldRole.PRIMARY, yaml_text=yaml_text)]

    gap_ids = sorted({cap.gap_id for cap in caps if cap.gap_id}, key=lambda g: g.value)
    return dict(
        gold_specs=specs,
        gold_source=source,
        gold_verification_level=level,
        gold_condition_verdicts=verdicts,
        # proxy_used is derived from proxy_cids by the record. A *violated*
        # condition is a different defect (the spec tried and got it wrong) and
        # must not be laundered into "a proxy was used".
        proxy_cids=sorted(set(proxied)),
        gold_is_refusal=refused,
        refusal_gap_ids=gap_ids if refused else [],
    )



def _drop_existing(path: Path, shas: set) -> int:
    """Remove rows for the texts about to be rewritten, so a re-run replaces.

    Matches on the hash of the source text however the record spells it. The
    public record stores ``text_sha256`` outright; the INTERNAL one does not
    carry it at all — it holds ``user_text_raw`` — so a lookup of that one
    key found nothing there and the internal file appended five rows every
    single run. It had reached forty rows for five conversations before
    anyone counted, and it looked exactly like a forty-case dataset.
    """
    if not path.exists():
        return 0
    kept, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        raw = row.get("user_text_raw")
        keys = {row.get("text_sha256")}
        if isinstance(raw, str) and raw:
            keys.add(S.sha256_hex(raw))
        if keys & shas:
            dropped += 1
        else:
            kept.append(line)
    path.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
    return dropped

def ingest(src: Path, public_out: Path, internal_out: Path, repo: Path) -> Dict[str, Any]:
    cases = _read(src / "cases.json") or {}
    if not cases:
        raise SystemExit(f"no cases.json under {src}")
    sha = _git_sha(repo)
    pinned_bundle_sha256 = _pinned_bundle_sha256(repo)
    summary: Dict[str, Any] = {"written": [], "skipped": [], "replaced": 0}

    # Idempotent by source text. Re-running after a fix must not double the corpus,
    # and case_id is freshly minted each run so it cannot be the dedupe key.

    incoming = set()
    for _cid, _meta in cases.items():
        _t = f"{_meta.get('first_query','')}\n{_meta.get('user_text','')}".strip()
        incoming.add(S.sha256_hex(_t))
    summary["replaced"] = _drop_existing(public_out, incoming)
    summary["replaced_internal"] = _drop_existing(internal_out, incoming)

    for cid, meta in sorted(cases.items()):
        conds_raw = _read(src / f"{cid}_conditions.json") or {}
        report = _read(src / f"{cid}_report.json")
        verdict = _read(src / f"{cid}_verdict.json")
        yaml_path = src / f"{cid}.yaml"
        yaml_text = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else None

        if verdict is None:
            summary["skipped"].append({"case": cid, "why": "no verdict"})
            continue

        user_text = f"{meta.get('first_query','')}\n{meta.get('user_text','')}".strip()
        n_sanitized = 0
        if yaml_text:
            yaml_text, n_sanitized = sanitize_yaml_for_public(yaml_text, user_text)
        conds = _conditions(conds_raw)
        n_conflict = _mark_conflicts(conds)
        caps = _capability(report, conds)
        receipt_state = "not_applicable"
        if yaml_text:
            receipt_state = _receipt_state(
                _receipt_path(src, cid),
                frozen_conditions_path=_frozen_conditions_path(src, cid),
                yaml_text=yaml_text,
                user_text=user_text,
                conditions=conds,
                pinned_bundle_sha256=pinned_bundle_sha256,
            )
        gold = _gold(
            verdict, yaml_text, conds, caps,
            receipt_verified=(receipt_state == "verified"),
        )

        text_sha = S.sha256_hex(user_text)
        case_id = S.case_id_for(text_sha)
        pseudo = S.hmac_pseudonym(f"demo:{cid}")
        quantified = sum(1 for c in conds if (c.measurability or "") == "quantified")
        scopes: Dict[str, int] = {}
        for c in conds:
            scopes[str(c.scope or "unknown")] = scopes.get(str(c.scope or "unknown"), 0) + 1

        case = S.CaseRecord(
            case_id=case_id,
            text_sha256=text_sha,
            canon_sha256=S.canon_sha256_of(user_text),
            # One-off demo picks, deliberately chosen to be distinct, so each is
            # its own cluster of one. mine.py assigns real clusters over the corpus.
            dup_cluster_id=S.cluster_id_for(S.canon_sha256_of(user_text)),
            dup_count=1,
            # Hand-picked real conversations, not product prompt cards.
            preset_case=False,
            pseudonym_id=pseudo,
            lang=str(meta.get("lang") or "zh"),
            # CSV carries zh_variant per row; the demo picks did not archive it, so
            # derive it from the text rather than defaulting (a wrong variant is a
            # wrong tokenisation for the model).
            zh_variant=(_zh_variant(user_text) if str(meta.get("lang") or "zh") == "zh" else None),
            month=str(meta.get("day") or "")[:7] or "2026-05",
            mining_source=S.MiningSource.HUMAN_CURATED,
            source_snapshot_id="trading_intent_chats_2026-05_07_zh_en",
            text_provenance=S.TextProvenance.VERBATIM_INTERNAL,
            conditions=conds,
            capability_map=caps,
            tier=_tier(caps, conds),
            # n_conditions / n_quantified / scope_counts are derived by the record
            # itself; passing them is rejected as a computed-field conflict.
            text_len_chars=len(user_text),
            repo_git_sha=sha,
            converter_model="claude-opus-5",
            temperature=0.0,
            **gold,
        )
        internal = S.InternalCaseRecord(
            case_id=case_id,
            user_text_raw=user_text,
            user_text_redacted=user_text,
            pseudonym_id=pseudo,
            user_id=f"demo:{cid}",
            conditions_with_quotes=conds_raw.get("conditions") or [],
        )
        S.write_case_pair(public_out, internal_out, case, internal)

        # No attempt records. Every demo case converted on the first try with zero
        # gate errors, so there is nothing here that teaches repair and no rejected
        # sibling that could form a preference pair. Worse, the demo never hashed
        # its prompts, and DPO validity is defined as *identical prompt* for chosen
        # and rejected -- writing a placeholder hash would let a later pair builder
        # believe it had that guarantee. An empty attempts log is the honest state.
        n_attempts_dropped = len((report or {}).get("attempts") or [])

        amb: Dict[str, int] = {}
        gran: Dict[str, int] = {}
        for c in conds:
            if c.ambiguity_type:
                amb[c.ambiguity_type.value] = amb.get(c.ambiguity_type.value, 0) + 1
            gran[c.granularity.value] = gran.get(c.granularity.value, 0) + 1

        summary["written"].append({
            "case": cid, "case_id": case_id,
            "n_conditions": len(conds), "tier": case.tier.value,
            "ambiguity": amb, "granularity": gran, "conflict_edges": n_conflict,
            "yaml_lines_sanitized": n_sanitized,
            "receipt": receipt_state,
            "attempts_dropped_no_prompt_hash": n_attempts_dropped,
            "level": int(case.gold_verification_level),
            "refusal": case.gold_is_refusal,
            "proxy_used": case.proxy_used,
            "eligible_sft": case.gold_eligible_for_sft,
            "eligible_loose": case.gold_eligible_loose,
        })
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(S.internal_root() / "raw_demo_2026-08-02"))
    ap.add_argument("--public-out", default="tools/nl2yaml/dataset/cases.jsonl")
    ap.add_argument("--internal-out", default=None)
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    public_out = Path(a.public_out)
    if not public_out.is_absolute():
        public_out = repo / public_out
    public_out.parent.mkdir(parents=True, exist_ok=True)
    # NOT cases_internal.jsonl: mine.py owns that file with a different record
    # shape (51k mined rows). Two record types in one JSONL is a silent corruption
    # that only shows up when a reader assumes one of them.
    internal_out = Path(a.internal_out) if a.internal_out else \
        S.internal_root() / "cases_demo_internal.jsonl"

    summary = ingest(Path(a.src).expanduser(), public_out, internal_out, repo)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\npublic   -> {public_out}")
    print(f"internal -> {internal_out}")


if __name__ == "__main__":
    main()
