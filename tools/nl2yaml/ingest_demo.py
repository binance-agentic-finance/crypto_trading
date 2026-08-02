#!/usr/bin/env python3
"""Turn the 2026-08-02 five-case end-to-end demo into dataset records.

Why this exists: the demo produced real, audited evidence — five user
conversations, the conditions extracted from them, the YAML each was converted
to, the actual run output on a pinned bundle, and an independent per-condition
verdict. That is exactly the shape of a training example, but it was written as
loose JSON in a session scratchpad, which is wiped. Nothing downstream can train
on or re-check a directory that no longer exists.

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
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import schema as S

# The demo ranked a frozen cross-section; a gold verified against it is only
# meaningful while that bundle is pinned, so the id travels with the record.
BUNDLE_REL = "tests/standard_bot/fixtures/universe_cross_section.json"
DEMO_RUN = "demo_2026-08-02_five_case"

#: Verdict vocabulary used by the demo's reviewer -> schema vocabulary.
#: ``not_expressed`` has no ConditionCheckStatus: a condition the spec never
#: attempted is not a *check* that came back negative. It is recorded as an
#: unexpressed cid on the case instead, so "we did not try" can never be read as
#: "we tried and it held".
_STATUS = {
    "satisfied": S.ConditionCheckStatus.SATISFIED,
    "violated": S.ConditionCheckStatus.VIOLATED,
    "unverifiable": S.ConditionCheckStatus.UNVERIFIABLE,
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


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


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
            scope=_SCOPE.get(str(c.get("scope") or "universe"), S.Scope.UNIVERSE),
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
          conds: List[S.Condition], caps: List[S.CapabilityEntry]) -> Dict[str, Any]:
    """Derive gold fields from the independent reviewer's verdict.

    Level is deliberately capped at 4 for every demo case. Level 5 means *all
    checkable conditions satisfied*, and no case cleared that: E1 and E3 each
    carry a ``violated``, and every case has conditions the spec never attempted.
    Marking any of them 5 would put a wrong answer into the training target,
    which is the failure this schema exists to prevent.
    """
    v = verdict or {}
    cvs = v.get("condition_verdicts") or []
    checked = {c.get("cid"): c for c in cvs}
    expressed = {c.cid for c, cap in zip(conds, caps)
                 if cap.verdict is S.CapabilityVerdict.SUPPORTED}

    verdicts: List[S.ConditionVerdict] = []
    proxied: List[str] = []
    violated = False
    for c in conds:
        raw = checked.get(c.cid) or {}
        status_raw = str(raw.get("verdict") or "").lower()
        if raw.get("silently_proxied"):
            proxied.append(c.cid)
            verdicts.append(S.ConditionVerdict(
                cid=c.cid, status=S.ConditionCheckStatus.PROXIED,
                checked_by=S.CheckedBy.EXECUTED))
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
                    checked_by=S.CheckedBy.EXECUTED))
            continue
        if status is S.ConditionCheckStatus.VIOLATED:
            violated = True
        verdicts.append(S.ConditionVerdict(
            cid=c.cid, status=status, checked_by=S.CheckedBy.EXECUTED))

    refused = not yaml_text
    if refused:
        level = S.GoldVerificationLevel.UNPARSED
        source = S.GoldSource.NONE
        specs: List[S.GoldSpec] = []
    else:
        # Ran on the pinned bundle and was reconciled by a reviewer, but never
        # all-conditions-clean -> 4, not 5.
        level = S.GoldVerificationLevel.EXECUTED_ON_PINNED_BUNDLE
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


def ingest(src: Path, public_out: Path, internal_out: Path, repo: Path) -> Dict[str, Any]:
    cases = _read(src / "cases.json") or {}
    if not cases:
        raise SystemExit(f"no cases.json under {src}")
    sha = _git_sha(repo)
    summary: Dict[str, Any] = {"written": [], "skipped": [], "replaced": 0}

    # Idempotent by source text. Re-running after a fix must not double the corpus,
    # and case_id is freshly minted each run so it cannot be the dedupe key.
    def _drop_existing(path: Path, shas: set) -> int:
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
            if row.get("text_sha256") in shas:
                dropped += 1
            else:
                kept.append(line)
        path.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
        return dropped

    incoming = set()
    for _cid, _meta in cases.items():
        _t = f"{_meta.get('first_query','')}\n{_meta.get('user_text','')}".strip()
        incoming.add(S.sha256_hex(_t))
    summary["replaced"] = _drop_existing(public_out, incoming)
    _drop_existing(internal_out, incoming)

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
        gold = _gold(verdict, yaml_text, conds, caps)

        case_id = S.new_case_id()
        pseudo = S.hmac_pseudonym(f"demo:{cid}")
        quantified = sum(1 for c in conds if (c.measurability or "") == "quantified")
        scopes: Dict[str, int] = {}
        for c in conds:
            scopes[str(c.scope or "unknown")] = scopes.get(str(c.scope or "unknown"), 0) + 1

        case = S.CaseRecord(
            case_id=case_id,
            text_sha256=S.sha256_hex(user_text),
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
