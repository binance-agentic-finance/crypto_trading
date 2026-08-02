"""Record schema: computed gates, cross-field invariants, JSON-Schema agreement.

The tests that matter most here are not "does a valid record validate" but the
ones that pin down what must stay *impossible*:

* a proxied condition can never produce an SFT-eligible gold, at any
  verification level (``test_proxy_used_forces_ineligible_at_every_level``);
* an attempt-2 sample can never be a DPO negative against an attempt-1 chosen,
  because the prompts differ (``test_dpo_negative_requires_identical_prompt``);
* ``level 5`` cannot coexist with a violated condition — the shape of the run
  that emitted five TradFi perpetuals for a user who asked to exclude TradFi.

Each of those is a rule somebody will eventually want to relax "just for this
batch". Relaxing it turns these tests red, which is the point.
"""

from __future__ import annotations

import json
import sys
import typing
from dataclasses import fields as dataclass_fields
from enum import Enum, IntEnum
from pathlib import Path

import jsonschema
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # tools/ is a namespace package under the repo root
    sys.path.insert(0, str(REPO))

from tools.nl2yaml import schema as S  # noqa: E402


# ---------------------------------------------------------------------------
# factories — no user text anywhere, everything synthetic
# ---------------------------------------------------------------------------

SYNTH_TEXT = "synthetic probe: screen binance futures shorts by liquidity"


def make_case(**over):
    text = over.pop("_text", SYNTH_TEXT)
    base = dict(
        case_id=S.new_case_id(),
        text_sha256=S.sha256_hex(text),
        canon_sha256=S.canon_sha256_of(text),
        dup_cluster_id=S.cluster_id_for(S.canon_sha256_of(text)),
        dup_count=1,
        preset_case=False,
        pseudonym_id=None,
        lang=S.Lang.EN,
        zh_variant=None,
        month="2026-05",
        mining_source=S.MiningSource.SYNTHETIC_TEMPLATE,
        source_snapshot_id="chats_2026-05_07_v1",
        text_provenance=S.TextProvenance.SYNTHETIC,
        text_len_chars=len(text),
    )
    base.update(over)
    return S.CaseRecord(**base)


def cond(cid="c1", **over):
    base = dict(cid=cid, polarity=S.Polarity.INCLUDE, subject="quote_volume_24h",
                operator=S.Operator.GT, value=2e6, unit=S.Unit.USD,
                scope=S.Scope.UNIVERSE, measurability=S.Measurability.QUANTIFIED)
    base.update(over)
    return S.Condition(**base)


def gold(text="strategy:\n  id: probe\n"):
    return S.GoldSpec(role=S.GoldRole.PRIMARY, yaml_text=text)


def make_attempt(**over):
    base = dict(
        case_id=S.new_case_id(),
        attempt_index=1,
        sampling_purpose=S.SamplingPurpose.CONVERT,
        prompt_sha256=S.sha256_hex("prompt-a"),
        yaml_text="strategy:\n  id: probe\n",
        stopped_at_gate=S.Gate.PASSED,
    )
    base.update(over)
    return S.AttemptRecord(**base)


def make_run(**over):
    base = dict(
        run_id="run-0001",
        bundle_id="universe_cross_section",
        bundle_sha256=S.sha256_hex("bundle"),
        bundle_decision_time="2026-08-01T00:00:00Z",
        bundle_snapshot_id="snap-2026-08-01",
        repo_git_sha="2d21006",
        python_version="3.11.15",
        pandas_version="2.2.3",
        signal_count=0,
        bundle_nodes=[S.BundleNode(node="universe", rows=727)],
    )
    base.update(over)
    return S.RunRecord(**base)


# ---------------------------------------------------------------------------
# 1. round trip
# ---------------------------------------------------------------------------

def test_case_round_trips_through_jsonl(tmp_path):
    case = make_case(
        conditions=[cond("c1"), cond("c2", polarity=S.Polarity.EXCLUDE,
                                     subject="underlying_type", operator=S.Operator.NEQ,
                                     value="EQUITY", unit=S.Unit.LABEL)],
        capability_map=[S.CapabilityEntry(cid="c1", verdict=S.CapabilityVerdict.SUPPORTED),
                        S.CapabilityEntry(cid="c2",
                                          verdict=S.CapabilityVerdict.UNSUPPORTED,
                                          gap_id=S.GapId.UNIVERSE_CONTRACT_META)],
        tier=S.CaseTier.T3_BLOCKED_BY_GAP,
        intent_slots=S.IntentSlots(intent=S.Intent.COIN_SELECTION,
                                   direction=S.Direction.SHORT,
                                   market_type=S.MarketType.FUTURES,
                                   symbols=["BTCUSDT"],
                                   universe_scope=S.UniverseScope.ALL_FUTURES,
                                   wants_ranking=True),
        block_refs=["universe.filter_quote_volume", "universe.top_losers"],
        prompt_tokens_by_component={"system": 120, "playbook": 900, "fewshot": 1400},
        gold_specs=[gold()],
        gold_source=S.GoldSource.HANDWRITTEN,
        gold_verification_level=S.GoldVerificationLevel.DRY_RAN,
        resolution_path=S.ResolutionPath.BIG_MODEL,
    )
    path = tmp_path / "cases.jsonl"
    S.write_case(path, case, source_text=S.NO_SOURCE_TEXT)
    back = list(S.read_cases(path))
    assert len(back) == 1
    assert back[0] == case


def test_all_record_kinds_round_trip(tmp_path):
    S.write_attempt(tmp_path / "a.jsonl", make_attempt(), source_text=S.NO_SOURCE_TEXT)
    S.write_gap(tmp_path / "g.jsonl", S.GapRecord(
        gap_id=S.GapId.UNIVERSE_CONTRACT_META, title="exchangeInfo contract metadata",
        needed_capability="universe.filter_underlying_type", hit_count=414,
        dup_weighted_count=1520))
    S.write_run(tmp_path / "r.jsonl", make_run(
        signal_count=2,
        candidates=[S.Candidate(rank=1, symbol="SNDKUSDT", side=S.CandidateSide.SHORT,
                                score=-11.76, attributes={"underlyingType": "EQUITY"}),
                    S.Candidate(rank=2, symbol="SOXLUSDT", side=S.CandidateSide.SHORT,
                                score=-14.31)]))
    assert len(list(S.read_attempts(tmp_path / "a.jsonl"))) == 1
    assert len(list(S.read_gaps(tmp_path / "g.jsonl"))) == 1
    runs = list(S.read_runs(tmp_path / "r.jsonl"))
    assert [c.symbol for c in runs[0].candidates] == ["SNDKUSDT", "SOXLUSDT"]


def _measure_gap(**over):
    """One valid aggregate Gate0 row, deliberately unlike per-case GapRecord."""
    base = {
        "gap_id": "GAP-MARKET-CAP",
        "detectable_by_this_pass": True,
        "undetectable_reason_code": None,
        "dup_weighted_count": 3,
        "distinct_requests": 2,
        "unique_canon": 2,
        "rows_unlocked_if_closed": 1,
        "distinct_requests_unlocked_if_closed": 1,
        "largest_group_share": 0.5,
        "largest_split_group_kind": "duplicate_cluster",
        "tier_rows": {"A": 1, "B": 1, "C": 1, "D": 0},
        "shape_rows": {"selection": 2, "trade": 1, "both": 0, "unclear": 0},
        "blocking_sources": {"threshold/market_cap": 3},
        "co_occurring_gaps": {"GAP-PER-SYMBOL-INDICATOR": 1},
    }
    base.update(over)
    return base


def test_measure_gap_stream_is_versioned_and_round_trips(tmp_path):
    path = tmp_path / "gaps.jsonl"
    row = _measure_gap()

    S.write_measure_gaps(path, [row])

    lines = path.read_text("ascii").splitlines()
    assert json.loads(lines[0]) == {
        "record_schema": S.MEASURE_GAP_RECORD_SCHEMA,
        "schema": S.MEASURE_GAP_FILE_SCHEMA,
    }
    assert list(S.read_measure_gaps(path)) == [row]


def test_measure_gap_stream_refuses_schema_drift_and_private_text_shapes(tmp_path):
    path = tmp_path / "gaps.jsonl"
    path.write_text(json.dumps(_measure_gap()) + "\n", encoding="ascii")
    with pytest.raises(S.RecordError, match="invalid measure_gap_header"):
        list(S.read_measure_gaps(path))

    # ``ensure_ascii=True`` hides the characters on disk.  The reader must
    # reject after JSON decoding, before schema diagnostics can echo a value.
    hidden_non_ascii = _measure_gap(undetectable_reason_code="\u4e0d\u5141\u8a31")
    header = {
        "schema": S.MEASURE_GAP_FILE_SCHEMA,
        "record_schema": S.MEASURE_GAP_RECORD_SCHEMA,
    }
    path.write_text("\n".join((
        json.dumps(header, ensure_ascii=True),
        json.dumps(hidden_non_ascii, ensure_ascii=True),
    )) + "\n", encoding="ascii")
    with pytest.raises(S.PrivacyError, match="non-ASCII"):
        list(S.read_measure_gaps(path))

    # The public artifact has no free-text slots: an ASCII user sentence cannot
    # masquerade as a harmless metric label either.
    with pytest.raises(S.RecordError, match="invalid measure_gap"):
        S.write_measure_gaps(path, [_measure_gap(
            undetectable_reason_code="free-form user sentence")])
    with pytest.raises(S.RecordError, match="invalid measure_gap"):
        S.write_measure_gaps(path, [_measure_gap(
            blocking_sources={"free-form user sentence": 3})])


# ---------------------------------------------------------------------------
# 2. the eligibility formula
# ---------------------------------------------------------------------------

def _level5_case(**over):
    """A clean level-5 case: one condition, satisfied, executed, no proxy."""
    base = dict(
        conditions=[cond("c1")],
        capability_map=[S.CapabilityEntry(cid="c1", verdict=S.CapabilityVerdict.SUPPORTED)],
        tier=S.CaseTier.T1_EXPRESSIBLE,
        gold_specs=[gold()],
        gold_source=S.GoldSource.HANDWRITTEN,
        gold_verification_level=S.GoldVerificationLevel.ALL_CHECKABLE_CONDITIONS_SATISFIED,
        gold_condition_verdicts=[S.ConditionVerdict(
            cid="c1", status=S.ConditionCheckStatus.SATISFIED,
            checked_by=S.CheckedBy.EXECUTED)],
        human_reviewed_by="synthetic-reviewer",
        human_reviewed_at="2026-08-02T00:00:00Z",
    )
    base.update(over)
    return make_case(**base)


def test_clean_level5_case_is_eligible():
    case = _level5_case()
    assert case.proxy_used is False
    assert case.gold_unverifiable_cids == []
    assert case.gold_eligible_for_sft is True
    assert case.gold_eligible_loose is True


@pytest.mark.parametrize("level,strict,loose", [
    (0, False, False),
    (1, False, False),
    (2, False, False),
    (3, False, False),
    (4, False, True),
    (5, True, True),
])
def test_eligibility_thresholds(level, strict, loose):
    """``strict`` needs level >= 5, ``loose`` needs level >= 4."""
    verdicts = [S.ConditionVerdict(cid="c1", status=S.ConditionCheckStatus.SATISFIED,
                                   checked_by=S.CheckedBy.EXECUTED)] if level >= 3 else []
    case = make_case(
        conditions=[cond("c1")],
        capability_map=[S.CapabilityEntry(cid="c1", verdict=S.CapabilityVerdict.SUPPORTED)],
        tier=S.CaseTier.T1_EXPRESSIBLE,
        gold_specs=[] if level == 0 else [gold()],
        gold_source=S.GoldSource.NONE if level == 0 else S.GoldSource.HANDWRITTEN,
        gold_verification_level=level,
        gold_condition_verdicts=verdicts,
        human_reviewed_by="synthetic-reviewer",
        human_reviewed_at="2026-08-02T00:00:00Z",
    )
    assert case.gold_eligible_for_sft is strict
    assert case.gold_eligible_loose is loose


def test_unreviewed_level5_case_is_not_sft_eligible_but_remains_loose():
    case = _level5_case(human_reviewed_by=None, human_reviewed_at=None)

    assert case.gold_eligible_for_sft is False
    assert case.gold_eligible_loose is True


def test_unverifiable_condition_blocks_strict_but_not_loose():
    case = _level5_case(gold_condition_verdicts=[
        S.ConditionVerdict(cid="c1", status=S.ConditionCheckStatus.UNVERIFIABLE,
                           checked_by=S.CheckedBy.EXECUTED)])
    assert case.gold_unverifiable_cids == ["c1"]
    assert case.gold_eligible_for_sft is False
    assert case.gold_eligible_loose is True


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4, 5])
def test_proxy_used_forces_ineligible_at_every_level(level):
    """🔴 The pin. A proxy disqualifies a gold no matter how well it verified.

    If someone "just" relaxes this for high-confidence proxies, this test goes
    red — which is the only place it can go red. A proxy passes ``validate``,
    passes ``run``, and produces a plausible basket, so no gate downstream of the
    dataset can notice that the model learned to substitute a correlated field.
    """
    verdicts = [S.ConditionVerdict(cid="c1", status=S.ConditionCheckStatus.SATISFIED,
                                   checked_by=S.CheckedBy.EXECUTED)] if level >= 3 else []
    if level >= 5:
        # Level 5 must retain a verdict for every extracted condition even
        # when one is already known to be proxied; otherwise c2 could vanish
        # from an ostensibly complete gold record.
        verdicts.append(S.ConditionVerdict(
            cid="c2", status=S.ConditionCheckStatus.PROXIED,
            checked_by=S.CheckedBy.EXECUTED))
    case = make_case(
        conditions=[cond("c1"), cond("c2", subject="supertrend_direction",
                                     operator=S.Operator.EQ, value="bearish",
                                     unit=S.Unit.LABEL, timeframe="4h")],
        capability_map=[
            S.CapabilityEntry(cid="c1", verdict=S.CapabilityVerdict.SUPPORTED),
            S.CapabilityEntry(cid="c2", verdict=S.CapabilityVerdict.PROXY,
                              gap_id=S.GapId.MULTI_TIMEFRAME_CONFLUENCE)],
        tier=S.CaseTier.T2_PROXY_NEEDED,
        gold_specs=[] if level == 0 else [gold()],
        gold_source=S.GoldSource.NONE if level == 0 else S.GoldSource.HANDWRITTEN,
        gold_verification_level=level,
        gold_condition_verdicts=verdicts,
    )
    assert case.proxy_used is True
    assert case.proxy_cids == ["c2"]
    assert case.gold_eligible_for_sft is False, "level %d proxy leaked into strict" % level
    assert case.gold_eligible_loose is False, "level %d proxy leaked into loose" % level


def test_proxy_detected_from_output_check_alone():
    """A ``proxied`` output verdict counts even if the adjudication said supported.

    The two sources disagree exactly when the converter did not realise it was
    substituting — the case we most need to catch.
    """
    case = _level5_case(gold_condition_verdicts=[
        S.ConditionVerdict(cid="c1", status=S.ConditionCheckStatus.PROXIED,
                           checked_by=S.CheckedBy.EXECUTED)])
    assert case.proxy_used is True and case.proxy_cids == ["c1"]
    assert case.gold_eligible_for_sft is False and case.gold_eligible_loose is False


def test_refusal_gold_does_not_inherit_adjudication_proxy():
    """A refusal names the gap instead of encoding the proxy, so it is not proxied."""
    case = make_case(
        conditions=[cond("c1", subject="retail_long_short_ratio", value=0.6,
                         unit=S.Unit.RATIO)],
        capability_map=[S.CapabilityEntry(cid="c1", verdict=S.CapabilityVerdict.PROXY,
                                          gap_id=S.GapId.UNIVERSE_LONG_SHORT_RATIO)],
        tier=S.CaseTier.T2_PROXY_NEEDED,
        gold_specs=[S.GoldSpec(role=S.GoldRole.PRIMARY,
                               yaml_text="# refuse: no long/short ratio block\n")],
        gold_source=S.GoldSource.HANDWRITTEN,
        gold_is_refusal=True,
        refusal_gap_ids=[S.GapId.UNIVERSE_LONG_SHORT_RATIO],
        gold_verification_level=S.GoldVerificationLevel.STATIC_VALID,
    )
    assert case.proxy_used is False and case.proxy_cids == []


def test_refusal_must_name_its_gaps():
    with pytest.raises(S.RecordError, match="refusal_gap_ids"):
        make_case(gold_specs=[gold()], gold_source=S.GoldSource.HANDWRITTEN,
                  gold_verification_level=1, gold_is_refusal=True)


# ---------------------------------------------------------------------------
# 3. computed fields are not caller input
# ---------------------------------------------------------------------------

WRONG_COMPUTED = {
    "n_conditions": 99,
    "n_quantified": 99,
    "scope_counts": {"universe": 99},
    "gold_unverifiable_cids": ["cX"],
    "proxy_used": True,
    "proxy_cids": ["cX"],
    "gold_eligible_for_sft": True,
    "gold_eligible_loose": True,
}


def test_wrong_computed_table_covers_every_computed_field():
    """Adding a computed field without extending this table fails here."""
    assert set(WRONG_COMPUTED) == set(S.CaseRecord.COMPUTED_FIELDS)


@pytest.mark.parametrize("name", sorted(WRONG_COMPUTED))
def test_caller_cannot_assert_a_computed_field(name):
    with pytest.raises(S.ComputedFieldConflict, match=name):
        make_case(**{name: WRONG_COMPUTED[name]})


def test_matching_computed_value_is_accepted_so_records_round_trip():
    case = make_case()
    again = S.CaseRecord.from_dict(json.loads(json.dumps(S.validate_record(case))))
    assert again == case


@pytest.mark.parametrize("name", ["error_signature", "is_valid_dpo_negative"])
def test_attempt_computed_fields_are_not_caller_input(name):
    wrong = {"error_signature": "0" * 16, "is_valid_dpo_negative": True}[name]
    with pytest.raises(S.ComputedFieldConflict, match=name):
        make_attempt(**{name: wrong})


def test_error_signature_is_stable_across_incidental_differences():
    a = make_attempt(stopped_at_gate=S.Gate.BUNDLE_INSUFFICIENT,
                     gate_errors=["DataFrame missing 'priceChangePercent' column "
                                  "(/tmp/run/17/universe.parquet)"])
    b = make_attempt(stopped_at_gate=S.Gate.BUNDLE_INSUFFICIENT,
                     gate_errors=["DataFrame missing 'priceChangePercent' column "
                                  "(/var/other/99/universe.parquet)"])
    assert a.error_signature == b.error_signature


# ---------------------------------------------------------------------------
# 4. DPO validity
# ---------------------------------------------------------------------------

PROMPT_A = S.sha256_hex("prompt-a")
PROMPT_B = S.sha256_hex("prompt-b: prompt-a plus repair feedback")


def test_dpo_negative_requires_identical_prompt():
    """🔴 (attempt 1 rejected, attempt 2 chosen) is not a preference pair.

    Attempt 2's prompt carries the repair feedback, so the two samples answer
    different questions; the gradient would teach "obey the correction you were
    just given", which is not the behaviour we sample at inference time.
    """
    rejected = make_attempt(attempt_index=1, prompt_sha256=PROMPT_A,
                            stopped_at_gate=S.Gate.SCHEMA_INVALID,
                            gate_errors=["'sizing' is a required property"],
                            sampling_purpose=S.SamplingPurpose.DPO_NEGATIVE,
                            dpo_chosen_prompt_sha256=PROMPT_B)
    assert rejected.is_valid_dpo_negative is False

    same_prompt = make_attempt(attempt_index=1, prompt_sha256=PROMPT_A,
                               stopped_at_gate=S.Gate.SCHEMA_INVALID,
                               gate_errors=["'sizing' is a required property"],
                               sampling_purpose=S.SamplingPurpose.DPO_NEGATIVE,
                               dpo_chosen_prompt_sha256=PROMPT_A)
    assert same_prompt.is_valid_dpo_negative is True


@pytest.mark.parametrize("gate", [S.Gate.PARSE_ERROR, S.Gate.BUNDLE_INSUFFICIENT])
def test_dpo_negative_excludes_format_and_data_failures(gate):
    """``parse_error`` teaches formatting; ``bundle_insufficient`` is not the
    model's fault at all."""
    attempt = make_attempt(prompt_sha256=PROMPT_A, stopped_at_gate=gate,
                           gate_errors=["boom"], dpo_chosen_prompt_sha256=PROMPT_A)
    assert attempt.is_valid_dpo_negative is False


def test_passing_attempt_is_never_a_negative():
    assert make_attempt(prompt_sha256=PROMPT_A,
                        dpo_chosen_prompt_sha256=PROMPT_A).is_valid_dpo_negative is False


def test_unpaired_attempt_is_not_a_negative():
    assert make_attempt(stopped_at_gate=S.Gate.INTENT_MISMATCH,
                        gate_errors=["5/5 candidates are TradFi perpetuals"]
                        ).is_valid_dpo_negative is False


def test_intent_mismatch_is_a_usable_negative():
    """The most valuable negative we have: it parsed, it ran, it answered wrong."""
    attempt = make_attempt(prompt_sha256=PROMPT_A, dpo_chosen_prompt_sha256=PROMPT_A,
                           stopped_at_gate=S.Gate.INTENT_MISMATCH,
                           gate_errors=["5/5 candidates have underlyingType=EQUITY"],
                           defect_class=S.DefectClass.SILENTLY_DROPPED_EXCLUSION)
    assert attempt.is_valid_dpo_negative is True


def test_attempt_chain_catches_feedback_that_never_reached_the_prompt():
    first = make_attempt(case_id="01KYZ4WCWMDA7N8WWE6JZBY9X1", attempt_index=1,
                         prompt_sha256=PROMPT_A, stopped_at_gate=S.Gate.SCHEMA_INVALID,
                         gate_errors=["'sizing' is a required property"],
                         feedback_given_to_next_attempt="add a sizing block")
    second = make_attempt(case_id="01KYZ4WCWMDA7N8WWE6JZBY9X1", attempt_index=2,
                          prompt_sha256=PROMPT_A,
                          sampling_purpose=S.SamplingPurpose.REPAIR)
    with pytest.raises(S.RecordError, match="never reached the prompt"):
        S.validate_attempt_chain([first, second])
    second_ok = make_attempt(case_id="01KYZ4WCWMDA7N8WWE6JZBY9X1", attempt_index=2,
                             prompt_sha256=PROMPT_B,
                             sampling_purpose=S.SamplingPurpose.REPAIR)
    S.validate_attempt_chain([first, second_ok])


def test_attempt_chain_rejects_index_gaps():
    a = make_attempt(case_id="01KYZ4WCWMDA7N8WWE6JZBY9X1", attempt_index=1)
    c = make_attempt(case_id="01KYZ4WCWMDA7N8WWE6JZBY9X1", attempt_index=3)
    with pytest.raises(S.RecordError, match="1..n"):
        S.validate_attempt_chain([a, c])


def test_repair_on_first_attempt_is_rejected():
    with pytest.raises(S.RecordError, match="nothing to repair"):
        make_attempt(attempt_index=1, sampling_purpose=S.SamplingPurpose.REPAIR)


def test_gate_and_errors_must_agree():
    with pytest.raises(S.RecordError, match="disagree"):
        make_attempt(stopped_at_gate=S.Gate.PASSED, gate_errors=["something"])
    with pytest.raises(S.RecordError, match="disagree"):
        make_attempt(stopped_at_gate=S.Gate.DRY_RUN_FAILED, gate_errors=[])


def test_cached_tokens_cannot_exceed_prompt_tokens():
    with pytest.raises(S.RecordError, match="cached_prompt_tokens"):
        make_attempt(prompt_tokens=100, cached_prompt_tokens=101)


# ---------------------------------------------------------------------------
# 5. "it ran" is not "it is correct"
# ---------------------------------------------------------------------------

def test_level5_rejects_a_violated_condition():
    with pytest.raises(S.RecordError, match="level 5"):
        _level5_case(gold_condition_verdicts=[
            S.ConditionVerdict(cid="c1", status=S.ConditionCheckStatus.VIOLATED,
                               checked_by=S.CheckedBy.EXECUTED)])


def test_level5_rejects_a_condition_the_converter_never_expressed():
    with pytest.raises(S.RecordError, match="omitted request conditions"):
        _level5_case(gold_condition_verdicts=[
            S.ConditionVerdict(cid="c1", status=S.ConditionCheckStatus.NOT_EXPRESSED,
                               checked_by=S.CheckedBy.STATIC)])


def test_level5_requires_that_something_was_actually_checked():
    with pytest.raises(S.RecordError, match="one verdict for every request condition"):
        _level5_case(gold_condition_verdicts=[])


def test_level5_requires_complete_condition_verdict_coverage():
    """One satisfied condition cannot silently stand in for a second request."""
    with pytest.raises(S.RecordError, match="one verdict for every request condition"):
        _level5_case(
            conditions=[
                cond("c1"),
                cond("c2", subject="stablecoin_exclusion",
                     operator=S.Operator.NEQ, value="USDC", unit=S.Unit.LABEL),
            ],
            capability_map=[
                S.CapabilityEntry(cid="c1", verdict=S.CapabilityVerdict.SUPPORTED),
                S.CapabilityEntry(cid="c2", verdict=S.CapabilityVerdict.SUPPORTED),
            ],
            gold_condition_verdicts=[S.ConditionVerdict(
                cid="c1", status=S.ConditionCheckStatus.SATISFIED,
                checked_by=S.CheckedBy.EXECUTED,
            )],
        )


def test_gold_condition_verdicts_cannot_repeat_a_cid():
    with pytest.raises(S.RecordError, match="duplicate cid in gold_condition_verdicts"):
        _level5_case(gold_condition_verdicts=[
            S.ConditionVerdict(cid="c1", status=S.ConditionCheckStatus.SATISFIED,
                               checked_by=S.CheckedBy.EXECUTED),
            S.ConditionVerdict(cid="c1", status=S.ConditionCheckStatus.SATISFIED,
                               checked_by=S.CheckedBy.EXECUTED),
        ])


def test_attempt_derived_gold_needs_intent_reconciliation():
    """Failure mode 1+2: promoting the attempt that "worked" is not labelling."""
    for level in (1, 2):
        with pytest.raises(S.RecordError, match="attempt_k"):
            make_case(gold_specs=[gold()], gold_source=S.GoldSource.ATTEMPT_K,
                      gold_verification_level=level)
    ok = make_case(gold_specs=[gold()], gold_source=S.GoldSource.ATTEMPT_K,
                   gold_verification_level=3)
    assert ok.gold_source is S.GoldSource.ATTEMPT_K


def test_satisfied_cannot_be_established_statically():
    with pytest.raises(S.RecordError, match="statically"):
        S.ConditionVerdict(cid="c1", status=S.ConditionCheckStatus.SATISFIED,
                           checked_by=S.CheckedBy.STATIC)


def test_gap_closure_forces_re_verification():
    with pytest.raises(S.RecordError, match="re-verified"):
        _level5_case(gold_invalidated_by_gap_closure=True)


def test_unresolved_disagreement_caps_the_level():
    with pytest.raises(S.RecordError, match="disagreement"):
        _level5_case(gold_disagreement=S.GoldDisagreement(
            n_annotators=2, disagreeing_cids=["c1"],
            resolution=S.DisagreementResolution.UNRESOLVED))
    case = _level5_case(gold_disagreement=S.GoldDisagreement(
        n_annotators=2, disagreeing_cids=["c1"],
        resolution=S.DisagreementResolution.THIRD_PARTY))
    assert case.gold_eligible_for_sft is True


def test_exactly_one_primary_gold_spec():
    with pytest.raises(S.RecordError, match="exactly one role=primary"):
        make_case(gold_specs=[gold("a: 1\n"), S.GoldSpec(role=S.GoldRole.PRIMARY,
                                                        yaml_text="b: 2\n")],
                  gold_source=S.GoldSource.HANDWRITTEN, gold_verification_level=1)


def test_gold_source_none_means_no_gold():
    with pytest.raises(S.RecordError, match="gold_source=none"):
        make_case(gold_specs=[gold()], gold_source=S.GoldSource.NONE)


def test_human_edited_gold_must_be_auditable():
    with pytest.raises(S.RecordError, match="human_edit_diff"):
        make_case(gold_specs=[gold()], gold_source=S.GoldSource.HUMAN_EDITED,
                  gold_verification_level=1)


def test_gold_yaml_sha_must_match_the_text():
    with pytest.raises(S.ComputedFieldConflict, match="yaml_sha256"):
        S.GoldSpec(role=S.GoldRole.PRIMARY, yaml_text="a: 1\n", yaml_sha256="0" * 64)


# ---------------------------------------------------------------------------
# 6. conditions and adjudication
# ---------------------------------------------------------------------------

def test_quantified_numeric_condition_needs_a_unit():
    with pytest.raises(S.RecordError, match="needs a unit"):
        cond(unit=None)


def test_vague_condition_cannot_carry_a_made_up_threshold():
    with pytest.raises(S.RecordError, match="unspecified"):
        cond(measurability=S.Measurability.UNMEASURABLE, operator=S.Operator.GT)
    ok = cond(measurability=S.Measurability.VAGUE, operator=S.Operator.UNSPECIFIED,
              value=None, unit=None)
    assert ok.value is None


def test_quantified_condition_cannot_be_unspecified():
    with pytest.raises(S.RecordError, match="made-up threshold"):
        cond(operator=S.Operator.UNSPECIFIED)


def test_exact_n_is_a_distinct_public_operator_from_legacy_top_n():
    exact = cond(subject="basket_size", operator=S.Operator.EXACT_N,
                 value=5, unit=S.Unit.COUNT)
    ceiling = cond(subject="basket_size", operator=S.Operator.TOP_N,
                   value=5, unit=S.Unit.COUNT)

    assert exact.operator is S.Operator.EXACT_N
    assert ceiling.operator is S.Operator.TOP_N


@pytest.mark.parametrize("operator", [S.Operator.EXACT_N, S.Operator.TOP_N,
                                        S.Operator.BOTTOM_N])
@pytest.mark.parametrize("value", [0, -1, True, 5.5, "5"])
def test_public_cardinality_condition_needs_a_positive_integer(operator, value):
    """Public records cannot serialize a coercible cardinality as an integer."""
    with pytest.raises(S.RecordError, match="positive integer count"):
        cond(subject="basket_size", operator=operator, value=value,
             unit=S.Unit.COUNT)


@pytest.mark.parametrize("operator", [S.Operator.EXACT_N, S.Operator.TOP_N,
                                        S.Operator.BOTTOM_N])
def test_public_cardinality_condition_requires_count_unit(operator):
    with pytest.raises(S.RecordError, match="needs unit=count"):
        cond(subject="basket_size", operator=operator, value=5, unit=S.Unit.USD)


def test_rank_direction_pairs_with_is_ranking():
    with pytest.raises(S.RecordError, match="rank_direction"):
        cond(is_ranking=True)
    with pytest.raises(S.RecordError, match="rank_direction"):
        cond(rank_direction=S.RankDirection.DESC)


def test_proxy_verdict_requires_a_named_gap():
    with pytest.raises(S.RecordError, match="requires"):
        S.CapabilityEntry(cid="c1", verdict=S.CapabilityVerdict.PROXY)
    with pytest.raises(S.RecordError, match="must not have"):
        S.CapabilityEntry(cid="c1", verdict=S.CapabilityVerdict.SUPPORTED,
                          gap_id=S.GapId.UNIVERSE_CONTRACT_META)


def test_capability_map_must_cover_every_condition():
    with pytest.raises(S.RecordError, match="adjudicate exactly"):
        make_case(conditions=[cond("c1"), cond("c2")],
                  capability_map=[S.CapabilityEntry(
                      cid="c1", verdict=S.CapabilityVerdict.SUPPORTED)],
                  tier=S.CaseTier.T1_EXPRESSIBLE)


def test_tier_cannot_contradict_the_adjudication():
    with pytest.raises(S.RecordError, match="contradicts capability_map"):
        make_case(conditions=[cond("c1")],
                  capability_map=[S.CapabilityEntry(
                      cid="c1", verdict=S.CapabilityVerdict.UNSUPPORTED,
                      gap_id=S.GapId.ONCHAIN_DATA)],
                  tier=S.CaseTier.T1_EXPRESSIBLE)


def test_counts_are_derived_from_the_conditions():
    case = make_case(
        conditions=[cond("c1"),
                    cond("c2", scope=S.Scope.ENTRY, measurability=S.Measurability.VAGUE,
                         operator=S.Operator.UNSPECIFIED, value=None, unit=None),
                    cond("c3", scope=S.Scope.ENTRY)])
    assert case.n_conditions == 3
    assert case.n_quantified == 2
    assert case.scope_counts == {"entry": 2, "universe": 1}


def test_unknown_enum_value_raises_with_an_actionable_message():
    with pytest.raises(S.RecordError, match="vocabulary is closed"):
        S.CapabilityEntry(cid="c1", verdict="probably_fine")
    with pytest.raises(S.RecordError, match="vocabulary is closed"):
        S.GapRecord(gap_id="something_new", title="t", needed_capability="c",
                    hit_count=1, dup_weighted_count=1)


def test_lang_and_variant_are_scoped_to_zh_en():
    with pytest.raises(S.RecordError, match="vocabulary is closed"):
        make_case(lang="ja")
    with pytest.raises(S.RecordError, match="zh_variant is set iff"):
        make_case(lang=S.Lang.ZH, zh_variant=None)
    with pytest.raises(S.RecordError, match="zh_variant is set iff"):
        make_case(lang=S.Lang.EN, zh_variant=S.ZhVariant.HANS)


def test_pseudonym_shape_is_enforced():
    """A raw user id in ``pseudonym_id`` is the leak this field exists to stop."""
    with pytest.raises(S.RecordError, match="pseudonym_id"):
        make_case(text_provenance=S.TextProvenance.VERBATIM_INTERNAL,
                  pseudonym_id="123456789")


def test_synthetic_cases_have_no_pseudonym():
    with pytest.raises(S.RecordError, match="pseudonym_id"):
        make_case(pseudonym_id="pid_" + "a" * 26)


# ---------------------------------------------------------------------------
# 7. gaps and runs
# ---------------------------------------------------------------------------

def test_dup_weighted_count_cannot_be_below_hit_count():
    with pytest.raises(S.RecordError, match="dup_weighted_count"):
        S.GapRecord(gap_id=S.GapId.UNIVERSE_CONTRACT_META, title="t",
                    needed_capability="c", hit_count=414, dup_weighted_count=200)


def test_closed_gap_names_the_commit():
    with pytest.raises(S.RecordError, match="closed_by_commit"):
        S.GapRecord(gap_id=S.GapId.UNIVERSE_CONTRACT_META, title="t",
                    needed_capability="c", hit_count=1, dup_weighted_count=1,
                    status=S.GapStatus.CLOSED)


def test_gap_examples_are_capped():
    with pytest.raises(S.RecordError, match="at most 5"):
        S.GapRecord(gap_id=S.GapId.ONCHAIN_DATA, title="t", needed_capability="c",
                    hit_count=6, dup_weighted_count=6,
                    example_case_ids=[S.new_case_id() for _ in range(6)])


def test_candidate_ranks_must_be_contiguous():
    with pytest.raises(S.RecordError, match="ranks must be"):
        make_run(signal_count=2,
                 candidates=[S.Candidate(rank=1, symbol="A", side=S.CandidateSide.SHORT),
                             S.Candidate(rank=3, symbol="B", side=S.CandidateSide.SHORT)])


def test_selection_signal_count_is_independent_of_embedded_basket_size():
    run = make_run(signal_count=1,
                   candidates=[S.Candidate(rank=1, symbol="A", side=S.CandidateSide.SHORT),
                               S.Candidate(rank=2, symbol="B", side=S.CandidateSide.SHORT)])
    assert run.signal_count == 1
    assert len(run.candidates) == 2


def test_embedded_basket_requires_an_emitted_signal():
    with pytest.raises(S.RecordError, match="signal_count"):
        make_run(signal_count=0,
                 candidates=[S.Candidate(rank=1, symbol="A", side=S.CandidateSide.SHORT)])


def test_case_linked_run_requires_complete_provenance_and_signal_hash():
    case_id = S.new_case_id()
    with pytest.raises(S.RecordError, match="all-or-none run provenance"):
        make_run(case_id=case_id)
    with pytest.raises(S.RecordError, match="requires signal_batch_sha256"):
        make_run(case_id=case_id, attempt_index=1,
                 yaml_sha256=S.sha256_hex("strategy: {}\n"))

    linked = make_run(case_id=case_id, attempt_index=1,
                      yaml_sha256=S.sha256_hex("strategy: {}\n"),
                      signal_batch_sha256=S.sha256_hex("signal-batch"))
    assert linked.case_id == case_id


def test_candidates_cannot_come_from_an_entirely_empty_bundle():
    with pytest.raises(S.RecordError, match="every frame is empty"):
        make_run(signal_count=1, bundle_nodes=[S.BundleNode(node="universe", rows=0)],
                 candidates=[S.Candidate(rank=1, symbol="A", side=S.CandidateSide.SHORT)])


def test_bundle_decision_time_needs_a_timezone():
    with pytest.raises(S.RecordError, match="point-in-time"):
        make_run(bundle_decision_time="2026-08-01 00:00:00")


# ---------------------------------------------------------------------------
# 8. JSON Schema agreement (drift guard)
# ---------------------------------------------------------------------------

ENUM_DEFS = {cls.__name__: cls for cls in [
    S.Polarity, S.Operator, S.Unit, S.Scope, S.Measurability, S.RankDirection,
    S.EvaluationGranularity, S.AmbiguityType,
    S.CapabilityVerdict, S.GapId, S.GapStatus, S.CaseTier, S.MiningSource,
    S.TextProvenance, S.Lang, S.ZhVariant, S.ResolutionPath, S.Intent, S.MarketType,
    S.Direction, S.UniverseScope, S.GoldRole, S.GoldSource, S.GoldVerificationLevel,
    S.ConditionCheckStatus, S.CheckedBy, S.DisagreementResolution, S.SamplingPurpose,
    S.Gate, S.DefectClass, S.CandidateSide]}


def _common():
    return json.loads((S.SCHEMA_DIR / "common.schema.json").read_text("utf-8"))


def test_every_enum_in_the_module_is_in_the_shared_schema():
    """A vocabulary that is closed in Python and open on disk is not closed."""
    in_module = {name: obj for name, obj in vars(S).items()
                 if isinstance(obj, type) and issubclass(obj, Enum)
                 and obj.__module__ == S.__name__}
    assert set(in_module) == set(ENUM_DEFS), (
        "ENUM_DEFS is out of date; enums only in the module: %s"
        % sorted(set(in_module) - set(ENUM_DEFS)))
    defs = _common()["$defs"]
    assert set(ENUM_DEFS) <= set(defs)


@pytest.mark.parametrize("name", sorted(ENUM_DEFS))
def test_enum_members_match_the_schema(name):
    cls = ENUM_DEFS[name]
    on_disk = _common()["$defs"][name]["enum"]
    expected = [int(m) for m in cls] if issubclass(cls, IntEnum) else [m.value for m in cls]
    assert on_disk == expected


@pytest.mark.parametrize("cls,kind", sorted(S.RECORD_KINDS.items(), key=lambda kv: kv[1]))
def test_record_schema_lists_exactly_the_dataclass_fields(cls, kind):
    schema = json.loads((S.SCHEMA_DIR / S.KIND_SCHEMA_FILES[kind]).read_text("utf-8"))
    names = [f.name for f in dataclass_fields(cls)]
    assert sorted(schema["properties"]) == sorted(names)
    assert sorted(schema["required"]) == sorted(names), (
        "every field is written on every line, so every field is required")
    assert schema["additionalProperties"] is False


def test_nested_schemas_also_forbid_extra_properties():
    """``additionalProperties: false`` everywhere is half of the privacy story:
    an unexpected ``quote`` key fails validation even if a guard is bypassed."""
    for name, body in _common()["$defs"].items():
        if body.get("type") == "object":
            assert body.get("additionalProperties") is False, name


def test_nested_dataclasses_match_their_shared_defs():
    for cls in (S.Condition, S.CapabilityEntry, S.IntentSlots, S.GoldSpec,
                S.ConditionVerdict, S.GoldDisagreement, S.BundleNode, S.Candidate):
        body = _common()["$defs"][cls.__name__]
        names = [f.name for f in dataclass_fields(cls)]
        assert sorted(body["properties"]) == sorted(names), cls.__name__
        assert sorted(body["required"]) == sorted(names), cls.__name__


def test_schema_rejects_an_unknown_key():
    payload = S.validate_record(make_case())
    payload["extra_field"] = 1
    with pytest.raises(jsonschema.ValidationError):
        S.validate_record(payload, "case")


def test_schema_rejects_a_bad_hash_shape():
    payload = S.validate_record(make_case())
    payload["text_sha256"] = "not-a-hash"
    with pytest.raises(jsonschema.ValidationError):
        S.validate_record(payload, "case")


@pytest.mark.parametrize("operator", ["exact_n", "top_n", "bottom_n"])
@pytest.mark.parametrize("value", [True, 5.5, "5"])
def test_json_schema_rejects_non_integral_cardinality_even_without_dataclass_validation(
    operator, value,
):
    """The on-disk schema is an independent fail-closed boundary."""
    record = make_case(
        conditions=[cond("c1", subject="basket_size", operator=S.Operator.EXACT_N,
                         value=5, unit=S.Unit.COUNT)],
        capability_map=[S.CapabilityEntry(
            cid="c1", verdict=S.CapabilityVerdict.SUPPORTED)],
        tier=S.CaseTier.T1_EXPRESSIBLE,
    )
    payload = S.validate_record(record)
    payload["conditions"][0]["operator"] = operator
    payload["conditions"][0]["value"] = value

    with pytest.raises(jsonschema.ValidationError):
        S.validate_record(payload, "case")


def test_validate_record_rejects_unknown_kind():
    with pytest.raises(S.RecordError, match="unknown record kind"):
        S.validate_record({}, "not_a_kind")


# ---------------------------------------------------------------------------
# 9. reading re-checks the invariants
# ---------------------------------------------------------------------------

def test_hand_edited_eligibility_fails_on_read(tmp_path):
    """A file edited to flip the gate must not load. Otherwise the cheapest way
    past every check in this module is a text editor."""
    path = tmp_path / "cases.jsonl"
    S.write_case(path, _level5_case(gold_verification_level=2,
                                    gold_condition_verdicts=[]),
                 source_text=S.NO_SOURCE_TEXT)
    payload = json.loads(path.read_text("utf-8"))
    payload["gold_eligible_for_sft"] = True
    path.write_text(json.dumps(payload) + "\n", "utf-8")
    with pytest.raises(S.ComputedFieldConflict):
        list(S.read_cases(path))


def test_read_reports_the_line_number(tmp_path):
    path = tmp_path / "cases.jsonl"
    S.write_case(path, make_case(), source_text=S.NO_SOURCE_TEXT)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{oops\n")
    with pytest.raises(S.RecordError, match=":2 "):
        list(S.read_cases(path))


def test_level_enum_is_an_ordered_chain():
    levels = list(S.GoldVerificationLevel)
    assert [int(x) for x in levels] == sorted(int(x) for x in levels) == [0, 1, 2, 3, 4, 5]
    assert (S.GoldVerificationLevel.ALL_CHECKABLE_CONDITIONS_SATISFIED
            > S.GoldVerificationLevel.EXECUTED_ON_PINNED_BUNDLE)


def test_serialiser_refuses_an_unplanned_type():
    """No ``str(value)`` fallback: a repr is how user text would sneak through."""
    with pytest.raises(S.RecordError, match="cannot serialise"):
        S._to_jsonable({"x": object()})


def test_typing_hints_resolve():
    """``from __future__ import annotations`` plus dataclasses is a common way to
    ship a record type whose hints cannot be resolved; the schema generator and
    every JSON-Schema drift test above depend on this working."""
    for cls in list(S.RECORD_KINDS) + [S.Condition, S.Candidate]:
        assert typing.get_type_hints(cls)
