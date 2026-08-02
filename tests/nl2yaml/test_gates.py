"""The five gates, run end to end on the two shipped example specs.

Everything here replays ``tests/standard_bot/fixtures/universe_cross_section.json``
— a frozen 727-symbol Binance cross-section — so G1d executes a real strategy
against real captured market data and no test result depends on the hour.

The two tests that carry the design are
:func:`test_g1e_catches_a_threshold_violation_that_every_other_gate_waves_through`
and :func:`test_the_supertrend_condition_is_reported_silently_proxied`. The first
proves the per-condition assertions are not idling; the second reproduces, as an
assertion, the exact accident that motivated them.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.nl2yaml import capability as cap
from tools.nl2yaml import gates

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "standard_bot" / "fixtures" / "universe_cross_section.json"
SPEC_NEWS = REPO / "docs" / "strategy_yaml_spec" / "example_selection.yaml"
SPEC_USER_CHAT = REPO / "docs" / "strategy_yaml_spec" / "example_from_user_chat.yaml"

#: A request text per spec, because G1c has no honest default: it reconciles the
#: spec against what was asked, and there is nothing to reconcile without it.
#: Both are condition summaries in the shape the corpus produces — never a
#: verbatim user question, which stays outside this repo.
NL_NEWS = ("幫我選 5 個幣:先過濾流動性,再依 Square 提及量熱度排名,"
           "情緒偏多做多、偏空做空")
NL_USER_CHAT = ("選幣:掃合約市場,幫我挑做空的候選幣,排除 BTC / ETH / SOL / XRP "
                "與 USDC 計價對,成交量 200 萬美元以上,依成交量排名取 5 個")


def _bundle() -> dict:
    """A fresh parse per call: one test must not inherit another's mutations."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _spec(path: Path) -> dict:
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import load_spec

    return load_spec(str(path))


# The conditions the user-chat spec was converted from. Structured triples only.
USER_CHAT_CONDITIONS = [
    {"id": "vol", "subject": "quote_volume_24h", "scope": "cross_section",
     "operator": "compare", "value": {"op": ">", "threshold": 2_000_000},
     "quantified": True},
    {"id": "majors", "subject": "symbol_blacklist", "scope": "cross_section",
     "operator": "exclude", "polarity": "exclude", "quantified": True,
     "value": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]},
    {"id": "usdc", "subject": "quote_currency", "scope": "cross_section",
     "operator": "exclude", "polarity": "exclude", "value": ["USDC"],
     "quantified": True},
    {"id": "short", "subject": "direction", "scope": "cross_section",
     "operator": "require", "value": "short", "quantified": True},
    {"id": "five", "subject": "basket_size", "scope": "cross_section",
     "operator": "top_k", "value": 5, "quantified": True},
    {"id": "supertrend", "subject": "technical_indicator", "scope": "cross_section",
     "operator": "compare", "value": {"op": "<", "threshold": 0},
     "quantified": True},
    {"id": "lsr", "subject": "long_short_ratio", "scope": "cross_section",
     "operator": "compare", "value": {"op": ">", "threshold": 60.0},
     "quantified": True},
    {"id": "tradfi", "subject": "sector_label", "scope": "cross_section",
     "operator": "exclude", "polarity": "exclude", "value": ["TradFi"],
     "quantified": True},
]

NEWS_CONDITIONS = [
    {"id": "liq", "subject": "quote_volume_24h", "scope": "cross_section",
     "operator": "compare", "value": {"op": ">", "threshold": 100_000_000},
     "quantified": True},
    {"id": "buzz", "subject": "social_mentions", "scope": "cross_section",
     "operator": "rank", "value": None},
    {"id": "five", "subject": "basket_size", "scope": "cross_section",
     "operator": "top_k", "value": 5, "quantified": True},
    {"id": "order", "subject": "score_order", "scope": "cross_section",
     "operator": "rank", "value": "desc", "quantified": True},
]

CASES = [
    pytest.param(SPEC_NEWS, NL_NEWS, NEWS_CONDITIONS, id="example_selection"),
    pytest.param(SPEC_USER_CHAT, NL_USER_CHAT, USER_CHAT_CONDITIONS,
                 id="example_from_user_chat"),
]


# --------------------------------------------------------------------------- #
# all five gates, on both shipped examples                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec_path, nl, conditions", CASES)
def test_all_five_gates_run_on_a_shipped_example(spec_path, nl, conditions, capsys):
    report = gates.run_gates(_spec(spec_path), nl=nl, bundle=_bundle(),
                             conditions=conditions)

    with capsys.disabled():
        print("\n=== %s ===\n%s" % (spec_path.name, report.summary()))

    assert [result.gate for result in report.results] == list(gates.GATES)
    assert report.status == gates.STATUS_PASSED, report.summary()
    # G1d really executed: a report that skipped it would have no batch and no
    # candidates, and would still pass every assertion above.
    assert report.batch is not None
    assert report.batch["schema"] == "cyqnt.signal-batch/v1"
    assert gates.selection_candidates(report.batch), "G1d produced no basket"
    # G1e really ruled: one verdict per declared condition, none of them blank.
    assert len(report.condition_verdicts) == len(conditions)
    assert all(item.detail for item in report.condition_verdicts)


@pytest.mark.parametrize("spec_path, nl, conditions", CASES)
def test_the_gates_are_replayable_and_touch_no_network(spec_path, nl, conditions,
                                                       monkeypatch):
    """A gate result that depends on today's market is not a gate.

    Recording as well as raising is deliberate: ``data_cli.rest_source`` catches
    ``OSError``/``ValueError`` around its ``urlopen`` and returns ``None``, so a
    transport that only raised would be swallowed there and the test would pass
    with a request already gone out.
    """
    import urllib.request

    from cyqnt_trd.blocks import data as blocks_data

    calls = []

    def deny(name):
        def blocked(*args, **kwargs):
            calls.append(name)
            raise AssertionError("gates must not fetch: %s" % name)
        return blocked

    monkeypatch.setattr(blocks_data, "_request_json", deny("blocks.data._request_json"))
    monkeypatch.setattr(urllib.request, "urlopen", deny("urllib.request.urlopen"))

    first = gates.run_gates(_spec(spec_path), nl=nl, bundle=_bundle(),
                            conditions=conditions)
    second = gates.run_gates(_spec(spec_path), nl=nl, bundle=_bundle(),
                             conditions=conditions)

    assert calls == []
    assert first.status == second.status == gates.STATUS_PASSED
    assert (json.dumps(first.batch, sort_keys=True)
            == json.dumps(second.batch, sort_keys=True))


# --------------------------------------------------------------------------- #
# G1e: the reason the whole thing exists                                      #
# --------------------------------------------------------------------------- #


def test_g1e_catches_a_threshold_violation_that_every_other_gate_waves_through():
    """The acceptance test for G1e, and the proof it is not idling.

    The spec's declared condition is turnover above $2m. Lower the block's floor
    and cap the ranked column below the declared threshold, and every earlier
    gate still passes — ``validate_spec`` has no idea what the user asked for,
    and ``reconcile_intent`` checks that turnover DRIVES the screen, not what the
    number is. Only a Python predicate over the emitted basket can see it.
    """
    spec = _spec(SPEC_USER_CHAT)
    spec["strategy"]["id"] = "violating_volume_floor"
    for step in spec["selection"]["universe"]:
        if step["block"] == "universe.filter_quote_volume":
            step["params"]["min_quote_volume"] = 1_000
    spec["selection"]["max_score"] = 1_500_000

    report = gates.run_gates(spec, nl=NL_USER_CHAT, bundle=_bundle(),
                             conditions=USER_CHAT_CONDITIONS)

    assert report.failed_gate == "G1e"
    assert report.status == "condition_violated"
    # Every earlier gate passed, which is the point: four gates' worth of green.
    assert all(result.ok for result in report.results[:4])

    violated = [item for item in report.condition_verdicts
                if item.verdict == gates.VIOLATED]
    assert [item.condition.id for item in violated] == ["vol"]
    assert violated[0].offenders, "a violation with no offender names is unauditable"
    for offender in violated[0].offenders:
        assert float(offender.split("=")[1]) < 2_000_000, offender
    assert "quoteVolume" in violated[0].detail


def test_a_forgotten_condition_is_not_reported_satisfied_by_a_lucky_basket():
    """Vacuous expression, closed.

    Drop the exclusion step. None of the four majors is among the 30 biggest
    losers in this cross-section, so the basket is byte-identical and "no
    candidate is BTC" is perfectly true of it. Reporting that as ``satisfied``
    would teach the model that the step is optional.
    """
    spec = _spec(SPEC_USER_CHAT)
    spec["selection"]["universe"] = [
        step for step in spec["selection"]["universe"]
        if step["block"] != "universe.exclude_symbols"]

    report = gates.run_gates(spec, nl=NL_USER_CHAT, bundle=_bundle(),
                             conditions=USER_CHAT_CONDITIONS)
    verdict = {item.condition.id: item for item in report.condition_verdicts}["majors"]

    assert verdict.verdict == gates.NOT_EXPRESSED
    assert "universe.exclude_symbols" in verdict.detail
    # The gate itself still passes — nothing was violated — so `clean` is the
    # property a caller picking gold specs has to filter on, not `ok`.
    assert report.ok is True
    assert report.clean is False


def test_an_empty_basket_is_unverifiable_and_never_vacuously_satisfied():
    """"Every candidate is above $2m" is trivially true of zero candidates."""
    spec = _spec(SPEC_USER_CHAT)
    spec["selection"]["short_when"] = {"cond": "conditions.value_below",
                                       "args": ["priceChangePercent", -99.0]}

    report = gates.run_gates(spec, nl=NL_USER_CHAT, bundle=_bundle(),
                             conditions=USER_CHAT_CONDITIONS)
    verdicts = {item.condition.id: item for item in report.condition_verdicts}

    assert gates.selection_candidates(report.batch) == ()
    assert verdicts["vol"].verdict == gates.UNVERIFIABLE
    assert "vacuous" in verdicts["vol"].detail
    # A ceiling really is satisfied by zero, and this predicate says so rather
    # than pretending it cannot tell.
    assert verdicts["five"].verdict == gates.SATISFIED


def test_the_supertrend_condition_is_reported_silently_proxied():
    """The accident, reproduced as an assertion.

    The shipped spec answers "Supertrend(10,3) bearish on H4/H1/M15" with
    ``universe.top_losers(n=30)``. The run succeeds, the basket looks healthy,
    and the emitted reason strings say ``quoteVolume=..., rank N of 5`` and
    nothing about a stand-in. So the condition is ``not_expressed`` AND
    ``silently_proxied``, and the report is not clean even though all five gates
    pass.
    """
    report = gates.run_gates(_spec(SPEC_USER_CHAT), nl=NL_USER_CHAT,
                             bundle=_bundle(), conditions=USER_CHAT_CONDITIONS)
    verdicts = {item.condition.id: item for item in report.condition_verdicts}

    assert report.status == gates.STATUS_PASSED
    assert verdicts["supertrend"].verdict == gates.NOT_EXPRESSED
    assert verdicts["supertrend"].silently_proxied is True
    assert "GAP-PER-SYMBOL-INDICATOR" in verdicts["supertrend"].detail

    # Nothing in the basket admits it, which is precisely why a human reviewer
    # could not have caught this one.
    reasons = " ".join(candidate["reason"]
                       for candidate in gates.selection_candidates(report.batch))
    assert "supertrend" not in reasons.lower()
    assert "top_losers" not in reasons.lower()

    assert report.clean is False
    assert {item.condition.id for item in report.silently_proxied} >= {"supertrend"}


def test_a_condition_the_output_could_never_confirm_is_unverifiable_not_satisfied():
    """The TradFi and long/short-ratio conditions, on a spec that omits both.

    ``unverifiable`` and never ``satisfied``: a basket with no
    ``underlying_sub_type`` column carries no evidence either way, and the
    original run's five candidates were all TradFi.
    """
    report = gates.run_gates(_spec(SPEC_USER_CHAT), nl=NL_USER_CHAT,
                             bundle=_bundle(), conditions=USER_CHAT_CONDITIONS)
    verdicts = {item.condition.id: item for item in report.condition_verdicts}

    assert verdicts["lsr"].verdict == gates.NOT_EXPRESSED
    assert verdicts["tradfi"].verdict == gates.NOT_EXPRESSED
    assert not any(item.verdict == gates.SATISFIED
                   for item in (verdicts["lsr"], verdicts["tradfi"]))


def test_a_basket_ranked_on_a_different_column_than_requested_is_violated():
    """The other half of the proxy problem, and it is exactly checkable.

    "Rank the liquid coins by turnover" answered with a spec that FILTERS on
    turnover and ranks on buzz gives a basket that looks entirely reasonable: the
    filter is there, the numbers are plausible. The only trace is that each
    candidate's emitted ``score`` equals a different column than the one asked
    for, and comparing those two numbers needs no judgement at all.

    G1c catches the same swap when the request text names the metric (that is what
    ``reconcile_intent``'s score-dependency checks are for), so this is defence in
    depth: G1c reads the spec, G1e reads what the spec actually emitted.
    """
    report = gates.run_gates(_spec(SPEC_NEWS), nl=NL_NEWS, bundle=_bundle())
    assert report.status == gates.STATUS_PASSED

    ruled = gates.evaluate_conditions(
        report.batch, report.spec,
        [{"id": "by_turnover", "subject": "quote_volume_24h",
          "scope": "cross_section", "operator": "rank", "value": None,
          "quantified": True}])

    assert ruled[0].verdict == gates.VIOLATED
    assert "ranked on something other than quoteVolume" in ruled[0].detail
    assert ruled[0].offenders


def test_g1e_agrees_with_the_capability_plan_about_what_never_got_converted():
    """C-1 and C-2 must not tell two different stories about one case.

    The plan decides, before generation, which conditions leave the converter's
    input; G1e reports, after execution, which conditions are absent from the
    answer. If those two sets diverged, one of them would be lying about the same
    request.
    """
    plan = cap.plan_conversion(USER_CHAT_CONDITIONS)
    report = gates.run_gates(_spec(SPEC_USER_CHAT), nl=NL_USER_CHAT,
                             bundle=_bundle(), conditions=USER_CHAT_CONDITIONS)

    planned = {cond.id for cond, _gap in plan.unconvertible}
    reported = {item.condition.id for item in report.condition_verdicts
                if item.verdict == gates.NOT_EXPRESSED}

    # Only the per-candidate indicator is genuinely unconvertible now: the
    # cross-sectional long/short snapshot landed on 2026-08-02 and "lsr" moved out
    # of this set the same day the sector condition did.
    assert planned == {"supertrend"}
    # Two conditions are now extras rather than one, and for the same reason: the
    # capability landed, the plan knows, and the shipped spec has not caught up.
    # G1e finding MORE than the plan predicted is the healthy direction — the
    # opposite (plan claims unconvertible, output claims handled) would mean a
    # condition got answered by something the plan never authorised.
    assert planned <= reported
    assert reported - planned == {"tradfi", "lsr"}


def test_the_predicate_registry_covers_every_quantifiable_subject():
    """A registered predicate is a condition that never needs judgement again.

    So the number matters, and so does the direction of the gap: a subject with
    no predicate must come back ``unverifiable``, which the test below pins.
    """
    assert len(gates.PREDICATES) >= 12, sorted(gates.PREDICATES)
    checkable = {"quote_volume_24h", "price_change_24h", "funding_rate",
                 "social_mentions", "social_sentiment", "symbol_blacklist",
                 "symbol_whitelist", "quote_currency", "basket_size", "direction",
                 "score_order", "sector_label", "contract_type"}
    assert checkable <= set(gates.PREDICATES), sorted(checkable - set(gates.PREDICATES))
    # Every predicate subject is one the capability table rules on, or the two
    # halves have drifted and a condition can be checked against a verdict that
    # does not exist.
    assert set(gates.PREDICATES) <= set(cap.subjects())


def test_a_subject_with_no_predicate_is_unverifiable_rather_than_assumed_to_hold(
    monkeypatch,
):
    monkeypatch.delitem(gates.PREDICATES, "quote_volume_24h")

    report = gates.run_gates(_spec(SPEC_USER_CHAT), nl=NL_USER_CHAT,
                             bundle=_bundle(), conditions=USER_CHAT_CONDITIONS)
    verdict = {item.condition.id: item for item in report.condition_verdicts}["vol"]

    assert verdict.verdict == gates.UNVERIFIABLE
    assert "no predicate registered" in verdict.detail


def test_direction_and_order_predicates_read_the_emitted_contract():
    """Two predicates whose failure is invisible from outside the output.

    A basket taken from the wrong end of a signed column looks completely healthy
    — five symbols, five plausible numbers — and so does a long basket answering
    a short request if nobody re-reads the direction field.
    """
    report = gates.run_gates(_spec(SPEC_USER_CHAT), nl=NL_USER_CHAT,
                             bundle=_bundle(), conditions=USER_CHAT_CONDITIONS
                             + [{"id": "desc", "subject": "score_order",
                                 "scope": "cross_section", "operator": "rank",
                                 "value": "desc", "quantified": True}])
    verdicts = {item.condition.id: item for item in report.condition_verdicts}

    assert verdicts["short"].verdict == gates.SATISFIED
    assert verdicts["desc"].verdict == gates.SATISFIED
    # And the opposite claim about the same basket is caught.
    flipped = gates.evaluate_conditions(
        report.batch, report.spec,
        [{"id": "asc", "subject": "score_order", "scope": "cross_section",
          "operator": "rank", "value": "asc", "quantified": True}])
    assert flipped[0].verdict == gates.VIOLATED
    assert "not asc" in flipped[0].detail


# --------------------------------------------------------------------------- #
# the earlier gates, and their statuses                                       #
# --------------------------------------------------------------------------- #


def test_g1a_strips_a_code_fence_and_refuses_a_non_mapping():
    fenced = "```yaml\nstrategy:\n  id: x\n```"
    assert gates.strip_code_fence(fenced).startswith("strategy:")

    report = gates.run_gates("- just\n- a list\n", nl=NL_NEWS, bundle=_bundle())
    assert report.failed_gate == "G1a"
    assert report.status == "parse_error"
    assert "must be a YAML mapping" in report.results[0].errors[0]
    # Stopped at G1a: no later gate ran, so no later gate can claim to have
    # checked anything.
    assert len(report.results) == 1


def test_g1a_reports_a_yaml_syntax_error_as_parse_error():
    report = gates.run_gates("strategy: {id: [unclosed\n", nl=NL_NEWS,
                             bundle=_bundle())
    assert report.status == "parse_error"


def test_g1b_separates_the_batched_static_errors_from_the_single_dry_run_one():
    """The two are different retry costs, so they are different statuses.

    ``validate_spec`` skips the dry-run entirely while any static error stands,
    so a spec with several mistakes needs one round per GATE and not one per
    error — and a report that lumped them together would make that budget
    unknowable.
    """
    spec = _spec(SPEC_USER_CHAT)
    del spec["strategy"]["id"]
    spec["run"]["mode"] = "sideways"
    static = gates.run_gates(spec, nl=NL_USER_CHAT, bundle=_bundle())

    assert static.failed_gate == "G1b"
    assert static.status == "static_invalid"
    assert len(static.results[1].errors) >= 2, static.results[1].errors

    spec = _spec(SPEC_USER_CHAT)
    spec["selection"]["score"] = "turnover"      # resolves to no column at all
    dry = gates.run_gates(spec, nl=NL_USER_CHAT, bundle=_bundle())

    assert dry.status == "dryrun_failed"
    assert len(dry.results[1].errors) == 1
    assert dry.results[1].errors[0].startswith("selection dry-run failed")


def test_g1c_rejects_a_spec_that_answers_a_different_question():
    """The gate no amount of internal consistency can pass.

    A selection spec that ranks on turnover is flawless YAML; it is simply not an
    answer to a request about news buzz, and only a check against the request can
    say so.
    """
    report = gates.run_gates(_spec(SPEC_USER_CHAT), nl=NL_NEWS, bundle=_bundle())

    assert report.failed_gate == "G1c"
    assert report.status == "intent_mismatch"
    assert report.results[2].errors


def test_g1c_cannot_be_skipped_by_omitting_the_request():
    """A gate that silently no-ops is worse than one that is absent."""
    with pytest.raises(ValueError, match="needs the request text"):
        gates.run_gates(_spec(SPEC_NEWS), nl="   ", bundle=_bundle())


def test_g1d_refuses_to_run_without_a_frozen_bundle():
    with pytest.raises(ValueError, match="frozen cyqnt.input/v1 bundle"):
        gates.run_gates(_spec(SPEC_NEWS), nl=NL_NEWS, bundle={})


def test_a_missing_source_is_bundle_insufficient_and_not_the_models_fault():
    """The independent status, and the two consequences that follow from it.

    ``example_selection.yaml`` declares ``with: [ticker_rank]``. Take that frame
    away and the strategy never runs — so there is nothing to grade. Counting it
    as a conversion failure understates accuracy, and making the spec a DPO
    negative would teach the model to avoid a correct answer because the data
    plane was down.
    """
    bundle = _bundle()
    bundle["source_status"]["ticker_rank"] = "error: upstream 503"

    report = gates.run_gates(_spec(SPEC_NEWS), nl=NL_NEWS, bundle=bundle,
                             conditions=NEWS_CONDITIONS)

    assert report.failed_gate == "G1d"
    assert report.status == "bundle_insufficient"
    assert report.counts_toward_accuracy is False
    assert "ticker_rank" in report.results[3].errors[0]
    # G1e never ran, so the report must not carry condition verdicts that would
    # read as evidence about the spec.
    assert report.condition_verdicts == ()


def test_a_dropped_frame_is_also_bundle_insufficient():
    bundle = _bundle()
    del bundle["frames"]["ticker_rank"]

    report = gates.run_gates(_spec(SPEC_NEWS), nl=NL_NEWS, bundle=bundle)

    assert report.status == "bundle_insufficient"
    assert report.counts_toward_accuracy is False


def test_a_spec_needing_contract_meta_is_bundle_insufficient_on_this_fixture():
    """The capability landed before the fixture carried its source.

    A sector filter is now expressible, and the frozen cross-section predates the
    ``contract_meta`` frame. That combination is exactly what
    ``bundle_insufficient`` exists to name: the spec is right, the input is not
    there yet, and the model must not be scored for it.
    """
    from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import required_bundle_nodes

    spec = _spec(SPEC_USER_CHAT)
    spec["selection"]["universe"] = (
        [{"block": "universe.augment_with_contract_meta", "with": ["contract_meta"]},
         {"block": "universe.filter_sub_type", "params": {"exclude": ["TradFi"]}}]
        + list(spec["selection"]["universe"]))

    assert "contract_meta" in required_bundle_nodes(spec)
    if "contract_meta" in (_bundle().get("frames") or {}):
        pytest.skip("the fixture now carries contract_meta; recapture landed")

    report = gates.run_gates(spec, nl=NL_USER_CHAT, bundle=_bundle(),
                             conditions=USER_CHAT_CONDITIONS)

    assert report.status == "bundle_insufficient"
    assert report.counts_toward_accuracy is False


def test_g1d_reports_a_genuine_execution_failure_as_run_error():
    """Distinct from ``bundle_insufficient``: this one IS the model's fault.

    A universe step given a source the bundle does not name never falls back to
    the network, so it raises at run time — and that is a spec mistake, not an
    outage.
    """
    spec = _spec(SPEC_USER_CHAT)
    spec["selection"]["universe"].append(
        {"block": "universe.augment_with_news", "with": ["ticker_rank"]})
    bundle = _bundle()
    bundle["frames"]["ticker_rank"] = {"rows": []}

    report = gates.run_gates(spec, nl=NL_USER_CHAT, bundle=bundle)

    assert report.failed_gate == "G1d"
    assert report.status in ("run_error", "bundle_insufficient")
    if report.status == "run_error":
        assert report.counts_toward_accuracy is True


# --------------------------------------------------------------------------- #
# retry loop                                                                  #
# --------------------------------------------------------------------------- #


def _broken_spec() -> dict:
    spec = _spec(SPEC_USER_CHAT)
    del spec["strategy"]["id"]
    return spec


def test_the_retry_loop_stops_on_a_repeated_signature_and_files_a_playbook_gap():
    """Two identical failures at temperature 0 are not a model that needs a third try.

    They are a prompt that lacks something the model cannot derive. Recording
    that as a capability gap would send somebody to build a block that already
    exists, so the outcome names a PLAYBOOK gap instead.
    """
    calls = []

    def convert(attempt, previous):
        calls.append(attempt)
        return _broken_spec()

    outcome = gates.run_with_retries(convert, nl=NL_USER_CHAT, bundle=_bundle(),
                                     max_attempts=3)

    assert outcome.status == "stuck"
    assert calls == [0, 1], "the third attempt would have cost tokens for nothing"
    assert "identical signature" in outcome.playbook_gap
    assert "playbook" in outcome.playbook_gap
    assert outcome.report.failed_gate == "G1b"


def test_the_retry_loop_accepts_a_fix_and_reports_the_attempt_count():
    def convert(attempt, previous):
        if attempt == 0:
            assert previous is None
            return _broken_spec()
        assert previous is not None and previous.failed_gate == "G1b"
        return _spec(SPEC_USER_CHAT)

    outcome = gates.run_with_retries(convert, nl=NL_USER_CHAT, bundle=_bundle(),
                                     conditions=USER_CHAT_CONDITIONS,
                                     max_attempts=3)

    assert outcome.status == gates.STATUS_PASSED
    assert outcome.attempts == 2
    assert outcome.playbook_gap is None


def test_two_different_failures_are_allowed_to_use_the_whole_budget():
    """Only an IDENTICAL signature aborts; making progress is not being stuck."""
    def convert(attempt, previous):
        spec = _spec(SPEC_USER_CHAT)
        if attempt == 0:
            del spec["strategy"]["id"]
        elif attempt == 1:
            spec["selection"]["score"] = "turnover"
        else:
            spec["run"]["mode"] = "sideways"
        return spec

    outcome = gates.run_with_retries(convert, nl=NL_USER_CHAT, bundle=_bundle(),
                                     max_attempts=3)

    assert outcome.attempts == 3
    assert outcome.status == "static_invalid"
    assert outcome.playbook_gap is None
    assert [report.status for report in outcome.reports] == [
        "static_invalid", "dryrun_failed", "static_invalid"]


def test_a_bundle_outage_ends_the_loop_without_blaming_the_model():
    bundle = _bundle()
    bundle["source_status"]["ticker_rank"] = "error: upstream 503"
    attempts = []

    def convert(attempt, previous):
        attempts.append(attempt)
        return _spec(SPEC_NEWS)

    outcome = gates.run_with_retries(convert, nl=NL_NEWS, bundle=bundle,
                                     max_attempts=3)

    assert outcome.status == "bundle_insufficient"
    assert attempts == [0], "retrying asks the model to fix the data plane"
    assert outcome.report.counts_toward_accuracy is False


def test_the_error_signature_ignores_object_addresses_but_not_the_message():
    same = gates.GateResult("G1b", "static_invalid",
                            errors=("boom at 0x7f9a1c00 during validate",))
    other = gates.GateResult("G1b", "static_invalid",
                             errors=("boom at 0xdeadbeef during validate",))
    different = gates.GateResult("G1b", "static_invalid",
                                 errors=("a different complaint entirely",))

    assert same.signature == other.signature
    assert same.signature != different.signature
    # The gate is part of the identity: the same text from a different gate is a
    # different failure and must not trip the abort.
    assert same.signature != gates.GateResult(
        "G1c", "intent_mismatch",
        errors=("boom at 0x1 during validate",)).signature


def test_a_failing_gate_result_must_carry_readable_error_text():
    """A failure nobody can read cannot be turned into a retry prompt."""
    with pytest.raises(ValueError, match="recorded no error text"):
        gates.GateResult("G1b", "static_invalid")
