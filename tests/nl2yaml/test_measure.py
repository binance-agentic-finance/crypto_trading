"""Tests for the Gate0 measurement pass.

What these pin, in order of how expensive the mistake would be:

1. **Direction.** Every approximation in the proxy map has to push the count of
   convertible cases UP, or the report's headline stops being an upper bound and
   nobody downstream can tell. The two known exceptions are declared and
   quantified, and there are tests for both.
2. **Nothing silently dropped.** An unmapped miner family, a gap that no longer
   appears in the ranking, a condition that vanishes on the way to
   ``plan_conversion`` — each of those makes a case look cheaper than it is.
3. **The privacy boundary.** The report is statistics; a test asserts the
   rendered markdown is pure ASCII, because the guard that catches a leaked
   Chinese quote is the ascii encode on write.
"""

from __future__ import annotations

import json

import pytest

from tools.nl2yaml import capability as cap
from tools.nl2yaml import measure, mine


# ---------------------------------------------------------------------------
# Fixtures: minimal records in the shape remine() produces
# ---------------------------------------------------------------------------

def record(conditions, shape="trade", tier="A", **overrides):
    row = {
        "canon_sha256": "0" * 64,
        "lang": "zh",
        "preset_case": "",
        "dup_cluster_id": "dup_0000000000000000",
        "dup_count": 1,
        "split_group_key": "dup:dup_0000000000000000",
        "families": sorted({c["family"] for c in conditions}),
        "n_families": len({c["family"] for c in conditions}),
        "n_conditions": len(conditions),
        "conditions": conditions,
        "tier": tier,
        "spec_shape": shape,
        "spec_shape_base": shape,
        "is_continuation_fragment": False,
        "fragment_reason": "",
        "leading_chatter": False,
        "is_candidate": True,
    }
    row.update(overrides)
    return row


def cond(family, subject, operator="exists", value=None, polarity="include"):
    return {"family": family, "subject": subject, "operator": operator,
            "value": value, "unit": None, "polarity": polarity}


# ---------------------------------------------------------------------------
# The map: completeness and coherence
# ---------------------------------------------------------------------------

def test_every_miner_family_subject_pair_is_mapped():
    """The map must cover every pair the miner can emit.

    An unmapped pair raises in ``map_condition``, which is the right behaviour,
    but only if this test tells us before a 51k-row run does.
    """
    from tools.nl2yaml.mine import (_ASSET_RX, _INDICATOR_NAMES, _RISK_RX,
                                    _SUBJECT_HINTS, _UNIVERSE_RX)

    expected = set()
    for name, _rx in _SUBJECT_HINTS:
        expected.add(("threshold", name))
    expected.add(("threshold", "unspecified"))
    expected.add(("rank_topn", "universe"))
    for name, _pat in _INDICATOR_NAMES:
        expected.add(("indicator", name))
    expected.add(("direction", "side"))
    for name, _rx in _RISK_RX:
        expected.add(("risk", name))
    for name, _rx in _UNIVERSE_RX:
        expected.add(("universe_filter", name))
    if _ASSET_RX:
        expected.add(("asset", "asset"))

    missing = sorted(expected - set(measure.PROXY_MAP))
    assert not missing, "unmapped miner pairs: %s" % (missing,)


def test_timeframe_is_not_in_the_map_because_it_is_row_level():
    """``timeframe`` is deliberately absent: one interval and three intervals are
    different capability questions, and only the row knows which it is."""
    assert not [k for k in measure.PROXY_MAP if k[0] == "timeframe"]


def test_map_subjects_all_exist_in_the_capability_table():
    ruled = set(cap.subjects())
    for key, row in measure.PROXY_MAP.items():
        if row.fidelity in ("non_condition", "unruled"):
            continue
        assert row.cap_subject in ruled, "%s -> %s" % (key, row.cap_subject)


def test_unruled_subjects_really_are_unruled():
    """If the table starts ruling one of these, the report would keep calling a
    known answer 'unknown' and the strict column would understate forever."""
    for subject in measure.UNRULED_SUBJECTS:
        assert cap.lookup(subject).verdict == cap.UNKNOWN


def test_unruled_subjects_are_all_used():
    used = {row.cap_subject for row in measure.PROXY_MAP.values()
            if row.fidelity == "unruled"}
    used |= {"bar_interval"}   # emitted by _timeframe_conditions, not the map
    assert set(measure.UNRULED_SUBJECTS) == used


def test_bad_fidelity_raises():
    with pytest.raises(ValueError, match="unknown fidelity"):
        measure.ProxySubject("market_cap", "compare", "*", "probably", "why")


def test_bad_operator_raises():
    with pytest.raises(ValueError, match="not a capability operator"):
        measure.ProxySubject("market_cap", "sort_of_compare", "*", "exact", "why")


def test_bad_scope_raises():
    with pytest.raises(ValueError, match="not a capability scope"):
        measure.ProxySubject("market_cap", "compare", "sideways", "exact", "why")


# ---------------------------------------------------------------------------
# Gap coverage
# ---------------------------------------------------------------------------

def test_every_gap_id_is_reachable_or_declared_undetectable():
    """The assertion that runs at import, run again explicitly.

    This is the check that keeps the priority ranking honest: a gap absent from
    the ranking must be absent for a stated reason, because otherwise "not in the
    list" reads as "not needed".
    """
    measure._assert_gap_coverage_is_declared()
    accounted = measure._reachable_gaps() | set(measure.UNDETECTABLE_GAPS)
    assert cap.GAP_IDS <= accounted


def test_undetectable_gaps_are_not_reachable():
    assert not (measure._reachable_gaps() & set(measure.UNDETECTABLE_GAPS))


def test_the_gaps_this_pass_can_see():
    """Pinned so that a change in coverage is a visible diff, not a silent one.

    It was five. ``GAP-SPREAD-DEPTH`` left this set when the cross-sectional half
    of it was CLOSED — the ``book_ticker`` node plus ``universe.augment_with_spread``
    landed, so ``('universe_filter','liquidity')`` scores expressible and no
    longer reaches the gap. The gap id still exists for the per-bar
    microstructure half, which no miner family looks for, so it moved to
    ``measure.UNDETECTABLE_GAPS`` and ``_assert_gap_coverage_is_declared`` is what
    forced that to be written down rather than dropped.
    """
    assert measure._reachable_gaps() == {
        "GAP-COMPOUND-SELECT-THEN-TRADE",
        "GAP-ENTRY-EXIT-PER-CANDIDATE",
        "GAP-MARKET-CAP",
        "GAP-PER-SYMBOL-INDICATOR",
    }
    assert "GAP-SPREAD-DEPTH" in measure.UNDETECTABLE_GAPS


def test_gaps_jsonl_lists_every_gap_id():
    rows = [record([cond("universe_filter", "market_cap")], shape="selection")]
    verdicts = {id(r): measure.plan_row(r) for r in rows}
    gaps = measure.build_gaps(rows, verdicts)
    assert {g["gap_id"] for g in gaps} == set(cap.GAP_IDS)
    ranked = [g for g in gaps if g["detectable_by_this_pass"]]
    assert [g["gap_id"] for g in ranked] == ["GAP-MARKET-CAP"]
    for gap in gaps:
        if not gap["detectable_by_this_pass"]:
            assert gap["undetectable_reason"]
            assert gap["dup_weighted_count"] == 0


# ---------------------------------------------------------------------------
# Scope: the axis every verdict hangs on
# ---------------------------------------------------------------------------

def test_indicator_in_a_trade_request_is_expressible():
    verdict = measure.plan_row(record([cond("indicator", "supertrend")],
                                      shape="trade"))
    assert verdict.gap_ids == ()
    assert verdict.n_convertible == 1


def test_indicator_in_a_selection_request_is_the_supertrend_accident():
    """The whole reason the capability table exists: the same words, the other
    frame, and the answer is a withheld proxy rather than a spec."""
    verdict = measure.plan_row(record([cond("indicator", "supertrend")],
                                      shape="selection"))
    assert verdict.gap_ids == ("GAP-PER-SYMBOL-INDICATOR",)
    assert verdict.n_convertible == 0
    assert verdict.has_not_expressible


def test_universe_filter_keeps_its_frame_inside_a_trade_request():
    """A universe filter is cross-sectional even when the request is a trade, so
    'coins under $50m market cap' must not become expressible by being asked
    about one symbol.

    The subject used to be ``liquidity``, which stopped demonstrating anything
    when the book cross-section landed and made it expressible in BOTH frames.
    ``market_cap`` has no source in this repo at either scope, so it is the
    subject that still isolates the frame rule this test is about.
    """
    verdict = measure.plan_row(record([cond("universe_filter", "market_cap")],
                                      shape="trade"))
    assert verdict.gap_ids == ("GAP-MARKET-CAP",)


def test_a_liquidity_filter_is_no_longer_a_gap_in_either_frame():
    """The other half of the test above, and the reason it had to change.

    ``universe.augment_with_spread`` joins ``bookTicker`` for the whole market, so
    "剔除流動性差的幣" is now answerable — and it is answerable in a trade-shaped
    request too, because the condition is still evaluated on the cross-section.
    Asserted in both shapes so a regression that re-blocks one of them is visible.
    """
    for shape in ("selection", "trade"):
        verdict = measure.plan_row(record([cond("universe_filter", "liquidity")],
                                          shape=shape))
        assert verdict.gap_ids == (), shape


def test_ambiguous_rows_take_the_better_frame():
    """An unclear row holding an indicator is scored on the series frame, where
    the indicator is expressible - the upward choice, and it is the choice this
    report has to make to stay a bound."""
    verdict = measure.plan_row(record([cond("indicator", "rsi")], shape="unclear"))
    assert verdict.scope == measure.SERIES
    assert verdict.gap_ids == ()


def test_ambiguous_tie_breaks_to_the_cross_section():
    """When both frames score the same, the cross-section wins: it is the frame
    the corpus's ambiguous rows more often mean, and it is the stricter one for
    indicators, so the tie-break cannot hand out the proxy for free."""
    verdict = measure.plan_row(record([cond("asset", "asset", "in", ["BTC"])],
                                      shape="unclear"))
    assert verdict.scope == measure.CROSS


# ---------------------------------------------------------------------------
# Timeframes
# ---------------------------------------------------------------------------

def test_one_interval_is_unruled_not_invented():
    verdict = measure.plan_row(record([cond("timeframe", "interval", "eq", "4h")]))
    assert verdict.unknown_keys == (("bar_interval", "*", "equals"),)
    assert verdict.gap_ids == ()
    assert not verdict.all_expressible_strict
    assert verdict.all_expressible_optimistic


def test_several_intervals_become_a_resonance_request():
    verdict = measure.plan_row(record([
        cond("timeframe", "interval", "eq", "15m"),
        cond("timeframe", "interval", "eq", "1h"),
        cond("timeframe", "interval", "eq", "4h"),
    ]))
    assert verdict.gap_ids == ("GAP-PER-SYMBOL-INDICATOR",)
    assert verdict.blockers == (("timeframe/interval[multi]",
                                "GAP-PER-SYMBOL-INDICATOR"),)


def test_the_same_interval_twice_is_still_one_interval():
    verdict = measure.plan_row(record([
        cond("timeframe", "interval", "eq", "4h"),
        cond("timeframe", "interval", "eq", "4h"),
    ]))
    assert verdict.gap_ids == ()


# ---------------------------------------------------------------------------
# The compound shape
# ---------------------------------------------------------------------------

def test_both_shape_is_refusal_gold_whatever_else_it_holds():
    """`validate_spec` refuses a spec that is both selection and signals, so no
    amount of otherwise-expressible content rescues these."""
    verdict = measure.plan_row(record([
        cond("rank_topn", "universe", "top_n", 5),
        cond("indicator", "rsi"),
    ], shape="both"))
    assert "GAP-COMPOUND-SELECT-THEN-TRADE" in verdict.gap_ids
    assert not verdict.all_expressible_optimistic


def test_entry_plan_in_a_selection_request_is_refused():
    """"five shorts with entry and stop" answers half the request; the YAML
    selection path leaves every candidate's trade slot null."""
    verdict = measure.plan_row(record([cond("risk", "stop_loss", "eq", 2.0)],
                                      shape="selection"))
    assert verdict.gap_ids == ("GAP-ENTRY-EXIT-PER-CANDIDATE",)


def test_entry_plan_in_a_trade_request_is_expressible():
    verdict = measure.plan_row(record([cond("risk", "stop_loss", "eq", 2.0)],
                                      shape="trade"))
    assert verdict.gap_ids == ()
    assert verdict.all_expressible_strict


# ---------------------------------------------------------------------------
# Nothing silently dropped
# ---------------------------------------------------------------------------

def test_unmapped_pair_raises_rather_than_being_skipped():
    with pytest.raises(KeyError, match="no proxy mapping"):
        measure.map_condition(cond("brand_new_family", "whatever"), 0,
                              measure.CROSS)


def test_non_conditions_are_dropped_and_counted():
    row = record([cond("universe_filter", "screen")], tier="C")
    verdict = measure.plan_row(row)
    assert verdict.n_mapped == 0
    bucket = measure._bucket([row], {id(row): verdict})
    assert bucket["no_mappable_condition"]["rows"] == 1
    assert bucket["upper_bound"]["rows"] == 0


def test_risk_control_alone_is_not_refusal_gold():
    """It was, and that was a bug: the bare word for 'risk management' turned a
    clean request into a refusal, which pushes the bound the wrong way."""
    row = record([cond("risk", "risk_control")], tier="C")
    verdict = measure.plan_row(row)
    assert verdict.gap_ids == ()
    assert verdict.n_mapped == 0


def test_every_mapped_condition_reaches_plan_conversion():
    """Ids must be unique or ``plan_conversion`` would collapse two conditions
    into one and the row would look cheaper than it is."""
    conditions = [
        cond("indicator", "rsi"),
        cond("threshold", "rsi", "lt", 30.0),
        cond("risk", "stop_loss", "eq", 2.0),
        cond("timeframe", "interval", "eq", "4h"),
        cond("asset", "asset", "in", ["BTC"]),
    ]
    row = record(conditions, shape="trade")
    mapped = measure.map_conditions(row, measure.SERIES)
    ids = [m.condition.id for m in mapped]
    assert len(ids) == len(set(ids))
    assert len(mapped) == len(conditions)


# ---------------------------------------------------------------------------
# Buckets and the direction of every approximation
# ---------------------------------------------------------------------------

def test_upper_bound_and_refusal_gold_are_mutually_exclusive():
    rows = [
        record([cond("indicator", "rsi")], shape="trade"),
        record([cond("universe_filter", "market_cap")], shape="selection"),
    ]
    verdicts = {id(r): measure.plan_row(r) for r in rows}
    bucket = measure._bucket(rows, verdicts)
    assert bucket["upper_bound"]["rows"] == 1
    assert bucket["refusal_gold"]["rows"] == 1
    assert bucket["total"]["rows"] == 2


def test_strict_is_never_larger_than_the_upper_bound():
    rows = [
        record([cond("timeframe", "interval", "eq", "4h")], tier="C"),
        record([cond("indicator", "rsi")], shape="trade"),
        record([cond("universe_filter", "liquidity")], shape="selection"),
    ]
    verdicts = {id(r): measure.plan_row(r) for r in rows}
    bucket = measure._bucket(rows, verdicts)
    assert bucket["strict"]["rows"] <= bucket["upper_bound"]["rows"]
    assert bucket["upper_bound_convertible"]["rows"] <= bucket["upper_bound"]["rows"]


def test_counts_are_deduplicated_by_split_group():
    """The number that matters for a training set is examples, not rows: 48% of
    this corpus is a repeat and the biggest group is over 1,500 rows."""
    rows = [record([cond("indicator", "rsi")], shape="trade",
                   split_group_key="preset:card", canon_sha256="a" * 64)
            for _ in range(5)]
    verdicts = {id(r): measure.plan_row(r) for r in rows}
    bucket = measure._bucket(rows, verdicts)
    assert bucket["upper_bound"]["rows"] == 5
    assert bucket["upper_bound"]["unique_canon"] == 1
    assert bucket["upper_bound"]["split_groups"] == 1


def test_leverage_is_counted_but_flagged_as_overstating():
    """SIZING_KEYS is {'size'} and no exit key carries leverage, so '10x' cannot
    be written. It stays in the bound (it is a bound) and it must be visible in
    the overstated column, or 4k conditions look expressible with no asterisk."""
    row = record([cond("risk", "leverage", "eq", 10.0)], shape="trade")
    verdict = measure.plan_row(row)
    assert verdict.all_expressible_optimistic
    assert verdict.fidelities["overstates"] == 1
    bucket = measure._bucket([row], {id(row): verdict})
    assert bucket["upper_bound"]["rows"] == 1
    assert bucket["upper_bound_minus_overstated"]["rows"] == 0


def test_trailing_stop_is_expressible_because_the_surface_has_it():
    """Verified against spec.py: VALID_EXIT_TYPES has atr_trailing_stop and
    EXIT_KEYS has trail_mult. The capability row's field list just does not
    enumerate them."""
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import (EXIT_KEYS,
                                                          VALID_EXIT_TYPES)
    assert "atr_trailing_stop" in VALID_EXIT_TYPES
    assert "trail_mult" in EXIT_KEYS
    assert measure.PROXY_MAP[("risk", "trailing_stop")].fidelity == "exact"


def test_leverage_really_is_absent_from_the_spec_surface():
    """Pins the evidence behind the 'overstates' verdict above. If a leverage key
    lands, this fails and the mapping should be upgraded."""
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import (DATA_KEYS, EXIT_KEYS,
                                                           SIZING_KEYS)
    assert not any("leverage" in key
                   for key in set(SIZING_KEYS) | set(EXIT_KEYS) | set(DATA_KEYS))


# ---------------------------------------------------------------------------
# Report shape and the privacy boundary
# ---------------------------------------------------------------------------

def _small_report():
    rows = [
        record([cond("indicator", "rsi"), cond("risk", "stop_loss", "eq", 2.0),
                cond("timeframe", "interval", "eq", "4h")], shape="trade"),
        record([cond("rank_topn", "universe", "top_n", 5),
                cond("universe_filter", "market_cap")], shape="selection",
               tier="B", split_group_key="preset:card", preset_case="card"),
        record([cond("indicator", "supertrend"),
                cond("rank_topn", "universe", "top_n", 30)], shape="both"),
        record([], tier="D", shape="unclear",
               is_continuation_fragment=True, fragment_reason="empty",
               is_candidate=False),
    ]
    return rows, measure.build_measure_report(rows)


def test_report_excludes_continuation_fragments_from_the_scored_population():
    rows, report = _small_report()
    assert report["population"]["total_rows"] == 4
    assert report["population"]["continuation_fragments"] == 1
    assert report["population"]["scored_rows"] == 3


def test_report_has_a_bucket_for_every_tier_and_shape():
    _rows, report = _small_report()
    assert set(report["by_tier"]) == set(measure.TIERS)
    assert set(report["by_shape"]) == set(measure.SHAPES)
    for shape in measure.SHAPES:
        assert set(report["by_shape_tier"][shape]) == set(measure.TIERS)


def test_rendered_markdown_is_pure_ascii():
    """The last line of defence on the privacy rule. The corpus holds user_id and
    verbatim Chinese questions and both remotes are public; if any of it ever
    reached a label in this report, the ascii write is what stops it."""
    rows, report = _small_report()
    funnel = mine.build_report(rows)
    markdown = measure.render_markdown(report, funnel)
    assert markdown.isascii()
    markdown.encode("ascii")


def test_report_json_is_serialisable_and_ascii():
    _rows, report = _small_report()
    payload = json.dumps(report, ensure_ascii=True, sort_keys=True, default=str)
    assert payload.isascii()


def test_gap_ranking_is_sorted_by_dup_weighted_count():
    # The second gap used to be GAP-SPREAD-DEPTH via ('universe_filter',
    # 'liquidity'); that condition is expressible now, so a per-symbol indicator
    # inside a SELECTION request is the second blocked condition here. It is
    # blocked for a different reason than market_cap (no per-candidate bar series
    # rather than no source at all), which is what makes it a distinct gap id.
    rows = [record([cond("universe_filter", "market_cap")], shape="selection")
            for _ in range(3)]
    rows += [record([cond("indicator", "rsi")], shape="selection")]
    for index, row in enumerate(rows):
        row["split_group_key"] = "dup:%d" % index
    verdicts = {id(r): measure.plan_row(r) for r in rows}
    gaps = [g for g in measure.build_gaps(rows, verdicts)
            if g["detectable_by_this_pass"]]
    assert [g["gap_id"] for g in gaps] == ["GAP-MARKET-CAP",
                                           "GAP-PER-SYMBOL-INDICATOR"]
    counts = [g["dup_weighted_count"] for g in gaps]
    assert counts == sorted(counts, reverse=True)


def test_unlocked_if_closed_counts_only_sole_blockers():
    """A row blocked by two gaps is unlocked by neither on its own, and a
    frequency ranking cannot see that."""
    # Second gap swapped from GAP-SPREAD-DEPTH to GAP-PER-SYMBOL-INDICATOR for the
    # reason given in the ranking test above: the book cross-section landed, so a
    # liquidity filter no longer blocks anything.
    single = record([cond("universe_filter", "market_cap")], shape="selection",
                    split_group_key="dup:1")
    double = record([cond("universe_filter", "market_cap"),
                     cond("indicator", "rsi")], shape="selection",
                    split_group_key="dup:2")
    rows = [single, double]
    verdicts = {id(r): measure.plan_row(r) for r in rows}
    gaps = {g["gap_id"]: g for g in measure.build_gaps(rows, verdicts)}
    assert gaps["GAP-MARKET-CAP"]["dup_weighted_count"] == 2
    assert gaps["GAP-MARKET-CAP"]["rows_unlocked_if_closed"] == 1
    assert gaps["GAP-PER-SYMBOL-INDICATOR"]["rows_unlocked_if_closed"] == 0
    assert (gaps["GAP-PER-SYMBOL-INDICATOR"]["co_occurring_gaps"]
            == {"GAP-MARKET-CAP": 1})


# ---------------------------------------------------------------------------
# Whole-pass integration on a slice of the real corpus
# ---------------------------------------------------------------------------

CSV = ("docs/user_demand_analysis/2026-05_07_trading_intent/"
       "trading_intent_chats_2026-05_07_zh_en.csv")


@pytest.fixture(scope="module")
def real_slice(request):
    from pathlib import Path
    path = Path(request.config.rootpath) / CSV
    if not path.exists():
        pytest.skip("corpus CSV not present: %s" % path)
    return measure.remine(path, limit=400)


def test_remine_produces_repo_safe_records(real_slice):
    for row in real_slice:
        mine.assert_repo_safe(row)


def test_remine_records_carry_no_verbatim_text(real_slice):
    """``remine`` must not reintroduce what ``mine`` was careful to keep out."""
    forbidden = {"first_query", "user_text_excerpt", "canon_text", "user_id",
                 "chat_id"}
    for row in real_slice:
        assert not (forbidden & set(row))
        for condition in row["conditions"]:
            assert "quote" not in condition


def test_every_real_condition_maps_without_raising(real_slice):
    for row in real_slice:
        measure.plan_row(row) if not row["is_continuation_fragment"] else None


def test_real_slice_renders(real_slice):
    report = measure.build_measure_report(real_slice)
    markdown = measure.render_markdown(report, mine.build_report(real_slice))
    assert markdown.isascii()
    assert "UPPER BOUND" in markdown


def test_verify_against_mine_rejects_a_different_population(tmp_path, real_slice):
    """The cross-check has to actually fail on a mismatch, or the full run's
    'identical' line means nothing."""
    funnel = mine.build_report(real_slice)
    path = tmp_path / "funnel.json"
    path.write_text(json.dumps(funnel, ensure_ascii=True, sort_keys=True),
                    encoding="ascii")
    measure.verify_against_mine(real_slice, path)          # matches
    with pytest.raises(AssertionError, match="does not reproduce"):
        measure.verify_against_mine(real_slice[:-10], path)


def test_verify_against_mine_raises_when_the_funnel_is_missing(tmp_path,
                                                              real_slice):
    with pytest.raises(FileNotFoundError, match="run tools.nl2yaml.mine first"):
        measure.verify_against_mine(real_slice, tmp_path / "nope.json")
