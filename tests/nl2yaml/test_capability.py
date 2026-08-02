"""The capability table's own invariants, and the anti-proxy mechanism it powers.

The tests worth reading first are
:func:`test_the_supertrend_accident_cannot_be_written_when_the_proxy_is_withheld`
and :func:`test_opening_the_proxy_is_per_condition_and_never_per_subject`. The
rest guard the table's shape so that those two keep meaning what they say.
"""

from __future__ import annotations

import pytest

from tools.nl2yaml import capability as cap


# --------------------------------------------------------------------------- #
# shape                                                                       #
# --------------------------------------------------------------------------- #


def test_the_table_covers_the_subjects_the_corpus_actually_asks_about():
    """Coverage is the point: an unlisted subject shelves the case for a human.

    So a thin table does not fail loudly, it quietly parks work. The named
    subjects are the ones the selection corpus keeps producing.
    """
    subjects = set(cap.subjects())
    assert len(subjects) >= 18, sorted(subjects)
    required = {
        "quote_volume_24h", "price_change_24h", "funding_rate", "social_mentions",
        "social_sentiment", "open_interest", "long_short_ratio", "market_cap",
        "onchain_holder_concentration", "sector_label", "contract_type",
        "technical_indicator", "multi_timeframe", "historical_range",
        "spread_liquidity", "entry_exit_plan", "alert_notify", "account_ops",
    }
    assert required <= subjects, sorted(required - subjects)


def test_every_row_carries_either_a_vocabulary_or_a_gap_id():
    """The acceptance condition, stated as an assertion over the whole table.

    A row with neither is the dangerous shape: it tells the converter nothing it
    may use AND gives the pipeline no closed label to record, so the case ends up
    with an unlabelled partial answer.
    """
    for row in cap.CAPABILITY_TABLE:
        if row.verdict == cap.EXPRESSIBLE:
            assert row.block_refs or row.fields, row.key
            assert row.gap_id is None, row.key
        elif row.verdict == cap.PROXY_ONLY:
            assert row.proxy_block_refs and row.degradation, row.key
            assert row.gap_id in cap.GAP_IDS, row.key
        else:
            assert row.gap_id in cap.GAP_IDS, row.key


def test_gap_ids_are_a_closed_vocabulary_so_a_refusal_needs_no_llm_judge():
    used = {row.gap_id for row in cap.CAPABILITY_TABLE if row.gap_id}
    assert used <= cap.GAP_IDS
    # Vacated ids stay in the set so historical case labels remain comparable.
    assert {"GAP-CONTRACT-META", "GAP-SECTOR-LABEL"} <= cap.GAP_IDS


def test_no_row_can_pre_authorise_its_own_proxy():
    """``allow_proxy`` is a per-case human decision, not a table property."""
    assert all(row.allow_proxy is False for row in cap.CAPABILITY_TABLE)
    with pytest.raises(ValueError, match="allow_proxy=True in the table"):
        cap.Capability(subject="x", scope="*", operator="*",
                       verdict=cap.NOT_EXPRESSIBLE, gap_id="GAP-MARKET-CAP",
                       allow_proxy=True, why="probe")


@pytest.mark.parametrize("kwargs, match", [
    ({"verdict": cap.EXPRESSIBLE}, "empty vocabulary"),
    ({"verdict": cap.NOT_EXPRESSIBLE}, "needs a gap_id"),
    ({"verdict": cap.NOT_EXPRESSIBLE, "gap_id": "GAP-INVENTED"}, "needs a gap_id"),
    ({"verdict": cap.PROXY_ONLY, "gap_id": "GAP-MARKET-CAP"}, "no proxy_block_refs"),
    ({"verdict": cap.PROXY_ONLY, "gap_id": "GAP-MARKET-CAP",
      "proxy_block_refs": ("universe.top_losers",)}, "must state what the proxy loses"),
    ({"verdict": cap.EXPRESSIBLE, "fields": ("x",), "why": "  "}, "must say why"),
])
def test_a_malformed_row_raises_instead_of_degrading(kwargs, match):
    """The repo's rule, applied to its own table: raise, never accept quietly.

    An expressible row with an empty payload is the specific shape that grants
    permission while naming nothing — which is how a model ends up inventing the
    name itself.
    """
    kwargs.setdefault("why", "probe")
    with pytest.raises(ValueError, match=match):
        cap.Capability(subject="probe", scope="cross_section", operator="compare",
                       **kwargs)


def test_lookup_prefers_the_specific_row_and_falls_back_to_the_wildcard():
    # The derivatives fan-out (2026-08-02) made both of these expressible at
    # cross_section scope; every other scope still falls through to the wildcard
    # and gets a NAMED gap rather than "undecidable", because a subject we
    # understand should never be sent to human triage.
    assert cap.lookup("open_interest", "cross_section", "rank").verdict == cap.EXPRESSIBLE
    assert cap.lookup("open_interest", "per_symbol_series", "compare").verdict == cap.EXPRESSIBLE
    assert cap.lookup("long_short_ratio", "cross_section", "rank").verdict == cap.EXPRESSIBLE
    for scope in ("per_symbol_series", "account", "side_channel"):
        row = cap.lookup("long_short_ratio", scope, "rank")
        assert row.verdict == cap.NOT_EXPRESSIBLE, scope
        assert row.gap_id == "GAP-LONG-SHORT-RATIO", scope


def test_an_unlisted_key_is_unknown_and_not_a_guess_in_either_direction():
    row = cap.lookup("gamma_exposure", "cross_section", "rank")
    assert row.verdict == cap.UNKNOWN
    assert row.granted == ()
    assert row.gap_id is None


def test_the_table_is_still_grounded_in_the_live_blocks_package():
    """Fails loudly when ``blocks.universe`` moves under the table.

    Both directions are real. A vanished ref puts a name in the prompt that
    cannot be written into a working spec. A NEW universe block that no row
    mentions is worse: a capability that landed while a row still says
    ``not_expressible`` makes the pipeline refuse requests it can now serve, and
    a refusal is never retried, so those cases are lost in silence. This test is
    the only thing that notices.
    """
    cap.assert_table_is_current()


def test_every_source_a_row_demands_is_one_the_bundle_can_carry():
    """``with: [funding]`` is not decoration.

    The blocks that take a source fetch it themselves when it is omitted, which
    turns validation of a frontend-supplied spec into outbound REST traffic. A
    row that grants such a block without naming its source would hand the
    converter a network call.
    """
    from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import FETCHES_WITHOUT_SOURCE

    for row in cap.CAPABILITY_TABLE:
        for ref in row.block_refs:
            needed = FETCHES_WITHOUT_SOURCE.get(ref)
            if needed:
                assert needed in row.requires_sources, (row.key, ref, needed)


def test_no_row_grants_a_block_the_interpreter_refuses():
    """The table and ``DENIED_FUNCTION_NAMES`` must not disagree.

    ``universe.fetch_perpetual_universe`` is the example that matters: it is a
    live REST call, it was reachable from a spec, and validating a
    frontend-supplied spec therefore fired outbound requests.
    """
    from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import DENIED_FUNCTION_NAMES

    for row in cap.CAPABILITY_TABLE:
        for ref in tuple(row.block_refs) + tuple(row.proxy_block_refs):
            assert ref.split(".")[-1] not in DENIED_FUNCTION_NAMES, (row.key, ref)


# --------------------------------------------------------------------------- #
# the mechanism                                                               #
# --------------------------------------------------------------------------- #


#: The originating request, as conditions. Structured only — the corpus row this
#: came from holds a user_id and a verbatim question, and neither may enter this
#: repo, so what travels is the triple and nothing else.
USER_CHAT_CONDITIONS = (
    {"id": "c1", "subject": "quote_volume_24h", "scope": "cross_section",
     "operator": "compare", "value": {"op": ">", "threshold": 2_000_000},
     "quantified": True},
    {"id": "c2", "subject": "symbol_blacklist", "scope": "cross_section",
     "operator": "exclude", "value": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
     "polarity": "exclude", "quantified": True},
    {"id": "c3", "subject": "quote_currency", "scope": "cross_section",
     "operator": "exclude", "value": ["USDC"], "polarity": "exclude",
     "quantified": True},
    {"id": "c4", "subject": "sector_label", "scope": "cross_section",
     "operator": "exclude", "value": ["TradFi"], "polarity": "exclude",
     "quantified": True},
    {"id": "c5", "subject": "long_short_ratio", "scope": "cross_section",
     "operator": "compare", "value": {"op": ">", "threshold": 60.0},
     "quantified": True},
    {"id": "c6", "subject": "technical_indicator", "scope": "cross_section",
     "operator": "compare", "value": {"op": "<", "threshold": 0},
     "quantified": True},
    {"id": "c7", "subject": "multi_timeframe", "scope": "cross_section",
     "operator": "resonance", "value": ["4h", "1h", "15m"], "quantified": False},
    {"id": "c8", "subject": "direction", "scope": "cross_section",
     "operator": "require", "value": "short", "quantified": True},
    {"id": "c9", "subject": "basket_size", "scope": "cross_section",
     "operator": "top_k", "value": 5, "quantified": True},
)


def test_the_supertrend_accident_cannot_be_written_when_the_proxy_is_withheld():
    """The whole design, in one assertion.

    ``universe.top_losers`` was the block that stood in for "Supertrend(10,3)
    bearish on H4/H1/M15". The spec validated, the run succeeded, and the output
    admitted nothing. With the proxy withheld the name is simply not in the
    converter's vocabulary for that condition, and the condition is recorded
    unconvertible under a closed gap id instead of being answered wrongly.
    """
    plan = cap.plan_conversion(USER_CHAT_CONDITIONS)

    assert "universe.top_losers" not in plan.vocabulary
    assert "universe.top_losers" in plan.refused_vocabulary

    unconvertible = {cond.id: gap for cond, gap in plan.unconvertible}
    assert unconvertible["c6"] == "GAP-PER-SYMBOL-INDICATOR"
    assert unconvertible["c7"] == "GAP-PER-SYMBOL-INDICATOR"
    # And the conditions that cannot be converted are ABSENT from what the
    # converter sees, not annotated inside it: a "cannot do this" note left in
    # the prompt is a condition the model will try to satisfy anyway.
    # c5 (retail long/short ratio) joined this set when the cross-sectional
    # snapshot landed. c6/c7 have not: those need a per-candidate indicator, and
    # the point of the assertion is that they stay OUT of the vocabulary rather
    # than being answered by whatever is nearest.
    assert {cond.id for cond in plan.converter_conditions} == {
        "c1", "c2", "c3", "c4", "c5", "c8", "c9"}


def test_the_gaps_the_same_request_still_hits_are_named_not_approximated():
    plan = cap.plan_conversion(USER_CHAT_CONDITIONS)

    # Down to one. GAP-LONG-SHORT-RATIO left this list when the cross-sectional
    # snapshot landed, the same way GAP-SECTOR-LABEL left it when the
    # contract-metadata blocks did. What remains is the per-candidate indicator,
    # which is the request's real blocker.
    assert plan.gap_ids == ("GAP-PER-SYMBOL-INDICATOR",)
    assert cap.lookup("long_short_ratio", "cross_section", "*").verdict == cap.EXPRESSIBLE
    assert "universe.filter_long_short_ratio" in plan.vocabulary
    assert "long_short_ratio_snapshot" in plan.required_sources
    # The TradFi exclusion is no longer one of them: the contract-metadata blocks
    # landed, so the condition that ruined the original run is now expressible.
    assert cap.lookup("sector_label", "cross_section", "exclude").verdict == cap.EXPRESSIBLE
    assert "universe.filter_sub_type" in plan.vocabulary
    assert "contract_meta" in plan.required_sources


def test_opening_the_proxy_is_per_condition_and_never_per_subject():
    """A reviewer accepting one degradation must not open it everywhere.

    Ids, not subjects, precisely so that "yes, 24h change is close enough THIS
    time" cannot become a standing permission.
    """
    plan = cap.plan_conversion(USER_CHAT_CONDITIONS, allow_proxy_for=["c6"])

    assert "universe.top_losers" in plan.vocabulary
    assert {cond.id for cond in plan.proxied} == {"c6"}
    # c7 asked for the same gap and was not opened, so it is still refused.
    assert "c7" in {cond.id for cond, _gap in plan.unconvertible}
    # Still reported as a proxy, never promoted to expressible: the basket's
    # reader has to keep being told the answer is a stand-in.
    assert {cond.id for cond in plan.expressible}.isdisjoint({"c6"})


def test_a_typo_in_allow_proxy_for_raises_instead_of_silently_refusing():
    """Otherwise the reviewer's decision is lost with no trace of the loss."""
    with pytest.raises(ValueError, match="not in this case"):
        cap.plan_conversion(USER_CHAT_CONDITIONS, allow_proxy_for=["c66"])


def test_a_granted_name_beats_a_withheld_one_within_the_same_case():
    """``top_losers`` is honest for "biggest fallers" and a proxy for Supertrend.

    Both conditions in one request must not produce a prompt that grants and
    refuses the same name — the caller would have contradictory instructions.
    Grant wins, and the indicator condition stays unconvertible, which is the
    honest reading: the block is there for the condition that asked for it.
    """
    plan = cap.plan_conversion([
        {"id": "a", "subject": "price_change_24h", "scope": "cross_section",
         "operator": "top_k", "value": 30, "quantified": True},
        {"id": "b", "subject": "technical_indicator", "scope": "cross_section",
         "operator": "compare", "value": {"op": "<", "threshold": 0}},
    ])

    assert "universe.top_losers" in plan.vocabulary
    assert "universe.top_losers" not in plan.refused_vocabulary
    assert [gap for _cond, gap in plan.unconvertible] == ["GAP-PER-SYMBOL-INDICATOR"]


def test_an_unknown_subject_blocks_the_case_rather_than_generating_something():
    plan = cap.plan_conversion([
        {"id": "z", "subject": "moon_phase", "operator": "compare", "value": 3},
    ])

    assert [cond.id for cond in plan.shelved] == ["z"]
    assert plan.blocked is True
    assert plan.vocabulary == frozenset()


# --------------------------------------------------------------------------- #
# privacy at the boundary                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ["quote", "text", "raw", "user_id", "utterance"])
def test_a_condition_carrying_user_text_is_refused_at_the_boundary(key):
    """Both remotes are public. One bad ``.gitignore`` edit is the whole leak.

    So the ban lives in code at the point conditions are accepted, not in a
    reviewer's memory of which directory is ignored today.
    """
    with pytest.raises(ValueError, match="stay outside"):
        cap.normalize_condition({"subject": "quote_volume_24h",
                                 "operator": "compare", "value": 1, key: "..."})


def test_a_misspelt_condition_key_raises_rather_than_being_ignored():
    """``quantifed`` would silently switch off that condition's G1e predicate."""
    with pytest.raises(ValueError, match="unknown condition key"):
        cap.normalize_condition({"subject": "quote_volume_24h",
                                 "operator": "compare", "quantifed": True})


def test_with_proxy_opened_shows_a_reviewer_the_cost_it_is_accepting():
    row = cap.lookup("technical_indicator", "cross_section", "compare")
    opened = cap.with_proxy_opened(row)

    assert opened.verdict == cap.EXPRESSIBLE
    assert "universe.top_losers" in opened.block_refs
    assert "PROXY OPENED" in opened.why and "top_losers(n=30)" in opened.why
    with pytest.raises(ValueError, match="not a proxy row"):
        cap.with_proxy_opened(cap.lookup("market_cap"))
