"""The hand-written capability table: which user conditions this repo can express.

Why a table and not an instruction
----------------------------------
The accident this module exists to prevent, verbatim: a request for
"Supertrend(10,3) bearish on H4/H1/M15 simultaneously" came back as a spec whose
trend filter was ``universe.top_losers(n=30)``. The spec validated. It ran. It
emitted five candidates. Nothing in the output admitted that the multi-timeframe
condition had been replaced by "biggest 24h fallers", and no downstream gate
could tell, because a proxied spec is internally consistent — it is only wrong
relative to a request the spec no longer contains.

Telling a model "do not substitute a similar field" does not survive contact with
a model that wants its YAML to run. So the table does not advise; it **decides
what the converter is allowed to name**:

===================  ===============================================================
``expressible``      the block refs / column names enter the converter's vocabulary
``proxy_only``      the proxy block is withheld unless a human opens it per case
``not_expressible``  the condition is REMOVED from the converter's input, and the
                     case is recorded as partially unconvertible under a gap id
``unknown``          the case is shelved for a human; nothing is generated
===================  ===============================================================

``universe.top_losers`` is still a legitimate block — it is the honest answer to
"biggest 24h fallers". What :func:`plan_conversion` withholds is its *name*, for
the cases where it would only ever be a stand-in. A prompt that never contains
the string ``top_losers`` cannot emit it.

This is the same principle as
``yaml_pipeline.interpreter.DENIED_FUNCTION_NAMES``: refuse at the layer that
parses, not at the layer that instructs.

Provenance
----------
Every row is grounded in a probe against the real validator, not read off a
docstring — see :data:`TABLE_DERIVED_FROM`. :func:`assert_table_is_current`
re-checks the grounding at test time, because ``cyqnt_trd.blocks`` is under
active development and a capability that lands while a row still says
``not_expressible`` is a silent downgrade: the pipeline would keep refusing a
request it can now serve.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Capability",
    "Condition",
    "ConversionPlan",
    "CAPABILITY_TABLE",
    "EXPRESSIBLE",
    "GAP_IDS",
    "NOT_EXPRESSIBLE",
    "OPERATORS",
    "PROXY_ONLY",
    "SCOPES",
    "TABLE_DERIVED_FROM",
    "UNKNOWN",
    "VERDICTS",
    "assert_table_is_current",
    "lookup",
    "normalize_condition",
    "plan_conversion",
    "subjects",
    "table_as_rows",
    "with_proxy_opened",
]


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

EXPRESSIBLE = "expressible"
PROXY_ONLY = "proxy_only"
NOT_EXPRESSIBLE = "not_expressible"
UNKNOWN = "unknown"

VERDICTS = frozenset({EXPRESSIBLE, PROXY_ONLY, NOT_EXPRESSIBLE, UNKNOWN})

#: Which frame a condition has to be evaluated on. This is the axis that decides
#: nearly every verdict in the table, because the repo has exactly two shapes and
#: they are not interchangeable:
#:
#: ``cross_section``
#:     one row per instrument, one instant — what ``selection:`` evaluates.
#:     There are no bars here, so no time-series indicator can be computed.
#: ``per_symbol_series``
#:     one row per bar for the ONE symbol named in ``data.symbol`` — what
#:     ``signals:`` evaluates.
#: ``per_candidate_series``
#:     a bar series for each of many candidates. This is the shape "scan the
#:     market, then run an indicator on each survivor" needs, and the repo has
#:     no such object at all — which is why it is its own scope rather than a
#:     variation of the two above.
#: ``account``
#:     positions / balance / leverage. Present in the runtime, absent from YAML.
#: ``side_channel``
#:     anything whose output is not a signal (a push notification, a webhook).
SCOPES = frozenset({
    "cross_section", "per_symbol_series", "per_candidate_series",
    "account", "side_channel", "*",
})

#: What the user is doing to the subject. Closed so that a case's conditions can
#: be keyed on it; ``*`` is the wildcard row used when the verdict does not
#: depend on the operator (nothing in this repo can rank on market cap, filter on
#: market cap, or exclude by market cap).
OPERATORS = frozenset({
    "compare",     # numeric threshold: >, <, between
    "rank",        # order the universe by this column
    "top_k",       # keep the first K
    "exclude",     # drop named members
    "require",     # keep only named members
    "equals",      # categorical == / tag membership
    "resonance",   # the same condition true on several timeframes at once
    "window",      # over a historical lookback (N-day high, range since X)
    "plan",        # entry / stop / target attached to the instrument
    "notify",      # tell me when, rather than decide what
    "execute",     # place the order
    "*",
})

#: Closed gap vocabulary. Closed so a refusal can be compared by string equality
#: instead of by asking a second model whether two refusals mean the same thing:
#: the converter's own output names one of these ids, and the expected id is
#: known before generation, so grading a refusal costs nothing and cannot drift.
#:
#: Ids are never deleted, only vacated. ``GAP-CONTRACT-META`` and
#: ``GAP-SECTOR-LABEL`` are vacant as of :data:`TABLE_DERIVED_FROM` (the contract
#: metadata blocks landed) and stay in this set so that cases labelled with them
#: before that remain comparable.
GAP_IDS = frozenset({
    "GAP-PER-SYMBOL-INDICATOR",
    "GAP-CONTRACT-META",
    "GAP-SECTOR-LABEL",
    "GAP-LONG-SHORT-RATIO",
    "GAP-OI-CROSS-SECTION",
    "GAP-LIQUIDATION-CROSS-SECTION",
    "GAP-MARKET-CAP",
    "GAP-ONCHAIN-CONCENTRATION",
    "GAP-HISTORICAL-WINDOW",
    "GAP-COMPOUND-SELECT-THEN-TRADE",
    "GAP-ENTRY-EXIT-PER-CANDIDATE",
    "GAP-VAGUE-CRITERION",
    "GAP-SPREAD-DEPTH",
    "GAP-NEWS-EVENT-TEXT",
    "GAP-ALERT-NOTIFY",
    "GAP-ACCOUNT-OPS",
})

#: What this table was written against, so a reader can tell whether it is stale
#: without re-deriving it. ``blocks`` and ``docs`` are being changed by other work
#: in parallel, so the git commit alone is not enough — the working tree is what
#: the probes ran against, and :func:`assert_table_is_current` re-runs the part of
#: that grounding a test can afford.
TABLE_DERIVED_FROM: Dict[str, Any] = {
    "commit": "2d2100620dd1e46d65aeba4ec8efd17bf9b79e54",
    "working_tree": (
        "dirty — cyqnt_trd/blocks/universe.py, yaml_pipeline/{interpreter,spec}.py "
        "and docs/strategy_yaml_spec/* carried uncommitted changes from parallel "
        "work when this table was derived (2026-08-02). The contract-metadata "
        "blocks below exist only in that working tree. REVISED later the same day "
        "after the derivatives fan-out landed: open_interest / "
        "open_interest_change / long_short_ratio flipped from not_expressible to "
        "expressible at cross_section scope, each re-probed to errors == []. "
        "All three are NARROWED fan-outs, so their augment step must follow the "
        "narrowing steps; probing one before filter_crypto_only reproduces the "
        "coverage guard at 87%. REVISED again the same day after the liquidity + "
        "funding-APR work landed: spread_liquidity flipped from not_expressible to "
        "expressible at cross_section scope for BOTH compare and rank (the "
        "`book_ticker` node and universe.augment_with_spread now exist; re-probed "
        "four ways to errors == []), and the funding rank row moved from "
        "fundingRatePct to fundingRateApr with funding_info as a second required "
        "source. Unlike the fan-outs, book_ticker is whole-market in ONE request, "
        "so its augment step carries no ordering requirement. TOUCHED a third time "
        "the same day for `universe_blocks` ONLY, when the per-candidate kline "
        "fan-out landed — the verdict rows it affects are NOT updated; see "
        "``pending_review``."
    ),
    #: ★ FOUR ROWS IN THIS TABLE ARE KNOWN STALE AND ARE STILL SAYING NO.
    #:
    #: ``universe.augment_with_indicator`` and the ``universe_bars`` catalog node
    #: landed on 2026-08-02. They make these four rows wrong, in the direction this
    #: table calls the worse one — refusing a request the repo can now serve, where
    #: refusals are not retried and the case is lost silently:
    #:
    #: ================================================  ===============  ==========
    #: row                                               says             should be
    #: ================================================  ===============  ==========
    #: technical_indicator / cross_section / compare      proxy_only       expressible
    #: technical_indicator / per_candidate_series / *     not_expressible  expressible
    #: multi_timeframe / cross_section / resonance        not_expressible  expressible
    #: historical_range / cross_section / window          not_expressible  expressible
    #: ================================================  ===============  ==========
    #:
    #: The vocabulary each should grant: ``universe.augment_with_indicator`` with
    #: ``requires_sources=("universe_bars",)``, fields ``indicator / timeframe / agg
    #: / window_bars / output / column / input / min_bars_multiple / as``, plus
    #: ``entry.all_of`` and ``conditions.value_below`` for the resonance row and
    #: ``indicators.range_gain_pct`` for the history row. ``universe.top_losers``
    #: should NOT survive as a fallback: the honest spelling exists now, and a proxy
    #: left in the vocabulary is a proxy that gets picked again.
    #:
    #: WHY IT IS NOT DONE HERE. The proxy_only verdict on the first row is the
    #: worked example eleven tests in ``tests/nl2yaml/`` are built on — the
    #: withheld-vocabulary mechanism, the G1e silent-proxy gate, the gap ranking and
    #: the corpus measurement all use it as their flagship case. Flipping it turns
    #: those red, and choosing their replacement example is a decision about that
    #: dataset, not about the blocks package. So the capability is real, the
    #: evidence is below, and the flip is left to the owner of this table.
    #:
    #: EVIDENCE, so the review does not have to re-derive it:
    #: ``docs/strategy_yaml_spec/example_multi_timeframe_supertrend.yaml`` and
    #: ``example_three_month_runup_screen.yaml`` both validate to errors == [] and
    #: both run end to end against
    #: ``tests/standard_bot/fixtures/universe_derivatives.json``. The history screen
    #: reproduced a hand-written Python answer exactly (+1300 % / +594 % / +589 % /
    #: +330 % / +126 % over a 3-month daily window, 5 instruments). On that same
    #: capture 0 of 5 candidates were Supertrend-bearish on 4h AND 1h AND 15m while
    #: 3 of 5 were bearish on at least one, so ``all_of`` and ``any_of`` are
    #: genuinely different answers and a 24h loser list is neither.
    #: ``tests/standard_bot/test_universe_indicator.py`` pins all of it.
    "pending_review": (
        "technical_indicator/cross_section/compare, "
        "technical_indicator/per_candidate_series/*, "
        "multi_timeframe/cross_section/resonance, "
        "historical_range/cross_section/window"
    ),
    "probed_on": "2026-08-02",
    "how": (
        "every 'expressible' row for the per_symbol_series scope was proved by "
        "building a minimal spec and running yaml_pipeline.spec.validate_spec on "
        "it until errors == []; every 'not_expressible' row for a column-backed "
        "subject was proved by the same probe FAILING with 'cannot resolve "
        "reference <column>' against the synthetic dry-run frame, which is the "
        "frame validate_spec offers a spec author"
    ),
    #: The ``blocks.universe`` surface at derivation time. A name appearing here
    #: that no longer resolves, or a name resolving that is not here, both mean
    #: the table needs review — see :func:`assert_table_is_current`.
    "universe_blocks": (
        "UniverseFilter", "augment_with_contract_meta", "augment_with_funding",
        "augment_with_indicator",
        "augment_with_long_short_ratio", "augment_with_news",
        "augment_with_oi_change", "augment_with_open_interest",
        "augment_with_spread",
        "exclude_symbols", "fetch_perpetual_universe", "filter_change_pct",
        "filter_crypto_only", "filter_funding_rate", "filter_long_short_ratio",
        "filter_oi_change", "filter_open_interest", "filter_quote_suffix",
        "filter_quote_volume", "filter_sentiment", "filter_spread",
        "filter_sub_type", "filter_top_of_book",
        "filter_underlying_type", "only_symbols", "top_bullish", "top_gainers",
        "top_losers", "top_mentioned",
    ),
}


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capability:
    """One ``(subject, scope, operator) -> verdict`` ruling.

    The payload fields are not decoration: ``block_refs`` and ``fields`` ARE the
    converter's vocabulary for this condition (see :func:`plan_conversion`), and
    ``gap_id`` IS the refusal the converter is graded against. An
    ``expressible`` row with an empty payload would therefore grant a condition
    permission to be converted while naming nothing it may use, so the
    invariants below raise rather than let that row exist.
    """

    subject: str
    scope: str
    operator: str
    verdict: str
    #: ``"<module>.<fn>"`` refs the converter may name for this condition.
    block_refs: Tuple[str, ...] = ()
    #: frame columns / spec keys the converter may name for this condition.
    fields: Tuple[str, ...] = ()
    #: bundle sources a step needs via ``with: [...]``; withheld with the blocks.
    requires_sources: Tuple[str, ...] = ()
    gap_id: Optional[str] = None
    #: the block a model reaches for when it wants the YAML to run anyway.
    proxy_block_refs: Tuple[str, ...] = ()
    #: what is lost by accepting the proxy. Mandatory on a proxy row: a proxy
    #: whose cost is not written down gets accepted by whoever is in a hurry.
    degradation: Optional[str] = None
    #: False by default and NOT settable from the table — see the class docstring
    #: of :class:`ConversionPlan`. Opening a proxy is a per-case human act, so it
    #: is an argument to :func:`plan_conversion`, never a property of a row.
    allow_proxy: bool = False
    why: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(
                "verdict must be one of %s, got %r for %s"
                % (sorted(VERDICTS), self.verdict, self.key))
        if self.scope not in SCOPES:
            raise ValueError(
                "scope must be one of %s, got %r for %s"
                % (sorted(SCOPES), self.scope, self.key))
        if self.operator not in OPERATORS:
            raise ValueError(
                "operator must be one of %s, got %r for %s"
                % (sorted(OPERATORS), self.operator, self.key))
        if self.allow_proxy:
            raise ValueError(
                "%s sets allow_proxy=True in the table. A proxy is opened for one "
                "case by a human, not for every case by a row — pass the "
                "condition id to plan_conversion(allow_proxy_for=...) instead."
                % (self.key,))

        if self.verdict == EXPRESSIBLE:
            if not (self.block_refs or self.fields):
                raise ValueError(
                    "%s is expressible but names no block_refs and no fields; "
                    "that grants the converter permission with an empty "
                    "vocabulary, which is how a proxy gets invented" % (self.key,))
            if self.gap_id or self.proxy_block_refs or self.degradation:
                raise ValueError(
                    "%s is expressible and must not carry a gap_id, a proxy or a "
                    "degradation note" % (self.key,))
        elif self.verdict == PROXY_ONLY:
            if not self.proxy_block_refs:
                raise ValueError(
                    "%s is proxy_only but names no proxy_block_refs; there is "
                    "nothing to withhold" % (self.key,))
            if not self.degradation:
                raise ValueError(
                    "%s is proxy_only and must state what the proxy loses; an "
                    "unstated cost is what makes a proxy look free" % (self.key,))
            if self.gap_id not in GAP_IDS:
                raise ValueError(
                    "%s is proxy_only and must name the gap the proxy stands in "
                    "for, from the closed set; got %r" % (self.key, self.gap_id))
            if self.block_refs:
                raise ValueError(
                    "%s is proxy_only; put the stand-in under proxy_block_refs so "
                    "it is withheld by default, not under block_refs where it "
                    "would be granted" % (self.key,))
        elif self.verdict == NOT_EXPRESSIBLE:
            if self.gap_id not in GAP_IDS:
                raise ValueError(
                    "%s is not_expressible and needs a gap_id from the closed set "
                    "%s, got %r — an open-vocabulary refusal cannot be graded by "
                    "string equality" % (self.key, sorted(GAP_IDS), self.gap_id))
            if self.block_refs or self.fields:
                raise ValueError(
                    "%s is not_expressible but names a vocabulary; the point of "
                    "the verdict is that the condition leaves the converter's "
                    "input entirely" % (self.key,))
        else:                                            # UNKNOWN
            if self.block_refs or self.fields or self.gap_id or self.proxy_block_refs:
                raise ValueError(
                    "%s is unknown and must carry no payload: the case is shelved "
                    "for a human, who decides what the row becomes" % (self.key,))
        if not self.why.strip():
            raise ValueError("%s must say why; a table nobody can audit is an "
                             "instruction with extra steps" % (self.key,))

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.subject, self.scope, self.operator)

    @property
    def granted(self) -> Tuple[str, ...]:
        """The names this row lets the converter use, blocks and fields together."""
        return tuple(self.block_refs) + tuple(self.fields)


def _row(subject: str, scope: str, operator: str, verdict: str, **kwargs: Any) -> Capability:
    return Capability(subject=subject, scope=scope, operator=operator,
                      verdict=verdict, **kwargs)


# The frame columns a cross-sectional condition can name. Grouped here because
# several rows share them and because the camelCase is not a style choice: the
# universe frame is Binance's 24h ticker response, so ``quote_volume`` does not
# resolve and ``quoteVolume`` does.
_TICKER_FIELDS = ("quoteVolume", "priceChangePercent", "instrument_id")

CAPABILITY_TABLE: Tuple[Capability, ...] = (

    # -- turnover -----------------------------------------------------------
    _row("quote_volume_24h", "cross_section", "compare", EXPRESSIBLE,
         block_refs=("universe.filter_quote_volume",),
         fields=("quoteVolume",),
         why="the column rides along with the 24h ticker that IS the universe "
             "frame, so no augment step is needed"),
    _row("quote_volume_24h", "cross_section", "rank", EXPRESSIBLE,
         fields=("quoteVolume", "selection.score", "selection.order"),
         why="score: quoteVolume. No block: ranking is a spec key, not a step"),
    _row("quote_volume_24h", "per_symbol_series", "compare", EXPRESSIBLE,
         fields=("quote_volume", "volume"),
         block_refs=("indicators.volume_zscore", "indicators.volume_ma",
                     "conditions.volume_surge", "conditions.volume_shrink"),
         why="OHLCV carries quote_volume on every bar; snake_case here and "
             "camelCase in the cross-section, which is a real trap"),

    # -- 24h change ---------------------------------------------------------
    _row("price_change_24h", "cross_section", "compare", EXPRESSIBLE,
         block_refs=("universe.filter_change_pct", "conditions.value_above",
                     "conditions.value_below", "conditions.value_in_range"),
         fields=("priceChangePercent",),
         why="same frame as turnover; reconcile_intent additionally insists the "
             "column actually drives ranking, direction or narrowing"),
    _row("price_change_24h", "cross_section", "rank", EXPRESSIBLE,
         fields=("priceChangePercent", "selection.score", "selection.order"),
         why="score: priceChangePercent + order: asc states 'biggest fallers' "
             "directly, without the top_losers detour"),
    _row("price_change_24h", "cross_section", "top_k", EXPRESSIBLE,
         block_refs=("universe.top_gainers", "universe.top_losers"),
         fields=("priceChangePercent",),
         why="these two narrow the universe to the N extremes of the column; the "
             "honest use of top_losers, as opposed to the proxy use below"),

    # -- funding ------------------------------------------------------------
    _row("funding_rate", "cross_section", "compare", EXPRESSIBLE,
         block_refs=("universe.augment_with_funding",
                     "universe.filter_funding_rate"),
         fields=("fundingRatePct",),
         requires_sources=("funding",),
         why="premiumIndex answers for the whole market in one call, so a real "
             "cross-section exists; `with: [funding]` is mandatory — the block "
             "fetches when the source is omitted and validate would hit the network"),
    _row("funding_rate", "cross_section", "rank", EXPRESSIBLE,
         block_refs=("universe.augment_with_funding",),
         fields=("fundingRateApr", "fundingIntervalHours", "fundingRatePct",
                 "selection.score", "selection.order", "selection.min_score",
                 "selection.max_score"),
         requires_sources=("funding", "funding_info"),
         why="'the five most negative' is score: fundingRateApr + order: asc. The "
             "ANNUALISED column, not fundingRatePct: the venue settles 443 of 743 "
             "perpetuals 4-hourly, 296 8-hourly and 4 hourly, so the raw rate's "
             "unit differs per row and ranking it puts 0.01%@1h (87.6%/yr) level "
             "with 0.01%@8h (10.95%/yr). Hence funding_info as a second source — "
             "without it the column is NaN for every instrument and "
             "spec._refuse_column_without_its_source rejects the spec rather than "
             "returning an empty basket. Annualising multiplies by a POSITIVE "
             "number, so order keeps its meaning. min_score/max_score are absolute "
             "and do not flip with order"),
    _row("funding_rate", "per_symbol_series", "compare", EXPRESSIBLE,
         block_refs=("derivatives.funding_rate_state",),
         fields=("data.derivatives", "funding_rate", "funding_rate_bps"),
         why="probed: data.derivatives + derivatives.funding_rate_state "
             "validates clean. The block wants the RAW ratio and converts to bps "
             "itself; feeding it bps raises"),

    # -- social / news ------------------------------------------------------
    _row("social_mentions", "cross_section", "rank", EXPRESSIBLE,
         block_refs=("universe.augment_with_news", "universe.top_mentioned"),
         fields=("news_mention_count", "news_mention_rank", "selection.score"),
         requires_sources=("ticker_rank",),
         why="Square ticker ranks arrive as a bundle frame. Counts are keyed on "
             "the BASE token, so dedupe_by: base_asset is not optional here — "
             "without it one asset takes several top_k slots at double weight"),
    _row("social_mentions", "cross_section", "compare", EXPRESSIBLE,
         block_refs=("universe.augment_with_news",),
         fields=("news_mention_count", "news_unique_authors"),
         requires_sources=("ticker_rank",),
         why="same join; a floor on mentions is conditions.value_above on the column"),
    _row("social_sentiment", "cross_section", "compare", EXPRESSIBLE,
         block_refs=("universe.augment_with_news", "universe.filter_sentiment"),
         fields=("news_bull_ratio", "news_bullish_count", "news_bearish_count"),
         requires_sources=("ticker_rank",),
         why="bull ratio is derived from the same rank frame; note it is a RATIO "
             "of a mention count, not a read on price"),
    _row("social_sentiment", "cross_section", "rank", EXPRESSIBLE,
         block_refs=("universe.augment_with_news", "universe.top_bullish"),
         fields=("news_bull_ratio", "selection.score"),
         requires_sources=("ticker_rank",),
         why="ranking on sentiment is what 'which coins look bullish' reduces to"),
    _row("news_event", "cross_section", "equals", NOT_EXPRESSIBLE,
         gap_id="GAP-NEWS-EVENT-TEXT",
         why="the rank frame carries aggregate counts per token, not headlines. "
             "'coins with a listing announcement' cannot be distinguished from "
             "'coins being talked about', and buzz is not the same request"),
    _row("news_event", "per_symbol_series", "*", NOT_EXPRESSIBLE,
         gap_id="GAP-NEWS-EVENT-TEXT",
         why="reconcile_intent refuses news in the trade path outright: "
             "make_signals(df) sees bars and cannot read an EventFrame, and the "
             "refusal exists because EMA/RSI were being passed off as news rules"),

    # -- open interest ------------------------------------------------------
    _row("open_interest", "cross_section", "*", EXPRESSIBLE,
         block_refs=("universe.augment_with_open_interest",
                     "universe.filter_open_interest"),
         fields=("oi_base", "oi_notional_usd"),
         requires_sources=("open_interest_snapshot",),
         why="probed 2026-08-02: the fan-out landed. GET /fapi/v1/openInterest "
             "still answers HTTP 400 without a symbol, so capture fans out over a "
             "NARROWED roster — the augment step must therefore come AFTER the "
             "narrowing steps or the coverage guard raises (87% against a frame "
             "still carrying TradFi rows; 100% once filter_crypto_only ran). "
             "Probe: filter_quote_volume -> augment_with_contract_meta -> "
             "filter_crypto_only -> augment_with_open_interest -> "
             "filter_open_interest validates with errors == []. Note oi_base is a "
             "COIN quantity; oi_notional_usd is the comparable one"),
    _row("open_interest", "per_symbol_series", "compare", EXPRESSIBLE,
         block_refs=("derivatives.oi_change_pct", "derivatives.oi_price_divergence"),
         fields=("data.derivatives", "open_interest", "open_interest_value",
                 "oi_change_bps"),
         why="probed: data.derivatives + derivatives.oi_change_pct validates "
             "clean. One declared symbol at a time is exactly what the fan-out "
             "gap means"),

    _row("open_interest_change", "cross_section", "*", EXPRESSIBLE,
         block_refs=("universe.augment_with_oi_change",
                     "universe.filter_oi_change"),
         fields=("oi_change_pct", "oi_change_notional_pct"),
         requires_sources=("oi_change_snapshot",),
         why="probed 2026-08-02: validates with errors == []. Backed by "
             "/futures/data/openInterestHist (~30d, per-symbol), so it is a "
             "narrowed fan-out with the same step-order constraint as "
             "open_interest. lookback_days defaults to 7, which is what "
             "'近一週持倉異動' asks for"),

    # -- retail positioning -------------------------------------------------
    _row("long_short_ratio", "cross_section", "*", EXPRESSIBLE,
         block_refs=("universe.augment_with_long_short_ratio",
                     "universe.filter_long_short_ratio"),
         fields=("long_short_ratio", "long_account_pct"),
         requires_sources=("long_short_ratio_snapshot",),
         why="probed 2026-08-02: validates with errors == []. "
             "globalLongShortAccountRatio is still per-symbol only, so this is a "
             "narrowed fan-out with the same step-order constraint as "
             "open_interest — place it after the narrowing steps"),
    # Catch-all AFTER the two specific rows. Without it, long_short_ratio at
    # account / per_candidate_series / side_channel scope resolves to undecidable,
    # and an undecidable row for a subject we understand perfectly well sends the
    # case to human triage instead of naming the gap.
    _row("long_short_ratio", "*", "*", NOT_EXPRESSIBLE,
         gap_id="GAP-LONG-SHORT-RATIO",
         why="only the cross-sectional snapshot landed on 2026-08-02; every other "
             "scope still has no feedable column"),
    _row("long_short_ratio", "per_symbol_series", "*", NOT_EXPRESSIBLE,
         gap_id="GAP-LONG-SHORT-RATIO",
         why="unchanged by the 2026-08-02 fan-out work, which added a "
             "CROSS-SECTIONAL snapshot only. derivatives.long_short_ratio_state "
             "exists, but long_short_ratio is in no DATA_SECTIONS, so the column "
             "is absent from the trade-path dry-run frame and the spec dies with "
             "'cannot resolve reference long_short_ratio'. A block you cannot "
             "feed is not a capability"),

    # -- liquidations -------------------------------------------------------
    _row("liquidation", "per_symbol_series", "compare", EXPRESSIBLE,
         block_refs=("derivatives.liquidation_imbalance",
                     "derivatives.liquidation_clusters"),
         fields=("data.liquidations", "long_liq_notional_usd",
                 "short_liq_notional_usd", "liq_imbalance_ratio"),
         why="probed: data.liquidations + derivatives.liquidation_imbalance "
             "validates clean"),
    _row("liquidation", "cross_section", "*", NOT_EXPRESSIBLE,
         gap_id="GAP-LIQUIDATION-CROSS-SECTION",
         why="the liquidations feed is a per-symbol event stream; no universe "
             "block joins it, so 'which coins just got liquidated hardest' has "
             "nothing to rank"),

    # -- market cap / on-chain ----------------------------------------------
    _row("market_cap", "*", "*", NOT_EXPRESSIBLE,
         gap_id="GAP-MARKET-CAP",
         why="no Binance public endpoint carries market cap or circulating "
             "supply. 24h turnover correlates with it and is the substitution "
             "that keeps getting made — 'small caps' answered with 'thin books' "
             "is a different screen, and the basket does not show which was run"),
    _row("onchain_holder_concentration", "*", "*", NOT_EXPRESSIBLE,
         gap_id="GAP-ONCHAIN-CONCENTRATION",
         why="chain data is not in this repo's sources at all; nothing on the "
             "exchange side is even a bad proxy for holder distribution"),

    # -- what the instrument IS ---------------------------------------------
    _row("sector_label", "cross_section", "equals", EXPRESSIBLE,
         block_refs=("universe.augment_with_contract_meta", "universe.filter_sub_type"),
         fields=("underlying_sub_type",),
         requires_sources=("contract_meta",),
         why="exchangeInfo's underlyingSubType, joined as a comma-separated "
             "scalar. Multi-valued: a coin can be 'Alpha,DeFi', so include= "
             "matches ANY named tag"),
    _row("sector_label", "cross_section", "exclude", EXPRESSIBLE,
         block_refs=("universe.augment_with_contract_meta", "universe.filter_sub_type"),
         fields=("underlying_sub_type",),
         requires_sources=("contract_meta",),
         why="exclude= drops a row carrying any named tag; this is one of the "
             "three non-equivalent spellings of 'no TradFi' (577 of 727)"),
    _row("contract_type", "cross_section", "equals", EXPRESSIBLE,
         block_refs=("universe.augment_with_contract_meta",
                     "universe.filter_underlying_type", "universe.filter_crypto_only"),
         fields=("contract_type", "underlying_type", "base_asset", "quote_asset"),
         requires_sources=("contract_meta",),
         why="'crypto only' is the whitelist underlying_type == COIN (575), not "
             "the blacklist contract_type != TRADIFI_PERPETUAL (577). The two "
             "differ by ALLUSDT and BTCDOMUSDT, synthetic INDEX baskets that the "
             "blacklist lets through as if they were coins — hence "
             "filter_crypto_only exists as its own name"),
    _row("contract_type", "cross_section", "exclude", EXPRESSIBLE,
         block_refs=("universe.augment_with_contract_meta",
                     "universe.filter_underlying_type"),
         fields=("contract_type", "underlying_type"),
         requires_sources=("contract_meta",),
         why="exclude=EQUITY etc. Matching is exact and case-insensitive, never "
             "a substring, so EQUITY does not silently take HK_EQUITY with it"),

    # -- technical indicators -----------------------------------------------
    _row("technical_indicator", "per_symbol_series", "compare", EXPRESSIBLE,
         block_refs=("indicators.supertrend", "indicators.ema", "indicators.sma",
                     "indicators.rsi", "indicators.macd", "indicators.adx",
                     "indicators.atr", "indicators.bollinger", "indicators.donchian",
                     "conditions.value_above", "conditions.value_below",
                     "conditions.ma_cross_above", "conditions.ma_cross_below",
                     "conditions.rsi_oversold", "conditions.rsi_overbought"),
         fields=("open", "high", "low", "close", "volume"),
         why="probed: indicators.supertrend(10, 3) on the primary interval "
             "validates clean. This is the ONE scope where the request the "
             "Supertrend accident came from is fully expressible"),
    # ⚠️ THESE TWO ROWS ARE STALE AS OF 2026-08-02 AND ARE LEFT THAT WAY
    # DELIBERATELY — see TABLE_DERIVED_FROM["pending_review"] for the whole reason
    # and for the flip that is already written out there. universe.augment_with_
    # indicator + the `universe_bars` node exist and were probed to errors == [],
    # so BOTH of these are now expressible; the verdict is wired into eleven tests
    # in tests/nl2yaml/ whose subject IS this gap, and flipping it is a decision
    # about that dataset rather than about the blocks package.
    _row("technical_indicator", "cross_section", "compare", PROXY_ONLY,
         gap_id="GAP-PER-SYMBOL-INDICATOR",
         proxy_block_refs=("universe.top_losers", "universe.top_gainers",
                           "universe.filter_change_pct"),
         degradation=(
             "24h change stands in for a trend indicator. It loses the "
             "indicator's period, its timeframe, its state (a Supertrend flip is "
             "not a percentage), and the distinction between 'downtrend' and "
             "'one bad hour'. THIS IS THE SUBSTITUTION THAT SHIPPED: "
             "'Supertrend(10,3) bearish on H4/H1/M15' became top_losers(n=30), "
             "the spec validated, the run succeeded, and the output named no proxy"),
         why="the cross-sectional frame has one row per instrument and no bars, "
             "so there is nothing to compute an indicator over. Withheld by "
             "default: with 'top_losers' absent from the prompt the substitution "
             "is unwritable, and the condition is recorded unconvertible instead"),
    _row("technical_indicator", "per_candidate_series", "*", NOT_EXPRESSIBLE,
         gap_id="GAP-PER-SYMBOL-INDICATOR",
         why="'scan the market, then run the indicator on each survivor' is the "
             "single most common shape in the selection corpus and the repo has "
             "no object for it: selection sees no bars, and signals: sees one "
             "declared symbol. Needs universe.augment_with_indicator"),

    # -- multi-timeframe ----------------------------------------------------
    _row("multi_timeframe", "per_symbol_series", "resonance", PROXY_ONLY,
         gap_id="GAP-PER-SYMBOL-INDICATOR",
         proxy_block_refs=("conditions.multi_timeframe_alignment",),
         degradation=(
             "data.htf attaches exactly one kind of higher-timeframe column, "
             "_htf_<tf>_sma_<period> — build_plugin takes (interval, sma_period) "
             "tuples and nothing else. So 'Supertrend bearish on H4' becomes "
             "'close below the 4h SMA', a different statement about a different "
             "indicator; only the timeframe survives"),
         why="the resonance combinator is real and works, but it can only be fed "
             "HTF SMAs, so it is honest for SMA alignment and a proxy for "
             "everything else. An SMA request should be routed to the "
             "technical_indicator / per_symbol_series row instead"),
    # ⚠️ ALSO STALE as of 2026-08-02 — see TABLE_DERIVED_FROM["pending_review"].
    _row("multi_timeframe", "cross_section", "resonance", NOT_EXPRESSIBLE,
         gap_id="GAP-PER-SYMBOL-INDICATOR",
         why="no bars in the frame at one timeframe, let alone three"),

    # -- history ------------------------------------------------------------
    _row("historical_range", "per_symbol_series", "window", EXPRESSIBLE,
         block_refs=("indicators.highest", "indicators.lowest",
                     "indicators.donchian", "indicators.rolling_quantile",
                     "conditions.close_above", "conditions.close_below"),
         fields=("high", "low", "close"),
         why="probed: indicators.highest(high, 30) + conditions.close_above "
             "validates clean. 'near its 30-day high' is a rolling window"),
    # ⚠️ ALSO STALE as of 2026-08-02 — see TABLE_DERIVED_FROM["pending_review"].
    _row("historical_range", "cross_section", "window", NOT_EXPRESSIBLE,
         gap_id="GAP-HISTORICAL-WINDOW",
         why="the universe frame is ONE instant. 'coins down 50% from their "
             "yearly high' needs a per-symbol history the frame does not carry, "
             "and priceChangePercent is a 24h number — a 20x shorter window "
             "wearing the same words"),

    # -- book / spread ------------------------------------------------------
    # Both cross_section rows were NOT_EXPRESSIBLE / GAP-SPREAD-DEPTH until the
    # `book_ticker` node + universe.augment_with_spread landed (2026-08-02).
    # Re-probed to errors == [] with validate_spec, four ways: filter_spread,
    # conditions.value_below on the column, score: spread_bps, and
    # filter_top_of_book. The gap id stays live — the per_symbol_series row below
    # still has no frame to stand on.
    _row("spread_liquidity", "cross_section", "compare", EXPRESSIBLE,
         block_refs=("universe.augment_with_spread", "universe.filter_spread",
                     "universe.filter_top_of_book", "conditions.value_below",
                     "conditions.value_above"),
         fields=("spread_bps", "top_of_book_usd"),
         requires_sources=("book_ticker",),
         why="bookTicker answers for all 727 symbols in ONE request, so this is a "
             "real cross-section and not a fan-out — the augment step may sit "
             "anywhere in the pipeline. `with: [book_ticker]` is mandatory (the "
             "block fetches when the source is omitted, so validate would hit the "
             "network). Turnover is NOT this condition: measured, SNXXUSDT cleared "
             "a $100m floor at 11 bps and TAKEUSDT turned over $24.5m with $0.03 "
             "at the ask"),
    _row("spread_liquidity", "cross_section", "rank", EXPRESSIBLE,
         block_refs=("universe.augment_with_spread",),
         fields=("spread_bps", "top_of_book_usd", "selection.score",
                 "selection.order", "selection.min_score", "selection.max_score"),
         requires_sources=("book_ticker",),
         why="'the most liquid coins' is score: top_of_book_usd + order: desc, or "
             "score: spread_bps + order: asc — two different questions that a "
             "turnover ranking collapses into one. Instruments with no quotable "
             "touch are NaN and are dropped by the ranker, never ranked first"),
    _row("spread_liquidity", "per_symbol_series", "compare", NOT_EXPRESSIBLE,
         gap_id="GAP-SPREAD-DEPTH",
         why="probed: microstructure.order_imbalance resolves, but its inputs "
             "(taker_buy_volume / taker_sell_volume) are in no DATA_SECTIONS, so "
             "the dry-run frame has no such columns and the spec cannot be fed"),

    # -- what to do about it ------------------------------------------------
    _row("entry_exit_plan", "per_symbol_series", "plan", EXPRESSIBLE,
         fields=("risk.exit.type", "risk.exit.stop_pct", "risk.exit.tp_pct",
                 "risk.exit.atr_period", "risk.exit.stop_mult", "risk.exit.tp_mult",
                 "sizing.size"),
         why="probed: risk.exit atr_stop_tp validates clean. EXIT_KEYS is closed "
             "because an unknown key used to be defaulted past in silence — "
             "'stop_pctt' cost a real stop-loss"),
    _row("entry_exit_plan", "cross_section", "plan", NOT_EXPRESSIBLE,
         gap_id="GAP-ENTRY-EXIT-PER-CANDIDATE",
         why="the candidate contract has a `trade` slot and the YAML selection "
             "path never fills it — every candidate comes back trade: null. "
             "'give me five shorts with entry and stop' answers the first half "
             "and silently drops the second"),
    _row("compound_select_then_trade", "*", "execute", NOT_EXPRESSIBLE,
         gap_id="GAP-COMPOUND-SELECT-THEN-TRADE",
         why="one spec is either selection: or signals: — validate_spec refuses "
             "both, because they emit different signal kinds. classify_request "
             "returns kind='ambiguous' for these and generation stops there, "
             "which is right: picking a half would discard the other in silence"),
    _row("alert_notify", "side_channel", "notify", NOT_EXPRESSIBLE,
         gap_id="GAP-ALERT-NOTIFY",
         why="'tell me when X happens' has no output slot: a spec emits signals, "
             "and there is no notification target in the schema. Answering it "
             "with a strategy that trades X is a different and more expensive act"),
    _row("account_ops", "account", "*", NOT_EXPRESSIBLE,
         gap_id="GAP-ACCOUNT-OPS",
         why="balances, leverage changes, closing an existing position: the "
             "runtime has them, the YAML surface deliberately does not. A "
             "generated spec must not be able to reach an account"),

    # -- naming instruments -------------------------------------------------
    _row("symbol_blacklist", "cross_section", "exclude", EXPRESSIBLE,
         block_refs=("universe.exclude_symbols",),
         fields=("instrument_id",),
         why="full pair names, listed one by one. 'exclude the majors' has no "
             "semantic block, so the four names have to be written out — and "
             "only the USDT halves unless the USDC ones are named too"),
    _row("symbol_whitelist", "cross_section", "require", EXPRESSIBLE,
         block_refs=("universe.only_symbols",),
         fields=("instrument_id",),
         why="reconcile_intent cross-checks this one against the names the user "
             "actually said, because narrowing the universe to symbols nobody "
             "asked for is the most consequential thing a model can invent"),
    _row("quote_currency", "cross_section", "exclude", EXPRESSIBLE,
         block_refs=("universe.filter_quote_suffix",),
         fields=("instrument_id", "quote_asset"),
         why="a suffix test, so it is about the QUOTE currency: excluding USDC "
             "keeps USDCUSDT. The hand-written 4-name list it replaced missed 34 "
             "of the 38 USDC pairs in the cross-section"),
    _row("quote_currency", "cross_section", "require", EXPRESSIBLE,
         block_refs=("universe.filter_quote_suffix",),
         fields=("instrument_id", "quote_asset"),
         why="'USDT pairs only' is the same block with exclude: false"),

    # -- shape of the answer ------------------------------------------------
    _row("basket_size", "cross_section", "top_k", EXPRESSIBLE,
         fields=("selection.top_k", "selection.dedupe_by"),
         why="a ceiling, never a quota: a spec three symbols clear returns three, "
             "never five padded with rows that failed the filter"),
    _row("direction", "cross_section", "require", EXPRESSIBLE,
         fields=("selection.long_when", "selection.short_when"),
         why="declaring only short_when drops the rows that fail it rather than "
             "emitting them as neutral. reconcile_intent also refuses a direction "
             "the user never asked for"),
    _row("score_order", "cross_section", "rank", EXPRESSIBLE,
         fields=("selection.order", "selection.min_score", "selection.max_score"),
         why="asc/desc, and min_score/max_score are absolute bounds that do NOT "
             "flip with it — floor stays a floor. A basket taken from the wrong "
             "end of a signed column looks perfectly healthy from outside"),
    _row("market_type", "*", "equals", EXPRESSIBLE,
         fields=("data.market_type",),
         why="spot vs futures. reconcile_intent enforces it against the request, "
             "since 'perpetuals' answered with spot is a different instrument"),

    # -- the residue --------------------------------------------------------
    _row("vague_criterion", "*", "*", NOT_EXPRESSIBLE,
         gap_id="GAP-VAGUE-CRITERION",
         why="'coins nobody has noticed yet', 'ones about to pump', 'good "
             "fundamentals'. There is no column, and the danger is not that the "
             "model fails — it is that mentions or turnover LOOK like an answer, "
             "so the output is confident and unfalsifiable"),
)


def _build_index() -> Dict[Tuple[str, str, str], Capability]:
    index: Dict[Tuple[str, str, str], Capability] = {}
    for row in CAPABILITY_TABLE:
        if row.key in index:
            raise ValueError(
                "duplicate capability row %s; two verdicts for one key means the "
                "answer depends on iteration order" % (row.key,))
        index[row.key] = row
    return index


_INDEX = _build_index()


def subjects() -> Tuple[str, ...]:
    """Every subject the table rules on, sorted."""
    return tuple(sorted({row.subject for row in CAPABILITY_TABLE}))


def table_as_rows() -> Tuple[Dict[str, Any], ...]:
    """The table as plain dicts, for reports and for diffing across commits."""
    return tuple(
        {
            "subject": row.subject, "scope": row.scope, "operator": row.operator,
            "verdict": row.verdict, "gap_id": row.gap_id,
            "block_refs": list(row.block_refs), "fields": list(row.fields),
            "requires_sources": list(row.requires_sources),
            "proxy_block_refs": list(row.proxy_block_refs),
        }
        for row in CAPABILITY_TABLE
    )


def lookup(subject: str, scope: str = "*", operator: str = "*") -> Capability:
    """The ruling for one condition. Most specific row wins.

    An unlisted key comes back as an ``unknown`` verdict rather than an
    exception: "we have not ruled on this" is a real state the pipeline has to
    represent (the case is shelved for a human, and a later pass proposes the new
    row). Returning ``expressible`` or ``not_expressible`` by default would be
    the guess this whole module exists to avoid.
    """
    for candidate in (
        (subject, scope, operator),
        (subject, scope, "*"),
        (subject, "*", operator),
        (subject, "*", "*"),
    ):
        row = _INDEX.get(candidate)
        if row is not None:
            return row
    return Capability(
        subject=subject, scope=scope, operator=operator, verdict=UNKNOWN,
        why="no row for (%s, %s, %s); shelved for a human to rule on, because "
            "guessing either way is how a proxy or a false refusal gets in"
            % (subject, scope, operator))


# ---------------------------------------------------------------------------
# Conditions and the per-case plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One structured condition out of a user request.

    Deliberately carries NO quote of what the user wrote. The corpus these come
    from holds user_id and verbatim questions and both remotes are public repos,
    so the structured triple is the only part allowed to travel — and the check
    is in :func:`normalize_condition`, at the boundary, rather than in a
    reviewer's memory.
    """

    subject: str
    operator: str
    scope: str = "cross_section"
    #: threshold / list / tag. Never free text the user typed.
    value: Any = None
    #: ``require`` (must hold) or ``exclude`` (must not hold).
    polarity: str = "require"
    #: True when the condition has a checkable number or member list, i.e. when
    #: G1e is obliged to register a predicate for it.
    quantified: bool = False
    id: str = ""

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("a condition needs a subject")
        if self.polarity not in ("require", "exclude"):
            raise ValueError("polarity must be require|exclude, got %r" % (self.polarity,))
        if self.operator not in OPERATORS:
            raise ValueError("operator must be one of %s, got %r"
                             % (sorted(OPERATORS), self.operator))
        if self.scope not in SCOPES:
            raise ValueError("scope must be one of %s, got %r"
                             % (sorted(SCOPES), self.scope))

    @property
    def capability(self) -> Capability:
        return lookup(self.subject, self.scope, self.operator)


#: Keys that would carry user text into a public repo if a caller passed them.
_FORBIDDEN_CONDITION_KEYS = ("quote", "text", "raw", "utterance", "message",
                             "user_id", "nl", "verbatim")


def normalize_condition(cond: Any) -> Condition:
    """Accept a :class:`Condition` or a plain mapping; refuse a leaky one.

    Mappings are accepted so that whatever upstream stage produces the corpus's
    ``conditions[]`` does not have to import this module's dataclass. Unknown
    keys raise instead of being ignored: a misspelt ``quantifed`` would otherwise
    turn off the G1e predicate for that condition in silence, and a condition
    with no predicate is one nobody ever checks again.
    """
    if isinstance(cond, Condition):
        return cond
    if not isinstance(cond, Mapping):
        raise TypeError("a condition must be a Condition or a mapping, got %s"
                        % type(cond).__name__)
    leaked = sorted(key for key in cond if key in _FORBIDDEN_CONDITION_KEYS)
    if leaked:
        raise ValueError(
            "condition carries %s. Verbatim user text and user ids stay outside "
            "the repo (see the corpus privacy rule); only the structured "
            "subject/operator/value may travel." % leaked)
    allowed = {"subject", "operator", "scope", "value", "polarity", "quantified", "id"}
    unknown = sorted(set(cond) - allowed)
    if unknown:
        raise ValueError(
            "unknown condition key(s) %s; allowed: %s. A misspelt key here is "
            "accepted in silence and quietly removes a G1e predicate"
            % (unknown, sorted(allowed)))
    return Condition(**{key: cond[key] for key in cond})


@dataclass(frozen=True)
class ConversionPlan:
    """What the converter is allowed to see, for ONE case.

    Per case, not per table, and that is the whole design. ``top_losers`` is a
    legitimate name for "biggest 24h fallers" and an illegitimate one for
    "Supertrend bearish across three timeframes"; a single global allowlist
    cannot say that, so the vocabulary is computed from the conditions actually
    present in this request.
    """

    conditions: Tuple[Condition, ...] = ()
    expressible: Tuple[Condition, ...] = ()
    proxied: Tuple[Condition, ...] = ()
    unconvertible: Tuple[Tuple[Condition, str], ...] = ()
    shelved: Tuple[Condition, ...] = ()
    vocabulary: frozenset = field(default_factory=frozenset)
    refused_vocabulary: frozenset = field(default_factory=frozenset)
    required_sources: Tuple[str, ...] = ()

    @property
    def gap_ids(self) -> Tuple[str, ...]:
        """The refusals this case must state, sorted and deduplicated."""
        return tuple(sorted({gap for _cond, gap in self.unconvertible}))

    @property
    def converter_conditions(self) -> Tuple[Condition, ...]:
        """The conditions that reach the converter at all.

        Unconvertible and shelved conditions are absent — not annotated,
        absent. A condition left in the prompt with a "cannot do this" note is a
        condition the model will try to satisfy anyway.
        """
        return self.expressible + self.proxied

    @property
    def blocked(self) -> bool:
        """True when nothing should be generated for this case."""
        return bool(self.shelved) or not self.converter_conditions


def plan_conversion(
    conditions: Iterable[Any],
    *,
    allow_proxy_for: Sequence[str] = (),
) -> ConversionPlan:
    """Split a case's conditions by verdict and compute its vocabulary.

    *allow_proxy_for* holds condition ids a human has reviewed and accepted the
    degradation for. It takes ids and not subjects so that opening the proxy for
    one condition in one case cannot silently open it for every similar condition
    in every other case.
    """
    opened = set(allow_proxy_for)
    normalized = tuple(normalize_condition(cond) for cond in conditions)
    unknown_ids = opened - {cond.id for cond in normalized if cond.id}
    if unknown_ids:
        raise ValueError(
            "allow_proxy_for names condition id(s) %s that are not in this case; "
            "a typo here reads as 'proxy refused' and the reviewer's decision is "
            "lost without a trace" % sorted(unknown_ids))

    expressible: list = []
    proxied: list = []
    unconvertible: list = []
    shelved: list = []
    granted: set = set()
    withheld: set = set()
    sources: set = set()

    for cond in normalized:
        row = cond.capability
        if row.verdict == EXPRESSIBLE:
            expressible.append(cond)
            granted.update(row.granted)
            sources.update(row.requires_sources)
        elif row.verdict == PROXY_ONLY:
            if cond.id and cond.id in opened:
                # Recorded as proxied, never as expressible: the case's report
                # has to keep saying the answer is a stand-in, or the next reader
                # of the basket has no way to know.
                proxied.append(cond)
                granted.update(row.proxy_block_refs)
                sources.update(row.requires_sources)
            else:
                withheld.update(row.proxy_block_refs)
                unconvertible.append((cond, row.gap_id))
        elif row.verdict == NOT_EXPRESSIBLE:
            unconvertible.append((cond, row.gap_id))
        else:
            shelved.append(cond)

    # A name granted by any condition in this case stays granted. Otherwise the
    # from-user-chat case would both grant top_losers (for its explicit "biggest
    # 24h fallers" condition) and refuse it (as the withheld Supertrend proxy),
    # and a caller building a prompt out of both sets would get contradictory
    # instructions. Grant wins, and the Supertrend condition is still recorded
    # unconvertible — which is the honest reading: the block is there for the
    # condition that asked for it, not for the one it would have stood in for.
    return ConversionPlan(
        conditions=normalized,
        expressible=tuple(expressible),
        proxied=tuple(proxied),
        unconvertible=tuple(unconvertible),
        shelved=tuple(shelved),
        vocabulary=frozenset(granted),
        refused_vocabulary=frozenset(withheld - granted),
        required_sources=tuple(sorted(sources)),
    )


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def assert_table_is_current() -> None:
    """Raise if the table's grounding no longer matches ``cyqnt_trd.blocks``.

    Two directions, both of which are real failures rather than housekeeping:

    * a granted ref that no longer resolves puts a name in the converter's
      vocabulary that cannot be written into a working spec — the model gets
      blamed for a table's stale promise;
    * a NEW ``universe`` block that no row mentions is worse. A capability that
      landed while a row still says ``not_expressible`` makes the pipeline refuse
      a request it can now serve, and refusals are not retried, so the case is
      lost silently. That is the failure mode this project keeps hitting, so it
      raises rather than warns.
    """
    from cyqnt_trd.blocks import universe as universe_blocks
    from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (
        DENIED_FUNCTION_NAMES,
        SpecError,
        resolve_block,
    )

    unresolvable = []
    for row in CAPABILITY_TABLE:
        for ref in tuple(row.block_refs) + tuple(row.proxy_block_refs):
            try:
                resolve_block(ref)
            except SpecError as exc:
                unresolvable.append("%s (row %s): %s" % (ref, row.key, exc))
    if unresolvable:
        raise AssertionError(
            "capability table names block(s) a spec can no longer use:\n  - "
            + "\n  - ".join(unresolvable)
            + "\nThe table was derived from %s; update the affected rows."
            % TABLE_DERIVED_FROM["commit"])

    live = {name for name in universe_blocks.__all__
            if name not in DENIED_FUNCTION_NAMES and not name[0].isupper()}
    recorded = {name for name in TABLE_DERIVED_FROM["universe_blocks"]
                if name not in DENIED_FUNCTION_NAMES and not name[0].isupper()}
    added = sorted(live - recorded)
    removed = sorted(recorded - live)
    if added or removed:
        raise AssertionError(
            "blocks.universe changed since the capability table was derived "
            "(commit %s): added=%s removed=%s.\nRevisit every not_expressible / "
            "proxy_only row a new block might now serve, then update "
            "TABLE_DERIVED_FROM['universe_blocks']. Leaving the table alone means "
            "refusing requests the repo can now answer."
            % (TABLE_DERIVED_FROM["commit"], added, removed))


def with_proxy_opened(row: Capability) -> Capability:
    """A copy of *row* with the proxy accepted, for a caller that wants to show
    a reviewer what opening it would grant. Never used by :func:`plan_conversion`
    — see :class:`Capability.allow_proxy`."""
    if row.verdict != PROXY_ONLY:
        raise ValueError("%s is not a proxy row" % (row.key,))
    return replace(row, verdict=EXPRESSIBLE, block_refs=row.proxy_block_refs,
                   proxy_block_refs=(), gap_id=None, degradation=None,
                   why=row.why + " [PROXY OPENED: " + (row.degradation or "") + "]")
