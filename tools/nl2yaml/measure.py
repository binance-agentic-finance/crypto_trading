"""Gate0 measurement: how many trainable gold cases can this corpus actually yield?

Joins the two halves that already exist — :mod:`tools.nl2yaml.mine` (regex
condition families over 51,595 chat rows) and :mod:`tools.nl2yaml.capability`
(the 49-row ``(subject, scope, operator) -> verdict`` adjudication) — and runs
them over the whole corpus with **no LLM in the loop**.

Why a proxy pass at all
-----------------------
``capability.plan_conversion`` wants ``conditions[]`` in its own vocabulary, and
that vocabulary is the output of the A1 converter (an LLM). This module stands
in for A1 with a hand-written map from the miner's ``(family, subject)`` pairs to
capability subjects, so the whole funnel can be costed before a single token is
spent.

Everything that map gets wrong pushes the answer the *same* way, which is what
makes the output usable as a bound rather than as an estimate:

* the miner's recall is a floor, not a ceiling. Its regexes find a **subset** of
  the conditions in a request, so a row counted "every condition expressible"
  can still hold an unmined ``not_expressible`` one. This is precisely how the
  "exclude tradfi" accident happened: the blocking condition was in the text and
  not in the machine's hands.
* the map is coarse in the generous direction. ``成交量`` (a coin count) is
  scored against ``quote_volume_24h`` (a quote-currency column) because that is
  the nearest ruling that exists; ``排除`` is scored as ``symbol_blacklist``
  because the miner does not know *what* is being excluded and every plausible
  target happens to be expressible.
* the headline number treats ``unknown`` (no table row) as if it were
  expressible. Some of those are genuine table omissions (``data.interval`` is a
  plain spec key that no row rules on) and some are questions nobody has ruled on
  yet (a backtest win-rate floor). Optimism here is deliberate: this is the upper
  bound, and the report's ``strict`` column gives the same count with ``unknown``
  blocking, next to the list of exactly which keys the optimism consists of.

So: **the true count is lower than every "upper bound" number in this report,
and the report never claims otherwise.** The one direction the map is *not*
generous in is scope, which it cannot guess for an ambiguous request; there it
takes the better of the two frames per row, again upward.

Privacy. This module reads the miner's repo-bound ``candidates.jsonl`` and the
source CSV's *structure* only; it never writes user text. Its outputs are counts,
enum ids and hashes. The CSV is re-read because tier D rows and continuation
fragments are (correctly) absent from ``candidates.jsonl`` and the funnel needs
them; the re-mined records are rebuilt with the miner's own functions so the
numbers cannot drift from Part B, and :func:`verify_against_mine` fails the run
if they do.

Usage::

    CSV=docs/user_demand_analysis/2026-05_07_trading_intent
    ./.venv-standard-bot/bin/python -m tools.nl2yaml.measure \
        --csv "$CSV/trading_intent_chats_2026-05_07_zh_en.csv" \
        --out tools/nl2yaml/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tools.nl2yaml import capability as cap
from tools.nl2yaml import mine

MEASURE_VERSION = "gate0-measure/1"

#: The two frames a mined condition can be scored against. ``account`` and
#: ``side_channel`` never arise from the miner's families, and
#: ``per_candidate_series`` is not used: a cross-sectional request that names an
#: indicator is scored on ``cross_section`` because that is literally the frame
#: the repo would hand it, and both scopes carry the same
#: ``GAP-PER-SYMBOL-INDICATOR`` verdict anyway.
CROSS = "cross_section"
SERIES = "per_symbol_series"

#: How a request's shape picks the frame. ``both`` and ``unclear`` have no single
#: answer, so those rows are scored under each frame and the better result is
#: kept (see :func:`plan_row`) — upward, like every other approximation here.
SHAPE_SCOPE = {"selection": (CROSS,), "trade": (SERIES,),
               "both": (CROSS, SERIES), "unclear": (CROSS, SERIES)}


# ---------------------------------------------------------------------------
# The proxy map
# ---------------------------------------------------------------------------

#: What a mapping is worth, so the report can subtract the parts that are only
#: true by charity.
#:
#: ``exact``
#:     the miner's subject and the table's subject are the same quantity.
#: ``coarse``
#:     the same *family* of quantity, measured differently or on an unknown
#:     target. Counts as expressible; a human would have to confirm.
#: ``overstates``
#:     the table rules the class expressible but the specific thing the user
#:     named has no implementation behind it. These are counted in the headline
#:     and subtracted again in ``upper_bound_minus_overstated``.
#: ``unruled``
#:     deliberately mapped onto a subject the table does NOT carry, so
#:     :func:`capability.lookup` returns ``unknown``. Not a guess dressed up as a
#:     verdict — an explicit "nobody has ruled on this", counted and listed.
#: ``non_condition``
#:     not a criterion at all ("篩選", "全市場"). Dropped before the capability
#:     pass rather than scored, because a request whose only mined condition is
#:     "please screen" has nothing a gold spec could be graded against.
FIDELITIES = ("exact", "coarse", "overstates", "unruled", "non_condition")


@dataclass(frozen=True)
class ProxySubject:
    """One ``(miner family, miner subject) -> capability subject`` mapping."""

    cap_subject: str
    cap_operator: str
    #: ``None`` means "take the frame from the request's shape". A fixed value
    #: means the quantity only exists in that frame whatever the request looks
    #: like — a universe filter is cross-sectional even inside a trade request.
    scope: str | None
    fidelity: str
    why: str

    def __post_init__(self) -> None:
        if self.fidelity not in FIDELITIES:
            raise ValueError("unknown fidelity %r" % (self.fidelity,))
        if self.fidelity == "non_condition":
            return
        if self.cap_operator not in cap.OPERATORS:
            raise ValueError("%r is not a capability operator" % (self.cap_operator,))
        if self.scope is not None and self.scope not in cap.SCOPES:
            raise ValueError("%r is not a capability scope" % (self.scope,))


def _drop(why: str) -> ProxySubject:
    return ProxySubject("", "*", None, "non_condition", why)


#: Subjects deliberately absent from the capability table, used so that a
#: question nobody has ruled on comes back ``unknown`` instead of being answered
#: by this module. Asserted absent at import time.
UNRULED_SUBJECTS = (
    "bar_interval",             # data.interval — a plain spec key, no row rules it
    "unattributed_threshold",   # "低於 30" with no subject in the 14 chars before it
    "backtest_win_rate",        # "勝率 > 60%" is a demand on results, not a condition
    "portfolio_drawdown_limit", # "最大回撤 < 20%" likewise
)

#: ``(family, subject)`` -> mapping. Every pair the miner can emit is listed; a
#: pair that is not raises in :func:`map_condition` rather than being skipped,
#: because a silently skipped condition is a condition that cannot block a case
#: and that is the failure mode this dataset is built to avoid.
PROXY_MAP: dict[tuple[str, str], ProxySubject] = {

    # -- threshold: "<subject> <comparator> <number>" ------------------------
    ("threshold", "volume"): ProxySubject(
        "quote_volume_24h", "compare", None, "coarse",
        "the miner's 成交量/volume is a COIN count; the cross-section column is "
        "quoteVolume, in quote currency. Nearest ruling that exists, and it is "
        "the substitution a converter would make anyway"),
    ("threshold", "quote_volume"): ProxySubject(
        "quote_volume_24h", "compare", None, "exact",
        "成交額/turnover is quoteVolume outright"),
    ("threshold", "market_cap"): ProxySubject(
        "market_cap", "compare", "*", "exact",
        "no Binance endpoint carries market cap; the table refuses it in every scope"),
    ("threshold", "price_change_pct"): ProxySubject(
        "price_change_24h", "compare", None, "coarse",
        "the window is assumed to be 24h because that is the only one the "
        "cross-section has. A trade-shaped '漲幅 5%' is per-bar and the table has "
        "no per_symbol_series row for it, so those come back unknown — a real "
        "table omission this sweep surfaces, not a capability gap"),
    ("threshold", "funding_rate"): ProxySubject(
        "funding_rate", "compare", None, "exact",
        "expressible in both frames, which is rare in this table"),
    ("threshold", "price"): ProxySubject(
        "technical_indicator", "compare", SERIES, "coarse",
        "a price level is conditions.value_above(close) on a bar series. Pinned to "
        "the series frame on purpose: in the cross-section the universe frame "
        "carries no lastPrice column, and routing it through the indicator row "
        "there would blame GAP-PER-SYMBOL-INDICATOR for a missing price column"),
    ("threshold", "rsi"): ProxySubject(
        "technical_indicator", "compare", None, "exact", "indicators.rsi"),
    ("threshold", "macd"): ProxySubject(
        "technical_indicator", "compare", None, "exact", "indicators.macd"),
    ("threshold", "atr"): ProxySubject(
        "technical_indicator", "compare", None, "exact", "indicators.atr"),
    ("threshold", "leverage"): ProxySubject(
        "entry_exit_plan", "plan", None, "overstates",
        "verified against spec.py at TABLE_DERIVED_FROM: SIZING_KEYS == {'size'}, "
        "and neither EXIT_KEYS nor DATA_KEYS carries a leverage key. So the YAML "
        "surface cannot express '10x' at all. Scored against the plan row so the "
        "request is not lost, counted in overstated_sources, and it needs its own "
        "GAP id in capability.GAP_IDS — this pass must not invent one"),
    ("threshold", "win_rate"): ProxySubject(
        "backtest_win_rate", "compare", "*", "unruled",
        "'only strategies with >60% win rate' is a constraint on the backtest "
        "report, not a predicate a spec can carry. Nobody has ruled on it"),
    ("threshold", "unspecified"): ProxySubject(
        "unattributed_threshold", "compare", "*", "unruled",
        "the miner found a comparator and a number and no subject within 14 "
        "characters. A1 would resolve most of these; guessing here would invent "
        "the very condition the gold spec is supposed to be graded on"),

    # -- rank / top-N -------------------------------------------------------
    ("rank_topn", "universe"): ProxySubject(
        "basket_size", "top_k", CROSS, "exact",
        "'top 5' is selection.top_k. NOTE this rules the SIZE of the basket only; "
        "what it is ranked ON is a separate condition, and if the miner did not "
        "catch that one the row looks cheaper than it is"),

    # -- timeframe ----------------------------------------------------------
    # Handled per row, not per condition: one interval is a spec key, two or more
    # is a resonance request. See :func:`_timeframe_conditions`.

    # -- direction ----------------------------------------------------------
    ("direction", "side"): ProxySubject(
        "direction", "require", CROSS, "coarse",
        "the table rules direction in the cross-section (selection.long_when / "
        "short_when) and does not rule it on a bar series, where it is equally "
        "trivial. Pinned to the ruled frame rather than left unknown for all "
        "12k trade rows; the borrowing is recorded in scope_borrowed"),

    # -- risk ---------------------------------------------------------------
    ("risk", "stop_loss"): ProxySubject(
        "entry_exit_plan", "plan", None, "exact", "risk.exit.stop_pct / stop_mult"),
    ("risk", "take_profit"): ProxySubject(
        "entry_exit_plan", "plan", None, "exact", "risk.exit.tp_pct / tp_mult"),
    ("risk", "position_size"): ProxySubject(
        "entry_exit_plan", "plan", None, "exact", "sizing.size"),
    ("risk", "trailing_stop"): ProxySubject(
        "entry_exit_plan", "plan", None, "exact",
        "verified against spec.py at TABLE_DERIVED_FROM: VALID_EXIT_TYPES carries "
        "atr_trailing_stop and EXIT_KEYS carries trail_mult. The capability row's "
        "field list does not enumerate them, but the surface has them"),
    ("risk", "leverage"): ProxySubject(
        "entry_exit_plan", "plan", None, "overstates", "as ('threshold','leverage')"),
    ("risk", "max_drawdown"): ProxySubject(
        "portfolio_drawdown_limit", "plan", "*", "unruled",
        "a drawdown ceiling is a portfolio-level constraint; the YAML surface has "
        "no slot and the table does not rule it"),
    ("risk", "risk_control"): _drop(
        "'風控' / 'risk management' is a topic word, not a criterion, and the "
        "specific demands inside it (止損 / 倉位 / 回撤) are mined as their own "
        "conditions. It was first scored as vague_criterion, which turned every "
        "otherwise-clean request that merely said the word into refusal gold - a "
        "downward push, and this report is only sound if every approximation "
        "pushes the other way. Consequence: GAP-VAGUE-CRITERION now has no "
        "detector at all; see UNDETECTABLE_GAPS"),

    # -- universe filters: cross-sectional by construction ------------------
    ("universe_filter", "volume"): ProxySubject(
        "quote_volume_24h", "compare", CROSS, "coarse", "as ('threshold','volume')"),
    ("universe_filter", "quote_volume"): ProxySubject(
        "quote_volume_24h", "compare", CROSS, "exact", "universe.filter_quote_volume"),
    ("universe_filter", "liquidity"): ProxySubject(
        "spread_liquidity", "compare", CROSS, "exact",
        "流動性/深度 is the book, and universe.augment_with_spread now joins "
        "bookTicker for the whole market — so this scores EXPRESSIBLE, where it "
        "used to be the main source of GAP-SPREAD-DEPTH. It stays mapped to "
        "spread_liquidity rather than to quote_volume_24h because the request is "
        "about the book: turnover was always the proxy, and reclassifying the "
        "subject would hide that the answer changed"),
    ("universe_filter", "market_cap"): ProxySubject(
        "market_cap", "compare", CROSS, "exact", "no market-cap source"),
    ("universe_filter", "exclude"): ProxySubject(
        "symbol_blacklist", "exclude", CROSS, "coarse",
        "the miner sees 排除/剔除 but not the target. Scored expressible because "
        "every plausible target is (symbols, sector tags, contract type, quote "
        "suffix) — generous, and the direction is stated"),
    ("universe_filter", "gainers"): ProxySubject(
        "price_change_24h", "top_k", CROSS, "exact", "universe.top_gainers"),
    ("universe_filter", "losers"): ProxySubject(
        "price_change_24h", "top_k", CROSS, "exact", "universe.top_losers"),
    ("universe_filter", "whitelist"): ProxySubject(
        "symbol_whitelist", "require", CROSS, "exact", "universe.only_symbols"),
    ("universe_filter", "blacklist"): ProxySubject(
        "symbol_blacklist", "exclude", CROSS, "exact", "universe.exclude_symbols"),
    ("universe_filter", "screen"): _drop(
        "篩選/掃描/screen is the request's verb, not a criterion. A row whose only "
        "mined condition is this one has nothing to grade a spec against"),
    ("universe_filter", "full_market"): _drop(
        "全市場 is the default universe (fetch_perpetual_universe), so it adds no "
        "requirement to satisfy"),

    # -- named instruments --------------------------------------------------
    ("asset", "asset"): ProxySubject(
        "symbol_whitelist", "require", CROSS, "coarse",
        "in a selection request this is universe.only_symbols; in a trade request "
        "it is data.symbol, which is trivially expressible too. One mapping "
        "covers both because both are expressible"),
}

#: Indicator names the miner recognises, all scored against the one table row
#: that rules indicators — with the frame taken from the request, which is where
#: the whole Supertrend accident lives: expressible on a bar series, a withheld
#: proxy in the cross-section.
_INDICATOR_EXACT = (
    "rsi", "macd", "ema", "sma", "ma", "bollinger", "atr", "supertrend", "adx",
    "breakout", "breakdown", "cross_up", "cross_down", "overbought", "oversold",
    # verified present in cyqnt_trd.blocks.indicators / .conditions at
    # TABLE_DERIVED_FROM: stochastic (kdj), vwap, obv, cci, ichimoku,
    # macd_bullish_divergence (divergence)
    "kdj", "vwap", "obv", "cci", "ichimoku", "divergence",
)
_INDICATOR_OVERSTATED = {
    "fibonacci": "no Fibonacci retracement block exists in cyqnt_trd.blocks "
                 "(pivot_points and zigzag are not the same thing), so the "
                 "indicator row's 'expressible' does not reach this request",
}
for _name in _INDICATOR_EXACT:
    PROXY_MAP[("indicator", _name)] = ProxySubject(
        "technical_indicator", "compare", None, "exact",
        "a block for this indicator exists; the frame decides the verdict")
for _name, _why in _INDICATOR_OVERSTATED.items():
    PROXY_MAP[("indicator", _name)] = ProxySubject(
        "technical_indicator", "compare", None, "overstates", _why)


#: Gaps this pass structurally cannot count, and why. The miner has eight
#: condition families and none of them looks for these subjects, so a request
#: that hits one is scored as if it did not.
#:
#: This is the single most important caveat on the gap ranking: it covers 5 of
#: the table's 16 gap ids. The 11 below are not rare — "which coins got
#: liquidated hardest", "tell me when BTC breaks 70k", "coins down 50% from their
#: yearly high", "close my position" are all ordinary requests — they are
#: *invisible*. So every refusal-gold count here is a FLOOR and every
#: expressible count is a CEILING, which is the same direction as the rest of the
#: report but for a different reason: recall, not charity.
UNDETECTABLE_GAPS: dict[str, str] = {
    "GAP-VAGUE-CRITERION":
        "'coins about to pump', 'good fundamentals'. A regex cannot recognise the "
        "absence of a criterion, and the table calls this the most dangerous "
        "class precisely because mentions or turnover LOOK like an answer",
    "GAP-NEWS-EVENT-TEXT":
        "no miner family looks for news / listing announcements / headlines",
    "GAP-ALERT-NOTIFY":
        "no miner family looks for 'notify me when' - and this one is a whole "
        "product shape, not a column",
    "GAP-ACCOUNT-OPS":
        "no miner family looks for balance / close-my-position / set-leverage-now",
    "GAP-HISTORICAL-WINDOW":
        "no miner family looks for 'N-day high' or '% off its yearly high'; the "
        "timeframe family catches intervals, not lookback windows",
    "GAP-OI-CROSS-SECTION":
        "no miner family looks for open interest",
    "GAP-LIQUIDATION-CROSS-SECTION":
        "no miner family looks for liquidations (baocang)",
    "GAP-LONG-SHORT-RATIO":
        "no miner family looks for the long/short account ratio",
    "GAP-ONCHAIN-CONCENTRATION":
        "no miner family looks for holder distribution; chain data is not in the "
        "repo's sources either, so this one is doubly invisible",
    "GAP-CONTRACT-META":
        "vacant in the capability table as of TABLE_DERIVED_FROM (the contract "
        "metadata blocks landed), so nothing should reach it",
    "GAP-SECTOR-LABEL":
        "vacant, as above",
    # ASCII only, like every other value here: these strings are rendered into
    # the markdown report and test_rendered_markdown_is_pure_ascii is the last
    # line of defence on the privacy rule, so an em dash fails the whole render.
    "GAP-SPREAD-DEPTH":
        "the CROSS-SECTIONAL half landed (universe.augment_with_spread joins "
        "bookTicker for all 727 symbols), so ('universe_filter','liquidity') now "
        "scores expressible and no longer reaches this gap. What still carries it "
        "is the per_symbol_series row: per-BAR microstructure, order imbalance on "
        "a kline series, which no miner family looks for. So the remaining half is "
        "invisible rather than absent",
}


def _reachable_gaps() -> set[str]:
    """Gap ids this module's map can actually produce, computed not asserted.

    Derived by looking up every mapping in every frame it can be scored in, so
    that a change to either the map or the capability table shows up as a
    changed set rather than as a quietly shorter ranking.
    """
    reachable: set[str] = set()
    synthetic = (("multi_timeframe", "resonance", SERIES),
                 ("compound_select_then_trade", "execute", "*"))
    keys = [(row.cap_subject, row.cap_operator, row.scope)
            for row in PROXY_MAP.values() if row.fidelity != "non_condition"]
    for subject, operator, scope in list(keys) + list(synthetic):
        for frame in ((scope,) if scope is not None else (CROSS, SERIES)):
            row = cap.lookup(subject, frame, operator)
            if row.gap_id:
                reachable.add(row.gap_id)
    return reachable


def _assert_gap_coverage_is_declared() -> None:
    """Fail if a gap is neither reachable nor declared undetectable.

    Without this, a mapping change that stops producing a gap turns into a
    ranking with a missing row and no sign that anything happened - and a missing
    row reads as "we do not need that block".
    """
    reachable = _reachable_gaps()
    overlap = reachable & set(UNDETECTABLE_GAPS)
    if overlap:
        raise AssertionError(
            "gap(s) %s are declared undetectable but the proxy map does reach "
            "them; remove them from UNDETECTABLE_GAPS" % sorted(overlap))
    unaccounted = cap.GAP_IDS - reachable - set(UNDETECTABLE_GAPS)
    if unaccounted:
        raise AssertionError(
            "gap(s) %s are neither reachable from PROXY_MAP nor listed in "
            "UNDETECTABLE_GAPS. Every gap must be one or the other, or the "
            "priority ranking silently drops a need." % sorted(unaccounted))


def _assert_map_is_sane() -> None:
    """Fail at import if the map and the table disagree about what exists.

    Two ways this goes wrong silently otherwise: an ``unruled`` subject that the
    table has since started ruling on (the report would keep calling a known
    answer unknown), and a non-unruled mapping whose subject is absent from the
    table (every case using it would be shelved and the funnel would blame the
    corpus for a typo here).
    """
    ruled = set(cap.subjects())
    for subject in UNRULED_SUBJECTS:
        if subject in ruled:
            raise AssertionError(
                "%r is listed as unruled but the capability table now rules it; "
                "move it into PROXY_MAP with a real fidelity" % (subject,))
    for key, row in PROXY_MAP.items():
        if row.fidelity == "non_condition":
            continue
        if row.fidelity == "unruled":
            if row.cap_subject not in UNRULED_SUBJECTS:
                raise AssertionError(
                    "%s maps to %r with fidelity 'unruled' but that subject is "
                    "not in UNRULED_SUBJECTS" % (key, row.cap_subject))
            continue
        if row.cap_subject not in ruled:
            raise AssertionError(
                "%s maps to %r, which the capability table does not carry"
                % (key, row.cap_subject))


_assert_map_is_sane()
_assert_gap_coverage_is_declared()


# ---------------------------------------------------------------------------
# Mined condition -> capability condition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mapped:
    """A capability condition plus the provenance of the guess behind it."""

    condition: cap.Condition
    fidelity: str
    source: str       # the "family/subject" it came from


def _timeframe_conditions(intervals: list[str], index: int) -> list[Mapped]:
    """Timeframes, decided per row rather than per condition.

    One stated interval is ``data.interval``, a spec key. Two or more is a
    different request entirely — "Supertrend bearish on H4 *and* H1 *and* M15" —
    and that is ``multi_timeframe``, which the table rules a withheld proxy at
    best. Scoring three intervals as three independent spec keys would turn the
    single most expensive shape in this corpus into the cheapest.
    """
    distinct = sorted(set(intervals))
    if not distinct:
        return []
    if len(distinct) >= 2:
        return [Mapped(
            cap.Condition(subject="multi_timeframe", operator="resonance",
                          scope=SERIES, value=len(distinct), quantified=True,
                          id="c%d" % index),
            "coarse", "timeframe/interval[multi]")]
    return [Mapped(
        cap.Condition(subject="bar_interval", operator="equals", scope="*",
                      value=distinct[0], quantified=True, id="c%d" % index),
        "unruled", "timeframe/interval")]


def map_condition(cond: dict, index: int, scope: str) -> Mapped | None:
    """One mined condition as a capability condition, or ``None`` if dropped.

    Raises on an unmapped ``(family, subject)`` pair. A new miner family that
    quietly produced no capability condition would make every case holding it
    look *more* convertible, which is backwards.
    """
    key = (cond["family"], cond["subject"])
    row = PROXY_MAP.get(key)
    if row is None:
        raise KeyError(
            "no proxy mapping for %s. Add it to PROXY_MAP (or mark it "
            "non_condition); leaving it out inflates every count in the report"
            % (key,))
    if row.fidelity == "non_condition":
        return None
    return Mapped(
        cap.Condition(
            subject=row.cap_subject,
            operator=row.cap_operator,
            scope=row.scope if row.scope is not None else scope,
            value=cond.get("value"),
            polarity="exclude" if cond.get("polarity") == "exclude" else "require",
            quantified=cond.get("value") is not None,
            id="c%d" % index,
        ),
        row.fidelity,
        "%s/%s" % key,
    )


def map_conditions(record: dict, scope: str) -> list[Mapped]:
    """Every mined condition of one row, in one frame.

    ``spec_shape == "both"`` adds a condition the miner cannot see: the request
    asks for a screen *and* per-bar execution, and one spec is either
    ``selection:`` or ``signals:``. ``classify_request`` already refuses these, so
    the refusal is part of the row's truth and not an artifact of this pass.
    """
    mapped: list[Mapped] = []
    intervals: list[str] = []
    for index, cond in enumerate(record["conditions"]):
        if cond["family"] == "timeframe":
            intervals.append(str(cond.get("value")))
            continue
        item = map_condition(cond, index, scope)
        if item is not None:
            mapped.append(item)
    mapped.extend(_timeframe_conditions(intervals, len(record["conditions"])))
    if record["spec_shape"] == "both":
        mapped.append(Mapped(
            cap.Condition(subject="compound_select_then_trade", operator="execute",
                          scope="*", id="c_compound"),
            "exact", "spec_shape/both"))
    return mapped


# ---------------------------------------------------------------------------
# Per-row verdict
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RowVerdict:
    """What this row could become, under the most generous honest reading."""

    scope: str
    n_mapped: int
    #: conditions the table rules expressible outright
    n_convertible: int
    #: gap ids that block it (a withheld proxy counts: plan_conversion records it
    #: unconvertible, which is the behaviour the pipeline actually has)
    gap_ids: tuple[str, ...]
    #: ``(subject, scope, operator)`` triples the table has no row for
    unknown_keys: tuple[tuple[str, str, str], ...]
    #: ``(miner source, gap_id)`` for each blocked condition. Without this a gap
    #: count says how often the pipeline stops and not what the user typed to
    #: stop it, and the two do not always agree: GAP-PER-SYMBOL-INDICATOR fires on
    #: trade-shaped rows too, via multi-timeframe resonance rather than via any
    #: indicator being unavailable.
    blockers: tuple[tuple[str, str], ...]
    fidelities: Counter
    #: no gaps and no unknowns: a spec could be written and every condition
    #: checked against a block that exists
    all_expressible_strict: bool
    #: no gaps; unknowns forgiven. The literal upper bound, and it includes rows
    #: whose conditions are ALL unknown (a lone "4h" and nothing else).
    all_expressible_optimistic: bool
    has_not_expressible: bool


def _score(record: dict, scope: str) -> RowVerdict:
    mapped = map_conditions(record, scope)
    plan = cap.plan_conversion([m.condition for m in mapped])
    unknown_keys = tuple(sorted(
        (c.subject, c.scope, c.operator) for c in plan.shelved))
    gaps = plan.gap_ids
    no_gap = not gaps
    source_of = {m.condition.id: m.source for m in mapped}
    return RowVerdict(
        scope=scope,
        n_mapped=len(mapped),
        n_convertible=len(plan.expressible),
        gap_ids=gaps,
        unknown_keys=unknown_keys,
        blockers=tuple(sorted(
            (source_of[cond.id], gap) for cond, gap in plan.unconvertible)),
        fidelities=Counter(m.fidelity for m in mapped),
        all_expressible_strict=bool(mapped) and no_gap and not unknown_keys,
        all_expressible_optimistic=bool(mapped) and no_gap,
        has_not_expressible=bool(gaps),
    )


def _rank(verdict: RowVerdict) -> tuple:
    """Order two candidate frames for the same row: fewer blockers wins.

    Ties break toward the cross-section, which is the frame the corpus's
    ambiguous rows more often mean (a screen), and — more importantly — the
    stricter of the two for indicators, so the tie-break does not hand out the
    Supertrend proxy for free.
    """
    return (len(verdict.gap_ids), len(verdict.unknown_keys),
            0 if verdict.scope == CROSS else 1)


def plan_row(record: dict) -> RowVerdict:
    """The row's verdict, under the frame most favourable to it.

    A ``selection`` or ``trade`` row has exactly one frame and gets it. An
    ``unclear`` row has none — ``classify_request`` returned ambiguous — and a
    ``both`` row needs two at once, which no spec can be. For those the row is
    scored in each frame and the better result is kept, so the number stays an
    upper bound instead of depending on a coin toss.
    """
    scopes = SHAPE_SCOPE[record["spec_shape"]]
    return min((_score(record, scope) for scope in scopes), key=_rank)


# ---------------------------------------------------------------------------
# Re-mining the full corpus (tier D and fragments included)
# ---------------------------------------------------------------------------

#: The fields ``mine.build_report`` reads. Rebuilding exactly these lets the
#: whole Part B funnel be recomputed and compared byte for byte, which is the
#: check that this module's population is the same one Part B measured.
def remine(csv_path: Path, limit: int | None = None) -> list[dict]:
    """Every row of the corpus as a minimal record, using the miner's own passes.

    ``candidates.jsonl`` holds tier A/B/C non-fragments only — correctly, that is
    what a candidate is — but the funnel has to account for the other 25,434 rows
    too. Rather than a second implementation of the miner, this calls
    :func:`mine.analyse_text` and :func:`mine.cluster_near_duplicates` and keeps
    only the repo-safe fields.
    """
    rows = mine.read_rows(csv_path, limit)
    analyses: dict[str, dict] = {}
    per_row: list[dict] = []
    for row in rows:
        key = f"{row['first_query']}\x1f{row['user_text_excerpt']}"
        analysis = analyses.get(key)
        if analysis is None:
            analysis = mine.analyse_text(row["first_query"], row["user_text_excerpt"])
            analyses[key] = analysis
        per_row.append(analysis)

    unique_texts: dict[str, int] = {}
    for analysis in per_row:
        unique_texts.setdefault(analysis["cluster_text"], len(unique_texts))
    texts = [""] * len(unique_texts)
    for text, index in unique_texts.items():
        texts[index] = text
    cluster_root = mine.cluster_near_duplicates(texts)
    root_members: dict[int, list[str]] = defaultdict(list)
    for text, index in unique_texts.items():
        root_members[cluster_root[index]].append(mine.sha256_hex(text))
    cluster_id = {root: mine.short_hash(min(members), "dup_")
                  for root, members in root_members.items()}
    cluster_rows: Counter = Counter(
        cluster_id[cluster_root[unique_texts[a["cluster_text"]]]] for a in per_row)

    records: list[dict] = []
    for row, analysis in zip(rows, per_row):
        dup_cluster_id = cluster_id[cluster_root[unique_texts[analysis["cluster_text"]]]]
        preset = (row["preset_case"] or "").strip()
        record = {
            "canon_sha256": analysis["canon_sha256"],
            "lang": row["lang"],
            "preset_case": preset,
            "dup_cluster_id": dup_cluster_id,
            "dup_count": cluster_rows[dup_cluster_id],
            "split_group_key": (f"preset:{preset}" if preset
                                else f"dup:{dup_cluster_id}"),
            "families": analysis["families"],
            "n_families": analysis["n_families"],
            "n_conditions": analysis["n_conditions"],
            "conditions": [mine.repo_condition(c) for c in analysis["conditions"]],
            "tier": analysis["tier"],
            "spec_shape": analysis["spec_shape"],
            "spec_shape_base": analysis["spec_shape_base"],
            "is_continuation_fragment": analysis["is_continuation_fragment"],
            "fragment_reason": analysis["fragment_reason"],
            "leading_chatter": analysis["leading_chatter"],
        }
        record["is_candidate"] = (
            not record["is_continuation_fragment"] and record["tier"] != "D")
        mine.assert_repo_safe(record)
        records.append(record)
    return records


def verify_against_mine(records: list[dict], funnel_path: Path) -> None:
    """Fail unless the re-mined population reproduces Part B's funnel exactly.

    Not belt-and-braces: if this module's rows differed from the ones
    ``candidates.jsonl`` was built from, every ratio below would be measured on a
    different corpus than the one the dataset will be cut from, and nothing
    downstream would ever notice.
    """
    if not funnel_path.exists():
        raise FileNotFoundError(
            "%s is missing; run tools.nl2yaml.mine first so this pass can be "
            "checked against it" % funnel_path)
    published = json.loads(funnel_path.read_text(encoding="ascii"))
    # Round-tripped, not compared in memory: mine.build_report keys
    # n_conditions_histogram on ints and JSON keys are strings, so a direct
    # comparison reports a mismatch on every run and the check would be muted
    # rather than fixed.
    recomputed = json.loads(json.dumps(mine.build_report(records),
                                       ensure_ascii=True, sort_keys=True))
    mismatches = []
    for key, expected in published.items():
        if key == "salt_fingerprint":     # not recomputed: no salt is read here
            continue
        got = recomputed.get(key)
        if got != expected:
            mismatches.append("%s: mine.py=%r measure.py=%r" % (key, expected, got))
    if mismatches:
        raise AssertionError(
            "re-mined corpus does not reproduce %s:\n  - %s"
            % (funnel_path, "\n  - ".join(mismatches)))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

TIERS = ("A", "B", "C", "D")
SHAPES = ("selection", "trade", "both", "unclear")


def table_staleness() -> str:
    """"ok", or the reason the capability table no longer matches ``blocks``.

    Recorded in the report rather than raised. The check exists to catch the
    project's recurring failure - a block lands, a row still says
    ``not_expressible``, and the pipeline keeps refusing a request it can now
    serve - and that failure is exactly what a parallel workstream editing
    ``cyqnt_trd/blocks`` produces mid-measurement. Raising here would make the
    corpus unmeasurable whenever anyone touches a block; swallowing it would
    publish verdicts known to be stale. So it is captured verbatim, printed at the
    top of the report, and left for the reader to weigh against the gap list.
    """
    try:
        cap.assert_table_is_current()
    except AssertionError as exc:
        return str(exc)
    return "ok"


def _counts(rows: Iterable[dict]) -> dict[str, int]:
    rows = list(rows)
    return {
        "rows": len(rows),
        "unique_canon": len({r["canon_sha256"] for r in rows}),
        "split_groups": len({r["split_group_key"] for r in rows}),
    }


def _bucket(rows: list[dict], verdicts: dict[int, RowVerdict]) -> dict[str, Any]:
    """The headline counts for one slice, each in rows/unique/groups.

    ``split_groups`` is the number that matters for a training set: 48% of this
    corpus is a repeated query and the biggest single group is 1,507 rows, so
    counting rows would report memorisation capacity, not example count.

    ``upper_bound_convertible`` exists because the literal upper bound is too
    literal: a row whose only mined condition is a bare "4h" has no gap and no
    expressible condition either, and calling that a trainable spec would be the
    proxy trap in a new costume. It is the same bound with "and at least one
    condition the table rules expressible" added.
    """
    strict, optimistic, convertible, refusal, no_condition = [], [], [], [], []
    overstated_free = []
    for row in rows:
        verdict = verdicts[id(row)]
        if verdict.n_mapped == 0:
            no_condition.append(row)
            continue
        if verdict.all_expressible_strict:
            strict.append(row)
        if verdict.all_expressible_optimistic:
            optimistic.append(row)
            if verdict.n_convertible:
                convertible.append(row)
            if not verdict.fidelities.get("overstates"):
                overstated_free.append(row)
        if verdict.has_not_expressible:
            refusal.append(row)
    return {
        "total": _counts(rows),
        "no_mappable_condition": _counts(no_condition),
        "upper_bound": _counts(optimistic),
        "upper_bound_convertible": _counts(convertible),
        "upper_bound_minus_overstated": _counts(overstated_free),
        "strict": _counts(strict),
        "refusal_gold": _counts(refusal),
    }


def build_measure_report(records: list[dict]) -> dict[str, Any]:
    kept = [r for r in records if not r["is_continuation_fragment"]]
    verdicts = {id(r): plan_row(r) for r in kept}

    report: dict[str, Any] = {
        "measure": MEASURE_VERSION,
        "miner": mine.MINER_VERSION,
        "capability_table_commit": cap.TABLE_DERIVED_FROM["commit"],
        "capability_table_probed_on": cap.TABLE_DERIVED_FROM["probed_on"],
        "capability_table_staleness": table_staleness(),
        "population": {
            "total_rows": len(records),
            "continuation_fragments": len(records) - len(kept),
            "scored_rows": len(kept),
        },
        "by_tier": {t: _bucket([r for r in kept if r["tier"] == t], verdicts)
                    for t in TIERS},
        "by_shape": {s: _bucket([r for r in kept if r["spec_shape"] == s], verdicts)
                     for s in SHAPES},
        "by_shape_tier": {
            s: {t: _bucket([r for r in kept
                            if r["spec_shape"] == s and r["tier"] == t], verdicts)
                for t in TIERS}
            for s in SHAPES},
    }

    # -- unruled (subject, scope, operator) keys, i.e. what the optimism buys --
    unknown_rows: Counter = Counter()
    unknown_groups: dict[tuple, set] = defaultdict(set)
    for row in kept:
        for key in set(verdicts[id(row)].unknown_keys):
            unknown_rows[key] += 1
            unknown_groups[key].add(row["split_group_key"])
    report["unruled_keys"] = [
        {"subject": k[0], "scope": k[1], "operator": k[2],
         "rows": v, "split_groups": len(unknown_groups[k])}
        for k, v in unknown_rows.most_common()]
    report["rows_blocked_only_by_unknown"] = _counts([
        r for r in kept
        if verdicts[id(r)].n_mapped
        and verdicts[id(r)].all_expressible_optimistic
        and not verdicts[id(r)].all_expressible_strict])

    # -- fidelity census ---------------------------------------------------
    fidelity_conditions: Counter = Counter()
    for row in kept:
        fidelity_conditions.update(verdicts[id(row)].fidelities)
    report["condition_fidelity"] = dict(fidelity_conditions.most_common())
    source_fidelity: Counter = Counter()
    for row in kept:
        for cond in row["conditions"]:
            mapping = PROXY_MAP.get((cond["family"], cond["subject"]))
            if mapping is not None:
                source_fidelity[(mapping.fidelity, "%s/%s"
                                 % (cond["family"], cond["subject"]))] += 1
    report["overstated_sources"] = [
        {"source": src, "conditions": n}
        for (fid, src), n in source_fidelity.most_common() if fid == "overstates"]

    # -- gaps, ranked ------------------------------------------------------
    report["gaps"] = build_gaps(kept, verdicts)
    report["undetectable_gaps"] = dict(sorted(UNDETECTABLE_GAPS.items()))
    report["detectable_gap_ids"] = sorted(_reachable_gaps())

    # -- the one approximation that pushes DOWNWARD, quantified ------------
    #: Rows whose only blocker is "the conversation mentioned two or more
    #: intervals". The miner reads the whole excerpt, so "let us look at 1h ...
    #: now 4h" is indistinguishable from "aligned on 1h AND 4h", and only the
    #: second is a resonance request. Scoring them as resonance moves the row out
    #: of the upper bound, which is the wrong direction for a bound - so the
    #: number that would flip back is stated instead of being argued about.
    report["multi_timeframe_only_blocker"] = {
        s: _counts([r for r in kept
                    if r["spec_shape"] == s
                    and verdicts[id(r)].blockers
                    and all(source == "timeframe/interval[multi]"
                            for source, _gap in verdicts[id(r)].blockers)])
        for s in SHAPES}

    # -- quantification: a level-5 gold needs a checkable number -----------
    #: Families whose condition is unusable without a number the USER stated.
    #: ``indicator`` is absent on purpose ("RSI 金叉" is a complete rule with no
    #: number in it); ``risk`` and ``rank_topn`` are here because "設個止損" and
    #: "幫我排名" force the annotator to pick the percentage or the N, and a gold
    #: spec whose stop was invented by the annotator teaches the model to invent
    #: stops. 14,650 of the corpus's 16,872 risk mentions carry no value at all.
    needs_value = ("risk", "rank_topn")

    def fully_quantified(row: dict) -> bool:
        return all(cond.get("value") is not None
                   for cond in row["conditions"]
                   if cond["family"] in needs_value)

    report["quantified_families"] = list(needs_value)
    report["risk_conditions_without_value"] = sum(
        1 for r in kept for c in r["conditions"]
        if c["family"] in needs_value and c.get("value") is None)
    report["upper_bound_quantified"] = {
        s: _counts([r for r in kept
                    if r["spec_shape"] == s
                    and verdicts[id(r)].all_expressible_optimistic
                    and verdicts[id(r)].n_convertible
                    and fully_quantified(r)])
        for s in SHAPES}

    # -- preset cards ------------------------------------------------------
    preset_rows: Counter = Counter(
        r["preset_case"] for r in records if r["preset_case"])
    preset_detail = []
    for name, rows in preset_rows.most_common(20):
        subset = [r for r in records if r["preset_case"] == name]
        scored = [r for r in subset if not r["is_continuation_fragment"]]
        preset_detail.append({
            "preset_case": name,
            "dup_count_rows": rows,
            "unique_canon": len({r["canon_sha256"] for r in subset}),
            "tier_rows": dict(Counter(r["tier"] for r in subset).most_common()),
            "shape_rows": dict(Counter(r["spec_shape"] for r in subset).most_common()),
            "upper_bound_rows": sum(
                1 for r in scored if verdicts[id(r)].all_expressible_optimistic),
            "refusal_rows": sum(
                1 for r in scored if verdicts[id(r)].has_not_expressible),
            "gap_ids": sorted({g for r in scored for g in verdicts[id(r)].gap_ids}),
        })
    report["top_preset_case"] = preset_detail
    return report


def build_gaps(kept: list[dict], verdicts: dict[int, RowVerdict]) -> list[dict]:
    """Gaps ranked by how much of the corpus they block.

    Three numbers per gap, because they answer different questions and the first
    one alone has misled this project before:

    ``dup_weighted_count``
        raw rows blocked. This is demand as users produced it, duplicates and
        all, and it is what "how often do we hit this" means.
    ``distinct_requests``
        split groups blocked. A gap that looks huge because one preset card fired
        500 times is not 500 problems, and ``largest_group_share`` says so.
    ``rows_unlocked_if_closed``
        rows whose *only* blockers are this gap. Closing a gap that is always
        accompanied by another buys nothing, and raw frequency cannot see that.
    """
    per_gap_rows: dict[str, list[dict]] = defaultdict(list)
    unlocked: Counter = Counter()
    unlocked_groups: dict[str, set] = defaultdict(set)
    for row in kept:
        verdict = verdicts[id(row)]
        for gap in verdict.gap_ids:
            per_gap_rows[gap].append(row)
        if len(verdict.gap_ids) == 1:
            # The only thing in the way. ``unknown`` keys are forgiven here for
            # the same reason as in the headline: they are table omissions, not
            # missing blocks, and a gap should not look cheap because some other
            # subject has not been ruled on yet.
            unlocked[verdict.gap_ids[0]] += 1
            unlocked_groups[verdict.gap_ids[0]].add(row["split_group_key"])

    gaps = []
    for gap, rows in per_gap_rows.items():
        group_sizes = Counter(r["split_group_key"] for r in rows)
        largest, largest_rows = group_sizes.most_common(1)[0]
        gaps.append({
            "gap_id": gap,
            "detectable_by_this_pass": True,
            "undetectable_reason": None,
            "dup_weighted_count": len(rows),
            "distinct_requests": len(group_sizes),
            "unique_canon": len({r["canon_sha256"] for r in rows}),
            "rows_unlocked_if_closed": unlocked[gap],
            "distinct_requests_unlocked_if_closed": len(unlocked_groups[gap]),
            "largest_group_share": round(largest_rows / len(rows), 4),
            "largest_split_group_key": largest,
            "tier_rows": {t: sum(1 for r in rows if r["tier"] == t) for t in TIERS},
            "shape_rows": {s: sum(1 for r in rows if r["spec_shape"] == s)
                           for s in SHAPES},
            "blocking_sources": dict(Counter(
                source for r in rows
                for source, blocked_gap in verdicts[id(r)].blockers
                if blocked_gap == gap).most_common(6)),
            "co_occurring_gaps": dict(Counter(
                other for r in rows for other in verdicts[id(r)].gap_ids
                if other != gap).most_common(5)),
        })
    # Every gap the table carries appears in the file, ranked or not. A gap
    # missing from the ranking because no regex looks for it reads exactly like a
    # gap nobody needs, and that misreading is expensive: it is a decision not to
    # build a block.
    for gap in sorted(cap.GAP_IDS - set(per_gap_rows)):
        gaps.append({
            "gap_id": gap,
            "detectable_by_this_pass": False,
            "undetectable_reason": UNDETECTABLE_GAPS.get(
                gap, "reachable from the proxy map but zero rows hit it"),
            "dup_weighted_count": 0,
            "distinct_requests": 0,
            "unique_canon": 0,
            "rows_unlocked_if_closed": 0,
            "distinct_requests_unlocked_if_closed": 0,
            "largest_group_share": 0.0,
            "largest_split_group_key": None,
            "tier_rows": {t: 0 for t in TIERS},
            "shape_rows": {s: 0 for s in SHAPES},
            "blocking_sources": {},
            "co_occurring_gaps": {},
        })
    gaps.sort(key=lambda g: (-g["dup_weighted_count"], g["gap_id"]))
    return gaps


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _row3(counts: dict[str, int]) -> str:
    return "%d / %d / %d" % (counts["rows"], counts["unique_canon"],
                             counts["split_groups"])


def render_markdown(report: dict, funnel: dict) -> str:
    out: list[str] = []
    w = out.append
    w("# Gate0 funnel: how much trainable gold is in the 51,595-row corpus")
    w("")
    w("Generated by `tools/nl2yaml/measure.py` (%s) over the whole corpus, "
      "**no LLM in the loop**." % report["measure"])
    w("Miner: `%s`. Capability table: commit `%s`, probed %s."
      % (report["miner"], report["capability_table_commit"][:12],
         report["capability_table_probed_on"]))
    w("")
    if report["capability_table_staleness"] != "ok":
        w("> **The capability table is stale as of this run.** "
          "`capability.assert_table_is_current()` says:")
        w(">")
        for line in report["capability_table_staleness"].splitlines():
            # backslashreplace, not a strip: the message is evidence, and losing a
            # character out of it silently is how evidence stops being evidence.
            w("> " + line.encode("ascii", "backslashreplace").decode("ascii"))
        w(">")
        w("> Read every `not_expressible` verdict below against that. A gap ranked "
          "in section 5 may already be closed. The gaps this pass can see at all "
          "are `%s` - check the block names in the message above against that "
          "list before trusting the ranking, and re-run once the table is "
          "re-derived." % "`, `".join(report["detectable_gap_ids"]))
        w("")
    dup_share = 1 - (funnel["split_groups"]["n_groups"] / funnel["funnel"]["total_rows"])
    w("Every count is written **rows / unique-canon-texts / split-groups**. "
      "Use split-groups as the example count: %.0f%% of the corpus collapses into "
      "a group with another row, the largest single group is %d rows, and a "
      "random split over rows measures memorisation."
      % (100 * dup_share, funnel["split_groups"]["max_group_rows"]))
    w("")
    w("## 0. Read this before using any number below")
    w("")
    w("`capability.plan_conversion` needs conditions in its own vocabulary, and "
      "producing those is A1's job (an LLM). This pass substitutes a hand-written "
      "map from the miner's regex `(family, subject)` pairs onto capability "
      "subjects, so **every number here is an UPPER BOUND**. The true yield is "
      "strictly lower, for four reasons that all push the same way:")
    w("")
    w("1. **Miner recall is a floor.** The regexes find a subset of a request's "
      "conditions. A row scored \"all expressible\" can still contain an unmined "
      "`not_expressible` one - which is exactly the 'exclude tradfi' accident: "
      "the blocking condition was in the text and not in the machine's hands.")
    w("2. **The map is coarse in the generous direction.** `volume` (a COIN count) "
      "is scored against `quoteVolume` (quote currency); `exclude` (the verb) is scored as "
      "`symbol_blacklist` without knowing *what* is excluded, because every "
      "plausible target happens to be expressible.")
    w("3. **`unknown` is forgiven in the headline.** %d rows / %d groups are "
      "\"upper bound\" only because a `(subject, scope, operator)` the table does "
      "not rule on was treated as expressible. The `strict` column removes them; "
      "section 5 lists exactly which keys they are."
      % (report["rows_blocked_only_by_unknown"]["rows"],
         report["rows_blocked_only_by_unknown"]["split_groups"]))
    w("4. **Scope is guessed for ambiguous rows.** `classify_request` returns "
      "ambiguous for %d rows. Those are scored in both frames and the better "
      "result kept - upward again."
      # shape_rows / shape_base_rows come from a Counter, so a kind with no
      # rows is absent rather than zero. Absence means zero here; it is not a
      # degraded reading of a present value.
      % funnel["shape_base_rows"].get("ambiguous", 0))
    w("")
    w("A fifth, separate discount: expressible is not gradeable. See section 7.")
    w("")
    w("Two things push the other way and are stated so they can be argued with:")
    w("")
    w("* **The gap ranking is blind to 11 of the table's 16 gap ids** (section 5). "
      "Requests that hit an undetectable gap are scored as clean, so refusal-gold "
      "counts are floors.")
    w("* **Two intervals in one conversation are scored as a resonance request.** "
      "The miner reads the whole excerpt, so \"look at 1h ... now 4h\" is "
      "indistinguishable from \"aligned on 1h AND 4h\", and only the second needs "
      "the missing capability. That pushes rows OUT of the upper bound, which is "
      "the wrong direction, so the size of the effect is stated rather than "
      "argued: %d selection / %d trade rows (%d / %d groups) are blocked by "
      "nothing else and would flip back into the bound if the heuristic is "
      "judged wrong."
      % (report["multi_timeframe_only_blocker"]["selection"]["rows"],
         report["multi_timeframe_only_blocker"]["trade"]["rows"],
         report["multi_timeframe_only_blocker"]["selection"]["split_groups"],
         report["multi_timeframe_only_blocker"]["trade"]["split_groups"]))
    w("")

    # -- 1. funnel ---------------------------------------------------------
    f = funnel["funnel"]
    w("## 1. Funnel")
    w("")
    w("```")
    w("total rows                                %7d" % f["total_rows"])
    w("- continuation fragments                  %7d   %s"
      % (f["continuation_fragments"], funnel["fragment_reasons"]))
    w("= after fragment filter                   %7d" % f["after_fragment_filter"])
    w("  unique canon texts                      %7d"
      % f["unique_canon_after_fragment_filter"])
    w("= candidates (tier A/B/C)                 %7d   unique %d   groups %d"
      % (f["candidates_rows"], f["candidates_unique_canon"],
         f["candidates_split_groups"]))
    w("  kept despite leading chatter            %7d" % funnel["leading_chatter_but_kept"])
    w("```")
    w("")
    w("| tier | rows | unique canon | split groups |")
    w("| --- | --- | --- | --- |")
    for tier in TIERS:
        w("| %s | %d | %d | %d |"
          % (tier, funnel["tier_rows"][tier], funnel["tier_unique_canon"][tier],
             funnel["tier_split_groups"][tier]))
    w("")
    w("Tier = how many machine-checkable condition families the whole "
      "conversation carries (A >=3, B 2, C 1, D 0). Tier D holds no mined "
      "condition at all, so it cannot yield a spec-shaped gold; it is kept in "
      "the tables below only so the denominators stay honest.")
    w("")

    # -- 2. shape x tier ---------------------------------------------------
    w("## 2. Spec shape x tier")
    w("")
    w("| shape | A | B | C | D | total rows |")
    w("| --- | --- | --- | --- | --- | --- |")
    for shape in SHAPES:
        cells = " | ".join(
            "%d / %d" % (funnel["shape_by_tier_rows"][shape][t],
                         funnel["shape_by_tier_unique_canon"][shape][t])
            for t in TIERS)
        w("| %s | %s | %d |"
          % (shape, cells, funnel["shape_rows"].get(shape, 0)))
    w("")
    w("(cells are rows / unique canon)")
    w("")
    w("`classify_request` base kinds: %s. `both` = the request screens a universe "
      "*and* states a per-bar entry rule; one spec cannot be both, so those "
      "%d rows are refusal gold by construction, not by any judgement of this "
      "pass." % (funnel["shape_base_rows"], funnel["shape_rows"].get("both", 0)))
    w("")

    # -- 3 & 4. per-tier expressibility ------------------------------------
    w("## 3 & 4. Per tier: all-expressible (level-5 ceiling) vs any-not-expressible")
    w("")
    w("Q3 is the `upper bound` column: no condition in the row carries a gap, so "
      "a level-5 (every-condition-verified) gold spec is *possible*. Q4 is "
      "`refusal gold`: at least one condition is `not_expressible` (or is a "
      "withheld `proxy_only`, which `plan_conversion` records as unconvertible), "
      "so the only honest gold is a refusal naming the gap.")
    w("")
    w("| tier | scored rows | upper bound | % of tier | + has >=1 expressible "
      "condition | strict (unknown blocks) | refusal gold | % of tier | no "
      "mappable condition |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for tier in TIERS:
        b = report["by_tier"][tier]
        total = b["total"]["rows"] or 1
        w("| %s | %s | %s | %.1f%% | %s | %s | %s | %.1f%% | %s |"
          % (tier, _row3(b["total"]), _row3(b["upper_bound"]),
             100 * b["upper_bound"]["rows"] / total,
             _row3(b["upper_bound_convertible"]), _row3(b["strict"]),
             _row3(b["refusal_gold"]), 100 * b["refusal_gold"]["rows"] / total,
             _row3(b["no_mappable_condition"])))
    w("")
    w("The buckets do not sum to the tier: a row with zero mappable conditions is "
      "in none of them, and `upper bound` and `refusal gold` are mutually "
      "exclusive by construction (any gap disqualifies the row from the bound).")
    w("")
    w("Same table by shape, which is what actually matters for training (a "
      "selection spec and a trade spec are different tasks):")
    w("")
    w("| shape | scored rows | upper bound | % | + >=1 expressible | minus "
      "overstated | strict | refusal gold | % |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for shape in SHAPES:
        b = report["by_shape"][shape]
        total = b["total"]["rows"] or 1
        w("| %s | %s | %s | %.1f%% | %s | %s | %s | %s | %.1f%% |"
          % (shape, _row3(b["total"]), _row3(b["upper_bound"]),
             100 * b["upper_bound"]["rows"] / total,
             _row3(b["upper_bound_convertible"]),
             _row3(b["upper_bound_minus_overstated"]), _row3(b["strict"]),
             _row3(b["refusal_gold"]),
             100 * b["refusal_gold"]["rows"] / total))
    w("")
    w("And the full cross: upper bound / refusal gold, in **split groups**, per "
      "shape x tier. This is the cell-level answer to Q3 and Q4.")
    w("")
    w("| shape | A | B | C | D |")
    w("| --- | --- | --- | --- | --- |")
    for shape in SHAPES:
        cells = []
        for tier in TIERS:
            b = report["by_shape_tier"][shape][tier]
            cells.append("%d / %d" % (b["upper_bound"]["split_groups"],
                                      b["refusal_gold"]["split_groups"]))
        w("| %s | %s |" % (shape, " | ".join(cells)))
    w("")

    # -- 5. gaps -----------------------------------------------------------
    w("## 5. Gap priority, by `dup_weighted_count`")
    w("")
    w("This table replaces any hand-estimated block wishlist. Machine-readable "
      "copy in `gaps.jsonl`, same order.")
    w("")
    w("* `dup_weighted_count` - raw rows blocked (demand as users produced it).")
    w("* `distinct` - split groups blocked. `top group` is the share of rows in "
      "the single biggest group: a high share means one preset card, not one "
      "widespread need.")
    w("* `unlocked if closed` - rows whose ONLY blocker is this gap. **Build "
      "order should follow this column, not the first one.**")
    w("")
    w("")
    w("**Before reading the order: this ranking sees %d of the capability "
      "table's %d gap ids.** The miner has eight regex families and none of them "
      "looks for liquidations, open interest, news, alerts, account operations, "
      "lookback windows, the long/short ratio, holder concentration, or a vague "
      "criterion. Requests hitting those are scored as if they had not hit "
      "anything, so **every refusal-gold count in this report is a floor and "
      "every expressible count is a ceiling.** The undetectable eleven are listed "
      "below the table with zero counts, and they are in `gaps.jsonl` too - a gap "
      "missing from a priority list reads as a gap nobody needs."
      % (len(report["detectable_gap_ids"]), len(cap.GAP_IDS)))
    w("")
    w("| # | gap_id | dup_weighted_count | distinct | unlocked if closed "
      "(rows / groups) | top group | main shape |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for index, gap in enumerate(report["gaps"], 1):
        if not gap["detectable_by_this_pass"]:
            continue
        shape = max(gap["shape_rows"].items(), key=lambda kv: kv[1])
        w("| %d | `%s` | %d | %d | %d / %d | %.0f%% | %s (%d) |"
          % (index, gap["gap_id"], gap["dup_weighted_count"],
             gap["distinct_requests"], gap["rows_unlocked_if_closed"],
             gap["distinct_requests_unlocked_if_closed"],
             100 * gap["largest_group_share"], shape[0], shape[1]))
    w("")
    w("Unranked because this pass is blind to them, NOT because they are rare:")
    w("")
    w("| gap_id | why this pass cannot count it |")
    w("| --- | --- |")
    for gap_id, why in report["undetectable_gaps"].items():
        w("| `%s` | %s |" % (gap_id, why))
    w("")
    w("What the user actually typed to hit each gap (`miner family/subject` -> "
      "blocked condition count). Worth reading against the shape column above: "
      "`GAP-PER-SYMBOL-INDICATOR` fires on trade-shaped rows too, and there it is "
      "never the indicator that is missing - it is multi-timeframe resonance, "
      "which the repo can only feed HTF SMAs.")
    w("")
    w("| gap_id | blocking sources |")
    w("| --- | --- |")
    for gap in report["gaps"]:
        sources = ", ".join("`%s` %d" % (src, n)
                            for src, n in gap["blocking_sources"].items())
        w("| `%s` | %s |" % (gap["gap_id"], sources or "-"))
    w("")
    w("### Unruled keys (what the headline's optimism is made of)")
    w("")
    w("These are `(subject, scope, operator)` triples the capability table has no "
      "row for, so `lookup` returns `unknown`. They are **not** gaps in the "
      "blocks - they are questions Part C has not ruled on. Each is either a spec "
      "key nobody wrote a row for, or a request that is not a spec condition at "
      "all.")
    w("")
    w("| subject | scope | operator | rows | split groups |")
    w("| --- | --- | --- | --- | --- |")
    for key in report["unruled_keys"]:
        w("| `%s` | %s | %s | %d | %d |"
          % (key["subject"], key["scope"], key["operator"], key["rows"],
             key["split_groups"]))
    w("")
    w("### Conditions whose 'expressible' overstates the repo")
    w("")
    w("The capability table rules a *class* expressible; these are the specific "
      "things inside that class with no implementation behind them. Rows relying "
      "on one are counted in `upper bound` and excluded from `upper bound minus "
      "overstated`.")
    w("")
    w("| source | conditions |")
    w("| --- | --- |")
    for item in report["overstated_sources"]:
        w("| `%s` | %d |" % (item["source"], item["conditions"]))
    w("")

    # -- 6. presets --------------------------------------------------------
    w("## 6. Top 20 `preset_case` by `dup_count`")
    w("")
    w("These are template-layer targets, not model targets: one fix to a card "
      "moves every row in it. Note `black-swan-insurance` at 1 row - the cards "
      "are wildly unequal, and a per-card fix is worth its row count, not 1/15th "
      "of the corpus.")
    w("")
    w("| preset_case | rows | unique canon | tiers | shapes | upper-bound rows | "
      "refusal rows | gaps hit |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in report["top_preset_case"]:
        w("| `%s` | %d | %d | %s | %s | %d | %d | %s |"
          % (item["preset_case"], item["dup_count_rows"], item["unique_canon"],
             item["tier_rows"], item["shape_rows"], item["upper_bound_rows"],
             item["refusal_rows"],
             ", ".join(g.replace("GAP-", "") for g in item["gap_ids"]) or "-"))
    w("")

    # -- 7. the verdict ----------------------------------------------------
    w("## 7. Is this enough to fine-tune? (500-5,000 structured outputs)")
    w("")
    w("The two dialects are separate tasks with separate grammars "
      "(`selection:` ranks a universe and emits candidates; `signals:` emits "
      "per-bar entries for one declared symbol). `validate_spec` refuses a spec "
      "that is both, so a mixed training set teaches a shape that cannot exist. "
      "Counted apart:")
    w("")
    w("| dialect | scored rows | upper bound rows | **upper bound, deduped "
      "(groups)** | + >=1 expressible (groups) | minus overstated (groups) | "
      "strict (groups) | + user stated every number (groups) |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for shape in ("selection", "trade"):
        b = report["by_shape"][shape]
        w("| %s | %d | %d | **%d** | %d | %d | %d | %d |"
          % (shape, b["total"]["rows"], b["upper_bound"]["rows"],
             b["upper_bound"]["split_groups"],
             b["upper_bound_convertible"]["split_groups"],
             b["upper_bound_minus_overstated"]["split_groups"],
             b["strict"]["split_groups"],
             report["upper_bound_quantified"][shape]["split_groups"]))
    w("")
    w("The last column adds \"and the user stated every number the spec needs\" "
      "(families %s must carry a value). It is not a nicety: an expressible-but-"
      "unquantified `add a stop-loss` forces the annotator to pick the "
      "percentage, and a "
      "gold spec whose stop was invented by the annotator trains the model to "
      "invent stops. %d of the corpus's risk conditions carry no value at all."
      % (report["quantified_families"],
         report["risk_conditions_without_value"]))
    w("")
    w("`both` (%d rows) and `unclear` (%d rows) are excluded from this table on "
      "purpose: `both` is refusal gold, and `unclear` has no decided shape, so "
      "its target dialect would be the annotator's guess."
      % (report["by_shape"]["both"]["total"]["rows"],
         report["by_shape"]["unclear"]["total"]["rows"]))
    w("")
    w("Refusal gold is a third, cheaper class and it is abundant:")
    w("")
    w("| dialect | refusal gold rows | deduped (groups) |")
    w("| --- | --- | --- |")
    for shape in SHAPES:
        b = report["by_shape"][shape]
        w("| %s | %d | %d |"
          % (shape, b["refusal_gold"]["rows"], b["refusal_gold"]["split_groups"]))
    w("")
    w("A refusal is not a consolation prize. It is the output that stops the "
      "accident this dataset exists because of: the gap vocabulary is closed, so "
      "a refusal is graded by string equality against an id known before "
      "generation, and grading it costs nothing and cannot drift. Both dialects "
      "need one.")
    w("")

    # -- the verdict -------------------------------------------------------
    sel = report["by_shape"]["selection"]
    tra = report["by_shape"]["trade"]
    both = report["by_shape"]["both"]
    unclear = report["by_shape"]["unclear"]
    gap_by_id = {g["gap_id"]: g for g in report["gaps"]}
    per_symbol = gap_by_id["GAP-PER-SYMBOL-INDICATOR"]
    compound = gap_by_id["GAP-COMPOUND-SELECT-THEN-TRADE"]
    sel_spec = sel["upper_bound"]["split_groups"]
    sel_refusal = sel["refusal_gold"]["split_groups"]
    tra_spec = tra["upper_bound"]["split_groups"]

    w("### The answer")
    w("")
    w("**Trade dialect: yes, with room to spare - the problem there is selection "
      "pressure, not scarcity.** %d distinct requests clear the bound, %d survive "
      "the strict reading, and %d also have every number the user stated rather "
      "than an annotator's guess. The band is 500-5,000; the tightest of those "
      "columns is already inside it, so the trade set can be cut for quality "
      "(quantified, tier A/B, one row per split group) instead of scraped for "
      "volume, and a genuine held-out test set is affordable."
      % (tra_spec, tra["strict"]["split_groups"],
         report["upper_bound_quantified"]["trade"]["split_groups"]))
    w("")
    w("**Selection dialect: no, not as spec emission.** %d distinct requests clear "
      "the bound - and that is the ceiling, before A1's own error rate and before "
      "a human rejects the ones the miner scored generously. Realistically that "
      "lands well under 300, against a floor of 500. Adding the refusals gets the "
      "selection *task* to %d distinct targets (%d specs + %d refusals), which "
      "does clear 500, but half of that set teaches the model to decline. Train "
      "that head, and do not claim a selection spec generator on %d examples."
      % (sel_spec, sel_spec + sel_refusal, sel_spec, sel_refusal, sel_spec))
    w("")
    w("Three ways to close the selection shortfall, in order of leverage. All "
      "three are worth more than more chat data, because the corpus is not short "
      "of selection *requests* - it is short of selection requests this repo can "
      "answer:")
    w("")
    w("1. **Decide the shape of the ambiguous rows.** `classify_request` returns "
      "`ambiguous` for %d of %d rows, and that bucket holds %d upper-bound "
      "distinct requests with no dialect assigned - more than the trade dialect "
      "yields in total. Only %d rows in the whole corpus are classified "
      "`selection` at base. Sharpening the classifier (or letting A1 decide the "
      "shape and recording its choice as a field) is a cheaper move than any "
      "block, and it is where the selection examples are hiding."
      % (funnel["shape_base_rows"].get("ambiguous", 0),
         funnel["funnel"]["total_rows"], unclear["upper_bound"]["split_groups"],
         funnel["shape_base_rows"].get("selection", 0)))
    w("2. **Close `GAP-PER-SYMBOL-INDICATOR` (`universe.augment_with_indicator`).** "
      "%d distinct requests are blocked by it and %d are blocked by nothing else, "
      "so closing it alone converts them. \"Scan the market, then run the "
      "indicator on each survivor\" is the shape the selection dialect is missing, "
      "and it is the same gap the Supertrend accident came from."
      % (per_symbol["distinct_requests"],
         per_symbol["distinct_requests_unlocked_if_closed"]))
    w("3. **Decide what a compound request should produce.** %d distinct requests "
      "(%d rows) ask for a screen AND per-bar execution, and today all of them "
      "are refusals - that bucket alone is %.1fx the entire selection spec yield. "
      "They do not need a new block so much as a decision: two specs, or a "
      "declared refusal. Either answer converts them into training targets; "
      "leaving it undecided keeps the largest single block of selection-flavoured "
      "demand in the corpus unusable."
      % (compound["distinct_requests"], compound["dup_weighted_count"],
         compound["distinct_requests"] / max(sel_spec, 1)))
    w("")
    w("**One practical note on cost.** A1 does not need %d calls. Conditions are "
      "computed per unique canonical text and cases split by group, so one pass "
      "over the %d candidate split groups covers the %d candidate rows - a %.1fx "
      "saving - and it is also the only way the train/test split stays honest."
      % (funnel["funnel"]["total_rows"], funnel["funnel"]["candidates_split_groups"],
         funnel["funnel"]["candidates_rows"],
         funnel["funnel"]["candidates_rows"]
         / max(funnel["funnel"]["candidates_split_groups"], 1)))
    w("")
    w("**And one warning about what this pass cannot tell you.** Section 5 sees 5 "
      "of 16 gaps and the miner's recall is a floor, so the refusal counts above "
      "are floors and the spec counts are ceilings. The number that would settle "
      "it is the one this pass deliberately did not spend: run A1 over a "
      "stratified sample of a few hundred split groups, have a human adjudicate "
      "every condition, and compare the realised level-5 rate against the %.0f%% "
      "and %.0f%% upper bounds in the shape table. Until then, treat the trade "
      "'yes' as firm and the selection 'no' as firm, and treat everything between "
      "them as unmeasured."
      % (100 * tra["upper_bound"]["rows"] / max(tra["total"]["rows"], 1),
         100 * sel["upper_bound"]["rows"] / max(sel["total"]["rows"], 1)))
    w("")

    # -- 8. the map itself -------------------------------------------------
    w("## 8. Appendix: the proxy map, in full")
    w("")
    w("Every number above depends on these %d rows, so they are printed rather "
      "than described. `scope` = `-` means the frame comes from the request's "
      "shape; a fixed frame means the quantity only exists there. `fidelity` is "
      "defined in `measure.FIDELITIES`." % len(PROXY_MAP))
    w("")
    w("| miner family/subject | capability subject | operator | scope | fidelity |")
    w("| --- | --- | --- | --- | --- |")
    for (family, subject), row in sorted(PROXY_MAP.items()):
        w("| `%s/%s` | %s | %s | %s | %s |"
          % (family, subject,
             ("`%s`" % row.cap_subject) if row.cap_subject else "_dropped_",
             row.cap_operator, row.scope or "-", row.fidelity))
    w("| `timeframe/interval` (1 distinct) | `bar_interval` | equals | * | unruled |")
    w("| `timeframe/interval` (>=2 distinct) | `multi_timeframe` | resonance | "
      "per_symbol_series | coarse |")
    w("| `spec_shape == both` | `compound_select_then_trade` | execute | * | exact |")
    w("")
    w("Condition fidelity census over all scored rows: %s"
      % report["condition_fidelity"])
    w("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def write_outputs(report: dict, funnel: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gaps_path = out_dir / "gaps.jsonl"
    with gaps_path.open("w", encoding="ascii") as handle:
        for gap in report["gaps"]:
            payload = json.dumps(gap, ensure_ascii=True, sort_keys=True)
            handle.write(payload + "\n")

    markdown = render_markdown(report, funnel)
    # The report is statistics only. An ascii-only write is the last line of
    # defence: if a user's Chinese ever reached a count's label, this raises
    # instead of committing it to a public repo.
    (out_dir / "funnel_report.md").write_text(markdown, encoding="ascii")

    payload = json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2,
                         default=str)
    if not payload.isascii():
        raise ValueError("measure report contains non-ascii content")
    (out_dir / "measure.json").write_text(payload + "\n", encoding="ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="repo-bound output dir (counts and enum ids only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N rows only; skips the funnel cross-check")
    args = parser.parse_args(argv)

    if not args.csv.exists():
        raise FileNotFoundError("input csv not found: %s" % args.csv)
    records = remine(args.csv, args.limit)
    funnel_path = args.out / "funnel.json"
    if args.limit is None:
        verify_against_mine(records, funnel_path)
        sys.stdout.write("funnel cross-check vs %s: identical\n" % funnel_path)
    else:
        sys.stdout.write("--limit given: funnel cross-check skipped\n")
    funnel = mine.build_report(records)
    report = build_measure_report(records)
    write_outputs(report, funnel, args.out)

    for shape in ("selection", "trade"):
        bucket = report["by_shape"][shape]
        sys.stdout.write(
            "%-10s upper bound %6d rows / %5d groups   refusal %6d rows\n"
            % (shape, bucket["upper_bound"]["rows"],
               bucket["upper_bound"]["split_groups"],
               bucket["refusal_gold"]["rows"]))
    sys.stdout.write("wrote %s, %s and %s\n"
                     % (args.out / "funnel_report.md", args.out / "gaps.jsonl",
                        args.out / "measure.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
