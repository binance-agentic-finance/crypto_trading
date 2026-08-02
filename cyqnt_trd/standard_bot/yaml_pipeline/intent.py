"""What the user asked for, extracted and checked independently of the model.

Two halves, and the split is the point:

* :func:`classify_request` reads the natural-language request and nothing else.
  It decides scope (basket vs single-instrument trade), the data sources the
  request depends on, the counts/periods/percentages it names, and which end of
  the ranked column it wants. It fails closed on genuine ambiguity.
* :func:`reconcile_intent` then checks a *generated* spec against that decision.
  A generated spec cannot validate its own claim about what was asked: a trade
  YAML looks internally consistent whether or not the user wanted a basket, and
  a news-ranked selection looks fine until you notice the request was about
  funding rates.

This used to live inside ``docs/strategy_yaml_spec/demo/server.py``. It moved
here because the conversion pipeline being built under ``tools/nl2yaml`` needs
the same two halves, and a tool must not import a module out of ``docs/``. The
demo server re-exports these names, so its HTTP routes and tests are unaffected.
"""

from __future__ import annotations

import re

__all__ = [
    "IntentDecision",
    "UNSUPPORTED_SELECTION_SOURCES",
    "candidate_trade_plan",
    "classify_request",
    "generated_strategy_kind",
    "infer_strategy_kind",
    "reconcile_intent",
]

#: Cross-sectional sources a selection spec cannot express yet, so a request
#: that depends on one is refused instead of answered with something else.
#:
#: Keep this list beside the front-door classifier, but derive every entry from
#: the *running* YAML path.  Open interest used to live here because Binance has
#: no whole-market endpoint.  The runtime now deliberately fans out over an
#: already-narrowed roster via ``open_interest_snapshot`` and
#: ``universe.augment_with_open_interest``, so refusing it here would make the
#: front door less capable than the Blocks path it fronts.
UNSUPPORTED_SELECTION_SOURCES = frozenset()


class IntentDecision:
    """The independently-checkable meaning extracted before YAML generation."""

    __slots__ = (
        "kind", "evidence", "requested_count", "sources", "excluded_sources",
        "bullish_preference", "unsupported_preferences", "candidate_trade_requested",
        "long_short_min_account_pct", "long_short_min_account_operator",
        "supertrend_parameters",
        "named_symbols",
        "included_symbols", "excluded_symbols", "intervals", "market_type",
        "technical_periods", "excluded_technical_periods", "stop_pct",
        "tp_pct", "size_fraction", "directions", "excluded_directions",
        "news_metrics", "triggers", "condition_join", "indicator_names",
        "excluded_indicator_names", "rsi_thresholds", "ranking_metric",
        "score_order", "score_order_metric", "exit_type", "exit_atr_period",
        "exit_trail_mult", "excluded_exit_types",
    )

    def __init__(
        self,
        *,
        kind: str,
        evidence=(),
        requested_count=None,
        sources=frozenset(),
        excluded_sources=frozenset(),
        bullish_preference=False,
        unsupported_preferences=(),
        candidate_trade_requested=False,
        long_short_min_account_pct=None,
        long_short_min_account_operator=None,
        supertrend_parameters=(),
        named_symbols=(),
        included_symbols=(),
        excluded_symbols=(),
        intervals=(),
        market_type=None,
        technical_periods=(),
        excluded_technical_periods=(),
        stop_pct=None,
        tp_pct=None,
        size_fraction=None,
        directions=(),
        excluded_directions=(),
        news_metrics=(),
        triggers=(),
        condition_join=None,
        indicator_names=(),
        excluded_indicator_names=(),
        rsi_thresholds=(),
        ranking_metric=None,
        score_order=None,
        score_order_metric=None,
        exit_type=None,
        exit_atr_period=None,
        exit_trail_mult=None,
        excluded_exit_types=(),
    ):
        self.kind = str(kind)
        self.evidence = tuple(evidence)
        self.requested_count = requested_count
        self.sources = frozenset(sources)
        self.excluded_sources = frozenset(excluded_sources)
        self.bullish_preference = bool(bullish_preference)
        self.unsupported_preferences = tuple(unsupported_preferences)
        self.candidate_trade_requested = bool(candidate_trade_requested)
        self.long_short_min_account_pct = long_short_min_account_pct
        self.long_short_min_account_operator = long_short_min_account_operator
        self.supertrend_parameters = tuple(supertrend_parameters)
        self.named_symbols = tuple(named_symbols)
        self.included_symbols = tuple(included_symbols)
        self.excluded_symbols = tuple(excluded_symbols)
        self.intervals = tuple(intervals)
        self.market_type = market_type
        self.technical_periods = tuple(technical_periods)
        self.excluded_technical_periods = tuple(excluded_technical_periods)
        self.stop_pct = stop_pct
        self.tp_pct = tp_pct
        self.size_fraction = size_fraction
        self.directions = tuple(directions)
        self.excluded_directions = tuple(excluded_directions)
        self.news_metrics = tuple(news_metrics)
        self.triggers = tuple(triggers)
        self.condition_join = condition_join
        self.indicator_names = tuple(indicator_names)
        self.excluded_indicator_names = tuple(excluded_indicator_names)
        self.rsi_thresholds = tuple(rsi_thresholds)
        self.ranking_metric = ranking_metric
        # Which end of the ranked column the request named ("asc"/"desc"), and
        # the frame column it named it for. Both None means the user did not say,
        # and nothing downstream may pretend otherwise.
        self.score_order = score_order
        self.score_order_metric = score_order_metric
        self.exit_type = exit_type
        self.exit_atr_period = exit_atr_period
        self.exit_trail_mult = exit_trail_mult
        self.excluded_exit_types = tuple(excluded_exit_types)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "evidence": list(self.evidence),
            "requested_count": self.requested_count,
            "sources": sorted(self.sources),
            "excluded_sources": sorted(self.excluded_sources),
            "bullish_preference": self.bullish_preference,
            "unsupported_preferences": list(self.unsupported_preferences),
            "candidate_trade_requested": self.candidate_trade_requested,
            "long_short_min_account_pct": self.long_short_min_account_pct,
            "long_short_min_account_operator": self.long_short_min_account_operator,
            "supertrend_parameters": [
                {"period": period, "multiplier": multiplier}
                for period, multiplier in self.supertrend_parameters
            ],
            "named_symbols": list(self.named_symbols),
            "included_symbols": list(self.included_symbols),
            "excluded_symbols": list(self.excluded_symbols),
            "intervals": list(self.intervals),
            "market_type": self.market_type,
            "technical_periods": [
                {"indicator": name, "period": period}
                for name, period in self.technical_periods
            ],
            "excluded_technical_periods": [
                {"indicator": name, "period": period}
                for name, period in self.excluded_technical_periods
            ],
            "stop_pct": self.stop_pct,
            "tp_pct": self.tp_pct,
            "size_fraction": self.size_fraction,
            "directions": list(self.directions),
            "excluded_directions": list(self.excluded_directions),
            "news_metrics": list(self.news_metrics),
            "triggers": list(self.triggers),
            "condition_join": self.condition_join,
            "indicator_names": list(self.indicator_names),
            "excluded_indicator_names": list(self.excluded_indicator_names),
            "rsi_thresholds": [
                {"relation": relation, "value": value}
                for relation, value in self.rsi_thresholds
            ],
            "ranking_metric": self.ranking_metric,
            "score_order": self.score_order,
            "score_order_metric": self.score_order_metric,
            "exit_type": self.exit_type,
            "exit_atr_period": self.exit_atr_period,
            "exit_trail_mult": self.exit_trail_mult,
            "excluded_exit_types": list(self.excluded_exit_types),
        }


_CANDIDATE_DEFAULT_CIDS = frozenset({
    "default_candidate_indicator",
    "default_candidate_rsi_period",
    "default_candidate_rsi_long",
    "default_candidate_rsi_short",
    "candidate_sell_point_as_short_entry",
    "default_candidate_size",
    "default_candidate_stop",
    "default_candidate_take_profit",
})


def _candidate_default(cid: str, reading: str, basis: str) -> dict:
    return {
        "cid": cid,
        "reading": reading,
        "alternatives": ["由使用者明確指定此值"],
        "basis": basis,
    }


def candidate_trade_plan(intent: IntentDecision) -> dict:
    """One source of truth for hybrid prompt defaults and semantic checks.

    Defaults remain useful for a conversational front door, but only when the
    generated YAML admits each choice under ``strategy.assumptions``.  The
    returned readings are canonical strings: reconciliation compares them
    exactly, and the runtime carries those same strings into signal warnings.
    """
    sides = tuple(side for side in ("long", "short") if side in intent.directions)
    if "bidirectional_trade_points" in intent.evidence:
        sides = ("long", "short")
    elif "candidate_buy_point" in intent.evidence and not sides:
        sides = ("long",)
    elif "candidate_sell_point" in intent.evidence and not sides:
        sides = ("short",)
    elif not sides:
        sides = ("long", "short")

    explicit = {relation: float(value) for relation, value in intent.rsi_thresholds}
    periods = [period for name, period in intent.technical_periods if name == "rsi"]
    period = periods[0] if periods else 14
    alias = "rsi%d" % period
    thresholds = {}
    assumptions = []

    requested_rsi = (
        "rsi" in intent.indicator_names
        or any(name == "rsi" for name, _period in intent.technical_periods)
        or bool(intent.rsi_thresholds)
    )
    if not requested_rsi:
        assumptions.append(_candidate_default(
            "default_candidate_indicator",
            "default candidate indicator: name=rsi, period=14",
            "使用者要求逐候選技術買賣點但未指定技術指標",
        ))
    elif not periods:
        assumptions.append(_candidate_default(
            "default_candidate_rsi_period",
            "default candidate RSI period: period=14",
            "使用者指定 RSI 但未指定週期",
        ))

    if "sell_point_requires_short_interpretation" in intent.evidence:
        assumptions.append({
            "cid": "candidate_sell_point_as_short_entry",
            "reading": (
                "candidate sell point interpretation: intent=open_short, "
                "product=usd_m_perpetual; not close_long"
            ),
            "alternatives": [
                "close_long for an existing position",
                "advisory-only sell point",
            ],
            "basis": "使用者說賣點但未明確指定做空或平多",
        })

    if "long" in sides:
        if "below" in explicit:
            thresholds["long"] = ("below", explicit["below"])
        elif "above" in explicit and "short" not in sides:
            thresholds["long"] = ("above", explicit["above"])
        else:
            thresholds["long"] = ("below", 30.0)
            assumptions.append(_candidate_default(
                "default_candidate_rsi_long",
                "default candidate long threshold: indicator=rsi, period=%d, "
                "relation=below, value=30" % period,
                "使用者要求逐候選買點但未指定做多 RSI 門檻",
            ))
    if "short" in sides:
        if "above" in explicit:
            thresholds["short"] = ("above", explicit["above"])
        elif "below" in explicit and "long" not in sides:
            thresholds["short"] = ("below", explicit["below"])
        else:
            thresholds["short"] = ("above", 70.0)
            assumptions.append(_candidate_default(
                "default_candidate_rsi_short",
                "default candidate short threshold: indicator=rsi, period=%d, "
                "relation=above, value=70" % period,
                "使用者要求逐候選賣點但未指定做空 RSI 門檻",
            ))

    size = intent.size_fraction if intent.size_fraction is not None else 0.05
    if intent.size_fraction is None:
        assumptions.append(_candidate_default(
            "default_candidate_size",
            "default candidate size: mode=equity_pct, value=0.05",
            "使用者未指定逐候選交易計畫的倉位",
        ))
    stop = intent.stop_pct if intent.stop_pct is not None else 0.02
    if intent.stop_pct is None:
        assumptions.append(_candidate_default(
            "default_candidate_stop",
            "default candidate stop: type=pct_stop_tp, stop_pct=0.02",
            "使用者未指定逐候選交易計畫的停損",
        ))
    take_profit = intent.tp_pct if intent.tp_pct is not None else 0.04
    if intent.tp_pct is None:
        assumptions.append(_candidate_default(
            "default_candidate_take_profit",
            "default candidate take profit: type=pct_stop_tp, tp_pct=0.04",
            "使用者未指定逐候選交易計畫的停利",
        ))

    return {
        "sides": sides,
        "period": period,
        "alias": alias,
        "thresholds": thresholds,
        "size": float(size),
        "stop_pct": float(stop),
        "tp_pct": float(take_profit),
        "assumptions": tuple(assumptions),
        "default_cids": _CANDIDATE_DEFAULT_CIDS,
    }


def _rules(*items):
    return tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in items)


_SELECTION_RULES = _rules(
    ("zh_explicit_selection", r"選幣"),
    # A counted basket often omits the final noun after the metric:
    # "選五個成交量最高" is still a selector, whereas "選 EMA 10 還是 26"
    # has no counted-candidate phrase and remains ambiguous.
    ("zh_counted_selection",
     r"(?:選|挑|篩).{0,24}(?:\d+|[一二兩三四五六七八九十]{1,3})\s*(?:個|名|檔)"),
    ("zh_asset_request",
     r"(?:選|挑|篩|找|列出|推薦).{0,60}(?:幣別|幣種|代幣|候選幣|幣)"),
    ("zh_which_assets", r"(?:哪些|哪幾個).{0,30}(?:幣別|幣種|代幣|幣)"),
    ("zh_asset_ranking",
     r"(?:幣別|幣種|代幣|候選幣|幣).{0,30}(?:排行|排名)|"
     r"(?:排行|排名).{0,60}(?:幣別|幣種|代幣|候選幣|幣)"),
    ("zh_hot_assets",
     r"(?:熱門|熱度|常提到|常被提及|提及最多).{0,40}(?:幣別|幣種|代幣|幣)"),
    ("en_select_assets",
     r"\b(?:select|pick|screen|rank)\b.{0,80}\b(?:coins?|tokens?)\b"),
    ("en_request_assets",
     r"\b(?:want|find|show|give|list|recommend|discover)\b.{0,80}"
     r"\b(?:coins?|tokens?)\b"),
    ("en_hot_assets",
     r"\b(?:hot|trending|popular|mentioned|undiscovered|under[- ]the[- ]radar)\b"
     r".{0,40}\b(?:coins?|tokens?)\b"),
    ("en_top_assets", r"\btop\s*\d*\s*(?:coins?|tokens?)\b"),
    ("en_selection", r"\b(?:coin|token)\s+selection\b|\bwhich\s+(?:coins?|tokens?)\b"),
)

_TRADE_RULES = _rules(
    # ``(?<!未)平倉``: 未平倉(量) is *open interest*, a cross-section data source,
    # and it contains 平倉 (close a position) as a substring. Without the
    # lookbehind, "選五個未平倉量最高的幣" scored as selection AND trade AND
    # execution, so it came back as "cannot tell whether you want a basket or a
    # trade" instead of the true answer, "open interest has no whole-market
    # endpoint yet".
    ("zh_trade_action",
     r"(?:(?:技術分析.{0,12})?(?:買賣點|買點|賣點)|"
     r"買進|賣出|買入|賣掉|做多|做空|進場|出場|"
     r"(?<!未)平倉|停損|停利|止損|止盈)"),
    ("zh_trade_trigger",
     r"(?:上穿|下穿|突破|跌破|黃金交叉|死亡交叉|交易策略|回測策略)"),
    ("en_trade_action",
     r"\b(?:buy|sell|long|short|entry|exit|close|reduce|flip)\b"),
    ("en_trade_trigger",
     r"\b(?:stop[- ]loss|take[- ]profit|breakout|cross(?:es|ing)?|trading strategy|backtest)\b"),
)

_PLURAL_SCOPE = re.compile(
    r"(?:一些|幾個|多個|數個|前\s*[一二三四五六七八九十百0-9]+\s*名|哪些|哪幾|候選|排行|排名|清單|幣別|幣種)"
    r"|\b(?:some|few|several|many|multiple|which|top\s*\d+)\b"
    r"|\b(?:coins|tokens)\b",
    re.IGNORECASE,
)
_LONG_SHORT_RATIO_TERM = (
    r"(?:散戶(?:帳戶)?多空比|多空(?:帳戶)?比|多空比|"
    r"long[\s/_-]*short(?:[\s_-]*(?:ratio|account))?|"
    r"long[\s_-]*account(?:s)?\s*(?:ratio|share)?)"
)
_LONG_SHORT_RATIO = re.compile(_LONG_SHORT_RATIO_TERM, re.IGNORECASE)
_SOURCE_RULES = _rules(
    ("news",
     r"(?:新聞|社群|熱度|熱門|提到|提及|Square|news|social|mention|mentioned|buzz|hot|trending|popular)"),
    ("funding", r"(?:資金費率|funding(?:\s+rate)?)"),
    ("open_interest", r"(?:未平倉|未平倉量|持倉量|open[\s_-]*interest|\bOI\b)"),
    ("long_short_ratio", _LONG_SHORT_RATIO_TERM),
    ("liquidity", r"(?:流動性|成交量|交易量|成交額|交易額|quote[\s_-]*volume|\b(?:volume|turnover)\b|liquidity)"),
    ("price_change", r"(?:漲幅|跌幅|漲最多|跌最多|price[\s_-]*change|gainers?|losers?)"),
)
_FUNDING_TERM = r"(?:資金費率|funding(?:\s*rate)?)"
#: Which end of a signed column, for columns whose name carries no sign of its
#: own. ``negative``/``positive`` cover ``most negative`` too, so those longer
#: forms are not repeated. "為負 / 是負 / 負的" are here because they are the
#: ordinary way to ask for the bottom of the funding column, and leaving them out
#: did not produce "no direction stated" — it produced a prompt that asserted the
#: opposite direction (see ``score_order is None`` handling in the demo prompt).
#: ``由低到高``/``ascending`` are here for the same reason: a user who spells the
#: sort out is the least ambiguous case there is, and leaving it unrecognised sent
#: it down the "user said nothing" path.
_LOW_END = (r"(?:最負|最低|最小|為負|是負|負的|由低到高|由小到大|升冪|"
            r"negative|lowest|smallest|ascending)")
_HIGH_END = (r"(?:最正|最高|最大|為正|是正|正的|由高到低|由大到小|降冪|"
             r"positive|highest|largest|biggest|descending)")

#: The 24h change column needs TWO words to name an end of it, and they do not
#: compose the way the funding pair does: 跌幅 is the *size of a fall*, so the
#: magnitude word decides which end it points at — 跌幅最大 is the bottom of the
#: column (most negative) while 跌幅最小 is the top of it. Matching the direction
#: word alone, which is what this used to do, read 跌幅最小 as "most negative",
#: and read the filters 跌幅小於 3% / 排除跌幅超過 10% as ranking requests.
#: :func:`reconcile_intent` *enforces* whatever direction lands here, so each of
#: those rejected the correct spec and accepted the inverted one.
_FALL_TERM = r"(?:跌幅|\blosers?\b|\bdrops?\b|\bdeclines?\b)"
_RISE_TERM = r"(?:漲幅|\bgainers?\b|\bgains?\b)"
_LARGEST = (r"(?:最大|最多|最高|最兇|最猛|最深|由大到小|前\s*\d{1,3}\s*名|"
            r"biggest|largest|highest|most|top)")
_SMALLEST = r"(?:最小|最少|最低|最輕|由小到大|smallest|fewest|lowest|least)"


def _sized_direction(direction: str, magnitude: str) -> str:
    """A magnitude word close enough to a direction word to be modifying it.

    The window is deliberately tight. "跌幅超過 10% 且成交量最大的幣" filters on the
    fall and ranks on volume; a funding-sized window would read that trailing
    最大 as the fall's magnitude and demand the wrong end of a different column.
    """
    return rf"(?:{direction}.{{0,6}}?{magnitude}|{magnitude}.{{0,6}}?{direction})"


#: Phrasings that name *which end* of the ranked column the user wants, each
#: bound to the frame column it is talking about.
#:
#: The lowest-funding rule used to be a refusal trigger: ``selection.score`` only
#: ranked descending, so "the five most negative funding rates" was answered with
#: ``unsupported`` rather than quietly handing back the five most *positive* ones.
#: ``selection.order`` now exists, so the same phrasing does two useful things
#: instead: it puts ``order: asc`` in the generation prompt, and it lets
#: :func:`reconcile_intent` check that the model actually wrote it. That check is
#: the part that matters — a basket taken from the wrong end of a signed column
#: looks completely healthy from the outside (five symbols, five plausible
#: rates), so nobody reading the output can catch it.
#:
#: Only signed columns are listed. "the least liquid coins" is not a screen
#: anyone runs, so ``quote_volume`` gets no direction rule and therefore no
#: chance to misfire on "highest volume".
#:
#: Every rule needs a direction word AND a word saying which end of it; a bare
#: 跌幅 or ``losers`` is as likely to be a filter as a ranking. Duplicate
#: ``(column, order)`` pairs are harmless — the caller collects them into a set,
#: so two phrasings agreeing on one end still count as one unambiguous answer.
_SCORE_ORDER_RULES = tuple(
    (column, order, re.compile(pattern, re.IGNORECASE))
    for column, order, pattern in (
        ("fundingRatePct", "asc",
         rf"(?:{_FUNDING_TERM}.{{0,20}}{_LOW_END}|{_LOW_END}.{{0,20}}{_FUNDING_TERM})"),
        ("fundingRatePct", "desc",
         rf"(?:{_FUNDING_TERM}.{{0,20}}{_HIGH_END}|{_HIGH_END}.{{0,20}}{_FUNDING_TERM})"),
        # Fall x largest = bottom of the column; fall x smallest = top of it.
        ("priceChangePercent", "asc", _sized_direction(_FALL_TERM, _LARGEST)),
        ("priceChangePercent", "desc", _sized_direction(_FALL_TERM, _SMALLEST)),
        ("priceChangePercent", "desc", _sized_direction(_RISE_TERM, _LARGEST)),
        ("priceChangePercent", "asc", _sized_direction(_RISE_TERM, _SMALLEST)),
        # Idioms that fuse both halves into one word, so the pair rules above
        # cannot see them: 跌最多 has no 幅, "worst performing" has no 最.
        ("priceChangePercent", "asc", r"(?:跌最多|跌最兇|跌最深|worst\s+perform)"),
        ("priceChangePercent", "desc", r"(?:漲最多|漲最兇|漲最猛|best\s+perform)"),
    )
)

#: Frame columns that carry the SAME user-facing metric, so a phrase about one is
#: a phrase about the other: ``the name a rule matched under -> every column that
#: satisfies it``.
#:
#: ``fundingRateApr`` is ``fundingRatePct`` multiplied by the contract's
#: settlements per year — a strictly POSITIVE factor, so the two agree on sign
#: and therefore on which end of the column 最負 / negative names. They are
#: aliased because :data:`_SCORE_ORDER_RULES` matches the phrase under the name
#: ``fundingRatePct`` while a generated spec now ranks on ``fundingRateApr`` (the
#: annualised column is the only one comparable across 8h / 4h / 1h contracts).
#: Without the alias, :func:`reconcile_intent` found no overlap between
#: ``score_order_metric`` and the spec's score dependencies and quietly stopped
#: enforcing the direction — and a basket taken from the wrong end of a signed
#: column looks completely healthy from the outside, which is the whole reason
#: that check exists.
_METRIC_COLUMN_ALIASES = {
    "fundingRatePct": ("fundingRatePct", "fundingRateApr"),
}


def _metric_columns(metric) -> set:
    """Every frame column that would satisfy a request for *metric*."""
    if metric is None:
        return set()
    return set(_METRIC_COLUMN_ALIASES.get(metric, (metric,)))


_EXCLUSION_VERB = (r"(?:排除|剔除|去掉|扣掉|不要|避開|避免|不含|"
                   r"\b(?:exclude|excluding|avoid|avoiding|without|except|skip)\b)")
_SELECTION_VERB = (r"(?:選|挑|篩|找|列出|推薦|"
                   r"\b(?:select|pick|screen|rank|show|find|list|want)\b)")

#: Polarity belongs to the extracted slot, not to an after-the-fact exception in
#: reconciliation.  Without this, ``不要使用 RSI`` and ``不要做多`` put the exact
#: same positive tokens in :class:`IntentDecision` as ``使用 RSI`` / ``做多``.
#: The reconciler then rejected the faithful YAML and accepted the forbidden one.
_NEGATED_TERM_PREFIX = re.compile(
    r"(?:不要|不用|不使用|不採用|不采用|不准|不準|不可|不能|不允許|不允许|禁止|避免|避開|避开|"
    r"排除|剔除|去掉|扣掉|不含|勿|別|别|除了|"
    r"\b(?:do\s+not|don't|never|without|exclude|excluding|avoid|avoiding|"
    r"except|skip|no)\b)"
    r"(?:\s*(?:再|去|使用|採用|采用|加入|用|做|開|开|建立|"
    r"allow|use|using|take|go))*\s*$",
    re.IGNORECASE,
)
_NEGATED_TERM_SUFFIX = re.compile(
    r"^\s*(?:不要|不用|不准|不準|禁止|避免|排除|剔除|勿|別用|别用|"
    r"\b(?:is\s+forbidden|not\s+allowed|must\s+not\s+be\s+used)\b)",
    re.IGNORECASE,
)


def _is_excluded_match(text: str, match: re.Match) -> bool:
    """Whether one concrete token occurrence is in a negative clause.

    The check is span-local.  A global ``"不要" in text`` would also negate the
    positive half of ``不要做多,只允許做空``; splitting at punctuation keeps the
    two clauses independent.  Looking on both sides covers the ordinary
    ``不要使用 RSI`` form and the less common ``RSI 不要用`` form.
    """

    left = re.split(r"[,，。;；、\n]", text[max(0, match.start() - 48):match.start()])[-1]
    right = re.split(r"[,，。;；、\n]", text[match.end():match.end() + 32])[0]
    return bool(_NEGATED_TERM_PREFIX.search(left) or _NEGATED_TERM_SUFFIX.search(right))

#: Exclusion phrasing attached to a direction word inverts what the direction
#: means: "排除跌幅最大的幣" names the end of the column the user does NOT want.
#: No ``order`` value means "rank everything else", so a match here drops the
#: direction instead of enforcing its opposite.
#:
#: The gap is tempered rather than a plain ``.{0,30}``: it may not cross a clause
#: break or a selection verb, because "排除 BTC,選跌幅最大的五個幣" excludes a
#: symbol and then states a direction that IS worth enforcing. Widening this to a
#: bare character window silently disarms the check for those requests.
_DIRECTION_EXCLUSION = re.compile(
    rf"{_EXCLUSION_VERB}(?:(?!{_SELECTION_VERB})[^,,。;;、\n]){{0,30}}?"
    rf"(?:{_FALL_TERM}|{_RISE_TERM}|{_FUNDING_TERM})",
    re.IGNORECASE,
)

#: A diminisher in front of the sign word points at the OTHER end than the sign
#: word does: "最不負的 funding" / "the least negative funding" asks for the rates
#: closest to zero, i.e. the top of the bottom half, not the bottom of the column.
#: Which end that is depends on how much of the column is negative today, so it is
#: not expressible as ``order`` at all — hence dropped rather than guessed.
_DIMINISHED_SIGN = re.compile(
    r"(?:最不|沒那麼|不太|不那麼|\b(?:least|less)\b)\s*(?:負|正|negative|positive)",
    re.IGNORECASE,
)
_BULLISH_PREFERENCE = re.compile(
    r"(?:可以漲|會漲|上漲|看漲|偏多|適合做多|可能漲|bullish|likely\s+to\s+(?:rise|go\s+up)|upside)",
    re.IGNORECASE,
)
# A named asset-class exclusion is a hard semantic constraint.  We deliberately
# retain the category name instead of guessing whether a venue's free-form
# metadata happens to use the same spelling.  The demo front door stops before
# an LLM call until there is an explicit, reviewed mapping for it.
_CATEGORY_EXCLUSION_RULES = (
    ("tradfi", re.compile(r"(?:\btradfi\b|傳統金融|傳統金融類)", re.IGNORECASE)),
    ("stock_token", re.compile(
        r"(?:股票代幣|股權代幣|代幣化(?:股票|股權)|"
        r"\b(?:stock|equity)[\s_-]*(?:tokens?|assets?)\b|"
        r"\btokeni[sz]ed[\s_-]*(?:stocks?|equities)\b)",
        re.IGNORECASE,
    )),
    ("stablecoin", re.compile(r"(?:穩定幣|\bstablecoins?\b)", re.IGNORECASE)),
)
_NEWS_MENTION_METRIC = re.compile(
    r"(?:常提到|常被提及|提及量|提及最多|熱門|熱度|"
    r"mentions?|mentioned|buzz|hot|trending|popular)",
    re.IGNORECASE,
)
_NEWS_SENTIMENT_METRIC = re.compile(
    r"(?:新聞情緒|社群情緒|情緒排行|sentiment|bullish|bearish)",
    re.IGNORECASE,
)
_EXPLICIT_NEWS_SOURCE = re.compile(
    r"(?:新聞|社群|提到|提及|Square|news|social|mentions?|mentioned|sentiment|buzz)",
    re.IGNORECASE,
)
_VOLUME_RANKING = re.compile(
    r"(?:by\s+(?:quote[\s_-]*)?(?:volume|turnover)|"
    r"(?:依|按|根據).{0,16}(?:成交量|交易量|成交額|交易額|流動性)|"
    r"(?:成交量|交易量|成交額|交易額|流動性).{0,12}(?:最大|最高|排行|排名|top))",
    re.IGNORECASE,
)
_UNSUPPORTED_DISCOVERY = re.compile(
    r"(?:少見|冷門|未被.{0,8}發現|沒人.{0,8}發現|尚未.{0,8}發現|"
    r"undiscovered|under[- ]the[- ]radar|havent\s+(?:find|found)|haven't\s+(?:find|found))",
    re.IGNORECASE,
)
# These names intentionally mirror the closed capability table in
# ``tools.nl2yaml.capability``.  The demo must refuse the named request before
# an LLM can substitute a correlated metric (for example turnover for market
# cap) or pretend a side effect was scheduled.  Keep the patterns narrow: they
# identify explicit unsupported requirements, not vague domain discussion.
_NAMED_UNSUPPORTED_GAPS = (
    ("GAP-MARKET-CAP", re.compile(r"(?:市值|market\s*cap|\bmcap\b)", re.IGNORECASE)),
    ("GAP-ONCHAIN-CONCENTRATION", re.compile(
        r"(?:鏈上.{0,20}(?:持幣|持有人|集中)|on[\s_-]*chain.{0,24}"
        r"(?:holder|concentration)|holder\s+concentration)", re.IGNORECASE)),
    ("GAP-ALERT-NOTIFY", re.compile(
        r"(?:Telegram|電報通知|推播|通知我|提醒我|webhook|alert\s+me|notify\s+me)",
        re.IGNORECASE)),
    ("GAP-ACCOUNT-OPS", re.compile(
        # A naked ``2x`` is not enough: it can be the multiplier in an ATR
        # trailing stop. Require an execution/account context around the
        # multiplier so the unsupported-account gate cannot shadow a fully
        # specified, runnable risk.exit contract.
        r"(?:槓桿|leverage|(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*x\s*"
        r"(?:槓桿|leverage|(?:open|enter)\s+(?:a\s+)?(?:position|trade)|"
        r"(?:開倉|开仓|進場|进场))|限價(?:單|進場)?|限价(?:单|进场)?|"
        r"\blimit\s+(?:entry|order)\b|\bstop\s+order\b)",
        re.IGNORECASE)),
)
_CROSS_SECTION_LIQUIDATION = re.compile(
    r"(?:清算|爆倉|爆仓|liquidations?)", re.IGNORECASE,
)
_FULL_SYMBOL = re.compile(
    r"\b[A-Z0-9]{2,12}(?:USDT|USDC|BUSD|FDUSD|USD|BTC|ETH)\b",
    re.IGNORECASE,
)
_KNOWN_BASE_SYMBOL = re.compile(
    r"\b(?:BTC|ETH|SOL|BNB|XRP|SUI|DOGE|ADA|AVAX|LINK|DOT|TON|TRX)\b",
    re.IGNORECASE,
)
_BARE_UPPER_SYMBOL = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")
_SYMBOL_STOPWORDS = {
    "ADX", "API", "ATR", "BUY", "CHOOSE", "EMA", "ENTER", "EXIT", "FIND",
    "LLM", "LONG", "MACD", "OHLCV", "OI", "PICK", "RANK", "RSI", "SELECT", "SELL",
    "SHORT", "SMA", "TOP", "TRADE", "USD", "USDC", "USDT", "VWAP", "YAML",
}
#: Separators that keep adjacent symbol mentions in the same list.  Clause
#: punctuation is allowed only when *nothing except another list separator* sits
#: between the two symbols.  Consequently ``排除 BTC, ETH`` propagates the
#: exclusion, while ``排除 BTC, 選 ETH`` does not cross the new selection clause.
_SYMBOL_LIST_SEPARATOR = re.compile(
    r"\s*(?:(?:[/／|、,，&]|以及|與|和|及|或(?:者)?|\b(?:and|or)\b)\s*)+",
    re.IGNORECASE,
)
_EXECUTION_ACTION = re.compile(
    # (?<!未)平倉 for the same reason as _TRADE_RULES: 未平倉量 is open interest.
    r"(?:(?:技術分析.{0,12})?(?:買賣點|買點|賣點)|"
    r"買進|買入|賣出|賣掉|下單|進場|出場|(?<!未)平倉|"
    r"自動交易|直接買|"
    r"\b(?:buy|sell|execute|enter|exit|entry\s+points?|exit\s+points?|"
    r"place\s+orders?|trade\s+them)\b)",
    re.IGNORECASE,
)
#: ``做空候選`` can be a direction label on a basket rather than an order.
#: Treat long/short wording as candidate execution only when it is attached to
#: an explicit RSI threshold; otherwise existing direction-only selections stay
#: selections instead of being turned into unsupported compound orders.
_CANDIDATE_DIRECTION_ACTION = re.compile(
    r"(?:做多|做空|\b(?:long|short)\b)", re.IGNORECASE,
)
_CANDIDATE_TRADE_POINT = re.compile(
    r"(?:技術分析.{0,12})?(?:買賣點|買點|賣點)|"
    r"\b(?:entry|exit|buy|sell)\s+points?\b",
    re.IGNORECASE,
)
_CANDIDATE_BUY_POINT = re.compile(
    r"(?:(?:技術分析.{0,12})?買點|\bbuy\s+points?\b|"
    r"RSI(?:\s*[-_]?\s*\d{1,4})?.{0,24}?"
    r"(?:低於|小於|低于|below|under)\s*\d+(?:\.\d+)?"
    r"(?:\s*(?:時|就))?\s*(?:買(?:進|入)?|\bbuy\b))",
    re.IGNORECASE,
)
_CANDIDATE_SELL_POINT = re.compile(
    r"(?:(?:技術分析.{0,12})?賣點|\bsell\s+points?\b|"
    r"RSI(?:\s*[-_]?\s*\d{1,4})?.{0,24}?"
    r"(?:高於|大於|高于|above|over)\s*\d+(?:\.\d+)?"
    r"(?:\s*(?:時|就))?\s*(?:賣(?:出|掉)?|\bsell\b))",
    re.IGNORECASE,
)
_BIDIRECTIONAL_TRADE_POINT = re.compile(
    r"(?:(?:技術分析.{0,12})?買賣點|買點.{0,12}賣點|賣點.{0,12}買點|"
    r"\bbuy\s+and\s+sell\s+points?\b)",
    re.IGNORECASE,
)
_INTERVAL = re.compile(
    r"(?<![A-Za-z0-9])(\d+)\s*"
    r"(minutes?|mins?|m|分鐘|分|hours?|hrs?|h|小時|小时|days?|d|天|日|weeks?|w|週|周)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_PREFIX_INTERVAL = re.compile(
    r"(?<![A-Za-z0-9])([mhdw])\s*(\d{1,4})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Binance's universe ticker exposes a rolling 24-hour quote turnover.  It is a
# *metric window*, not a candles timeframe.  Without this distinction a request
# such as ``依 24 小時成交額排名`` asks the model for quoteVolume and then the
# reconciliation layer incorrectly demands ``data.primary.interval: 24h``.
_TURNOVER_METRIC_WINDOW = re.compile(
    r"(?:24\s*(?:小時|小时|h(?:ours?)?)\s*(?:的\s*)?"
    r"(?:成交量|交易量|成交額|交易額|(?:quote[\s_-]*)?(?:volume|turnover))|"
    r"(?:成交量|交易量|成交額|交易額|(?:quote[\s_-]*)?(?:volume|turnover))\s*"
    r"(?:的\s*)?24\s*(?:小時|小时|h(?:ours?)?))",
    re.IGNORECASE,
)
_TECHNICAL_PERIOD = re.compile(
    r"(?<![A-Za-z])(EMA|SMA|RSI|ADX)\s*[-_]?\s*(\d{1,4})(?!\d)",
    re.IGNORECASE,
)
_TECHNICAL_NAME = re.compile(
    r"(?<![A-Za-z])(EMA|SMA|RSI|ADX|MACD|SUPERTREND|VWAP)(?![A-Za-z])",
    re.IGNORECASE,
)
_SUPERTREND_PARAMETERS = re.compile(
    r"(?<![A-Za-z])supertrend\s*[（(]\s*(\d{1,4})\s*[,，]\s*"
    r"(\d+(?:\.\d+)?)\s*[)）]",
    re.IGNORECASE,
)
_RSI_THRESHOLD = re.compile(
    r"RSI(?:\s*[-_]?\s*\d{1,4})?.{0,24}?"
    r"(低於|小於|低于|below|under|高於|大於|高于|above|over)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
#: A second threshold commonly inherits the indicator named by the first:
#: ``RSI14 低於 30 做多，高於 70 做空``.  Requiring the action plus a clause
#: join keeps an unrelated later comparison (for example quote volume > 1m)
#: from being relabelled as an RSI threshold merely because RSI appeared earlier.
_RSI_FOLLOWUP_THRESHOLD = re.compile(
    r"\s*(?:時|就)?\s*"
    r"(?:做多|做空|買進|買入|賣出|賣掉|買|賣|買點|賣點|"
    r"\b(?:long|short|buy|sell)\b)?\s*"
    r"(?:[,，、;；/]|且|並且|而|同時|和|與|及|以及|或(?:者)?|\b(?:and|or|then)\b)+\s*"
    r"(低於|小於|低于|below|under|高於|大於|高于|above|over)\s*"
    r"(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_LONG_DIRECTION = re.compile(r"(?:買進|買入|做多|多方|\b(?:buy|long)\b)", re.IGNORECASE)
_SHORT_DIRECTION = re.compile(r"(?:做空|空方|\bshort\b)", re.IGNORECASE)
# A bare ``賣出`` / ``sell`` is not an entry-side declaration.  In a
# single-instrument strategy it can mean either "close my long" or "open a
# short"; treating it as the latter changes both required capability and the
# emitted intent.  Candidate-trade selection has a separately documented,
# confirmation-only sell-point convention, so this guard is applied only to a
# top-level trade below.
_BARE_SELL_ACTION = re.compile(r"(?:賣出|賣掉|\bsell\b)", re.IGNORECASE)
_ATR_TERM = (
    r"(?:(?<![A-Za-z0-9])ATR(?:\s*\(\s*\d{1,3}\s*\)|\s*\d{1,3})?"
    r"(?![A-Za-z0-9])|平均真實波幅)"
)
_TRAILING_TERM = r"(?:追蹤|跟蹤|移動|trailing)"
_STOP_TERM = r"(?:停損|止損|stop(?:[- ]?loss)?)"
# All spellings require ATR + trailing + stop. A request to inspect an ATR
# indicator must not become permission to invent an ATR-based exit.
_ATR_TRAILING_STOP = re.compile(
    rf"(?:{_ATR_TERM}.{{0,40}}?{_TRAILING_TERM}.{{0,24}}?{_STOP_TERM}|"
    rf"{_TRAILING_TERM}.{{0,24}}?{_STOP_TERM}.{{0,40}}?{_ATR_TERM}|"
    rf"{_TRAILING_TERM}.{{0,24}}?{_ATR_TERM}.{{0,24}}?{_STOP_TERM}|"
    rf"{_STOP_TERM}.{{0,24}}?{_TRAILING_TERM}.{{0,24}}?{_ATR_TERM})",
    re.IGNORECASE,
)
_ATR_PERIOD = re.compile(
    r"(?<![A-Za-z0-9])ATR\s*(?:\(\s*(?P<paren>\d{1,3})\s*\)|"
    r"(?:period\s*[-_:：=]?\s*)?(?P<plain>\d{1,3})"
    r"(?![\d.]|\s*(?:x|倍)))",
    re.IGNORECASE,
)
_ATR_TRAIL_MULTIPLIER = re.compile(
    r"(?:(?<![A-Za-z0-9])(?P<before>\d+(?:\.\d+)?)\s*(?:x|倍)\s*"
    r"(?:[- ]?\s*)?ATR(?:\s*\(\s*\d{1,3}\s*\)|\s*\d{1,3})?"
    r"(?![A-Za-z0-9])|(?<![A-Za-z0-9])ATR(?:\s*\(\s*\d{1,3}\s*\)|"
    r"\s*\d{1,3})?(?![A-Za-z0-9])[^,，。;；\n]{0,16}?"
    r"(?P<after>\d+(?:\.\d+)?)\s*(?:x|倍))",
    re.IGNORECASE,
)
_CONDITION_OR = re.compile(r"(?:\b(?:or|either)\b|或(?:者)?|任一)", re.IGNORECASE)
_CONDITION_AND = re.compile(
    r"(?:\b(?:and|then)\b|且|並且|同時|而|和|與|及|以及)", re.IGNORECASE,
)
_TRIGGER_RULES = _rules(
    ("cross_above", r"(?:上穿|黃金交叉|golden[\s_-]*cross|cross(?:es|ing)?\s+above)"),
    ("cross_below", r"(?:下穿|死亡交叉|death[\s_-]*cross|cross(?:es|ing)?\s+below)"),
    ("breakout_high", r"(?:突破(?:近期|前)?高|向上突破|breakout(?:\s+high)?)"),
    ("breakout_low", r"(?:跌破(?:近期|前)?低|向下突破|breakdown|breakout\s+low)"),
)

_CHINESE_DIGITS = {
    "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

_ENGLISH_COUNTS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


def _parse_chinese_count(value: str) -> int | None:
    value = str(value)
    if value in _CHINESE_DIGITS:
        return _CHINESE_DIGITS[value]
    if "十" not in value:
        return None
    left, _, right = value.partition("十")
    tens = _CHINESE_DIGITS.get(left, 1) if left else 1
    units = _CHINESE_DIGITS.get(right, 0) if right else 0
    return tens * 10 + units


def _requested_count(text: str) -> int | None:
    for pattern in (
        r"\b(?:top|select|pick)?\s*(\d+)(?:\s+[a-z-]+){0,3}\s+(?:coins?|tokens?)\b",
        r"\btop\s*(\d+)\s*(?:coins?|tokens?)?\b",
        r"(?:前|選出|挑選)?\s*(\d+)\s*(?:個|名|檔|種|coins?|tokens?)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    english = "|".join(_ENGLISH_COUNTS)
    match = re.search(
        rf"\b(?:top|select|pick)?\s*({english})(?:\s+[a-z-]+){{0,3}}\s+"
        rf"(?:coins?|tokens?)\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return _ENGLISH_COUNTS[match.group(1).lower()]
    match = re.search(r"([一二兩三四五六七八九十]{1,3})\s*(?:個|名|檔|種)", text)
    return _parse_chinese_count(match.group(1)) if match else None


def _symbol_mentions(text: str) -> tuple[tuple[str, bool], ...]:
    """Return ``(symbol, excluded)`` without losing the mention's polarity."""

    found = []
    for pattern in (_FULL_SYMBOL, _KNOWN_BASE_SYMBOL, _BARE_UPPER_SYMBOL):
        for match in pattern.finditer(text):
            symbol = match.group(0).upper()
            if pattern is _BARE_UPPER_SYMBOL and (
                symbol in _SYMBOL_STOPWORDS
            or re.fullmatch(r"(?:EMA|SMA|RSI|ADX|ATR|MACD|SUPERTREND)\d*", symbol)
                # H4/H1/M15 are ordinary chart intervals, not ticker symbols.
                # Keeping them in ``named_symbols`` caused an MTF request to
                # become an accidental universe allowlist downstream.
                or re.fullmatch(r"(?:[MHDW]\d{1,4})", symbol)
            ):
                continue
            found.append((match.start(), match.end(), symbol, match))
    # The broad bare-upper pattern also sees symbols found by a narrower pattern.
    # Collapse those overlapping matches before carrying an exclusion through a
    # joined list; otherwise the duplicate at one span becomes the "previous"
    # symbol and breaks propagation to the next real mention.
    mentions = []
    seen_spans = set()
    for start, end, symbol, match in sorted(found, key=lambda item: item[:3]):
        key = (start, end, symbol)
        if key not in seen_spans:
            seen_spans.add(key)
            mentions.append((start, end, symbol, match))

    classified = []
    previous_end = None
    previous_excluded = False
    for start, end, symbol, match in mentions:
        excluded = _is_excluded_match(text, match)
        if not excluded and previous_excluded and previous_end is not None:
            separator = text[previous_end:start]
            excluded = bool(_SYMBOL_LIST_SEPARATOR.fullmatch(separator))
        classified.append((symbol, excluded))
        previous_end = end
        previous_excluded = excluded

    # Preserve textual order while deduplicating the same symbol/polarity pair.
    ordered = []
    seen = set()
    for symbol, excluded in classified:
        key = (symbol, excluded)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return tuple(ordered)


def _source_mentions(text: str) -> tuple[tuple[str, bool], ...]:
    """Return source names with their local positive/negative polarity.

    Source lists follow the same grammar as symbol lists: a negative clause
    covering news or funding negates both items, while a later independent
    clause starts a new positive request. Treating the second list item as
    positive is dangerous because reconciliation would require data the user
    expressly forbade.
    """
    found = []
    for name, rule in _SOURCE_RULES:
        for match in rule.finditer(text):
            found.append((match.start(), match.end(), name, match))

    mentions = []
    seen_spans = set()
    for start, end, name, match in sorted(found, key=lambda item: item[:3]):
        key = (start, end, name)
        if key not in seen_spans:
            seen_spans.add(key)
            mentions.append((start, end, name, match))

    classified = []
    previous_end = None
    previous_excluded = False
    for start, end, name, match in mentions:
        excluded = _is_excluded_match(text, match)
        if not excluded and previous_excluded and previous_end is not None:
            separator = text[previous_end:start]
            excluded = bool(_SYMBOL_LIST_SEPARATOR.fullmatch(separator))
        classified.append((name, excluded))
        previous_end = end
        previous_excluded = excluded

    ordered = []
    seen = set()
    for name, excluded in classified:
        key = (name, excluded)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return tuple(ordered)


def _named_symbols(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(symbol for symbol, _excluded in _symbol_mentions(text)))


def _requested_intervals(text: str) -> tuple[str, ...]:
    unit_map = {
        "m": "m", "min": "m", "mins": "m", "minute": "m", "minutes": "m",
        "分鐘": "m", "分": "m",
        "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
        "小時": "h", "小时": "h",
        "d": "d", "day": "d", "days": "d", "天": "d", "日": "d",
        "w": "w", "week": "w", "weeks": "w", "週": "w", "周": "w",
    }
    turnover_windows = [match.span() for match in _TURNOVER_METRIC_WINDOW.finditer(text)]
    values = []
    for match in _INTERVAL.finditer(text):
        # Do not turn the rolling universe metric's ``24h`` into a bar
        # timeframe.  Other intervals in the same request remain meaningful,
        # for example "24h turnover plus an H1 EMA filter".
        if any(start <= match.start() < end for start, end in turnover_windows):
            continue
        unit = unit_map[match.group(2).lower()]
        values.append((match.start(), "%d%s" % (int(match.group(1)), unit)))
    # Chart users commonly spell timeframes as H4/H1/M15 rather than 4h/1h/15m.
    # Parse both forms in textual order so a generated MTF selector can preserve
    # the requested timeframes exactly.
    for match in _PREFIX_INTERVAL.finditer(text):
        values.append((match.start(), "%d%s" % (
            int(match.group(2)), match.group(1).lower()
        )))
    return tuple(dict.fromkeys(value for _start, value in sorted(values)))


def _clauses(text: str) -> tuple[str, ...]:
    """Small clause split for context-sensitive lexical meanings.

    ``偏多`` in a crowd-positioning clause describes the *crowd*; it must not
    turn into a request for a news-bullish proxy.  Likewise Supertrend bearish
    is an indicator state, not a generic preference.  Punctuation boundaries
    are enough here and deliberately avoid a brittle natural-language parser.
    """

    parts = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "(（":
            depth += 1
        elif char in ")）" and depth:
            depth -= 1
        elif depth == 0 and char in ",，。;；、\n":
            part = text[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return tuple(parts)


def _generic_bullish_preference(text: str) -> bool:
    for match in _BULLISH_PREFERENCE.finditer(text):
        clause = next(
            (item for item in _clauses(text)
             if match.group(0) in item),
            "",
        )
        if _LONG_SHORT_RATIO.search(clause) or "supertrend" in clause.lower():
            continue
        return True
    return False


def _long_short_min_account_requirement(text: str) -> tuple[float, str] | None:
    """Extract a crowd-long threshold and whether its lower bound is strict.

    ``> 60%`` and ``>= 60%`` select different baskets at the exact boundary.
    Preserve that operator here rather than collapsing it into a numeric floor;
    the generated YAML then selects the matching inclusive or exclusive Block
    parameter.  Raw ratio thresholds and an unspecified ``偏多`` remain a
    clarification, never an invented comparator.
    """

    comparison_after = re.compile(
        r"(?P<op>>=|>|大於等於|高於等於|至少|不低於|以上|大於|高於|超過|"
        r"above|over|at\s+least)\s*(?P<value>\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    )
    comparison_before = re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*%\s*(?P<op>以上|或以上|起|above|over)",
        re.IGNORECASE,
    )
    inclusive = {">=", "大於等於", "高於等於", "至少", "不低於", "以上", "或以上",
                 "起", "at least"}
    for clause in _clauses(text):
        if not _LONG_SHORT_RATIO.search(clause):
            continue
        if not re.search(r"(?:偏多|多方|long(?:[\s_-]*accounts?)?)", clause,
                         re.IGNORECASE):
            continue
        match = comparison_after.search(clause) or comparison_before.search(clause)
        if match:
            operator = str(match.group("op")).lower()
            return (float(match.group("value")), ">=" if operator in inclusive else ">")
    return None


def _supertrend_directions(text: str) -> tuple[str, ...]:
    """Return explicit Supertrend direction states without borrowing ``偏多``.

    This keeps ``散戶多空比偏多 > 60%`` as a crowd filter rather than a request
    for ``long_when`` or a news-sentiment proxy.
    """

    directions = []
    for clause in _clauses(text):
        if "supertrend" not in clause.lower():
            continue
        if re.search(r"(?:偏多|bullish)", clause, re.IGNORECASE):
            directions.append("long")
        if re.search(r"(?:偏空|bearish)", clause, re.IGNORECASE):
            directions.append("short")
    return tuple(dict.fromkeys(directions))


def _category_exclusions(text: str) -> tuple[str, ...]:
    if not re.search(_EXCLUSION_VERB, text, re.IGNORECASE):
        return ()
    return tuple(name for name, rule in _CATEGORY_EXCLUSION_RULES if rule.search(text))


def _percent_near(text: str, labels: str) -> float | None:
    after = re.search(
        rf"(?:{labels})\s*(?:為|是|=|:)?\s*(\d+(?:\.\d+)?)\s*%",
        text,
        re.IGNORECASE,
    )
    if after:
        return float(after.group(1)) / 100.0
    before = re.search(
        rf"(\d+(?:\.\d+)?)\s*%\s*(?:的)?\s*(?:{labels})",
        text,
        re.IGNORECASE,
    )
    return float(before.group(1)) / 100.0 if before else None


def _normalise_rsi_relation(value: str) -> str:
    return (
        "below"
        if value.lower() in {"低於", "小於", "低于", "below", "under"}
        else "above"
    )


def _requested_rsi_thresholds(text: str) -> tuple[tuple[str, float], ...]:
    """Extract explicit and inherited RSI thresholds in textual order."""
    thresholds = []
    for match in _RSI_THRESHOLD.finditer(text):
        if _is_excluded_match(text, match):
            continue
        thresholds.append(
            (_normalise_rsi_relation(match.group(1)), float(match.group(2)))
        )

        # A coordinated second comparison may omit the already-established
        # subject: "RSI below 30 long, above 70 short".  Match only from the end
        # of this concrete RSI comparison; do not search an arbitrary tail.
        cursor = match.end()
        while True:
            inherited = _RSI_FOLLOWUP_THRESHOLD.match(text, cursor)
            if inherited is None or _is_excluded_match(text, inherited):
                break
            thresholds.append(
                (_normalise_rsi_relation(inherited.group(1)),
                 float(inherited.group(2)))
            )
            cursor = inherited.end()

    return tuple(dict.fromkeys(thresholds))


_ENTRY_JOIN_DISQUALIFIER = re.compile(
    r"(?:停損|止損|停利|止盈|風控|风险|risk|stop(?:[- ]?loss)?|"
    r"take[- ]?profit|倉位|仓位|size|position)",
    re.IGNORECASE,
)


def _entry_condition_join(
        text: str,
        triggers: tuple[str, ...],
        rsi_thresholds: tuple[tuple[str, float], ...],
) -> str | None:
    """Infer AND/OR only when the connector joins recognised entry predicates.

    A global search for an OR word turns phrases such as stop loss or take
    profit into an OR entry rule. We inspect only gaps between recognised
    positive entry predicates: trigger↔RSI as well as RSI↔RSI.  The latter is
    necessary for ``RSI14 below 30 or above 70``; otherwise its second threshold
    is silently dropped from the boolean contract.  Risk/sizing clauses remain
    excluded from this inference.
    """
    if not rsi_thresholds:
        return None

    spans = []
    for name, rule in _TRIGGER_RULES:
        if name not in triggers:
            continue
        for match in rule.finditer(text):
            if not _is_excluded_match(text, match):
                spans.append((match.start(), match.end(), "trigger"))
    for match in _RSI_THRESHOLD.finditer(text):
        if not _is_excluded_match(text, match):
            spans.append((match.start(), match.end(), "rsi"))
            cursor = match.end()
            while True:
                inherited = _RSI_FOLLOWUP_THRESHOLD.match(text, cursor)
                if inherited is None or _is_excluded_match(text, inherited):
                    break
                # Only the comparison itself is an entry predicate.  The
                # connector before it is precisely the gap we need to read.
                spans.append((inherited.start(1), inherited.end(2), "rsi"))
                cursor = inherited.end()

    ordered = sorted(spans)
    if len(ordered) < 2:
        return None
    joins = []
    for (_left_start, left_end, left_kind), (right_start, _right_end, right_kind) in zip(
            ordered, ordered[1:]):
        gap = text[left_end:right_start]
        if _ENTRY_JOIN_DISQUALIFIER.search(gap):
            continue
        if _CONDITION_OR.search(gap):
            joins.append("any_of")
            continue
        # Two RSI comparisons only get an AND meaning when the connector
        # explicitly says so.  A comma alone may delimit separate long/short
        # clauses and must not be made into an impossible conjunctive entry.
        if _CONDITION_AND.search(gap):
            joins.append("all_of")
    if len(set(joins)) > 1:
        # A scalar ``condition_join`` can faithfully represent one connector
        # across all recognised leaves, but not e.g. EMA AND (RSI<30 OR
        # RSI>70).  Do not flatten that into an arbitrary any_of/all_of tree;
        # the preflight caller will ask for a form the current IR can prove.
        return "mixed"
    if joins:
        return joins[0]
    # Trigger + RSI historically has an AND default: it is one ordinary entry
    # sentence with two predicates. Preserve that narrow default; a pair of RSI
    # comparisons without an explicit connector remains unspecified instead.
    return "all_of" if triggers and rsi_thresholds else None


def _requested_market_type(text: str) -> str | None:
    if re.search(r"(?:現貨|\bspot\b)", text, re.IGNORECASE):
        return "spot"
    if re.search(r"(?:永續|合約|期貨|\b(?:futures?|perpetuals?)\b)", text, re.IGNORECASE):
        return "futures"
    return None


def _atr_trailing_matches(text: str, *, excluded: bool | None = None) -> tuple[re.Match, ...]:
    """Return ATR trailing-stop spans with their request-local polarity."""

    matches = []
    for match in _ATR_TRAILING_STOP.finditer(text):
        is_excluded = _is_excluded_match(text, match)
        if excluded is None or is_excluded == excluded:
            matches.append(match)
    return tuple(matches)


def _requested_atr_trailing_exit(text: str) -> tuple[str | None, int | None, float | None]:
    """Extract the fully-specified ATR trailing-stop contract when requested.

    ``atr_trailing_stop`` has two load-bearing parameters.  Treating a vague
    mention as a permission to apply the generic 2% stop is exactly the silent
    substitution this front door is meant to prevent, so a present-but-partial
    request intentionally returns ``(type, None, None)`` for the caller to
    clarify before generation.
    """
    matches = _atr_trailing_matches(text, excluded=False)
    if not matches:
        return None, None, None
    if len(matches) != 1:
        # One risk.exit can only carry one trailing configuration.  Taking the
        # first of several positive requests silently discards the rest, so use
        # the existing deterministic preflight to request a single plan.
        return "atr_trailing_stop", None, None
    # Bind parameters to the positive trailing-stop phrase rather than scanning
    # the whole request.  In "do not use ATR14 2x ...; use ATR10 1.5x ..." the
    # prohibited values must not become the live exit plan.  A multiplier can
    # begin just before ATR (``2x ATR14``), so overlap rather than containment
    # is the correct relation.
    match = matches[0]
    def overlaps(candidate: re.Match) -> bool:
        return candidate.start() < match.end() and match.start() < candidate.end()

    period_match = next(
        (candidate for candidate in _ATR_PERIOD.finditer(text) if overlaps(candidate)), None,
    )
    multiplier_match = next(
        (candidate for candidate in _ATR_TRAIL_MULTIPLIER.finditer(text) if overlaps(candidate)),
        None,
    )
    multiplier = None
    if multiplier_match is not None:
        value = multiplier_match.group("before") or multiplier_match.group("after")
        try:
            multiplier = float(value)
        except (TypeError, ValueError):
            multiplier = None
    period = None
    if period_match is not None:
        value = period_match.group("paren") or period_match.group("plain")
        try:
            period = int(value)
        except (TypeError, ValueError):
            period = None
    return (
        "atr_trailing_stop",
        period,
        multiplier,
    )


def _excluded_exit_types(text: str) -> tuple[str, ...]:
    """Return explicit exit mechanisms the user forbade, never silently omit."""

    # A later explicit positive ATR plan can qualify a preceding prohibition
    # ("not ATR14/2x; use ATR20/3x"). This coarse field records mechanism-level
    # exclusions, not parameter-level blacklists, so do not turn that explicit
    # replacement into an impossible request.
    return (
        ("atr_trailing_stop",)
        if (_atr_trailing_matches(text, excluded=True)
            and not _atr_trailing_matches(text, excluded=False))
        else ()
    )


def classify_request(nl: str) -> IntentDecision:
    """Classify scope and fail closed when the request is genuinely ambiguous.

    This is intentionally independent of the YAML returned by the model. A
    generated trade spec cannot validate its own claim that a basket request was
    actually a trade request.
    """
    text = " ".join(str(nl or "").split())
    def positive_matches(rule):
        return [match for match in rule.finditer(text)
                if not _is_excluded_match(text, match)]

    selection = [name for name, rule in _SELECTION_RULES if rule.search(text)]
    trade = [name for name, rule in _TRADE_RULES if positive_matches(rule)]
    plural_scope = bool(_PLURAL_SCOPE.search(text))
    rsi_thresholds = _requested_rsi_thresholds(text)
    exit_type, exit_atr_period, exit_trail_mult = _requested_atr_trailing_exit(text)
    excluded_exit_types = _excluded_exit_types(text)
    buy_point = bool(positive_matches(_CANDIDATE_BUY_POINT))
    sell_point = bool(positive_matches(_CANDIDATE_SELL_POINT))
    bidirectional_point = bool(positive_matches(_BIDIRECTIONAL_TRADE_POINT)) \
        or (buy_point and sell_point)
    has_rsi = bool(positive_matches(re.compile(
        r"(?<![A-Za-z])RSI(?:\s*[-_]?\s*\d{1,4})?", re.IGNORECASE)))
    named_candidate_indicators = {
        match.group(1).lower()
        for match in positive_matches(_TECHNICAL_NAME)
    }
    excluded_candidate_indicators = {
        match.group(1).lower()
        for match in _TECHNICAL_NAME.finditer(text)
        if _is_excluded_match(text, match)
    }
    generic_point_is_supported = (
        (buy_point or sell_point or bidirectional_point)
        and named_candidate_indicators.issubset({"rsi"})
        and "rsi" not in excluded_candidate_indicators
    )
    explicit_short = bool(positive_matches(_SHORT_DIRECTION))
    requested_market_type = _requested_market_type(text)
    compound_execution = bool(
        positive_matches(_EXECUTION_ACTION)
        or (
            (positive_matches(_CANDIDATE_DIRECTION_ACTION) or buy_point or sell_point)
            and rsi_thresholds
        )
    )
    # ``selection.candidate_trade`` can express an entry plan driven by the
    # current per-candidate indicator snapshot.  It still cannot express an EMA
    # *cross* over time, so a generic "pick then buy" or crossover request stays
    # closed instead of being upgraded merely because an indicator name appears.
    candidate_trade_requested = bool(
        selection
        and compound_execution
        and (
            (_CANDIDATE_TRADE_POINT.search(text)
             and has_rsi)
            or rsi_thresholds
            or generic_point_is_supported
        )
    )
    # A spot candidate cannot carry the inferred short entry used for a sell
    # point.  Stop before generation instead of silently changing spot to
    # futures or pretending that an open_short instruction means close_long.
    if candidate_trade_requested and requested_market_type == "spot" \
            and (sell_point or bidirectional_point or explicit_short):
        candidate_trade_requested = False

    if selection and compound_execution:
        if candidate_trade_requested:
            kind = "selection"
            extra = ["compound_select_then_trade"]
            if bidirectional_point:
                extra.append("bidirectional_trade_points")
            elif buy_point:
                extra.append("candidate_buy_point")
            elif sell_point:
                extra.append("candidate_sell_point")
            if not has_rsi:
                extra.append("default_candidate_indicator_rsi")
            if (sell_point or bidirectional_point) and not explicit_short:
                extra.append("sell_point_requires_short_interpretation")
            evidence = tuple(selection + trade + extra)
        else:
            # Direct execution without a per-candidate threshold, and temporal
            # crossovers that the snapshot grammar cannot represent, must still
            # stop before an LLM can invent a different rule.
            kind = "ambiguous"
            evidence = tuple(selection + trade + ["compound_selection_execution"])
    elif selection and (plural_scope or not trade):
        kind = "selection"
        evidence = tuple(selection + (["plural_scope"] if plural_scope else []))
    elif trade:
        kind = "trade"
        evidence = tuple(trade)
    elif selection:
        kind = "selection"
        evidence = tuple(selection)
    else:
        kind = "ambiguous"
        evidence = ()

    # Keep this fact in the independently-derived intent so every caller of
    # ``reconcile_intent`` can refuse a model that smuggles the down-cross into
    # a long entry.  The demo front door also sees it before sending any text to
    # an external model.
    if kind == "trade" and positive_matches(_BARE_SELL_ACTION) and not explicit_short:
        evidence = tuple((*evidence, "bare_sell_direction_ambiguous"))

    unsupported = []
    if _UNSUPPORTED_DISCOVERY.search(text):
        unsupported.append("under_discovered")
    unsupported.extend(
        "gap:%s" % gap_id
        for gap_id, pattern in _NAMED_UNSUPPORTED_GAPS
        if pattern.search(text)
    )
    if kind == "selection" and _CROSS_SECTION_LIQUIDATION.search(text):
        unsupported.append("gap:GAP-LIQUIDATION-CROSS-SECTION")
    # A request such as "排除 TradFi / 股票代幣 / 穩定幣" is not safely
    # equivalent to an arbitrary string comparison.  Retain the category so the
    # front door can stop before an LLM invents a mapping to exchange metadata.
    unsupported.extend(
        "category_exclusion:%s" % name for name in _category_exclusions(text)
    )
    source_mentions = _source_mentions(text)
    sources = {name for name, excluded in source_mentions if not excluded}
    excluded_sources = {name for name, excluded in source_mentions if excluded}
    if _VOLUME_RANKING.search(text):
        ranking_metric = "liquidity"
        # In "hot coins by volume", hot means high activity; it is not enough
        # evidence to force a Square/news dependency when volume is explicit.
        if "news" in sources and not _EXPLICIT_NEWS_SOURCE.search(text):
            sources.discard("news")
    elif _NEWS_SENTIMENT_METRIC.search(text):
        ranking_metric = "sentiment"
    elif _NEWS_MENTION_METRIC.search(text):
        ranking_metric = "mentions"
    else:
        ranking_metric = None
    sources = frozenset(sources)
    excluded_sources = frozenset(excluded_sources)

    technical_periods = []
    excluded_technical_periods = []
    for match in _TECHNICAL_PERIOD.finditer(text):
        item = (match.group(1).lower(), int(match.group(2)))
        # ``ATR14 2x trailing stop`` configures the exit engine, whose plugin
        # computes ATR itself.  It is not a request to add a separate ATR entry
        # indicator, and treating it as one made a faithful trailing-stop YAML
        # fail reconciliation for omitting an unused indicator.
        if exit_type == "atr_trailing_stop" and item == ("atr", exit_atr_period):
            continue
        target = excluded_technical_periods if _is_excluded_match(text, match) \
            else technical_periods
        if item not in target:
            target.append(item)

    indicator_names = []
    excluded_indicator_names = []
    for match in _TECHNICAL_NAME.finditer(text):
        name = match.group(1).lower()
        if exit_type == "atr_trailing_stop" and name == "atr":
            continue
        target = excluded_indicator_names if _is_excluded_match(text, match) \
            else indicator_names
        if name not in target:
            target.append(name)

    supertrend_parameters = []
    for match in _SUPERTREND_PARAMETERS.finditer(text):
        if _is_excluded_match(text, match):
            continue
        item = (int(match.group(1)), float(match.group(2)))
        if item not in supertrend_parameters:
            supertrend_parameters.append(item)

    directions = []
    excluded_directions = []
    for direction, pattern in (("long", _LONG_DIRECTION), ("short", _SHORT_DIRECTION)):
        matches = list(pattern.finditer(text))
        if any(not _is_excluded_match(text, match) for match in matches):
            directions.append(direction)
        if any(_is_excluded_match(text, match) for match in matches):
            excluded_directions.append(direction)
    for direction in _supertrend_directions(text):
        if direction not in directions:
            directions.append(direction)
    symbol_mentions = _symbol_mentions(text)
    included_symbols = tuple(
        dict.fromkeys(symbol for symbol, excluded in symbol_mentions if not excluded)
    )
    excluded_symbols = tuple(
        dict.fromkeys(symbol for symbol, excluded in symbol_mentions if excluded)
    )
    news_metrics = []
    if "news" in sources and _NEWS_MENTION_METRIC.search(text) \
            and ranking_metric != "liquidity":
        news_metrics.append("mentions")
    if "news" in sources and _NEWS_SENTIMENT_METRIC.search(text):
        news_metrics.append("sentiment")
    # One unambiguous direction phrase, or none at all. "最低與最高 funding" and
    # "漲幅或跌幅" match two rules that contradict each other; guessing one would
    # invent a preference the user never stated, and the downstream check would
    # then reject a spec for disagreeing with a coin flip.
    ordered = {(column, order) for column, order, rule in _SCORE_ORDER_RULES
               if rule.search(text)}
    if _DIRECTION_EXCLUSION.search(text) or _DIMINISHED_SIGN.search(text):
        ordered = set()
    score_order_metric, score_order = ordered.pop() if len(ordered) == 1 else (None, None)
    long_short_requirement = _long_short_min_account_requirement(text)
    triggers = tuple(
        name for name, rule in _TRIGGER_RULES
        if any(not _is_excluded_match(text, match) for match in rule.finditer(text))
    )
    # A request that names a temporal trigger and a threshold states two entry
    # predicates.  In ordinary strategy wording their default composition is
    # AND; the only supported opposite reading must be spelled with an OR term.
    # Recording this before YAML generation lets reconciliation prove that an
    # ``any_of`` did not turn an AND strategy into two independent entries.
    condition_join = (
        _entry_condition_join(text, triggers, rsi_thresholds)
        if kind == "trade" else None
    )
    return IntentDecision(
        kind=kind,
        evidence=evidence,
        requested_count=_requested_count(text),
        sources=sources,
        excluded_sources=excluded_sources,
        bullish_preference=_generic_bullish_preference(text),
        unsupported_preferences=tuple(unsupported),
        candidate_trade_requested=candidate_trade_requested,
        long_short_min_account_pct=(long_short_requirement[0]
                                    if long_short_requirement is not None else None),
        long_short_min_account_operator=(long_short_requirement[1]
                                         if long_short_requirement is not None else None),
        supertrend_parameters=supertrend_parameters,
        named_symbols=tuple(dict.fromkeys(symbol for symbol, _ in symbol_mentions)),
        included_symbols=included_symbols,
        excluded_symbols=excluded_symbols,
        intervals=_requested_intervals(text),
        market_type=requested_market_type,
        technical_periods=technical_periods,
        excluded_technical_periods=excluded_technical_periods,
        stop_pct=_percent_near(text, r"停損|止損|stop[- ]?loss"),
        tp_pct=_percent_near(text, r"停利|止盈|take[- ]?profit"),
        size_fraction=_percent_near(text, r"倉位|資金|投入|size|position[\s_-]*size"),
        directions=directions,
        excluded_directions=excluded_directions,
        news_metrics=news_metrics,
        triggers=triggers,
        condition_join=condition_join,
        indicator_names=indicator_names,
        excluded_indicator_names=excluded_indicator_names,
        rsi_thresholds=rsi_thresholds,
        ranking_metric=ranking_metric,
        score_order=score_order,
        score_order_metric=score_order_metric,
        exit_type=exit_type,
        exit_atr_period=exit_atr_period,
        exit_trail_mult=exit_trail_mult,
        excluded_exit_types=excluded_exit_types,
    )


def infer_strategy_kind(nl: str) -> str:
    """Compatibility helper used by the HTTP prompt route and tests."""
    return classify_request(nl).kind


def generated_strategy_kind(spec: dict) -> str | None:
    """Which strategy shape the model actually emitted, or None if mixed."""
    has_selection = isinstance(spec.get("selection"), dict)
    has_trade = isinstance(spec.get("signals"), dict)
    if has_selection and not has_trade:
        return "selection"
    if has_trade and not has_selection:
        return "trade"
    return None


def _base_asset(value: str) -> str:
    """Use the same pair-to-base normalisation as the news join and dedupe."""
    from cyqnt_trd.blocks.news_feed import base_token

    return base_token(str(value).upper())


def _close_enough(actual, expected: float) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _greater_than(actual, threshold: float) -> bool:
    try:
        return float(actual) > float(threshold)
    except (TypeError, ValueError):
        return False


def _feature_dependencies(selection: dict, token, seen=None) -> set[str]:
    """Resolve a selection feature reference to the frame columns it reads.

    Checking feature *names* is not sufficient: a model could name a
    quote-volume feature ``news_bull_ratio``. Only the inputs in this dependency
    graph count as evidence that the requested data affects ranking/direction.
    """
    if not isinstance(token, str):
        return set()
    features = selection.get("features") or {}
    if token not in features or not isinstance(features[token], dict):
        return {token}
    seen = set() if seen is None else set(seen)
    if token in seen:
        return set()
    seen.add(token)
    feature = features[token]
    if "inputs" in feature and isinstance(feature["inputs"], (list, tuple)):
        refs = list(feature["inputs"])
    elif "input" in feature:
        refs = [feature["input"]]
    else:
        refs = ["close"]
    out: set[str] = set()
    for ref in refs:
        out.update(_feature_dependencies(selection, ref, seen))
    return out


def _condition_dependencies(selection: dict, node) -> set[str]:
    if isinstance(node, list):
        out: set[str] = set()
        for item in node:
            out.update(_condition_dependencies(selection, item))
        return out
    if not isinstance(node, dict):
        return set()
    out: set[str] = set()
    args = node.get("args")
    if isinstance(args, (list, tuple)):
        for ref in args:
            out.update(_feature_dependencies(selection, ref))
    for key, value in node.items():
        if key not in {"args", "params", "cond"}:
            out.update(_condition_dependencies(selection, value))
    return out


def _selection_usage(selection: dict):
    steps = [step for step in (selection.get("universe") or [])
             if isinstance(step, dict)]
    blocks = {str(step.get("block") or "") for step in steps}
    score_dependencies = _feature_dependencies(selection, selection.get("score"))
    direction_dependencies = set()
    for key in ("long_when", "short_when"):
        direction_dependencies.update(_condition_dependencies(selection, selection.get(key)))
    return steps, blocks, score_dependencies, direction_dependencies


def _selection_indicator_specs(steps) -> tuple[tuple[str, object, str], ...]:
    """``(indicator, period, output_alias)`` for real per-candidate indicator steps."""

    out = []
    for step in steps:
        if step.get("block") != "universe.augment_with_indicator":
            continue
        params = step.get("params") or {}
        if not isinstance(params, dict):
            continue
        name = str(params.get("indicator") or "").lower()
        timeframe = str(params.get("timeframe") or "")
        alias = str(params.get("as") or "%s_%s" % (name, timeframe))
        out.append((name, params.get("period"), alias))
    return tuple(out)


def _selection_supertrend_aliases(
    steps, *, period: int, multiplier: float, timeframe: str,
) -> set[str]:
    """Aliases from exact, runnable Supertrend-direction universe steps.

    ``output: 1`` is not cosmetic: output zero is the Supertrend price line,
    which is positive whether the trend is bearish or bullish.  Treating it as
    a direction would make a structurally valid selector answer the opposite
    semantic question.
    """

    aliases = set()
    for step in steps:
        if step.get("block") != "universe.augment_with_indicator":
            continue
        if not isinstance(step.get("with"), (list, tuple)) \
                or "universe_bars" not in step.get("with"):
            continue
        params = step.get("params") or {}
        if not isinstance(params, dict):
            continue
        if str(params.get("indicator") or "").lower() != "supertrend":
            continue
        if str(params.get("timeframe") or "").lower() != timeframe:
            continue
        if not _close_enough(params.get("period"), period) \
                or not _close_enough(params.get("multiplier"), multiplier) \
                or not _close_enough(params.get("output"), 1):
            continue
        aliases.add(str(params.get("as") or "supertrend_%s" % timeframe))
    return aliases


def _mandatory_supertrend_aliases(node, *, aliases: set[str], relation: str) -> set[str] | None:
    """Return direction aliases every truth path of *node* must satisfy.

    Presence anywhere in an AST is insufficient for a ``同時`` request.  For
    example, ``any_of: [st_4h < 0, quoteVolume > 0]`` mentions ``st_4h`` but a
    liquid asset can pass without being bearish on 4h.  Boolean necessity is
    deliberately conservative:

    * an ``all_of`` needs the union of every child requirement;
    * an ``any_of`` needs only the intersection shared by *every* branch; and
    * ``not`` / ``exclude_when`` reject the proof because a direction alias
      nested there does not establish the requested positive direction.

    ``None`` means the expression has a form this semantic checker refuses to
    reason about.  The YAML interpreter may still run such a form, but it may
    not claim to meet a multi-timeframe Supertrend request.
    """
    if not isinstance(node, dict):
        return None

    if "cond" in node:
        if not str(node.get("cond") or "").endswith(relation):
            return set()
        args = node.get("args") or []
        if (not isinstance(args, (list, tuple)) or not args
                or not _close_enough(_leaf_threshold(node), 0)):
            return set()
        alias = str(args[0])
        return {alias} if alias in aliases else set()

    if "all_of" in node:
        children = node.get("all_of")
        if not isinstance(children, list) or not children:
            return None
        required: set[str] = set()
        for child in children:
            child_required = _mandatory_supertrend_aliases(
                child, aliases=aliases, relation=relation)
            if child_required is None:
                return None
            required.update(child_required)
        return required

    if "any_of" in node:
        children = node.get("any_of")
        if not isinstance(children, list) or not children:
            return None
        branch_requirements = []
        for child in children:
            child_required = _mandatory_supertrend_aliases(
                child, aliases=aliases, relation=relation)
            if child_required is None:
                return None
            branch_requirements.append(child_required)
        return set.intersection(*branch_requirements)

    # The requested "all timeframes bearish/bullish" property is positive. A
    # negated or exclusion tree may be a legitimate strategy expression, but it
    # cannot be mechanically shown to mean the requested Supertrend state.
    if "not" in node or "exclude_when" in node:
        return None
    return None


def _iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _condition_leaves(node):
    if isinstance(node, list):
        for item in node:
            yield from _condition_leaves(item)
    elif isinstance(node, dict):
        if isinstance(node.get("cond"), str):
            yield node
        for key, value in node.items():
            if key not in {"cond", "args", "params"}:
                yield from _condition_leaves(value)


def _condition_required_on_every_path(node, predicate) -> bool:
    """Whether a leaf matching *predicate* is necessary for every true path.

    This is the Boolean dual of a flat leaf search.  ``all_of`` needs every
    child, so a match in any required child is mandatory; ``any_of`` offers
    alternate true paths, so every branch must itself require the match.  It is
    deliberately conservative for negation/exclusion trees: those forms may be
    valid YAML, but they cannot prove that a positively requested condition is
    always enforced.
    """
    if not isinstance(node, dict):
        return False
    if "cond" in node:
        return bool(predicate(node))
    if "all_of" in node:
        children = node.get("all_of")
        return (isinstance(children, list) and bool(children)
                and any(_condition_required_on_every_path(child, predicate)
                        for child in children))
    if "any_of" in node:
        children = node.get("any_of")
        return (isinstance(children, list) and bool(children)
                and all(_condition_required_on_every_path(child, predicate)
                        for child in children))
    return False


def _condition_requested_paths(node, predicates):
    """Return the requested-predicate set for every satisfiable AST branch.

    The result is a small symbolic proof, not an evaluation over market data.
    A leaf contributes the requested predicates it fulfils; all_of unions its
    child requirements and any_of keeps alternatives separate. Negation and
    exclusion forms are deliberately unprovable for a positive user request.
    """
    if not isinstance(node, dict):
        return None
    if "cond" in node:
        return {frozenset(
            name for name, matcher in predicates if matcher(node)
        )}
    if "all_of" in node:
        children = node.get("all_of")
        if not isinstance(children, list) or not children:
            return None
        paths = {frozenset()}
        for child in children:
            child_paths = _condition_requested_paths(child, predicates)
            if child_paths is None:
                return None
            paths = {
                left | right for left in paths for right in child_paths
            }
            # An adversarially huge expression is not a semantic proof. The
            # YAML grammar normally has only a handful of predicates, so this
            # bound is never reached by ordinary strategies.
            if len(paths) > 128:
                return None
        return paths
    if "any_of" in node:
        children = node.get("any_of")
        if not isinstance(children, list) or not children:
            return None
        paths = set()
        for child in children:
            child_paths = _condition_requested_paths(child, predicates)
            if child_paths is None:
                return None
            paths.update(child_paths)
            if len(paths) > 128:
                return None
        return paths
    return None


def _indicator_aliases(indicators: dict, name: str, period=None) -> set[str]:
    out = set()
    expected_block = "indicators.%s" % name
    for alias, item in indicators.items():
        if not isinstance(item, dict) or str(item.get("block") or "").lower() != expected_block:
            continue
        if period is not None and not _close_enough((item.get("params") or {}).get("period"), period):
            continue
        out.add(str(alias))
    return out


def _leaf_threshold(leaf: dict):
    params = leaf.get("params") or {}
    if "threshold" in params:
        return params.get("threshold")
    args = leaf.get("args") or []
    return args[1] if isinstance(args, (list, tuple)) and len(args) > 1 else None


def _reconcile_trade(intent: IntentDecision, spec: dict) -> list[str]:
    errors: list[str] = []
    data = spec.get("data") or {}
    actual_symbol = str(data.get("symbol") or "").upper()

    if intent.condition_join == "mixed":
        errors.append(
            "需求的 entry 條件混合了 AND 與 OR；目前扁平 intent 無法證明其巢狀分組，"
            "拒絕將它降格為任一 all_of/any_of 規則"
        )

    if "bare_sell_direction_ambiguous" in intent.evidence:
        errors.append(
            "需求中的『賣出/sell』可能是平多或開空；請明確指定 close_long 或 open_short，"
            "系統不會把下穿條件偷偷併入開多 entry"
        )

    if intent.included_symbols:
        requested_bases = {_base_asset(item) for item in intent.included_symbols}
        if len(requested_bases) > 1:
            errors.append(
                "trade YAML 一次只能執行一個標的,但需求指定了多個標的: %s"
                % ", ".join(intent.included_symbols)
            )
        elif _base_asset(actual_symbol) not in requested_bases:
            errors.append(
                "使用者指定標的是 %s,但 YAML 的 data.symbol=%s"
                % (", ".join(intent.included_symbols), actual_symbol or None)
            )
    excluded_bases = {_base_asset(item) for item in intent.excluded_symbols}
    if actual_symbol and _base_asset(actual_symbol) in excluded_bases:
        errors.append(
            "使用者明確排除標的 %s,但 YAML 的 data.symbol=%s"
            % (", ".join(intent.excluded_symbols), actual_symbol)
        )

    if intent.intervals:
        primary = str((data.get("primary") or {}).get("interval") or "").lower()
        htf = {
            str(item.get("interval") or "").lower()
            for item in (data.get("htf") or []) if isinstance(item, dict)
        }
        if len(intent.intervals) == 1 and primary != intent.intervals[0]:
            errors.append(
                "使用者指定週期 %s,但 YAML 的 primary.interval=%s"
                % (intent.intervals[0], primary or None)
            )
        elif len(intent.intervals) > 1:
            missing = sorted(set(intent.intervals) - ({primary} | htf))
            if missing:
                errors.append("YAML 未包含使用者指定的週期: %s" % ", ".join(missing))

    if intent.market_type and str(data.get("market_type") or "").lower() != intent.market_type:
        errors.append(
            "使用者指定 market_type=%s,但 YAML 是 %s"
            % (intent.market_type, data.get("market_type"))
        )

    signals = spec.get("signals") or {}
    indicators = signals.get("indicators") or {}
    for indicator_name, period in intent.technical_periods:
        matched = any(
            isinstance(item, dict)
            and str(item.get("block") or "").lower() == "indicators.%s" % indicator_name
            and _close_enough((item.get("params") or {}).get("period"), period)
            for item in indicators.values()
        )
        if not matched:
            errors.append(
                "需求指定 %s%d,但 YAML 沒有使用對應的 Block period"
                % (indicator_name.upper(), period)
            )

    period_names = {name for name, _period in intent.technical_periods}
    for indicator_name in set(intent.indicator_names) - period_names:
        if not _indicator_aliases(indicators, indicator_name):
            errors.append(
                "需求指定 %s,但 YAML 沒有使用對應的 indicator Block"
                % indicator_name.upper()
            )

    excluded_period_names = {
        name for name, _period in intent.excluded_technical_periods
    }
    for indicator_name, period in intent.excluded_technical_periods:
        if _indicator_aliases(indicators, indicator_name, period):
            errors.append(
                "使用者明確要求不要使用 %s%d,但 YAML 仍宣告了該 indicator Block"
                % (indicator_name.upper(), period)
            )
    for indicator_name in set(intent.excluded_indicator_names) - excluded_period_names:
        if _indicator_aliases(indicators, indicator_name):
            errors.append(
                "使用者明確要求不要使用 %s,但 YAML 仍宣告了該 indicator Block"
                % indicator_name.upper()
            )

    entry = signals.get("entry") or {}
    if "long" in intent.directions and not entry.get("long"):
        errors.append("需求包含做多/買進,但 YAML 沒有 signals.entry.long")
    if "short" in intent.directions and not entry.get("short"):
        errors.append("需求包含做空,但 YAML 沒有 signals.entry.short")
    if "long" in intent.excluded_directions and entry.get("long"):
        errors.append("使用者明確要求不要做多,但 YAML 仍包含 signals.entry.long")
    if "short" in intent.excluded_directions and entry.get("short"):
        errors.append("使用者明確要求不要做空,但 YAML 仍包含 signals.entry.short")
    if intent.directions == ("long",) and entry.get("short"):
        errors.append("使用者只要求做多/買進,但模型擅自加入 signals.entry.short")
    if intent.directions == ("short",) and entry.get("long"):
        errors.append("使用者只要求做空,但模型擅自加入 signals.entry.long")
    if (not intent.directions and not intent.excluded_directions
            and entry.get("short")):
        errors.append(
            "使用者未指定方向時系統預設 long-only,但模型擅自加入 "
            "signals.entry.short"
        )

    requested_nodes = [entry.get(side) for side in intent.directions if entry.get(side)]
    if not requested_nodes:
        requested_nodes = [value for value in entry.values() if isinstance(value, dict)]

    def required_on_every_entry_path(predicate) -> bool:
        return bool(requested_nodes) and all(
            _condition_required_on_every_path(node, predicate)
            for node in requested_nodes
        )

    leaves = [leaf for node in requested_nodes for leaf in _condition_leaves(node)]
    condition_refs = {str(leaf.get("cond")) for leaf in leaves}
    used_args = {
        str(arg) for leaf in leaves for arg in (leaf.get("args") or [])
        if isinstance(arg, str)
    }

    for indicator_name, period in intent.technical_periods:
        aliases = _indicator_aliases(indicators, indicator_name, period)
        if aliases and not aliases.intersection(used_args):
            errors.append(
                "YAML 雖宣告 %s%d,但使用者要求的 entry 條件沒有引用它"
                % (indicator_name.upper(), period)
            )
    for indicator_name in set(intent.indicator_names) - period_names:
        aliases = _indicator_aliases(indicators, indicator_name)
        if aliases and not aliases.intersection(used_args):
            errors.append(
                "YAML 雖宣告 %s,但使用者要求的 entry 條件沒有引用它"
                % indicator_name.upper()
            )

    requested_indicators = {name for name, _period in intent.technical_periods}
    or_predicates = []
    for trigger in intent.triggers:
        if trigger == "cross_above":
            accepted = (
                {"conditions.ma_cross_above"}
                if requested_indicators & {"ema", "sma"}
                else {"conditions.ma_cross_above", "conditions.macd_golden_cross"}
            )
        elif trigger == "cross_below":
            accepted = (
                {"conditions.ma_cross_below"}
                if requested_indicators & {"ema", "sma"}
                else {"conditions.ma_cross_below", "conditions.macd_death_cross"}
            )
        elif trigger == "breakout_high":
            accepted = {"conditions.breakout_high"}
        else:
            accepted = {"conditions.breakout_low"}
        if trigger in {"cross_above", "cross_below"} and "vwap" in intent.indicator_names:
            # ``close_above`` / ``close_below`` describe a persistent state,
            # not an edge.  Treating either as an 上穿/下穿 lets a strategy
            # enter repeatedly long after the crossing bar.  The generic MA
            # cross block accepts close and VWAP series, and the range-based
            # block is a legitimate intrabar touch/cross reading when its
            # direction is explicit.
            accepted = set(accepted) | {
                "conditions.ma_cross_above" if trigger == "cross_above"
                else "conditions.ma_cross_below",
                "conditions.price_touch_or_cross",
            }
        trigger_matcher = (
            lambda leaf, accepted=frozenset(accepted): leaf.get("cond") in accepted
        )
        or_predicates.append(("trigger:%s" % trigger, trigger_matcher))
        if not accepted.intersection(condition_refs):
            errors.append(
                "需求指定 %s,但 YAML entry 沒有使用對應條件 Block (%s)"
                % (trigger, ", ".join(sorted(accepted)))
            )
        elif intent.condition_join == "all_of" and not required_on_every_entry_path(
                trigger_matcher):
            errors.append(
                "需求以 AND 同時要求 %s,但 YAML 的 any_of/分支可在沒有該條件時進場"
                % trigger
            )

        ma_periods = [
            (name, period) for name, period in intent.technical_periods
            if name in {"ema", "sma"}
        ]
        if trigger in {"cross_above", "cross_below"} and len(ma_periods) >= 2:
            first_name, first_period = ma_periods[0]
            second_name, second_period = ma_periods[1]
            first_aliases = _indicator_aliases(indicators, first_name, first_period)
            second_aliases = _indicator_aliases(indicators, second_name, second_period)
            required = (
                "conditions.ma_cross_above" if trigger == "cross_above"
                else "conditions.ma_cross_below"
            )
            wired = any(
                leaf.get("cond") == required
                and isinstance(leaf.get("args"), (list, tuple))
                and len(leaf["args"]) >= 2
                and str(leaf["args"][0]) in first_aliases
                and str(leaf["args"][1]) in second_aliases
                for leaf in leaves
            )
            if not wired:
                errors.append(
                    "需求指定 %s%d 與 %s%d 的 %s,但 entry args 沒有接到這兩個指標"
                    % (first_name.upper(), first_period, second_name.upper(),
                       second_period, trigger)
                )

        if trigger in {"cross_above", "cross_below"} and "macd" in intent.indicator_names:
            line_aliases = {
                alias for alias in _indicator_aliases(indicators, "macd")
                if int((indicators[alias].get("output", 0))) == 0
            }
            signal_aliases = {
                alias for alias in _indicator_aliases(indicators, "macd")
                if int((indicators[alias].get("output", 0))) == 1
            }
            required = (
                "conditions.macd_golden_cross" if trigger == "cross_above"
                else "conditions.macd_death_cross"
            )
            wired = any(
                leaf.get("cond") == required
                and isinstance(leaf.get("args"), (list, tuple))
                and len(leaf["args"]) >= 2
                and str(leaf["args"][0]) in line_aliases
                and str(leaf["args"][1]) in signal_aliases
                for leaf in leaves
            )
            if not wired:
                errors.append(
                    "需求指定 MACD %s,但 entry 沒有引用 MACD line/signal 輸出"
                    % trigger
                )

        if trigger in {"cross_above", "cross_below"} and "vwap" in intent.indicator_names:
            vwap_aliases = _indicator_aliases(indicators, "vwap")
            crossover_condition = (
                "conditions.ma_cross_above" if trigger == "cross_above"
                else "conditions.ma_cross_below"
            )
            touch_direction = "up" if trigger == "cross_above" else "down"

            def vwap_matcher(
                    leaf, aliases=frozenset(vwap_aliases),
                    crossover_condition=crossover_condition, touch_direction=touch_direction,
            ):
                args = leaf.get("args")
                if (not isinstance(args, (list, tuple)) or len(args) < 2
                        or str(args[1]) not in aliases):
                    return False
                if (leaf.get("cond") == crossover_condition
                        and str(args[0]) == "close"):
                    return True
                return (
                    leaf.get("cond") == "conditions.price_touch_or_cross"
                    and str(args[0]) == "df"
                    and str((leaf.get("params") or {}).get("direction") or "").lower()
                    == touch_direction
                )

            if not any(vwap_matcher(leaf) for leaf in leaves):
                errors.append(
                    "需求指定價格 %s VWAP,但 entry 必須以 conditions.%s(close, vwap)"
                    " 或 conditions.price_touch_or_cross(df, vwap, direction=%s)"
                    % ("上穿" if trigger == "cross_above" else "下穿",
                       crossover_condition.rsplit(".", 1)[1], touch_direction)
                )
            elif intent.condition_join == "all_of" and not required_on_every_entry_path(
                    vwap_matcher):
                errors.append(
                    "需求以 AND 同時要求價格 %s VWAP,但 YAML 的 any_of/分支可繞過 VWAP"
                    % ("上穿" if trigger == "cross_above" else "下穿")
                )
            if intent.condition_join == "any_of":
                or_predicates[-1] = ("trigger:%s" % trigger, vwap_matcher)

    rsi_periods = [period for name, period in intent.technical_periods if name == "rsi"]
    rsi_aliases = set()
    for period in rsi_periods or [None]:
        rsi_aliases.update(_indicator_aliases(indicators, "rsi", period))
    for relation, value in intent.rsi_thresholds:
        acceptable = (
            {"conditions.rsi_oversold", "conditions.value_below"}
            if relation == "below"
            else {"conditions.rsi_overbought", "conditions.value_above"}
        )
        rsi_matcher = (
            lambda leaf, acceptable=frozenset(acceptable), aliases=frozenset(rsi_aliases),
            value=value: (
                leaf.get("cond") in acceptable
                and isinstance(leaf.get("args"), (list, tuple))
                and bool(leaf.get("args"))
                and str(leaf["args"][0]) in aliases
                and _close_enough(_leaf_threshold(leaf), value)
            )
        )
        or_predicates.append((
            "rsi:%s:%.12g" % (relation, value), rsi_matcher,
        ))
        wired = any(rsi_matcher(leaf) for leaf in leaves)
        if not wired:
            errors.append(
                "需求指定 RSI %s %.4g,但 entry 沒有以該 RSI 與門檻建立條件"
                % (relation, value)
            )
        elif intent.condition_join == "all_of" and not required_on_every_entry_path(
                rsi_matcher):
            errors.append(
                "需求以 AND 同時要求 RSI %s %.4g,但 YAML 的 any_of/分支可繞過該門檻"
                % (relation, value)
            )

    if intent.condition_join == "any_of":
        for node in requested_nodes:
            paths = _condition_requested_paths(node, or_predicates)
            independent = (
                paths is not None
                and all(paths)
                and all(
                    any(path == frozenset({name}) for path in paths)
                    for name, _matcher in or_predicates
                )
            )
            if not independent:
                errors.append(
                    "需求以 OR 要求每個 entry 條件可各自成立,但 YAML 沒有兩條"
                    "獨立且無可繞過分支的進場路徑"
                )
                break

    exit_cfg = (spec.get("risk") or {}).get("exit") or {}
    if intent.exit_type == "atr_trailing_stop":
        if exit_cfg.get("type") != "atr_trailing_stop":
            errors.append(
                "使用者要求 ATR 追蹤停損,但 YAML risk.exit.type 必須是 atr_trailing_stop"
            )
        if not _close_enough(exit_cfg.get("atr_period"), intent.exit_atr_period):
            errors.append(
                "使用者指定 ATR period=%r,但 YAML risk.exit.atr_period=%r"
                % (intent.exit_atr_period, exit_cfg.get("atr_period"))
            )
        if not _close_enough(exit_cfg.get("trail_mult"), intent.exit_trail_mult):
            errors.append(
                "使用者指定 ATR trail_mult=%r,但 YAML risk.exit.trail_mult=%r"
                % (intent.exit_trail_mult, exit_cfg.get("trail_mult"))
            )
    if (intent.stop_pct is not None or intent.tp_pct is not None) \
            and exit_cfg.get("type") != "pct_stop_tp":
        errors.append(
            "使用者指定百分比停損/停利時 risk.exit.type 必須是 pct_stop_tp，"
            "其他 exit type 不會套用 stop_pct/tp_pct"
        )
    for label, expected, key in (
        ("停損", intent.stop_pct, "stop_pct"),
        ("停利", intent.tp_pct, "tp_pct"),
    ):
        if expected is not None and not _close_enough(exit_cfg.get(key), expected):
            errors.append(
                "使用者指定%s %.4g,但 YAML risk.exit.%s=%r"
                % (label, expected, key, exit_cfg.get(key))
            )
    if intent.size_fraction is not None:
        actual_size = (spec.get("sizing") or {}).get("size")
        if not _close_enough(actual_size, intent.size_fraction):
            errors.append(
                "使用者指定倉位 %.4g,但 YAML sizing.size=%r"
                % (intent.size_fraction, actual_size)
            )

    functional_tokens = set(_iter_strings(signals))
    source_is_used = {
        "funding": any("funding" in item for item in functional_tokens),
        "open_interest": any(
            "open_interest" in item or "oi_" in item for item in functional_tokens),
        "liquidity": any(
            "quote_volume" in item or "liquidity" in item or "volume" in item
            for item in functional_tokens),
        "news": any("news_" in item for item in functional_tokens),
    }
    for source in sorted(intent.excluded_sources):
        if source_is_used.get(source, False):
            errors.append(
                "使用者明確要求不要使用 %s,但 YAML 仍宣告或讀取該資料來源"
                % source
            )
    if "funding" in intent.sources and not any("funding_rate" in item for item in functional_tokens):
        errors.append("需求指定 funding rate,但交易規則沒有實際讀取 funding_rate 欄位")
    if "open_interest" in intent.sources and not any(
        "open_interest" in item for item in functional_tokens
    ):
        errors.append("需求指定 open interest,但交易規則沒有實際讀取 open_interest 欄位")
    if "liquidity" in intent.sources and not any(
        "quote_volume" in item or "liquidity" in item for item in functional_tokens
    ):
        errors.append("需求指定成交量/流動性,但交易規則沒有實際讀取對應欄位或 Block")
    return errors


def _reconcile_excluded_exit_types(intent: IntentDecision, spec: dict) -> list[str]:
    """Reject an execution mechanism the user explicitly ruled out."""

    actual_type = ((spec.get("risk") or {}).get("exit") or {}).get("type")
    errors = []
    if "atr_trailing_stop" in intent.excluded_exit_types \
            and actual_type == "atr_trailing_stop":
        errors.append(
            "使用者明確要求不要使用 ATR 追蹤停損,但 YAML risk.exit.type 是 "
            "atr_trailing_stop"
        )
    return errors


def reconcile_intent(intent: IntentDecision, spec: dict) -> tuple[list[str], list[str]]:
    """Check meaning and data dependencies after structural validation."""
    errors: list[str] = []
    warnings: list[str] = []
    generated = generated_strategy_kind(spec)

    if generated != intent.kind:
        errors.append(
            "使用者需求是 %s,但模型產生的是 %s;拒絕把需求改成另一種策略"
            % (intent.kind, generated or "mixed/unknown")
        )
        return errors, warnings

    if intent.kind == "trade":
        if "news" in intent.sources:
            errors.append(
                "目前 trade YAML 的逐根 make_signals(df) 路徑不能讀取新聞 EventFrame;"
                "拒絕用 EMA/RSI 等技術指標冒充新聞交易條件"
            )
        errors.extend(_reconcile_trade(intent, spec))
        errors.extend(_reconcile_excluded_exit_types(intent, spec))
        return errors, warnings

    if intent.kind != "selection":
        return errors, warnings

    errors.extend(_reconcile_excluded_exit_types(intent, spec))

    selection = spec["selection"]
    steps, blocks, score_dependencies, direction_dependencies = _selection_usage(selection)
    functional_dependencies = score_dependencies | direction_dependencies

    # Positive-source checks below prove a requested source reaches ranking or
    # direction. The inverse is equally important: a model must not sneak a
    # forbidden source into an otherwise plausible selector, even as an unused
    # universe augmentation that makes future edits accidentally depend on it.
    source_is_used = {
        "news": (
            "universe.augment_with_news" in blocks
            or bool({"universe.filter_sentiment", "universe.top_mentioned",
                     "universe.top_bullish"}.intersection(blocks))
            or any(item.startswith("news_") for item in functional_dependencies)
        ),
        "funding": (
            "universe.augment_with_funding" in blocks
            or "universe.filter_funding_rate" in blocks
            or any("funding" in item.lower() for item in functional_dependencies)
        ),
        "open_interest": (
            bool({"universe.augment_with_open_interest", "universe.augment_with_oi_change",
                  "universe.filter_open_interest", "universe.filter_oi_change"}
                 .intersection(blocks))
            or any(
                "open_interest" in item.lower() or item.lower().startswith("oi_")
                for item in functional_dependencies
            )
        ),
        "long_short_ratio": (
            bool({"universe.augment_with_long_short_ratio",
                  "universe.filter_long_short_ratio"}.intersection(blocks))
            or any(
                "long_short" in item.lower() or "longaccount" in item.lower()
                or "shortaccount" in item.lower()
                for item in functional_dependencies
            )
        ),
        "liquidity": (
            "universe.filter_quote_volume" in blocks
            or bool({"quoteVolume", "quote_volume"}.intersection(functional_dependencies))
        ),
        "price_change": (
            bool({"universe.top_gainers", "universe.top_losers",
                  "universe.filter_change_pct"}.intersection(blocks))
            or bool({"priceChangePercent", "price_change_pct"}
                    .intersection(functional_dependencies))
        ),
    }
    for source in sorted(intent.excluded_sources):
        if source_is_used.get(source, False):
            errors.append(
                "使用者明確要求不要使用 %s,但 YAML 的 selection 仍宣告或讀取該資料來源"
                % source
            )

    selection_indicators = _selection_indicator_specs(steps)
    candidate_trade = selection.get("candidate_trade")
    candidate_plan = (
        candidate_trade_plan(intent) if intent.candidate_trade_requested else None
    )
    if intent.candidate_trade_requested and not isinstance(candidate_trade, dict):
        errors.append(
            "需求包含逐候選技術買賣點,但 YAML 沒有 selection.candidate_trade"
        )
    if not intent.candidate_trade_requested and candidate_trade is not None:
        errors.append(
            "使用者只要求選幣/方向標註,但模型擅自加入 selection.candidate_trade 交易計畫"
        )
    if candidate_plan is not None:
        declared = {
            str(item.get("cid")): item
            for item in ((spec.get("strategy") or {}).get("assumptions") or [])
            if isinstance(item, dict) and item.get("cid")
        }
        required = {
            str(item["cid"]): item for item in candidate_plan["assumptions"]
        }
        for cid, expected in required.items():
            actual = declared.get(cid)
            if actual is None:
                errors.append(
                    "逐候選交易使用未由使用者指定的預設,但 strategy.assumptions "
                    "缺少 cid=%s" % cid
                )
            elif actual.get("reading") != expected["reading"]:
                errors.append(
                    "strategy.assumptions cid=%s 必須精確宣告 %r,但目前是 %r"
                    % (cid, expected["reading"], actual.get("reading"))
                )
        unexpected_defaults = sorted(
            set(declared).intersection(candidate_plan["default_cids"]) - set(required)
        )
        if unexpected_defaults:
            errors.append(
                "使用者已明確指定對應值,但 YAML 仍宣稱使用未採用的預設: %s"
                % ", ".join(unexpected_defaults)
            )

        actual_size = (spec.get("sizing") or {}).get("size")
        if not _close_enough(actual_size, candidate_plan["size"]):
            errors.append(
                "逐候選交易 sizing.size 必須是 %.4g,但 YAML 是 %r"
                % (candidate_plan["size"], actual_size)
            )
        actual_exit = (spec.get("risk") or {}).get("exit") or {}
        if actual_exit.get("type") != "pct_stop_tp":
            errors.append(
                "逐候選交易的百分比停損/停利必須使用 risk.exit.type=pct_stop_tp，"
                "其他 exit type 不會套用 stop_pct/tp_pct"
            )
        for label, key in (("停損", "stop_pct"), ("停利", "tp_pct")):
            expected = candidate_plan[key]
            if not _close_enough(actual_exit.get(key), expected):
                errors.append(
                    "逐候選交易%s %s 必須是 %.4g,但 YAML 是 %r"
                    % (label, key, expected, actual_exit.get(key))
                )
    data = spec.get("data") or {}
    if intent.market_type and str(data.get("market_type") or "").lower() != intent.market_type:
        errors.append(
            "使用者指定 market_type=%s,但 YAML 是 %s"
            % (intent.market_type, data.get("market_type"))
        )
    if len(intent.intervals) == 1:
        actual_interval = str((data.get("primary") or {}).get("interval") or "").lower()
        if actual_interval != intent.intervals[0]:
            errors.append(
                "使用者指定週期 %s,但 YAML 的 primary.interval=%s"
                % (intent.intervals[0], actual_interval or None)
            )

    sentiment_filters = [
        step for step in steps if step.get("block") == "universe.filter_sentiment"
    ]
    meaningful_sentiment_filter = any(
        isinstance(step.get("params"), dict)
        and _greater_than((step.get("params") or {}).get("min_bull_ratio", 0.5), 0.5)
        for step in sentiment_filters
    )
    bullish_direction = any(
        leaf.get("cond") == "conditions.value_above"
        and isinstance(leaf.get("args"), (list, tuple))
        and leaf["args"]
        and "news_bull_ratio" in _feature_dependencies(selection, leaf["args"][0])
        and _greater_than(_leaf_threshold(leaf), 0.5)
        for leaf in _condition_leaves(selection.get("long_when"))
        if _leaf_threshold(leaf) is not None
    )

    unsupported_sources = sorted(intent.sources & UNSUPPORTED_SELECTION_SOURCES)
    if unsupported_sources:
        errors.append(
            "目前 selection runtime 尚未把 %s 的跨幣別 frame 接到 UniverseBundle;"
            "拒絕改用新聞或技術分析代替"
            % ", ".join(unsupported_sources)
        )

    # Only enforced when the phrase and the ranking are about the SAME column.
    # example_from_user_chat.yaml says "24h 跌幅前 30 名,依成交額排名": the
    # direction phrase describes a narrowing step (``universe.top_losers``) while
    # ``score`` is ``quoteVolume``. Demanding ``order: asc`` there would reject a
    # spec that is doing exactly what was asked.
    if intent.score_order and (
        _metric_columns(intent.score_order_metric) & score_dependencies
    ):
        actual_order = str(selection.get("order", "desc")).lower()
        if actual_order != intent.score_order:
            errors.append(
                "需求要求以 %s 由%s排名(selection.order: %s),但 YAML 是 %s;"
                "拒絕交出排名方向相反的籃子"
                % (intent.score_order_metric,
                   "低到高" if intent.score_order == "asc" else "高到低",
                   intent.score_order, actual_order)
            )

    if "price_change" in intent.sources:
        # The column needs no augment step — it arrives with the Binance 24h
        # ticker that IS the universe frame — so the only thing to check is that
        # it actually drives ranking, direction or narrowing, rather than being
        # quietly swapped for news buzz.
        change_blocks = {"universe.top_gainers", "universe.top_losers",
                         "universe.filter_change_pct"}
        if ("priceChangePercent" not in functional_dependencies
                and not change_blocks.intersection(blocks)):
            errors.append(
                "需求以 24h 漲跌幅選幣,但 selection 沒有實際讀取 priceChangePercent"
                "(排名、方向條件或 universe.top_gainers / top_losers 任一)"
            )

    if "funding" in intent.sources:
        augment = [step for step in steps
                   if step.get("block") == "universe.augment_with_funding"]
        if not augment or not any(
            isinstance(step.get("with"), (list, tuple))
            and "funding" in step.get("with")
            for step in augment
        ):
            errors.append(
                "需求提到 funding rate,selection 必須使用 "
                "universe.augment_with_funding 並傳入 funding"
            )
        # Either funding column counts: the per-settlement rate and the annualised
        # one are the same fact in two units, and the annualised one is the only
        # one comparable across contracts that settle 8h / 4h / 1h apart.
        funding_columns = _metric_columns("fundingRatePct")
        funding_used = (
            bool(funding_columns & functional_dependencies)
            or "universe.filter_funding_rate" in blocks
        )
        if not funding_used:
            errors.append(
                "需求提到 funding rate,但 selection 的排名或過濾沒有實際讀取 %s"
                % " / ".join(sorted(funding_columns))
            )

    if "open_interest" in intent.sources:
        oi_augments = {
            "universe.augment_with_open_interest": "open_interest_snapshot",
            "universe.augment_with_oi_change": "oi_change_snapshot",
        }
        wired_augments = [
            step for step in steps
            if step.get("block") in oi_augments
            and isinstance(step.get("with"), (list, tuple))
            and oi_augments[step.get("block")] in step.get("with")
        ]
        if not wired_augments:
            errors.append(
                "需求提到 open interest,selection 必須使用 "
                "universe.augment_with_open_interest(with: [open_interest_snapshot])"
                " 或 universe.augment_with_oi_change(with: [oi_change_snapshot])"
            )
        oi_columns = {
            "oi_base", "oi_notional_usd", "oi_change_pct",
            "oi_base_change_pct", "oi_change_notional_pct",
        }
        oi_filters = {"universe.filter_open_interest", "universe.filter_oi_change"}
        if not (oi_columns & functional_dependencies or oi_filters & blocks):
            errors.append(
                "需求提到 open interest,但 selection 的排名或過濾沒有實際讀取 OI 欄位"
            )

    if "long_short_ratio" in intent.sources:
        lsr_augments = [
            step for step in steps
            if step.get("block") == "universe.augment_with_long_short_ratio"
            and isinstance(step.get("with"), (list, tuple))
            and "long_short_ratio_snapshot" in step.get("with")
        ]
        if not lsr_augments:
            errors.append(
                "需求提到散戶多空比,selection 必須使用 "
                "universe.augment_with_long_short_ratio(with: "
                "[long_short_ratio_snapshot])"
            )
        minimum = intent.long_short_min_account_pct
        filters = [
            step for step in steps
            if step.get("block") == "universe.filter_long_short_ratio"
        ]
        comparator = intent.long_short_min_account_operator
        if minimum is None or comparator not in {">", ">="}:
            errors.append(
                "需求提到散戶多空比但未明確指定可驗證的多方帳戶百分比門檻;"
                "不得猜測 min_long_account_pct"
            )
        else:
            parameter = ("min_long_account_pct_exclusive"
                         if comparator == ">" else "min_long_account_pct")
            wording = "大於" if comparator == ">" else "至少"
            matched = any(
                isinstance(step.get("params"), dict)
                and _close_enough(
                    (step.get("params") or {}).get(parameter), minimum
                )
                for step in filters
            )
            if not matched:
                errors.append(
                    "需求要求散戶多方帳戶佔比%s %.4g%%,但 selection 沒有 "
                    "filter_long_short_ratio(%s: %.4g)"
                    % (wording, minimum, parameter, minimum)
                )

    if "news" in intent.sources:
        augment = [step for step in steps
                   if step.get("block") == "universe.augment_with_news"]
        if not augment or not any(
            isinstance(step.get("with"), (list, tuple))
            and "ticker_rank" in step.get("with")
            for step in augment
        ):
            errors.append(
                "需求提到新聞/社群/熱度,selection 必須使用 "
                "universe.augment_with_news 並傳入 ticker_rank"
            )
        news_blocks = {"universe.filter_sentiment", "universe.top_mentioned",
                       "universe.top_bullish"}
        if (not any(item.startswith("news_") for item in functional_dependencies)
                and not news_blocks.intersection(blocks)):
            errors.append(
                "需求提到新聞/熱度,但 selection 的排名與條件沒有實際讀取 news_* 欄位"
            )
        if "mentions" in intent.news_metrics:
            mentions_used = "news_mention_count" in score_dependencies
            if not mentions_used:
                errors.append(
                    "需求指定新聞提及量/熱度排名,但 selection.score 沒有實際依賴"
                    " news_mention_count"
                )
        if "sentiment" in intent.news_metrics:
            sentiment_used = "news_bull_ratio" in score_dependencies
            if not sentiment_used:
                errors.append(
                    "需求指定新聞/社群情緒排名,但 selection 沒有實際使用 news_bull_ratio"
                )

    if intent.bullish_preference:
        bullish_used = (
            meaningful_sentiment_filter
            or "news_bull_ratio" in score_dependencies
            or bullish_direction
        )
        if not bullish_used:
            errors.append(
                "需求偏好可能上漲/偏多候選,但 YAML 沒有使用 "
                "min_bull_ratio > 0.5 或 news_bull_ratio 排名作可驗證代理"
            )

    if "liquidity" in intent.sources:
        liquidity_blocks = {"universe.filter_quote_volume"}
        score_uses_volume = bool(
            {"quote_volume", "quoteVolume"}.intersection(score_dependencies)
        )
        # "Liquid coins ranked by <something else>" is a filter, not a ranking.
        # This used to accept only news as that something else, which turned the
        # sanest possible price-change screen — filter_quote_volume then rank on
        # priceChangePercent, because the top of the 24h gainers board is thin
        # coins nobody can fill — into a rejection for not ranking on volume.
        combined_filter = (
            intent.ranking_metric != "liquidity"
            and bool(intent.sources - {"liquidity"})
            and bool(liquidity_blocks.intersection(blocks))
        )
        if not score_uses_volume and not combined_filter:
            errors.append(
                "需求以成交量/流動性作選幣依據,但 selection.score 沒有實際依賴"
                " quoteVolume"
            )

    if "supertrend" in intent.indicator_names:
        if not intent.supertrend_parameters:
            errors.append(
                "需求指定 Supertrend,但沒有可驗證的 (period, multiplier) 參數"
            )
        if not intent.intervals:
            errors.append("需求指定 Supertrend,但沒有可驗證的 timeframe")
        requested_sides = [side for side in ("long", "short") if side in intent.directions]
        if not requested_sides:
            errors.append("需求指定 Supertrend,但沒有明確偏多或偏空方向")
        for period, multiplier in intent.supertrend_parameters:
            aliases = set()
            for timeframe in intent.intervals:
                found = _selection_supertrend_aliases(
                    steps, period=period, multiplier=multiplier, timeframe=timeframe,
                )
                if not found:
                    errors.append(
                        "需求指定 Supertrend(%d,%.4g) %s,但 selection 沒有 "
                        "universe.augment_with_indicator(with: [universe_bars], "
                        "output: 1) 的同參數 step"
                        % (period, multiplier, timeframe)
                    )
                aliases.update(found)

            for side in requested_sides:
                node = selection.get("%s_when" % side)
                relation = "value_above" if side == "long" else "value_below"
                mandatory_aliases = _mandatory_supertrend_aliases(
                    node, aliases=aliases, relation=relation,
                )
                missing_aliases = (sorted(aliases - mandatory_aliases)
                                   if mandatory_aliases is not None
                                   else sorted(aliases))
                if missing_aliases:
                    errors.append(
                        "需求指定 Supertrend(%d,%.4g) 偏%s,但 %s_when 沒有以 "
                        "conditions.%s(_, 0) 在每條可行判斷路徑實際要求: %s"
                        % (period, multiplier, "多" if side == "long" else "空", side,
                           relation, ", ".join(missing_aliases))
                    )
                if len(intent.intervals) > 1 and aliases and (
                        mandatory_aliases is None or not aliases.issubset(mandatory_aliases)):
                    errors.append(
                        "需求指定多時框 Supertrend 同時偏%s,但 %s_when 存在可繞過某個 "
                        "timeframe 的布林分支；每條可行路徑都必須要求所有 timeframe"
                        % ("多" if side == "long" else "空", side)
                    )

    if intent.candidate_trade_requested:
        if "long" in intent.directions and not selection.get("long_when"):
            errors.append("需求明確要求做多買點,但 YAML 沒有 selection.long_when")
        if "short" in intent.directions and not selection.get("short_when"):
            errors.append("需求明確要求做空賣點,但 YAML 沒有 selection.short_when")
        if "bidirectional_trade_points" in intent.evidence:
            if not selection.get("long_when"):
                errors.append("需求要求買賣點,但 YAML 沒有 selection.long_when 買點")
            if not selection.get("short_when"):
                errors.append("需求要求買賣點,但 YAML 沒有 selection.short_when 賣點")
    elif not intent.directions and not intent.bullish_preference:
        if selection.get("long_when") or selection.get("short_when"):
            errors.append("使用者只要求排名,但模型擅自加入 long_when/short_when 方向條件")
    else:
        if "long" in intent.directions and not selection.get("long_when"):
            errors.append("需求明確要求做多候選,但 YAML 沒有 selection.long_when")
        if "short" in intent.directions and not selection.get("short_when"):
            errors.append("需求明確要求做空候選,但 YAML 沒有 selection.short_when")
    if "long" in intent.excluded_directions and selection.get("long_when"):
        errors.append("使用者明確要求不要做多,但 YAML 仍包含 selection.long_when")
    if "short" in intent.excluded_directions and selection.get("short_when"):
        errors.append("使用者明確要求不要做空,但 YAML 仍包含 selection.short_when")

    if intent.candidate_trade_requested:
        period_names = {name for name, _period in intent.technical_periods}
        requested_aliases = set()
        for name, period in intent.technical_periods:
            aliases = {
                alias for actual_name, actual_period, alias in selection_indicators
                if actual_name == name and _close_enough(actual_period, period)
            }
            if not aliases:
                errors.append(
                    "需求指定 %s%d,但 selection 沒有用 "
                    "universe.augment_with_indicator 計算該指標"
                    % (name.upper(), period)
                )
            elif not aliases.intersection(direction_dependencies):
                errors.append(
                    "selection 雖計算 %s%d,但其輸出沒有實際影響 long_when/short_when"
                    % (name.upper(), period)
                )
            requested_aliases.update(aliases)
        for name in set(intent.indicator_names) - period_names:
            aliases = {
                alias for actual_name, _actual_period, alias in selection_indicators
                if actual_name == name
            }
            if not aliases:
                errors.append(
                    "需求指定 %s,但 selection 沒有用 "
                    "universe.augment_with_indicator 計算該指標"
                    % name.upper()
                )
            elif not aliases.intersection(direction_dependencies):
                errors.append(
                    "selection 雖計算 %s,但其輸出沒有實際影響 long_when/short_when"
                    % name.upper()
                )
            requested_aliases.update(aliases)

        # A generic "technical buy/sell point" request intentionally defaults
        # to RSI14.  There is no user-named indicator for the loops above to
        # discover, so validate the resolved plan itself as well: the model must
        # actually compute RSI at the declared/defaulted period and wire that
        # alias into the direction conditions.
        plan_aliases = {
            alias for actual_name, actual_period, alias in selection_indicators
            if actual_name == "rsi"
            and _close_enough(actual_period, candidate_plan["period"])
        }
        if not plan_aliases:
            errors.append(
                "逐候選交易計畫要求 RSI%d,但 selection 沒有用 "
                "universe.augment_with_indicator 計算該指標"
                % candidate_plan["period"]
            )
        elif not plan_aliases.intersection(direction_dependencies):
            errors.append(
                "selection 雖計算逐候選交易計畫的 RSI%d,但其輸出沒有實際影響 "
                "long_when/short_when" % candidate_plan["period"]
            )
        requested_aliases.update(plan_aliases)

        # The resolved plan includes both user-stated values and any declared
        # defaults.  Check each side separately: finding ``above 70`` somewhere
        # in the tree is not enough when the model attached it to the buy side.
        for side, (relation, value) in candidate_plan["thresholds"].items():
            side_leaves = list(_condition_leaves(selection.get("%s_when" % side)))
            suffixes = (
                ("value_below", "rsi_oversold")
                if relation == "below"
                else ("value_above", "rsi_overbought")
            )
            wired = any(
                str(leaf.get("cond") or "").endswith(suffixes)
                and bool(requested_aliases.intersection({
                    str(item) for item in (leaf.get("args") or [])
                }))
                and _close_enough(_leaf_threshold(leaf), value)
                for leaf in side_leaves
            )
            if not wired:
                errors.append(
                    "逐候選 %s_when 必須使用 RSI %s %.4g,且門檻與宣告/需求一致"
                    % (side, relation, value)
                )

        # A threshold is part of the request, not a suggestion.  Merely wiring
        # the RSI alias into a different comparison would still produce a healthy
        # basket answering a different strategy.
        direction_leaves = list(_condition_leaves(selection.get("long_when")))
        direction_leaves += list(_condition_leaves(selection.get("short_when")))
        for relation, value in intent.rsi_thresholds:
            suffixes = (("value_below", "rsi_oversold") if relation == "below"
                        else ("value_above", "rsi_overbought"))
            wired = any(
                str(leaf.get("cond") or "").endswith(suffixes)
                and bool(requested_aliases.intersection({
                    str(item) for item in (leaf.get("args") or [])
                }))
                and _close_enough(_leaf_threshold(leaf), value)
                for leaf in direction_leaves
            )
            if not wired:
                errors.append(
                    "需求指定 RSI %s %.4g,但 selection.long_when/short_when "
                    "沒有以該門檻建立候選交易條件" % (relation, value)
                )

    excluded_indicator_period_names = {
        name for name, _period in intent.excluded_technical_periods
    }
    for name, period in intent.excluded_technical_periods:
        if any(actual_name == name and _close_enough(actual_period, period)
               for actual_name, actual_period, _alias in selection_indicators):
            errors.append(
                "使用者明確要求不要使用 %s%d,但 selection 仍計算該指標"
                % (name.upper(), period)
            )
    for name in set(intent.excluded_indicator_names) - excluded_indicator_period_names:
        if any(actual_name == name for actual_name, _period, _alias in selection_indicators):
            errors.append(
                "使用者明確要求不要使用 %s,但 selection 仍計算該指標"
                % name.upper()
            )

    if intent.requested_count is not None:
        try:
            actual_count = int(selection.get("top_k"))
        except (TypeError, ValueError):
            actual_count = None
        if actual_count != intent.requested_count:
            errors.append(
                "使用者要求 %d 個候選,但 selection.top_k=%r"
                % (intent.requested_count, selection.get("top_k"))
            )

    def step_symbols(block_name):
        return {
            str(item).upper()
            for step in steps if step.get("block") == block_name
            for item in (((step.get("params") or {}).get("symbols") or []))
        }

    only_symbols = step_symbols("universe.only_symbols")
    excluded_symbols = step_symbols("universe.exclude_symbols")
    requested_included_bases = {_base_asset(item) for item in intent.included_symbols}
    requested_excluded_bases = {_base_asset(item) for item in intent.excluded_symbols}
    actual_only_bases = {_base_asset(item) for item in only_symbols}
    actual_excluded_bases = {_base_asset(item) for item in excluded_symbols}

    missing_included = sorted(requested_included_bases - actual_only_bases)
    if missing_included:
        errors.append(
            "使用者把選幣宇宙限定為 %s,但 YAML 的 universe.only_symbols 未完整保留: %s"
            % (", ".join(intent.included_symbols), ", ".join(missing_included))
        )
    unexpected_included = sorted(
        symbol for symbol in only_symbols
        if _base_asset(symbol) not in requested_included_bases
    )
    if unexpected_included:
        errors.append(
            "模型擅自把選幣宇宙限制為使用者未指定的標的: %s"
            % ", ".join(unexpected_included)
        )

    missing_excluded = sorted(requested_excluded_bases - actual_excluded_bases)
    if missing_excluded:
        errors.append(
            "使用者明確排除 %s,但 YAML 的 universe.exclude_symbols 未完整排除: %s"
            % (", ".join(intent.excluded_symbols), ", ".join(missing_excluded))
        )
    unmapped_categories = [
        item.split(":", 1)[1]
        for item in intent.unsupported_preferences
        if item.startswith("category_exclusion:")
    ]
    if unmapped_categories:
        errors.append(
            "使用者要求排除資產類別(%s),但沒有經核對的 metadata 映射;"
            "拒絕以近似的 universe filter 假裝已排除"
            % ", ".join(unmapped_categories)
        )
    if "under_discovered" in intent.unsupported_preferences:
        warnings.append(
            "「少見/尚未被市場發現」目前沒有直接資料欄位;本策略只能使用"
            "流動性、Square 提及量與情緒作代理,不能宣稱已證明尚未被發現"
        )
    return errors, warnings
