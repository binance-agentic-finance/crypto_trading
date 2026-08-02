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
    "classify_request",
    "generated_strategy_kind",
    "infer_strategy_kind",
    "reconcile_intent",
]

#: Cross-sectional sources a selection spec cannot express yet, so a request
#: that depends on one is refused instead of answered with something else.
#:
#: ``open_interest``: Binance publishes no whole-market OI endpoint —
#: ``GET /fapi/v1/openInterest`` without a symbol answers HTTP 400 — so there is
#: no cross-section frame to rank on, only a per-symbol fan-out that nobody has
#: built. ``price_change`` used to be listed here and was simply wrong:
#: ``priceChangePercent`` rides along in the universe RANK frame already, and
#: ``universe.top_gainers`` / ``top_losers`` have always consumed it.
UNSUPPORTED_SELECTION_SOURCES = frozenset({"open_interest"})


class IntentDecision:
    """The independently-checkable meaning extracted before YAML generation."""

    __slots__ = (
        "kind", "evidence", "requested_count", "sources",
        "bullish_preference", "unsupported_preferences", "named_symbols",
        "intervals", "market_type", "technical_periods", "stop_pct",
        "tp_pct", "size_fraction", "directions", "news_metrics", "triggers",
        "indicator_names", "rsi_thresholds", "ranking_metric",
        "score_order", "score_order_metric",
    )

    def __init__(
        self,
        *,
        kind: str,
        evidence=(),
        requested_count=None,
        sources=frozenset(),
        bullish_preference=False,
        unsupported_preferences=(),
        named_symbols=(),
        intervals=(),
        market_type=None,
        technical_periods=(),
        stop_pct=None,
        tp_pct=None,
        size_fraction=None,
        directions=(),
        news_metrics=(),
        triggers=(),
        indicator_names=(),
        rsi_thresholds=(),
        ranking_metric=None,
        score_order=None,
        score_order_metric=None,
    ):
        self.kind = str(kind)
        self.evidence = tuple(evidence)
        self.requested_count = requested_count
        self.sources = frozenset(sources)
        self.bullish_preference = bool(bullish_preference)
        self.unsupported_preferences = tuple(unsupported_preferences)
        self.named_symbols = tuple(named_symbols)
        self.intervals = tuple(intervals)
        self.market_type = market_type
        self.technical_periods = tuple(technical_periods)
        self.stop_pct = stop_pct
        self.tp_pct = tp_pct
        self.size_fraction = size_fraction
        self.directions = tuple(directions)
        self.news_metrics = tuple(news_metrics)
        self.triggers = tuple(triggers)
        self.indicator_names = tuple(indicator_names)
        self.rsi_thresholds = tuple(rsi_thresholds)
        self.ranking_metric = ranking_metric
        # Which end of the ranked column the request named ("asc"/"desc"), and
        # the frame column it named it for. Both None means the user did not say,
        # and nothing downstream may pretend otherwise.
        self.score_order = score_order
        self.score_order_metric = score_order_metric

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "evidence": list(self.evidence),
            "requested_count": self.requested_count,
            "sources": sorted(self.sources),
            "bullish_preference": self.bullish_preference,
            "unsupported_preferences": list(self.unsupported_preferences),
            "named_symbols": list(self.named_symbols),
            "intervals": list(self.intervals),
            "market_type": self.market_type,
            "technical_periods": [
                {"indicator": name, "period": period}
                for name, period in self.technical_periods
            ],
            "stop_pct": self.stop_pct,
            "tp_pct": self.tp_pct,
            "size_fraction": self.size_fraction,
            "directions": list(self.directions),
            "news_metrics": list(self.news_metrics),
            "triggers": list(self.triggers),
            "indicator_names": list(self.indicator_names),
            "rsi_thresholds": [
                {"relation": relation, "value": value}
                for relation, value in self.rsi_thresholds
            ],
            "ranking_metric": self.ranking_metric,
            "score_order": self.score_order,
            "score_order_metric": self.score_order_metric,
        }


def _rules(*items):
    return tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in items)


_SELECTION_RULES = _rules(
    ("zh_explicit_selection", r"選幣"),
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
     r"(?:買進|賣出|買入|賣掉|做多|做空|進場|出場|(?<!未)平倉|停損|停利|止損|止盈)"),
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
_SOURCE_RULES = _rules(
    ("news",
     r"(?:新聞|社群|熱度|熱門|提到|提及|Square|news|social|mention|mentioned|buzz|hot|trending|popular)"),
    ("funding", r"(?:資金費率|funding(?:\s+rate)?)"),
    ("open_interest", r"(?:未平倉|未平倉量|持倉量|open[\s_-]*interest|\bOI\b)"),
    ("liquidity", r"(?:流動性|成交量|交易量|quote[\s_-]*volume|\bvolume\b|liquidity)"),
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
    r"(?:by\s+(?:quote[\s_-]*)?volume|(?:依|按|根據).{0,16}(?:成交量|交易量|流動性)|"
    r"(?:成交量|交易量|流動性).{0,12}(?:最大|最高|排行|排名|top))",
    re.IGNORECASE,
)
_UNSUPPORTED_DISCOVERY = re.compile(
    r"(?:少見|冷門|未被.{0,8}發現|沒人.{0,8}發現|尚未.{0,8}發現|"
    r"undiscovered|under[- ]the[- ]radar|havent\s+(?:find|found)|haven't\s+(?:find|found))",
    re.IGNORECASE,
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
    "LLM", "LONG", "MACD", "OHLCV", "PICK", "RANK", "RSI", "SELECT", "SELL",
    "SHORT", "SMA", "TOP", "TRADE", "USD", "USDC", "USDT", "VWAP", "YAML",
}
_EXECUTION_ACTION = re.compile(
    # (?<!未)平倉 for the same reason as _TRADE_RULES: 未平倉量 is open interest.
    r"(?:買進|買入|賣出|賣掉|下單|進場|出場|(?<!未)平倉|自動交易|直接買|"
    r"\b(?:buy|sell|execute|enter|exit|place\s+orders?|trade\s+them)\b)",
    re.IGNORECASE,
)
_INTERVAL = re.compile(
    r"(?<![A-Za-z0-9])(\d+)\s*"
    r"(minutes?|mins?|m|分鐘|分|hours?|hrs?|h|小時|小时|days?|d|天|日|weeks?|w|週|周)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_TECHNICAL_PERIOD = re.compile(
    r"(?<![A-Za-z])(EMA|SMA|RSI|ADX)\s*[-_]?\s*(\d{1,4})(?!\d)",
    re.IGNORECASE,
)
_TECHNICAL_NAME = re.compile(r"(?<![A-Za-z])(EMA|SMA|RSI|ADX|MACD)(?![A-Za-z])", re.IGNORECASE)
_RSI_THRESHOLD = re.compile(
    r"RSI(?:\s*[-_]?\s*\d{1,4})?.{0,24}?"
    r"(低於|小於|低于|below|under|高於|大於|高于|above|over)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_LONG_DIRECTION = re.compile(r"(?:買進|買入|做多|多方|\b(?:buy|long)\b)", re.IGNORECASE)
_SHORT_DIRECTION = re.compile(r"(?:做空|空方|\bshort\b)", re.IGNORECASE)
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


def _named_symbols(text: str) -> tuple[str, ...]:
    found = [match.group(0).upper() for match in _FULL_SYMBOL.finditer(text)]
    found.extend(match.group(0).upper() for match in _KNOWN_BASE_SYMBOL.finditer(text))
    found.extend(
        match.group(0)
        for match in _BARE_UPPER_SYMBOL.finditer(text)
        if match.group(0) not in _SYMBOL_STOPWORDS
        and not re.fullmatch(r"(?:EMA|SMA|RSI|ADX|MACD)\d*", match.group(0))
    )
    return tuple(dict.fromkeys(found))


def _requested_intervals(text: str) -> tuple[str, ...]:
    unit_map = {
        "m": "m", "min": "m", "mins": "m", "minute": "m", "minutes": "m",
        "分鐘": "m", "分": "m",
        "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
        "小時": "h", "小时": "h",
        "d": "d", "day": "d", "days": "d", "天": "d", "日": "d",
        "w": "w", "week": "w", "weeks": "w", "週": "w", "周": "w",
    }
    values = []
    for match in _INTERVAL.finditer(text):
        unit = unit_map[match.group(2).lower()]
        values.append("%d%s" % (int(match.group(1)), unit))
    return tuple(dict.fromkeys(values))


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


def _requested_market_type(text: str) -> str | None:
    if re.search(r"(?:現貨|\bspot\b)", text, re.IGNORECASE):
        return "spot"
    if re.search(r"(?:永續|合約|期貨|\b(?:futures?|perpetuals?)\b)", text, re.IGNORECASE):
        return "futures"
    return None


def classify_request(nl: str) -> IntentDecision:
    """Classify scope and fail closed when the request is genuinely ambiguous.

    This is intentionally independent of the YAML returned by the model. A
    generated trade spec cannot validate its own claim that a basket request was
    actually a trade request.
    """
    text = " ".join(str(nl or "").split())
    selection = [name for name, rule in _SELECTION_RULES if rule.search(text)]
    trade = [name for name, rule in _TRADE_RULES if rule.search(text)]
    plural_scope = bool(_PLURAL_SCOPE.search(text))

    if selection and trade and _EXECUTION_ACTION.search(text):
        # There is no one-spec grammar for "rank a universe, then run this
        # single-symbol entry rule on every winner". Picking either half would
        # silently discard user intent, so stop before an LLM can improvise.
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

    unsupported = ()
    if _UNSUPPORTED_DISCOVERY.search(text):
        unsupported = ("under_discovered",)
    sources = {name for name, rule in _SOURCE_RULES if rule.search(text)}
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
    technical_periods = tuple(
        (match.group(1).lower(), int(match.group(2)))
        for match in _TECHNICAL_PERIOD.finditer(text)
    )
    directions = []
    if _LONG_DIRECTION.search(text):
        directions.append("long")
    if _SHORT_DIRECTION.search(text):
        directions.append("short")
    news_metrics = []
    if "news" in sources and _NEWS_MENTION_METRIC.search(text) \
            and ranking_metric != "liquidity":
        news_metrics.append("mentions")
    if "news" in sources and _NEWS_SENTIMENT_METRIC.search(text):
        news_metrics.append("sentiment")
    rsi_thresholds = []
    for match in _RSI_THRESHOLD.finditer(text):
        relation = match.group(1).lower()
        relation = "below" if relation in {"低於", "小於", "低于", "below", "under"} else "above"
        rsi_thresholds.append((relation, float(match.group(2))))
    # One unambiguous direction phrase, or none at all. "最低與最高 funding" and
    # "漲幅或跌幅" match two rules that contradict each other; guessing one would
    # invent a preference the user never stated, and the downstream check would
    # then reject a spec for disagreeing with a coin flip.
    ordered = {(column, order) for column, order, rule in _SCORE_ORDER_RULES
               if rule.search(text)}
    if _DIRECTION_EXCLUSION.search(text) or _DIMINISHED_SIGN.search(text):
        ordered = set()
    score_order_metric, score_order = ordered.pop() if len(ordered) == 1 else (None, None)
    return IntentDecision(
        kind=kind,
        evidence=evidence,
        requested_count=_requested_count(text),
        sources=sources,
        bullish_preference=bool(_BULLISH_PREFERENCE.search(text)),
        unsupported_preferences=unsupported,
        named_symbols=_named_symbols(text),
        intervals=_requested_intervals(text),
        market_type=_requested_market_type(text),
        technical_periods=technical_periods,
        stop_pct=_percent_near(text, r"停損|止損|stop[- ]?loss"),
        tp_pct=_percent_near(text, r"停利|止盈|take[- ]?profit"),
        size_fraction=_percent_near(text, r"倉位|資金|投入|size|position[\s_-]*size"),
        directions=directions,
        news_metrics=news_metrics,
        triggers=[name for name, rule in _TRIGGER_RULES if rule.search(text)],
        indicator_names=tuple(dict.fromkeys(
            match.group(1).lower() for match in _TECHNICAL_NAME.finditer(text)
        )),
        rsi_thresholds=rsi_thresholds,
        ranking_metric=ranking_metric,
        score_order=score_order,
        score_order_metric=score_order_metric,
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

    if intent.named_symbols:
        requested_bases = {_base_asset(item) for item in intent.named_symbols}
        if len(requested_bases) > 1:
            errors.append(
                "trade YAML 一次只能執行一個標的,但需求指定了多個標的: %s"
                % ", ".join(intent.named_symbols)
            )
        elif _base_asset(actual_symbol) not in requested_bases:
            errors.append(
                "使用者指定標的是 %s,但 YAML 的 data.symbol=%s"
                % (", ".join(intent.named_symbols), actual_symbol or None)
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

    entry = signals.get("entry") or {}
    if "long" in intent.directions and not entry.get("long"):
        errors.append("需求包含做多/買進,但 YAML 沒有 signals.entry.long")
    if "short" in intent.directions and not entry.get("short"):
        errors.append("需求包含做空,但 YAML 沒有 signals.entry.short")
    if intent.directions == ("long",) and entry.get("short"):
        errors.append("使用者只要求做多/買進,但模型擅自加入 signals.entry.short")
    if intent.directions == ("short",) and entry.get("long"):
        errors.append("使用者只要求做空,但模型擅自加入 signals.entry.long")

    requested_nodes = [entry.get(side) for side in intent.directions if entry.get(side)]
    if not requested_nodes:
        requested_nodes = [value for value in entry.values() if isinstance(value, dict)]
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
        if not accepted.intersection(condition_refs):
            errors.append(
                "需求指定 %s,但 YAML entry 沒有使用對應條件 Block (%s)"
                % (trigger, ", ".join(sorted(accepted)))
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
        wired = any(
            leaf.get("cond") in acceptable
            and isinstance(leaf.get("args"), (list, tuple))
            and leaf["args"]
            and str(leaf["args"][0]) in rsi_aliases
            and _close_enough(_leaf_threshold(leaf), value)
            for leaf in leaves
        )
        if not wired:
            errors.append(
                "需求指定 RSI %s %.4g,但 entry 沒有以該 RSI 與門檻建立條件"
                % (relation, value)
            )

    exit_cfg = (spec.get("risk") or {}).get("exit") or {}
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
        return errors, warnings

    if intent.kind != "selection":
        return errors, warnings

    selection = spec["selection"]
    steps, blocks, score_dependencies, direction_dependencies = _selection_usage(selection)
    functional_dependencies = score_dependencies | direction_dependencies
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

    if not intent.directions and not intent.bullish_preference:
        if selection.get("long_when") or selection.get("short_when"):
            errors.append("使用者只要求排名,但模型擅自加入 long_when/short_when 方向條件")
    elif "long" in intent.directions and not selection.get("long_when"):
        errors.append("需求明確要求做多候選,但 YAML 沒有 selection.long_when")
    elif "short" in intent.directions and not selection.get("short_when"):
        errors.append("需求明確要求做空候選,但 YAML 沒有 selection.short_when")

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

    requested_symbols = {_base_asset(item) for item in intent.named_symbols}
    for step in steps:
        if step.get("block") != "universe.only_symbols":
            continue
        symbols = {str(item).upper() for item in
                   ((step.get("params") or {}).get("symbols") or [])}
        unexpected = sorted(
            symbol for symbol in symbols if _base_asset(symbol) not in requested_symbols
        )
        if unexpected:
            errors.append(
                "模型擅自把選幣宇宙限制為使用者未指定的標的: %s"
                % ", ".join(unexpected)
            )

    if "under_discovered" in intent.unsupported_preferences:
        warnings.append(
            "「少見/尚未被市場發現」目前沒有直接資料欄位;本策略只能使用"
            "流動性、Square 提及量與情緒作代理,不能宣稱已證明尚未被發現"
        )
    return errors, warnings
