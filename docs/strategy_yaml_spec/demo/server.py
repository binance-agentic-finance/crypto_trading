"""Demo server: 自然語言 → (LLM/LiteLLM) → YAML → 標準訊號 / 回測.

這是一個把自然語言轉成交易或選幣 YAML，再交給確定性 runtime 執行的展示,不是 agent。
瀏覽器 → 本後端 → LiteLLM(OpenAI 相容 /chat/completions)。API key 只在這次請求中轉送。

啟動:
    PYTHONPATH=<repo_root> <venv>/bin/python docs/strategy_yaml_spec/demo/server.py
    # 然後開 http://127.0.0.1:8799
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- 讓後端找得到 cyqnt_trd(repo root = 這個檔往上三層)---
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
SPEC_DIR = HERE.parents[1]                      # docs/strategy_yaml_spec
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import requests
import yaml

from cyqnt_trd.standard_bot.yaml_pipeline import build_make_signals, validate_spec
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import resolve_block
from cyqnt_trd.standard_bot.simulation.vectorized_backtest import run_vectorized_backtest

PORT = 8799
SCHEMA_PATH = SPEC_DIR / "strategy.schema.yaml"
FIXTURE_DIR = REPO_ROOT / "tests" / "blocks" / "fixtures"


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

def _prompt_signature(ref: str) -> str:
    """Render parameter names from the callable the YAML runtime will invoke.

    The prompt used to duplicate these names by hand and taught the model
    ``fast_period`` while the running MACD block accepts ``fast``. Deriving the
    hint from :func:`resolve_block` makes prompt drift a testable contract bug
    instead of a model hallucination.
    """
    import inspect

    parameters = inspect.signature(resolve_block(ref)).parameters.values()
    names = [item.name for item in parameters]
    return "%s(%s)" % (ref, ",".join(names))


def _build_blocks_cheatsheet() -> str:
    indicators = [
        "indicators.ema", "indicators.sma", "indicators.rsi",
        "indicators.atr", "indicators.adx", "indicators.macd",
    ]
    conditions = [
        "conditions.ma_cross_above", "conditions.ma_cross_below",
        "conditions.rsi_in_range", "conditions.rsi_overbought",
        "conditions.rsi_oversold", "conditions.adx_trending",
        "conditions.breakout_high", "conditions.breakout_low",
        "conditions.macd_golden_cross", "conditions.macd_death_cross",
    ]
    return (
        "可用 indicators(input=close 或自動 df;參數名取自實際 Blocks):\n  "
        + " ".join(_prompt_signature(ref) for ref in indicators)
        + "\n  indicators.adx 用 output:0 取 adx;indicators.macd 用 output:0/1/2"
          " 取 macd線/signal線/hist。\n"
        + "可用 conditions(回傳 bool):\n  "
        + " ".join(_prompt_signature(ref) for ref in conditions)
        + "\n組合器(可任意巢狀):{all_of:[...]} {any_of:[...]} {not:<node>} "
          "葉節點:{cond:\"conditions.xxx\",args:[...],params:{...}}\n"
        + "出場 risk.exit.type:pct_stop_tp{stop_pct,tp_pct,max_bars} / "
          "atr_stop_tp{atr_period,stop_mult,tp_mult,max_bars} / "
          "time_only{max_bars} / opposite_signal{max_bars}\n"
    )


BLOCKS_CHEATSHEET = _build_blocks_cheatsheet()

EXAMPLE_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: btc_ema_rsi_1h
  description: "EMA 交叉 + RSI 過濾"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h", poll_interval: 3570 }
  source: { type: binance_rest }
signals:
  indicators:
    ema_fast: { block: indicators.ema, input: close, params: { period: 12 } }
    ema_slow: { block: indicators.ema, input: close, params: { period: 26 } }
    rsi14:    { block: indicators.rsi, input: close, params: { period: 14 } }
  entry:
    long:
      all_of:
        - { cond: conditions.ma_cross_above, args: [ema_fast, ema_slow] }
        - not: { cond: conditions.rsi_overbought, args: [rsi14], params: { threshold: 75 } }
    short:
      all_of:
        - { cond: conditions.ma_cross_below, args: [ema_fast, ema_slow] }
        - not: { cond: conditions.rsi_oversold, args: [rsi14], params: { threshold: 25 } }
sizing: { size: 0.95 }
risk:
  exit: { type: pct_stop_tp, stop_pct: 0.02, tp_pct: 0.04, max_bars: 96 }
  fees: { commission_bps: 4.0, slippage_bps: 2.0 }
backtest: { initial_capital: 10000, execution_model: next_bar_open }
"""


SELECTION_EXAMPLE_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: square_news_buzz_selector
  description: "依 Binance Square 最近提及量挑選候選幣"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h" }
selection:
  universe:
    - block: universe.filter_quote_volume
      params: { min_quote_volume: 100000000 }
    - block: universe.augment_with_news
      with: [ticker_rank]
  score: news_mention_count
  top_k: 5
  min_score: 1.0
  dedupe_by: base_asset
"""

LIQUIDITY_SELECTION_EXAMPLE_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: quote_volume_selector
  description: "依 24 小時 USDT 成交額挑選候選幣"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h" }
selection:
  universe:
    - block: universe.filter_quote_volume
      params: { min_quote_volume: 100000000 }
  score: quote_volume
  top_k: 5
  min_score: 1.0
  dedupe_by: base_asset
"""

FUNDING_SELECTION_EXAMPLE_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: funding_rate_selector
  description: "依目前跨幣別資金費率挑選候選幣"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h" }
selection:
  universe:
    - block: universe.filter_quote_volume
      params: { min_quote_volume: 100000000 }
    - block: universe.augment_with_funding
      with: [funding]
  score: fundingRatePct
  top_k: 5
  dedupe_by: base_asset
"""


class IntentDecision:
    """The independently-checkable meaning extracted before YAML generation."""

    __slots__ = (
        "kind", "evidence", "requested_count", "sources",
        "bullish_preference", "unsupported_preferences", "named_symbols",
        "intervals", "market_type", "technical_periods", "stop_pct",
        "tp_pct", "size_fraction", "directions", "news_metrics", "triggers",
        "indicator_names", "rsi_thresholds", "ranking_metric",
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
    ("zh_trade_action",
     r"(?:買進|賣出|買入|賣掉|做多|做空|進場|出場|平倉|停損|停利|止損|止盈)"),
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
_LOWEST_FUNDING_RANKING = re.compile(
    r"(?:funding.{0,20}(?:最負|最低|most\s+negative|lowest)|"
    r"(?:最負|最低|most\s+negative|lowest).{0,20}funding)",
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
    r"(?:買進|買入|賣出|賣掉|下單|進場|出場|平倉|自動交易|直接買|"
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
    )


def infer_strategy_kind(nl: str) -> str:
    """Compatibility helper used by the HTTP prompt route and tests."""
    return classify_request(nl).kind


def _generated_strategy_kind(spec: dict) -> str | None:
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
    generated = _generated_strategy_kind(spec)

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

    unsupported_sources = sorted(intent.sources & {"open_interest", "price_change"})
    if unsupported_sources:
        errors.append(
            "目前 selection runtime 尚未把 %s 的跨幣別 frame 接到 UniverseBundle;"
            "拒絕改用新聞或技術分析代替"
            % ", ".join(unsupported_sources)
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
        funding_used = (
            "fundingRatePct" in functional_dependencies
            or "universe.filter_funding_rate" in blocks
        )
        if not funding_used:
            errors.append(
                "需求提到 funding rate,但 selection 的排名或過濾沒有實際讀取 "
                "fundingRatePct"
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
        combined_filter = (
            intent.ranking_metric != "liquidity"
            and "news" in intent.sources
            and bool(liquidity_blocks.intersection(blocks))
        )
        if not score_uses_volume and not combined_filter:
            errors.append(
                "需求以成交量/流動性作選幣依據,但 selection.score 沒有實際依賴"
                " quote_volume"
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


def build_system_prompt(
    kind: str = "trade", intent: IntentDecision | None = None,
) -> str:
    if kind not in {"trade", "selection"}:
        raise ValueError("strategy kind must be trade or selection, got %r" % kind)
    if kind == "selection":
        if intent is not None and "funding" in intent.sources:
            return (
                "你是一個把自然語言選幣需求轉成 StandardBot YAML 的轉換器。"
                "只輸出一份合法 YAML,不要 markdown、解釋或多餘文字。\n\n"
                "這次需求指定跨幣別 funding rate。頂層只用 selection:,不得產生"
                " signals:、sizing:、risk: 或 backtest:。使用 data.symbol=BTCUSDT"
                " 作排程代表標的,它不是候選結果。必須先用"
                " universe.filter_quote_volume 過濾最低流動性,再精確使用"
                " universe.augment_with_funding 並寫 with: [funding]。"
                "selection.score 必須是 fundingRatePct,使 funding 真正控制候選排名;"
                "不得改用新聞、EMA/RSI 或自行猜單一幣種。top_k 必須依使用者要求,"
                "未指定時為 5。這是當下截面選幣訊號,不是歷史回測。\n\n"
                "=== 可用的 funding 選幣 BLOCKS / 欄位 ===\n"
                "universe.filter_quote_volume\n"
                "universe.augment_with_funding (with: [funding])\n"
                "universe.filter_funding_rate\n"
                "fundingRatePct, quote_volume\n\n"
                "=== 範例輸出 ===\n" + FUNDING_SELECTION_EXAMPLE_YAML
            )
        if intent is not None and "liquidity" in intent.sources \
                and "news" not in intent.sources:
            return (
                "你是一個把自然語言選幣需求轉成 StandardBot YAML 的轉換器。"
                "只輸出一份合法 YAML,不要 markdown、解釋或多餘文字。\n\n"
                "這次需求指定成交量/流動性。頂層只用 selection:,不得產生 signals:。"
                "使用 data.symbol=BTCUSDT 作排程代表標的；它不是候選結果。"
                "使用 universe.filter_quote_volume 過濾最低流動性，並以"
                " score: quote_volume 排名。不得改用 news_mention_count、技術指標或"
                "自行猜單一幣種。top_k 必須依使用者要求。\n\n=== 範例輸出 ===\n"
                + LIQUIDITY_SELECTION_EXAMPLE_YAML
            )
        ranking_note = "依 Square 提及量排序時使用 score: news_mention_count。"
        selection_example = SELECTION_EXAMPLE_YAML
        if intent is not None and "sentiment" in intent.news_metrics \
                and "mentions" not in intent.news_metrics:
            ranking_note = (
                "這次使用者要求新聞/社群情緒排行,selection.score 必須使用"
                " news_bull_ratio；不得偷換成 news_mention_count。"
            )
            selection_example = SELECTION_EXAMPLE_YAML.replace(
                "score: news_mention_count", "score: news_bull_ratio"
            )
        return (
            "你是一個把自然語言選幣需求轉成 StandardBot YAML 的轉換器。"
            "只輸出一份合法 YAML,不要有 markdown 圍欄(```)、不要解釋、不要多餘文字。\n\n"
            "這次輸出必須是截面選幣策略:頂層只用 selection:,不得產生 signals:、sizing:、risk:"
            "或 backtest:。執行後系統會由 payload 推導 kind=selection,不要自行新增 kind 欄位。\n"
            "必須保留 spec_version、target、strategy、run、data。run.mode 使用 backtest;"
            "選幣仍以 data.symbol=BTCUSDT 作為排程代表標的,market_type=futures,"
            "data.primary.interval=1h。data.symbol 只是排程代表標的;使用者要求一組候選時,"
            "不得自行猜 SUIUSDT 或其他單一候選。只有使用者明確列出候選宇宙時,"
            "才可用 universe.only_symbols 限制在那些標的。\n\n"
            "新聞或社群熱門度選幣必須使用現有 Blocks 路徑:先用"
            " universe.filter_quote_volume 過濾流動性,再用 universe.augment_with_news,"
            "並精確寫 with: [ticker_rank]。" + ranking_note +
            "top_k 依使用者要求,未指定時為 5;"
            "min_score: 1.0;dedupe_by: base_asset。這條 live 資料目前代表 Square 最近 24 小時"
            "的 ticker_rank,不要虛構不支援的 window 或 data.news 欄位。\n"
            "若使用者只要求挑選或排名,不要加 long_when/short_when,候選方向會是 neutral。"
            "只有使用者明確要求依情緒做多/做空時,才可加入 long_when/short_when。\n\n"
            "若使用者說『可能上漲、看漲、偏多』,只能把它保守映射為新聞情緒代理:"
            "在 augment_with_news 後加入 universe.filter_sentiment,參數"
            " min_bull_ratio: 0.55;不得承諾真的會上漲。『少見、未被市場發現』目前沒有"
            "直接欄位,不要在 description 宣稱已驗證;只能說使用流動性、提及量與情緒代理。\n\n"
            "=== 可用的新聞選幣 BLOCKS / 欄位 ===\n"
            "universe.filter_quote_volume\n"
            "universe.augment_with_news (with: [ticker_rank])\n"
            "universe.filter_sentiment (params: {min_bull_ratio: 0.55})\n"
            "news_mention_count, news_bull_ratio, news_unique_authors, quote_volume\n\n"
            "=== 範例輸出 ===\n" + selection_example
        )

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    return (
        "你是一個把「自然語言交易策略描述」轉成 YAML 規格的轉換器。"
        "只輸出一份合法 YAML,不要有 markdown 圍欄(```)、不要解釋、不要多餘文字。\n\n"
        "必須嚴格遵守下面的 schema 與可用 block 清單;只能使用清單內的 block,參數名要精確。\n"
        "不要使用 data.htf(單一時間框即可)。若使用者沒指定,採保守預設:"
        "market_type=futures、interval=1h、fees commission_bps=4 slippage_bps=2、size=0.95、"
        "出場預設 pct_stop_tp{stop_pct:0.02,tp_pct:0.04,max_bars:96}。\n"
        "若使用者語意含『下穿做空 / 空方』就同時給 entry.long 與 entry.short;否則 long-only(只給 long)。\n\n"
        "=== SCHEMA ===\n" + schema + "\n\n"
        "=== 可用 BLOCKS ===\n" + BLOCKS_CHEATSHEET + "\n\n"
        "=== 範例輸出 ===\n" + EXAMPLE_YAML
    )


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        # drop first fence line (``` or ```yaml) and trailing fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def call_llm(api_base: str, api_key: str, model: str, nl: str) -> str:
    base = api_base.rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    intent = classify_request(nl)
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": build_system_prompt(intent.kind, intent)},
            {"role": "user", "content": nl},
        ],
    }
    resp = requests.post(url, headers=headers, json=body, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return _strip_fences(content)


def convert_nl(api_base: str, api_key: str, model: str, nl: str) -> dict:
    """Convert, validate and ensure the generated YAML matches user intent."""
    intent = classify_request(nl)
    if intent.kind == "ambiguous":
        compound = "compound_selection_execution" in intent.evidence
        message = (
            "目前一份 YAML 不能同時表達『先跨幣別選幣，再對每個候選執行交易規則』;"
            "請拆成選幣與交易兩個需求"
            if compound else
            "無法可靠判斷你要『挑選一組候選幣』還是『為單一標的建立交易規則』;"
            "系統已停止,不會預設成技術分析"
        )
        return {
            "ok": True,
            "status": "needs_clarification",
            "yaml": "",
            "valid": False,
            "strategy_kind": "ambiguous",
            "generated_strategy_kind": None,
            "intent": intent.to_dict(),
            "errors": [message],
            "warnings": [],
        }

    if intent.kind == "selection":
        if "funding" in intent.sources and _LOWEST_FUNDING_RANKING.search(nl):
            return {
                "ok": True,
                "status": "unsupported",
                "yaml": "",
                "valid": False,
                "strategy_kind": "selection",
                "generated_strategy_kind": None,
                "intent": intent.to_dict(),
                "errors": [
                    "目前 funding selection 的 score 只支援由高到低排名;"
                    "『最負/最低 funding』需要明確的 ascending 排名規格,系統已停止,"
                    "不會反向選錯幣"
                ],
                "warnings": [],
            }
        unsupported = sorted(intent.sources & {"open_interest", "price_change"})
        if unsupported:
            return {
                "ok": True,
                "status": "unsupported",
                "yaml": "",
                "valid": False,
                "strategy_kind": "selection",
                "generated_strategy_kind": None,
                "intent": intent.to_dict(),
                "errors": [
                    "目前 selection runtime 尚未把 %s 的跨幣別 frame 接到 UniverseBundle;"
                    "系統已停止,不會偷偷改用新聞或技術分析"
                    % ", ".join(unsupported)
                ],
                "warnings": [],
            }
        if not intent.sources:
            return {
                "ok": True,
                "status": "needs_clarification",
                "yaml": "",
                "valid": False,
                "strategy_kind": "selection",
                "generated_strategy_kind": None,
                "intent": intent.to_dict(),
                "errors": ["請指定選幣依據，例如 Square 新聞熱度或 24h 成交量/流動性"],
                "warnings": [],
            }

    yaml_text = call_llm(api_base, api_key, model, nl)
    errors: list[str] = []
    warnings: list[str] = []
    generated_kind = None

    try:
        spec = yaml.safe_load(yaml_text)
    except Exception as exc:
        spec = None
        errors.append(f"YAML 解析失敗:{exc}")

    if not errors:
        if not isinstance(spec, dict):
            errors.append("YAML 不是有效的 mapping")
        else:
            validation_errors, validation_warnings = validate_spec(spec)
            errors.extend(validation_errors)
            warnings.extend(validation_warnings)
            generated_kind = _generated_strategy_kind(spec)
            # Semantic reconciliation expects a structurally valid tree. Running
            # it after (for example) ``with: 123`` turns a useful validation error
            # into an unrelated TypeError and a generic HTTP 500.
            if not validation_errors:
                alignment_errors, alignment_warnings = reconcile_intent(intent, spec)
                errors.extend(alignment_errors)
                warnings.extend(alignment_warnings)

    # Keep one actionable copy when structural and semantic gates report the
    # same underlying mismatch in slightly different phases.
    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))

    return {
        "ok": True,
        "status": "valid" if not errors else "rejected",
        "yaml": yaml_text,
        "valid": not errors,
        # Keep the independently inferred route even when model output is invalid, so the
        # UI can show the YAML in the correct editor without trying to execute it.
        "strategy_kind": intent.kind,
        "generated_strategy_kind": generated_kind,
        "intent": intent.to_dict(),
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_KLINES_HOST = {"spot": "https://api.binance.com/api/v3/klines",
                "futures": "https://fapi.binance.com/fapi/v1/klines"}
_FIXTURES = {
    ("BTCUSDT", "1h"): "BTCUSDT_1h_500bars.parquet",
    ("BTCUSDT", "4h"): "BTCUSDT_4h_300bars.parquet",
    ("BTCUSDT", "15m"): "BTCUSDT_15m_500bars.parquet",
    ("ETHUSDT", "1h"): "ETHUSDT_1h_500bars.parquet",
}


def _fixture_df(symbol: str, interval: str):
    name = _FIXTURES.get((symbol.upper(), interval))
    if not name:
        return None
    p = FIXTURE_DIR / name
    if not p.exists():
        return None
    return pd.read_parquet(p).reset_index(drop=True)


def fetch_klines(symbol: str, interval: str, market_type: str, limit: int = 1000):
    """Return an OHLCV DataFrame. Try Binance public REST, fall back to fixture."""
    host = _KLINES_HOST.get(market_type, _KLINES_HOST["futures"])
    try:
        r = requests.get(host, params={"symbol": symbol.upper(), "interval": interval,
                                       "limit": min(limit, 1000)}, timeout=20)
        r.raise_for_status()
        rows = r.json()
        df = pd.DataFrame({
            "open_time": [int(k[0]) for k in rows],
            "open": [float(k[1]) for k in rows],
            "high": [float(k[2]) for k in rows],
            "low": [float(k[3]) for k in rows],
            "close": [float(k[4]) for k in rows],
            "volume": [float(k[5]) for k in rows],
            "close_time": [int(k[6]) for k in rows],
            "quote_volume": [float(k[7]) for k in rows],
        })
        df["timestamp"] = df["close_time"]
        return df, "binance_%s" % market_type
    except Exception as exc:
        fx = _fixture_df(symbol, interval)
        if fx is not None:
            fx["timestamp"] = fx["close_time"]
            return fx, "fixture(offline: %s)" % type(exc).__name__
        raise RuntimeError(f"無法取得 {symbol} {interval} 行情:{exc}") from exc


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def run_backtest(yaml_text: str) -> dict:
    spec = yaml.safe_load(yaml_text)
    if not isinstance(spec, dict):
        return {"ok": False, "error": "YAML 不是有效的 mapping"}
    errors, warnings = validate_spec(spec)
    if errors:
        return {"ok": False, "error": "spec 驗證失敗", "errors": errors, "warnings": warnings}

    data = spec.get("data") or {}
    symbol = data["symbol"].upper()
    interval = (data.get("primary") or {})["interval"]
    market_type = data.get("market_type", "futures")
    entry = (spec.get("signals") or {}).get("entry") or {}
    exit_cfg = (spec.get("risk") or {}).get("exit")
    fees = (spec.get("risk") or {}).get("fees") or {}
    size = float((spec.get("sizing") or {}).get("size", 0.95))
    initial_capital = float((spec.get("backtest") or {}).get("initial_capital", 10000.0))
    long_only = (market_type == "spot") or (not entry.get("short"))

    df, source = fetch_klines(symbol, interval, market_type)
    if df is None or len(df) < 50:
        return {"ok": False, "error": "行情資料不足(需 ≥ 50 根)"}

    make_signals = build_make_signals(spec)
    result = run_vectorized_backtest(
        df=df, signal_fn=make_signals, exit_cfg=exit_cfg, timeframe=interval,
        size=size, fee_bps=float(fees.get("commission_bps", 4.0)),
        slippage_bps=float(fees.get("slippage_bps", 2.0)),
        initial_capital=initial_capital, long_only=long_only,
    )

    equity = result.equity_curve
    if equity is None:
        equity = np.full(len(df), initial_capital, dtype=float)
    equity = np.asarray(equity, dtype=float)

    # --- BTC buy-and-hold 基準(同區間,initial_capital 全押)---
    if symbol == "BTCUSDT":
        btc_close = df["close"].to_numpy(dtype=float)
    else:
        btc_df, _ = fetch_klines("BTCUSDT", interval, market_type, limit=len(df))
        btc_close = btc_df["close"].to_numpy(dtype=float)
    m = min(len(equity), len(btc_close), len(df))
    equity = equity[-m:]
    btc_close = btc_close[-m:]
    baseline = initial_capital * (btc_close / btc_close[0])
    ts = df["close_time"].to_numpy()[-m:].astype("int64").tolist()

    # 下採樣避免傳太大(圖用)
    def _ds(arr, k=600):
        arr = list(arr)
        if len(arr) <= k:
            return arr
        step = len(arr) / k
        return [arr[int(i * step)] for i in range(k)]

    bh_return = float(baseline[-1] / baseline[0] - 1.0)

    # --- 交易加上實際時間(entry_idx/exit_idx → close_time → 可讀時間)---
    from datetime import datetime, timezone

    def _iso(ms):
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    close_times = df["close_time"].to_numpy()
    enriched_trades = []
    for t in (result.trades[-8:] if result.trades else []):
        nt = {}
        ei, xi = t.get("entry_idx"), t.get("exit_idx")
        if ei is not None and 0 <= ei < len(close_times):
            nt["entry_time"] = _iso(close_times[ei])
        if xi is not None and 0 <= xi < len(close_times):
            nt["exit_time"] = _iso(close_times[xi])
        nt.update(t)
        enriched_trades.append(nt)

    return {
        "ok": True,
        "symbol": symbol, "interval": interval, "market_type": market_type,
        "bars": int(m), "data_source": source,
        "period": {"start": _iso(ts[0]), "end": _iso(ts[-1])} if ts else None,
        "metrics": {
            "total_return": result.total_return,
            "total_pnl": result.total_pnl,
            "final_equity": result.final_equity if result.final_equity != 1.0 else float(equity[-1]),
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "trade_count": result.trade_count,
            "avg_trade_pnl": result.avg_trade_pnl,
            "exposure": result.exposure,
        },
        "baseline": {
            "label": "BTC buy & hold",
            "total_return": bh_return,
            "total_pnl": float(baseline[-1] - baseline[0]),
            "final_equity": float(baseline[-1]),
        },
        "chart": {
            "timestamps": _ds(ts),
            "strategy": _ds(equity),
            "baseline": _ds(baseline),
            "initial_capital": initial_capital,
        },
        "trades_sample": enriched_trades,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 馬上抓資料 / 產生訊號 / 選幣
# ---------------------------------------------------------------------------


def fetch_live_bundle(symbol: str, interval: str, limit: int = 300) -> dict:
    """Fetch every catalog node live and return one ``cyqnt.input/v1`` dict.

    This is the input half of the demo. It matters that it is ONE call producing
    ONE artifact: the same ``build_live_bundle`` that paper/live uses, gated once
    on ``available_time``, with a status for every source it was asked for —
    including the ones that came back empty.
    """
    import time

    from cyqnt_trd.standard_bot.data.live_bundle import build_live_bundle

    started = time.time()
    bundle = build_live_bundle(symbol=symbol.upper(), interval=interval, limit=limit)
    elapsed = time.time() - started

    raw = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    frames = bundle.get("frames") or {}

    # Skeleton = everything except the rows, so the shape is readable at a glance.
    skeleton = {k: v for k, v in bundle.items() if k != "frames"}
    skeleton["frames"] = {
        key: "{ shape: %s, rows: [ … %d 列 … ] }" % (spec.get("shape"), len(spec.get("rows") or []))
        for key, spec in frames.items()
    }

    table = [
        {"node": key,
         "shape": spec.get("shape"),
         "rows": len(spec.get("rows") or []),
         "status": str((bundle.get("source_status") or {}).get(key, "")),
         # one real row — "here is the format" says nothing without a value in it
         "sample": (spec.get("rows") or [None])[-1]}
        for key, spec in frames.items()
    ]
    # Sources that were asked for and produced no frame at all still belong in the
    # report: "not fetched" and "fetched and empty" are different failures.
    for key, status in (bundle.get("source_status") or {}).items():
        if key not in frames:
            table.append({"node": key, "shape": "-", "rows": 0,
                          "status": str(status), "sample": None})

    return {
        "ok": True,
        "schema": bundle.get("schema"),
        "decision_time": bundle.get("decision_time"),
        "decision_time_basis": bundle.get("decision_time_basis"),
        "symbol": symbol.upper(), "interval": interval,
        "elapsed_sec": round(elapsed, 2),
        "bytes": len(raw.encode("utf-8")),
        "node_count": len(bundle.get("source_status") or {}),
        "row_total": sum(len(s.get("rows") or []) for s in frames.values()),
        "warnings": bundle.get("warnings") or [],
        "skeleton": skeleton,
        "table": sorted(table, key=lambda r: -r["rows"]),
    }


def _spec_from_yaml(yaml_text: str):
    """Parse + validate, returning ``(spec, error_payload)``."""
    spec = yaml.safe_load(yaml_text)
    if not isinstance(spec, dict):
        return None, {"ok": False, "error": "YAML 不是有效的 mapping"}
    errors, warnings = validate_spec(spec)
    if errors:
        return None, {"ok": False, "error": "spec 驗證失敗",
                      "errors": errors, "warnings": warnings}
    return spec, None


def make_signal(yaml_text: str) -> dict:
    """YAML → the latest ``cyqnt.signal/v2`` signal.

    The backtest answers "would this have made money". This answers "what does
    the strategy say RIGHT NOW, in the format a consumer receives" — which is the
    part of the contract a downstream team actually has to implement against.
    """
    spec, bad = _spec_from_yaml(yaml_text)
    if bad:
        return bad
    if isinstance(spec.get("selection"), dict):
        return {"ok": False, "error": "這是選幣 spec,請用下面的『執行選幣』"}

    import time

    from cyqnt_trd.standard_bot.data.live_snapshot import build_live_snapshot
    from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import (
        live_sections_for_spec, run_bundle)

    data = spec.get("data") or {}
    symbol = str(data["symbol"]).upper()
    interval = str((data.get("primary") or {})["interval"])
    market_type = data.get("market_type", "futures")

    started = time.time()
    _snapshot_obj, bundle = build_live_snapshot(
        sections=live_sections_for_spec(spec), symbol=symbol, interval=interval,
        market_type=market_type,
    )
    output = run_bundle(spec, bundle)
    signal = output["signals"][0] if output["signals"] else None
    bars = len(((bundle.get("frames") or {}).get("klines") or {}).get("rows") or [])
    if signal is None:
        return {"ok": True, "signal": None,
                "batch": output, "status": output["source_status"],
                "bars": bars, "as_of": output["decision_time"],
                "elapsed_sec": round(time.time() - started, 2),
                "note": "最後一根沒有觸發訊號 —— 這是正常結果,不是錯誤。"
                        "策略大多數時間應該是不動作的。"}
    return {"ok": True, "status": output["source_status"], "bars": bars,
            "as_of": output["decision_time"], "batch": output,
            "elapsed_sec": round(time.time() - started, 2),
            "envelope_version": signal["schema"],
            "signal": signal, "key_count": len(signal)}


def run_selection(yaml_text: str) -> dict:
    """Selection YAML → live universe → ranked basket → one v2 signal.

    Deliberately the same output schema as :func:`make_signal`: a consumer parses
    one contract and branches on ``kind``.
    """
    spec, bad = _spec_from_yaml(yaml_text)
    if bad:
        return bad
    if not isinstance(spec.get("selection"), dict):
        return {"ok": False, "error": "這不是選幣 spec(缺 selection:),請用上面的『產生訊號』"}

    import time

    from cyqnt_trd.standard_bot.data.live_snapshot import build_live_snapshot
    from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import (
        live_sections_for_spec, run_bundle)

    market_type = (spec.get("data") or {}).get("market_type", "futures")
    started = time.time()
    data = spec.get("data") or {}
    symbol = str(data.get("symbol") or "BTCUSDT").upper()
    interval = str((data.get("primary") or {}).get("interval") or "1h")
    _snapshot_obj, bundle = build_live_snapshot(
        sections=live_sections_for_spec(spec), symbol=symbol, interval=interval,
        market_type=market_type,
    )
    batch = run_bundle(spec, bundle)
    elapsed = time.time() - started
    signal = batch["signals"][0] if batch["signals"] else None
    if signal is None:
        return {"ok": False, "error": "選幣沒有產出 v2 signal",
                "status": batch["source_status"], "batch": batch}

    candidates = signal.get("candidates") or []
    frames = bundle.get("frames") or {}
    out = {"ok": True, "status": batch["source_status"],
           "as_of": batch["decision_time"], "batch": batch,
           "elapsed_sec": round(elapsed, 2),
           "universe_size": signal.get("universe_size"),
           "universe_rows": len((frames.get("universe") or {}).get("rows") or []),
           "rank_rows": len((frames.get("ticker_rank") or {}).get("rows") or []),
           "candidates": candidates,
           "envelope_version": signal["schema"]}
    if not candidates:
        out["note"] = ("篩選後沒有候選。常見原因:ticker_rank 回空(Square 快取冷)"
                       "或門檻設太嚴。這是真實結果,不是程式錯誤。")
        return out
    out["signal"] = signal
    out["key_count"] = len(signal)
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def log_message(self, fmt, *args):
        sys.stderr.write("[demo] " + (fmt % args) + "\n")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (HERE.parent / "index.html").read_text(encoding="utf-8")
            return self._send(200, html.encode("utf-8"), ctype="text/html")
        if self.path == "/api/schema":
            return self._send(200, {"schema": SCHEMA_PATH.read_text(encoding="utf-8")})
        if self.path.startswith("/api/example"):
            name = "example_selection.yaml"
            if "kind=trade" in self.path:
                name = "example_multi_source.yaml"
            path = SPEC_DIR / name
            if not path.exists():
                return self._send(404, {"ok": False, "error": "找不到 %s" % name})
            return self._send(200, {"ok": True, "name": name,
                                    "yaml": path.read_text(encoding="utf-8")})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/api/convert":
                b = self._read_json()
                if not b.get("nl", "").strip():
                    return self._send(400, {"ok": False, "error": "請輸入自然語言描述"})
                if not b.get("api_base") or not b.get("model"):
                    return self._send(400, {"ok": False, "error": "請填 LLM API Base URL 與 model"})
                return self._send(200, convert_nl(
                    b["api_base"], b.get("api_key", ""), b["model"], b["nl"]
                ))
            if self.path == "/api/backtest":
                b = self._read_json()
                return self._send(200, run_backtest(b.get("yaml", "")))
            if self.path == "/api/fetch":
                b = self._read_json()
                return self._send(200, fetch_live_bundle(
                    b.get("symbol", "BTCUSDT"), b.get("interval", "1h"),
                    int(b.get("limit", 300))))
            if self.path == "/api/signal":
                b = self._read_json()
                return self._send(200, make_signal(b.get("yaml", "")))
            if self.path == "/api/selection":
                b = self._read_json()
                return self._send(200, run_selection(b.get("yaml", "")))
            return self._send(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            sys.stderr.write(traceback.format_exc())
            return self._send(200, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[demo] serving on http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    print(f"[demo] repo_root={REPO_ROOT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] bye")


if __name__ == "__main__":
    main()
