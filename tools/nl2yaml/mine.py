"""Gate0 mining: turn the trading-intent chat export into split-safe candidates.

What this produces and why each piece exists:

* **Near-duplicate clusters, not just exact hashes.** ~48% of the corpus is a
  repeated query and the biggest single preset prompt card fires hundreds of
  times. A random train/test split over rows measures memorisation, not
  generalisation, so every row carries a ``split_group_key`` that a splitter
  must keep whole. Exact hashing is not enough: the preset cards reappear with a
  different ticker and a few extra lines of chit-chat, which is a different hash
  and a ~0.95 Jaccard neighbour, hence the MinHash/LSH pass.

* **A convertibility tier computed over the whole conversation.** Conditions
  live in later turns far more often than in the opening line: the same regex
  families score 1,572 rows as tier A on ``first_query`` alone and 3,826 rows
  once ``user_text_excerpt`` is included. Mining the first line only would throw
  away 59% of the convertible material.

* **A continuation-fragment flag.** Long chats had their opening turns dropped
  by context compression, so ``first_query`` is often "現在呢" / "process" /
  "策略状态". Handing those to an annotator invites invented conditions, so they
  are flagged and excluded from ``candidates.jsonl``.

* **Provenance, not labels.** ``primary_intent`` / ``is_coin_selection`` and
  friends were audited at 0.530 inter-annotator agreement on the 13 classes and
  0.639 self-agreement on *verbatim identical* input. They are recorded under
  ``mining_source`` as a cheap recall filter and are never used as a label or as
  a stratification variable.

Privacy. Both git remotes are public. User text and anything identifying is
written **only** to ``--internal-dir`` (outside the repo, mode 600); the
repo-bound record carries hashes, enums, counts and structured conditions with
the quote stripped. That split is enforced by :func:`assert_repo_safe`, which
rejects any repo-bound string containing whitespace or a non-ASCII character —
a Chinese quote sneaking into a ``conditions[]`` entry raises instead of
shipping.

The house rule throughout: raise rather than degrade silently. A missing column,
an unreadable salt, a wrong salt mode or an unexpected string in a repo-bound
record is an error, not a warning.

Usage::

    CSV="$NL2YAML_INTERNAL_ROOT/user_demand_analysis/2026-05_07_trading_intent"
    ./.venv-standard-bot/bin/python -m tools.nl2yaml.mine \
        --csv "$CSV/trading_intent_chats_2026-05_07_zh_en.csv" \
        --out tools/nl2yaml/dataset
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# The scope classifier is deliberately imported, not reimplemented. It is a
# deterministic regex pass with 40+ tests pinning it, and a second private copy
# of "is this a basket request or a single-symbol trade" would drift away from
# the one the conversion pipeline actually runs. If it moves or breaks, this
# module must fail loudly rather than fall back to a lookalike of its own.
from cyqnt_trd.standard_bot.yaml_pipeline.intent import classify_request

MINER_VERSION = "gate0-mine/1"
#: Names the tier vocabulary in every record. ``schema.CaseTier`` is a different
#: thing with the same short name — it grades what the block vocabulary can
#: express (t0..t4, from the capability adjudication), while this one counts how
#: many machine-checkable condition families the text contains (A..D). Two
#: artifacts of one dataset carrying a bare "tier" each would eventually be
#: joined on the wrong one.
TIER_SCHEME = "gate0_condition_families_abcd"

#: Columns this miner reads. A missing one is a schema change, and silently
#: mining a subset of the signal is exactly how a dataset ends up measuring
#: something other than what it claims to.
REQUIRED_COLUMNS = (
    "user_id", "chat_id", "month", "day", "lang", "zh_variant",
    "primary_intent", "secondary_intents", "label_source", "is_coin_selection",
    "selection_basis", "wants_automation", "wants_backtest", "wants_strategy",
    "n_user_msgs", "n_assistant_msgs", "preset_case", "kw_primary",
    "first_query", "user_text_excerpt",
)


# ---------------------------------------------------------------------------
# 1. Traditional/simplified folding
# ---------------------------------------------------------------------------

#: The corpus is 25,082 zh-Hans rows against 1,585 zh-Hant. A lexicon written in
#: Traditional characters only — 止損, 槓桿, 倉位, 篩選, 買進 — misses the
#: simplified form of the same word, which is where nearly all of the Chinese
#: volume is. Rather than enumerate both spellings of every phrase, patterns are
#: written once in Traditional and every character with a distinct simplified
#: form is expanded into a two-character class.
#:
#: Only the characters the patterns below actually use are listed; this is not a
#: general OpenCC replacement.
#: ``了`` is deliberately absent (it is both its own word and the simplified form
#: of ``瞭``): folding it would rewrite "好了" into "好瞭". Phrases that need it
#: spell both variants out.
_ZH_PAIRS = (
    "於于 過过 內内 萬万 億亿 個个 檔档 強强 幣币 標标 對对 幾几 兩两 線线 離离 黃黄 "
    "買买 賣卖 頭头 鐘钟 時时 週周 盤盘 單单 開开 進进 損损 槓杠 桿杆 倉仓 資资 風风 "
    "動动 額额 篩筛 選选 掃扫 漲涨 場场 謝谢 確确 認认 繼继 續续 後后 現现 著着 "
    "來来 執执 啟启 暫暂 狀状 態态 麼么 樣样 號号 檢检 說说 請请 體体 譯译 複复 詳详 "
    "細细 這这 們们 嗎吗 縮缩 監监 觸触 訊讯 級级 轉转 圍围 帶带 軌轨 產产 貨货 約约 "
    "條条 濾滤 種种 準准 網网 響响 錄录 記记 論论 討讨 觀观 應应 該该 會会 為为 "
    "價价 勢势 較较 點点 數数 險险 賺赚 幫帮 給给 東东 華华 遠远 別别 熱热 門门 "
    "薦荐 測测 聞闻 費费 錢钱 歷历 驗验 隱隐 顯显 慮虑 圖图 據据 樓楼 況况 運运 "
    "當当 勝胜 僅仅 實实 寫写 處处 問问 沒没 趨趋 隨随 題题 釋释 蹤踪 機机 餅饼 "
    "負负 冪幂 兇凶"
)

_ZH_CLASS: dict[str, str] = {}
#: Simplified -> traditional, single direction, for feeding text to a classifier
#: whose own vocabulary is written in Traditional (see :func:`spec_shape`).
_SIMP_TO_TRAD: dict[str, str] = {}
for _pair in _ZH_PAIRS.split():
    _trad, _simp = _pair[0], _pair[1]
    _cls = f"[{_trad}{_simp}]"
    _ZH_CLASS[_trad] = _cls
    _ZH_CLASS[_simp] = _cls
    _SIMP_TO_TRAD[_simp] = _trad

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def zh_fold(pattern: str) -> str:
    """Expand CJK characters in ``pattern`` into traditional/simplified classes.

    CJK characters carry no regex meaning, so a plain textual substitution is
    safe *outside* a character class. Inside one it is not: expanding ``兩`` in
    ``[一二兩]`` would nest ``[兩两]`` in the class and silently change what the
    pattern matches (the class would end at the inner ``]``, and the rest would
    become a required literal). That mistake is undetectable by eye in a wall of
    CJK, so it raises here instead. Write alternations, not classes.
    """
    depth = 0
    previous = ""
    for ch in pattern:
        if ch == "[" and previous != "\\":
            depth += 1
        elif ch == "]" and previous != "\\" and depth:
            depth -= 1
        elif depth and _CJK.match(ch):
            raise ValueError(
                f"CJK character {ch!r} inside a character class in {pattern!r}; "
                "use an alternation so trad/simp folding stays safe"
            )
        previous = ch
    return "".join(_ZH_CLASS.get(ch, ch) for ch in pattern)


def to_traditional(text: str) -> str:
    """Rewrite the simplified characters this module knows about."""
    return "".join(_SIMP_TO_TRAD.get(ch, ch) for ch in text)


def _rx(pattern: str) -> re.Pattern:
    return re.compile(zh_fold(pattern), re.IGNORECASE)


# ---------------------------------------------------------------------------
# 2. Normalisation and hashing
# ---------------------------------------------------------------------------

#: Turns of a chat are joined with " --- " by the upstream extractor.
_MSG_SEP = re.compile(r"\s+---\s+")
#: Turns that are not utterances at all. They reached the excerpt because the
#: extractor took whatever sat in the user role: platform notices, cron failure
#: alerts pasted back into the chat, the agent runtime's own loop marker, and the
#: preset-card injection (whose identity is already carried by ``preset_case``).
#: Mining them manufactures conditions nobody asked for — a cron alert quoting
#: "1 hour" is not a user asking for the 1h timeframe.
_INJECTED = re.compile(
    r"^(?:"
    r"<system>|"
    r"\[system\b|"
    r"system:\s*\[|"
    r"⚠️?\s*cron job\b|"
    r"cron job \".*?\" (?:failed|interrupted)|"
    r"continue the openclaw runtime event|"
    r"user selected the [a-z0-9-]+ case\.|"
    r"conversation info \(untrusted metadata\)"
    r")",
    re.IGNORECASE,
)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HANDLE = re.compile(r"(?<![\w/])@[A-Za-z0-9_]{2,}")
#: Zero-width and bidi controls: invisible in review, but they change a hash.
_INVISIBLE = re.compile("[​-‏‪-‮⁠﻿]")


def canonicalise(text: str) -> str:
    """NFKC, drop URLs/@handles, casefold, collapse whitespace.

    NFKC matters more here than usual: the export contains full-width digits and
    Latin ("１０倍"), which would otherwise miss every numeric pattern and hash
    apart from its half-width twin.
    """
    out = unicodedata.normalize("NFKC", text or "")
    out = _INVISIBLE.sub("", out)
    out = _URL.sub(" ", out)
    out = _HANDLE.sub(" ", out)
    out = out.casefold()
    return " ".join(out.split())


def sha256_hex(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def short_hash(text: str, prefix: str, size: int = 16) -> str:
    return prefix + hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:size]


def split_messages(first_query: str, excerpt: str) -> list[str]:
    """Canonical turns of one chat, first turn first, without double counting.

    ``user_text_excerpt`` normally already starts with ``first_query``, but the
    two fields are truncated at different lengths (1,000 vs 1,200 chars), so the
    overlap can be a prefix rather than an exact repeat. The prefix test is
    applied *only* between ``first_query`` and the excerpt's opening turn, where
    that truncation is the known cause; applying it to every pair would delete
    real turns, because a user who writes "做多" and then "做多 BTC 4h" has said
    two different things and the second is the one worth mining. Keeping a
    duplicate opening turn instead would distort the "is every turn chit-chat"
    ratio the fragment detector depends on.
    """
    opening = canonicalise(first_query)
    if _INJECTED.match(opening):
        opening = ""
    rest = [p for p in (canonicalise(p) for p in _MSG_SEP.split(excerpt or ""))
            if p and not _INJECTED.match(p)]
    kept: list[str] = [opening] if opening else []
    for index, part in enumerate(rest):
        if index == 0 and opening and (
                part.startswith(opening) or opening.startswith(part)):
            if len(part) > len(opening):
                kept[0] = part      # the excerpt held more of the opening turn
            continue
        if part in kept:
            continue
        kept.append(part)
    return kept


# ---------------------------------------------------------------------------
# 3. Machine-checkable condition families
# ---------------------------------------------------------------------------

#: Families whose presence pins down a *rule* rather than a mere preference.
#: ``asset`` and ``direction`` say which instrument and which side; they never
#: say when to act, so a request carrying only those two is not a two-condition
#: strategy (see :func:`tier_of`).
RULE_FAMILIES = frozenset({
    "threshold", "rank_topn", "indicator", "timeframe", "risk", "universe_filter",
})
ALL_FAMILIES = tuple(sorted(RULE_FAMILIES | {"direction", "asset"}))

_CJK_NUM = {"一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _to_int(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        return int(token)
    return _CJK_NUM.get(token)


# --- threshold -------------------------------------------------------------
#
# The published spelling only matched "operator then number" (``低於 30``). In
# Chinese the comparator just as often *follows* the number — "成交量在 1000 萬
# 以上", "勝率 60% 以上" — so both orders are matched. ``突破``/``跌破`` are
# deliberately absent here even though they are comparisons: they already count
# under ``indicator``, and matching them twice would inflate the tier of a
# single stated condition.
_THRESHOLD_PRE = _rx(
    r"(?P<op>>=|<=|=>|=<|>|<|≥|≤|大於|小於|高於|低於|不低於|不高於|不超過|超過|至少|最少|最多|不足)"
    r"\s*\$?(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>%|萬|億|k|m|b)?"
)
_THRESHOLD_POST = _rx(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>%|萬|億|k|m|b)?\s*(?P<op>以上|以下|之上|之下|以內)"
)
#: Comparators, in the vocabulary of ``schema.Operator`` (``gte``/``lte``, not
#: ``ge``/``le``). Gate0 emits a subset of that enum and never invents a member;
#: a test pins the subset against the enum so a rename there fails here.
_OP_CANON = {
    ">": "gt", ">=": "gte", "=>": "gte", "<": "lt", "<=": "lte", "=<": "lte",
    "≥": "gte", "≤": "lte",
    "大於": "gt", "大于": "gt", "高於": "gt", "高于": "gt", "超過": "gt", "超过": "gt",
    "小於": "lt", "小于": "lt", "低於": "lt", "低于": "lt", "不足": "lt",
    "不低於": "gte", "不低于": "gte", "至少": "gte", "最少": "gte",
    "不高於": "lte", "不高于": "lte", "不超過": "lte", "不超过": "lte", "最多": "lte",
    "以上": "gte", "之上": "gt", "以下": "lte", "之下": "lt", "以內": "lte", "以内": "lte",
}
_UNIT_SCALE = {"萬": 1e4, "万": 1e4, "億": 1e8, "亿": 1e8, "k": 1e3, "m": 1e6, "b": 1e9}

#: What the number in front of a comparator is measured on. Only the 12
#: characters before the match are inspected, so "rsi 低於 30" resolves and a
#: bare "低於 30" stays ``unspecified`` instead of being guessed.
_SUBJECT_HINTS = (
    ("rsi", _rx(r"rsi")),
    ("macd", _rx(r"macd")),
    ("volume", _rx(r"成交量|交易量|\bvolume\b|\bvol\b")),
    ("quote_volume", _rx(r"成交額|交易額|turnover")),
    ("market_cap", _rx(r"市值|market\s*cap|\bmcap\b")),
    ("price", _rx(r"價格|價位|股價|\bprice\b")),
    ("price_change_pct", _rx(r"漲幅|跌幅|漲跌|change|gain|drop")),
    ("funding_rate", _rx(r"資金費率|費率|funding")),
    ("win_rate", _rx(r"勝率|win\s*rate")),
    ("leverage", _rx(r"槓桿|leverage")),
    ("atr", _rx(r"\batr\b")),
)


def _subject_before(text: str, start: int, window: int = 14) -> str:
    left = text[max(0, start - window):start]
    for name, rx in _SUBJECT_HINTS:
        if rx.search(left):
            return name
    return "unspecified"


def _scaled(num: str, unit: str | None) -> float:
    value = float(num)
    if unit:
        value *= _UNIT_SCALE.get(unit.lower(), 1.0)
    return value


#: Subjects whose thresholds legitimately run into the billions, so a large
#: number next to them is a market quantity rather than an identifier.
_LARGE_MAGNITUDE_SUBJECTS = frozenset({"volume", "quote_volume", "market_cap", "price"})

#: Unit of a threshold value, keyed by subject, in the vocabulary of
#: ``schema.Unit``. A bare number is not a condition: "> 2 million" has been read
#: as 2, 2e6 and 2_000_000, so the unit travels with the value or the value is
#: recorded as unitless on purpose. 成交量 is a coin count while 成交額 is quote
#: currency, which is exactly the distinction ``count`` vs ``usd`` carries.
_SUBJECT_UNIT = {
    "volume": "count",
    "quote_volume": "usd",
    "market_cap": "usd",
    "price": "usd",
    "price_change_pct": "pct",
    "win_rate": "pct",
    "funding_rate": "pct",
    "leverage": "leverage_x",
    "rsi": "indicator_value",
    "macd": "indicator_value",
    "atr": "indicator_value",
}
NUMBER_REDACTION_LIMIT = 1e6


def safe_number(subject: str, value: float | None) -> tuple[float | None, bool]:
    """Drop numbers big enough to be an identifier rather than a threshold.

    Found while auditing the first full run: a user had pasted their own
    Binance/Telegram id into the chat, "…> 76916505…" matched the threshold
    pattern, and the id was written verbatim into the repo-bound record as a
    ``threshold`` value with ``subject: unspecified``. Structured conditions are
    allowed in the repo, but a raw eight-digit number lifted out of a message is
    a quasi-identifier, and paired with ``pseudonym_id`` it is a re-identification
    handle.

    So: a value at or above a million is kept only when the subject says it is a
    market quantity. Nothing of modelling value is lost — an unattributed
    ``> 1244817253`` was never a condition a spec could express — and the shape of
    the finding survives as ``value_redacted``.
    """
    if value is None or value < NUMBER_REDACTION_LIMIT:
        return value, False
    if subject in _LARGE_MAGNITUDE_SUBJECTS:
        return value, False
    return None, True


# --- rank / top-N ----------------------------------------------------------
#
# The published spelling included a bare ``\d+\s*個``, which fires on "8 個小時"
# and every other counted noun in the language. A top-N condition needs either
# an explicit ranking word or a counted *instrument*, so the counter noun is
# required to be one that denotes a coin/pair.
_INSTRUMENT = r"(?:幣|標的|代幣|交易對|貨幣|檔|支|coins?|tokens?|pairs?|symbols?|altcoins?)"
_CJK_DIGIT = r"(?:一|二|三|四|五|六|七|八|九|十|兩)"
_RANK_PATTERNS = (
    _rx(r"前\s*(?P<n>\d+|" + _CJK_DIGIT + r")\s*(?:名|強|大|" + _INSTRUMENT + r")?"),
    _rx(r"\btop\s*(?P<n>\d+)\b"),
    _rx(r"(?P<n>\d+)\s*(?:個|支|檔|只)?\s*" + _INSTRUMENT),
)
_RANK_BARE = _rx(r"排名|排行|榜單|漲幅榜|跌幅榜|\branking\b|\brank\s+(?:by|the)\b|leaderboard")

# --- indicator -------------------------------------------------------------
_INDICATOR_NAMES = (
    ("rsi", r"\brsi\b|相對強弱"),
    ("macd", r"\bmacd\b"),
    ("ema", r"\bema\s*\d*|指數均線"),
    ("sma", r"\bsma\s*\d*"),
    ("ma", r"\bma\s*\d+|均線|移動平均"),
    ("bollinger", r"布林|bollinger|\bbb\b|\bboll\b"),
    ("kdj", r"\bkdj\b|隨機指標|stoch"),
    ("atr", r"\batr\b|真實波幅"),
    ("supertrend", r"supertrend|超級趨勢"),
    ("vwap", r"\bvwap\b"),
    ("obv", r"\bobv\b"),
    ("adx", r"\badx\b"),
    ("cci", r"\bcci\b"),
    ("ichimoku", r"ichimoku|一目均衡"),
    ("fibonacci", r"斐波|fibonacci|\bfib\b|黃金比例"),
    ("breakout", r"突破|breakout|break\s+above|站上"),
    ("breakdown", r"跌破|breakdown|break\s+below"),
    ("cross_up", r"金叉|黃金交叉|golden\s*cross|cross(?:es|ed)?\s+above"),
    ("cross_down", r"死叉|死亡交叉|death\s*cross|cross(?:es|ed)?\s+below"),
    ("divergence", r"背離|divergence"),
    ("overbought", r"超買|overbought"),
    ("oversold", r"超賣|oversold"),
)
_INDICATOR_RX = tuple((name, _rx(pat)) for name, pat in _INDICATOR_NAMES)

# --- timeframe -------------------------------------------------------------
#
# Single-letter units are required to sit flush against the digits ("4h",
# "15m"). Allowing a space there turns "$10 M" of market cap and "1000 d" of
# anything into a timeframe.
_TF_COMPACT = _rx(r"\b(?P<n>\d{1,3})(?P<u>[mhdw])\b")
_TF_SPACED = _rx(
    r"\b(?P<n>\d{1,3})\s*(?P<u>分鐘|分|小時|天|日|週|月|min(?:ute)?s?|hours?|hrs?|days?|weeks?)\b"
)
_TF_NAMED = _rx(r"日線|週線|月線|小時線|\bdaily\b|\bweekly\b|\bhourly\b|\bintraday\b|盤中|日內")
_TF_UNIT_CANON = {
    "m": "m", "分": "m", "分鐘": "m", "分钟": "m", "min": "m", "mins": "m",
    "minute": "m", "minutes": "m",
    "h": "h", "小時": "h", "小时": "h", "hour": "h", "hours": "h", "hr": "h", "hrs": "h",
    "d": "d", "天": "d", "日": "d", "day": "d", "days": "d",
    "w": "w", "週": "w", "周": "w", "week": "w", "weeks": "w",
    "月": "M",
}
_TF_NAMED_CANON = {
    "日線": "1d", "日线": "1d", "週線": "1w", "周线": "1w", "月線": "1M", "月线": "1M",
    "小時線": "1h", "小时线": "1h", "daily": "1d", "weekly": "1w", "hourly": "1h",
    "intraday": "intraday", "盤中": "intraday", "盘中": "intraday",
    "日內": "intraday", "日内": "intraday",
}

# --- direction -------------------------------------------------------------
#
# ``\blong\b`` alone matches "as long as" and "long term", which is how a chat
# about nothing at all acquires a side. Both are excluded explicitly; the
# Chinese forms need no such guard.
_DIRECTION_RX = (
    ("long", _rx(r"做多|開多|加多|多單|看多|買進|買入|go\s+long|(?<!as )\blongs?\b(?!\s*(?:term|run|-term))")),
    ("short", _rx(r"做空|開空|加空|空單|看空|賣出|go\s+short|\bshorts?\b(?!\s*(?:term|-term))")),
)

# --- risk ------------------------------------------------------------------
_RISK_RX = (
    ("stop_loss", _rx(r"止損|停損|stop\s*-?\s*loss|\bsl\b")),
    ("take_profit", _rx(r"止盈|停利|take\s*-?\s*profit|\btp\b")),
    ("trailing_stop", _rx(r"移動止損|追蹤止損|trailing\s*stop")),
    ("leverage", _rx(r"槓桿|leverage|\d+\s*倍|\b\d+x\b")),
    ("position_size", _rx(r"倉位|部位|position\s*siz|資金管理|下注比例")),
    ("max_drawdown", _rx(r"最大回撤|回撤|drawdown")),
    ("risk_control", _rx(r"風控|風險控制|risk\s*(?:control|management)")),
)
#: A percent or multiple sitting next to the risk word, e.g. "止損 2%" — or
#: "90%止損", because Chinese puts the quantity on either side.
_RISK_VALUE = _rx(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>%|倍|x)")
RISK_VALUE_WINDOW = 8


def nearest_risk_value(text: str, span: tuple[int, int]):
    """The quantity closest to a risk keyword, looking both ways.

    Looking only *after* the keyword read "挂90%止损单跟…200%的半仓止盈单" as a
    200% stop loss: the stop's own quantity sits in front of it and the number it
    found belonged to the take-profit on the other side of the clause. A stop of
    200% instead of 90% is not a rounding error, and once it is in a gold spec
    nothing downstream can tell it was ever wrong.
    """
    before = text[max(0, span[0] - RISK_VALUE_WINDOW):span[0]]
    after = text[span[1]:span[1] + RISK_VALUE_WINDOW]
    candidates = [(len(before) - m.end(), m) for m in _RISK_VALUE.finditer(before)]
    # Ties go to the trailing form ("止損 2%"), the more explicit of the two.
    candidates += [(m.start() - 0.5, m) for m in _RISK_VALUE.finditer(after)]
    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair[0])[1]

# --- universe filter -------------------------------------------------------
#
# ``只要`` and ``不要`` from the published spelling are dropped: they are
# ordinary Chinese function words ("只要你…", "不要跟我說"), and they were
# turning conversations with no screening intent into universe filters.
_UNIVERSE_RX = (
    ("volume", _rx(r"成交量|交易量|\bvolume\b|放量")),
    ("quote_volume", _rx(r"成交額|交易額|turnover")),
    ("liquidity", _rx(r"流動性|liquidity|深度")),
    ("market_cap", _rx(r"市值|market\s*cap|\bmcap\b")),
    ("exclude", _rx(r"排除|剔除|\bexclude\b|不包含|不包括|去掉")),
    ("screen", _rx(r"篩選|掃描|篩出|挑出|選出|screen(?:er|ing)?\b|\bscan\b|\bfilter\b")),
    ("gainers", _rx(r"漲幅榜|漲幅排行|gainers|top\s*movers")),
    ("losers", _rx(r"跌幅榜|跌幅排行|losers")),
    ("whitelist", _rx(r"白名單|whitelist|只交易|僅限")),
    ("blacklist", _rx(r"黑名單|blacklist")),
    ("full_market", _rx(r"全市場|所有交易對|全部幣種|whole\s*market|all\s*pairs")),
)

# --- asset -----------------------------------------------------------------
#
# Tickers that are also English words (op, near, link, sui, ton, arb) are left
# out: "near the top" is not a mention of NEAR, and a false asset is a false
# universe restriction in the resulting spec.
_ASSET_RX = (
    ("BTC", _rx(r"\bbtc\b|bitcoin|比特幣|大餅")),
    ("ETH", _rx(r"\beth\b|ethereum|以太")),
    ("BNB", _rx(r"\bbnb\b")),
    ("SOL", _rx(r"\bsol\b|solana")),
    ("XRP", _rx(r"\bxrp\b|ripple|瑞波")),
    ("DOGE", _rx(r"\bdoge\b|狗狗幣")),
    ("ADA", _rx(r"\bada\b|cardano")),
    ("AVAX", _rx(r"\bavax\b")),
    ("PEPE", _rx(r"\bpepe\b")),
    ("SHIB", _rx(r"\bshib\b")),
    ("LTC", _rx(r"\bltc\b")),
    ("TRX", _rx(r"\btrx\b")),
)

MAX_CONDITIONS = 40


def extract_conditions(text: str) -> tuple[list[dict], Counter, int]:
    """Structured, machine-checkable conditions found in ``text``.

    Returns ``(conditions, per_family_match_counts, n_dropped)``. Each condition
    carries an internal-only ``quote`` (the matched span) so a human can audit
    the extraction; :func:`repo_condition` strips it before the record leaves for
    the repo.

    ``operator``, ``unit`` and ``polarity`` use the vocabularies of
    ``schema.Operator`` / ``schema.Unit`` / ``schema.Polarity`` so a Gate0
    condition can be lifted into a ``schema.Condition`` without a translation
    table in between. ``polarity`` in particular is worth carrying this early:
    exclusions ("exclude tradfi") are the conditions that get silently dropped
    downstream, and they cannot be counted if the miner never recorded them.
    """
    found: list[dict] = []
    seen: set[tuple] = set()
    hits: Counter = Counter()
    dropped = 0

    def add(family: str, subject: str, operator: str, value, span,
            unit: str | None = None, polarity: str = "include") -> None:
        nonlocal dropped
        hits[family] += 1
        redacted = False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Keep the original object so an int top-N stays an int.
            _, redacted = safe_number(subject, float(value))
            if redacted:
                value = None
                unit = None
        key = (family, subject, operator, repr(value), unit, polarity, redacted)
        if key in seen:
            return
        seen.add(key)
        if len(found) >= MAX_CONDITIONS:
            dropped += 1
            return
        condition = {
            "family": family, "subject": subject, "operator": operator,
            "value": value, "unit": unit, "polarity": polarity,
            "quote": text[max(0, span[0] - 10):span[1] + 6],
        }
        if redacted:
            condition["value_redacted"] = True
        found.append(condition)

    def threshold_unit(subject: str, unit_token: str | None) -> str | None:
        # An explicit "%" in the text beats the subject's default: "漲幅 > 5%" is
        # a percentage whatever the column is usually measured in.
        if unit_token == "%":
            return "pct"
        return _SUBJECT_UNIT.get(subject)

    for pattern, lowercase_op in ((_THRESHOLD_PRE, True), (_THRESHOLD_POST, False)):
        for match in pattern.finditer(text):
            token = match.group("op")
            op = _OP_CANON.get(token.lower() if lowercase_op else token)
            if op is None:
                raise AssertionError(f"unmapped threshold operator: {token!r}")
            subject = _subject_before(text, match.start())
            add("threshold", subject, op,
                _scaled(match.group("num"), match.group("unit")), match.span(),
                unit=threshold_unit(subject, match.group("unit")))

    for rx in _RANK_PATTERNS:
        for match in rx.finditer(text):
            n = _to_int(match.group("n"))
            if n is None or not 1 <= n <= 500:
                continue
            add("rank_topn", "universe", "top_n", n, match.span(), unit="count")
    for match in _RANK_BARE.finditer(text):
        # A ranking with no N stated. "unspecified" rather than top_n with a
        # guessed count: inventing "top 10" here is how a proxy becomes gold.
        add("rank_topn", "universe", "unspecified", None, match.span())

    for name, rx in _INDICATOR_RX:
        match = rx.search(text)
        if match:
            add("indicator", name, "exists", None, match.span())

    def add_interval(value: str, span) -> None:
        add("timeframe", "interval", "eq", value, span, unit="label")

    for match in _TF_COMPACT.finditer(text):
        unit = _TF_UNIT_CANON[match.group("u").lower()]
        add_interval(f"{int(match.group('n'))}{unit}", match.span())
    for match in _TF_SPACED.finditer(text):
        unit = _TF_UNIT_CANON.get(match.group("u").lower())
        if unit is None:
            continue
        add_interval(f"{int(match.group('n'))}{unit}", match.span())
    for match in _TF_NAMED.finditer(text):
        canon = _TF_NAMED_CANON.get(match.group(0).lower())
        if canon is None:
            continue
        add_interval(canon, match.span())

    for side, rx in _DIRECTION_RX:
        match = rx.search(text)
        if match:
            add("direction", "side", "eq", side, match.span(), unit="label")

    for subject, rx in _RISK_RX:
        match = rx.search(text)
        if not match:
            continue
        value_match = nearest_risk_value(text, match.span())
        value = float(value_match.group("num")) if value_match else None
        unit = None
        if value is not None:
            token = value_match.group("unit").lower()
            unit = "leverage_x" if token in {"倍", "x"} else "pct"
        add("risk", subject, "eq" if value is not None else "exists", value,
            match.span(), unit=unit)

    for subject, rx in _UNIVERSE_RX:
        match = rx.search(text)
        if match:
            add("universe_filter", subject, "exists", None, match.span(),
                polarity="exclude" if subject in {"exclude", "blacklist"} else "include")

    assets: list[str] = []
    first_span = None
    for ticker, rx in _ASSET_RX:
        match = rx.search(text)
        if match:
            assets.append(ticker)
            if first_span is None or match.start() < first_span[0]:
                first_span = match.span()
    if assets:
        add("asset", "asset", "in", sorted(assets), first_span, unit="symbol")

    return found, hits, dropped


def tier_of(families: set[str]) -> str:
    """Convertibility tier from the set of families present.

    ``>=3 -> A``, ``2 -> B``, ``1 -> C``, ``0 -> D`` as specified, with one
    adjustment: a row whose only two families are ``asset`` and ``direction``
    is demoted from B to C. "Should I long BTC?" names an instrument and a side
    and nothing a backtest could check, so calling it a two-condition request
    would seed the training set with specs whose entry rule was invented by the
    annotator. Note this is the *only* case the adjustment can touch — any three
    families necessarily include a rule family.
    """
    n = len(families)
    if n == 0:
        return "D"
    if not families & RULE_FAMILIES:
        return "C"
    if n >= 3:
        return "A"
    return "B" if n == 2 else "C"


# ---------------------------------------------------------------------------
# 4. Continuation fragments
# ---------------------------------------------------------------------------

#: Turns that carry no request of their own: acknowledgements, "and now?",
#: status polls, and instructions about the reply language. Matched against a
#: whole turn (punctuation and emoji stripped), never as a substring, so
#: "繼續分析 BTC 4h 的均線" is not mistaken for a bare "繼續".
_CHATTER_ALTERNATIVES = (
    # acknowledgement / assent
    r"好|好的|好喔|嗯+|知道了|收到|瞭解|明白|沒問題|謝謝|感謝|thanks|thank\s*you|"
    r"ok|okay|k|yes|yep|no|nope|y|n|確認|confirm|confirmed|是|對|可以|行|同意|了解|"
    # continue / control
    r"繼續|繼續啊|然後呢|然後|接著|接下來|現在呢|那現在呢|那現在|再來|再一次|下一步|"
    r"go|go\s*on|go\s*ahead|continue|next|proceed|process|run|start|開始|執行|"
    r"重啟|重新開始|停|停止|暫停|stop|pause|restart|resume|退出|exit|quit|"
    # status polling. The "…好了嗎 / 目前狀況" forms are the observed shape of a
    # follow-up whose antecedent was compressed away, so the optional verb and
    # the interchangeable 情況/狀況/狀態 tail are spelled out rather than left to
    # a fixed phrase list.
    r"狀態|策略狀態|現在狀態|目前狀態|status|strategy\s*status|positions?|持倉|我的持倉|"
    r"(?:改|修|弄|做|處理|跑|測|生成|寫)?好了嗎?|完成了嗎?|可以了嗎?|卡了嗎?|過來了嗎?|"
    r"(?:目前|現在|當前)?(?:運行|執行)?(?:情況|狀況|狀態)如何?|"
    r"(?:目前|現在|當前)(?:運行|執行)?(?:情況|狀況|狀態)|"
    r"怎麼樣了|怎麼樣|怎麼了|如何|有信號嗎?|有沒有信號|"
    r"更新|更新一下|更新了嗎?|any\s*updates?|updates?\s*now|"
    r"update|check|檢查|檢查一下|ready|done|完成|"
    # meta / language instructions
    r"用中文回答我?|用中文|說中文|請說中文|請用中文回答我?|請用中文|中文|"
    r"用繁體|用繁體中文|請用英文|用英文|english|in\s*english|answer\s*in\s*\w+|"
    r"reply\s*in\s*\w+|翻譯|翻譯一下|再說一次|重複|重複一次|詳細一點|詳細點|說清楚|"
    r"說明|解釋|why|為什麼|怎麼說|"
    # bare deixis
    r"這個|那個|它|他|他們|這|那|this|that|it"
)
_CHATTER = re.compile(zh_fold(rf"^(?:{_CHATTER_ALTERNATIVES})$"), re.IGNORECASE)
#: Stripped before the whole-turn match: trailing punctuation, emoji, symbols.
_TRIM = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def is_chatter(message: str) -> bool:
    core = _TRIM.sub("", message)
    if not core:
        return True
    if _CHATTER.match(core):
        return True
    # A one- or two-character turn with no digit and no Latin word cannot hold a
    # checkable condition; it is a grunt or a truncation artefact.
    return len(core) <= 2 and not re.search(r"\d|[a-z]{2,}", core)


def fragment_verdict(messages: list[str], n_families: int) -> tuple[bool, str, bool]:
    """``(is_continuation_fragment, reason, leading_chatter)``.

    Two ways a row is unusable. Either every turn we can see is chit-chat, or
    the chat opens on a follow-up ("現在呢") *and* nothing checkable survives
    anywhere in the excerpt — in both cases the request itself was lost to
    context compression and an annotator asked to produce a spec would have to
    invent one.

    A row that opens on chit-chat but does carry conditions later is kept, with
    ``leading_chatter`` set so downstream review knows the opening context is
    missing.
    """
    if not messages:
        return True, "empty", True
    leading = is_chatter(messages[0])
    if all(is_chatter(m) for m in messages):
        return True, "all_chatter", leading
    if leading and n_families == 0:
        return True, "leading_chatter_no_conditions", True
    return False, "", leading


# ---------------------------------------------------------------------------
# 5. Spec shape
# ---------------------------------------------------------------------------

def spec_shape(text: str, families: set[str]) -> tuple[str, str, tuple[str, ...]]:
    """``(shape, base_kind_from_classify_request, evidence)``.

    ``classify_request`` is the authority on selection-vs-trade. It has no
    "both" bucket — a request that ranks a universe *and* states a per-bar entry
    rule comes back ``ambiguous`` with ``compound_selection_execution`` in the
    evidence, because there is no single-spec grammar for it. Gate0 does want to
    count those, so that evidence tag is promoted to ``both``, and two narrow
    upgrades are layered on top:

    * ``selection`` + a stated side/risk and a trigger -> ``both``;
    * ``trade`` + an explicit top-N -> ``both``.

    Only ``rank_topn`` counts as cross-sectional evidence for the second rule.
    ``universe_filter`` does not: "long BTC when volume expands" filters one
    symbol over time, not a cross-section, and treating it as selection would
    relabel a large slice of ordinary trade requests.

    The text is script-normalised on the way in. Every Chinese rule in
    ``intent.py`` is written in Traditional (選幣, 買進, 做多, 幣種), while 94% of
    the Chinese in this corpus is simplified — feeding it raw would return
    ``ambiguous`` for almost every Chinese row and make this column meaningless.
    Normalising the *input* keeps the classifier the single source of truth for
    scope; the alternative, editing its rules, would fork the pinned behaviour.
    """
    decision = classify_request(to_traditional(text))
    base = decision.kind
    trade_signal = bool(families & {"direction", "risk"}) and bool(
        families & {"indicator", "threshold", "timeframe"})
    selection_signal = "rank_topn" in families

    if base == "ambiguous":
        if "compound_selection_execution" in decision.evidence:
            return "both", base, decision.evidence
        return "unclear", base, decision.evidence
    if base == "selection":
        return ("both" if trade_signal else "selection"), base, decision.evidence
    if base == "trade":
        return ("both" if selection_signal else "trade"), base, decision.evidence
    raise AssertionError(f"classify_request returned an unknown kind: {base!r}")


# ---------------------------------------------------------------------------
# 6. Near-duplicate clustering (MinHash + LSH, verified)
# ---------------------------------------------------------------------------

NUM_PERM = 128
BAND_ROWS = 16          # 8 bands x 16 rows -> LSH threshold ~0.88
SHINGLE_W = 3
JACCARD_MIN = 0.9
_MOD = (1 << 31) - 1    # Mersenne prime; a*x stays inside uint64 without wrap

_TOKEN = re.compile(r"[a-z0-9]+|[㐀-鿿豈-﫿]")


def tokenize(canon: str) -> list[str]:
    """Latin words/numbers as single tokens, each CJK character as its own token.

    A shared tokenizer for both scripts is what makes one Jaccard threshold work
    across a corpus that is half Chinese: word shingles alone would leave every
    Chinese message as a single token, and character shingles alone would make
    unrelated English messages look similar through shared letters.
    """
    return _TOKEN.findall(canon)


def shingle_set(canon: str) -> frozenset[int]:
    tokens = tokenize(canon)
    if not tokens:
        return frozenset()
    if len(tokens) <= SHINGLE_W:
        grams = [" ".join(tokens)]
    else:
        grams = [" ".join(tokens[i:i + SHINGLE_W])
                 for i in range(len(tokens) - SHINGLE_W + 1)]
    return frozenset(
        int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest(), "big")
        & 0x7FFFFFFF
        for g in grams
    )


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster_near_duplicates(canon_texts: list[str], seed: int = 20260802) -> list[int]:
    """Cluster ids over ``canon_texts`` at Jaccard >= 0.9, one id per input.

    LSH proposes; exact Jaccard on the shingle sets decides. Every member of a
    band bucket is only tested against that bucket's lowest-index member, which
    keeps the work linear in the corpus; chains still merge transitively through
    the union-find across the 8 bands.
    """
    n = len(canon_texts)
    shingles = [shingle_set(t) for t in canon_texts]
    rng = np.random.default_rng(seed)
    coef_a = rng.integers(1, _MOD, size=NUM_PERM, dtype=np.uint64)
    coef_b = rng.integers(0, _MOD, size=NUM_PERM, dtype=np.uint64)

    signatures = np.full((n, NUM_PERM), _MOD, dtype=np.uint64)
    for i, shingle in enumerate(shingles):
        if not shingle:
            continue
        values = np.fromiter(shingle, dtype=np.uint64, count=len(shingle))
        signatures[i] = ((coef_a[:, None] * values[None, :] + coef_b[:, None]) % _MOD).min(axis=1)

    uf = UnionFind(n)
    n_bands = NUM_PERM // BAND_ROWS
    for band in range(n_bands):
        buckets: dict[bytes, int] = {}
        block = signatures[:, band * BAND_ROWS:(band + 1) * BAND_ROWS]
        for i in range(n):
            if not shingles[i]:
                continue
            key = hashlib.blake2b(block[i].tobytes(), digest_size=8).digest()
            head = buckets.setdefault(key, i)
            if head == i or uf.find(head) == uf.find(i):
                continue
            a, b = shingles[head], shingles[i]
            if len(a & b) / len(a | b) >= JACCARD_MIN:
                uf.union(head, i)
    return [uf.find(i) for i in range(n)]


# ---------------------------------------------------------------------------
# 7. Pseudonymisation
# ---------------------------------------------------------------------------

def load_or_create_salt(internal_dir: Path) -> bytes:
    """HMAC salt at ``<internal_dir>/salt``, mode 600, used as raw file bytes.

    The salt file is shared with the other tools in this pipeline, and it already
    existed (written base64-encoded) when this miner was added. **The key is the
    file's bytes exactly as stored** — no hex/base64 decoding step. That rule is
    the only one that cannot disagree with a sibling tool over an encoding, and a
    pseudonym that disagrees is worse than useless: the same user would appear
    under two ids and a group-wise split would leak across them.
    :func:`salt_fingerprint` is published so a mismatch is detectable.

    A wrong mode raises instead of being fixed in place: if the salt has been
    readable by other accounts then the pseudonyms it produced are already
    reversible by anyone who can enumerate user ids, and quietly tightening the
    bits would hide that it happened.
    """
    internal_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(internal_dir, 0o700)
    path = internal_dir / "salt"
    if not path.exists():
        path.write_bytes(secrets.token_bytes(32))
        os.chmod(path, 0o600)
    mode = path.stat().st_mode & 0o777
    if mode != 0o600:
        raise PermissionError(
            f"{path} has mode {mode:o}, expected 600 — pseudonyms derived from a "
            f"salt other accounts can read are not pseudonyms. Rotate it."
        )
    salt = path.read_bytes()
    if len(salt) < 16:
        raise ValueError(f"{path} holds {len(salt)} bytes of salt; expected >= 16")
    return salt


def _base32(digest: bytes) -> str:
    return base64.b32encode(digest).decode("ascii").rstrip("=").lower()


def salt_fingerprint(salt: bytes) -> str:
    """A keyed, publishable fingerprint of the salt.

    Keyed rather than a plain digest so the value can sit in a public repo: it
    proves two runs used the same salt without publishing anything about it.
    """
    return hmac.new(salt, b"nl2yaml-salt-fingerprint", hashlib.sha256).hexdigest()[:12]


def pseudonym(salt: bytes, user_id: str) -> str:
    """``pid_`` + 26 base32 chars of HMAC-SHA256(salt, user_id).

    The encoding matches ``tools/nl2yaml/schema.hmac_pseudonym`` exactly, and a
    test asserts the two agree for a shared salt — two id formats for the same
    user across the two artifacts of one dataset would break every join and every
    group-wise split built on them. It is spelled out here rather than imported
    because that function reads the salt from a process-wide env-driven path,
    while this one is handed the salt whose mode it just verified.

    Base32 rather than hex, for their reason: a hex digest can contain a run of
    eight digits, which a privacy scan flags as a Telegram-id, and a scanner that
    cries wolf on our own pseudonyms is a scanner that gets switched off.
    """
    if not user_id:
        raise ValueError("user_id must be non-empty to derive a pseudonym")
    digest = hmac.new(salt, user_id.encode("utf-8"), hashlib.sha256).digest()
    return "pid_" + _base32(digest)[:26]


# ---------------------------------------------------------------------------
# 8. Repo-safety gate
# ---------------------------------------------------------------------------

#: Anything bound for the repo must look like a hash, an enum, an id, a ticker
#: or a number. No spaces, no non-ASCII. This is the last line of defence: the
#: leak this dataset has to avoid is one Chinese quote riding along inside a
#: ``conditions[]`` entry, and that is exactly what this rejects.
_REPO_TOKEN = re.compile(r"^[A-Za-z0-9_.:+\-/]*$")
MAX_REPO_STRING = 96


def assert_repo_safe(value, path: str = "$") -> None:
    if isinstance(value, str):
        if len(value) > MAX_REPO_STRING:
            raise ValueError(f"repo-bound string too long at {path}: {len(value)} chars")
        if not _REPO_TOKEN.match(value):
            raise ValueError(
                f"repo-bound string at {path} is not an enum/hash/id: {value[:24]!r}"
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_repo_safe(key, f"{path}.{key}")
            assert_repo_safe(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_repo_safe(item, f"{path}[{index}]")
        return
    if isinstance(value, (int, float, bool)) or value is None:
        return
    raise TypeError(f"unexpected type in repo-bound record at {path}: {type(value)}")


def repo_condition(cond: dict) -> dict:
    """A condition with the verbatim quote removed."""
    return {k: v for k, v in cond.items() if k != "quote"}


# ---------------------------------------------------------------------------
# 9. Per-text analysis (computed once per unique canonical text)
# ---------------------------------------------------------------------------

def analyse_text(first_query: str, excerpt: str) -> dict:
    messages = split_messages(first_query, excerpt)
    text = " --- ".join(messages)
    # Cluster on the request core, i.e. the turns that are not chit-chat. The
    # duplicate shape this corpus actually has is "the same prompt card, plus a
    # couple of trailing remarks", and on a short card two extra tokens drag
    # Jaccard to 0.889 — under the 0.9 bar, so the two rows would land in
    # different split groups and a splitter could put one in train and its twin
    # in test. Stripping chit-chat first makes them identical. Rows that are
    # nothing but chit-chat keep their full text so they do not all collapse into
    # one meaningless cluster.
    core = " --- ".join(m for m in messages if not is_chatter(m)) or text
    conditions, hits, dropped = extract_conditions(text)
    families = {c["family"] for c in conditions}
    fragment, reason, leading = fragment_verdict(messages, len(families))
    shape, shape_base, evidence = spec_shape(text, families)
    return {
        "canon_text": text,
        "cluster_text": core,
        "canon_sha256": sha256_hex(text),
        "n_messages_seen": len(messages),
        "canon_len": len(text),
        "conditions": conditions,
        "family_counts": {f: hits[f] for f in ALL_FAMILIES if hits[f]},
        "families": sorted(families),
        "n_families": len(families),
        "n_rule_families": len(families & RULE_FAMILIES),
        "n_conditions": len(conditions),
        "n_conditions_dropped": dropped,
        "tier": tier_of(families),
        "is_continuation_fragment": fragment,
        "fragment_reason": reason,
        "leading_chatter": leading,
        "spec_shape": shape,
        "spec_shape_base": shape_base,
        "spec_shape_evidence": list(evidence),
    }


def read_rows(csv_path: Path, limit: int | None = None) -> list[dict]:
    csv.field_size_limit(1 << 30)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        rows = []
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"{csv_path} has no data rows")
    return rows


def mine(csv_path: Path, out_dir: Path, internal_dir: Path,
         limit: int | None = None) -> dict:
    """Run the whole pass and write both artifacts. Returns the funnel report."""
    rows = read_rows(csv_path, limit)
    salt = load_or_create_salt(internal_dir)

    # One analysis per unique canonical text. Beyond the ~2x saving, this is what
    # guarantees two verbatim-identical chats get byte-identical fields — the
    # property the 0.639 self-agreement audit found the upstream labels lacked.
    analyses: dict[str, dict] = {}
    per_row: list[dict] = []
    for row in rows:
        cache_key = f"{row['first_query']}\x1f{row['user_text_excerpt']}"
        analysis = analyses.get(cache_key)
        if analysis is None:
            analysis = analyse_text(row["first_query"], row["user_text_excerpt"])
            analyses[cache_key] = analysis
        per_row.append(analysis)

    # Cluster over unique cluster texts only; rows sharing one share a cluster by
    # construction, and the LSH pass never sees the same string twice.
    unique_texts: dict[str, int] = {}
    for analysis in per_row:
        unique_texts.setdefault(analysis["cluster_text"], len(unique_texts))
    texts = [""] * len(unique_texts)
    for text, index in unique_texts.items():
        texts[index] = text
    cluster_root = cluster_near_duplicates(texts)

    # Name each cluster after the smallest hash it contains, so the id is a
    # function of the cluster's content and not of row order.
    root_members: dict[int, list[str]] = defaultdict(list)
    for text, index in unique_texts.items():
        root_members[cluster_root[index]].append(sha256_hex(text))
    cluster_id = {root: short_hash(min(members), "dup_")
                  for root, members in root_members.items()}

    canon_row_counts: Counter = Counter(a["canon_sha256"] for a in per_row)
    cluster_row_counts: Counter = Counter(
        cluster_id[cluster_root[unique_texts[a["cluster_text"]]]] for a in per_row)

    records: list[dict] = []
    internal: list[dict] = []
    for row, analysis in zip(rows, per_row):
        dup_cluster_id = cluster_id[cluster_root[unique_texts[analysis["cluster_text"]]]]
        preset = (row["preset_case"] or "").strip()
        raw_first = row["first_query"] or ""
        raw_excerpt = row["user_text_excerpt"] or ""
        record = {
            "row_id": short_hash(f"{row['chat_id']}|{row['day']}", "r_"),
            "pseudonym_id": pseudonym(salt, row["user_id"]),
            "chat_ref": short_hash(row["chat_id"], "c_"),
            "month": row["month"],
            "day": row["day"],
            "lang": row["lang"],
            "zh_variant": row["zh_variant"],
            "text_sha256": sha256_hex(f"{raw_first}\x1f{raw_excerpt}"),
            "canon_sha256": analysis["canon_sha256"],
            "canon_len": analysis["canon_len"],
            "n_user_msgs": int(row["n_user_msgs"] or 0),
            "n_assistant_msgs": int(row["n_assistant_msgs"] or 0),
            "n_messages_seen": analysis["n_messages_seen"],
            "preset_case": preset,
            "dup_cluster_id": dup_cluster_id,
            "dup_count": cluster_row_counts[dup_cluster_id],
            "canon_dup_count": canon_row_counts[analysis["canon_sha256"]],
            # Prefixed rather than a bare coalesce: a preset-card slug and a
            # cluster id must not be able to collide into one split group.
            "split_group_key": f"preset:{preset}" if preset else f"dup:{dup_cluster_id}",
            "is_continuation_fragment": analysis["is_continuation_fragment"],
            "fragment_reason": analysis["fragment_reason"],
            "leading_chatter": analysis["leading_chatter"],
            "families": analysis["families"],
            "family_counts": analysis["family_counts"],
            "n_families": analysis["n_families"],
            "n_rule_families": analysis["n_rule_families"],
            "n_conditions": analysis["n_conditions"],
            "n_conditions_dropped": analysis["n_conditions_dropped"],
            "tier": analysis["tier"],
            "conditions": [repo_condition(c) for c in analysis["conditions"]],
            "spec_shape": analysis["spec_shape"],
            "spec_shape_base": analysis["spec_shape_base"],
            "spec_shape_evidence": analysis["spec_shape_evidence"],
            # Recall filter and provenance only. Audited at 0.530 inter-annotator
            # agreement over 13 classes; never a label, never a strata.
            "mining_source": {
                "primary_intent": row["primary_intent"],
                "secondary_intents": [s for s in (row["secondary_intents"] or "").split("|") if s],
                "label_source": row["label_source"],
                "is_coin_selection": row["is_coin_selection"],
                "selection_basis": [s for s in (row["selection_basis"] or "").split("|") if s],
                "wants_automation": row["wants_automation"],
                "wants_backtest": row["wants_backtest"],
                "wants_strategy": row["wants_strategy"],
                "kw_primary": row["kw_primary"],
                "labels_are_not_gold": True,
            },
            "tier_scheme": TIER_SCHEME,
            "mined_by": MINER_VERSION,
        }
        record["is_candidate"] = (
            not record["is_continuation_fragment"] and record["tier"] != "D")
        assert_repo_safe(record)
        records.append(record)
        internal.append({
            "row_id": record["row_id"],
            "user_id": row["user_id"],
            "chat_id": row["chat_id"],
            "pseudonym_id": record["pseudonym_id"],
            "day": row["day"],
            "lang": row["lang"],
            "tier": record["tier"],
            "spec_shape": record["spec_shape"],
            "split_group_key": record["split_group_key"],
            "is_candidate": record["is_candidate"],
            "fragment_reason": record["fragment_reason"],
            "first_query": raw_first,
            "user_text_excerpt": raw_excerpt,
            "canon_text": analysis["canon_text"],
            "conditions": analysis["conditions"],
        })

    report = build_report(records)
    report["salt_fingerprint"] = salt_fingerprint(salt)
    write_outputs(records, internal, report, out_dir, internal_dir)
    return report


def write_outputs(records: list[dict], internal: list[dict], report: dict,
                  out_dir: Path, internal_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = out_dir / "candidates.jsonl"
    with candidates.open("w", encoding="ascii") as handle:
        for record in records:
            if not record["is_candidate"]:
                continue
            # ensure_ascii is not cosmetic here: with an ascii-only stream, any
            # user text that slipped past assert_repo_safe would raise on write
            # instead of landing in a public repo.
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")

    payload = json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2)
    if not payload.isascii():
        raise ValueError("funnel report contains non-ascii content")
    (out_dir / "funnel.json").write_text(payload + "\n", encoding="ascii")

    internal_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(internal_dir, 0o700)
    cases = internal_dir / "cases_internal.jsonl"
    with cases.open("w", encoding="utf-8") as handle:
        for row in internal:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.chmod(cases, 0o600)


# ---------------------------------------------------------------------------
# 10. Funnel report
# ---------------------------------------------------------------------------

def build_report(records: list[dict]) -> dict:
    total = len(records)
    fragments = [r for r in records if r["is_continuation_fragment"]]
    kept = [r for r in records if not r["is_continuation_fragment"]]
    candidates = [r for r in records if r["is_candidate"]]

    def unique(rows: list[dict]) -> int:
        return len({r["canon_sha256"] for r in rows})

    def groups(rows: list[dict]) -> int:
        return len({r["split_group_key"] for r in rows})

    tier_rows = Counter(r["tier"] for r in kept)
    tier_unique = {t: unique([r for r in kept if r["tier"] == t]) for t in "ABCD"}
    tier_groups = {t: groups([r for r in kept if r["tier"] == t]) for t in "ABCD"}

    crosstab: dict[str, dict[str, int]] = {}
    crosstab_unique: dict[str, dict[str, int]] = {}
    for shape in ("selection", "trade", "both", "unclear"):
        subset = [r for r in kept if r["spec_shape"] == shape]
        crosstab[shape] = {t: sum(1 for r in subset if r["tier"] == t) for t in "ABCD"}
        crosstab_unique[shape] = {
            t: unique([r for r in subset if r["tier"] == t]) for t in "ABCD"}

    group_sizes = Counter(r["split_group_key"] for r in records)
    dup_counts = Counter(r["dup_count"] for r in records)

    def bucket(value: int) -> str:
        for edge in (1, 2, 5, 10, 50, 100, 500):
            if value <= edge:
                return f"<={edge}"
        return ">500"

    return {
        "miner": MINER_VERSION,
        "funnel": {
            "total_rows": total,
            "continuation_fragments": len(fragments),
            "after_fragment_filter": len(kept),
            "unique_canon_after_fragment_filter": unique(kept),
            "candidates_rows": len(candidates),
            "candidates_unique_canon": unique(candidates),
            "candidates_split_groups": groups(candidates),
        },
        "fragment_reasons": dict(Counter(r["fragment_reason"] for r in fragments)),
        "leading_chatter_but_kept": sum(1 for r in kept if r["leading_chatter"]),
        "tier_rows": {t: tier_rows[t] for t in "ABCD"},
        "tier_unique_canon": tier_unique,
        "tier_split_groups": tier_groups,
        "shape_by_tier_rows": crosstab,
        "shape_by_tier_unique_canon": crosstab_unique,
        "shape_rows": dict(Counter(r["spec_shape"] for r in kept)),
        "shape_base_rows": dict(Counter(r["spec_shape_base"] for r in kept)),
        "family_rows": {
            f: sum(1 for r in records if f in r["families"]) for f in ALL_FAMILIES},
        "split_groups": {
            "n_groups": len(group_sizes),
            "max_group_rows": max(group_sizes.values()) if group_sizes else 0,
            "groups_from_preset_case": len(
                {r["split_group_key"] for r in records if r["preset_case"]}),
            "rows_in_preset_groups": sum(1 for r in records if r["preset_case"]),
            "size_histogram": dict(sorted(
                Counter(bucket(v) for v in group_sizes.values()).items())),
            "largest": [{"split_group_key": k, "rows": v}
                        for k, v in group_sizes.most_common(20)],
        },
        "dup_count_histogram": dict(sorted(
            Counter(bucket(v) for v in dup_counts.elements()).items())),
        "top_preset_case": [
            {"preset_case": k, "rows": v} for k, v in
            Counter(r["preset_case"] for r in records if r["preset_case"]).most_common(20)],
        "n_conditions_histogram": dict(sorted(
            Counter(min(r["n_conditions"], 10) for r in records).items(),
            key=lambda kv: kv[0])),
    }


def print_report(report: dict) -> None:
    funnel = report["funnel"]
    write = sys.stdout.write
    write("\n=== Gate0 funnel ===\n")
    write(f"total rows                              {funnel['total_rows']:>7}\n")
    write(f"- continuation fragments                {funnel['continuation_fragments']:>7}"
          f"   {report['fragment_reasons']}\n")
    write(f"= after fragment filter                 {funnel['after_fragment_filter']:>7}\n")
    write(f"  unique canon texts                    "
          f"{funnel['unique_canon_after_fragment_filter']:>7}\n")
    write(f"= candidates (tier A/B/C)               {funnel['candidates_rows']:>7}"
          f"   unique {funnel['candidates_unique_canon']}"
          f"   split groups {funnel['candidates_split_groups']}\n")
    write(f"  kept despite leading chatter          {report['leading_chatter_but_kept']:>7}\n")

    write("\n=== tier (after fragment filter) ===\n")
    write("tier     rows   unique   groups\n")
    for tier in "ABCD":
        write(f"{tier:<5}{report['tier_rows'][tier]:>8}"
              f"{report['tier_unique_canon'][tier]:>9}"
              f"{report['tier_split_groups'][tier]:>9}\n")

    write("\n=== spec shape x tier (rows / unique canon) ===\n")
    write(f"{'shape':<12}{'A':>14}{'B':>14}{'C':>14}{'D':>14}\n")
    for shape in ("selection", "trade", "both", "unclear"):
        cells = "".join(
            f"{report['shape_by_tier_rows'][shape][t]:>8}"
            f"/{report['shape_by_tier_unique_canon'][shape][t]:<5}" for t in "ABCD")
        write(f"{shape:<12}{cells}\n")
    write(f"classify_request base kinds: {report['shape_base_rows']}\n")

    write("\n=== condition families (rows with >=1 hit, all rows) ===\n")
    for family, count in sorted(report["family_rows"].items(), key=lambda kv: -kv[1]):
        write(f"  {family:<18}{count:>7}\n")
    write(f"n_conditions histogram (10 = 10+): {report['n_conditions_histogram']}\n")

    groups = report["split_groups"]
    write("\n=== split_group_key ===\n")
    write(f"groups                {groups['n_groups']:>7}\n")
    write(f"max group (rows)      {groups['max_group_rows']:>7}\n")
    write(f"groups from preset    {groups['groups_from_preset_case']:>7}"
          f"   rows {groups['rows_in_preset_groups']}\n")
    write(f"group size histogram  {groups['size_histogram']}\n")
    write(f"dup_count histogram   {report['dup_count_histogram']}\n")
    write("largest 20 groups:\n")
    for item in groups["largest"]:
        write(f"  {item['rows']:>6}  {item['split_group_key']}\n")

    write("\n=== top preset_case (template layer, not model targets) ===\n")
    for item in report["top_preset_case"]:
        write(f"  {item['rows']:>6}  {item['preset_case']}\n")
    write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="repo-bound output dir (hashes/enums/counts only)")
    # Same env var and same default as ``schema.internal_root``. If this tool
    # wrote its salt somewhere else, the pseudonyms in candidates.jsonl and the
    # ones the schema module derives would silently disagree, and the two halves
    # of the dataset could no longer be joined or split together.
    parser.add_argument("--internal-dir", type=Path,
                        default=Path(os.environ.get("NL2YAML_INTERNAL_ROOT")
                                     or Path.home() / "nl2yaml_internal").expanduser(),
                        help="outside-the-repo dir for salt and verbatim text "
                             "(default: $NL2YAML_INTERNAL_ROOT or ~/nl2yaml_internal)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.csv.exists():
        raise FileNotFoundError(f"input csv not found: {args.csv}")
    report = mine(args.csv, args.out, args.internal_dir, args.limit)
    print_report(report)
    sys.stdout.write(f"wrote {args.out / 'candidates.jsonl'}"
                     f" and {args.out / 'funnel.json'}\n")
    sys.stdout.write(f"verbatim text stayed in {args.internal_dir / 'cases_internal.jsonl'}"
                     " (mode 600)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
