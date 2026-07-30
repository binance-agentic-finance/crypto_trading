"""News feature extraction — ``EventFrame`` in, enriched events + metrics out.

Why this exists
---------------
Three advisory bots each wrote their own keyword matching against raw news
rows. That is the wrong layer: a strategy should be able to ask "what is the
news state of BTCUSDT right now" the same way it asks for funding, without
re-implementing entity matching. So the pipeline is:

    EventFrame (raw)
        → dedupe_events        first_seen_at / duplicate_of / lead_time
        → resolve_instruments  which coin, with disambiguation and refusal
        → classify_events      event_type + expected half-life
        → score_sentiment      sentiment_score / confidence / who judged it
        → enrich_events        all of the above, in order
        → aggregate_to_metrics MetricFrame, readable by any strategy

This is a **deliberately simple first version**: the sentiment model is a
lexicon, the classifier is keyword rules, dedup is exact-title plus a shingle
overlap. Each one is replaceable behind the same function signature. What is
*not* simplified is the honesty: every derived field records who derived it and
how confident it is, and anything the rules cannot resolve stays empty rather
than being guessed.

Two things worth knowing before using it
----------------------------------------
1. **Square's ``tendency`` and ``tickers[]`` are publisher-supplied.** They are
   the author's own tags, not an independent read — a KOL tagging ten coins to
   ride a trend produces ten tickers. They are kept as a *fallback* and always
   labelled ``sentiment_source="publisher"``, never silently promoted.

2. **News is FORWARD_ONLY: there is no PIT history.** Nothing here is
   backtestable today. That is precisely why ``first_seen_at`` matters now — if
   we do not start recording when a story first appeared, in six months a news
   backtest cannot even ask the question.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "EVENT_TYPES",
    "EVENT_HALF_LIFE_SECONDS",
    "BULLISH_TERMS",
    "BEARISH_TERMS",
    "dedupe_events",
    "resolve_instruments",
    "classify_events",
    "score_sentiment",
    "enrich_events",
    "aggregate_to_metrics",
]


# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------

#: event_type -> keywords. Ordered: the first match wins, so put the specific
#: categories before the general ones.
EVENT_TYPES: Dict[str, Tuple[str, ...]] = {
    "security_incident": ("hack", "exploit", "breach", "stolen", "drained",
                          "rug pull", "被盗", "攻击", "漏洞"),
    "delisting": ("delist", "will remove", "removal of", "下架", "退市"),
    "listing": ("will list", "listing", "lists ", "new listing", "launchpool",
                "上币", "上线交易"),
    "token_unlock": ("unlock", "vesting", "cliff", "解锁"),
    "regulation": ("sec ", "cftc", "lawsuit", "regulator", "ban", "sanction",
                   "监管", "诉讼", "禁令"),
    "mainnet_upgrade": ("mainnet", "hard fork", "upgrade", "halving",
                        "主网", "升级", "减半"),
    "governance": ("proposal", "governance", "vote", "dao ", "治理", "投票"),
    "partnership": ("partnership", "integrat", "collaborat", "合作"),
    "funding_round": ("raises", "funding round", "series a", "series b",
                      "investment", "融资"),
    "macro": ("cpi", "fomc", "fed ", "interest rate", "nonfarm", "inflation",
              "gdp", "非农", "通胀", "加息", "降息"),
    "etf_flow": ("etf", "spot etf", "inflow", "outflow"),
    "maintenance": ("maintenance", "suspend", "pause", "halt", "维护", "暂停"),
}

#: How long an event of each type plausibly still matters. Drives ``valid_until``
#: — a listing pulse is minutes, an unlock overhang is days. Compressing all of
#: them into one horizon is the main thing "bullish/bearish" throws away.
EVENT_HALF_LIFE_SECONDS: Dict[str, int] = {
    "listing": 1_800,
    "delisting": 14_400,
    "security_incident": 7_200,
    "maintenance": 3_600,
    "macro": 14_400,
    "etf_flow": 86_400,
    "regulation": 86_400,
    "mainnet_upgrade": 172_800,
    "token_unlock": 259_200,
    "governance": 259_200,
    "partnership": 43_200,
    "funding_round": 43_200,
    "unclassified": 3_600,
}

#: Direction each type implies **for the asset**, when the rules are confident.
#: ``0`` means the type carries no inherent direction.
EVENT_TYPE_BIAS: Dict[str, float] = {
    "listing": 0.6,
    "delisting": -0.7,
    "security_incident": -0.8,
    "maintenance": 0.0,
    "token_unlock": -0.4,
    "mainnet_upgrade": 0.3,
    "governance": 0.0,
    "partnership": 0.3,
    "funding_round": 0.3,
    "regulation": -0.3,
    "macro": 0.0,
    "etf_flow": 0.0,
    "unclassified": 0.0,
}

BULLISH_TERMS: Tuple[str, ...] = (
    "surge", "rally", "soar", "jump", "bullish", "record high", "all-time high",
    "approve", "approval", "adopt", "breakthrough", "gain", "upgrade",
    "上涨", "大涨", "突破", "利好", "获批", "新高",
)

BEARISH_TERMS: Tuple[str, ...] = (
    "plunge", "crash", "slump", "tumble", "bearish", "sell-off", "selloff",
    "reject", "denied", "halt", "liquidat", "exploit", "hack", "decline",
    "下跌", "暴跌", "跌破", "利空", "抛售", "清算", "被拒",
)

#: publisher tendency code -> score. Square: 0 neutral / 1 bullish / 2 bearish.
_TENDENCY_SCORE = {0: 0.0, 1: 0.5, 2: -0.5}

_QUOTES = ("USDT", "USDC", "FDUSD", "TUSD", "BUSD", "USD")
_WORD = re.compile(r"[a-z0-9]+")


def _text_of(row: pd.Series) -> str:
    parts = [row.get("title", ""), row.get("summary", ""), row.get("body", "")]
    return " ".join(str(part) for part in parts if part).lower()


def _contains(text: str, terms: Iterable[str]) -> List[str]:
    """Word-boundary match for ASCII terms, substring for CJK."""
    found = []
    for term in terms:
        value = str(term).lower().strip()
        if not value:
            continue
        if value.isascii():
            if re.search(r"(?<![a-z0-9])%s" % re.escape(value), text):
                found.append(value)
        elif value in text:
            found.append(value)
    return found


def _base_token(symbol: str) -> str:
    value = str(symbol).upper()
    for quote in _QUOTES:
        if value.endswith(quote) and len(value) > len(quote):
            return value[: -len(quote)]
    return value


# ---------------------------------------------------------------------------
# 1. dedupe + freshness
# ---------------------------------------------------------------------------


def _shingles(text: str, size: int = 4) -> frozenset:
    words = _WORD.findall(text)
    if len(words) < size:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(
        " ".join(words[i:i + size]) for i in range(len(words) - size + 1)
    )


def dedupe_events(
    frame: pd.DataFrame,
    *,
    similarity: float = 0.6,
    known: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Collapse reposts and record how late we are to each story.

    The same story arrives from ten accounts. Trading the eighth repost is
    trading a stale signal, so the *copy's* timestamp is not the useful one —
    ``first_seen_at`` is.

    Adds:

    ``content_hash``       exact-content key
    ``first_seen_at``      earliest ``event_time`` across the duplicate cluster
    ``duplicate_of``       ``event_id`` of the first copy (empty for the first)
    ``is_duplicate``       convenience flag
    ``repost_count``       size of the cluster (a crude attention measure)
    ``lead_time_seconds``  ``available_time - first_seen_at`` — how late we are

    ``known`` is an optional frame of previously-seen events (the running
    corpus); passing it lets a live poll recognise a story it saw an hour ago
    instead of treating every poll as a fresh world.
    """
    if frame is None or frame.empty:
        return frame

    out = frame.copy().reset_index(drop=True)
    out["content_hash"] = [
        hashlib.sha1(_text_of(row).encode("utf-8")).hexdigest()[:16]
        for _, row in out.iterrows()
    ]
    out["_shingles"] = [_shingles(_text_of(row)) for _, row in out.iterrows()]

    # order by when the story happened so the first copy really is first
    order = pd.to_datetime(out.get("event_time"), utc=True, errors="coerce")
    out["_order"] = order
    out = out.sort_values("_order", kind="stable").reset_index(drop=True)

    #: cluster id per row
    cluster: List[int] = [-1] * len(out)
    reps: List[Tuple[int, frozenset, str]] = []   # (cluster, shingles, hash)

    prior: Dict[str, Any] = {}
    if known is not None and not known.empty and "content_hash" in known:
        for _, row in known.iterrows():
            prior.setdefault(row["content_hash"], row.get("event_id", ""))

    for index, row in out.iterrows():
        assigned = None
        for cluster_id, shingle_set, digest in reps:
            if digest == row["content_hash"]:
                assigned = cluster_id
                break
            union = len(shingle_set | row["_shingles"])
            if union and len(shingle_set & row["_shingles"]) / union >= similarity:
                assigned = cluster_id
                break
        if assigned is None:
            assigned = len(reps)
            reps.append((assigned, row["_shingles"], row["content_hash"]))
        cluster[index] = assigned

    out["_cluster"] = cluster
    grouped = out.groupby("_cluster")
    out["first_seen_at"] = grouped["_order"].transform("min")
    out["repost_count"] = grouped["event_id"].transform("size").astype(int)
    first_ids = grouped["event_id"].transform("first")
    out["duplicate_of"] = np.where(out["event_id"] != first_ids, first_ids, "")
    # a story already in the running corpus is a repost even if this batch is its first copy
    if prior:
        seen_before = out["content_hash"].map(prior).fillna("")
        out["duplicate_of"] = np.where(
            (out["duplicate_of"] == "") & (seen_before != ""),
            seen_before, out["duplicate_of"],
        )
    out["is_duplicate"] = out["duplicate_of"].astype(bool)

    available = pd.to_datetime(out.get("available_time"), utc=True, errors="coerce")
    out["lead_time_seconds"] = (
        (available - out["first_seen_at"]).dt.total_seconds().fillna(0.0).clip(lower=0.0)
    )

    return out.drop(columns=["_shingles", "_order", "_cluster"])


# ---------------------------------------------------------------------------
# 2. entity resolution
# ---------------------------------------------------------------------------


def resolve_instruments(
    frame: pd.DataFrame,
    *,
    universe: Sequence[str],
    require_mention: bool = True,
    max_tickers: int = 3,
) -> pd.DataFrame:
    """Decide which instrument each event is about — or refuse to.

    Matching alone is not enough. The publisher's ``tickers[]`` is a self-applied
    tag, so a post riding a trend arrives labelled with ten coins. Rules:

    * only tickers whose base token is in ``universe`` are considered;
    * with ``require_mention``, a tagged ticker must also appear in the text —
      a tag nobody wrote about is a tag, not a subject;
    * an event tagged with more than ``max_tickers`` names is treated as
      untargeted (``instrument_id=None``, ``entity_source="ambiguous"``);
    * nothing resolvable → ``instrument_id=None``. **Attaching news to the wrong
      coin is worse than attaching it to none.**

    Adds ``instrument_id`` (may be null), ``entity_source``
    (``text`` | ``publisher`` | ``ambiguous`` | ``none``) and
    ``entity_confidence``.
    """
    if frame is None or frame.empty:
        return frame

    allowed = {_base_token(symbol): str(symbol).upper() for symbol in universe}
    out = frame.copy().reset_index(drop=True)

    resolved: List[Optional[str]] = []
    sources: List[str] = []
    confidence: List[float] = []

    for _, row in out.iterrows():
        text = _text_of(row)
        tagged = []
        for key in ("tickers", "user_input_tickers"):
            value = row.get(key)
            if isinstance(value, (list, tuple)):
                tagged.extend(str(item).upper().lstrip("$") for item in value)
        existing = row.get("instrument_id")
        if isinstance(existing, str) and existing:
            tagged.append(_base_token(existing))
        tagged = [token for token in dict.fromkeys(tagged) if token in allowed]

        if len(tagged) > max_tickers:
            resolved.append(None)
            sources.append("ambiguous")
            confidence.append(0.0)
            continue

        in_text = [token for token in tagged if _contains(text, [token.lower()])]
        if in_text:
            resolved.append(allowed[in_text[0]])
            sources.append("text")
            confidence.append(0.9 if len(in_text) == 1 else 0.6)
        elif tagged and not require_mention:
            resolved.append(allowed[tagged[0]])
            sources.append("publisher")
            confidence.append(0.4)
        else:
            resolved.append(None)
            sources.append("none" if not tagged else "publisher_only")
            confidence.append(0.0)

    out["instrument_id"] = resolved
    out["entity_source"] = sources
    out["entity_confidence"] = confidence
    return out


# ---------------------------------------------------------------------------
# 3. classification
# ---------------------------------------------------------------------------


def classify_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign ``event_type`` and its expected half-life.

    Bullish/bearish is too coarse to act on: a listing is a minutes-long pulse
    and an unlock is a multi-day overhang, and the two want completely
    different holding periods.
    """
    if frame is None or frame.empty:
        return frame

    out = frame.copy().reset_index(drop=True)
    types: List[str] = []
    matched: List[List[str]] = []
    for _, row in out.iterrows():
        text = _text_of(row)
        chosen = "unclassified"
        hits: List[str] = []
        for event_type, terms in EVENT_TYPES.items():
            found = _contains(text, terms)
            if found:
                chosen, hits = event_type, found
                break
        types.append(chosen)
        matched.append(hits)

    out["event_type"] = types
    out["matched_terms"] = matched
    out["expected_half_life_seconds"] = [
        EVENT_HALF_LIFE_SECONDS.get(item, EVENT_HALF_LIFE_SECONDS["unclassified"])
        for item in types
    ]
    return out


# ---------------------------------------------------------------------------
# 4. sentiment
# ---------------------------------------------------------------------------


def score_sentiment(
    frame: pd.DataFrame,
    *,
    min_confidence: float = 0.35,
    use_publisher_fallback: bool = True,
) -> pd.DataFrame:
    """Score direction in [-1, 1] with an explicit confidence and provenance.

    A lexicon count plus the event type's inherent bias. Crude on purpose — the
    point of the first version is the *shape*: three fields instead of one
    string, so a consumer can require confidence before acting on direction.

    ``sentiment_source`` is one of ``lexicon`` / ``event_type`` /
    ``publisher`` / ``none``. Square's ``tendency`` is only ever used as the
    fallback and is always labelled as such — it is the author's own tag, and
    promoting it silently would launder a self-assessment into a measurement.

    ``sentiment_disagrees_with_publisher`` flags the rows where our read and the
    author's tag point opposite ways: often a headline written to sound like
    the opposite of what it says.
    """
    if frame is None or frame.empty:
        return frame

    out = frame.copy().reset_index(drop=True)
    scores: List[float] = []
    confidences: List[float] = []
    sources: List[str] = []

    for _, row in out.iterrows():
        text = _text_of(row)
        bull = len(_contains(text, BULLISH_TERMS))
        bear = len(_contains(text, BEARISH_TERMS))
        type_bias = EVENT_TYPE_BIAS.get(str(row.get("event_type", "")), 0.0)

        if bull or bear:
            lexicon = (bull - bear) / float(bull + bear)
            score = float(np.clip(0.7 * lexicon + 0.3 * type_bias, -1.0, 1.0))
            # more matched terms -> more confident, saturating
            confidence = float(np.clip(0.35 + 0.12 * (bull + bear), 0.0, 0.85))
            source = "lexicon"
        elif type_bias:
            score, confidence, source = type_bias, 0.5, "event_type"
        else:
            score, confidence, source = 0.0, 0.0, "none"

        if source == "none" and use_publisher_fallback:
            tendency = row.get("tendency")
            if tendency in _TENDENCY_SCORE:
                score = _TENDENCY_SCORE[tendency]
                # deliberately low: this is the author grading their own post
                confidence = 0.3
                source = "publisher"

        scores.append(round(score, 4))
        confidences.append(round(confidence, 4))
        sources.append(source)

    out["sentiment_score"] = scores
    out["sentiment_confidence"] = confidences
    out["sentiment_source"] = sources
    # below the bar we keep the number but refuse the label — a consumer that
    # wants a direction must check the confidence.
    out["sentiment_label"] = [
        "neutral" if confidence < min_confidence or score == 0
        else ("bullish" if score > 0 else "bearish")
        for score, confidence in zip(scores, confidences)
    ]

    if "tendency" in out.columns:
        publisher = out["tendency"].map(_TENDENCY_SCORE)
        out["sentiment_disagrees_with_publisher"] = (
            publisher.notna()
            & (np.sign(publisher.fillna(0)) != 0)
            & (np.sign(out["sentiment_score"]) != 0)
            & (np.sign(publisher.fillna(0)) != np.sign(out["sentiment_score"]))
        )
    else:
        out["sentiment_disagrees_with_publisher"] = False
    return out


# ---------------------------------------------------------------------------
# 5. the pipeline
# ---------------------------------------------------------------------------


def enrich_events(
    frame: pd.DataFrame,
    *,
    universe: Sequence[str],
    known: Optional[pd.DataFrame] = None,
    similarity: float = 0.6,
    require_mention: bool = True,
    min_confidence: float = 0.35,
) -> pd.DataFrame:
    """Run the whole pipeline in the order the stages depend on each other.

    Classification runs before sentiment because the event type contributes to
    the score; dedup runs first so a repost cluster is scored once.
    """
    if frame is None or frame.empty:
        return frame
    out = dedupe_events(frame, similarity=similarity, known=known)
    out = resolve_instruments(out, universe=universe, require_mention=require_mention)
    out = classify_events(out)
    out = score_sentiment(out, min_confidence=min_confidence)
    return out


# ---------------------------------------------------------------------------
# 6. events -> metrics
# ---------------------------------------------------------------------------


def aggregate_to_metrics(
    frame: pd.DataFrame,
    *,
    as_of: Optional[Any] = None,
    windows: Sequence[str] = ("1h", "4h", "24h"),
    exclude_duplicates: bool = True,
    venue: str = "binance",
    product: str = "mixed",
) -> pd.DataFrame:
    """Turn an enriched ``EventFrame`` into ``MetricFrame`` rows.

    This is the bridge that makes news usable: after this, a strategy reads
    ``news_count_1h`` the same way it reads ``funding_rate_8h``, and no bot has
    to write keyword matching again.

    Metrics per instrument per window::

        news_count_<w>                 events seen
        news_bull_ratio_<w>            share of confident bullish events
        news_sentiment_mean_<w>        confidence-weighted mean score
        news_max_reliability_<w>       best source_reliability seen
        news_repost_max_<w>            largest repost cluster (attention proxy)
        news_min_lead_time_<w>         freshest lead time, seconds

    Plus one window-independent metric::

        news_seconds_since_last        time since the most recent event
    """
    columns = ["event_time", "available_time", "instrument_id", "metric", "value",
               "venue", "product", "unit", "window", "source_id", "quality"]
    if frame is None or frame.empty or "instrument_id" not in frame:
        return pd.DataFrame(columns=columns)

    events = frame[frame["instrument_id"].notna()].copy()
    if exclude_duplicates and "is_duplicate" in events:
        events = events[~events["is_duplicate"].astype(bool)]
    if events.empty:
        return pd.DataFrame(columns=columns)

    events["_available"] = pd.to_datetime(
        events["available_time"], utc=True, errors="coerce"
    )
    cutoff = (
        pd.to_datetime(as_of, utc=True) if as_of is not None
        else events["_available"].max()
    )
    events = events[events["_available"] <= cutoff]
    if events.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, Any]] = []

    def _emit(instrument: str, metric: str, value: Any, window: str, unit: str) -> None:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return
        rows.append({
            "event_time": cutoff, "available_time": cutoff,
            "instrument_id": instrument, "metric": metric, "value": float(value),
            "venue": venue, "product": product, "unit": unit, "window": window,
            "source_id": "blocks.news_features", "quality": "derived",
        })

    for instrument, group in events.groupby("instrument_id"):
        for window in windows:
            delta = pd.Timedelta(window)
            recent = group[group["_available"] > cutoff - delta]
            _emit(instrument, "news_count_%s" % window, len(recent), window, "count")
            if recent.empty:
                continue

            confident = recent[
                recent.get("sentiment_confidence", pd.Series(0, index=recent.index)) >= 0.35
            ]
            if not confident.empty:
                bullish = (confident["sentiment_score"] > 0).sum()
                _emit(instrument, "news_bull_ratio_%s" % window,
                      bullish / float(len(confident)), window, "ratio")
                weights = confident["sentiment_confidence"].astype(float)
                if weights.sum() > 0:
                    _emit(instrument, "news_sentiment_mean_%s" % window,
                          float((confident["sentiment_score"] * weights).sum() / weights.sum()),
                          window, "score")
            if "source_reliability" in recent:
                best = pd.to_numeric(recent["source_reliability"], errors="coerce").max()
                _emit(instrument, "news_max_reliability_%s" % window, best, window, "ratio")
            if "repost_count" in recent:
                _emit(instrument, "news_repost_max_%s" % window,
                      pd.to_numeric(recent["repost_count"], errors="coerce").max(),
                      window, "count")
            if "lead_time_seconds" in recent:
                _emit(instrument, "news_min_lead_time_%s" % window,
                      pd.to_numeric(recent["lead_time_seconds"], errors="coerce").min(),
                      window, "seconds")

        last = group["_available"].max()
        _emit(instrument, "news_seconds_since_last",
              (cutoff - last).total_seconds(), "point", "seconds")

    return pd.DataFrame(rows, columns=columns)
