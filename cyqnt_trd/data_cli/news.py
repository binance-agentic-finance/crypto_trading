"""PUBLIC Binance Square news / social data → typed, cached pandas DataFrames.

Layer 1 of the news integration (see also ``cyqnt_trd/blocks/news_feed.py`` for
the lookahead-safe feature layer). Shape mirrors ``data_cli/kline.py``: each
``fetch_*`` checks the in-memory TTL cache, calls the vendored PUBLIC client,
and returns a typed DataFrame.

Envelope policy (matches the acceptance contract)
-------------------------------------------------
The upstream envelope is ``{"code": "000000", "data": ...}``. A response with
``code == '000000'`` **and** ``data is None`` is a legitimate *cache miss*, not
an error — in that case (and for any transport error) these functions return an
**empty typed DataFrame** rather than raising. Empty frames are never written
to the cache (mirrors the ``_cache`` policy), so a later successful call still
populates it.

These live fetchers are for real-time / selection use. For backtests you want
the point-in-time, lookahead-safe features in ``blocks.news_feed``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._cache import cache_get, cache_set, TTL_NEWS
from ._vendor.binance_bigdata_client import get_public_client, is_ok_envelope

__all__ = [
    "fetch_news",
    "fetch_search",
    "fetch_sentiment",
    "fetch_ticker_rank",
    "fetch_topic_trending",
    "fetch_hot_post",
    "NEWS_COLUMNS",
    "POST_COLUMNS",
    "SENTIMENT_COLUMNS",
    "TICKER_RANK_COLUMNS",
    "TOPIC_TRENDING_COLUMNS",
]

# ---------------------------------------------------------------------------
# Typed column schemas (used for both parsed and empty cache-miss frames)
# ---------------------------------------------------------------------------
# News + posts share the Square "content item" shape.
_POST_FIELDS = [
    ("id", str),
    ("title", str),
    ("summary", str),
    ("body", str),
    ("web_link", str),
    ("date", "int"),                 # content time (ms) — do NOT use for gating
    ("latest_release_time", "int"),  # content time (ms) — do NOT use for gating
    ("content_type", "int"),
    ("tendency", "int"),             # 0/1/2 enum (speculative, unofficial)
    ("detected_lang", str),
    ("is_created_by_ai", bool),
    ("author_name", str),
    ("author_role", str),
    ("like_count", "int"),
    ("comment_count", "int"),
    ("share_count", "int"),
    ("view_count", "int"),
    ("quote_count", "int"),
    ("bookmark_count", "int"),
    ("rank", "int"),
    ("score", "float"),
    ("tickers", object),             # list[str]
    ("user_input_tickers", object),  # list[str]
    ("hashtag_list", object),        # list[str]
    ("generated_at", "int"),         # API content-generation time (ms)
]
POST_COLUMNS = [name for name, _ in _POST_FIELDS]
NEWS_COLUMNS = POST_COLUMNS  # getNews and getHotPost/getSearch/getFeed share it

SENTIMENT_COLUMNS = [
    "token", "bullish_value", "bearish_value", "total_value",
    "poll_status", "generated_at", "bull_ratio",
]

TICKER_RANK_COLUMNS = [
    "rank", "ticker", "mention_count", "unique_authors", "total_engagement",
    "bullish_count", "bearish_count", "neutral_count", "generated_at",
]

TOPIC_TRENDING_COLUMNS = [
    "rank", "hashtag", "content_count_total", "content_count_7days",
    "content_count_30days", "view_count", "window_mention_count",
    "window_unique_authors", "window_total_engagement", "window_bullish_count",
    "window_bearish_count", "window_neutral_count", "generated_at",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _empty(columns) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _as_list(v) -> list:
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


def _envelope_data(resp) -> Optional[dict]:
    """Return ``resp['data']`` iff *resp* is an OK envelope, else ``None``.

    A ``code=='000000'`` with ``data is None`` (cache miss) and any transport
    error both map to ``None`` — callers return an empty typed frame.
    """
    if is_ok_envelope(resp):
        return resp["data"]
    return None


def _parse_post_items(items) -> pd.DataFrame:
    if not items:
        return _empty(POST_COLUMNS)
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append({
            "id": str(it.get("id", "")),
            "title": it.get("title") or "",
            "summary": it.get("summary") or "",
            "body": it.get("body") or "",
            "web_link": it.get("webLink") or "",
            "date": _as_int(it.get("date")),
            "latest_release_time": _as_int(it.get("latestReleaseTime")),
            "content_type": _as_int(it.get("contentType")),
            "tendency": _as_int(it.get("tendency")),
            "detected_lang": it.get("detectedLang") or "",
            "is_created_by_ai": bool(it.get("isCreatedByAI")),
            "author_name": it.get("authorName") or "",
            "author_role": it.get("authorRole") or "",
            "like_count": _as_int(it.get("likeCount")),
            "comment_count": _as_int(it.get("commentCount")),
            "share_count": _as_int(it.get("shareCount")),
            "view_count": _as_int(it.get("viewCount")),
            "quote_count": _as_int(it.get("quoteCount")),
            "bookmark_count": _as_int(it.get("bookmarkCount")),
            "rank": _as_int(it.get("rank")),
            "score": _as_float(it.get("score")),
            "tickers": _as_list(it.get("tickers")),
            "user_input_tickers": _as_list(it.get("userInputTickers")),
            "hashtag_list": _as_list(it.get("hashtagList")),
            "generated_at": 0,  # filled by caller from data.generatedAt
        })
    if not rows:
        return _empty(POST_COLUMNS)
    return pd.DataFrame(rows, columns=POST_COLUMNS)


# ---------------------------------------------------------------------------
# Public fetchers
# ---------------------------------------------------------------------------
def fetch_news(
    lang: str = "en",
    page_index: int = 1,
    page_size: int = 20,
    *,
    env: str = "prod",
    ttl: Optional[int] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Binance News official feed → typed DataFrame (columns: :data:`NEWS_COLUMNS`)."""
    ttl_sec = ttl if ttl is not None else TTL_NEWS
    key = ("news", lang, page_index, page_size, env)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    resp = get_public_client(env=env).get_news(lang=lang, page_index=page_index, page_size=page_size)
    data = _envelope_data(resp)
    if data is None:
        return _empty(NEWS_COLUMNS)
    df = _parse_post_items(data.get("items"))
    if not df.empty:
        df["generated_at"] = _as_int(data.get("generatedAt"))
    cache_set(key, df, ttl=ttl_sec)
    return df


def fetch_hot_post(
    sort: str = "HEAT",
    window: str = "24h",
    limit: int = 10,
    lang: str = "en",
    *,
    env: str = "prod",
    ttl: Optional[int] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Global hot posts → typed DataFrame (columns: :data:`POST_COLUMNS`)."""
    ttl_sec = ttl if ttl is not None else TTL_NEWS
    key = ("hot_post", sort, window, limit, lang, env)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    resp = get_public_client(env=env).get_hot_post(sort=sort, window=window, limit=limit, lang=lang)
    data = _envelope_data(resp)
    if data is None:
        return _empty(POST_COLUMNS)
    df = _parse_post_items(data.get("items"))
    if not df.empty:
        df["generated_at"] = _as_int(data.get("generatedAt"))
    cache_set(key, df, ttl=ttl_sec)
    return df


def fetch_search(
    keyword: str = "",
    author: str = "",
    min_likes: int = 0,
    window: str = "24h",
    page_index: int = 1,
    page_size: int = 20,
    lang: str = "en",
    *,
    env: str = "prod",
    ttl: Optional[int] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Square keyword/author search → typed DataFrame.

    At least one of ``keyword``, ``author`` or ``min_likes`` must be set.
    ``getFeed`` is deliberately not used because it is empty on Prod.
    """
    if not keyword and not author and not min_likes:
        raise ValueError("fetch_search requires keyword, author, or min_likes")
    ttl_sec = ttl if ttl is not None else TTL_NEWS
    key = ("search", keyword, author, min_likes, window, page_index, page_size, lang, env)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    resp = get_public_client(env=env).get_search(
        keyword=keyword,
        author=author,
        min_likes=min_likes,
        window=window,
        page_index=page_index,
        page_size=page_size,
        lang=lang,
    )
    data = _envelope_data(resp)
    if data is None:
        return _empty(POST_COLUMNS)
    df = _parse_post_items(data.get("items"))
    if not df.empty:
        df["generated_at"] = _as_int(data.get("generatedAt"))
    cache_set(key, df, ttl=ttl_sec)
    return df


def fetch_sentiment(
    token: str = "BTC",
    *,
    env: str = "prod",
    ttl: Optional[int] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Token community sentiment poll → single-row typed DataFrame.

    Columns: :data:`SENTIMENT_COLUMNS`. ``bull_ratio`` is
    ``bullish / (bullish + bearish)`` (NaN when both are zero).
    """
    ttl_sec = ttl if ttl is not None else TTL_NEWS
    token = str(token).upper()
    key = ("sentiment", token, env)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    resp = get_public_client(env=env).get_sentiment(token=token)
    data = _envelope_data(resp)
    if data is None:
        return _empty(SENTIMENT_COLUMNS)
    bull = _as_int(data.get("bullishValue"))
    bear = _as_int(data.get("bearishValue"))
    denom = bull + bear
    bull_ratio = (bull / denom) if denom > 0 else float("nan")
    df = pd.DataFrame([{
        "token": token,
        "bullish_value": bull,
        "bearish_value": bear,
        "total_value": _as_int(data.get("totalValue")),
        "poll_status": data.get("pollStatus"),
        "generated_at": _as_int(data.get("generatedAt")),
        "bull_ratio": bull_ratio,
    }], columns=SENTIMENT_COLUMNS)
    cache_set(key, df, ttl=ttl_sec)
    return df


def fetch_ticker_rank(
    window: str = "24h",
    limit: int = 20,
    lang: str = "en",
    *,
    env: str = "prod",
    ttl: Optional[int] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Ticker mention ranking → typed DataFrame (columns: :data:`TICKER_RANK_COLUMNS`).

    ``rank`` is 1-based, preserving the upstream ordering (by mention count).
    """
    ttl_sec = ttl if ttl is not None else TTL_NEWS
    key = ("ticker_rank", window, limit, lang, env)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    resp = get_public_client(env=env).get_ticker_rank(window=window, limit=limit, lang=lang)
    data = _envelope_data(resp)
    if data is None:
        return _empty(TICKER_RANK_COLUMNS)
    items = data.get("items") or []
    generated_at = _as_int(data.get("generatedAt"))
    rows = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        rows.append({
            "rank": i + 1,
            "ticker": str(it.get("ticker", "")).upper(),
            "mention_count": _as_int(it.get("mentionCount")),
            "unique_authors": _as_int(it.get("uniqueAuthors")),
            "total_engagement": _as_int(it.get("totalEngagement")),
            "bullish_count": _as_int(it.get("bullishCount")),
            "bearish_count": _as_int(it.get("bearishCount")),
            "neutral_count": _as_int(it.get("neutralCount")),
            "generated_at": generated_at,
        })
    if not rows:
        return _empty(TICKER_RANK_COLUMNS)
    df = pd.DataFrame(rows, columns=TICKER_RANK_COLUMNS)
    cache_set(key, df, ttl=ttl_sec)
    return df


def fetch_topic_trending(
    window: str = "24h",
    limit: int = 10,
    lang: str = "en",
    *,
    env: str = "prod",
    ttl: Optional[int] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Trending topics → typed DataFrame (columns: :data:`TOPIC_TRENDING_COLUMNS`)."""
    ttl_sec = ttl if ttl is not None else TTL_NEWS
    key = ("topic_trending", window, limit, lang, env)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    resp = get_public_client(env=env).get_topic_trending(window=window, limit=limit, lang=lang)
    data = _envelope_data(resp)
    if data is None:
        return _empty(TOPIC_TRENDING_COLUMNS)
    items = data.get("items") or []
    generated_at = _as_int(data.get("generatedAt"))
    rows = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        rows.append({
            "rank": i + 1,
            "hashtag": it.get("hashtagForDisplay") or "",
            "content_count_total": _as_int(it.get("contentCountTotal")),
            "content_count_7days": _as_int(it.get("contentCount7days")),
            "content_count_30days": _as_int(it.get("contentCount30days")),
            "view_count": _as_int(it.get("viewCount")),
            "window_mention_count": _as_int(it.get("windowMentionCount")),
            "window_unique_authors": _as_int(it.get("windowUniqueAuthors")),
            "window_total_engagement": _as_int(it.get("windowTotalEngagement")),
            "window_bullish_count": _as_int(it.get("windowBullishCount")),
            "window_bearish_count": _as_int(it.get("windowBearishCount")),
            "window_neutral_count": _as_int(it.get("windowNeutralCount")),
            "generated_at": generated_at,
        })
    if not rows:
        return _empty(TOPIC_TRENDING_COLUMNS)
    df = pd.DataFrame(rows, columns=TOPIC_TRENDING_COLUMNS)
    cache_set(key, df, ttl=ttl_sec)
    return df
