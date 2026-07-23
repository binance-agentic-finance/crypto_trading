"""Lookahead-safe news / social feature layer (point-in-time).

This is layer 2 of the news integration. It reads the *point-in-time* (PIT)
capture tree produced by the upstream forward-only collector and projects
lookahead-safe ``_news_*`` feature columns onto a base-timeframe OHLCV frame —
exactly mirroring how ``blocks/strategy.py`` attaches ``_htf_*`` columns.

The one rule that matters
-------------------------
**Alignment is always keyed on *availability time*, never content time.**

* Availability time = when we actually *captured* the data
  (``captured_at_ms`` / ``endpoint_received_at`` / ``capture_completed_at`` from
  ``capture_manifest.json``). This is the only thing a live system could have
  known at bar time.
* Content time = ``news.date`` / ``generatedAt`` — when the article/poll was
  *authored*. Using this to gate features would leak the future, because a
  capture taken at 12:00 routinely contains articles dated 11:30 that a live
  system would only have learned about at 12:00.

For a bar with decision timestamp ``T`` we take the most recent capture with
``avail_ts <= T`` (``np.searchsorted(avail_ts, T, "right") - 1``). Perturbing or
adding any *future* capture (``avail_ts > T``) provably cannot change the value
attached to that bar — see ``tests/test_news_feed.py``.

PIT tree layout (produced upstream, forward-only)::

    <pit_root>/YYYYMMDD/capture_YYYYMMDDTHHMMSSZ/
        capture_manifest.json      # captured_at_ms, endpoint_received_at, ...
        news_page_01.json ...      # getNews (global)
        square_ticker_rank.json    # getTickerRank
        square_sentiment_<TOK>.json# getSentiment
        square_topic_trending.json # getTopicTrending
        square_hot_post_*.json     # getHotPost
        square_search_<TOK>.json   # getSearch (per-token; getFeed is empty on Prod)
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "load_pit_index",
    "build_pit_feature_frame",
    "attach_news_features",
    "ticker_rank_universe",
    "base_token",
    "NEWS_FEATURE_COLUMNS",
]

# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------
# Each feature carries a "kind" that dictates alignment + warmup fill:
#   snapshot_*  → state carried forward from the latest capture (avail <= T)
#   flow_*      → per-bar first-seen event counts (bucketed onto the bar grid)
# Warmup (no capture available yet, or state missing for the token):
#   ratio / rank  → NaN   (absence is not zero)
#   count / flag  → 0.0
_RATIO_FILL = float("nan")
_RANK_FILL = float("nan")
_COUNT_FILL = 0.0
_FLAG_FILL = 0.0

# name -> (kind, source_column_in_capture_frame, warmup_fill)
_FEATURE_SPEC = {
    "_news_sentiment_bull_ratio": ("snapshot", "sentiment_bull_ratio", _RATIO_FILL),
    "_news_ticker_mention_rank":  ("snapshot", "ticker_mention_rank", _RANK_FILL),
    "_news_ticker_mention_count": ("snapshot", "ticker_mention_count", _COUNT_FILL),
    "_news_ticker_bull_ratio":    ("snapshot", "ticker_bull_ratio", _RATIO_FILL),
    "_news_count":                ("flow_count", "news_new_count", _COUNT_FILL),
    "_news_event_flag":           ("flow_flag", "news_new_count", _FLAG_FILL),
}
NEWS_FEATURE_COLUMNS = list(_FEATURE_SPEC.keys())

# Columns emitted by build_pit_feature_frame (one row per capture).
_CAPTURE_FRAME_COLUMNS = [
    "capture_dir", "avail_ts",
    "sentiment_bull_ratio", "ticker_mention_rank", "ticker_mention_count",
    "ticker_bull_ratio", "news_new_count",
]

_QUOTE_SUFFIXES = ("USDT", "USDC", "FDUSD", "TUSD", "BUSD", "USD", "BTC", "ETH", "BNB")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def base_token(symbol: str) -> str:
    """Normalise a market symbol to its base token.

    ``"BTCUSDT" -> "BTC"``, ``"SOLUSDC" -> "SOL"``, ``"BTC" -> "BTC"``. Only the
    first matching quote suffix is stripped, and never the whole string.
    """
    s = str(symbol).upper().strip()
    for q in _QUOTE_SUFFIXES:
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


def _iso_to_ms(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _ticker_matches(item: dict, base: str) -> bool:
    """True if a Square content item mentions *base* (already upper-cased)."""
    tickers = []
    for key in ("tickers", "userInputTickers"):
        v = item.get(key)
        if isinstance(v, list):
            tickers.extend(v)
    for t in tickers:
        tu = str(t).upper()
        if tu == base or base_token(tu) == base:
            return True
    return False


# ---------------------------------------------------------------------------
# PIT index
# ---------------------------------------------------------------------------
def load_pit_index(
    pit_root: str,
    *,
    avail_field: str = "capture_completed_at",
) -> pd.DataFrame:
    """Index every capture under *pit_root*, sorted by availability time.

    Parameters
    ----------
    pit_root : str
        Root of the PIT tree (contains ``YYYYMMDD/`` day directories).
    avail_field : str
        Which manifest timestamp to treat as the capture's availability time:

        * ``"capture_completed_at"`` (default, safest) — the moment the *last*
          endpoint in the capture arrived; guaranteed ``>=`` every per-endpoint
          arrival, so nothing in the capture can ever be seen "too early".
        * ``"captured_at"`` — the documented ``captured_at_ms`` (capture start).

    Returns
    -------
    pd.DataFrame
        Columns ``capture_dir, captured_at_ms, capture_completed_at_ms,
        avail_ts`` — one row per capture, sorted ascending by ``avail_ts``.
        Empty (typed) frame if *pit_root* does not exist / has no captures.
    """
    cols = ["capture_dir", "captured_at_ms", "capture_completed_at_ms", "avail_ts"]
    if not pit_root or not os.path.isdir(pit_root):
        return pd.DataFrame(columns=cols)

    rows = []
    manifests = sorted(glob.glob(os.path.join(pit_root, "*", "*", "capture_manifest.json")))
    for man_path in manifests:
        man = _load_json(man_path)
        if not isinstance(man, dict):
            continue
        captured_at_ms = man.get("captured_at_ms")
        if captured_at_ms is None:
            captured_at_ms = _iso_to_ms(man.get("captured_at_utc"))
        if captured_at_ms is None:
            continue
        completed_ms = _iso_to_ms(man.get("capture_completed_at_utc"))
        if completed_ms is None:
            # Fall back to the latest per-endpoint arrival, else capture start.
            ep = man.get("endpoint_received_at_utc") or {}
            ep_ms = [m for m in (_iso_to_ms(v) for v in ep.values()) if m is not None]
            completed_ms = max(ep_ms) if ep_ms else int(captured_at_ms)

        if avail_field == "captured_at":
            avail_ts = int(captured_at_ms)
        else:  # "capture_completed_at" (default, safest)
            avail_ts = int(max(completed_ms, captured_at_ms))

        rows.append({
            "capture_dir": os.path.dirname(man_path),
            "captured_at_ms": int(captured_at_ms),
            "capture_completed_at_ms": int(completed_ms),
            "avail_ts": int(avail_ts),
        })

    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    # Sort by availability, then capture start, to guarantee a monotonic
    # searchsorted key even if two captures share an availability timestamp.
    df = df.sort_values(["avail_ts", "captured_at_ms"], kind="mergesort").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Per-capture feature frame (for one symbol)
# ---------------------------------------------------------------------------
def _capture_news_ids(capture_dir: str, base: str, news_cfg: dict) -> set:
    """Set of Square content ids in *capture_dir* that mention *base*.

    Sources (deduped by id):
      * global ``news_page_*.json`` filtered to items mentioning the token;
      * per-token ``square_search_<BASE>.json`` (every item counts — it was
        searched for the token). ``getFeed`` is intentionally ignored because it
        returns empty on Prod.
    """
    ids: set = set()
    sources = news_cfg.get("news_sources", ("news_pages", "search"))

    if "news_pages" in sources:
        for page_path in glob.glob(os.path.join(capture_dir, "news_page_*.json")):
            resp = _load_json(page_path)
            data = resp.get("data") if isinstance(resp, dict) else None
            items = (data or {}).get("items") or []
            for it in items:
                if isinstance(it, dict) and _ticker_matches(it, base):
                    _id = str(it.get("id", ""))
                    if _id:
                        ids.add(_id)

    if "search" in sources:
        search_path = os.path.join(capture_dir, f"square_search_{base}.json")
        resp = _load_json(search_path)
        data = resp.get("data") if isinstance(resp, dict) else None
        items = (data or {}).get("items") or []
        for it in items:
            if isinstance(it, dict):
                _id = str(it.get("id", ""))
                if _id:
                    ids.add(_id)

    return ids


def _capture_sentiment(capture_dir: str, base: str) -> float:
    resp = _load_json(os.path.join(capture_dir, f"square_sentiment_{base}.json"))
    data = resp.get("data") if isinstance(resp, dict) else None
    if not isinstance(data, dict):
        return float("nan")
    bull = _as_int(data.get("bullishValue"))
    bear = _as_int(data.get("bearishValue"))
    denom = bull + bear
    return (bull / denom) if denom > 0 else float("nan")


def _capture_ticker_rank(capture_dir: str, base: str):
    """Return ``(rank, mention_count, bull_ratio)`` for *base* in this capture.

    ``rank`` is 1-based; ``NaN`` rank / ``NaN`` bull_ratio and ``0`` count mean
    "token not present / no data" (distinct from a genuine warmup).
    """
    resp = _load_json(os.path.join(capture_dir, "square_ticker_rank.json"))
    data = resp.get("data") if isinstance(resp, dict) else None
    items = (data or {}).get("items") if isinstance(data, dict) else None
    if not items:
        return float("nan"), 0.0, float("nan")
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        if str(it.get("ticker", "")).upper() == base:
            bull = _as_int(it.get("bullishCount"))
            bear = _as_int(it.get("bearishCount"))
            denom = bull + bear
            bull_ratio = (bull / denom) if denom > 0 else float("nan")
            return float(i + 1), float(_as_int(it.get("mentionCount"))), bull_ratio
    return float("nan"), 0.0, float("nan")


def build_pit_feature_frame(
    pit_index: pd.DataFrame,
    symbol: str,
    news_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """Build a per-capture feature frame for one *symbol*.

    Returns one row per capture (sorted by ``avail_ts``) with the raw feature
    values known *as of that capture*. ``news_new_count`` is the number of
    **first-seen** content ids mentioning the token in that capture (an id seen
    in an earlier capture is never counted again — this is the first-seen
    dedup). ``attach_news_features`` then aligns/buckets these onto a bar grid.
    """
    news_cfg = dict(news_cfg or {})
    base = base_token(symbol)

    if pit_index is None or pit_index.empty:
        return pd.DataFrame(columns=_CAPTURE_FRAME_COLUMNS)

    seen_ids: set = set()
    rows = []
    for _, cap in pit_index.iterrows():
        capture_dir = cap["capture_dir"]
        avail_ts = int(cap["avail_ts"])

        # ---- first-seen dedup for flow count ----
        ids = _capture_news_ids(capture_dir, base, news_cfg)
        new_ids = ids - seen_ids
        seen_ids |= ids

        # ---- snapshot state ----
        bull_ratio = _capture_sentiment(capture_dir, base)
        t_rank, t_count, t_bull = _capture_ticker_rank(capture_dir, base)

        rows.append({
            "capture_dir": capture_dir,
            "avail_ts": avail_ts,
            "sentiment_bull_ratio": bull_ratio,
            "ticker_mention_rank": t_rank,
            "ticker_mention_count": t_count,
            "ticker_bull_ratio": t_bull,
            "news_new_count": float(len(new_ids)),
        })

    return pd.DataFrame(rows, columns=_CAPTURE_FRAME_COLUMNS)


# ---------------------------------------------------------------------------
# Attach onto a base-timeframe frame
# ---------------------------------------------------------------------------
def _base_ts_array(df: pd.DataFrame, as_of: str) -> np.ndarray:
    """Per-bar decision timestamp (ms): ``open_time`` (as_of='open') or
    ``close_time`` (as_of='close', default).

    Lookahead safety: we must NEVER silently fall back to a *later* timestamp
    than the caller asked for. When ``as_of='open'`` and ``open_time`` is
    missing we raise rather than fall back to ``close_time`` — using close-time
    would let an open-aligned bar see captures that only became available later
    in the same bar, i.e. a within-bar future leak.
    """
    if as_of not in ("open", "close"):
        raise ValueError(f"as_of must be 'open' or 'close', got {as_of!r}")

    if as_of == "open":
        if "open_time" not in df.columns:
            raise ValueError(
                "as_of='open' requires an 'open_time' column; refusing to fall "
                "back to close_time (that would leak within-bar future data)."
            )
        col = "open_time"
    else:  # "close": the bar is actionable at its close
        if "close_time" in df.columns:
            col = "close_time"
        elif "timestamp" in df.columns:  # framework alias for close_time
            col = "timestamp"
        else:
            raise ValueError(
                "as_of='close' requires a 'close_time' (or 'timestamp') column."
            )
    return df[col].astype("int64").to_numpy()


def attach_news_features(
    df: pd.DataFrame,
    news_cfg: Optional[dict],
    symbol: str,
    as_of: str = "close",
) -> pd.DataFrame:
    """Attach lookahead-safe ``_news_*`` columns to *df* for *symbol*.

    Parameters
    ----------
    df : pd.DataFrame
        Base-timeframe OHLCV frame; must carry ``close_time`` (or ``open_time``
        when ``as_of='open'``) as integer epoch-ms.
    news_cfg : dict
        Configuration. Recognised keys:

        * ``pit_root`` (str) — PIT tree root. **Required** unless
          ``strict=False``.
        * ``avail_field`` — see :func:`load_pit_index` (default
          ``"capture_completed_at"``).
        * ``news_sources`` — iterable subset of ``{"news_pages", "search"}``.
        * ``features`` — optional subset of :data:`NEWS_FEATURE_COLUMNS`.
        * ``strict`` (bool, default ``True``) — if ``pit_root`` is missing/empty,
          raise when ``True`` else attach all-warmup columns.
    symbol : str
        Market symbol, e.g. ``"BTCUSDT"``.
    as_of : str
        ``"close"`` (default) aligns on each bar's ``close_time``; ``"open"``
        aligns on ``open_time`` (stricter — only news available before the bar
        opened is visible).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the requested ``_news_*`` columns added.
    """
    news_cfg = dict(news_cfg or {})
    out = df.copy()
    features = list(news_cfg.get("features", NEWS_FEATURE_COLUMNS))

    if out.empty:
        for name in features:
            out[name] = pd.Series(dtype="float64")
        return out

    pit_root = news_cfg.get("pit_root")
    strict = news_cfg.get("strict", True)
    if not pit_root or not os.path.isdir(str(pit_root)):
        if strict:
            raise ValueError(
                "attach_news_features requires news_cfg['pit_root'] to point at "
                "a PIT capture tree (set strict=False to attach warmup columns)"
            )
        return _attach_all_warmup(out, features)

    pit_index = load_pit_index(
        str(pit_root),
        avail_field=news_cfg.get("avail_field", "capture_completed_at"),
    )
    frame = build_pit_feature_frame(pit_index, symbol, news_cfg)

    base_ts = _base_ts_array(out, as_of)
    n = len(base_ts)

    if frame.empty:
        return _attach_all_warmup(out, features)

    avail = frame["avail_ts"].to_numpy(dtype="int64")

    # Snapshot index: latest capture with avail <= base_ts (searchsorted right-1).
    snap_idx = np.searchsorted(avail, base_ts, side="right") - 1
    snap_valid = snap_idx >= 0

    # Flow bucketing: assign each capture's first-seen count to the first bar
    # whose decision time is >= the capture's availability time. Sort base_ts to
    # keep searchsorted valid, then scatter results back to df order.
    order = np.argsort(base_ts, kind="mergesort")
    base_sorted = base_ts[order]
    cap_new = frame["news_new_count"].to_numpy(dtype="float64")
    bar_pos_sorted = np.searchsorted(base_sorted, avail, side="left")
    flow_count_sorted = np.zeros(n, dtype="float64")
    for k in range(len(avail)):
        b = int(bar_pos_sorted[k])
        if 0 <= b < n:
            flow_count_sorted[b] += cap_new[k]
    flow_count = np.empty(n, dtype="float64")
    flow_count[order] = flow_count_sorted

    for name in features:
        kind, src_col, warmup = _FEATURE_SPEC[name]
        if kind == "snapshot":
            vals = frame[src_col].to_numpy(dtype="float64")
            aligned = np.full(n, warmup, dtype="float64")
            aligned[snap_valid] = vals[snap_idx[snap_valid]]
            out[name] = aligned
        elif kind == "flow_count":
            out[name] = flow_count
        elif kind == "flow_flag":
            out[name] = (flow_count > 0).astype("float64")

    return out


def _attach_all_warmup(out: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    n = len(out)
    for name in features:
        _, _, warmup = _FEATURE_SPEC[name]
        out[name] = np.full(n, warmup, dtype="float64")
    return out


# ---------------------------------------------------------------------------
# Selection helper
# ---------------------------------------------------------------------------
def ticker_rank_universe(
    pit_index: pd.DataFrame,
    as_of_ms: int,
    *,
    top_n: int = 20,
    by: str = "mention_count",
) -> pd.DataFrame:
    """Lookahead-safe ticker-mention universe as of *as_of_ms*.

    Reads ``square_ticker_rank.json`` from the most recent capture with
    ``avail_ts <= as_of_ms`` and returns the top-*n* tickers. Empty (typed)
    frame if no capture is available yet.

    Returns columns: ``ticker, mention_count, unique_authors, bullish_count,
    bearish_count, neutral_count, bull_ratio, rank`` (``rank`` is 1-based in the
    returned ordering).
    """
    cols = ["ticker", "mention_count", "unique_authors", "bullish_count",
            "bearish_count", "neutral_count", "bull_ratio", "rank"]
    if pit_index is None or pit_index.empty:
        return pd.DataFrame(columns=cols)

    avail = pit_index["avail_ts"].to_numpy(dtype="int64")
    idx = int(np.searchsorted(avail, int(as_of_ms), side="right") - 1)
    if idx < 0:
        return pd.DataFrame(columns=cols)

    capture_dir = pit_index.iloc[idx]["capture_dir"]
    resp = _load_json(os.path.join(capture_dir, "square_ticker_rank.json"))
    data = resp.get("data") if isinstance(resp, dict) else None
    items = (data or {}).get("items") if isinstance(data, dict) else None
    if not items:
        return pd.DataFrame(columns=cols)

    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        bull = _as_int(it.get("bullishCount"))
        bear = _as_int(it.get("bearishCount"))
        denom = bull + bear
        rows.append({
            "ticker": str(it.get("ticker", "")).upper(),
            "mention_count": _as_int(it.get("mentionCount")),
            "unique_authors": _as_int(it.get("uniqueAuthors")),
            "bullish_count": bull,
            "bearish_count": bear,
            "neutral_count": _as_int(it.get("neutralCount")),
            "bull_ratio": (bull / denom) if denom > 0 else float("nan"),
        })
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    sort_col = by if by in df.columns else "mention_count"
    df = df.sort_values(sort_col, ascending=False, kind="mergesort").head(int(top_n)).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df[cols]
