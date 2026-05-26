"""square-buzz-screener — scanner_report runner.

Replaces the old subprocess pipeline that scraped Binance Square. The new
contract: the AI agent (or a future pro adapter) supplies a structured
snapshot via ``inputs.square_snapshot``; this case dedupes / ranks /
validates against kline data.

snapshot JSON shape::

    {
      "square_snapshot": {
        "en": {
          "most_searched": ["BTC", "ETH"],
          "rapid_riser": ["SOL", "WIF"],
          "hashtags": [{"name": "memecoins", "tokens": ["WIF", "PEPE"]}, ...]
        },
        "zh": { same shape }
      },
      "bars_by_symbol": {"BTCUSDT": [{open, high, low, close, volume, timestamp}, ...]}
    }

Symbols are inferred by appending "USDT" if no quote pair is supplied.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import pandas as pd

from ai_pro_trading_library.library.features.indicators import (
    ema,
    rolling_zscore,
    rsi,
)


def _normalize_symbol(token: str) -> str:
    sym = token.upper().strip()
    if sym.endswith("USDT") or sym.endswith("USDC") or sym.endswith("BUSD"):
        return sym
    return f"{sym}USDT"


def _collect_candidates(snapshot: dict, scoring: dict) -> dict[str, dict]:
    """Score each symbol by section/locale recurrence."""
    sections = ("most_searched", "rapid_riser", "hashtags")
    weight = {
        "most_searched": scoring["most_searched_weight"],
        "rapid_riser": scoring["rapid_riser_weight"],
        "hashtags": scoring["hashtag_weight"],
    }
    seen: dict[str, dict] = {}
    locale_hits: Counter[str] = Counter()
    section_hits: dict[str, set] = {}

    for locale_key, locale_payload in snapshot.items():
        for section in sections:
            entries = locale_payload.get(section) or []
            tokens: list[str] = []
            if section == "hashtags":
                for tag in entries:
                    tokens.extend(tag.get("tokens", []))
            else:
                tokens = list(entries)
            for tok in tokens:
                sym = _normalize_symbol(tok)
                rec = seen.setdefault(sym, {
                    "symbol": sym,
                    "social_score": 0.0,
                    "sections": [],
                    "locales": set(),
                })
                rec["social_score"] += weight[section]
                rec["sections"].append(section)
                rec["locales"].add(locale_key)
                section_hits.setdefault(sym, set()).add(section)
                locale_hits[sym] += 1

    # Apply overlap bonuses.
    for sym, rec in seen.items():
        if len(rec["locales"]) > 1:
            rec["social_score"] += scoring["en_zh_overlap_bonus"]
        if len(section_hits.get(sym, set())) > 1:
            rec["social_score"] += scoring["section_overlap_bonus"]
        rec["recurrence"] = locale_hits[sym]
        rec["locales"] = sorted(rec["locales"])
    return seen


def _technical_score(df: pd.DataFrame, scoring: dict) -> tuple[float, dict]:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    ema_s = float(ema(close, scoring["ema_short"]).iloc[-1])
    ema_l = float(ema(close, scoring["ema_long"]).iloc[-1])
    rsi_value = float(rsi(close, scoring["rsi_period"]).iloc[-1])
    vol_z = float(rolling_zscore(volume, scoring["volume_window"]).iloc[-1])
    last = float(close.iloc[-1])

    if last > ema_s > ema_l:
        trend = 100
        bias = "long"
    elif last < ema_s < ema_l:
        trend = 100
        bias = "short"
    elif last > ema_s:
        trend = 60
        bias = "long"
    else:
        trend = 30
        bias = "watch"

    if rsi_value > 70:
        momentum = 90 if bias == "short" else 30
    elif rsi_value < 30:
        momentum = 90 if bias == "long" else 30
    else:
        momentum = 60

    if vol_z >= scoring["volume_zscore_threshold"]:
        volume_score = 100
    elif vol_z >= 0.5:
        volume_score = 60
    else:
        volume_score = 30

    composite = (trend + momentum + volume_score) / 3
    return composite, {
        "trend": trend,
        "momentum": momentum,
        "volume": volume_score,
        "rsi": round(rsi_value, 2),
        "volume_zscore": round(vol_z, 2),
        "bias": bias,
    }


def _verdict(score: float) -> str:
    if score >= 75:
        return "STRONG_CANDIDATE"
    if score >= 55:
        return "CANDIDATE"
    if score >= 35:
        return "WATCHLIST"
    if score >= 20:
        return "SKIP"
    return "AVOID"


# ---------------------------------------------------------------------------
# Live mode: assemble the square_snapshot from a BinanceProAdapter
# ---------------------------------------------------------------------------


def _extract_symbols(payload: Any, *keys: str, limit: int = 10) -> list[str]:
    """Best-effort extraction of token tickers from a Pro REST response.

    Pro endpoints return a list-of-dicts under various keys (`symbol`,
    `baseAsset`, `name`); we walk the supplied key candidates in order.
    """
    if not payload:
        return []
    if isinstance(payload, dict):
        items = payload.get("list") or payload.get("rank") or payload.get("data") or []
    else:
        items = payload
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value:
                out.append(value)
                break
    return out


def _build_snapshot_from_adapter(adapter: Any) -> dict:
    """Call BinanceProAdapter endpoints and assemble the
    `square_snapshot.{en,zh}.{most_searched,rapid_riser,hashtags}` shape.

    Endpoint mapping:
      most_searched ← `spot_hot_rank` top-N by view count
      rapid_riser   ← `spot_hot_rank` top-N by 24h price change (best effort)
      hashtags      ← `ai_trending` topics with their related tokens

    The Pro REST endpoints return locale-aware payloads; we call each
    endpoint twice with `lang=en` and `lang=zh-CN` to populate both
    locales. Failures are tolerated — a missing endpoint produces an
    empty list for that section.
    """
    snapshot: dict = {"en": {}, "zh": {}}

    for locale_key, locale_param in (("en", "en"), ("zh", "zh-CN")):
        try:
            hot = adapter.spot_hot_rank(lang=locale_param)
        except Exception:
            hot = None
        snapshot[locale_key]["most_searched"] = _extract_symbols(
            hot, "symbol", "baseAsset", "name", limit=10
        )
        snapshot[locale_key]["rapid_riser"] = _extract_symbols(
            hot, "baseAsset", "symbol", "name", limit=10
        )

        try:
            trending = adapter.ai_trending(lang=locale_param)
        except Exception:
            trending = None
        hashtags: list[dict] = []
        if isinstance(trending, dict):
            topics = trending.get("topics") or trending.get("list") or []
        elif isinstance(trending, list):
            topics = trending
        else:
            topics = []
        for topic in topics[:8]:
            if not isinstance(topic, dict):
                continue
            name = topic.get("topic") or topic.get("name") or topic.get("title")
            tokens = topic.get("tokens") or topic.get("symbols") or []
            if isinstance(tokens, list):
                token_list = [
                    str(t.get("symbol") if isinstance(t, dict) else t)
                    for t in tokens
                    if t
                ]
            else:
                token_list = []
            if name and token_list:
                hashtags.append({"name": str(name), "tokens": token_list})
        snapshot[locale_key]["hashtags"] = hashtags

    return snapshot


def _has_pro_endpoints(adapter: Any) -> bool:
    """True iff `adapter` looks like a BinanceProAdapter (has spot_hot_rank)."""
    return adapter is not None and hasattr(adapter, "spot_hot_rank")


def build_report(
    *,
    preset: dict,
    adapter: Any | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict:
    inputs = dict(inputs or {})
    # "key absent" triggers adapter backfill; explicit empty dict is honored.
    snapshot_supplied = "square_snapshot" in inputs
    snapshot = inputs.get("square_snapshot") or {}
    if not snapshot_supplied and _has_pro_endpoints(adapter):
        snapshot = _build_snapshot_from_adapter(adapter)
        inputs["square_snapshot"] = snapshot
    if not snapshot:
        return {
            "signals": [],
            "summary": "No square_snapshot input — supply social heat data or pass a BinanceProAdapter.",
            "recommended_next_step": "Provide inputs.square_snapshot.{en,zh}.{most_searched,rapid_riser,hashtags} or pass adapter=BinanceProAdapter().",
            "candidates": [],
        }

    timeframe = preset.get("timeframe", "1h")
    lookback = int(preset.get("lookback_bars", 200))
    scoring = preset["scoring"]

    raw_candidates = _collect_candidates(snapshot, scoring["social"])
    if not raw_candidates:
        return {
            "signals": [],
            "summary": "Snapshot contained no symbols.",
            "recommended_next_step": "Verify snapshot has tokens under most_searched/rapid_riser/hashtags.",
            "candidates": [],
        }

    bars_by_symbol = inputs.get("bars_by_symbol", {})
    weights = scoring["weights"]
    technical = scoring["technical"]

    rows: list[dict] = []
    for sym, base in raw_candidates.items():
        df_rows = bars_by_symbol.get(sym)
        if df_rows:
            df = pd.DataFrame(df_rows)
        elif adapter is not None:
            from ai_pro_trading_library.library.data.interfaces import KlineRequest

            try:
                bars = adapter.fetch_klines(
                    KlineRequest(instrument_id=sym, timeframe=timeframe, limit=lookback)
                )
                df = pd.DataFrame([
                    {"open": b.open, "high": b.high, "low": b.low,
                     "close": b.close, "volume": b.volume, "timestamp": b.timestamp}
                    for b in bars
                ])
            except Exception:
                df = None
        else:
            df = None

        if df is None or len(df) < technical["ema_long"]:
            base.update({
                "tech_score": None,
                "verdict": "INSUFFICIENT_DATA",
                "composite": round(base["social_score"], 2),
                "bias": "watch",
                "tech_detail": None,
            })
            rows.append(base)
            continue

        tech_score, tech_detail = _technical_score(df, technical)
        # Normalize social score into 0-100 region (cap at 30 raw -> 100).
        social_normalized = min(base["social_score"] / 30.0 * 100, 100)
        composite = (
            social_normalized * weights["social"]
            + tech_score * weights["technical"]
        )
        base.update({
            "tech_score": round(tech_score, 2),
            "social_score_normalized": round(social_normalized, 2),
            "composite": round(composite, 2),
            "verdict": _verdict(composite),
            "bias": tech_detail["bias"],
            "tech_detail": tech_detail,
        })
        rows.append(base)

    rows.sort(
        key=lambda r: (r["verdict"] != "INSUFFICIENT_DATA", r.get("composite", 0)),
        reverse=True,
    )

    signals = [
        {
            "symbol": r["symbol"],
            "side": r.get("bias", "watch"),
            "score": r.get("composite", 0.0),
            "reason": (
                f"verdict={r['verdict']}; social={r['social_score']:.1f}; "
                f"sections={','.join(sorted(set(r['sections'])))}; locales={r['locales']}"
            ),
        }
        for r in rows
        if r.get("verdict") in ("STRONG_CANDIDATE", "CANDIDATE", "WATCHLIST")
    ]

    strong = [r for r in rows if r.get("verdict") == "STRONG_CANDIDATE"]
    return {
        "signals": signals,
        "summary": (
            f"Scored {len(rows)} candidate(s) from social snapshot. "
            f"{len(strong)} STRONG_CANDIDATE."
        ),
        "recommended_next_step": (
            "Feed STRONG_CANDIDATE symbols into structured-trade-plan to derive entry zones."
        ),
        "candidates": rows,
    }
