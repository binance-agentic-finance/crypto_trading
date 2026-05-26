"""black-swan-insurance — scanner_report runner.

Computes a 12-factor panic score and emits a tier-based hedge allocation plan.
This is **planning only** — no orders are placed; the agent must hand the
allocation off to a BAW swap workflow if the tier crosses `alert`.

Input shape::

    {
      "market_signals": {
        "btc_drawdown_1h_pct": 4.2,
        "btc_drawdown_24h_pct": 9.8,
        "top10_cap_drawdown_24h_pct": 7.5,
        "stablecoin_depeg_bps": 25.0,
        "vix_proxy_change_24h_pct": 22.0,
        "spy_qqq_drawdown_1h_pct": 1.6,
        "gold_change_24h_pct": 1.4,
        "defi_tvl_drawdown_24h_pct": 4.0
      },
      "narrative_signals": {
        "negative_keyword_hit_count": 1,
        "social_sentiment_min": -0.20,
        "smart_money_inflow_top10_avg_usdc": -40000,
        "hotrank_majors_missing": 0,
        "ai_tradesignal_bearish_pct": 0.45
      },
      "portfolio": {
        "total_value_usdc": 10000.0,
        "holdings": [
          {"symbol": "BTC", "value_usdc": 6000},
          {"symbol": "ETH", "value_usdc": 3000},
          {"symbol": "SOL", "value_usdc": 1000}
        ]
      }
    }
"""

from __future__ import annotations

from typing import Any, Mapping


def _signal_score(value: float, panic_threshold: float, extreme_threshold: float, direction: str = "positive") -> float:
    """Score 0..1: 0 below panic_threshold, 1 at extreme_threshold and beyond."""
    if direction == "negative":
        # value should be <= threshold to fire (e.g. drawdown stored as -X, sentiment as -X).
        if value > panic_threshold:
            return 0.0
        if value <= extreme_threshold:
            return 1.0
        span = panic_threshold - extreme_threshold
        if span <= 0:
            return 1.0
        return (panic_threshold - value) / span
    # positive direction (default): higher value = more panic
    if value < panic_threshold:
        return 0.0
    if value >= extreme_threshold:
        return 1.0
    span = extreme_threshold - panic_threshold
    if span <= 0:
        return 1.0
    return (value - panic_threshold) / span


def _evaluate_signal_block(payload: dict, definitions: dict) -> tuple[float, list[dict]]:
    breakdown: list[dict] = []
    total = 0.0
    for signal_name, defn in definitions.items():
        if signal_name not in payload:
            breakdown.append({
                "signal": signal_name, "weight": defn["weight"],
                "value": None, "score": 0.0, "contribution": 0.0,
                "status": "missing",
            })
            continue
        value = float(payload[signal_name])
        score = _signal_score(
            value,
            float(defn["panic_threshold"]),
            float(defn["extreme_threshold"]),
            direction=defn.get("direction", "positive"),
        )
        contribution = score * defn["weight"]
        total += contribution
        breakdown.append({
            "signal": signal_name,
            "weight": defn["weight"],
            "value": value,
            "score": round(score, 3),
            "contribution": round(contribution, 3),
            "status": "fired" if score > 0 else "calm",
        })
    return total, breakdown


def _select_tier(panic_score: float, tiers: dict) -> tuple[str, dict]:
    # Tiers are listed in ascending max_score; pick the first whose max_score is greater.
    ordered = sorted(tiers.items(), key=lambda kv: kv[1]["max_score"])
    for name, cfg in ordered:
        if panic_score < cfg["max_score"]:
            return name, cfg
    last_name, last_cfg = ordered[-1]
    return last_name, last_cfg


# ---------------------------------------------------------------------------
# Live mode: derive narrative_signals from a BinanceProAdapter
# ---------------------------------------------------------------------------


_NEGATIVE_KEYWORDS = (
    "hack", "exploit", "sec", "depeg", "war", "liquidation", "rug", "regulator",
    "delist", "freeze", "default", "bankruptcy",
)


def _has_pro_endpoints(adapter: Any) -> bool:
    return adapter is not None and all(
        hasattr(adapter, m)
        for m in ("ai_trending", "social_hype_rank", "smart_money_inflow", "ai_signal_rank", "spot_hot_rank")
    )


def _count_negative_keywords(adapter: Any) -> int:
    try:
        trending = adapter.ai_trending()
    except Exception:
        return 0
    if isinstance(trending, dict):
        topics = trending.get("topics") or trending.get("list") or []
    elif isinstance(trending, list):
        topics = trending
    else:
        return 0
    hits = 0
    for t in topics:
        text_parts: list[str] = []
        if isinstance(t, dict):
            for k in ("topic", "name", "title", "summary", "description"):
                v = t.get(k)
                if isinstance(v, str):
                    text_parts.append(v.lower())
        elif isinstance(t, str):
            text_parts.append(t.lower())
        text = " ".join(text_parts)
        if any(kw in text for kw in _NEGATIVE_KEYWORDS):
            hits += 1
    return hits


def _min_social_sentiment(adapter: Any) -> float | None:
    try:
        ranks = adapter.social_hype_rank()
    except Exception:
        return None
    items = ranks if isinstance(ranks, list) else (ranks.get("list") or [])
    sentiments: list[float] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("sentiment") or item.get("sentimentScore")
        if value is not None:
            try:
                sentiments.append(float(value))
            except (TypeError, ValueError):
                continue
    return min(sentiments) if sentiments else None


def _smart_money_top10_inflow_avg(adapter: Any) -> float | None:
    try:
        ranks = adapter.smart_money_inflow(period="1h")
    except Exception:
        return None
    items = ranks if isinstance(ranks, list) else (ranks.get("list") or [])
    inflows: list[float] = []
    for item in items[:10]:
        if not isinstance(item, dict):
            continue
        value = item.get("inflow") or item.get("netInflow") or item.get("inflowUsd")
        if value is not None:
            try:
                inflows.append(float(value))
            except (TypeError, ValueError):
                continue
    return sum(inflows) / len(inflows) if inflows else None


def _hotrank_majors_missing(adapter: Any) -> int:
    try:
        hot = adapter.spot_hot_rank()
    except Exception:
        return 0
    items = hot if isinstance(hot, list) else (hot.get("list") or hot.get("rank") or [])
    top10 = []
    for item in items[:10]:
        if not isinstance(item, dict):
            continue
        sym = item.get("baseAsset") or item.get("symbol") or item.get("name")
        if isinstance(sym, str):
            top10.append(sym.upper())
    missing = 0
    for major in ("BTC", "ETH"):
        if not any(major in s for s in top10):
            missing += 1
    return missing


def _ai_tradesignal_bearish_pct(adapter: Any) -> float | None:
    try:
        ranks = adapter.ai_signal_rank()
    except Exception:
        return None
    items = ranks if isinstance(ranks, list) else (ranks.get("list") or [])
    if not items:
        return None
    bearish = 0
    total = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        side = item.get("direction") or item.get("side") or item.get("bias")
        if isinstance(side, str):
            total += 1
            if side.lower() in ("bearish", "short", "sell"):
                bearish += 1
    return bearish / total if total else None


def _derive_narrative_from_adapter(adapter: Any) -> dict:
    out: dict = {}
    out["negative_keyword_hit_count"] = _count_negative_keywords(adapter)
    sentiment = _min_social_sentiment(adapter)
    if sentiment is not None:
        out["social_sentiment_min"] = sentiment
    inflow = _smart_money_top10_inflow_avg(adapter)
    if inflow is not None:
        out["smart_money_inflow_top10_avg_usdc"] = inflow
    out["hotrank_majors_missing"] = _hotrank_majors_missing(adapter)
    bearish_pct = _ai_tradesignal_bearish_pct(adapter)
    if bearish_pct is not None:
        out["ai_tradesignal_bearish_pct"] = bearish_pct
    return out


def build_report(
    *,
    preset: dict,
    adapter: Any | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict:
    inputs = dict(inputs or {})
    market_signals = dict(inputs.get("market_signals") or {})
    narrative_signals = dict(inputs.get("narrative_signals") or {})
    portfolio = dict(inputs.get("portfolio") or {})

    # Live mode: backfill narrative_signals from the Pro adapter when it's
    # available and inputs didn't already supply them. Price signals stay
    # caller-supplied because they need a market data adapter (not Pro).
    if _has_pro_endpoints(adapter) and not narrative_signals:
        narrative_signals = _derive_narrative_from_adapter(adapter)

    if not market_signals:
        return {
            "signals": [],
            "summary": "No market_signals provided; cannot evaluate panic.",
            "recommended_next_step": "Supply inputs.market_signals with the 8 price-based metrics (sourced from a market data adapter; the BinanceProAdapter only covers narrative_signals).",
            "panic_score": None,
            "tier": "unknown",
        }

    scoring = preset["scoring"]
    price_score, price_breakdown = _evaluate_signal_block(
        market_signals, scoring["price_signals"]
    )
    narrative_score, narrative_breakdown = _evaluate_signal_block(
        narrative_signals, scoring["narrative_signals"]
    )

    raw_total = price_score + narrative_score
    panic_score = round(min(raw_total, 100.0), 2)

    tier_name, tier_cfg = _select_tier(panic_score, preset["tiers"])
    allocation_pcts = tier_cfg["allocation"]

    total_value = float(portfolio.get("total_value_usdc") or 0.0)
    allocation_usdc: dict[str, float] = {}
    if total_value > 0:
        for asset, pct in allocation_pcts.items():
            allocation_usdc[asset] = round(total_value * pct, 2)
    else:
        allocation_usdc = {asset: pct for asset, pct in allocation_pcts.items()}

    allocation_lines = [
        f"{asset}: {pct*100:.0f}%" + (f" (~${allocation_usdc[asset]})" if total_value > 0 else "")
        for asset, pct in allocation_pcts.items()
    ]

    signals = [
        {
            "symbol": "PORTFOLIO",
            "side": "hedge" if tier_name in ("alert", "black_swan", "extreme") else "hold",
            "score": panic_score,
            "reason": (
                f"tier={tier_name}; "
                f"fired_signals={sum(1 for b in price_breakdown + narrative_breakdown if b['status'] == 'fired')}"
            ),
        }
    ]

    return {
        "signals": signals,
        "summary": (
            f"panic_score={panic_score}/100, tier={tier_name}. "
            f"Hedge allocation: {', '.join(allocation_lines)}."
        ),
        "recommended_next_step": (
            f"Tier '{tier_name}': "
            + (
                "no action — keep monitoring."
                if tier_name == "calm"
                else "execute the recommended_allocation via BAW swap; arm the watcher with the unwind conditions."
            )
        ),
        "panic_score": panic_score,
        "raw_total": round(raw_total, 2),
        "tier": tier_name,
        "tier_max_score": tier_cfg["max_score"],
        "recommended_allocation_pct": allocation_pcts,
        "recommended_allocation_usdc": allocation_usdc if total_value > 0 else None,
        "price_signal_breakdown": price_breakdown,
        "narrative_signal_breakdown": narrative_breakdown,
        "unwind_conditions": preset.get("unwind_conditions", []),
    }
