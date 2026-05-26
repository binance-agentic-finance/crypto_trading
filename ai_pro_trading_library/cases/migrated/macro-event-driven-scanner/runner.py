"""macro-event-driven-scanner — scanner_report runner.

Consumes a macro event calendar (FOMC/CPI/NFP/PMI/...) supplied by the AI
agent via WebSearch per macro_layer.md, fans out each event to its
affected_assets, and aggregates attention scores per ticker.

Input shape::

    {
      "as_of": "2026-05-22",
      "macro_events": [
        {
          "event_type": "fomc", "event_name": "FOMC Decision",
          "event_date": "2026-06-18", "impact_level": "high",
          "consensus": "hold 4.25%", "previous": "hold 4.25%",
          "affected_assets": ["QQQ", "SPY", "TLT", "XLF"],
          "affected_sectors": ["technology", "finance", "etf"],
          "notes": "..."
        }
      ],
      "tradfi_universe": ["NVDA", "TSLA", ..., "QQQ", "SPY", "TLT", "XLF"],
      "ondo_universe": [{"ticker": "QQQ", "symbol": "QQQon", "chain": "solana"}]
    }
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _proximity_score(event_date: date, as_of: date, max_horizon: int, floor: float) -> float:
    delta = (event_date - as_of).days
    if delta < 0 or delta > max_horizon:
        return 0.0
    raw = 1.0 - (delta / max_horizon) * (1.0 - floor)
    return raw  # in [floor, 1.0]


def _has_ondo_endpoints(adapter: Any) -> bool:
    return adapter is not None and hasattr(adapter, "tokenized_stock_list")


def _ondo_universe_from_adapter(adapter: Any) -> list[dict]:
    """Live mode: pull the Ondo tokenized universe from BinanceProAdapter."""
    try:
        raw = adapter.tokenized_stock_list()
    except Exception:
        return []
    items = raw if isinstance(raw, list) else (raw.get("list") or raw.get("data") or [])
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = item.get("ticker") or item.get("symbol")
        if not ticker:
            continue
        ticker_clean = str(ticker)
        if ticker_clean.endswith("on"):
            ticker_clean = ticker_clean[:-2]
        out.append({
            "ticker": ticker_clean.upper(),
            "symbol": item.get("symbol") or item.get("ticker"),
            "chain": item.get("chain") or item.get("network") or "unknown",
        })
    return out


def build_report(
    *,
    preset: dict,
    adapter: Any | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict:
    inputs = dict(inputs or {})
    events = list(inputs.get("macro_events") or [])
    tradfi_universe = set(t.upper() for t in inputs.get("tradfi_universe") or [])
    # `key absent` triggers adapter backfill; explicit empty list is honored.
    ondo_supplied = "ondo_universe" in inputs
    ondo_universe = list(inputs.get("ondo_universe") or [])
    if not ondo_supplied and _has_ondo_endpoints(adapter):
        ondo_universe = _ondo_universe_from_adapter(adapter)
    as_of = _parse_date(inputs.get("as_of"))
    if as_of is None:
        as_of = date.today()

    if not events:
        return {
            "signals": [],
            "summary": "No macro_events input.",
            "recommended_next_step": "Provide inputs.macro_events via WebSearch per macro_layer.md.",
            "as_of": as_of.isoformat(),
            "candidates": [],
        }

    ondo_by_ticker = {o["ticker"].upper(): o for o in ondo_universe if "ticker" in o}
    scoring = preset["scoring"]
    impact_weights = scoring["impact_weights"]
    event_alpha = scoring["event_type_alpha"]
    max_horizon = int(scoring["max_horizon_days"])
    base_pts = scoring["base_event_points"]

    accumulators: dict[str, dict] = {}
    used_events: list[dict] = []

    for ev in events:
        ed = _parse_date(ev.get("event_date") or ev.get("event_time"))
        if ed is None or ed < as_of or (ed - as_of).days > max_horizon:
            continue
        impact = (ev.get("impact_level") or "medium").lower()
        impact_w = impact_weights.get(impact, impact_weights["medium"])
        alpha = event_alpha.get((ev.get("event_type") or "default").lower(), event_alpha["default"])
        prox = _proximity_score(ed, as_of, max_horizon, scoring["days_proximity_floor"])
        used_events.append({
            "event_type": ev.get("event_type"),
            "event_name": ev.get("event_name"),
            "event_date": ev.get("event_date"),
            "impact_level": impact,
            "days_to_event": (ed - as_of).days,
            "weight": round(impact_w * alpha * prox, 4),
        })

        affected = list(ev.get("affected_assets") or [])
        for raw_ticker in affected:
            ticker = str(raw_ticker).upper()
            has_tradfi = ticker in tradfi_universe
            ondo_match = ondo_by_ticker.get(ticker)
            if not has_tradfi and not ondo_match:
                continue

            if has_tradfi and ondo_match:
                venue_score = scoring["venue_bonus_both"]
                venues = ["tradfi_perp", "ondo"]
            elif has_tradfi:
                venue_score = scoring["venue_bonus_tradfi_only"]
                venues = ["tradfi_perp"]
            else:
                venue_score = scoring["venue_bonus_ondo_only"]
                venues = ["ondo"]

            event_points = base_pts * impact_w * alpha * prox + venue_score

            rec = accumulators.setdefault(ticker, {
                "ticker": ticker,
                "venues": venues,
                "ondo_symbol": ondo_match.get("symbol") if ondo_match else None,
                "events": [],
                "score": 0.0,
            })
            rec["score"] += event_points
            rec["events"].append({
                "event_type": ev.get("event_type"),
                "event_date": ev.get("event_date"),
                "impact_level": impact,
                "weighted_points": round(event_points, 2),
            })

    if not accumulators:
        return {
            "signals": [],
            "summary": f"As of {as_of.isoformat()}: no events within {max_horizon}d touched matched tickers.",
            "recommended_next_step": "Verify affected_assets refer to symbols present in tradfi_universe or ondo_universe.",
            "as_of": as_of.isoformat(),
            "candidates": [],
        }

    rows = list(accumulators.values())
    for row in rows:
        row["score"] = round(row["score"], 2)
        row["event_count"] = len(row["events"])
    rows.sort(key=lambda r: r["score"], reverse=True)
    top_n = int(preset.get("top_n", 10))
    top = rows[:top_n]

    signals = [
        {
            "symbol": r["ticker"],
            "side": "watch",
            "score": r["score"],
            "reason": (
                f"{r['event_count']} event(s); venues={','.join(r['venues'])}; "
                f"top_event={r['events'][0]['event_type']}"
            ),
        }
        for r in top
    ]

    return {
        "signals": signals,
        "summary": (
            f"As of {as_of.isoformat()}: {len(used_events)} events within "
            f"{max_horizon}d, {len(rows)} tickers affected. Top {len(top)} returned."
        ),
        "recommended_next_step": (
            "Pick high-scoring tickers and run structured-trade-plan or three-tier-strategy."
        ),
        "as_of": as_of.isoformat(),
        "horizon_days": max_horizon,
        "events_used": used_events,
        "candidates": top,
        "all_ranked": rows,
    }
