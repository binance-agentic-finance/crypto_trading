"""earnings-tradfi-scanner — scanner_report runner.

Cross-references an external earnings calendar against TradFi perpetual
contracts and Ondo tokenized equities. The earnings calendar is supplied
as ``inputs.earnings_calendar`` (per macro_layer.md WebSearch contract);
this case never WebSearches itself.

Input shape::

    {
      "as_of": "2026-05-22",
      "earnings_calendar": [
        {"ticker": "NVDA", "company": "NVIDIA Corp", "report_date": "2026-05-28",
         "period": "Q1 2027", "eps_estimate": 0.85, "eps_prior": 0.61,
         "revenue_estimate_b": 28.0, "revenue_prior_b": 18.0,
         "market_cap_category": "mega", "sector": "semiconductor", "notes": "..."}
      ],
      "tradfi_universe": ["NVDA", "TSLA", "AAPL", "QQQ", "SPY", ...],
      "ondo_universe": [{"ticker": "NVDA", "symbol": "NVDAon", "chain": "solana"}, ...]
    }
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Mapping


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _eps_surprise_score(entry: dict) -> float:
    estimate = entry.get("eps_estimate")
    prior = entry.get("eps_prior")
    if estimate is None or prior is None or prior == 0:
        return 0.5
    growth = (estimate - prior) / abs(prior)
    # 0% growth → 0.5; +50% → 1.0; -25% → 0.0
    score = 0.5 + growth
    return max(0.0, min(1.0, score))


def _days_to_report_score(entry: dict, as_of: date | None, max_horizon: int) -> float:
    rd = _parse_date(entry.get("report_date"))
    if rd is None or as_of is None:
        return 0.5
    delta = (rd - as_of).days
    if delta < 0:
        return 0.0
    if delta > max_horizon:
        return 0.0
    # Closer to report = higher score (linear).
    return 1.0 - (delta / max_horizon) * 0.5  # 1d → ~0.93, 7d → 0.5


def _venue_match(ticker: str, tradfi: set[str], ondo_by_ticker: dict[str, dict]) -> tuple[bool, dict | None]:
    has_tradfi = ticker in tradfi
    ondo = ondo_by_ticker.get(ticker)
    return has_tradfi, ondo


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
        # Strip the "on" suffix (e.g., "QQQon" → "QQQ") to match earnings tickers.
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
    earnings = list(inputs.get("earnings_calendar") or [])
    tradfi_universe = set(t.upper() for t in inputs.get("tradfi_universe") or [])
    # Distinguish "key absent" (None → adapter backfills) from "explicit
    # empty list" (caller declared no ondo coverage). Same convention is
    # used by macro-event-driven-scanner.
    ondo_supplied = "ondo_universe" in inputs
    ondo_universe = list(inputs.get("ondo_universe") or [])
    as_of = _parse_date(inputs.get("as_of"))
    if as_of is None:
        as_of = date.today()

    # Live mode: backfill ondo_universe from the Pro adapter only when
    # the caller didn't mention it at all. earnings_calendar stays
    # caller/agent supplied per macro_layer.md (no live API for that).
    if not ondo_supplied and _has_ondo_endpoints(adapter):
        ondo_universe = _ondo_universe_from_adapter(adapter)

    if not earnings:
        return {
            "signals": [],
            "summary": "No earnings_calendar input.",
            "recommended_next_step": "Provide inputs.earnings_calendar via macro_layer.md WebSearch (the agent supplies this; the library does not WebSearch).",
            "as_of": as_of.isoformat(),
            "candidates": [],
        }

    ondo_by_ticker = {o["ticker"].upper(): o for o in ondo_universe if "ticker" in o}
    scoring = preset["scoring"]
    weights = scoring["weights"]
    max_horizon = int(scoring["max_horizon_days"])

    rows: list[dict] = []
    for entry in earnings:
        ticker = str(entry.get("ticker", "")).upper()
        if not ticker:
            continue
        rd = _parse_date(entry.get("report_date"))
        if rd is None or rd < as_of or (rd - as_of).days > max_horizon:
            continue

        has_tradfi, ondo_match = _venue_match(ticker, tradfi_universe, ondo_by_ticker)
        if not has_tradfi and not ondo_match:
            continue

        eps_score = _eps_surprise_score(entry)
        time_score = _days_to_report_score(entry, as_of, max_horizon)
        cap_tier = entry.get("market_cap_category", "default")
        cap_score = scoring["market_cap_scores"].get(cap_tier, scoring["market_cap_scores"].get("default", 0.0))
        sector = (entry.get("sector") or "default").lower()
        sector_score = scoring["sector_alpha_scores"].get(sector, scoring["sector_alpha_scores"]["default"])

        if has_tradfi and ondo_match:
            venue_score = weights["venue_bonus_both"]
            venues = ["tradfi_perp", "ondo"]
        elif has_tradfi:
            venue_score = weights["venue_bonus_tradfi_only"]
            venues = ["tradfi_perp"]
        else:
            venue_score = weights["venue_bonus_ondo_only"]
            venues = ["ondo"]

        composite = (
            time_score * weights["days_to_report"]
            + eps_score * weights["eps_surprise_potential"]
            + cap_score * weights["market_cap_tier"]
            + sector_score * weights["sector_alpha"]
            + venue_score
        )

        rows.append({
            "ticker": ticker,
            "company": entry.get("company"),
            "report_date": entry.get("report_date"),
            "days_to_report": (rd - as_of).days,
            "period": entry.get("period"),
            "market_cap_category": cap_tier,
            "sector": entry.get("sector"),
            "eps_estimate": entry.get("eps_estimate"),
            "eps_prior": entry.get("eps_prior"),
            "venues": venues,
            "ondo_symbol": ondo_match.get("symbol") if ondo_match else None,
            "score_breakdown": {
                "time_score": round(time_score, 3),
                "eps_surprise_score": round(eps_score, 3),
                "market_cap_score": cap_score,
                "sector_score": sector_score,
                "venue_score": venue_score,
            },
            "composite": round(composite, 2),
        })

    rows.sort(key=lambda r: r["composite"], reverse=True)
    top_n = int(preset.get("top_n", 5))
    top = rows[:top_n]

    signals = [
        {
            "symbol": r["ticker"],
            "side": "long" if (r["score_breakdown"]["eps_surprise_score"] >= 0.5) else "watch",
            "score": r["composite"],
            "reason": (
                f"earnings in {r['days_to_report']}d; venues={','.join(r['venues'])}; "
                f"sector={r['sector']}; cap={r['market_cap_category']}"
            ),
        }
        for r in top
    ]

    return {
        "signals": signals,
        "summary": (
            f"As of {as_of.isoformat()}: {len(earnings)} earnings events, "
            f"{len(rows)} matched venues within {max_horizon}d. Top {len(top)} ranked."
        ),
        "recommended_next_step": (
            "Drop top-N tickers into structured-trade-plan or three-tier-strategy "
            "for entry zone analysis."
        ),
        "as_of": as_of.isoformat(),
        "horizon_days": max_horizon,
        "candidates": top,
        "all_ranked": rows,
    }
