"""ondo-daily — scanner_report runner.

Consumes a snapshot of the Ondo universe + alpha hotrank + 24h movers and
produces ranked discovery candidates. The fixture mirrors what
``binance-pro-cli web3 tokenized list / hotrank alpha`` would return; future
versions can plug in a real adapter that calls those endpoints.

Input shape::

    {
      "ondo_market_status": "open" | "closed" | "pre" | "post",
      "ondo_universe": [
        {"ticker": "QQQ", "symbol": "QQQon", "chain": "solana", "contract": "...",
         "multiplier": 1.0, "volume_24h": 12345.0}
      ],
      "alpha_hotrank": [
        {"rank": 1, "symbol": "QQQon", "score": 0.95},
        {"rank": 2, "symbol": "TSLAon", "score": 0.91}
      ],
      "movers_24h": [
        {"symbol": "TSLAon", "change_pct": 6.2, "premium_pct": 0.4},
        {"symbol": "NVDAon", "change_pct": -4.1, "premium_pct": -0.3}
      ]
    }
"""

from __future__ import annotations

from typing import Any, Mapping


def _build_index(items: list[dict], key: str) -> dict[str, dict]:
    return {item[key]: item for item in items if key in item}


# ---------------------------------------------------------------------------
# Live mode: assemble the ondo snapshot from a BinanceProAdapter
# ---------------------------------------------------------------------------


def _has_ondo_endpoints(adapter: Any) -> bool:
    """True iff `adapter` exposes the Ondo endpoints (BinanceProAdapter)."""
    return adapter is not None and all(
        hasattr(adapter, m)
        for m in ("tokenized_stock_list", "tokenized_market_status", "alpha_hot_rank")
    )


def _build_universe_from_adapter(adapter: Any) -> list[dict]:
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
        out.append({
            "ticker": str(ticker).replace("on", "").upper() if str(ticker).endswith("on") else str(ticker).upper(),
            "symbol": item.get("symbol") or item.get("ticker"),
            "chain": item.get("chain") or item.get("network") or "unknown",
            "contract": item.get("contract") or item.get("address") or "",
            "multiplier": float(item.get("multiplier", 1.0) or 1.0),
            "volume_24h": float(item.get("volume24h") or item.get("volume_24h") or 0.0),
        })
    return out


def _build_market_status_from_adapter(adapter: Any) -> str:
    try:
        raw = adapter.tokenized_market_status()
    except Exception:
        return "unknown"
    if isinstance(raw, dict):
        status = raw.get("status") or raw.get("marketStatus")
        if isinstance(status, str):
            return status.lower()
    if isinstance(raw, str):
        return raw.lower()
    return "unknown"


def _build_hotrank_from_adapter(adapter: Any) -> list[dict]:
    try:
        raw = adapter.alpha_hot_rank()
    except Exception:
        return []
    items = raw if isinstance(raw, list) else (raw.get("list") or raw.get("rank") or [])
    out: list[dict] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        sym = item.get("symbol") or item.get("ticker")
        if not sym:
            continue
        out.append({
            "rank": int(item.get("rank", idx) or idx),
            "symbol": str(sym),
            "score": float(item.get("score", 0.0) or 0.0),
        })
    return out


def _build_movers_from_universe(universe: list[dict], adapter: Any) -> list[dict]:
    """24h movers come from per-token dynamic queries. We cap the fan-out
    to the top 30 universe entries by 24h volume to avoid pathological
    fan-out on the Ondo endpoint."""
    if not hasattr(adapter, "tokenized_dynamic"):
        return []
    sorted_universe = sorted(
        universe, key=lambda u: float(u.get("volume_24h", 0.0) or 0.0), reverse=True
    )[:30]
    movers: list[dict] = []
    for entry in sorted_universe:
        sym = entry.get("symbol")
        if not sym:
            continue
        try:
            dyn = adapter.tokenized_dynamic(sym)
        except Exception:
            continue
        if not isinstance(dyn, dict):
            continue
        change = dyn.get("priceChange24h") or dyn.get("change_pct")
        premium = dyn.get("premium") or dyn.get("premium_pct")
        if change is None:
            continue
        try:
            movers.append({
                "symbol": sym,
                "change_pct": float(change),
                "premium_pct": float(premium) if premium is not None else 0.0,
            })
        except (TypeError, ValueError):
            continue
    return movers


def build_report(
    *,
    preset: dict,
    adapter: Any | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict:
    inputs = dict(inputs or {})
    # "key absent" triggers adapter backfill; explicit empty value is honored.
    universe_supplied = "ondo_universe" in inputs
    hotrank_supplied = "alpha_hotrank" in inputs
    movers_supplied = "movers_24h" in inputs
    universe = list(inputs.get("ondo_universe") or [])
    hotrank = list(inputs.get("alpha_hotrank") or [])
    movers = list(inputs.get("movers_24h") or [])
    market_status = inputs.get("ondo_market_status")

    if _has_ondo_endpoints(adapter):
        if not universe_supplied:
            universe = _build_universe_from_adapter(adapter)
        if market_status is None:
            market_status = _build_market_status_from_adapter(adapter)
        if not hotrank_supplied:
            hotrank = _build_hotrank_from_adapter(adapter)
        if not movers_supplied and universe:
            movers = _build_movers_from_universe(universe, adapter)

    market_status = str(market_status or "unknown")

    if not universe:
        return {
            "signals": [],
            "summary": "No ondo_universe in inputs.",
            "recommended_next_step": "Provide inputs.ondo_universe + alpha_hotrank + movers_24h, or pass adapter=BinanceProAdapter().",
            "ondo_market_status": market_status,
            "candidates": [],
        }

    scoring = preset["scoring"]
    universe_by_symbol = _build_index(universe, "symbol")

    hotrank_top = [
        h for h in hotrank if h.get("rank", 999) <= scoring["hotrank_top_n_threshold"]
    ]
    hotrank_top.sort(key=lambda h: h["rank"])
    movers_filtered = [
        m for m in movers
        if abs(m.get("change_pct", 0)) >= scoring["abs_change_floor_pct"]
    ]
    movers_filtered.sort(key=lambda m: abs(m.get("change_pct", 0)), reverse=True)
    movers_top = movers_filtered[: scoring["mover_top_n_threshold"]]

    candidates: dict[str, dict] = {}

    # Score hotrank entries.
    hotrank_rank_count = max(len(hotrank_top), 1)
    for h in hotrank_top:
        sym = h.get("symbol")
        if sym not in universe_by_symbol:
            continue
        rank_pts = (hotrank_rank_count - h["rank"] + 1) * scoring["hotrank_position_weight"]
        rec = candidates.setdefault(sym, {
            "symbol": sym,
            "ticker": universe_by_symbol[sym].get("ticker"),
            "chain": universe_by_symbol[sym].get("chain"),
            "score": 0.0,
            "sources": [],
            "hotrank": None,
            "mover": None,
        })
        rec["score"] += rank_pts
        rec["sources"].append("hotrank")
        rec["hotrank"] = {"rank": h["rank"], "score": h.get("score")}

    # Score mover entries.
    mover_count = max(len(movers_top), 1)
    for idx, m in enumerate(movers_top):
        sym = m.get("symbol")
        if sym not in universe_by_symbol:
            continue
        rank_pts = (mover_count - idx) * scoring["mover_position_weight"]
        rec = candidates.setdefault(sym, {
            "symbol": sym,
            "ticker": universe_by_symbol[sym].get("ticker"),
            "chain": universe_by_symbol[sym].get("chain"),
            "score": 0.0,
            "sources": [],
            "hotrank": None,
            "mover": None,
        })
        rec["score"] += rank_pts
        rec["sources"].append("mover")
        rec["mover"] = {
            "change_pct": m.get("change_pct"),
            "premium_pct": m.get("premium_pct"),
        }

    # Cross-source bonus.
    for rec in candidates.values():
        unique_sources = sorted(set(rec["sources"]))
        rec["sources"] = unique_sources
        if len(unique_sources) > 1:
            rec["score"] += scoring["cross_source_bonus"]

    ranked = sorted(candidates.values(), key=lambda r: r["score"], reverse=True)
    top_n = int(preset.get("top_n", 5))
    top = ranked[:top_n]

    signals = [
        {
            "symbol": c["symbol"],
            "side": "long" if (c.get("mover") and (c["mover"].get("change_pct") or 0) > 0) else "watch",
            "score": round(c["score"], 2),
            "reason": (
                f"sources={','.join(c['sources'])}; "
                f"hotrank={c['hotrank']['rank'] if c['hotrank'] else 'n/a'}; "
                f"24h={c['mover']['change_pct'] if c['mover'] else 'n/a'}"
            ),
        }
        for c in top
    ]

    return {
        "signals": signals,
        "summary": (
            f"Ondo market: {market_status}. Universe: {len(universe)} tokens. "
            f"Hotrank top: {len(hotrank_top)}. Movers: {len(movers_top)}. "
            f"Top {len(top)} merged candidates."
        ),
        "recommended_next_step": (
            "Cross-reference with earnings-tradfi-scanner; "
            "use US market open/close timing to gate any execution."
        ),
        "ondo_market_status": market_status,
        "ondo_universe_size": len(universe),
        "candidates": top,
        "all_ranked": ranked,
    }
