"""Module ⑥ EXECUTION — place market + stop orders for qualifying rows.

Both `cfg.execution.enabled` AND `cfg.execution.live` must be True for
real orders. Any one being False → dry-run (compute the plan, don't send).

Reads:
    cfg.execution.{enabled, live, mode, profile, min_verdict,
                   max_concurrent_positions, max_price_deviation_pct}
"""
from __future__ import annotations

from atomic_strategy_lib.execution.position import (
    exchange_filter_fetch, quantize, round_to_tick, set_leverage,
)
from atomic_strategy_lib.execution.orders import market_order, stop_order
from atomic_strategy_lib.risk.limits import price_deviation_check
from atomic_strategy_lib.orchestration.state import position_create


_VERDICT_RANK = {"SKIP": 0, "WATCHLIST": 1, "CANDIDATE": 2, "STRONG_CANDIDATE": 3}


def execute_all(results: list[dict], cfg: dict) -> dict:
    """Iterate `results`, dispatch orders for rows passing `min_verdict`.

    Returns a summary dict; individual per-order results attach to each
    result under `results[i]["execution"]`.
    """
    exc = cfg.get("execution", {})
    if not exc.get("enabled"):
        return {"skipped": True, "reason": "execution disabled"}

    live       = bool(exc.get("live"))
    mode       = exc.get("mode", "futures")
    profile    = exc.get("profile", "default")
    min_verdict = exc.get("min_verdict", "STRONG_CANDIDATE")
    max_pos    = int(exc.get("max_concurrent_positions", 3))
    max_dev    = float(exc.get("max_price_deviation_pct", 2.0))
    min_rank   = _VERDICT_RANK.get(min_verdict, 3)

    filled_count = 0
    executed = []

    for row in results:
        if filled_count >= max_pos:
            row["execution"] = {"status": "skipped", "reason": "max_positions"}
            continue
        if _VERDICT_RANK.get(row["score"]["verdict"], 0) < min_rank:
            row["execution"] = {"status": "skipped",
                                "reason": f"verdict<{min_verdict}"}
            continue
        if row["decision"]["direction"] == "WATCH":
            row["execution"] = {"status": "skipped", "reason": "watch-only"}
            continue

        outcome = _place_one(row, mode, profile, live, max_dev)
        row["execution"] = outcome
        executed.append(outcome)
        if outcome.get("status") == "filled":
            filled_count += 1

    return {"skipped": False, "live": live, "mode": mode,
            "count_filled": filled_count, "executed": executed}


def _place_one(row: dict, mode: str, profile: str, live: bool, max_dev: float) -> dict:
    """Send market + stop order pair. Dry-run when live=False."""
    symbol   = row["symbol"]
    decision = row["decision"]
    entry    = row["price"]

    # (a) price deviation guard — refuse if price moved > max_dev between
    # signal computation and order dispatch
    ok, cur_price = price_deviation_check(symbol, entry, max_dev, profile)
    if not ok:
        return {"status": "aborted",
                "reason": f"price moved > {max_dev}% ({entry} → {cur_price})"}

    # (b) fetch exchange filters and quantize size / stop
    filt = exchange_filter_fetch(symbol, market=mode, profile=profile) or {}
    qty  = quantize(decision["size_usdt"] / entry, filt)
    stop_px = round_to_tick(decision.get("stop_price") or 0, filt)

    if not live:
        return {
            "status": "dry_run",
            "symbol": symbol,
            "side":   decision["direction"],
            "qty":    qty,
            "entry":  entry,
            "stop":   stop_px,
        }

    # (c) set leverage (futures only)
    if mode == "futures" and decision.get("leverage"):
        set_leverage(symbol, decision["leverage"], profile=profile)

    # (d) market entry
    entry_res = market_order(symbol, side=decision["direction"],
                             qty=qty, market=mode, profile=profile)

    # (e) protective stop
    stop_res = stop_order(symbol, side="SELL" if decision["direction"]=="LONG" else "BUY",
                          qty=qty, stop_price=stop_px,
                          market=mode, profile=profile)

    # (f) record position
    position_create(symbol=symbol, side=decision["direction"], qty=qty,
                    entry=entry, stop=stop_px, mode=mode)

    return {
        "status":   "filled",
        "symbol":   symbol,
        "side":     decision["direction"],
        "qty":      qty,
        "entry":    entry,
        "stop":     stop_px,
        "orders":   {"entry": entry_res, "stop": stop_res},
    }
