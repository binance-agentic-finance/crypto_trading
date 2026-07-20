"""Module ⑥ EXECUTION — same pattern as btc_multi_factor_trend/m06.

We deliberately duplicate the file rather than share via _shared/ because
execution semantics diverge over time (e.g. buzz screener may want a
different reject/retry policy). Duplication keeps the two demos
independently readable at 200 lines each.

If you find yourself editing this in both strategies for the same reason
3+ times, promote to _shared/execution_helpers.py.
"""
from __future__ import annotations

from atomic_strategy_lib.execution.position import (
    exchange_filter_fetch, quantize, round_to_tick, set_leverage,
)
from atomic_strategy_lib.execution.orders import market_order, stop_order
from atomic_strategy_lib.risk.limits import price_deviation_check
from atomic_strategy_lib.orchestration.state import position_create


_VERDICT_RANK = {"SKIP": 0, "AVOID": 0, "WATCHLIST": 1, "CANDIDATE": 2, "STRONG_CANDIDATE": 3}


def execute_all(results: list[dict], cfg: dict) -> dict:
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


def _place_one(row, mode, profile, live, max_dev):
    symbol, decision, entry = row["symbol"], row["decision"], row["price"]

    ok, cur_price = price_deviation_check(symbol, entry, max_dev, profile)
    if not ok:
        return {"status": "aborted",
                "reason": f"price moved > {max_dev}% ({entry} → {cur_price})"}

    filt = exchange_filter_fetch(symbol, market=mode, profile=profile) or {}
    qty = quantize(decision["size_usdt"] / entry, filt) if entry else 0
    stop_px = round_to_tick(entry * (1 - decision["stop_pct"]/100)
                             if decision["direction"]=="LONG"
                             else entry * (1 + decision["stop_pct"]/100),
                             filt)

    if not live:
        return {"status": "dry_run", "symbol": symbol,
                "side": decision["direction"], "qty": qty,
                "entry": entry, "stop": stop_px}

    if mode == "futures" and decision.get("leverage"):
        set_leverage(symbol, decision["leverage"], profile=profile)

    entry_res = market_order(symbol, side=decision["direction"],
                             qty=qty, market=mode, profile=profile)
    stop_res  = stop_order(symbol,
                           side="SELL" if decision["direction"]=="LONG" else "BUY",
                           qty=qty, stop_price=stop_px,
                           market=mode, profile=profile)
    position_create(symbol=symbol, side=decision["direction"], qty=qty,
                    entry=entry, stop=stop_px, mode=mode)
    return {"status": "filled", "symbol": symbol,
            "side": decision["direction"], "qty": qty,
            "entry": entry, "stop": stop_px,
            "orders": {"entry": entry_res, "stop": stop_res}}
