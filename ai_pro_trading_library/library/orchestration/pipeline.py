"""Pipeline / scheduler helpers — L2 parity vs `atomic.orchestration.pipeline` and `scheduler`."""

from __future__ import annotations

import time
from typing import Callable

from ai_pro_trading_library.library.scoring.atomic_compat import AtomicVerdict


# ---------------------------------------------------------------------------
# Sequential / two-stage pipeline
# ---------------------------------------------------------------------------


def sequential_workflow(steps: list[Callable], initial_data: dict) -> dict:
    """Execute a sequence of (data: dict) -> dict steps; record progress."""
    data = dict(initial_data)
    data["_steps_completed"] = []
    data["_errors"] = []
    for i, step in enumerate(steps):
        step_name = getattr(step, "__name__", f"step_{i}")
        try:
            data = step(data)
            data["_steps_completed"].append(step_name)
        except Exception as e:
            data["_errors"].append({"step": step_name, "error": str(e)})
            break
    data["_completed"] = len(data["_errors"]) == 0
    return data


def two_stage_pipeline(
    scan_fn: Callable[[], list],
    validate_fn: Callable[[str], dict],
    max_candidates: int = 10,
) -> list[dict]:
    """Two-stage pipeline: scan for candidates, then validate each."""
    candidates = scan_fn()
    symbols: list[str] = []
    for c in candidates:
        if isinstance(c, str):
            symbols.append(c)
        elif isinstance(c, dict):
            symbols.append(c.get("symbol", ""))
        if len(symbols) >= max_candidates:
            break
    results: list[dict] = []
    for sym in symbols:
        if not sym:
            continue
        result = validate_fn(sym)
        results.append(result)
    return results


def scan_rank_select(
    candidates: list[dict],
    score_key: str = "score",
    min_verdict: str = "CANDIDATE",
    max_select: int = 3,
) -> list[dict]:
    """Rank candidates by score, filter by minimum verdict, select top N."""
    min_rank = AtomicVerdict.RANK.get(min_verdict, 1)
    qualifying = [
        c
        for c in candidates
        if AtomicVerdict.RANK.get(c.get("verdict", "SKIP"), 99) <= min_rank
        and c.get("direction", "NO_TRADE") in ("LONG", "SHORT")
    ]
    qualifying.sort(key=lambda c: -c.get(score_key, 0))
    return qualifying[:max_select]


def candidate_filter_rank(
    candidates: list[dict],
    filters: list[Callable[[dict], bool]] | None = None,
    sort_key: str = "score",
    sort_reverse: bool = True,
    limit: int = 0,
) -> list[dict]:
    """Generic candidate filtering and ranking."""
    result = list(candidates)
    if filters:
        for f in filters:
            result = [c for c in result if f(c)]
    result.sort(key=lambda c: c.get(sort_key, 0), reverse=sort_reverse)
    if limit > 0:
        result = result[:limit]
    return result


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def fixed_interval_loop(
    fn: Callable[[], None],
    interval_seconds: int = 900,
    max_iterations: int = 0,
) -> int:
    """Run a function at fixed intervals; returns iterations completed."""
    iterations = 0
    while True:
        try:
            fn()
            iterations += 1
            if max_iterations > 0 and iterations >= max_iterations:
                break
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            break
    return iterations


def rescan_pass(
    positions: list[dict],
    price_fetcher: Callable[[str], float],
    trend_classifier: Callable[[str], str],
    decision_fn: Callable[[dict, float, str], dict],
    action_fn: Callable[[dict, dict, float], dict] | None = None,
) -> list[dict]:
    """Execute a single rescan pass over positions."""
    decisions: list[dict] = []
    for pos in positions:
        symbol = pos.get("symbol", "")
        if not symbol:
            continue
        current_price = price_fetcher(symbol)
        if current_price <= 0:
            decisions.append(
                {
                    "symbol": symbol,
                    "action": "SKIP",
                    "reason": "Cannot fetch price",
                }
            )
            continue
        trend = trend_classifier(symbol)
        decision = decision_fn(pos, current_price, trend)
        if action_fn and decision.get("action") not in ("HOLD", "SKIP"):
            action_result = action_fn(pos, decision, current_price)
            decision["action_result"] = action_result
        decision["symbol"] = symbol
        decision["current_price"] = current_price
        decision["trend"] = trend
        decisions.append(decision)
    return decisions


def rescan_loop(
    pass_fn: Callable[[], list[dict]],
    interval_seconds: int = 900,
    max_iterations: int = 0,
    on_complete: Callable[[list[dict]], None] | None = None,
) -> int:
    """Run rescan passes at fixed intervals."""
    iterations = 0
    while True:
        try:
            decisions = pass_fn()
            if on_complete:
                on_complete(decisions)
            iterations += 1
            if max_iterations > 0 and iterations >= max_iterations:
                break
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            break
    return iterations


__all__ = [
    "candidate_filter_rank",
    "fixed_interval_loop",
    "rescan_loop",
    "rescan_pass",
    "scan_rank_select",
    "sequential_workflow",
    "two_stage_pipeline",
]
