"""
cyqnt_trd.standard_bot.orchestration.scheduler
================================================

Scheduler loops — fixed-interval cron-like loop, rescan pass over positions,
and a combined rescan loop.

Ported from atomic_strategy_lib.orchestration.scheduler (L8-05 to L8-07).

Design notes
------------
* **Main-thread only** — no ``threading``.  Loops use ``time.sleep`` and
  break on ``KeyboardInterrupt``.
* ``rescan_pass`` is fully decoupled from data sources via callbacks, making
  it easy to test with mocks and reuse across different live / paper setups.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional

__all__ = ["fixed_interval_loop", "rescan_pass", "rescan_loop"]


# ---------------------------------------------------------------------------
# L8-05  Fixed interval loop
# ---------------------------------------------------------------------------

def fixed_interval_loop(
    fn: Callable[[], None],
    interval_seconds: int = 900,
    max_iterations: int = 0,
) -> int:
    """Run a function at fixed intervals.

    Parameters
    ----------
    fn:
        Zero-arg callable executed each iteration.
    interval_seconds:
        Sleep duration between iterations (default 15 min = 900 s).
    max_iterations:
        ``0`` = run indefinitely until ``KeyboardInterrupt``.

    Returns
    -------
    int
        Number of iterations completed before stopping.

    Example
    -------
    ::

        def run_scan():
            results = full_market_scan()
            print(results)

        fixed_interval_loop(run_scan, interval_seconds=300)  # every 5 min
    """
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


# ---------------------------------------------------------------------------
# L8-06  Rescan pass (single pass over open positions)
# ---------------------------------------------------------------------------

def rescan_pass(
    positions: list[dict],
    price_fetcher: Callable[[str], float],
    trend_classifier: Callable[[str], str],
    decision_fn: Callable[[dict, float, str], dict],
    action_fn: Optional[Callable[[dict, dict, float], dict]] = None,
) -> list[dict]:
    """Execute a single rescan pass over a list of open positions.

    All data-source dependencies are injected via callbacks so the function
    has no hard dependency on any particular data layer.

    Parameters
    ----------
    positions:
        List of position dicts (must include ``"symbol"``).
    price_fetcher:
        ``(symbol) -> current_price_float``.  Return ``<= 0`` to skip.
    trend_classifier:
        ``(symbol) -> trend_string``  e.g. ``"BULLISH"``, ``"BEARISH"``,
        ``"MIXED"``.
    decision_fn:
        ``(position, current_price, trend) -> decision_dict``.
        Must return a dict with at least an ``"action"`` key.
    action_fn:
        Optional ``(position, decision, current_price) -> result_dict``.
        Called only when ``decision["action"]`` is not ``"HOLD"`` or
        ``"SKIP"``.

    Returns
    -------
    list[dict]
        Decision dict for each position, augmented with ``"symbol"``,
        ``"current_price"``, and ``"trend"``.
    """
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
            decision["action_result"] = action_fn(pos, decision, current_price)

        decision["symbol"] = symbol
        decision["current_price"] = current_price
        decision["trend"] = trend
        decisions.append(decision)

    return decisions


# ---------------------------------------------------------------------------
# L8-07  Rescan loop
# ---------------------------------------------------------------------------

def rescan_loop(
    pass_fn: Callable[[], list[dict]],
    interval_seconds: int = 900,
    max_iterations: int = 0,
    on_complete: Optional[Callable[[list[dict]], None]] = None,
) -> int:
    """Run rescan passes at fixed intervals.

    Parameters
    ----------
    pass_fn:
        Zero-arg callable that executes one rescan pass and returns a list of
        decision dicts (e.g. a closure over :func:`rescan_pass`).
    interval_seconds:
        Sleep duration between passes.
    max_iterations:
        ``0`` = run indefinitely until ``KeyboardInterrupt``.
    on_complete:
        Optional callback invoked after each pass with the list of decisions.

    Returns
    -------
    int
        Number of passes completed before stopping.

    Example
    -------
    ::

        def one_pass():
            positions = position_open_list("state/positions.json")
            return rescan_pass(positions, get_price, classify_trend, make_decision)

        rescan_loop(one_pass, interval_seconds=300, on_complete=print)
    """
    iterations = 0

    while True:
        try:
            _now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            decisions = pass_fn()

            if on_complete is not None:
                on_complete(decisions)

            iterations += 1
            if max_iterations > 0 and iterations >= max_iterations:
                break

            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            break

    return iterations
