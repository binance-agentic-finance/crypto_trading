"""
cyqnt_trd.standard_bot.monitoring.signals
==========================================

Signal tracking — record signals to JSONL, check outcomes, generate
performance reports.

Ported from atomic_strategy_lib.monitoring.signals (L7-03 to L7-05).
Function signatures are identical to atomic so existing call sites work
without modification.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["signal_record", "signal_outcome_check", "performance_report"]


# ---------------------------------------------------------------------------
# L7-03  Signal recording
# ---------------------------------------------------------------------------

def signal_record(
    symbol: str,
    direction: str,
    verdict: str,
    score: float,
    entry_price: float,
    stop_price: float = 0.0,
    stop_pct: float = 0.0,
    leverage: float = 0.0,
    notional: float = 0.0,
    max_loss: float = 0.0,
    metadata: dict | None = None,
    signals_file: str = "tmp/signals_log.jsonl",
) -> dict:
    """Record a trading signal to a JSONL file for later outcome tracking.

    Parameters
    ----------
    symbol:
        Trading pair, e.g. ``"BTCUSDT"``.
    direction:
        ``"LONG"`` or ``"SHORT"``.
    verdict:
        Verdict string from the scoring layer, e.g. ``"STRONG_CANDIDATE"``.
    score:
        Numeric composite score.
    entry_price:
        Entry price at signal time.
    stop_price:
        Absolute stop-loss price (0 = not set).
    stop_pct:
        Stop-loss distance as a percentage.
    leverage:
        Leverage used (0 = unknown).
    notional:
        Position size in quote currency (0 = unknown).
    max_loss:
        Maximum allowed loss in quote currency (0 = unknown).
    metadata:
        Optional extra fields to embed in the record.
    signals_file:
        Path to the JSONL log file.  Parent directories are created if
        they do not exist.

    Returns
    -------
    dict
        The signal dict that was written.
    """
    signal: dict = {
        "symbol": symbol,
        "direction": direction,
        "verdict": verdict,
        "score": score,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "stop_pct": stop_pct,
        "leverage": leverage,
        "notional": notional,
        "max_loss": max_loss,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "outcomes": {},
    }
    if metadata:
        signal["metadata"] = metadata

    path = Path(signals_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(signal, ensure_ascii=False) + "\n")

    return signal


# ---------------------------------------------------------------------------
# L7-04  Signal outcome check
# ---------------------------------------------------------------------------

def signal_outcome_check(
    entry_price: float,
    current_price: float,
    direction: str,
    leverage: float = 1.0,
    stop_price: float = 0.0,
) -> dict:
    """Compute hypothetical PnL for a recorded signal at a given mark price.

    Parameters
    ----------
    entry_price:
        Price at signal entry.
    current_price:
        Current or exit price to evaluate against.
    direction:
        ``"LONG"`` or ``"SHORT"``.
    leverage:
        Leverage multiplier.
    stop_price:
        Stop level — used to flag ``stop_hit`` (0 = not tracked).

    Returns
    -------
    dict with keys:
        ``price``             – current_price passed in
        ``pnl_pct``           – unleveraged % PnL (None if inputs invalid)
        ``pnl_leveraged_pct`` – leveraged % PnL
        ``stop_hit``          – bool or None
    """
    if entry_price <= 0 or current_price <= 0:
        return {"pnl_pct": None, "pnl_leveraged_pct": None, "stop_hit": None}

    direction_upper = direction.upper()
    if direction_upper == "LONG":
        pnl_pct = (current_price - entry_price) / entry_price * 100.0
        stop_hit = (current_price <= stop_price) if stop_price > 0 else None
    else:
        pnl_pct = (entry_price - current_price) / entry_price * 100.0
        stop_hit = (current_price >= stop_price) if stop_price > 0 else None

    return {
        "price": current_price,
        "pnl_pct": round(pnl_pct, 2),
        "pnl_leveraged_pct": round(pnl_pct * leverage, 2),
        "stop_hit": stop_hit,
    }


# ---------------------------------------------------------------------------
# L7-05  Performance report
# ---------------------------------------------------------------------------

def performance_report(signals: list[dict]) -> dict:
    """Generate performance statistics from recorded signals with outcomes.

    Each signal dict is expected to contain:
        ``direction``, ``verdict``, ``score``,
        ``outcomes``  – sub-dict keyed by horizon (``"current"``, ``"1h"``,
                        ``"4h"``, ``"24h"``); each value is a dict with a
                        ``"pnl_pct"`` key.

    The function picks the longest available horizon for each signal.

    Parameters
    ----------
    signals:
        List of signal dicts as produced by :func:`signal_record` and
        updated by callers with outcome data.

    Returns
    -------
    dict with keys:
        ``total``, ``evaluated``, ``win_rate``, ``wins``, ``losses``,
        ``avg_win``, ``avg_loss``, ``expectancy``,
        ``by_verdict``, ``by_direction``
    """
    if not signals:
        return {"total": 0, "evaluated": 0}

    total = len(signals)
    by_verdict: dict[str, dict] = {}
    by_direction: dict[str, dict] = {}
    wins = 0
    losses = 0
    win_pcts: list[float] = []
    loss_pcts: list[float] = []

    for sig in signals:
        verdict = sig.get("verdict", "?")
        direction = sig.get("direction", "?")
        outcomes = sig.get("outcomes", {})

        # Pick the longest available interval
        best_outcome = None
        for key in ("24h", "4h", "1h", "current"):
            if key in outcomes and outcomes[key].get("pnl_pct") is not None:
                best_outcome = outcomes[key]
                break

        if best_outcome is None:
            continue

        pnl: float = best_outcome["pnl_pct"]

        # Verdict bucket
        if verdict not in by_verdict:
            by_verdict[verdict] = {"count": 0, "wins": 0, "pnls": []}
        by_verdict[verdict]["count"] += 1
        by_verdict[verdict]["pnls"].append(pnl)

        # Direction bucket
        if direction not in by_direction:
            by_direction[direction] = {"count": 0, "wins": 0, "pnls": []}
        by_direction[direction]["count"] += 1
        by_direction[direction]["pnls"].append(pnl)

        if pnl > 0:
            wins += 1
            win_pcts.append(pnl)
            by_verdict[verdict]["wins"] += 1
            by_direction[direction]["wins"] += 1
        else:
            losses += 1
            loss_pcts.append(pnl)

    evaluated = wins + losses
    win_rate = round(wins / evaluated * 100.0, 1) if evaluated > 0 else 0.0
    avg_win = round(sum(win_pcts) / len(win_pcts), 2) if win_pcts else 0.0
    avg_loss = round(sum(loss_pcts) / len(loss_pcts), 2) if loss_pcts else 0.0
    expectancy = (
        round(
            (win_rate / 100.0 * avg_win) + ((1.0 - win_rate / 100.0) * avg_loss),
            2,
        )
        if evaluated > 0
        else 0.0
    )

    def _bucket_summary(bucket: dict) -> dict:
        count = bucket["count"]
        w = bucket["wins"]
        return {
            "count": count,
            "wins": w,
            "win_rate": round(w / count * 100.0, 1) if count > 0 else 0.0,
        }

    return {
        "total": total,
        "evaluated": evaluated,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "by_verdict": {k: _bucket_summary(v) for k, v in by_verdict.items()},
        "by_direction": {k: _bucket_summary(v) for k, v in by_direction.items()},
    }
