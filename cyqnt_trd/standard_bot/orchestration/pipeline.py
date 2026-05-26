"""
cyqnt_trd.standard_bot.orchestration.pipeline
===============================================

Pipeline orchestration helpers — sequential workflow, two-stage scan/validate
pipeline, scan-rank-select, and generic candidate filter/rank.

Ported from atomic_strategy_lib.orchestration.pipeline (L8-01 to L8-04).

Design notes
------------
* Functions are **pure** (no I/O, no side effects) so they compose cleanly
  into pandas pipelines: ``sequential_workflow([fetch_data, compute_signals,
  score_entries])`` works naturally.
* ``Verdict`` is defined locally so this module is self-contained and usable
  without the full atomic type system.  It mirrors ``atomic.core.types.Verdict``
  exactly (same RANK dict) for 1:1 compatibility.
"""

from __future__ import annotations

from typing import Callable, Optional

__all__ = [
    "Verdict",
    "sequential_workflow",
    "two_stage_pipeline",
    "scan_rank_select",
    "candidate_filter_rank",
]


# ---------------------------------------------------------------------------
# Verdict class (mirrors atomic.core.types.Verdict)
# ---------------------------------------------------------------------------

class Verdict:
    """Signal verdict constants and their ranking order.

    Lower rank = better signal.

    Examples
    --------
    >>> Verdict.RANK["STRONG_CANDIDATE"]
    0
    >>> Verdict.RANK["SKIP"]
    3
    """

    STRONG_CANDIDATE = "STRONG_CANDIDATE"
    CANDIDATE = "CANDIDATE"
    WATCHLIST = "WATCHLIST"
    SKIP = "SKIP"
    AVOID = "AVOID"

    RANK: dict[str, int] = {
        "STRONG_CANDIDATE": 0,
        "CANDIDATE": 1,
        "WATCHLIST": 2,
        "SKIP": 3,
        "AVOID": 4,
    }


# ---------------------------------------------------------------------------
# L8-01  Sequential workflow
# ---------------------------------------------------------------------------

def sequential_workflow(
    steps: list[Callable],
    initial_data: dict,
) -> dict:
    """Execute a sequence of processing steps that each consume and return a dict.

    Each ``step`` is a callable ``(data: dict) -> dict``.  The output of step
    *N* becomes the input to step *N+1*.  Execution halts on the first
    exception; the error is recorded in ``_errors``.

    Parameters
    ----------
    steps:
        Ordered list of callables.  Works naturally with pandas-style
        pipeline functions::

            sequential_workflow(
                [fetch_data, compute_signals, score_entries],
                {"symbol": "BTCUSDT", "interval": "15m"},
            )

    initial_data:
        Seed dict passed to the first step.

    Returns
    -------
    dict
        The data dict after all steps (or after the first error), augmented
        with ``_steps_completed``, ``_errors``, and ``_completed``.
    """
    data = dict(initial_data)
    data.setdefault("_steps_completed", [])
    data.setdefault("_errors", [])

    for i, step in enumerate(steps):
        step_name = getattr(step, "__name__", f"step_{i}")
        try:
            result = step(data)
            # Allow steps that return None (no-op pass-through)
            if result is not None:
                data = result
            data.setdefault("_steps_completed", []).append(step_name)
        except Exception as exc:  # noqa: BLE001
            data.setdefault("_errors", []).append(
                {"step": step_name, "error": str(exc)}
            )
            break

    data["_completed"] = len(data.get("_errors", [])) == 0
    return data


# ---------------------------------------------------------------------------
# L8-02  Two-stage pipeline (scan → validate)
# ---------------------------------------------------------------------------

def two_stage_pipeline(
    scan_fn: Callable[[], list],
    validate_fn: Callable[[str], dict],
    max_candidates: int = 10,
) -> list[dict]:
    """Two-stage pipeline: scan for candidates then validate each one.

    Stage 1 runs ``scan_fn()`` to obtain a candidate list (strings or dicts
    with a ``"symbol"`` key).  Stage 2 runs ``validate_fn(symbol)`` for each
    candidate (up to *max_candidates*).

    Parameters
    ----------
    scan_fn:
        Zero-arg callable that returns a list of symbols (str) or dicts
        with a ``"symbol"`` key.
    validate_fn:
        Callable ``(symbol: str) -> dict``.
    max_candidates:
        Cap on how many symbols to validate.

    Returns
    -------
    list[dict]
        Collected results from ``validate_fn``.
    """
    raw_candidates = scan_fn()

    symbols: list[str] = []
    for cand in raw_candidates:
        if isinstance(cand, str):
            symbols.append(cand)
        elif isinstance(cand, dict):
            sym = cand.get("symbol", "")
            if sym:
                symbols.append(sym)
        if len(symbols) >= max_candidates:
            break

    results: list[dict] = []
    for sym in symbols:
        result = validate_fn(sym)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# L8-03  Scan → rank → select
# ---------------------------------------------------------------------------

def scan_rank_select(
    candidates: list[dict],
    score_key: str = "score",
    min_verdict: str = "CANDIDATE",
    max_select: int = 3,
) -> list[dict]:
    """Rank candidates by score, filter by minimum verdict, select top N.

    Parameters
    ----------
    candidates:
        List of candidate dicts.  Each must have ``"verdict"``,
        ``"direction"``, and the ``score_key`` field.
    score_key:
        Dict key to sort by (descending).
    min_verdict:
        Minimum acceptable verdict level.  Candidates with a worse verdict
        (higher rank number) are excluded.  Defaults to ``"CANDIDATE"``.
    max_select:
        Maximum number of candidates to return.

    Returns
    -------
    list[dict]
        Top-ranked qualifying candidates, sorted by *score_key* descending.
    """
    min_rank = Verdict.RANK.get(min_verdict, 1)

    qualifying = [
        c for c in candidates
        if Verdict.RANK.get(c.get("verdict", "SKIP"), 99) <= min_rank
        and c.get("direction", "NO_TRADE") in ("LONG", "SHORT")
    ]

    qualifying.sort(key=lambda c: -c.get(score_key, 0.0))
    return qualifying[:max_select]


# ---------------------------------------------------------------------------
# L8-04  Candidate filter and rank (generic)
# ---------------------------------------------------------------------------

def candidate_filter_rank(
    candidates: list[dict],
    filters: Optional[list[Callable[[dict], bool]]] = None,
    sort_key: str = "score",
    sort_reverse: bool = True,
    limit: int = 0,
) -> list[dict]:
    """Generic candidate filtering and ranking.

    Parameters
    ----------
    candidates:
        List of candidate dicts.
    filters:
        List of predicate functions; a candidate must satisfy **all**
        predicates to be kept.
    sort_key:
        Dict key to sort by.
    sort_reverse:
        ``True`` (default) = descending (highest first).
    limit:
        Maximum results (``0`` = unlimited).

    Returns
    -------
    list[dict]
        Filtered and sorted candidates.
    """
    result = list(candidates)

    if filters:
        for predicate in filters:
            result = [c for c in result if predicate(c)]

    result.sort(key=lambda c: c.get(sort_key, 0), reverse=sort_reverse)

    if limit > 0:
        result = result[:limit]

    return result
