"""Atomic-shape scoring helpers — Verdict ladder, gates, combinators.

Sourced from ``ai_pro_trading_library.library.scoring.atomic_compat`` to fill
the gap identified in the atomic→cyqnt_trd audit. These are dict-out / atomic
``Signal``-style helpers, complementing the pandas-vectorized combinators in
:mod:`cyqnt_trd.blocks.scoring`.

Use this module when:

* You need the **5-tier Verdict ladder**
  (``STRONG_CANDIDATE`` / ``CANDIDATE`` / ``WATCHLIST`` / ``SKIP`` / ``AVOID``).
* Your inputs are :class:`AtomicSignal` (``name`` / ``value`` / ``direction`` /
  ``strength``) rather than ``pd.Series``.
* You want one-shot ``hard_gate``, ``soft_factor``, ``cross_validate``,
  ``conflict_detect``, or ``normalize_score`` helpers.

For pandas-vectorized scoring across an entire DataFrame, use
:mod:`cyqnt_trd.blocks.scoring` (``ScoringSystem``, module-level
``additive_combine`` / ``weighted_composite`` / etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Atomic-style schema (mirrors atomic_strategy_lib.core.types but standalone)
# ---------------------------------------------------------------------------


@dataclass
class AtomicSignal:
    """Atomic-shape signal: name + value + direction + strength."""

    name: str
    value: float
    direction: str = "NEUTRAL"  # "BULLISH" / "BEARISH" / "NEUTRAL"
    strength: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AtomicScore:
    """Atomic-shape score: numeric value + reason list + factor breakdown."""

    value: float
    reasons: list = field(default_factory=list)
    factors: dict = field(default_factory=dict)


class AtomicVerdict:
    """Verdict ladder constants + RANK lookup (lower = better)."""

    STRONG_CANDIDATE = "STRONG_CANDIDATE"
    CANDIDATE = "CANDIDATE"
    WATCHLIST = "WATCHLIST"
    SKIP = "SKIP"
    AVOID = "AVOID"

    RANK: Dict[str, int] = {
        "STRONG_CANDIDATE": 0,
        "CANDIDATE": 1,
        "WATCHLIST": 2,
        "SKIP": 3,
        "AVOID": 4,
    }


# ---------------------------------------------------------------------------
# Combinators (atomic Signal in / AtomicScore out)
# ---------------------------------------------------------------------------


def additive_combine(signals: List[AtomicSignal]) -> AtomicScore:
    """Sum signals with direction sign. Bullish adds, bearish subtracts."""
    total = 0.0
    reasons: List[str] = []
    factors: Dict[str, float] = {}
    for sig in signals:
        if sig.direction == "BULLISH":
            contribution = sig.strength * sig.value
        elif sig.direction == "BEARISH":
            contribution = -sig.strength * abs(sig.value)
        else:
            contribution = 0.0
        total += contribution
        factors[sig.name] = contribution
        if contribution != 0:
            reasons.append(
                "%s: %+.2f (%s, strength=%.2f)"
                % (sig.name, contribution, sig.direction, sig.strength)
            )
    return AtomicScore(value=round(total, 2), reasons=reasons, factors=factors)


def sequential_filter(
    signals: List[AtomicSignal],
    required_bullish: int = 0,
    required_bearish: int = 0,
    min_strength: float = 0.3,
) -> AtomicScore:
    """Count signals passing min_strength; fire when bullish/bearish quota met."""
    bullish_count = 0
    bearish_count = 0
    reasons: List[str] = []
    for sig in signals:
        if sig.strength < min_strength:
            reasons.append(
                "%s: strength %.2f below threshold %.2f — filtered out"
                % (sig.name, sig.strength, min_strength)
            )
            continue
        if sig.direction == "BULLISH":
            bullish_count += 1
        elif sig.direction == "BEARISH":
            bearish_count += 1
    passed = False
    if required_bullish > 0 and bullish_count >= required_bullish:
        score_val = float(bullish_count)
        reasons.append(
            "Sequential: %d bullish signals pass (need %d)"
            % (bullish_count, required_bullish)
        )
        passed = True
    elif required_bearish > 0 and bearish_count >= required_bearish:
        score_val = -float(bearish_count)
        reasons.append(
            "Sequential: %d bearish signals pass (need %d)"
            % (bearish_count, required_bearish)
        )
        passed = True
    else:
        score_val = 0.0
        reasons.append(
            "Sequential: %d bullish, %d bearish — insufficient"
            % (bullish_count, bearish_count)
        )
    return AtomicScore(
        value=score_val,
        reasons=reasons,
        factors={
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "passed": int(passed),
        },
    )


def weighted_composite(
    signals: List[AtomicSignal],
    weights: Optional[Dict[str, float]] = None,
) -> AtomicScore:
    """Weighted sum of atomic-shaped signals."""
    if weights is None:
        weights = {}
    total = 0.0
    reasons: List[str] = []
    factors: Dict[str, float] = {}
    for sig in signals:
        w = weights.get(sig.name, 1.0)
        if sig.direction == "BULLISH":
            contribution = w * sig.value
        elif sig.direction == "BEARISH":
            contribution = -w * abs(sig.value)
        else:
            contribution = 0.0
        total += contribution
        factors[sig.name] = contribution
        if contribution != 0:
            reasons.append(
                "%s: %+.2f (weight=%.1f, %s)"
                % (sig.name, contribution, w, sig.direction)
            )
    return AtomicScore(value=round(total, 2), reasons=reasons, factors=factors)


def majority_vote(signals: List[AtomicSignal], min_strength: float = 0.0) -> AtomicScore:
    """Direction by majority vote. Score = (bullish - bearish) / total."""
    bullish = 0
    bearish = 0
    neutral = 0
    for sig in signals:
        if sig.strength < min_strength:
            continue
        if sig.direction == "BULLISH":
            bullish += 1
        elif sig.direction == "BEARISH":
            bearish += 1
        else:
            neutral += 1
    total = bullish + bearish + neutral
    if total == 0:
        return AtomicScore(value=0, reasons=["No qualifying signals"], factors={})
    net = bullish - bearish
    score = net / total
    if net > 0:
        direction = "BULLISH"
    elif net < 0:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"
    return AtomicScore(
        value=round(score, 2),
        reasons=[
            "%s majority: %d bullish, %d bearish, %d neutral"
            % (direction, bullish, bearish, neutral)
        ],
        factors={
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "direction": direction,
        },
    )


def hierarchical_combine(
    tiers_or_signals: list,
    weights: Optional[Dict[str, float]] = None,
) -> AtomicScore:
    """Hierarchical tiered scoring. Two calling conventions:

    1. ``hierarchical_combine(tiers)`` where
       ``tiers = [(name, signals, weight), ...]``
    2. ``hierarchical_combine(signals, weights={...})`` — delegates to
       :func:`weighted_composite`.
    """
    is_signal_list = bool(tiers_or_signals) and hasattr(tiers_or_signals[0], "direction")
    if weights is not None or is_signal_list:
        return weighted_composite(tiers_or_signals, weights)
    tiers = tiers_or_signals
    total = 0.0
    reasons: List[str] = []
    factors: Dict[str, float] = {}
    for tier_name, signals, tier_weight in tiers:
        tier_score = 0.0
        tier_reasons: List[str] = []
        for sig in signals:
            if sig.direction == "BULLISH":
                contribution = sig.value
            elif sig.direction == "BEARISH":
                contribution = -abs(sig.value)
            else:
                contribution = 0.0
            tier_score += contribution
            if contribution != 0:
                tier_reasons.append("%s=%+.1f" % (sig.name, contribution))
        weighted = tier_score * tier_weight
        total += weighted
        factors[tier_name] = weighted
        detail = ", ".join(tier_reasons) if tier_reasons else "no signal"
        reasons.append(
            "%s: %+.1f × %.1f = %+.1f [%s]"
            % (tier_name, tier_score, tier_weight, weighted, detail)
        )
    return AtomicScore(value=round(total, 2), reasons=reasons, factors=factors)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def hard_gate(
    value: float,
    threshold: float,
    name: str = "gate",
    comparison: str = "gte",
) -> dict:
    """Binary pass/fail gate.

    *comparison* in ``{"gte", "lte", "gt", "lt"}``.
    """
    ops = {
        "gte": lambda v, t: v >= t,
        "lte": lambda v, t: v <= t,
        "gt": lambda v, t: v > t,
        "lt": lambda v, t: v < t,
    }
    op = ops.get(comparison, ops["gte"])
    passed = op(value, threshold)
    return {
        "passed": passed,
        "name": name,
        "value": value,
        "threshold": threshold,
        "reason": "%s: %.4g %s %.4g → %s"
        % (name, value, comparison, threshold, "PASS" if passed else "FAIL"),
    }


def enum_gate(value: Any, allowed: Iterable, name: str = "gate") -> dict:
    """Pass/fail gate for enum-like values without overloading numeric :func:`hard_gate`."""
    allowed_values = tuple(allowed)
    passed = value in allowed_values
    return {
        "passed": passed,
        "name": name,
        "value": value,
        "allowed": allowed_values,
        "reason": "%s: %s in %s → %s"
        % (name, value, allowed_values, "PASS" if passed else "FAIL"),
    }


def soft_factor(
    value: float,
    thresholds: List[Tuple[float, float, str]],
    name: str = "factor",
) -> dict:
    """Tiered scoring factor; first matching threshold wins.

    *thresholds* is a list of ``(threshold, score_contribution, label)`` tuples,
    ordered from highest threshold to lowest.
    """
    for threshold, score, label in thresholds:
        if value >= threshold:
            return {
                "score": score,
                "label": label,
                "name": name,
                "reason": "%s: %.4g >= %.4g → %s (%+.1f)"
                % (name, value, threshold, label, score),
            }
    return {
        "score": 0.0,
        "label": "BELOW_ALL",
        "name": name,
        "reason": "%s: %.4g below all thresholds" % (name, value),
    }


def verdict_classify(
    score: float,
    strong_min: float = 8.0,
    candidate_min: float = 5.0,
    watchlist_min: float = 2.0,
    skip_min: float = 0.0,
) -> str:
    """Classify a numeric score into a verdict string from :class:`AtomicVerdict`."""
    if score >= strong_min:
        return AtomicVerdict.STRONG_CANDIDATE
    if score >= candidate_min:
        return AtomicVerdict.CANDIDATE
    if score >= watchlist_min:
        return AtomicVerdict.WATCHLIST
    if score >= skip_min:
        return AtomicVerdict.SKIP
    return AtomicVerdict.AVOID


def verdict_with_gate(
    score: float,
    gates: List[dict],
    cap_verdict: str = "WATCHLIST",
    strong_min: float = 8.0,
    candidate_min: float = 5.0,
    watchlist_min: float = 2.0,
    skip_min: float = 0.0,
) -> dict:
    """Verdict classify with gate enforcement; cap if any gate failed."""
    raw_verdict = verdict_classify(
        score, strong_min, candidate_min, watchlist_min, skip_min
    )
    failed_gates = [g for g in gates if not g.get("passed", True)]
    if failed_gates and AtomicVerdict.RANK.get(raw_verdict, 99) < AtomicVerdict.RANK.get(
        cap_verdict, 99
    ):
        return {
            "verdict": cap_verdict,
            "raw_verdict": raw_verdict,
            "gated": True,
            "failed_gates": [g["name"] for g in failed_gates],
            "reason": "Verdict capped from %s to %s due to: %s"
            % (
                raw_verdict,
                cap_verdict,
                ", ".join(g["reason"] for g in failed_gates),
            ),
        }
    return {
        "verdict": raw_verdict,
        "raw_verdict": raw_verdict,
        "gated": False,
        "failed_gates": [],
        "reason": "Score %.1f → %s" % (score, raw_verdict),
    }


def cross_validate(scores: List[AtomicScore]) -> dict:
    """Check if multiple :class:`AtomicScore` objects agree on direction."""
    if not scores:
        return {"agreed": False, "direction": "NEUTRAL", "agreement_pct": 0.0}
    directions: List[str] = []
    for s in scores:
        if s.value > 0:
            directions.append("BULLISH")
        elif s.value < 0:
            directions.append("BEARISH")
        else:
            directions.append("NEUTRAL")
    counts: Dict[str, int] = {}
    for d in directions:
        counts[d] = counts.get(d, 0) + 1
    dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
    agreement = counts[dominant] / len(directions)
    return {
        "agreed": agreement >= 0.66,
        "direction": dominant,
        "agreement_pct": round(agreement * 100, 1),
        "breakdown": dict(counts),
    }


def conflict_detect(signals_a: List[dict], signals_b: List[dict]) -> dict:
    """Detect direction conflicts between two sets of dict-shaped signals."""
    conflicts: List[Tuple[dict, dict]] = []
    for a in signals_a:
        for b in signals_b:
            a_dir = a.get("direction", "NEUTRAL")
            b_dir = b.get("direction", "NEUTRAL")
            if (a_dir == "BULLISH" and b_dir == "BEARISH") or (
                a_dir == "BEARISH" and b_dir == "BULLISH"
            ):
                conflicts.append((a, b))
    return {
        "has_conflict": len(conflicts) > 0,
        "conflict_count": len(conflicts),
        "conflicting_pairs": [
            {"a": a.get("name", str(a)), "b": b.get("name", str(b))}
            for a, b in conflicts[:5]
        ],
    }


def normalize_score(
    score: float,
    min_val: float = -20.0,
    max_val: float = 20.0,
    target_min: float = 0.0,
    target_max: float = 100.0,
) -> float:
    """Linearly normalize *score* to ``[target_min, target_max]``, clamping outside."""
    clamped = max(min_val, min(score, max_val))
    if max_val == min_val:
        return (target_min + target_max) / 2
    normalized = (clamped - min_val) / (max_val - min_val)
    return round(target_min + normalized * (target_max - target_min), 2)


__all__ = [
    # Schema
    "AtomicScore",
    "AtomicSignal",
    "AtomicVerdict",
    # Combinators
    "additive_combine",
    "hierarchical_combine",
    "majority_vote",
    "sequential_filter",
    "weighted_composite",
    # Gates / verdict
    "conflict_detect",
    "cross_validate",
    "enum_gate",
    "hard_gate",
    "normalize_score",
    "soft_factor",
    "verdict_classify",
    "verdict_with_gate",
]
