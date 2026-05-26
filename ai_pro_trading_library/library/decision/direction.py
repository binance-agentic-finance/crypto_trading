"""Direction-determination helpers — L2 parity vs `atomic_strategy_lib.decision.direction`.

Both helpers return dicts with `direction`, `reasons`, `confidence` etc.
The exact threshold semantics, weights, and reason strings are preserved
to keep cross-checks against atomic deterministic.
"""

from __future__ import annotations


def trend_direction_determine(
    change_7d: float | None = None,
    change_24h: float | None = None,
    funding_rate: float | None = None,
    sentiment_en: str | None = None,
    sentiment_cn: str | None = None,
    short_funding_threshold: float = 0.0003,
) -> dict:
    """Determine LONG/SHORT/WATCH/NO_TRADE from trend + funding + sentiment."""
    reasons: list[str] = []
    confidence = 0.0

    is_short = (
        change_7d is not None and change_7d < 0
        and funding_rate is not None and funding_rate > short_funding_threshold
        and sentiment_en == "short_bias"
        and sentiment_cn == "short_bias"
    )
    if is_short:
        reasons.append(
            "7d %.1f%% down + funding %.4f%% crowded + both bearish → SHORT"
            % (change_7d, (funding_rate or 0) * 100)
        )
        return {"direction": "SHORT", "reasons": reasons, "confidence": 0.8}

    if change_7d is not None and change_7d > 0:
        reasons.append("7d +%.1f%% uptrend → LONG" % change_7d)
        confidence = 0.6
        if funding_rate is not None and funding_rate < -0.0001:
            reasons.append("Funding %.4f%% squeeze supports long" % (funding_rate * 100))
            confidence += 0.1
        return {"direction": "LONG", "reasons": reasons, "confidence": round(confidence, 2)}

    if funding_rate is not None and funding_rate < -0.0001:
        reasons.append("Funding %.4f%% squeeze favors longs" % (funding_rate * 100))
        return {"direction": "LONG", "reasons": reasons, "confidence": 0.5}

    if change_24h is not None and change_24h > 5:
        reasons.append("24h +%.1f%% fresh momentum → LONG" % change_24h)
        return {"direction": "LONG", "reasons": reasons, "confidence": 0.4}

    reasons.append("No clear directional signal → WATCH")
    return {"direction": "WATCH", "reasons": reasons, "confidence": 0.0}


def direction_from_multi_factor(
    trend_signal: str | None = None,
    momentum_signal: str | None = None,
    volume_signal: str | None = None,
    derivatives_signal: str | None = None,
    structure_signal: str | None = None,
) -> dict:
    """Vote multiple categorical signals; weighted by name."""
    weights = {
        "trend": 2.0,
        "momentum": 1.5,
        "volume": 1.0,
        "derivatives": 1.5,
        "structure": 1.0,
    }
    inputs = {
        "trend": trend_signal,
        "momentum": momentum_signal,
        "volume": volume_signal,
        "derivatives": derivatives_signal,
        "structure": structure_signal,
    }

    bullish_weight = 0.0
    bearish_weight = 0.0
    total_weight = 0.0
    reasons: list[str] = []

    for name, signal in inputs.items():
        if signal is None:
            continue
        w = weights[name]
        total_weight += w
        if signal == "BULLISH":
            bullish_weight += w
            reasons.append("%s: BULLISH (×%.1f)" % (name, w))
        elif signal == "BEARISH":
            bearish_weight += w
            reasons.append("%s: BEARISH (×%.1f)" % (name, w))
        else:
            reasons.append("%s: NEUTRAL" % name)

    if total_weight == 0:
        return {
            "direction": "NO_TRADE",
            "agreement_pct": 0,
            "confidence": 0,
            "reasons": ["No signals"],
        }

    if bullish_weight > bearish_weight:
        direction = "LONG"
        agreement = bullish_weight / total_weight
    elif bearish_weight > bullish_weight:
        direction = "SHORT"
        agreement = bearish_weight / total_weight
    else:
        direction = "WATCH"
        agreement = 0.5

    confidence = agreement * 0.8

    return {
        "direction": direction,
        "agreement_pct": round(agreement * 100, 1),
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "bullish_weight": bullish_weight,
        "bearish_weight": bearish_weight,
    }


__all__ = ["direction_from_multi_factor", "trend_direction_determine"]
