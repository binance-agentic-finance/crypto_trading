"""Direction determination and multi-factor decision blocks.

Ported from atomic_strategy_lib/decision/direction.py (L4-01 / L4-02).

Functions
---------
* :func:`trend_direction_determine` — LONG/SHORT/WATCH from trend + market signals
* :func:`direction_from_multi_factor` — weighted categorical voting across signal dimensions

Examples
--------
>>> from cyqnt_trd.blocks import decision
>>> result = decision.trend_direction_determine(change_7d=15.0, change_24h=3.0, funding_rate=-0.0002)
>>> result["direction"]
'LONG'
>>> vote = decision.direction_from_multi_factor(
...     trend_signal="BULLISH", momentum_signal="BULLISH", volume_signal="NEUTRAL"
... )
>>> vote["direction"]
'LONG'
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "trend_direction_determine",
    "direction_from_multi_factor",
]


# ---------------------------------------------------------------------------
# L4-01  Trend-based direction determination
# ---------------------------------------------------------------------------

def trend_direction_determine(
    change_7d: Optional[float] = None,
    change_24h: Optional[float] = None,
    funding_rate: Optional[float] = None,
    sentiment_en: Optional[str] = None,
    sentiment_cn: Optional[str] = None,
    short_funding_threshold: float = 0.0003,
) -> dict:
    """Determine trade direction from trend and market signals.

    Ported 1-to-1 from ``atomic_strategy_lib.decision.direction.trend_direction_determine``.

    Priority order (first match wins):

    1. **SHORT** — 7-day negative AND funding rate crowded (> *short_funding_threshold*)
       AND both sentiment inputs bearish.
    2. **LONG** — 7-day uptrend.  Confidence boosted if funding squeeze detected.
    3. **LONG** — Funding squeeze only (negative funding rate).
    4. **LONG** — Fresh 24-hour momentum (> 5 %).
    5. **WATCH** — No clear directional signal.

    Parameters
    ----------
    change_7d:
        7-day price change percentage (positive = bullish).
    change_24h:
        24-hour price change percentage.
    funding_rate:
        Current perpetual funding rate (e.g. 0.0003 = 0.03 %).
        Negative values indicate funding squeeze (longs paid by shorts).
    sentiment_en, sentiment_cn:
        Social sentiment label. Expected values: ``"short_bias"``, ``"long_bias"``,
        ``"neutral"`` (or None to skip).
    short_funding_threshold:
        Funding rate threshold above which the market is considered crowded long
        (used in the SHORT condition).

    Returns
    -------
    dict with keys:

    * ``direction`` — ``"LONG"``, ``"SHORT"``, ``"WATCH"``, or ``"NO_TRADE"``
    * ``reasons`` — list of human-readable reasoning strings
    * ``confidence`` — float in [0, 1]
    """
    reasons: list[str] = []
    confidence = 0.0

    # SHORT: 7d down + crowded long funding + both bearish sentiments
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
        confidence = 0.8
        return {"direction": "SHORT", "reasons": reasons, "confidence": confidence}

    # LONG: 7-day uptrend
    if change_7d is not None and change_7d > 0:
        reasons.append("7d +%.1f%% uptrend → LONG" % change_7d)
        confidence = 0.6
        if funding_rate is not None and funding_rate < -0.0001:
            reasons.append("Funding %.4f%% squeeze supports long" % (funding_rate * 100))
            confidence += 0.1
        return {"direction": "LONG", "reasons": reasons, "confidence": round(confidence, 2)}

    # LONG: funding squeeze
    if funding_rate is not None and funding_rate < -0.0001:
        reasons.append("Funding %.4f%% squeeze favors longs" % (funding_rate * 100))
        confidence = 0.5
        return {"direction": "LONG", "reasons": reasons, "confidence": confidence}

    # LONG: 24h fresh momentum
    if change_24h is not None and change_24h > 5:
        reasons.append("24h +%.1f%% fresh momentum → LONG" % change_24h)
        confidence = 0.4
        return {"direction": "LONG", "reasons": reasons, "confidence": confidence}

    # No clear signal
    reasons.append("No clear directional signal → WATCH")
    return {"direction": "WATCH", "reasons": reasons, "confidence": 0.0}


# ---------------------------------------------------------------------------
# L4-02  Multi-factor direction from categorical signals
# ---------------------------------------------------------------------------

def direction_from_multi_factor(
    trend_signal: Optional[str] = None,
    momentum_signal: Optional[str] = None,
    volume_signal: Optional[str] = None,
    derivatives_signal: Optional[str] = None,
    structure_signal: Optional[str] = None,
) -> dict:
    """Determine direction from multiple categorical signal inputs via weighted voting.

    Ported 1-to-1 from ``atomic_strategy_lib.decision.direction.direction_from_multi_factor``.

    Each input should be ``"BULLISH"``, ``"BEARISH"``, or ``"NEUTRAL"`` (or ``None``
    to exclude that dimension from voting).

    Weights used:

    * trend: 2.0
    * momentum: 1.5
    * derivatives: 1.5
    * volume: 1.0
    * structure: 1.0

    Parameters
    ----------
    trend_signal, momentum_signal, volume_signal, derivatives_signal, structure_signal:
        Categorical signal strings or None.

    Returns
    -------
    dict with keys:

    * ``direction`` — ``"LONG"``, ``"SHORT"``, ``"WATCH"``, or ``"NO_TRADE"``
    * ``agreement_pct`` — percentage of total weight on winning side
    * ``confidence`` — ``agreement_pct * 0.8`` (max 80 % from voting alone)
    * ``reasons`` — list of per-factor labels
    * ``bullish_weight``, ``bearish_weight``
    """
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
            "bullish_weight": 0.0,
            "bearish_weight": 0.0,
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

    confidence = agreement * 0.8  # max 80 % from factor voting alone

    return {
        "direction": direction,
        "agreement_pct": round(agreement * 100, 1),
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "bullish_weight": bullish_weight,
        "bearish_weight": bearish_weight,
    }
