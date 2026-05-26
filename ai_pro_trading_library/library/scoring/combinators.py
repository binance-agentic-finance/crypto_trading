"""Signal combinators for scoring-layer inputs."""

from __future__ import annotations

from typing import Iterable, Mapping

from ai_pro_trading_library.library.core.protocols import Signal, SignalEnvelope


def weighted_score(signals: Iterable[Signal]) -> float:
    return sum(signal.weight for signal in signals if signal.passed)


def envelope_score(envelope: SignalEnvelope) -> float:
    return weighted_score(envelope.signals)


def additive_combine(signals: Iterable[Signal]) -> float:
    """Plain sum of `signal.weight` over passed signals."""
    return weighted_score(signals)


def weighted_composite(
    signals: Iterable[Signal],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Per-feature weighted composite.

    If `weights` is None, falls back to `additive_combine`. Otherwise each
    passed signal's contribution is `signal.weight * weights[signal.feature_name]`.
    Missing features get weight 0 (they are ignored).
    """
    sigs = list(signals)
    if weights is None:
        return weighted_score(sigs)
    return float(
        sum(
            signal.weight * float(weights.get(signal.feature_name, 0.0))
            for signal in sigs
            if signal.passed
        )
    )
