"""Atomic-shape derivatives detectors — L2 parity vs `atomic.signals.derivatives`.

Returns dicts with `direction` / `is_extreme` / `strength` keys for the
atomic decision pipeline. The pandas-Series equivalents live in
`library.features.derivatives` (e.g. `funding_rate_state`).
"""

from __future__ import annotations


def funding_extreme_detect(
    rate: float = 0.0,
    squeeze_threshold: float = -0.0001,
    crowded_threshold: float = 0.0005,
    threshold: float | None = None,
) -> dict:
    """Classify funding rate. Accepts float or duck-typed FundingRate."""
    if hasattr(rate, "rate"):
        rate = rate.rate
    if threshold is not None:
        squeeze_threshold = -abs(threshold)
        crowded_threshold = abs(threshold)
    if rate < squeeze_threshold:
        distance = abs(rate - squeeze_threshold)
        strength = (
            min(distance / abs(squeeze_threshold) * 0.5, 1.0)
            if squeeze_threshold != 0
            else 1.0
        )
        return {
            "zone": "SQUEEZE",
            "signal_direction": "BULLISH",
            "direction": "BULLISH",
            "is_extreme": True,
            "strength": round(strength, 2),
            "rate": rate,
        }
    if rate < 0:
        return {
            "zone": "MILD_SQUEEZE",
            "signal_direction": "BULLISH",
            "direction": "BULLISH",
            "is_extreme": False,
            "strength": 0.3,
            "rate": rate,
        }
    if rate > crowded_threshold:
        distance = rate - crowded_threshold
        strength = (
            min(distance / crowded_threshold * 0.5, 1.0)
            if crowded_threshold != 0
            else 1.0
        )
        return {
            "zone": "CROWDED",
            "signal_direction": "BEARISH",
            "direction": "BEARISH",
            "is_extreme": True,
            "strength": round(strength, 2),
            "rate": rate,
        }
    return {
        "zone": "NEUTRAL",
        "signal_direction": "NEUTRAL",
        "direction": "NEUTRAL",
        "is_extreme": False,
        "strength": 0.0,
        "rate": rate,
    }


def oi_anomaly_detect(history: list, threshold_pct: float = 10.0) -> dict:
    """Classify OI anomaly from oldest→newest history (duck-typed `.oi_value`)."""
    if len(history) < 2:
        return {"is_anomaly": False, "delta_pct": 0.0, "direction": "STABLE"}
    oldest = history[0].oi_value
    newest = history[-1].oi_value
    if oldest <= 0:
        return {"is_anomaly": False, "delta_pct": 0.0, "direction": "STABLE"}
    delta_pct = (newest - oldest) / oldest * 100
    if delta_pct > threshold_pct:
        direction = "BUILDING"
        is_anomaly = True
    elif delta_pct < -threshold_pct:
        direction = "UNWINDING"
        is_anomaly = True
    else:
        direction = "STABLE"
        is_anomaly = False
    acceleration = None
    if len(history) >= 4:
        mid = len(history) // 2
        first_half_change = (
            (history[mid].oi_value - history[0].oi_value) / history[0].oi_value * 100
            if history[0].oi_value > 0
            else 0
        )
        second_half_change = (
            (history[-1].oi_value - history[mid].oi_value) / history[mid].oi_value * 100
            if history[mid].oi_value > 0
            else 0
        )
        if first_half_change != 0:
            acceleration = (
                round(second_half_change / first_half_change, 2)
                if first_half_change != 0
                else None
            )
    return {
        "is_anomaly": is_anomaly,
        "delta_pct": round(delta_pct, 2),
        "direction": direction,
        "oldest_oi": oldest,
        "newest_oi": newest,
        "acceleration": acceleration,
    }


def crowding_detect(
    funding_rate: float | None = None,
    long_short_ratio: float | None = None,
    oi_delta_pct: float | None = None,
    funding_crowded_threshold: float = 0.0005,
    ls_crowded_threshold: float = 2.0,
    oi_building_threshold: float = 10.0,
) -> dict:
    """Multi-factor crowding detector across funding / LS-ratio / OI delta."""
    signals: list[str] = []
    long_crowding = 0
    short_crowding = 0
    if funding_rate is not None:
        if funding_rate > funding_crowded_threshold:
            long_crowding += 1
            signals.append(
                "Funding %.4f%% indicates long crowding" % (funding_rate * 100)
            )
        elif funding_rate < -funding_crowded_threshold:
            short_crowding += 1
            signals.append(
                "Funding %.4f%% indicates short crowding" % (funding_rate * 100)
            )
    if long_short_ratio is not None:
        if long_short_ratio > ls_crowded_threshold:
            long_crowding += 1
            signals.append("L/S ratio %.2f indicates long crowding" % long_short_ratio)
        elif long_short_ratio < 1.0 / ls_crowded_threshold:
            short_crowding += 1
            signals.append("L/S ratio %.2f indicates short crowding" % long_short_ratio)
    if oi_delta_pct is not None:
        if oi_delta_pct > oi_building_threshold:
            signals.append("OI building +%.1f%% — positions accumulating" % oi_delta_pct)
        elif oi_delta_pct < -oi_building_threshold:
            signals.append("OI unwinding %.1f%% — positions closing" % oi_delta_pct)
    total_signals = long_crowding + short_crowding
    if long_crowding > short_crowding and long_crowding >= 1:
        return {
            "is_crowded": True,
            "crowded_direction": "LONG",
            "confidence": round(long_crowding / max(total_signals, 2), 2),
            "signals": signals,
        }
    if short_crowding > long_crowding and short_crowding >= 1:
        return {
            "is_crowded": True,
            "crowded_direction": "SHORT",
            "confidence": round(short_crowding / max(total_signals, 2), 2),
            "signals": signals,
        }
    return {
        "is_crowded": False,
        "crowded_direction": "NONE",
        "confidence": 0.0,
        "signals": signals,
    }


__all__ = ["crowding_detect", "funding_extreme_detect", "oi_anomaly_detect"]
