"""Atomic volume detectors — L2 parity vs `atomic.signals.volume`."""

from __future__ import annotations


def volume_surge_detect(
    candles: list,
    lookback: int = 20,
    surge_threshold: float = 2.0,
    multiplier: float | None = None,
) -> dict:
    """Detect volume surge vs prior `lookback` average. Accepts Candle list or floats."""
    if multiplier is not None:
        surge_threshold = multiplier
    if len(candles) < lookback + 1:
        return {"is_surge": False, "volume_ratio": None, "ratio": None}
    if candles and hasattr(candles[0], "volume"):
        vols = [
            c.quote_volume if c.quote_volume > 0 else c.volume for c in candles
        ]
    else:
        vols = [float(v) for v in candles]
    latest = vols[-1]
    prior = vols[-(lookback + 1) : -1]
    avg_vol = sum(prior) / len(prior) if prior else 1
    ratio = latest / avg_vol if avg_vol > 0 else 0
    return {
        "is_surge": ratio >= surge_threshold,
        "volume_ratio": round(ratio, 2),
        "ratio": round(ratio, 2),
        "latest_volume": latest,
        "avg_volume": round(avg_vol, 2),
    }


def volume_surge_normalize(result) -> dict:
    """Normalize a surge payload to the canonical key contract."""
    if not isinstance(result, dict):
        return {
            "is_surge": False,
            "volume_ratio": 1.0,
            "ratio": 1.0,
            "latest_volume": 0,
            "avg_volume": 0,
        }
    volume_ratio = result.get("volume_ratio", result.get("ratio", 1.0))
    return {
        "is_surge": result.get("is_surge", False),
        "volume_ratio": volume_ratio,
        "ratio": volume_ratio,
        "latest_volume": result.get("latest_volume", 0),
        "avg_volume": result.get("avg_volume", 0),
    }


def volume_trend(
    candles: list,
    short_window: int = 5,
    long_window: int = 20,
) -> dict:
    """Classify volume trend by short vs long average + buy-volume proxy."""
    if len(candles) < long_window:
        return {"trend": "NEUTRAL", "buy_ratio": 0.5}
    vols = [c.quote_volume if c.quote_volume > 0 else c.volume for c in candles]
    short_avg = sum(vols[-short_window:]) / short_window
    long_avg = sum(vols[-long_window:]) / long_window
    buy_vol = sum(
        (c.quote_volume if c.quote_volume > 0 else c.volume)
        for c in candles[-short_window:]
        if c.close >= c.open
    )
    total_short = sum(
        (c.quote_volume if c.quote_volume > 0 else c.volume)
        for c in candles[-short_window:]
    )
    buy_ratio = buy_vol / total_short if total_short > 0 else 0.5
    vol_ratio = short_avg / long_avg if long_avg > 0 else 1.0
    if vol_ratio > 1.2 and buy_ratio > 0.6:
        trend = "ACCUMULATING"
    elif vol_ratio > 1.2 and buy_ratio < 0.4:
        trend = "DISTRIBUTING"
    elif vol_ratio < 0.5:
        trend = "DRYING_UP"
    else:
        trend = "NEUTRAL"
    return {
        "trend": trend,
        "buy_ratio": round(buy_ratio, 2),
        "short_avg": round(short_avg, 2),
        "long_avg": round(long_avg, 2),
        "volume_ratio": round(vol_ratio, 2),
    }


__all__ = ["volume_surge_detect", "volume_surge_normalize", "volume_trend"]
