"""Atomic structural detectors — L2 parity vs `atomic.signals.structure`.

All helpers accept duck-typed Candle objects (anything with `.high`,
`.low`, `.open`, `.close`, `.timestamp`) so atomic Candle dataclasses
work directly.
"""

from __future__ import annotations


def fibonacci_levels(
    swing_high: float,
    swing_low: float,
    direction: str = "UP",
) -> dict[str, float]:
    """Compute Fibonacci retracement levels."""
    diff = swing_high - swing_low
    ratios = {
        "0.0": 0.0,
        "0.236": 0.236,
        "0.382": 0.382,
        "0.5": 0.5,
        "0.618": 0.618,
        "0.786": 0.786,
        "1.0": 1.0,
        "1.618": 1.618,
        "2.618": 2.618,
    }
    levels: dict[str, float] = {}
    for name, ratio in ratios.items():
        if direction == "UP":
            levels[name] = round(swing_high - diff * ratio, 8)
        else:
            levels[name] = round(swing_low + diff * ratio, 8)
    return levels


def candlestick_pattern_detect(candles: list) -> list[str]:
    """Detect common candle patterns from the last 1-3 candles. Returns label list."""
    patterns: list[str] = []
    if not candles:
        return patterns
    c = candles[-1]
    body = abs(c.close - c.open)
    hl_range = c.high - c.low
    if hl_range == 0:
        return patterns
    body_ratio = body / hl_range
    upper_wick = c.high - max(c.open, c.close)
    lower_wick = min(c.open, c.close) - c.low
    is_green = c.close >= c.open
    if body_ratio < 0.1:
        patterns.append("DOJI")
    if lower_wick > body * 2 and upper_wick < body * 0.5 and body_ratio < 0.4:
        patterns.append("HAMMER" if is_green else "HANGING_MAN")
    if upper_wick > body * 2 and lower_wick < body * 0.5 and body_ratio < 0.4:
        patterns.append("SHOOTING_STAR" if not is_green else "INVERTED_HAMMER")
    if body_ratio > 0.85:
        patterns.append("BULLISH_MARUBOZU" if is_green else "BEARISH_MARUBOZU")
    if len(candles) >= 2:
        prev = candles[-2]
        curr_body_low = min(c.open, c.close)
        curr_body_high = max(c.open, c.close)
        prev_body_low = min(prev.open, prev.close)
        prev_body_high = max(prev.open, prev.close)
        if (
            is_green
            and prev.close < prev.open
            and curr_body_low < prev_body_low
            and curr_body_high > prev_body_high
        ):
            patterns.append("BULLISH_ENGULFING")
        if (
            not is_green
            and prev.close >= prev.open
            and curr_body_low < prev_body_low
            and curr_body_high > prev_body_high
        ):
            patterns.append("BEARISH_ENGULFING")
    if len(candles) >= 3:
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        c1_range = c1.high - c1.low
        c2_body = abs(c2.close - c2.open)
        c2_range = c2.high - c2.low
        if c1_range > 0 and c2_body / (c2_range if c2_range > 0 else 1) < 0.3:
            if (
                c1.close < c1.open
                and c3.close > c3.open
                and c3.close > (c1.open + c1.close) / 2
            ):
                patterns.append("MORNING_STAR")
            if (
                c1.close > c1.open
                and c3.close < c3.open
                and c3.close < (c1.open + c1.close) / 2
            ):
                patterns.append("EVENING_STAR")
    return patterns


def structural_pivot_identify(candles: list, lookback: int = 5) -> dict:
    """Pivot highs/lows over a ±lookback window."""
    pivot_highs: list[dict] = []
    pivot_lows: list[dict] = []
    for i in range(lookback, len(candles) - lookback):
        window_highs = [candles[j].high for j in range(i - lookback, i + lookback + 1)]
        window_lows = [candles[j].low for j in range(i - lookback, i + lookback + 1)]
        if candles[i].high == max(window_highs):
            pivot_highs.append(
                {
                    "index": i,
                    "price": candles[i].high,
                    "timestamp": candles[i].timestamp,
                }
            )
        if candles[i].low == min(window_lows):
            pivot_lows.append(
                {
                    "index": i,
                    "price": candles[i].low,
                    "timestamp": candles[i].timestamp,
                }
            )
    return {"pivot_highs": pivot_highs, "pivot_lows": pivot_lows}


def box_range_construct(
    candles: list,
    max_range_pct: float = 5.0,
    min_bars: int = 10,
):
    """Detect a consolidation box if last `min_bars` have range < max_range_pct."""
    if len(candles) < min_bars:
        return None
    window = candles[-min_bars:]
    high = max(c.high for c in window)
    low = min(c.low for c in window)
    mid = (high + low) / 2
    if mid == 0:
        return None
    range_pct = (high - low) / mid * 100
    if range_pct > max_range_pct:
        return None
    return {
        "box_high": high,
        "box_low": low,
        "box_mid": mid,
        "range_pct": round(range_pct, 2),
        "bars": min_bars,
        "breakout_up": candles[-1].close > high,
        "breakout_down": candles[-1].close < low,
    }


def top_bottom_count(candles: list) -> dict:
    """TD-Sequential-style up/down count vs 4 bars ago."""
    if len(candles) < 5:
        return {"up_count": 0, "down_count": 0}
    up_count = 0
    down_count = 0
    for i in range(4, len(candles)):
        if candles[i].close > candles[i - 4].close:
            up_count += 1
            down_count = 0
        elif candles[i].close < candles[i - 4].close:
            down_count += 1
            up_count = 0
        else:
            up_count = 0
            down_count = 0
    return {"up_count": up_count, "down_count": down_count}


def symmetry_structure_detect(
    candles: list,
    lookback: int = 5,
    tolerance_pct: float = 1.0,
) -> dict:
    """Detect double top / double bottom from latest two pivot pairs."""
    pivots = structural_pivot_identify(candles, lookback)
    highs = pivots["pivot_highs"]
    lows = pivots["pivot_lows"]
    pattern = "NONE"
    p1 = 0.0
    p2 = 0.0
    if len(highs) >= 2:
        h1 = highs[-2]["price"]
        h2 = highs[-1]["price"]
        if h1 > 0 and abs(h2 - h1) / h1 * 100 <= tolerance_pct:
            pattern = "DOUBLE_TOP"
            p1 = h1
            p2 = h2
    if len(lows) >= 2:
        l1 = lows[-2]["price"]
        l2 = lows[-1]["price"]
        if l1 > 0 and abs(l2 - l1) / l1 * 100 <= tolerance_pct:
            if pattern == "NONE":
                pattern = "DOUBLE_BOTTOM"
                p1 = l1
                p2 = l2
    return {"pattern": pattern, "pivot_1": p1, "pivot_2": p2}


def structural_quality_score(candles: list) -> dict:
    """Composite 0-100 quality score combining patterns + pivots + body + range."""
    if len(candles) < 10:
        return {"score": 0, "components": {}}
    patterns = candlestick_pattern_detect(candles)
    pattern_score = min(len(patterns) * 15, 30)
    pivots = structural_pivot_identify(candles, min(5, len(candles) // 3))
    has_structure = len(pivots["pivot_highs"]) >= 1 and len(pivots["pivot_lows"]) >= 1
    pivot_score = 25 if has_structure else 0
    body_ratios = []
    for c in candles[-10:]:
        hl = c.high - c.low
        if hl > 0:
            body_ratios.append(abs(c.close - c.open) / hl)
    avg_body_ratio = sum(body_ratios) / len(body_ratios) if body_ratios else 0
    body_score = int(avg_body_ratio * 25)
    ranges = [c.high - c.low for c in candles[-10:]]
    if len(ranges) >= 5:
        recent = sum(ranges[-3:]) / 3
        older = sum(ranges[:3]) / 3
        expansion = recent / older if older > 0 else 1
        range_score = min(int(expansion * 10), 20)
    else:
        range_score = 10
    total = pattern_score + pivot_score + body_score + range_score
    return {
        "score": min(total, 100),
        "components": {
            "pattern_score": pattern_score,
            "pivot_score": pivot_score,
            "body_score": body_score,
            "range_score": range_score,
        },
        "patterns_found": patterns,
    }


__all__ = [
    "box_range_construct",
    "candlestick_pattern_detect",
    "fibonacci_levels",
    "structural_pivot_identify",
    "structural_quality_score",
    "symmetry_structure_detect",
    "top_bottom_count",
]
