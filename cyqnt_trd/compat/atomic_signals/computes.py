"""Atomic-shape compute helpers — L2 parity vs `atomic_strategy_lib.signals`.

Per `docs/migration/overlap-policy.md` Class B: atomic's pure-Python
algorithms differ from `library.features.indicators` at warm-up (atomic
uses SMA-then-Wilder for the first `period` bars; pandas canonical uses
pure-Wilder with NaN padding). To keep L2 dict-equal parity vs atomic,
these wrappers **port atomic's algorithm verbatim**, not delegate to
the canonical pandas surface.

The two surfaces converge after warm-up, but the first ~period bars
will not match the canonical pandas indicators — that's documented as
`intentional-divergence` in `symbols-atomic.md`.

All helpers accept duck-typed Candle objects (atomic Candle, our local
`_AtomicCandle`, or anything with `.close/.high/.low/.open/.volume/.quote_volume`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _extract_closes(values: list) -> list[float]:
    """Convert input to list[float]. Accepts Candle list or float list."""
    if not values:
        return []
    if hasattr(values[0], "close"):
        return [float(c.close) for c in values]
    return [float(v) for v in values]


@dataclass
class VelocityMetrics:
    """Velocity metrics — mirrors `atomic.core.types.VelocityMetrics`."""

    timeframe: str = ""
    vol_vs_avg: float | None = None
    vol_accel_3v3: float | None = None
    range_vs_avg: float | None = None
    green_streak: int = 0
    red_streak: int = 0
    price_change_pct: float = 0.0
    last3_change_pct: float = 0.0


# ---------------------------------------------------------------------------
# RSI / MACD / StochRSI (momentum)
# ---------------------------------------------------------------------------


def rsi_compute(values: list, period: int = 14) -> list[float]:
    """Wilder RSI (atomic shape: list[float] in / list[float] out, shorter by `period`)."""
    values = _extract_closes(values)
    if len(values) < period + 1:
        return []
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [max(-d, 0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    result: list[float] = []
    if avg_loss == 0:
        result.append(100.0)
    else:
        rs = avg_gain / avg_loss
        result.append(100.0 - 100.0 / (1.0 + rs))
    for i in range(period, len(deltas)):
        gain = max(deltas[i], 0)
        loss = max(-deltas[i], 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100.0 - 100.0 / (1.0 + rs))
    return result


def rsi_current(values: list, period: int = 14) -> float | None:
    series = rsi_compute(values, period)
    return round(series[-1], 2) if series else None


def rsi_zone_detect(
    rsi_value: float,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> str:
    """Classify RSI into `OVERBOUGHT` / `OVERSOLD` / `NEUTRAL`."""
    if rsi_value >= overbought:
        return "OVERBOUGHT"
    if rsi_value <= oversold:
        return "OVERSOLD"
    return "NEUTRAL"


def _ema_series_internal(values: list[float], period: int) -> list[float]:
    if not values or period <= 0:
        return []
    k = 2.0 / (period + 1)
    ema = values[0]
    result = [ema]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
        result.append(ema)
    return result


def macd_compute(
    values: list,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    fast_period: int | None = None,
    slow_period: int | None = None,
    signal_period: int | None = None,
) -> dict:
    """MACD line / signal / histogram (atomic dict shape)."""
    if fast_period is not None:
        fast = fast_period
    if slow_period is not None:
        slow = slow_period
    if signal_period is not None:
        signal = signal_period
    values = _extract_closes(values)
    if len(values) < slow:
        return {
            "macd": [], "signal": [], "histogram": [],
            "current_macd": None, "current_signal": None, "current_histogram": None,
        }
    fast_ema = _ema_series_internal(values, fast)
    slow_ema = _ema_series_internal(values, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    sig_line = _ema_series_internal(macd_line, signal)
    hist = [m - s for m, s in zip(macd_line, sig_line)]
    cur_macd = round(macd_line[-1], 6) if macd_line else None
    cur_signal = round(sig_line[-1], 6) if sig_line else None
    cur_hist = round(hist[-1], 6) if hist else None
    return {
        "macd": macd_line,
        "signal": sig_line,
        "histogram": hist,
        "current_macd": cur_macd,
        "current_signal": cur_signal,
        "current_histogram": cur_hist,
        "macd_line": cur_macd,
        "signal_line": cur_signal,
    }


def stochrsi_compute(
    values: list,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> dict:
    values = _extract_closes(values)
    rsi_vals = rsi_compute(values, rsi_period)
    if len(rsi_vals) < stoch_period:
        return {"k": [], "d": [], "current_k": None, "current_d": None}
    stoch_rsi: list[float] = []
    for i in range(stoch_period - 1, len(rsi_vals)):
        window = rsi_vals[i - stoch_period + 1 : i + 1]
        lo = min(window)
        hi = max(window)
        if hi == lo:
            stoch_rsi.append(50.0)
        else:
            stoch_rsi.append((rsi_vals[i] - lo) / (hi - lo) * 100)
    k_line: list[float] = []
    for i in range(k_period - 1, len(stoch_rsi)):
        k_line.append(sum(stoch_rsi[i - k_period + 1 : i + 1]) / k_period)
    d_line: list[float] = []
    for i in range(d_period - 1, len(k_line)):
        d_line.append(sum(k_line[i - d_period + 1 : i + 1]) / d_period)
    return {
        "k": k_line,
        "d": d_line,
        "current_k": round(k_line[-1], 2) if k_line else None,
        "current_d": round(d_line[-1], 2) if d_line else None,
    }


# ---------------------------------------------------------------------------
# Velocity (multi-tf candle metrics)
# ---------------------------------------------------------------------------


def tf_velocity_compute(candles: list) -> VelocityMetrics:
    if len(candles) < 4:
        return VelocityMetrics(timeframe="")
    vols = [
        c.quote_volume if getattr(c, "quote_volume", 0) > 0 else c.volume for c in candles
    ]
    ranges = [c.high - c.low for c in candles]
    latest_vol = vols[-1]
    prev_vols = vols[:-1]
    avg_vol = sum(prev_vols) / len(prev_vols) if prev_vols else 1
    latest_range = ranges[-1]
    prev_ranges = ranges[:-1]
    avg_range = sum(prev_ranges) / len(prev_ranges) if prev_ranges else 1
    vol_accel: float | None = None
    if len(vols) >= 6:
        recent_3 = sum(vols[-3:])
        prev_3 = sum(vols[-6:-3])
        vol_accel = round(recent_3 / prev_3, 2) if prev_3 > 0 else None
    green_streak = 0
    for c in reversed(candles):
        if c.close >= c.open:
            green_streak += 1
        else:
            break
    red_streak = 0
    for c in reversed(candles):
        if c.close < c.open:
            red_streak += 1
        else:
            break
    first_open = candles[0].open
    last_close = candles[-1].close
    price_change_pct = (
        round((last_close - first_open) / first_open * 100, 2) if first_open > 0 else 0
    )
    last3_open = candles[-3].open if len(candles) >= 3 else first_open
    last3_change_pct = (
        round((last_close - last3_open) / last3_open * 100, 2) if last3_open > 0 else 0
    )
    return VelocityMetrics(
        timeframe="",
        vol_vs_avg=round(latest_vol / avg_vol, 2) if avg_vol > 0 else None,
        vol_accel_3v3=vol_accel,
        range_vs_avg=round(latest_range / avg_range, 2) if avg_range > 0 else None,
        green_streak=green_streak,
        red_streak=red_streak,
        price_change_pct=price_change_pct,
        last3_change_pct=last3_change_pct,
    )


def multi_tf_velocity(tf_candles: dict) -> dict:
    result: dict[str, VelocityMetrics] = {}
    for tf, candles in tf_candles.items():
        vm = tf_velocity_compute(candles)
        vm.timeframe = tf
        result[tf] = vm
    return result


# ---------------------------------------------------------------------------
# EMA / SuperTrend / ADX / cross / trend (trend)
# ---------------------------------------------------------------------------


def ema_compute(values: list, period: int) -> list[float]:
    if not values or period <= 0:
        return []
    return _ema_series_internal(_extract_closes(values), period)


def ema_current(values: list, period: int) -> float | None:
    series = ema_compute(values, period)
    return series[-1] if series else None


def supertrend_compute(
    candles: list,
    period: int = 10,
    multiplier: float = 3.0,
) -> list[dict]:
    if len(candles) < period + 1:
        return []
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    trs: list[float] = []
    for i in range(1, len(candles)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    atr_vals: list[float] = []
    if len(trs) < period:
        return []
    atr = sum(trs[:period]) / period
    atr_vals.append(atr)
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        atr_vals.append(atr)
    offset = period
    results: list[dict] = []
    prev_upper = 0.0
    prev_lower = 0.0
    prev_direction = 1
    for j in range(len(atr_vals)):
        idx = offset + j
        hl2 = (highs[idx] + lows[idx]) / 2
        atr_j = atr_vals[j]
        upper = hl2 + multiplier * atr_j
        lower = hl2 - multiplier * atr_j
        if j > 0:
            if not (lower > prev_lower or closes[idx - 1] < prev_lower):
                lower = prev_lower
            if not (upper < prev_upper or closes[idx - 1] > prev_upper):
                upper = prev_upper
        if j == 0:
            direction = 1 if closes[idx] > upper else -1
        else:
            if prev_direction == 1:
                direction = -1 if closes[idx] < lower else 1
            else:
                direction = 1 if closes[idx] > upper else -1
        st = lower if direction == 1 else upper
        results.append({"supertrend": st, "direction": direction})
        prev_upper = upper
        prev_lower = lower
        prev_direction = direction
    return results


def adx_compute(candles: list, period: int = 14) -> list[float]:
    if len(candles) < period + 1:
        return []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    trs: list[float] = []
    for i in range(1, len(candles)):
        h_diff = candles[i].high - candles[i - 1].high
        l_diff = candles[i - 1].low - candles[i].low
        pdm = h_diff if h_diff > l_diff and h_diff > 0 else 0.0
        mdm = l_diff if l_diff > h_diff and l_diff > 0 else 0.0
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        )
        plus_dm.append(pdm)
        minus_dm.append(mdm)
        trs.append(tr)
    if len(trs) < period:
        return []

    def wilder_smooth(data: list[float], p: int) -> list[float]:
        s = sum(data[:p])
        result = [s]
        for i in range(p, len(data)):
            s = s - s / p + data[i]
            result.append(s)
        return result

    sm_plus = wilder_smooth(plus_dm, period)
    sm_minus = wilder_smooth(minus_dm, period)
    sm_tr = wilder_smooth(trs, period)
    dx_vals: list[float] = []
    for i in range(len(sm_tr)):
        tr_v = sm_tr[i]
        if tr_v == 0:
            dx_vals.append(0.0)
            continue
        di_plus = sm_plus[i] / tr_v * 100
        di_minus = sm_minus[i] / tr_v * 100
        di_sum = di_plus + di_minus
        if di_sum == 0:
            dx_vals.append(0.0)
        else:
            dx_vals.append(abs(di_plus - di_minus) / di_sum * 100)
    if len(dx_vals) < period:
        return dx_vals
    adx = sum(dx_vals[:period]) / period
    result = [adx]
    for i in range(period, len(dx_vals)):
        adx = (adx * (period - 1) + dx_vals[i]) / period
        result.append(adx)
    return result


def ema_cross_detect(
    values: list,
    fast_period: int = 9,
    slow_period: int = 21,
) -> dict:
    values = _extract_closes(values)
    if len(values) < max(fast_period, slow_period) + 1:
        return {"cross": "NONE", "fast_ema": None, "slow_ema": None, "spread_pct": 0}
    fast = ema_compute(values, fast_period)
    slow = ema_compute(values, slow_period)
    curr_fast, prev_fast = fast[-1], fast[-2]
    curr_slow, prev_slow = slow[-1], slow[-2]
    spread_pct = (curr_fast - curr_slow) / curr_slow * 100 if curr_slow != 0 else 0
    if prev_fast <= prev_slow and curr_fast > curr_slow:
        cross = "GOLDEN"
    elif prev_fast >= prev_slow and curr_fast < curr_slow:
        cross = "DEATH"
    else:
        cross = "NONE"
    return {
        "cross": cross,
        "fast_ema": curr_fast,
        "slow_ema": curr_slow,
        "spread_pct": round(spread_pct, 4),
    }


def trend_classify(*args: Any, **kwargs: Any) -> str:
    """Multi-shape trend classifier — see atomic.signals.trend.trend_classify."""
    if kwargs.get("price") is not None or kwargs.get("ema_fast") is not None:
        price = kwargs.get("price", args[0] if args else 0)
        ema_fast = kwargs.get("ema_fast", args[1] if len(args) > 1 else None)
        ema_slow = kwargs.get("ema_slow", args[2] if len(args) > 2 else None)
        if ema_fast is not None and ema_slow is not None:
            if ema_fast > ema_slow:
                return "BULLISH"
            if ema_fast < ema_slow:
                return "BEARISH"
            return "MIXED"
        if ema_fast is not None:
            return "BULLISH" if price > ema_fast else "BEARISH"
        return "UNKNOWN"
    if kwargs.get("ema_alignment") is not None or kwargs.get("adx_value") is not None:
        alignment = kwargs.get("ema_alignment", "")
        adx = kwargs.get("adx_value", 0)
        supertrend = kwargs.get("supertrend_direction", "")
        if isinstance(alignment, str) and alignment:
            return alignment
        if adx and adx > 25:
            return "BULLISH" if supertrend == "UP" else "BEARISH"
        return "MIXED"
    if not args:
        return "UNKNOWN"
    data = args[0]
    if isinstance(data, (int, float)):
        price = data
        emas = list(args[1:])
        if not emas:
            return "UNKNOWN"
        if isinstance(emas[0], list):
            emas = emas[0]
        avg = sum(e for e in emas if isinstance(e, (int, float))) / max(len(emas), 1)
        return "BULLISH" if price > avg else "BEARISH" if price < avg else "MIXED"
    if isinstance(data, list):
        if not data:
            return "UNKNOWN"
        if hasattr(data[0], "close") and hasattr(data[0], "open"):
            green = sum(1 for c in data if c.close >= c.open)
            if green == len(data):
                return "BULLISH"
            if green == 0:
                return "BEARISH"
            return "MIXED"
        if isinstance(data[0], (int, float)):
            if len(data) < 2:
                return "UNKNOWN"
            if data[-1] > data[0]:
                return "BULLISH"
            if data[-1] < data[0]:
                return "BEARISH"
            return "MIXED"
    return "UNKNOWN"


def multi_tf_resonance(tf_directions: dict) -> dict:
    if not tf_directions:
        return {"aligned": False, "dominant": "MIXED", "score": 0.0}
    counts: dict[str, int] = {}
    for direction in tf_directions.values():
        counts[direction] = counts.get(direction, 0) + 1
    total = len(tf_directions)
    dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
    score = counts[dominant] / total
    return {
        "aligned": score == 1.0,
        "dominant": dominant,
        "score": round(score, 2),
        "breakdown": dict(counts),
    }


def multi_tf_resonance_from_klines(tf_candles: dict) -> dict:
    tf_directions = {tf: trend_classify(c) for tf, c in tf_candles.items() if c}
    result = multi_tf_resonance(tf_directions)
    breakdown = result.get("breakdown", {})
    return {
        **result,
        "alignment": result.get("dominant", "MIXED"),
        "aligned_count": max(breakdown.values() or [0]),
        "total": len(tf_directions),
        "directions": tf_directions,
    }


def trend_strength(candles: list) -> dict:
    if not candles:
        return {"green_ratio": 0.0, "avg_body_pct": 0.0, "strength": 0.0}
    green = 0
    body_pcts: list[float] = []
    for c in candles:
        if c.close >= c.open:
            green += 1
        body = abs(c.close - c.open)
        hl_range = c.high - c.low
        body_pcts.append(body / hl_range * 100 if hl_range > 0 else 0)
    green_ratio = green / len(candles)
    avg_body = sum(body_pcts) / len(body_pcts) if body_pcts else 0
    strength = green_ratio * 0.6 + min(avg_body / 80, 1.0) * 0.4
    return {
        "green_ratio": round(green_ratio, 2),
        "avg_body_pct": round(avg_body, 2),
        "strength": round(strength, 2),
    }


# ---------------------------------------------------------------------------
# ATR / Bollinger / squeeze / dual-speed (volatility)
# ---------------------------------------------------------------------------


def atr_compute(candles: list, period: int = 14) -> list[float]:
    if len(candles) < 2:
        return []
    trs: list[float] = []
    for i in range(1, len(candles)):
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        )
        trs.append(tr)
    if len(trs) < period:
        return [sum(trs) / len(trs)] if trs else []
    atr = sum(trs[:period]) / period
    result = [atr]
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        result.append(atr)
    return result


def atr_current(candles: list, period: int = 14) -> float | None:
    series = atr_compute(candles, period)
    return round(series[-1], 8) if series else None


def atr_ratio(candles: list, period: int = 14) -> float | None:
    atr = atr_current(candles, period)
    if atr is None:
        return None
    last_price = candles[-1].close
    if last_price <= 0:
        return None
    return round(atr / last_price * 100, 2)


def bollinger_compute(
    values: list,
    period: int = 20,
    num_std: float = 2.0,
    std_dev: float | None = None,
) -> dict:
    if std_dev is not None:
        num_std = std_dev
    values = _extract_closes(values)
    if len(values) < period:
        return {
            "upper": [], "middle": [], "lower": [],
            "bandwidth": [], "pct_b": [],
            "current_upper": None, "current_middle": None, "current_lower": None,
            "current_bandwidth": None, "current_pct_b": None,
        }
    upper: list[float] = []
    middle: list[float] = []
    lower: list[float] = []
    bandwidth: list[float] = []
    pct_b: list[float] = []
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        sma = sum(window) / period
        variance = sum((x - sma) ** 2 for x in window) / period
        std = variance ** 0.5
        u = sma + num_std * std
        l = sma - num_std * std
        middle.append(sma)
        upper.append(u)
        lower.append(l)
        bw = (u - l) / sma * 100 if sma != 0 else 0
        bandwidth.append(bw)
        pb = (values[i] - l) / (u - l) if (u - l) != 0 else 0.5
        pct_b.append(pb)
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "pct_b": pct_b,
        "current_upper": round(upper[-1], 8) if upper else None,
        "current_middle": round(middle[-1], 8) if middle else None,
        "current_lower": round(lower[-1], 8) if lower else None,
        "current_bandwidth": round(bandwidth[-1], 2) if bandwidth else None,
        "current_pct_b": round(pct_b[-1], 4) if pct_b else None,
    }


def bb_squeeze_detect(
    values: list,
    period: int = 20,
    num_std: float = 2.0,
    squeeze_threshold: float = 2.0,
    lookback: int = 20,
    std_dev: float | None = None,
    threshold_pct: float | None = None,
) -> dict:
    if std_dev is not None:
        num_std = std_dev
    if threshold_pct is not None:
        squeeze_threshold = threshold_pct
    values = _extract_closes(values)
    bb = bollinger_compute(values, period, num_std)
    bw = bb["bandwidth"]
    if len(bw) < lookback:
        return {"is_squeeze": False, "current_bandwidth": None}
    recent_bw = bw[-lookback:]
    current_bw = bw[-1]
    avg_bw = sum(recent_bw) / len(recent_bw)
    min_bw = min(recent_bw)
    ratio = current_bw / avg_bw if avg_bw > 0 else 1.0
    is_squeeze = current_bw <= squeeze_threshold or ratio < 0.5
    return {
        "is_squeeze": is_squeeze,
        "current_bandwidth": round(current_bw, 2),
        "avg_bandwidth": round(avg_bw, 2),
        "min_bandwidth": round(min_bw, 2),
        "bandwidth_ratio": round(ratio, 2),
    }


def dual_speed_atr(
    candles: list,
    fast_period: int = 7,
    slow_period: int = 21,
) -> dict:
    fast = atr_current(candles, fast_period)
    slow = atr_current(candles, slow_period)
    if fast is None or slow is None:
        return {"fast_atr": fast, "slow_atr": slow, "ratio": None, "expanding": False}
    ratio = fast / slow if slow > 0 else 0
    return {
        "fast_atr": round(fast, 8),
        "slow_atr": round(slow, 8),
        "ratio": round(ratio, 2),
        "expanding": ratio > 1.0,
    }


__all__ = [
    "VelocityMetrics",
    "adx_compute",
    "atr_compute",
    "atr_current",
    "atr_ratio",
    "bb_squeeze_detect",
    "bollinger_compute",
    "dual_speed_atr",
    "ema_compute",
    "ema_cross_detect",
    "ema_current",
    "macd_compute",
    "multi_tf_resonance",
    "multi_tf_resonance_from_klines",
    "multi_tf_velocity",
    "rsi_compute",
    "rsi_current",
    "rsi_zone_detect",
    "stochrsi_compute",
    "supertrend_compute",
    "tf_velocity_compute",
    "trend_classify",
    "trend_strength",
]
