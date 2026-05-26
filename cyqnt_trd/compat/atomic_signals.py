"""Atomic-style signal wrappers — adapter layer between atomic and cyqnt_trd.

This module provides the **atomic_strategy_lib.signals.\\*** API surface
backed by ``cyqnt_trd.blocks`` implementations. Existing atomic-style
case scripts can switch their imports without changing their call sites:

    # Old:
    from atomic_strategy_lib.signals.momentum import rsi_compute, rsi_current
    from atomic_strategy_lib.signals.volume import volume_surge_detect

    # New (drop-in replacement):
    from cyqnt_trd.compat.atomic_signals import rsi_compute, rsi_current
    from cyqnt_trd.compat.atomic_signals import volume_surge_detect

I/O conventions (matching atomic):

* ``*_compute`` functions take ``list[float]`` (or ``list[Candle]``) and
  return ``list[float]``.
* ``*_current`` functions return a single ``float`` (the most recent value).
* ``*_detect`` functions return a ``dict`` with at least one boolean flag.

Internally each wrapper converts inputs to ``pd.Series`` /
:class:`pandas.DataFrame`, delegates to the canonical pandas-vectorized
``cyqnt_trd.blocks.*`` implementation, and converts the result back to
the atomic shape.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd

from cyqnt_trd.blocks import conditions as cond
from cyqnt_trd.blocks import indicators as ind
from cyqnt_trd.compat.adapters import candles_to_df

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _series_from_floats_or_candles(values: Sequence) -> pd.Series:
    """Convert ``list[float]`` or ``list[Candle]`` to a ``pd.Series`` of closes."""
    if values is None:
        return pd.Series(dtype=float)
    if not isinstance(values, (list, tuple, pd.Series)):
        # already a Series-like
        try:
            return pd.Series(values, dtype=float)
        except Exception:
            return pd.Series([], dtype=float)
    if len(values) == 0:
        return pd.Series(dtype=float)
    first = values[0]
    if hasattr(first, "close"):  # list[Candle]
        return pd.Series([float(c.close) for c in values], dtype=float)
    return pd.Series([float(v) for v in values], dtype=float)


def _df_from_candles(candles: Sequence) -> pd.DataFrame:
    """Convert ``list[Candle]`` to OHLCV DataFrame."""
    if not candles:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "quote_volume", "trades"]
        )
    return candles_to_df(list(candles))


def _safe_last_float(series: pd.Series, default: float = 0.0, ndigits: int = 4) -> float:
    """Return last value of *series* as a Python float, or *default* if empty/NaN."""
    if series is None or len(series) == 0:
        return default
    last = series.iloc[-1]
    if pd.isna(last):
        return default
    return round(float(last), ndigits)


def _series_to_list(series: pd.Series, ndigits: int = 4) -> List[float]:
    """Series → list[float], NaN → 0.0, rounded."""
    if series is None or len(series) == 0:
        return []
    return [
        round(float(v), ndigits) if not pd.isna(v) else 0.0 for v in series
    ]


# ---------------------------------------------------------------------------
# signals.momentum  (RSI / MACD / StochRSI)
# ---------------------------------------------------------------------------


def rsi_compute(values: Sequence, period: int = 14) -> List[float]:
    """List of RSI values for the input series.

    Drop-in replacement for ``atomic_strategy_lib.signals.momentum.rsi_compute``.
    """
    series = _series_from_floats_or_candles(values)
    return _series_to_list(ind.rsi(series, period), ndigits=2)


def rsi_current(values: Sequence, period: int = 14) -> float:
    """Latest RSI value (single float).

    Drop-in replacement for ``atomic_strategy_lib.signals.momentum.rsi_current``.
    """
    series = _series_from_floats_or_candles(values)
    return _safe_last_float(ind.rsi(series, period), default=0.0, ndigits=2)


def rsi_zone_detect(
    values: Sequence,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> dict:
    """RSI zone classification.

    Returns
    -------
    dict
        ``{"value", "zone", "oversold", "overbought"}`` where ``zone`` is one
        of ``"OVERSOLD"``, ``"OVERBOUGHT"``, ``"NEUTRAL"``.
    """
    rsi_now = rsi_current(values, period)
    if rsi_now < oversold:
        zone = "OVERSOLD"
    elif rsi_now > overbought:
        zone = "OVERBOUGHT"
    else:
        zone = "NEUTRAL"
    return {
        "value": rsi_now,
        "zone": zone,
        "oversold": oversold,
        "overbought": overbought,
    }


def macd_compute(
    values: Sequence,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict:
    """MACD line / signal / histogram.

    Returns
    -------
    dict
        ``macd``, ``signal``, ``histogram`` (lists) plus ``current_macd``,
        ``current_signal``, ``current_histogram`` (floats).
    """
    series = _series_from_floats_or_candles(values)
    macd_line, signal_line, hist = ind.macd(series, fast, slow, signal)
    return {
        "macd": _series_to_list(macd_line),
        "signal": _series_to_list(signal_line),
        "histogram": _series_to_list(hist),
        "current_macd": _safe_last_float(macd_line),
        "current_signal": _safe_last_float(signal_line),
        "current_histogram": _safe_last_float(hist),
    }


def stochrsi_compute(
    values: Sequence,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> dict:
    """StochRSI %K and %D series."""
    series = _series_from_floats_or_candles(values)
    k, d = ind.stochrsi(series, period, smooth_k, smooth_d)
    return {
        "k": _series_to_list(k, ndigits=2),
        "d": _series_to_list(d, ndigits=2),
        "current_k": _safe_last_float(k, ndigits=2),
        "current_d": _safe_last_float(d, ndigits=2),
    }


# ---------------------------------------------------------------------------
# signals.trend  (EMA / SMA / ADX / SuperTrend)
# ---------------------------------------------------------------------------


def ema_compute(values: Sequence, period: int) -> List[float]:
    """List of EMA values."""
    series = _series_from_floats_or_candles(values)
    return _series_to_list(ind.ema(series, period))


def ema_current(values: Sequence, period: int) -> float:
    """Latest EMA value."""
    series = _series_from_floats_or_candles(values)
    return _safe_last_float(ind.ema(series, period))


def ema_cross_detect(
    values: Sequence,
    fast: int = 9,
    slow: int = 21,
    lookback: int = 5,
) -> dict:
    """Detect recent EMA golden / death cross.

    Returns
    -------
    dict
        ``cross_type`` is one of ``"GOLDEN"`` (recent fast-over-slow up cross),
        ``"DEATH"`` (down cross), or ``"NONE"``.
    """
    series = _series_from_floats_or_candles(values)
    ema_fast = ind.ema(series, fast)
    ema_slow = ind.ema(series, slow)
    cross_up = cond.ma_cross_above(ema_fast, ema_slow)
    cross_dn = cond.ma_cross_below(ema_fast, ema_slow)

    recent_up = bool(cross_up.tail(lookback).any()) if len(cross_up) else False
    recent_dn = bool(cross_dn.tail(lookback).any()) if len(cross_dn) else False

    if recent_up:
        cross_type = "GOLDEN"
    elif recent_dn:
        cross_type = "DEATH"
    else:
        cross_type = "NONE"

    return {
        "cross_type": cross_type,
        "recent": recent_up or recent_dn,
        "lookback": lookback,
        "current_fast": _safe_last_float(ema_fast),
        "current_slow": _safe_last_float(ema_slow),
    }


def adx_compute(candles: Sequence, period: int = 14) -> List[float]:
    """List of ADX values from candles."""
    df = _df_from_candles(candles)
    result, _, _ = ind.adx(df, period)
    return _series_to_list(result, ndigits=2)


def supertrend_compute(
    candles: Sequence,
    period: int = 10,
    multiplier: float = 3.0,
) -> dict:
    """SuperTrend line + direction."""
    df = _df_from_candles(candles)
    line, direction = ind.supertrend(df, period, multiplier)
    return {
        "line": _series_to_list(line),
        "direction": _series_to_list(direction, ndigits=0),
        "current_line": _safe_last_float(line),
        "current_direction": int(_safe_last_float(direction, ndigits=0, default=0)),
    }


def trend_classify(candles: Sequence, period: int = 14) -> dict:
    """Classify trend regime from ADX."""
    adx_values = adx_compute(candles, period)
    if not adx_values:
        return {"trend": "NEUTRAL", "adx_value": 0.0}
    adx_now = adx_values[-1]
    if adx_now > 25:
        trend = "STRONG_TREND"
    elif adx_now > 20:
        trend = "WEAK_TREND"
    else:
        trend = "RANGING"
    return {"trend": trend, "adx_value": adx_now}


# ---------------------------------------------------------------------------
# signals.volatility  (ATR / Bollinger / BB-squeeze)
# ---------------------------------------------------------------------------


def atr_compute(candles: Sequence, period: int = 14) -> List[float]:
    """List of ATR values."""
    df = _df_from_candles(candles)
    return _series_to_list(ind.atr(df, period))


def atr_current(candles: Sequence, period: int = 14) -> float:
    """Latest ATR value."""
    df = _df_from_candles(candles)
    return _safe_last_float(ind.atr(df, period))


def atr_ratio_current(candles: Sequence, period: int = 14) -> float:
    """ATR / latest close, expressed as fraction (e.g. 0.025 = 2.5%)."""
    df = _df_from_candles(candles)
    if df.empty or "close" not in df.columns:
        return 0.0
    return _safe_last_float(ind.atr_ratio(df, period), ndigits=4)


def bollinger_compute(
    values: Sequence,
    period: int = 20,
    std: float = 2.0,
) -> dict:
    """Bollinger Bands."""
    series = _series_from_floats_or_candles(values)
    upper, middle, lower = ind.bollinger(series, period, std)
    return {
        "upper": _series_to_list(upper),
        "middle": _series_to_list(middle),
        "lower": _series_to_list(lower),
        "current_upper": _safe_last_float(upper),
        "current_middle": _safe_last_float(middle),
        "current_lower": _safe_last_float(lower),
    }


def bb_squeeze_detect(
    values: Sequence,
    period: int = 20,
    threshold: float = 0.02,
) -> dict:
    """Detect Bollinger Band squeeze (bandwidth below threshold)."""
    series = _series_from_floats_or_candles(values)
    bw = ind.bb_bandwidth(series, period)
    current_bw = _safe_last_float(bw, ndigits=4)
    return {
        "squeeze": current_bw < threshold,
        "bandwidth": current_bw,
        "bandwidth_pct": round(current_bw * 100, 2),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# signals.volume  (volume surge / trend)
# ---------------------------------------------------------------------------


def volume_surge_detect(
    candles: Sequence,
    multiplier: float = 2.0,
    lookback: int = 20,
) -> dict:
    """Detect volume surge on the latest bar.

    Returns
    -------
    dict
        ``surge`` (bool), ``ratio`` (current vol / avg vol), ``current_volume``,
        ``avg_volume``, ``multiplier``.
    """
    df = _df_from_candles(candles)
    if df.empty or "volume" not in df.columns or len(df) < 2:
        return {
            "surge": False,
            "ratio": 0.0,
            "current_volume": 0.0,
            "avg_volume": 0.0,
            "multiplier": multiplier,
        }
    vol_ma = df["volume"].rolling(lookback).mean()
    surge_series = cond.volume_surge(df, vol_ma, multiplier=multiplier)
    current_vol = float(df["volume"].iloc[-1])
    avg_vol = (
        float(vol_ma.iloc[-1])
        if len(vol_ma) and not pd.isna(vol_ma.iloc[-1])
        else 0.0
    )
    ratio = current_vol / avg_vol if avg_vol > 0 else 0.0
    return {
        "surge": bool(surge_series.iloc[-1]) if len(surge_series) else False,
        "ratio": round(ratio, 2),
        "current_volume": round(current_vol, 2),
        "avg_volume": round(avg_vol, 2),
        "multiplier": multiplier,
    }


def volume_trend(candles: Sequence, lookback: int = 10) -> dict:
    """Classify recent volume direction."""
    df = _df_from_candles(candles)
    if df.empty or "volume" not in df.columns or len(df) < lookback + 1:
        return {"trend": "FLAT", "slope_pct": 0.0}
    recent = df["volume"].tail(lookback)
    first = float(recent.iloc[0])
    last = float(recent.iloc[-1])
    if first <= 0:
        return {"trend": "FLAT", "slope_pct": 0.0}
    slope_pct = (last - first) / first * 100.0
    if slope_pct > 10:
        trend = "ACCUMULATING"
    elif slope_pct < -10:
        trend = "DRYING_UP"
    else:
        trend = "FLAT"
    return {"trend": trend, "slope_pct": round(slope_pct, 2)}


# ---------------------------------------------------------------------------
# signals.derivatives  (funding / OI / crowding)
# ---------------------------------------------------------------------------


def funding_extreme_detect(
    funding_rate: float,
    squeeze_threshold: float = -0.0001,
    crowded_threshold: float = 0.0005,
) -> dict:
    """Classify a funding-rate value into squeeze / normal / crowded.

    Returns
    -------
    dict
        ``state`` is one of ``"SQUEEZE"``, ``"NORMAL"``, ``"CROWDED_LONG"``;
        ``direction`` is the implied bias (``"BULLISH"`` for squeeze,
        ``"BEARISH"`` for crowded).
    """
    if funding_rate <= squeeze_threshold:
        state = "SQUEEZE"
        direction = "BULLISH"
    elif funding_rate >= crowded_threshold:
        state = "CROWDED_LONG"
        direction = "BEARISH"
    else:
        state = "NORMAL"
        direction = "NEUTRAL"
    return {
        "state": state,
        "direction": direction,
        "value": float(funding_rate),
        "squeeze_threshold": squeeze_threshold,
        "crowded_threshold": crowded_threshold,
    }


def oi_anomaly_detect(
    oi_current: float,
    oi_prev: float,
    threshold_pct: float = 10.0,
) -> dict:
    """Detect anomalous OI change between two snapshots."""
    if oi_prev <= 0:
        return {"anomaly": False, "change_pct": 0.0, "direction": "FLAT"}
    change_pct = (float(oi_current) - float(oi_prev)) / float(oi_prev) * 100.0
    if change_pct > 0:
        direction = "INCREASE"
    elif change_pct < 0:
        direction = "DECREASE"
    else:
        direction = "FLAT"
    return {
        "anomaly": abs(change_pct) >= threshold_pct,
        "change_pct": round(change_pct, 2),
        "direction": direction,
        "threshold_pct": threshold_pct,
    }


def crowding_detect(
    long_short_ratio: float,
    funding_rate: Optional[float] = None,
    extreme_ratio: float = 4.0,
    funding_extreme: float = 0.0005,
) -> dict:
    """Multi-factor crowding detector.

    Combines long/short ratio with funding rate (if provided) to estimate
    whether positioning is one-sided enough to risk a squeeze.
    """
    crowded_long = long_short_ratio >= extreme_ratio
    crowded_short = long_short_ratio <= 1.0 / extreme_ratio if extreme_ratio > 0 else False

    risk_level = "LOW"
    if crowded_long or crowded_short:
        risk_level = "MODERATE"
        if funding_rate is not None:
            if funding_rate > funding_extreme and crowded_long:
                risk_level = "HIGH"
            elif funding_rate < -funding_extreme and crowded_short:
                risk_level = "HIGH"

    return {
        "crowded_long": crowded_long,
        "crowded_short": crowded_short,
        "ls_ratio": float(long_short_ratio),
        "funding_rate": funding_rate,
        "risk_level": risk_level,
    }


__all__ = [
    # momentum
    "rsi_compute",
    "rsi_current",
    "rsi_zone_detect",
    "macd_compute",
    "stochrsi_compute",
    # trend
    "ema_compute",
    "ema_current",
    "ema_cross_detect",
    "adx_compute",
    "supertrend_compute",
    "trend_classify",
    # volatility
    "atr_compute",
    "atr_current",
    "atr_ratio_current",
    "bollinger_compute",
    "bb_squeeze_detect",
    # volume
    "volume_surge_detect",
    "volume_trend",
    # derivatives
    "funding_extreme_detect",
    "oi_anomaly_detect",
    "crowding_detect",
]
