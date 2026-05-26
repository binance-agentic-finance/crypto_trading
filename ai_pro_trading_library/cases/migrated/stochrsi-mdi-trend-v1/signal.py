"""StochRSI/MDI trend v1 — long signal.

Combines:
  - StochRSI %K oversold rebound (cross above oversold threshold).
  - MDI (minus_di from ADX) below threshold — confirms downside momentum
    is fading.

Long signal fires when **both** conditions hold on the current bar.

StochRSI = stochastic oscillator computed on the RSI series rather than
on price. We use:
    rsi_series = rsi(close, rsi_period)
    %K = (rsi - rsi.rolling(k).min) / (rsi.rolling(k).max - rsi.rolling(k).min)

When the rolling RSI range is degenerate (all bars equal), we fall back
to %K = 0.5 (neutral).

This replaces the prior bootstrap SMA10>SMA20 placeholder.
"""

from __future__ import annotations

import pandas as pd

from ai_pro_trading_library.library.core.protocols import StrategySpec
from ai_pro_trading_library.library.features.indicators import adx, rsi


def _stochrsi_k(close: pd.Series, rsi_period: int, k_period: int, smooth: int) -> pd.Series:
    rsi_series = rsi(close, rsi_period)
    lo = rsi_series.rolling(window=k_period, min_periods=k_period).min()
    hi = rsi_series.rolling(window=k_period, min_periods=k_period).max()
    rng = (hi - lo).replace(0.0, float("nan"))
    raw_k = (rsi_series - lo) / rng
    raw_k = raw_k.fillna(0.5)
    if smooth and smooth > 1:
        raw_k = raw_k.rolling(window=smooth, min_periods=smooth).mean()
    return raw_k


def build_signal(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    params = spec.parameters
    rsi_period = int(params.get("rsi_period", 14))
    k_period = int(params.get("stoch_k_period", 14))
    smooth = int(params.get("stoch_smooth", 3))
    oversold = float(params.get("stochrsi_oversold", 0.20))
    mdi_threshold = float(params.get("mdi_threshold", 25.0))
    adx_period = int(params.get("adx_period", 14))

    close = data["close"].astype(float)
    if "high" not in data.columns or "low" not in data.columns:
        return pd.Series(False, index=data.index)

    k = _stochrsi_k(close, rsi_period, k_period, smooth)
    # Fresh cross above the oversold threshold (was below, is now above).
    crossed_up = (k > oversold) & (k.shift(1) <= oversold)

    _adx_series, _plus_di, minus_di = adx(data, adx_period)
    mdi_relaxing = minus_di < mdi_threshold

    return (crossed_up & mdi_relaxing).fillna(False)
