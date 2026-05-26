"""Microstructure / whale / order-flow detectors.

L1 parity vs `cyqnt_trd.blocks.microstructure`. All functions are pure
pandas — they consume taker-buy/sell volume, OI, etc. produced by the
data layer and return boolean / numeric Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_pro_trading_library.library.features.indicators import rolling_zscore


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _safe_divide(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def whale_buy_signal(
    buy_volume: pd.Series,
    rolling_period: int = 96,
    threshold_quantile: float = 0.95,
) -> pd.Series:
    """True when buy_volume in top `threshold_quantile` of last `rolling_period` bars."""
    rolling_period = _positive_int(rolling_period, "rolling_period")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError(f"threshold_quantile must be in (0,1), got {threshold_quantile}")
    threshold = buy_volume.rolling(window=rolling_period, min_periods=rolling_period).quantile(
        threshold_quantile
    )
    return (buy_volume >= threshold).fillna(False)


def whale_sell_signal(
    sell_volume: pd.Series,
    rolling_period: int = 96,
    threshold_quantile: float = 0.95,
) -> pd.Series:
    return whale_buy_signal(sell_volume, rolling_period, threshold_quantile)


def smart_money_inflow(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    period: int = 12,
) -> pd.Series:
    """Rolling net taker-buy USD over `period` bars."""
    period = _positive_int(period, "period")
    return (buy_volume.astype(float) - sell_volume.astype(float)).rolling(
        window=period, min_periods=1
    ).sum()


def order_imbalance(buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
    """Per-bar (buy - sell) / (buy + sell) in [-1, 1]."""
    b = buy_volume.astype(float)
    s = sell_volume.astype(float)
    total = b + s
    return _safe_divide(b - s, total, fill=0.0)


def large_print_zscore(volume: pd.Series, period: int = 96) -> pd.Series:
    """Rolling z-score of bar volume."""
    return rolling_zscore(volume, period)


__all__ = [
    "large_print_zscore",
    "order_imbalance",
    "smart_money_inflow",
    "whale_buy_signal",
    "whale_sell_signal",
]
