"""Order-flow / whale / market-maker microstructure helpers.

Most of these are *derived* signals built from the data fetched by
:mod:`cyqnt_trd.blocks.data` (taker buy/sell volume, OI, long/short
ratio, liquidations, large-trade aggregates). They are intentionally
simple — heuristics, not models — because the user requests in T01 mostly
ask for "show me where whales/MMs are" rather than predictive accuracy.

Examples
--------
>>> from cyqnt_trd.blocks import microstructure as micro, data
>>> df = data.fetch_klines("BTCUSDT", "5m", limit=288)
>>> tk = data.fetch_taker_buy_sell_ratio("BTCUSDT", "5m", limit=288)
>>> whale = micro.whale_buy_signal(tk["buy_volume"], threshold_quantile=0.95)
"""

from __future__ import annotations

import pandas as pd

from ._utils import ensure_series, positive_int, safe_divide

__all__ = [
    "whale_buy_signal",
    "whale_sell_signal",
    "smart_money_inflow",
    "order_imbalance",
    "large_print_zscore",
]


def whale_buy_signal(
    buy_volume: pd.Series,
    rolling_period: int = 96,
    threshold_quantile: float = 0.95,
) -> pd.Series:
    """Boolean: bar's taker-buy volume is in the top *threshold_quantile* of last N bars.

    Heuristic for spotting unusually large taker-buy prints (whale
    aggression). 96 5-min bars = 8 hours of context.
    """
    rolling_period = positive_int(rolling_period, "rolling_period")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError(f"threshold_quantile must be within (0, 1), got {threshold_quantile}")
    s = ensure_series(buy_volume)
    threshold = s.rolling(window=rolling_period, min_periods=rolling_period).quantile(
        threshold_quantile
    )
    return (s >= threshold).fillna(False)


def whale_sell_signal(
    sell_volume: pd.Series,
    rolling_period: int = 96,
    threshold_quantile: float = 0.95,
) -> pd.Series:
    """Boolean: bar's taker-sell volume is in the top quantile."""
    return whale_buy_signal(sell_volume, rolling_period, threshold_quantile)


def smart_money_inflow(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    period: int = 12,
) -> pd.Series:
    """Rolling net taker-buy USD over *period* bars.

    Positive = net buying pressure. Used to corroborate trend
    decisions (e.g. only enter long if ``smart_money_inflow > 0``).
    """
    period = positive_int(period, "period")
    b = ensure_series(buy_volume).astype(float)
    s = ensure_series(sell_volume).astype(float)
    return (b - s).rolling(window=period, min_periods=1).sum()


def order_imbalance(buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
    """Per-bar buy/sell imbalance in ``[-1, 1]``.

    +1 = only taker buys, -1 = only taker sells.
    """
    b = ensure_series(buy_volume).astype(float)
    s = ensure_series(sell_volume).astype(float)
    total = b + s
    return safe_divide(b - s, total, fill=0.0)


def large_print_zscore(
    volume: pd.Series, period: int = 96
) -> pd.Series:
    """Rolling z-score of bar volume — useful for tagging unusually large prints."""
    from .indicators import rolling_zscore  # local import to avoid cycles

    return rolling_zscore(volume, period)
