"""Funding rate scanner — extreme-funding mean-reversion entries.

Long signal fires when the most recent funding rate is in the
`bearish_squeeze` zone (deeply negative funding → shorts crowded /
bears about to be squeezed → contrarian long entry).

Note: per `library.features.derivatives.funding_rate_state`,
`bullish_squeeze` actually labels HIGH POSITIVE funding (longs over-
extended, contrarian short bias) and `bearish_squeeze` labels DEEPLY
NEGATIVE funding (shorts over-extended, contrarian long bias). The
label names describe *which side gets squeezed*, not the resulting
trade direction.

The case expects a `funding_rate` column on `data` (fraction, e.g.
0.0001 = 1 bp). Use the adapter chain to populate it:

    from ai_pro_trading_library.library.data import (
        BinanceRestAdapter, MarketBundleClient, bars_to_frame,
    )
    from ai_pro_trading_library.library.features.derivatives import (
        funding_rate_state,
    )

    client = MarketBundleClient(BinanceRestAdapter())
    bars = client.klines("BTCUSDT", "8h", limit=200)
    df = bars_to_frame(bars)
    funding = client.funding("BTCUSDT", limit=200)
    df = df.merge(
        funding.set_index("timestamp")["funding_rate"],
        left_on="close_time", right_index=True, how="left",
    )

If `funding_rate` is missing, the scanner returns all-False so smoke
backtests on plain OHLCV (e.g. `runtime/smoke.py::synthetic_ohlcv`)
do not raise.
"""

from __future__ import annotations

import pandas as pd

from ai_pro_trading_library.library.core.protocols import StrategySpec
from ai_pro_trading_library.library.features.derivatives import funding_rate_state


def build_signal(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    if "funding_rate" not in data.columns:
        return pd.Series(False, index=data.index)
    p = spec.parameters
    high_threshold_bps = float(p.get("high_threshold_bps", 5.0))
    low_threshold_bps = float(p.get("low_threshold_bps", -5.0))
    state = funding_rate_state(
        data["funding_rate"],
        high_threshold_bps=high_threshold_bps,
        low_threshold_bps=low_threshold_bps,
    )
    # `bearish_squeeze` label = deeply negative funding (bears squeezed) →
    # contrarian long entry. See module docstring for label semantics.
    return (state == "bearish_squeeze").fillna(False)
