"""Module ② DATA — fetch market data needed by the 6 signal groups.

What we pull for one symbol:
  - kline batches across every timeframe in `signals.timeframes.*`
    (primary + confirmation + entry)
  - current funding rate (perp)
  - open interest snapshot

All fetches delegate to `atomic_strategy_lib` — no direct HTTP here.
"""
from __future__ import annotations

import time

from atomic_strategy_lib.data.market_bundle import (
    kline_fetch_multi_tf,
    funding_rate_current,
    open_interest_fetch,
)


def fetch_market_bundle(symbol: str, cfg: dict) -> dict:
    """Return a bundle keyed by:
        - price:      latest close on the primary timeframe (float)
        - tf_candles: {tf → [Candle, …]}
        - funding:    funding rate object (or None if fetch failed)
        - oi:         OI snapshot dict (or None)
    """
    tf_cfg  = cfg.get("signals", {}).get("timeframes", {})
    primary = tf_cfg.get("primary", "4h")
    confirm = tf_cfg.get("confirmation", ["1h", "1d"])
    entry   = tf_cfg.get("entry", "15m")
    all_tfs = sorted({primary, entry, *confirm})

    delay   = cfg.get("_delay_sec", 0.3)
    profile = cfg.get("_profile")

    tf_candles = kline_fetch_multi_tf(
        symbol,
        timeframes=all_tfs,
        limits={tf: 250 for tf in all_tfs},
        market="spot",
        profile=profile,
    ) or {}
    time.sleep(delay)

    try:
        funding = funding_rate_current(symbol, profile=profile)
        time.sleep(delay)
    except Exception:
        funding = None

    try:
        oi = open_interest_fetch(symbol, profile=profile)
        time.sleep(delay)
    except Exception:
        oi = None

    primary_candles = tf_candles.get(primary, [])
    price = primary_candles[-1].close if primary_candles else 0.0

    return {
        "symbol":     symbol,
        "price":      price,
        "primary_tf": primary,
        "tf_candles": tf_candles,
        "funding":    funding,
        "oi":         oi,
    }
