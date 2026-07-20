"""Module ② DATA — same shape as btc_multi_factor_trend/m02_data.py.

Difference: multiple symbols per run instead of one, so we don't cache
the primary_tf name in the config globally — it's read fresh each call.
Additionally we pull the 24h ticker to compute volume signals in m03.
"""
from __future__ import annotations

import time

from atomic_strategy_lib.data.market_bundle import (
    kline_fetch_multi_tf,
    funding_rate_current,
    open_interest_fetch,
    ticker_24h_fetch,
)


def fetch_market_bundle(symbol: str, cfg: dict) -> dict:
    tf_cfg = cfg.get("signals", {}).get("timeframes", {})
    primary = tf_cfg.get("primary", "4h")
    confirm = tf_cfg.get("confirmation", ["1h", "1d"])
    all_tfs = sorted({primary, *confirm})

    delay   = cfg.get("_delay_sec", 0.3)
    profile = cfg.get("_profile")

    try:
        tf_candles = kline_fetch_multi_tf(
            symbol, timeframes=all_tfs,
            limits={tf: 200 for tf in all_tfs},
            market="spot", profile=profile,
        ) or {}
    except Exception:
        tf_candles = {}
    time.sleep(delay)

    try:
        ticker = ticker_24h_fetch(symbol, profile=profile)
        time.sleep(delay)
    except Exception:
        ticker = None

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
    price = primary_candles[-1].close if primary_candles else \
        (getattr(ticker, "last_price", 0.0) if ticker else 0.0)

    return {
        "symbol":     symbol,
        "price":      price,
        "primary_tf": primary,
        "tf_candles": tf_candles,
        "ticker":     ticker,
        "funding":    funding,
        "oi":         oi,
    }
