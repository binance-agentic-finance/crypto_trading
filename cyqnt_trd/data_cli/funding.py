"""
Funding rate data via binance-cli subprocess → pandas DataFrame.

Atomic source: atomic_strategy_lib/data/funding.py
CLI commands replicated:
  history:  binance-cli futures-usds get-funding-rate-history --symbol S --limit N
  current:  binance-cli futures-usds mark-price --symbol S
  info:     binance-cli futures-usds get-funding-rate-info

Sample fixture (funding-rate-history --symbol BTCUSDT --limit 2):
    [
      {"symbol": "BTCUSDT", "fundingRate": "0.00010000",
       "fundingTime": 1700000000000, "markPrice": "35000.0"},
      {"symbol": "BTCUSDT", "fundingRate": "-0.00005000",
       "fundingTime": 1699971200000, "markPrice": "34900.0"}
    ]

Expected DataFrame columns (funding_history):
    symbol, rate, timestamp, mark_price
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._subprocess import run_binance_cli
from ._cache import cache_get, cache_set, TTL_FUNDING


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def fetch_funding_history(
    symbol: str,
    limit: int = 100,
    *,
    ttl: int = TTL_FUNDING,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch historical funding rates for a symbol.

    CLI::

        binance-cli futures-usds get-funding-rate-history \\
            --symbol BTCUSDT --limit 100

    Returns
    -------
    pd.DataFrame
        Columns: symbol, rate (float), timestamp (int ms), mark_price (float)
        Sorted chronologically (oldest first).

    Sample fixture::

        [{"symbol": "BTCUSDT", "fundingRate": "0.00010000",
          "fundingTime": 1700000000000, "markPrice": "35000.0"}, ...]
    """
    key = ("funding_history", symbol, limit)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    args = ["futures-usds", "get-funding-rate-history",
            "--symbol", symbol, "--limit", str(limit)]
    raw = run_binance_cli(args)
    items = raw if isinstance(raw, list) else []

    rows = [
        {
            "symbol":     item.get("symbol", symbol),
            "rate":       _safe_float(item.get("fundingRate")),
            "timestamp":  int(item.get("fundingTime", 0)),
            "mark_price": _safe_float(item.get("markPrice")),
        }
        for item in items
    ]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["symbol", "rate", "timestamp", "mark_price"])
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    cache_set(key, df, ttl=ttl)
    return df


def fetch_funding_rate(
    symbol: str,
    *,
    ttl: int = TTL_FUNDING,
    refresh: bool = False,
) -> float:
    """
    Fetch the current (latest) funding rate for a symbol.

    CLI::

        binance-cli futures-usds mark-price --symbol BTCUSDT

    Returns
    -------
    float
        Current funding rate (e.g. 0.0001 = 0.01 %).

    Sample fixture (mark-price response):
        {"symbol": "BTCUSDT", "markPrice": "35000.0",
         "lastFundingRate": "0.00010000", "nextFundingTime": 1700028800000}
    """
    key = ("funding_rate_current", symbol)
    if not refresh:
        cached = cache_get(key)
        if cached is not None and not cached.empty:
            return float(cached.iloc[0]["rate"])

    args = ["futures-usds", "mark-price", "--symbol", symbol]
    raw = run_binance_cli(args)
    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    rate = _safe_float(raw.get("lastFundingRate"))
    df = pd.DataFrame([{
        "symbol":           raw.get("symbol", symbol),
        "rate":             rate,
        "mark_price":       _safe_float(raw.get("markPrice")),
        "next_funding_time": int(raw.get("nextFundingTime", 0)) or None,
        "timestamp":        int(raw.get("time", 0)) or None,
    }])
    cache_set(key, df, ttl=ttl)
    return rate


def fetch_funding_rate_info() -> list[dict]:
    """
    Fetch funding rate limits/intervals for all symbols.

    CLI::

        binance-cli futures-usds get-funding-rate-info

    Returns
    -------
    list[dict]
        Raw list of per-symbol funding info dicts.
    """
    args = ["futures-usds", "get-funding-rate-info"]
    raw = run_binance_cli(args)
    return raw if isinstance(raw, list) else []
