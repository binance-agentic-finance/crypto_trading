"""
Open interest data via binance-cli subprocess → pandas DataFrame.

Atomic source: atomic_strategy_lib/data/open_interest.py
CLI commands replicated:
  current: binance-cli futures-usds open-interest --symbol S
  history: binance-cli futures-usds open-interest-statistics
               --symbol S --period P --limit N

Sample fixture (open-interest-statistics --symbol BTCUSDT --period 1h --limit 2):
    [
      {"symbol": "BTCUSDT", "sumOpenInterest": "60000.0",
       "sumOpenInterestValue": "2100000000.0", "timestamp": 1700000000000},
      {"symbol": "BTCUSDT", "sumOpenInterest": "62000.0",
       "sumOpenInterestValue": "2170000000.0", "timestamp": 1700003600000}
    ]

Expected DataFrame columns (oi_history):
    timestamp, oi_base, oi_value, oi_change_bps
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._subprocess import run_binance_cli
from ._cache import cache_get, cache_set, TTL_OI


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def fetch_open_interest(
    symbol: str,
    *,
    ttl: int = TTL_OI,
    refresh: bool = False,
) -> dict:
    """
    Fetch current open interest snapshot.

    CLI::

        binance-cli futures-usds open-interest --symbol BTCUSDT

    Returns
    -------
    dict
        ``{"symbol": str, "oi_base": float, "timestamp": int | None}``

    Sample fixture (open-interest response):
        {"symbol": "BTCUSDT", "openInterest": "60000.0", "time": 1700000000000}
    """
    key = ("oi_current", symbol)
    if not refresh:
        cached = cache_get(key)
        if cached is not None and not cached.empty:
            return cached.iloc[0].to_dict()

    args = ["futures-usds", "open-interest", "--symbol", symbol]
    raw = run_binance_cli(args)
    if isinstance(raw, list):
        raw = raw[0] if raw else {}

    result = {
        "symbol":    raw.get("symbol", symbol),
        "oi_base":   _safe_float(raw.get("openInterest")),
        "timestamp": int(raw.get("time", 0)) or None,
    }
    df = pd.DataFrame([result])
    cache_set(key, df, ttl=ttl)
    return result


def fetch_oi_history(
    symbol: str,
    period: str = "1h",
    limit: int = 48,
    *,
    ttl: int = TTL_OI,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch open interest history for delta computation.

    CLI::

        binance-cli futures-usds open-interest-statistics \\
            --symbol BTCUSDT --period 1h --limit 48

    Parameters
    ----------
    period : str
        ``"5m"``, ``"15m"``, ``"30m"``, ``"1h"``, ``"2h"``, ``"4h"``,
        ``"6h"``, ``"12h"``, ``"1d"``

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (int ms), oi_base (float), oi_value (float),
                 oi_change_bps (float)
        Sorted chronologically (oldest first).

    Sample fixture::

        [{"symbol": "BTCUSDT", "sumOpenInterest": "60000.0",
          "sumOpenInterestValue": "2100000000.0", "timestamp": 1700000000000}, ...]
    """
    key = ("oi_history", symbol, period, limit)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    args = ["futures-usds", "open-interest-statistics",
            "--symbol", symbol, "--period", period, "--limit", str(limit)]
    raw = run_binance_cli(args)
    items = raw if isinstance(raw, list) else []

    rows = [
        {
            "timestamp": int(item.get("timestamp", 0)),
            "oi_base":   _safe_float(item.get("sumOpenInterest")),
            "oi_value":  _safe_float(item.get("sumOpenInterestValue")),
        }
        for item in items
    ]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp", "oi_base", "oi_value"])

    if not df.empty and len(df) >= 2:
        df = df.sort_values("timestamp").reset_index(drop=True)
        oldest_val = df["oi_value"].iloc[0]
        if oldest_val != 0:
            df["oi_change_bps"] = ((df["oi_value"] - oldest_val) / oldest_val) * 10000
        else:
            df["oi_change_bps"] = 0.0
    elif not df.empty:
        df["oi_change_bps"] = 0.0

    cache_set(key, df, ttl=ttl)
    return df


def oi_delta_pct(df: pd.DataFrame) -> Optional[float]:
    """
    Compute percentage change between oldest and newest OI value in the history DF.

    Parameters
    ----------
    df : pd.DataFrame
        Output from :func:`fetch_oi_history`.

    Returns
    -------
    float | None
        Percentage change or None if not enough data.
    """
    if df is None or len(df) < 2:
        return None
    oldest = float(df["oi_value"].iloc[0])
    newest = float(df["oi_value"].iloc[-1])
    if oldest == 0:
        return None
    return ((newest - oldest) / oldest) * 100
