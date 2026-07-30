"""
In-memory DataFrame cache with per-key TTL.

Keys are arbitrary hashable tuples, typically:
    (function_name, symbol, interval, limit, market, ...)

Values are ``pandas.DataFrame`` objects stored alongside an expiry
timestamp (``float`` seconds since epoch).

Thread-safety
-------------
A single ``threading.Lock`` guards all reads and writes.  This is safe
for multi-threaded scanner loops but does NOT prevent parallel fetches for
the same key (two threads may both decide the cache is stale and fetch
concurrently).  That is acceptable — the last writer wins and results are
idempotent.

Persistence (optional)
-----------------------
Set ``CYQNT_TRD_PERSIST_CACHE=1`` to also write/read JSON files under
``~/.cache/cyqnt_trd/data_cli/``.  Disabled by default.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# TTL presets (seconds) — callers can pass custom ttl= to override
# ---------------------------------------------------------------------------
TTL_KLINE: dict[str, int] = {
    "1m": 45, "3m": 90, "5m": 120, "15m": 300, "30m": 300,
    "1h": 600, "2h": 900, "4h": 1200, "6h": 1200, "8h": 1800,
    "12h": 2400, "1d": 3600, "3d": 7200, "1w": 7200, "1M": 86400,
}
TTL_FUNDING = 300
TTL_OI = 300
TTL_TICKER = 30
TTL_ORDERBOOK = 5
TTL_SCANNER = 120
TTL_PRO = 60
TTL_ACCOUNT = 10  # balances/positions change quickly
TTL_NEWS = 300  # Square news/social — upstream PIT cadence is ~30 min; 5 min is safe
TTL_REGIME = 3600  # Fear&Greed / AHR999 — daily cadence upstream; 1h cache is safe

_LOCK = threading.Lock()
# {key: (DataFrame, expiry_epoch_float)}
_STORE: dict[tuple, tuple[pd.DataFrame, float]] = {}


def cache_get(key: tuple) -> Optional[pd.DataFrame]:
    """Return cached DataFrame if key exists and TTL has not elapsed, else None."""
    with _LOCK:
        entry = _STORE.get(key)
    if entry is None:
        return None
    df, expiry = entry
    if time.monotonic() > expiry:
        return None
    return df


def cache_set(key: tuple, df: pd.DataFrame, ttl: int = 60) -> None:
    """Store *df* under *key* with *ttl* seconds time-to-live."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        # Do not cache empty results — mirrors atomic market_bundle policy
        return
    expiry = time.monotonic() + ttl
    with _LOCK:
        _STORE[key] = (df, expiry)


def cache_clear(key: Optional[tuple] = None) -> None:
    """Clear a specific key or the entire cache if key is None."""
    with _LOCK:
        if key is None:
            _STORE.clear()
        else:
            _STORE.pop(key, None)


def cache_size() -> int:
    """Return number of entries currently in the cache."""
    with _LOCK:
        return len(_STORE)


def kline_ttl(interval: str) -> int:
    """Return per-interval TTL seconds for kline data."""
    return TTL_KLINE.get(interval, 600)
