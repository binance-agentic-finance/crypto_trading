"""SQLite-backed cache decorator over any data adapter.

Per `docs/architecture/data-layer-design.md`, the cache layer is **a
decorator over any adapter**, not a separate adapter type. Cache key is
`(method_name, request_kwargs)`; stored value is the pickled return
object with a `fetched_at` timestamp for TTL checks.

Replaces the per-endpoint `read_klines / write_klines / read_funding /
write_funding / ...` API of `atomic_strategy_lib.data.market_cache` with
a single generic decorator that works for any adapter method.

Default TTLs (per `docs/architecture/dependency-strategy.md` decision):

| Method | TTL |
| --- | --- |
| `fetch_klines` | 60 s (latest bar may still be partial) |
| `fetch_klines_multi_tf` | 60 s |
| `fetch_funding_history` | 1 h |
| `fetch_premium_index` | 30 s |
| `fetch_open_interest` | 60 s |
| `fetch_oi_history` | 5 min |
| `fetch_long_short_ratio` | 5 min |
| `fetch_taker_buy_sell` | 5 min |
| `fetch_24h_tickers` | 30 s |
| `fetch_ticker_price` | 5 s |
| `fetch_exchange_info` | 1 d |
| `fetch_orderbook` | 0 (never cache — too volatile) |
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS adapter_cache (
    cache_key TEXT PRIMARY KEY,
    method_name TEXT NOT NULL,
    payload BLOB NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_method ON adapter_cache(method_name);
"""

_DEFAULT_TTLS: dict[str, int] = {
    "fetch_klines": 60,
    "fetch_klines_multi_tf": 60,
    "fetch_funding_history": 3600,
    "fetch_premium_index": 30,
    "fetch_open_interest": 60,
    "fetch_oi_history": 300,
    "fetch_long_short_ratio": 300,
    "fetch_taker_buy_sell": 300,
    "fetch_24h_tickers": 30,
    "fetch_ticker_price": 5,
    "fetch_exchange_info": 86_400,
    "fetch_orderbook": 0,  # never cache
}


def _resolve_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("AI_PRO_TRADING_CACHE_PATH")
    if override:
        return Path(override)
    return Path.home() / ".ai_pro_trading_library" / "cache.sqlite"


def _kwargs_signature(req: Any) -> str:
    """Build a stable string fingerprint for the request object."""
    if is_dataclass(req) and not isinstance(req, type):
        d = asdict(req)
    elif isinstance(req, dict):
        d = req
    elif isinstance(req, (list, tuple)):
        d = {"args": list(req)}
    else:
        d = {"value": repr(req)}
    return json.dumps(d, sort_keys=True, default=str)


def _cache_key(method_name: str, *args: Any, **kwargs: Any) -> str:
    payload = {
        "method": method_name,
        "args": [_kwargs_signature(a) for a in args],
        "kwargs": {k: _kwargs_signature(v) for k, v in kwargs.items()},
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class CachedAdapter:
    """SQLite-backed cache decorator. Wrap any adapter (`MarketDataAdapter` /
    `DerivativesDataAdapter` / ...); intercept method calls, return cached
    result if fresh, otherwise call the wrapped adapter and store.

    Example:
        from library.data.adapters.binance_rest import BinanceRestAdapter
        from library.data.adapters.market_cache import CachedAdapter

        upstream = BinanceRestAdapter()
        adapter = CachedAdapter(upstream)
        bars = adapter.fetch_klines(KlineRequest("BTCUSDT", "1h", limit=500))
    """

    def __init__(
        self,
        wrapped: Any,
        path: str | Path | None = None,
        ttls: dict[str, int] | None = None,
    ) -> None:
        self._wrapped = wrapped
        self._path = _resolve_path(path)
        self._ttls = {**_DEFAULT_TTLS, **(ttls or {})}
        self._lock = threading.Lock()
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        path = getattr(self._local, "path", None)
        if conn is not None and path == self._path:
            return conn
        if conn is not None:
            conn.close()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        self._local.conn = conn
        self._local.path = self._path
        return conn

    # ------------------------------------------------------------------
    # Cache primitives
    # ------------------------------------------------------------------

    def _read(self, key: str, ttl: int) -> Any:
        if ttl <= 0:
            return _MISS
        cur = self._conn().execute(
            "SELECT payload, fetched_at FROM adapter_cache WHERE cache_key = ?",
            (key,),
        )
        row = cur.fetchone()
        if row is None:
            return _MISS
        payload, fetched_at = row
        # Use float seconds so a 1.05s gap counts as expired vs ttl=1
        age = time.time() - float(fetched_at)
        if age >= ttl:
            return _MISS
        try:
            return pickle.loads(payload)
        except Exception:
            return _MISS

    def _write(self, key: str, method_name: str, value: Any) -> None:
        try:
            blob = pickle.dumps(value)
        except Exception:
            return  # un-picklable → silently skip caching
        with self._lock:
            self._conn().execute(
                "INSERT OR REPLACE INTO adapter_cache "
                "(cache_key, method_name, payload, fetched_at) VALUES (?,?,?,?)",
                (key, method_name, blob, time.time()),
            )
            self._conn().commit()

    def invalidate(self, method_name: str | None = None) -> int:
        """Drop cached entries for a method (or all if None). Returns row count."""
        with self._lock:
            if method_name is None:
                cur = self._conn().execute("DELETE FROM adapter_cache")
            else:
                cur = self._conn().execute(
                    "DELETE FROM adapter_cache WHERE method_name = ?", (method_name,)
                )
            self._conn().commit()
            return cur.rowcount

    def stats(self) -> dict[str, int]:
        """Return cache row count per method."""
        cur = self._conn().execute(
            "SELECT method_name, COUNT(*) FROM adapter_cache GROUP BY method_name"
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # Attribute proxy: every method call goes through cache
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Defer to wrapped adapter for any unknown attribute
        target = getattr(self._wrapped, name)
        if not callable(target):
            return target
        ttl = self._ttls.get(name, 0)
        if ttl <= 0:
            return target  # uncached pass-through

        def cached_call(*args: Any, **kwargs: Any) -> Any:
            key = _cache_key(name, *args, **kwargs)
            hit = self._read(key, ttl)
            if hit is not _MISS:
                return hit
            value = target(*args, **kwargs)
            self._write(key, name, value)
            return value

        cached_call.__name__ = name
        cached_call.__qualname__ = f"CachedAdapter.{name}"
        return cached_call


# Sentinel — distinct from None so callers can cache None explicitly
class _Miss:
    pass


_MISS = _Miss()


__all__ = ["CachedAdapter"]
