"""
Regime / macro-state data via PUBLIC endpoints → typed, cached pandas DataFrames.

These are the ``R`` (regime / macro) strategy-type axes that had **no fetcher
anywhere** in the repo, yet are ``✅ BACKTESTABLE`` per the bn-data catalog
(deep daily public history, real walk-forward possible on this machine):

  * Fear & Greed index  — public sub: ``api.alternative.me/fng``  (deep daily)
  * AHR999 index        — reconstructed from public BTC daily close

Both are **stdlib-only** (``urllib``), so this module adds no new runtime
dependency — mirroring ``data_cli/_vendor/binance_bigdata_client.py``.

Availability (bn-data)
----------------------
Internal endpoints exist (``market/fearAndGreedHistory``,
``ahr999/getAhr999List``) but are unreachable off-VPN. The public substitutes
below carry equivalent deep daily history and are the reachable path for real
backtests, exactly as ``策略开发/_shared/public_data.py`` documents.

  * Fear & Greed: ``https://api.alternative.me/fng/?limit=0&format=json``
    → deep daily history from 2018-02-01. This is the SAME upstream Binance's
      internal ``market/fearAndGreed`` mirrors.
  * AHR999: no free index API. Reconstructed from BTC daily close using the
    canonical published formula (see ``ahr999`` below). The internal
    ``ahr999/getAhr999List`` is the authoritative source when on-VPN.

Envelope / empty-frame policy (matches data_cli/news.py)
--------------------------------------------------------
Any transport error, non-2xx, or empty payload returns an **empty typed
DataFrame** rather than raising. Empty frames are never cached, so a later
successful call still populates the cache.

Point-in-time note
-------------------
Both series are daily and lookahead-safe *as stored* (each row is dated by its
own settlement day). AHR999 uses only trailing BTC close (200-day geometric
mean + genesis-age model), so row ``t`` depends on closes ``<= t`` only.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Optional

import pandas as pd

from ._cache import cache_get, cache_set, TTL_REGIME

__all__ = [
    "fetch_fear_greed",
    "fetch_ahr999",
    "FEAR_GREED_COLUMNS",
    "AHR999_COLUMNS",
]

FEAR_GREED_COLUMNS = ["timestamp", "date", "value", "classification"]
AHR999_COLUMNS = ["timestamp", "date", "close", "ahr999", "geomean_200d", "model_price"]

# Public endpoints (stdlib urllib, no auth) ---------------------------------
_FNG_URL = "https://api.alternative.me/fng/"
_BTC_KLINE_URL = "https://api.binance.com/api/v3/klines"

# Bitcoin genesis block: 2009-01-03. AHR999 "coin age" counts days from here.
_BTC_GENESIS_MS = 1230940800_000  # 2009-01-03T00:00:00Z in ms

_UA = "cyqnt_trd/data_cli.regime (public-only)"


def _empty(columns) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _get_json(url: str, timeout: int = 20):
    """GET *url*, parse JSON. Returns parsed body or ``None`` on any error."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        return None


def _as_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Fear & Greed  (✅ BACKTESTABLE — deep daily, public alternative.me)
# ---------------------------------------------------------------------------
def fetch_fear_greed(
    limit: int = 0,
    *,
    ttl: int = TTL_REGIME,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch the crypto Fear & Greed index (daily).

    Public source::

        GET https://api.alternative.me/fng/?limit=0&format=json

    Parameters
    ----------
    limit:
        Number of most-recent days. ``0`` (default) = full history
        (deep daily from 2018-02-01).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (int ms), date (str YYYY-MM-DD), value (int 0-100),
        classification (str). Sorted chronologically (oldest first).

    Notes
    -----
    Equivalent to internal ``market/fearAndGreedHistory``. On transport error
    or empty payload, returns an empty typed frame (never cached).
    """
    key = ("fear_greed", int(limit))
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    body = _get_json(f"{_FNG_URL}?limit={int(limit)}&format=json")
    items = (body or {}).get("data") if isinstance(body, dict) else None
    if not items:
        return _empty(FEAR_GREED_COLUMNS)

    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        ts_sec = int(_as_float(it.get("timestamp")))
        ts_ms = ts_sec * 1000
        rows.append({
            "timestamp":      ts_ms,
            "date":           pd.Timestamp(ts_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
            "value":          int(_as_float(it.get("value"))),
            "classification": str(it.get("value_classification", "")),
        })
    df = pd.DataFrame(rows) if rows else _empty(FEAR_GREED_COLUMNS)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    cache_set(key, df, ttl=ttl)
    return df


# ---------------------------------------------------------------------------
# AHR999  (✅ BACKTESTABLE — reconstructed from public BTC daily close)
# ---------------------------------------------------------------------------
def _fetch_btc_daily_close(limit: int = 1000) -> pd.DataFrame:
    """Public BTC daily OHLCV close → DataFrame[timestamp(ms), close]."""
    limit = max(1, min(int(limit), 1000))  # Binance klines hard cap
    url = f"{_BTC_KLINE_URL}?symbol=BTCUSDT&interval=1d&limit={limit}"
    raw = _get_json(url)
    if not isinstance(raw, list) or not raw:
        return _empty(["timestamp", "close"])
    rows = [
        {"timestamp": int(k[6]), "close": _as_float(k[4])}  # k[6]=close_time, k[4]=close
        for k in raw if isinstance(k, (list, tuple)) and len(k) >= 7
    ]
    df = pd.DataFrame(rows) if rows else _empty(["timestamp", "close"])
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_ahr999(
    limit: int = 1000,
    *,
    ttl: int = TTL_REGIME,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Reconstruct the AHR999 valuation index from public BTC daily close.

    Formula (canonical, as published by author ``ahr999``)::

        ahr999 = (price / geomean_200d) * (price / model_price)
        model_price = 10 ** (5.84 * log10(coin_age_days) - 17.01)

    where ``coin_age_days`` is days since Bitcoin genesis (2009-01-03) and
    ``geomean_200d`` is the 200-day geometric mean of daily close. Below ~0.45
    is historically "bottom / DCA" territory; above ~1.2 is "overheated".

    Parameters
    ----------
    limit:
        Days of BTC close to fetch (max 1000 = Binance cap). The first 200
        rows are consumed as warmup for the geometric mean and dropped.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (int ms), date (str), close, ahr999, geomean_200d,
        model_price. Sorted chronologically (oldest first). Warmup rows with an
        undefined 200-day mean are dropped.

    Notes
    -----
    Public substitute for internal ``ahr999/getAhr999List``. Lookahead-safe:
    row ``t`` uses only closes ``<= t``. Returns empty typed frame if the BTC
    close fetch fails or yields < 200 rows.
    """
    key = ("ahr999", int(limit))
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    btc = _fetch_btc_daily_close(limit)
    if btc.empty or len(btc) < 200:
        return _empty(AHR999_COLUMNS)

    close = btc["close"].astype(float)
    # 200-day geometric mean = exp(rolling mean of log close)
    geomean_200d = pd.Series(
        pd.Series(close.apply(lambda x: math.log(x) if x > 0 else float("nan")))
        .rolling(window=200, min_periods=200).mean().apply(math.exp),
        index=btc.index,
    )

    # coin-age model price from genesis
    age_days = ((btc["timestamp"] - _BTC_GENESIS_MS) / 86_400_000).clip(lower=1)
    model_price = age_days.apply(lambda d: 10 ** (5.84 * math.log10(d) - 17.01))

    ahr999 = (close / geomean_200d) * (close / model_price)

    df = pd.DataFrame({
        "timestamp":    btc["timestamp"].astype("int64"),
        "date":         pd.to_datetime(btc["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d"),
        "close":        close,
        "ahr999":       ahr999,
        "geomean_200d": geomean_200d,
        "model_price":  model_price,
    })
    df = df.dropna(subset=["ahr999"]).reset_index(drop=True)
    if df.empty:
        return _empty(AHR999_COLUMNS)
    cache_set(key, df, ttl=ttl)
    return df
