"""
Kline / OHLCV data via binance-cli subprocess → pandas DataFrame.

Atomic source: atomic_strategy_lib/data/kline.py
CLI commands replicated:
  spot:    binance-cli spot klines --symbol S --interval I --limit N
  futures: binance-cli futures-usds kline-candlestick-data --symbol S --interval I --limit N
  listing: binance-cli spot klines --symbol S --interval 1M --limit 1

Sample fixture (binance-cli raw array response):
    [
      [1700000000000, "35000.0", "35100.0", "34900.0", "35050.0",
       "100.5", 1700003599999, "3519000.0", 2500, "60.0", "2100000.0", "0"],
      ...
    ]

Expected DataFrame columns:
    open_time, open, high, low, close, volume, quote_volume, close_time, trades
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd

from ._subprocess import run_binance_cli
from ._cache import cache_get, cache_set, kline_ttl

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "_ignore",
]

DEFAULT_MULTI_TF_LIMITS: dict[str, int] = {
    "1m": 60, "5m": 60, "1h": 48, "4h": 24, "1d": 14, "1w": 8,
}


def _parse_raw_klines(raw) -> pd.DataFrame:
    """
    Parse binance-cli kline response into a DataFrame.

    Raw format: list of lists
      [open_time_ms, open, high, low, close, volume, close_time_ms,
       quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]

    Or list of dicts (newer CLI versions may return dicts).

    Returns DataFrame with columns:
        open_time (int ms), open, high, low, close, volume,
        quote_volume (float), close_time (int ms), trades (int)
    """
    if not raw:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close",
                                     "volume", "quote_volume", "close_time", "trades"])

    items = raw if isinstance(raw, list) else []
    rows = []
    for item in items:
        if isinstance(item, list) and len(item) >= 6:
            rows.append({
                "open_time":    int(item[0]),
                "open":         float(item[1]),
                "high":         float(item[2]),
                "low":          float(item[3]),
                "close":        float(item[4]),
                "volume":       float(item[5]),
                "close_time":   int(item[6]) if len(item) > 6 else 0,
                "quote_volume": float(item[7]) if len(item) > 7 else 0.0,
                "trades":       int(item[8]) if len(item) > 8 else 0,
            })
        elif isinstance(item, dict):
            rows.append({
                "open_time":    int(item.get("openTime", item.get("t", 0))),
                "open":         float(item.get("open", item.get("o", 0))),
                "high":         float(item.get("high", item.get("h", 0))),
                "low":          float(item.get("low", item.get("l", 0))),
                "close":        float(item.get("close", item.get("c", 0))),
                "volume":       float(item.get("volume", item.get("v", 0))),
                "close_time":   int(item.get("closeTime", item.get("T", 0))),
                "quote_volume": float(item.get("quoteVolume", item.get("q", 0))),
                "trades":       int(item.get("count", item.get("n", 0))),
            })

    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close",
                                     "volume", "quote_volume", "close_time", "trades"])
    df = pd.DataFrame(rows)
    return df[["open_time", "open", "high", "low", "close", "volume",
               "quote_volume", "close_time", "trades"]]


def fetch_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 100,
    market: str = "futures",
    *,
    ttl: Optional[int] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data.

    Parameters
    ----------
    symbol : str
        e.g. ``"BTCUSDT"``
    interval : str
        e.g. ``"1h"``, ``"15m"``, ``"4h"``
    limit : int
        Number of candles to return (max 1500 for futures, 1000 for spot).
    market : str
        ``"futures"`` (default) or ``"spot"``.
    ttl : int, optional
        Cache TTL seconds.  Defaults to per-interval preset.
    refresh : bool
        If True, bypass the cache.

    Returns
    -------
    pd.DataFrame
        Columns: open_time, open, high, low, close, volume,
                 quote_volume, close_time, trades

    Sample binance-cli fixture::

        binance-cli futures-usds kline-candlestick-data --symbol BTCUSDT \\
            --interval 1h --limit 2
        [
          [1700000000000, "35000.0", "35100.0", "34900.0", "35050.0",
           "100.5", 1700003599999, "3519000.0", 2500, "60.0", "2100000.0", "0"],
          [1700003600000, "35050.0", "35200.0", "35000.0", "35150.0",
           "90.0", 1700007199999, "3163500.0", 2100, "50.0", "1750000.0", "0"]
        ]
    """
    ttl_sec = ttl if ttl is not None else kline_ttl(interval)
    key = ("klines", symbol, interval, limit, market)

    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    if market == "futures":
        args = ["futures-usds", "kline-candlestick-data",
                "--symbol", symbol, "--interval", interval, "--limit", str(limit)]
    else:
        args = ["spot", "klines",
                "--symbol", symbol, "--interval", interval, "--limit", str(limit)]

    raw = run_binance_cli(args)
    df = _parse_raw_klines(raw)
    cache_set(key, df, ttl=ttl_sec)
    return df


def fetch_klines_multi_tf(
    symbol: str,
    timeframes: list[str],
    limits: Optional[dict[str, int]] = None,
    limit: Optional[int] = None,
    market: str = "futures",
    *,
    ttl: Optional[int] = None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch klines across multiple timeframes concurrently.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{timeframe: DataFrame}``
    """
    if limit is not None:
        resolved = {tf: limit for tf in timeframes}
    else:
        resolved = {tf: (limits or DEFAULT_MULTI_TF_LIMITS).get(tf, 100)
                    for tf in timeframes}

    workers = max(1, min(len(timeframes), 6))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            tf: pool.submit(
                fetch_klines, symbol, tf, resolved[tf], market,
                ttl=ttl, refresh=refresh,
            )
            for tf in timeframes
        }
        return {tf: f.result() for tf, f in futs.items()}


def listing_age_fetch(
    symbol: str,
) -> dict:
    """
    Estimate coin listing age via earliest monthly kline.

    CLI: binance-cli spot klines --symbol S --interval 1M --limit 1

    Returns
    -------
    dict
        ``{"days_since_listing": int | None, "first_kline_date": str | None}``
    """
    from datetime import datetime, timezone

    args = ["spot", "klines", "--symbol", symbol, "--interval", "1M", "--limit", "1"]
    raw = run_binance_cli(args)

    if not raw or not isinstance(raw, list) or len(raw) == 0:
        return {"days_since_listing": None, "first_kline_date": None}

    first = raw[0]
    ts = int(first[0]) if isinstance(first, list) else int(first.get("openTime", 0))
    first_date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    days = (now - first_date).days
    return {
        "days_since_listing": days,
        "first_kline_date": first_date.strftime("%Y-%m-%d"),
    }


def df_to_candles(df: pd.DataFrame) -> list[dict]:
    """
    Adapter: convert kline DataFrame to list-of-dicts (atomic Candle style).

    Each dict has keys: timestamp, open, high, low, close, volume, quote_volume, trades.
    Useful for feeding into legacy atomic-style signal functions.
    """
    return df.rename(columns={"open_time": "timestamp"}).to_dict(orient="records")
