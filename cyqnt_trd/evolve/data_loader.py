"""Data loader: download 90-day OHLCV from Binance, cache to parquet, IS/OOS split.

90 days × 15m = 8640 bars. Binance kline endpoint caps at 1000 bars/call,
so we paginate by walking ``startTime`` forward.

Cache layout:
    sessions/<id>/data/<symbol>_<interval>.parquet  ← full 90-day series
    sessions/<id>/data/meta.json                    ← {is_split_idx, ...}

Splits:
    IS  = first 60 days
    OOS = last 30 days
The split is computed once at session init so it's stable across all
re-loads (even if you re-download fresher bars).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

from cyqnt_trd.standard_bot.data._ssl_utils import resolve_ca_bundle

logger = logging.getLogger(__name__)

_SPOT_KLINES_URL = "https://api.binance.com/api/v3/uiKlines"


# ── Time helpers ──────────────────────────────────────────────────────────

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


# ── Download ──────────────────────────────────────────────────────────────

def _fetch_klines_page(
    symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000
) -> list:
    """Single page fetch from Binance spot API."""
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": int(limit),
    }
    resp = requests.get(
        _SPOT_KLINES_URL,
        params=params,
        timeout=30,
        verify=resolve_ca_bundle(),
    )
    resp.raise_for_status()
    return resp.json()


def fetch_ohlcv(
    symbol: str,
    interval: str,
    days: int = 90,
    *,
    end_ts_ms: int | None = None,
    sleep_between_calls: float = 0.05,
) -> pd.DataFrame:
    """Download ``days`` of OHLCV from Binance with pagination.

    Returns a DataFrame with DatetimeIndex (UTC) and columns
    ['open', 'high', 'low', 'close', 'volume', 'close_time'].
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported interval: {interval}")

    if end_ts_ms is None:
        end_ts_ms = int(time.time() * 1000)
    start_ts_ms = end_ts_ms - days * 86_400_000
    step_ms = INTERVAL_MS[interval] * 1000  # max 1000 bars per call

    all_rows: list = []
    cursor = start_ts_ms
    page = 0
    while cursor < end_ts_ms:
        page_end = min(cursor + step_ms, end_ts_ms)
        rows = _fetch_klines_page(symbol, interval, cursor, page_end, limit=1000)
        if not rows:
            # No data in this window — advance cursor to avoid infinite loop
            cursor = page_end
            continue
        all_rows.extend(rows)
        last_open = int(rows[-1][0])
        next_cursor = last_open + INTERVAL_MS[interval]
        if next_cursor <= cursor:
            cursor = page_end  # safety
        else:
            cursor = next_cursor
        page += 1
        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)
        # Safety: cap at 100 pages (~100k bars)
        if page > 100:
            logger.warning("fetch_ohlcv aborted at page 100, returning %d rows", len(all_rows))
            break

    if not all_rows:
        raise RuntimeError(f"no data for {symbol} {interval} in last {days}d")

    # Dedup by open_time (Binance can return overlapping pages on boundary)
    seen = {}
    for r in all_rows:
        seen[int(r[0])] = r
    rows_sorted = [seen[k] for k in sorted(seen.keys())]

    df = pd.DataFrame(
        [
            {
                "open_time": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
                "close_time": int(r[6]),
            }
            for r in rows_sorted
        ]
    )
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"])
    df.index.name = "timestamp"
    logger.info(
        "fetched %d bars for %s %s (%s → %s)",
        len(df), symbol, interval, df.index[0], df.index[-1],
    )
    return df


# ── Cache + IS/OOS split ──────────────────────────────────────────────────

def cache_path(session_dir: Path, symbol: str, interval: str) -> Path:
    return session_dir / "data" / f"{symbol.upper()}_{interval}.parquet"


def load_or_download(
    *,
    session_dir: Path,
    symbol: str,
    interval: str,
    days: int = 90,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load cached parquet if present (and not refresh), else download + cache."""
    p = cache_path(session_dir, symbol, interval)
    if p.exists() and not refresh:
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        return df
    p.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_ohlcv(symbol, interval, days=days)
    df.to_parquet(p)
    return df


def split_is_oos(
    df: pd.DataFrame, *, is_days: int = 60, oos_days: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into in-sample (first is_days) and out-of-sample (last oos_days).

    Strict: the IS+OOS time spans must be contiguous and non-overlapping.
    Both blocks are fully OOS-protected when used during evolution: only
    df_is is passed to evolve, df_oos is held for final validation.
    """
    if df.empty:
        return df, df
    end = df.index[-1]
    oos_start = end - pd.Timedelta(days=oos_days)
    is_start = oos_start - pd.Timedelta(days=is_days)
    df_is = df.loc[(df.index >= is_start) & (df.index < oos_start)].copy()
    df_oos = df.loc[df.index >= oos_start].copy()
    return df_is, df_oos
