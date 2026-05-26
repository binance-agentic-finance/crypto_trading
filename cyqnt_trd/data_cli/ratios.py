"""
Derivatives ratio data via binance-cli subprocess → pandas DataFrame.

Atomic source: atomic_strategy_lib/data/derivatives.py
CLI commands replicated:
  long-short ratio:            binance-cli futures-usds long-short-ratio --symbol S --period P --limit N
  top trader LS ratio:         binance-cli futures-usds top-trader-long-short-ratio-positions ...
  taker buy/sell volume:       binance-cli futures-usds taker-buy-sell-volume ...
  basis:                       binance-cli futures-usds basis --pair P --contract-type C --period P --limit N
  leverage brackets:           binance-cli futures-usds notional-and-leverage-brackets --symbol S

Sample fixture (long-short-ratio --symbol BTCUSDT --period 1h --limit 2):
    [
      {"symbol": "BTCUSDT", "longAccount": "0.52", "shortAccount": "0.48",
       "longShortRatio": "1.0833", "timestamp": 1700000000000},
      {"symbol": "BTCUSDT", "longAccount": "0.50", "shortAccount": "0.50",
       "longShortRatio": "1.0000", "timestamp": 1700003600000}
    ]

Expected DataFrame columns:
    timestamp, long_account, short_account, long_short_ratio
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


def _parse_ls_rows(raw, symbol_fallback: str = "") -> list[dict]:
    items = raw if isinstance(raw, list) else []
    return [
        {
            "timestamp":        int(item.get("timestamp", 0)),
            "long_account":     _safe_float(item.get("longAccount")),
            "short_account":    _safe_float(item.get("shortAccount")),
            "long_short_ratio": _safe_float(item.get("longShortRatio")),
        }
        for item in items
    ]


def fetch_long_short_ratio(
    symbol: str,
    period: str = "1h",
    limit: int = 30,
    *,
    ttl: int = TTL_OI,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch global long/short account ratio.

    CLI::

        binance-cli futures-usds long-short-ratio \\
            --symbol BTCUSDT --period 1h --limit 30

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, long_account, short_account, long_short_ratio
        Sorted chronologically (oldest first).

    Sample fixture::

        [{"longAccount": "0.52", "shortAccount": "0.48",
          "longShortRatio": "1.0833", "timestamp": 1700000000000}, ...]
    """
    key = ("ls_ratio", symbol, period, limit)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    args = ["futures-usds", "long-short-ratio",
            "--symbol", symbol, "--period", period, "--limit", str(limit)]
    raw = run_binance_cli(args)
    rows = _parse_ls_rows(raw)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp", "long_account", "short_account", "long_short_ratio"])
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    cache_set(key, df, ttl=ttl)
    return df


def fetch_top_trader_ls_ratio(
    symbol: str,
    period: str = "1h",
    limit: int = 30,
    *,
    ttl: int = TTL_OI,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch top trader long/short position ratio.

    CLI::

        binance-cli futures-usds top-trader-long-short-ratio-positions \\
            --symbol BTCUSDT --period 1h --limit 30

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, long_account, short_account, long_short_ratio
    """
    key = ("top_trader_ls", symbol, period, limit)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    args = ["futures-usds", "top-trader-long-short-ratio-positions",
            "--symbol", symbol, "--period", period, "--limit", str(limit)]
    raw = run_binance_cli(args)
    rows = _parse_ls_rows(raw)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp", "long_account", "short_account", "long_short_ratio"])
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    cache_set(key, df, ttl=ttl)
    return df


def fetch_taker_volume(
    symbol: str,
    period: str = "1h",
    limit: int = 30,
    *,
    ttl: int = TTL_OI,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch taker buy/sell volume.

    CLI::

        binance-cli futures-usds taker-buy-sell-volume \\
            --symbol BTCUSDT --period 1h --limit 30

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, buy_vol, sell_vol, buy_sell_ratio

    Sample fixture::

        [{"timestamp": 1700000000000, "buyVol": "1200.0",
          "sellVol": "1000.0", "buySellRatio": "1.2"}, ...]
    """
    key = ("taker_vol", symbol, period, limit)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    args = ["futures-usds", "taker-buy-sell-volume",
            "--symbol", symbol, "--period", period, "--limit", str(limit)]
    raw = run_binance_cli(args)
    items = raw if isinstance(raw, list) else []
    rows = [
        {
            "timestamp":     int(item.get("timestamp", 0)),
            "buy_vol":        _safe_float(item.get("buyVol")),
            "sell_vol":       _safe_float(item.get("sellVol")),
            "buy_sell_ratio": _safe_float(item.get("buySellRatio")),
        }
        for item in items
    ]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp", "buy_vol", "sell_vol", "buy_sell_ratio"])
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    cache_set(key, df, ttl=ttl)
    return df


def fetch_basis(
    pair: str,
    period: str = "1h",
    limit: int = 30,
    contract_type: str = "PERPETUAL",
    *,
    ttl: int = TTL_OI,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch basis (futures premium over spot).

    CLI::

        binance-cli futures-usds basis \\
            --pair BTCUSDT --contract-type PERPETUAL --period 1h --limit 30

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, index_price, contract_price, basis, basis_rate
    """
    key = ("basis", pair, period, limit, contract_type)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    args = ["futures-usds", "basis",
            "--pair", pair, "--contract-type", contract_type,
            "--period", period, "--limit", str(limit)]
    raw = run_binance_cli(args)
    items = raw if isinstance(raw, list) else []
    rows = [
        {
            "timestamp":      int(item.get("timestamp", 0)),
            "index_price":    _safe_float(item.get("indexPrice")),
            "contract_price": _safe_float(item.get("contractPrice")),
            "basis":          _safe_float(item.get("basis")),
            "basis_rate":     _safe_float(item.get("basisRate")),
        }
        for item in items
    ]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp", "index_price", "contract_price", "basis", "basis_rate"])
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    cache_set(key, df, ttl=ttl)
    return df


def fetch_leverage_brackets(symbol: str) -> list[dict]:
    """
    Fetch notional and leverage brackets for a symbol.

    CLI::

        binance-cli futures-usds notional-and-leverage-brackets --symbol BTCUSDT

    Returns
    -------
    list[dict]
        Raw brackets list.
    """
    args = ["futures-usds", "notional-and-leverage-brackets", "--symbol", symbol]
    raw = run_binance_cli(args)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "brackets" in raw:
        return raw["brackets"]
    return []
