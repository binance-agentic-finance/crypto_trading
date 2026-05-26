"""
Full market instrument scanning via binance-cli / binance-pro-cli.

Atomic source: atomic_strategy_lib/data/scanner.py
CLI commands replicated:
  futures exchange info: binance-cli futures-usds exchange-information
  spot exchange info:    binance-cli spot exchange-info
  futures all tickers:  binance-cli futures-usds ticker24hr-price-change-statistics
  spot all tickers:     binance-cli spot ticker24hr
  hotrank:              binance-pro-cli search hotrank <market> --lang en --no-compact
  leaderboard:          binance-pro-cli workflow leaderboard --market M --lang en
  top-strategies:       binance-pro-cli workflow top-strategies --direction D

Expected output:
  full_market_scan  → list[str] of symbols
  scan_with_filter  → pd.DataFrame with columns: symbol, volume_quote
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._subprocess import run_binance_cli, run_binance_pro_cli
from ._cache import cache_get, cache_set, TTL_SCANNER, TTL_TICKER


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def full_market_scan(
    market: str = "futures",
    *,
    ttl: int = TTL_SCANNER,
    refresh: bool = False,
) -> list[str]:
    """
    Enumerate all tradable USDT instruments.

    CLI::

        binance-cli futures-usds exchange-information
        binance-cli spot exchange-info

    Returns
    -------
    list[str]
        Sorted list of symbol strings (e.g. ``["BTCUSDT", "ETHUSDT", ...]``).

    Sample fixture (partial)::

        {"symbols": [
          {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT"},
          {"symbol": "XRPUSDT", "status": "TRADING", "quoteAsset": "USDT"},
          ...
        ]}
    """
    key = ("market_scan", market)
    if not refresh:
        cached = cache_get(key)
        if cached is not None and not cached.empty:
            return cached["symbol"].tolist()

    if market == "futures":
        args = ["futures-usds", "exchange-information"]
    else:
        args = ["spot", "exchange-info"]

    raw = run_binance_cli(args)
    symbols_data = raw.get("symbols", []) if isinstance(raw, dict) else []

    result = []
    for s in symbols_data:
        sym = s.get("symbol", "")
        status = s.get("status", s.get("contractStatus", ""))
        quote = s.get("quoteAsset", s.get("marginAsset", ""))
        if status == "TRADING" and quote == "USDT" and sym.endswith("USDT"):
            result.append(sym)
    result.sort()
    df = pd.DataFrame({"symbol": result})
    cache_set(key, df, ttl=ttl)
    return result


def scan_with_filter(
    market: str = "futures",
    min_volume: float = 0.0,
    *,
    ttl: int = TTL_TICKER,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Scan instruments, optionally filtered by 24h quote volume.

    CLI::

        (exchange-info) + (ticker24hr)

    Returns
    -------
    pd.DataFrame
        Columns: symbol (str), volume_quote (float)
        Sorted by volume_quote descending if min_volume > 0, else by symbol.

    Sample fixture::

        symbol      volume_quote
        BTCUSDT     1000000000.0
        ETHUSDT      500000000.0
    """
    key = ("scan_filter", market, min_volume)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    symbols = full_market_scan(market=market, refresh=refresh)

    if min_volume <= 0:
        df = pd.DataFrame({"symbol": symbols, "volume_quote": [0.0] * len(symbols)})
        cache_set(key, df, ttl=ttl)
        return df

    if market == "futures":
        args = ["futures-usds", "ticker24hr-price-change-statistics"]
    else:
        args = ["spot", "ticker24hr"]

    raw = run_binance_cli(args)
    items = raw if isinstance(raw, list) else []
    vol_map = {
        item.get("symbol"): _safe_float(item.get("quoteVolume"))
        for item in items
    }

    rows = [
        {"symbol": s, "volume_quote": vol_map.get(s, 0.0)}
        for s in symbols
        if vol_map.get(s, 0.0) >= min_volume
    ]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["symbol", "volume_quote"])
    if not df.empty:
        df = df.sort_values("volume_quote", ascending=False).reset_index(drop=True)
    cache_set(key, df, ttl=ttl)
    return df


def hotrank_scan(
    market: str = "futures",
    lang: str = "en",
    *,
    ttl: int = TTL_SCANNER,
    refresh: bool = False,
) -> list[dict]:
    """
    Scan for hot/trending coins via binance-pro-cli search hotrank.

    CLI::

        binance-pro-cli search hotrank futures --lang en --no-compact

    Parameters
    ----------
    market : str
        ``"spot"``, ``"futures"``, ``"alpha"``, ``"web3"``, ``"tradingbot"``

    Returns
    -------
    list[dict]
        Raw list of ranked items from the CLI.
    """
    key = ("hotrank_scan", market, lang)
    if not refresh:
        cached = cache_get(key)
        if cached is not None and not cached.empty:
            return cached.to_dict(orient="records")

    args = ["search", "hotrank", market, "--lang", lang, "--no-compact"]
    raw = run_binance_pro_cli(args)
    result = raw if isinstance(raw, list) else []
    if result:
        df = pd.DataFrame(result)
        cache_set(key, df, ttl=ttl)
    return result


def leaderboard_scan(
    market: str = "all",
    lang: str = "en",
) -> dict:
    """
    Run discovery workflow via binance-pro-cli workflow leaderboard.

    CLI::

        binance-pro-cli workflow leaderboard --market all --lang en

    Returns
    -------
    dict
    """
    args = ["workflow", "leaderboard", "--market", market, "--lang", lang]
    return run_binance_pro_cli(args)


def top_strategies_scan(
    direction: str = "neutral",
) -> dict:
    """
    Top trading bot strategies via binance-pro-cli.

    CLI::

        binance-pro-cli workflow top-strategies --direction neutral

    Parameters
    ----------
    direction : str
        ``"neutral"``, ``"long"``, ``"short"``
    """
    args = ["workflow", "top-strategies", "--direction", direction]
    return run_binance_pro_cli(args)
