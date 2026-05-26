"""
Ticker / price data via binance-cli subprocess → pandas DataFrame.

Atomic source: atomic_strategy_lib/data/ticker.py
CLI commands replicated:
  spot price:    binance-cli spot ticker-price --symbol S
  fut  price:    binance-cli futures-usds symbol-price-ticker --symbol S
  spot 24h:      binance-cli spot ticker24hr --symbol S
  fut  24h:      binance-cli futures-usds ticker24hr-price-change-statistics --symbol S
  spot all 24h:  binance-cli spot ticker24hr  (no --symbol = all)
  fut  all 24h:  binance-cli futures-usds ticker24hr-price-change-statistics

Sample fixture (binance-cli spot ticker24hr --symbol BTCUSDT):
    {
      "symbol": "BTCUSDT",
      "priceChange": "500.0",
      "priceChangePercent": "1.45",
      "weightedAvgPrice": "34800.0",
      "lastPrice": "35000.0",
      "highPrice": "35200.0",
      "lowPrice": "34500.0",
      "volume": "20000.0",
      "quoteVolume": "697000000.0",
      "count": 150000
    }

Expected DataFrame columns (24h ticker):
    symbol, price, change_pct, high_24h, low_24h,
    volume_base, volume_quote, trades, weighted_avg_price
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._subprocess import run_binance_cli
from ._cache import cache_get, cache_set, TTL_TICKER


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _parse_ticker_row(item: dict, symbol_fallback: str = "") -> dict:
    return {
        "symbol":             item.get("symbol", symbol_fallback),
        "price":              _safe_float(item.get("lastPrice", item.get("price"))),
        "change_pct":         _safe_float(item.get("priceChangePercent")),
        "high_24h":           _safe_float(item.get("highPrice")),
        "low_24h":            _safe_float(item.get("lowPrice")),
        "volume_base":        _safe_float(item.get("volume")),
        "volume_quote":       _safe_float(item.get("quoteVolume")),
        "trades":             int(item.get("count", 0)),
        "weighted_avg_price": _safe_float(item.get("weightedAvgPrice")),
    }


def fetch_24h_ticker(
    symbol: str,
    market: str = "futures",
    *,
    ttl: int = TTL_TICKER,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch 24-hour price change statistics for a single symbol.

    CLI::

        # futures
        binance-cli futures-usds ticker24hr-price-change-statistics --symbol BTCUSDT
        # spot
        binance-cli spot ticker24hr --symbol BTCUSDT

    Returns
    -------
    pd.DataFrame
        One row. Columns: symbol, price, change_pct, high_24h, low_24h,
        volume_base, volume_quote, trades, weighted_avg_price

    Sample fixture (single symbol response):
        {"symbol": "BTCUSDT", "lastPrice": "35000.0", "priceChangePercent": "1.45",
         "highPrice": "35200.0", "lowPrice": "34500.0", "volume": "20000.0",
         "quoteVolume": "697000000.0", "count": 150000, "weightedAvgPrice": "34800.0"}
    """
    key = ("ticker_24h", symbol, market)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    if market == "futures":
        args = ["futures-usds", "ticker24hr-price-change-statistics",
                "--symbol", symbol]
    else:
        args = ["spot", "ticker24hr", "--symbol", symbol]

    raw = run_binance_cli(args)
    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    row = _parse_ticker_row(raw, symbol)
    df = pd.DataFrame([row])
    cache_set(key, df, ttl=ttl)
    return df


def fetch_ticker_price(
    symbol: str,
    market: str = "spot",
    *,
    ttl: int = TTL_TICKER,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch latest price for a symbol as a single-row DataFrame.

    Drop-in replacement for ``atomic_strategy_lib.data.ticker.ticker_price_fetch``.
    Atomic's implementation already short-circuited to the 24h endpoint
    (which carries ``lastPrice``) to avoid a second roundtrip; we do the
    same here. For a scalar float, use :func:`fetch_price` instead.

    Parameters
    ----------
    symbol : str
        e.g. ``"BTCUSDT"``.
    market : str
        ``"spot"`` (atomic default) or ``"futures"``.
    ttl, refresh : optional cache controls (forwarded to :func:`fetch_24h_ticker`).

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with the same columns as :func:`fetch_24h_ticker`:
        symbol, price, change_pct, high_24h, low_24h, volume_base,
        volume_quote, trades, weighted_avg_price.
    """
    return fetch_24h_ticker(symbol, market=market, ttl=ttl, refresh=refresh)


def fetch_price(
    symbol: str,
    market: str = "futures",
    *,
    ttl: int = TTL_TICKER,
    refresh: bool = False,
) -> float:
    """
    Fetch latest price for a symbol (convenience wrapper).

    CLI::

        binance-cli futures-usds symbol-price-ticker --symbol BTCUSDT
        binance-cli spot ticker-price --symbol BTCUSDT

    Returns
    -------
    float
    """
    key = ("price", symbol, market)
    if not refresh:
        cached = cache_get(key)
        if cached is not None and not cached.empty:
            return float(cached.iloc[0]["price"])

    if market == "futures":
        args = ["futures-usds", "symbol-price-ticker", "--symbol", symbol]
    else:
        args = ["spot", "ticker-price", "--symbol", symbol]

    raw = run_binance_cli(args)
    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    price = _safe_float(raw.get("price"))
    df = pd.DataFrame([{"symbol": symbol, "price": price}])
    cache_set(key, df, ttl=ttl)
    return price


def fetch_book_ticker(
    symbol: str,
    market: str = "futures",
) -> dict:
    """
    Fetch best bid/ask (book ticker).

    CLI::

        binance-cli futures-usds book-ticker --symbol BTCUSDT
        binance-cli spot ticker-book --symbol BTCUSDT

    Returns
    -------
    dict
        ``{"symbol": str, "bid_price": float, "bid_qty": float,
           "ask_price": float, "ask_qty": float}``
    """
    if market == "futures":
        args = ["futures-usds", "book-ticker", "--symbol", symbol]
    else:
        args = ["spot", "ticker-book", "--symbol", symbol]

    raw = run_binance_cli(args)
    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    return {
        "symbol":    raw.get("symbol", symbol),
        "bid_price": _safe_float(raw.get("bidPrice")),
        "bid_qty":   _safe_float(raw.get("bidQty")),
        "ask_price": _safe_float(raw.get("askPrice")),
        "ask_qty":   _safe_float(raw.get("askQty")),
    }


def fetch_gainers(
    top_n: int = 20,
    min_change_pct: float = 3.0,
    market: str = "futures",
    *,
    ttl: int = TTL_TICKER,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch top gainers by 24h change (all USDT pairs, filtered + sorted).

    CLI::

        binance-cli futures-usds ticker24hr-price-change-statistics
        binance-cli spot ticker24hr

    Returns
    -------
    pd.DataFrame
        Top *top_n* rows, sorted by change_pct desc.
        Columns: symbol, change_pct, volume_quote, price

    Sample fixture (partial all-tickers response):
        [{"symbol": "BTCUSDT", "priceChangePercent": "5.2", "quoteVolume": "1e9", ...}, ...]
    """
    key = ("gainers", market, min_change_pct, top_n)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    if market == "futures":
        args = ["futures-usds", "ticker24hr-price-change-statistics"]
    else:
        args = ["spot", "ticker24hr"]

    raw = run_binance_cli(args)
    items = raw if isinstance(raw, list) else []

    rows = []
    for item in items:
        sym = item.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        change = _safe_float(item.get("priceChangePercent"))
        if change < min_change_pct:
            continue
        rows.append({
            "symbol":       sym,
            "change_pct":   change,
            "volume_quote": _safe_float(item.get("quoteVolume")),
            "price":        _safe_float(item.get("lastPrice")),
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["symbol", "change_pct", "volume_quote", "price"])
    if not df.empty:
        df = df.sort_values("change_pct", ascending=False).head(top_n).reset_index(drop=True)
    cache_set(key, df, ttl=ttl)
    return df
