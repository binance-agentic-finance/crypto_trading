"""
Order book data via binance-cli subprocess → pandas DataFrame.

Atomic source: atomic_strategy_lib/data/orderbook.py
CLI commands replicated:
  futures: binance-cli futures-usds order-book --symbol S --limit N
  spot:    binance-cli spot depth --symbol S --limit N

Sample fixture (order-book response):
    {
      "lastUpdateId": 123456789,
      "bids": [["35000.0", "2.5"], ["34999.0", "1.0"]],
      "asks": [["35001.0", "1.8"], ["35002.0", "3.0"]]
    }

Expected DataFrame columns:
    price (float), qty (float), side (str: "bid" | "ask")
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._subprocess import run_binance_cli
from ._cache import cache_get, cache_set, TTL_ORDERBOOK


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def fetch_orderbook_depth(
    symbol: str,
    limit: int = 20,
    market: str = "futures",
    *,
    ttl: int = TTL_ORDERBOOK,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch order book depth as a tidy DataFrame.

    CLI::

        # futures
        binance-cli futures-usds order-book --symbol BTCUSDT --limit 20
        # spot
        binance-cli spot depth --symbol BTCUSDT --limit 20

    Returns
    -------
    pd.DataFrame
        Columns: price (float), qty (float), side (``"bid"`` | ``"ask"``)
        Bids first (highest price first), then asks (lowest price first).

    Sample fixture::

        {"bids": [["35000.0", "2.5"], ["34999.0", "1.0"]],
         "asks": [["35001.0", "1.8"], ["35002.0", "3.0"]]}

    Expected result::

        price    qty   side
        35000.0  2.5   bid
        34999.0  1.0   bid
        35001.0  1.8   ask
        35002.0  3.0   ask
    """
    key = ("orderbook", symbol, limit, market)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    if market == "futures":
        args = ["futures-usds", "order-book", "--symbol", symbol, "--limit", str(limit)]
    else:
        args = ["spot", "depth", "--symbol", symbol, "--limit", str(limit)]

    raw = run_binance_cli(args)
    if not isinstance(raw, dict):
        raw = {}

    rows = []
    for b in raw.get("bids", []):
        rows.append({"price": _safe_float(b[0]), "qty": _safe_float(b[1]), "side": "bid"})
    for a in raw.get("asks", []):
        rows.append({"price": _safe_float(a[0]), "qty": _safe_float(a[1]), "side": "ask"})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["price", "qty", "side"])
    cache_set(key, df, ttl=ttl)
    return df


def orderbook_imbalance(df: pd.DataFrame, depth: int = 5) -> float:
    """
    Compute bid/ask volume imbalance ratio for top *depth* levels.

    Parameters
    ----------
    df : pd.DataFrame
        Output from :func:`fetch_orderbook_depth`.
    depth : int
        Number of levels per side to include.

    Returns
    -------
    float
        ``bid_vol / ask_vol``.  > 1 means bid-heavy; < 1 means ask-heavy.
        Returns 1.0 if asks are zero (both sides empty).
    """
    if df is None or df.empty:
        return 1.0
    bids = df[df["side"] == "bid"].head(depth)
    asks = df[df["side"] == "ask"].head(depth)
    bid_vol = bids["qty"].sum()
    ask_vol = asks["qty"].sum()
    if ask_vol == 0:
        return float("inf") if bid_vol > 0 else 1.0
    return float(bid_vol / ask_vol)


def orderbook_spread(df: pd.DataFrame) -> float:
    """
    Compute best bid-ask spread in basis points.

    Parameters
    ----------
    df : pd.DataFrame
        Output from :func:`fetch_orderbook_depth`.

    Returns
    -------
    float
        Spread in bps = ((best_ask - best_bid) / mid) * 10000.
    """
    if df is None or df.empty:
        return 0.0
    bids = df[df["side"] == "bid"]
    asks = df[df["side"] == "ask"]
    if bids.empty or asks.empty:
        return 0.0
    best_bid = float(bids["price"].max())
    best_ask = float(asks["price"].min())
    mid = (best_bid + best_ask) / 2
    if mid == 0:
        return 0.0
    return ((best_ask - best_bid) / mid) * 10_000
