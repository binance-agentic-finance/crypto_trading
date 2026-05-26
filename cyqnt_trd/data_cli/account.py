"""
Account balance and position data via binance-cli subprocess → pandas DataFrame.

Atomic source: atomic_strategy_lib/data/account.py
CLI commands replicated:
  futures balances: binance-cli futures-usds futures-account-balance-v3
  spot balances:    binance-cli spot get-account
  wallet balances:  binance-cli wallet user-asset
  positions:        binance-cli futures-usds position-information-v3 [--symbol S]
  account info:     binance-cli futures-usds account-information-v3
                    binance-cli spot get-account

Sample fixture (futures-account-balance-v3):
    [
      {"asset": "USDT", "balance": "1000.0", "availableBalance": "800.0",
       "crossWalletBalance": "1000.0"},
      {"asset": "BNB", "balance": "2.5", "availableBalance": "2.5",
       "crossWalletBalance": "2.5"}
    ]

Expected DataFrame columns (balances):
    asset, free, locked, total

Expected DataFrame columns (positions):
    symbol, direction, entry_price, quantity, unrealized_pnl,
    leverage, margin_type, notional, mark_price
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._subprocess import run_binance_cli
from ._cache import cache_get, cache_set, TTL_ACCOUNT


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def fetch_account_balance(
    profile: str = "default",
    wallet: str = "futures",
    *,
    ttl: int = TTL_ACCOUNT,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch account balances.

    Parameters
    ----------
    wallet : str
        ``"futures"`` (default) | ``"spot"`` | ``"wallet"``

    CLI::

        # futures
        binance-cli futures-usds futures-account-balance-v3
        # spot
        binance-cli spot get-account
        # wallet
        binance-cli wallet user-asset

    Returns
    -------
    pd.DataFrame
        Columns: asset, free, locked, total
        Only non-zero balances are returned.

    Sample fixture (futures)::

        [{"asset": "USDT", "balance": "1000.0", "availableBalance": "800.0"}, ...]

    Expected result::

        asset   free    locked  total
        USDT    800.0   200.0   1000.0
    """
    key = ("balance", profile, wallet)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    if wallet == "spot":
        args = ["spot", "get-account"]
        raw = run_binance_cli(args)
        items = raw.get("balances", []) if isinstance(raw, dict) else []
        rows = []
        for b in items:
            free = _safe_float(b.get("free"))
            locked = _safe_float(b.get("locked"))
            total = free + locked
            if total > 0:
                rows.append({"asset": b.get("asset", ""),
                             "free": free, "locked": locked, "total": total})

    elif wallet == "wallet":
        args = ["wallet", "user-asset"]
        raw = run_binance_cli(args)
        items = raw if isinstance(raw, list) else []
        rows = [
            {
                "asset":  b.get("asset", ""),
                "free":   _safe_float(b.get("free")),
                "locked": _safe_float(b.get("locked")),
                "total":  _safe_float(b.get("btcValuation")),
            }
            for b in items
        ]
    else:  # futures
        args = ["futures-usds", "futures-account-balance-v3"]
        raw = run_binance_cli(args)
        items = raw if isinstance(raw, list) else []
        rows = []
        for b in items:
            total = _safe_float(b.get("balance"))
            free = _safe_float(b.get("availableBalance"))
            if total > 0:
                rows.append({
                    "asset":  b.get("asset", ""),
                    "free":   free,
                    "locked": total - free,
                    "total":  total,
                })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["asset", "free", "locked", "total"])
    cache_set(key, df, ttl=ttl)
    return df


def fetch_positions(
    symbol: Optional[str] = None,
    profile: str = "default",
    *,
    ttl: int = TTL_ACCOUNT,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch open futures positions.

    CLI::

        binance-cli futures-usds position-information-v3 [--symbol BTCUSDT]

    Returns
    -------
    pd.DataFrame
        Only non-zero positions are returned.
        Columns: symbol, direction, entry_price, quantity, unrealized_pnl,
                 leverage, margin_type, notional, mark_price

    Sample fixture::

        [{"symbol": "BTCUSDT", "positionAmt": "0.5", "entryPrice": "35000.0",
          "unRealizedProfit": "25.0", "leverage": "10", "marginType": "cross",
          "notional": "17500.0", "markPrice": "35050.0"}, ...]
    """
    key = ("positions", profile, symbol)
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    args = ["futures-usds", "position-information-v3"]
    if symbol:
        args.extend(["--symbol", symbol])

    raw = run_binance_cli(args)
    items = raw if isinstance(raw, list) else []

    rows = []
    for p in items:
        amt = _safe_float(p.get("positionAmt"))
        if amt == 0:
            continue
        rows.append({
            "symbol":         p.get("symbol", ""),
            "direction":      "LONG" if amt > 0 else "SHORT",
            "entry_price":    _safe_float(p.get("entryPrice")),
            "quantity":       abs(amt),
            "unrealized_pnl": _safe_float(p.get("unRealizedProfit")),
            "leverage":       int(_safe_float(p.get("leverage"), 1)),
            "margin_type":    p.get("marginType", "").upper(),
            "notional":       _safe_float(p.get("notional")),
            "mark_price":     _safe_float(p.get("markPrice")),
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["symbol", "direction", "entry_price", "quantity",
                 "unrealized_pnl", "leverage", "margin_type", "notional", "mark_price"])
    cache_set(key, df, ttl=ttl)
    return df


def fetch_account_info(
    profile: str = "default",
    wallet: str = "futures",
) -> dict:
    """
    Fetch full account info dict.

    CLI::

        binance-cli futures-usds account-information-v3
        binance-cli spot get-account

    Returns
    -------
    dict
        Raw account info from the CLI.
    """
    if wallet == "futures":
        args = ["futures-usds", "account-information-v3"]
    else:
        args = ["spot", "get-account"]
    return run_binance_cli(args)
