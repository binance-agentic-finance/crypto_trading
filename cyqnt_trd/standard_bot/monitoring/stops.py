"""
cyqnt_trd.standard_bot.monitoring.stops
========================================

Stop-order health monitoring — detect stale / misplaced stops and query
open orders via ``binance-cli``.

Ported from atomic_strategy_lib.monitoring.stops (L7-06, L7-07).

``query_open_orders`` delegates to the CLI subprocess layer.  If
``cyqnt_trd.data_cli`` is available (task #33) it is used automatically;
otherwise a local fallback that calls ``binance-cli`` directly is used so
this module works before data_cli is fully ported.
"""

from __future__ import annotations

import json
import subprocess
from typing import Optional

__all__ = ["STOP_TYPES", "stale_stop_detect", "query_open_orders"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOP_TYPES: frozenset[str] = frozenset(
    {
        "STOP_MARKET",
        "STOP_LOSS_LIMIT",
        "TAKE_PROFIT_LIMIT",
        "TAKE_PROFIT_MARKET",
        "TRAILING_STOP_MARKET",
    }
)


# ---------------------------------------------------------------------------
# L7-06  Stale stop detection
# ---------------------------------------------------------------------------

def stale_stop_detect(open_orders: list[dict]) -> dict:
    """Check whether any open order is a stop-type order.

    Parameters
    ----------
    open_orders:
        List of order dicts as returned by the exchange (or
        :func:`query_open_orders`).  Each dict must contain ``"type"``.

    Returns
    -------
    dict with keys:
        ``has_stop``    – bool
        ``stop_count``  – int
        ``stop_orders`` – list of summarised stop order dicts
    """
    stops = [o for o in open_orders if o.get("type", "") in STOP_TYPES]

    return {
        "has_stop": len(stops) > 0,
        "stop_count": len(stops),
        "stop_orders": [
            {
                "orderId": o.get("orderId"),
                "type": o.get("type"),
                "stopPrice": o.get("stopPrice"),
                "side": o.get("side"),
            }
            for o in stops
        ],
    }


# ---------------------------------------------------------------------------
# L7-07  Open orders query
# ---------------------------------------------------------------------------

def query_open_orders(
    symbol: str,
    market: str = "futures",
    profile: Optional[str] = None,
    binary: str = "binance-cli",
) -> list[dict]:
    """Fetch all open orders for a symbol via ``binance-cli``.

    Parameters
    ----------
    symbol:
        Trading pair, e.g. ``"BTCUSDT"``.
    market:
        ``"futures"`` (default) or ``"spot"``.
    profile:
        Optional CLI profile name (``--profile <name>``).
    binary:
        CLI binary name (default ``"binance-cli"``).

    Returns
    -------
    list[dict]
        List of open order dicts.  Returns an empty list on any error
        so callers do not need to handle exceptions.
    """
    # Prefer data_cli if already ported and importable
    try:
        from cyqnt_trd.data_cli import query_open_orders as _dc_query  # type: ignore[import]
        return _dc_query(symbol=symbol, market=market, profile=profile)
    except ImportError:
        pass

    # Fallback: direct subprocess
    if market == "futures":
        cmd = [binary, "futures-usds", "get-all-open-orders", "--symbol", symbol]
    else:
        cmd = [binary, "spot", "get-open-orders", "--symbol", symbol]

    if profile:
        cmd.insert(1, "--profile")
        cmd.insert(2, profile)

    cmd.append("--json")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    stdout = result.stdout.strip()
    if not stdout:
        return []

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if data.get("error"):
            return []
        return [data]
    return []
