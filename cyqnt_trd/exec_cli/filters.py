"""L5-03 to L5-05: Exchange filters — fetch, quantize, tick-round.

Port of atomic_strategy_lib.execution.position (exchange_filter_fetch,
quantize, round_to_tick).

These are pure-logic helpers that do NOT require dry_run because they
are read-only (filter fetch) or purely mathematical (quantize/tick-round).
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Optional

from ._subprocess import CLIError, ExchangeFilter, run_cli, safe_float


# ---------------------------------------------------------------------------
# L5-03: Exchange filter fetch
# ---------------------------------------------------------------------------

def exchange_filter_fetch(
    symbol: str,
    market: str = "futures",
    profile: Optional[str] = None,
    binary: str = "binance-cli",
) -> ExchangeFilter:
    """Fetch exchange trading rules for a symbol (step_size, tick_size, etc.).

    Calls ``binance-cli futures-usds exchange-information`` (or spot equivalent),
    parses the LOT_SIZE / PRICE_FILTER / MIN_NOTIONAL filters, and returns an
    ``ExchangeFilter`` dataclass.

    Falls back to conservative defaults if the symbol is not found or the
    CLI call fails — so callers never have to handle None.

    Args:
        symbol: Trading pair, e.g. "BTCUSDT"
        market: "futures" (default) or "spot"
        profile: optional binance-cli profile name
        binary: CLI binary name
    """
    if market == "futures":
        cmd = [binary, "futures-usds", "exchange-information"]
    else:
        cmd = [binary, "spot", "exchange-info", "--symbol", symbol]

    try:
        raw = run_cli(cmd, profile=profile)
    except CLIError:
        return _default_filter(symbol)

    symbols_data = raw.get("symbols", []) if isinstance(raw, dict) else []

    for s in symbols_data:
        if s.get("symbol") != symbol:
            continue

        filters_raw = s.get("filters", [])
        filters: dict[str, dict] = {
            f.get("filterType", ""): f for f in filters_raw
        }

        if market == "futures":
            step_size = safe_float(
                filters.get("LOT_SIZE", {}).get("stepSize")
                or filters.get("MARKET_LOT_SIZE", {}).get("stepSize"),
                0.001,
            )
            tick_size = safe_float(
                filters.get("PRICE_FILTER", {}).get("tickSize"),
                0.01,
            )
            min_notional = safe_float(
                filters.get("MIN_NOTIONAL", {}).get("notional"),
                5.0,
            )
            min_qty = safe_float(
                filters.get("LOT_SIZE", {}).get("minQty")
                or filters.get("MARKET_LOT_SIZE", {}).get("minQty"),
                0.001,
            )
            qty_prec = int(s.get("quantityPrecision", 3))
            price_prec = int(s.get("pricePrecision", 2))
        else:
            step_size = safe_float(
                filters.get("LOT_SIZE", {}).get("stepSize"), 0.001
            )
            tick_size = safe_float(
                filters.get("PRICE_FILTER", {}).get("tickSize"), 0.01
            )
            min_notional = safe_float(
                filters.get("NOTIONAL", {}).get("minNotional")
                or filters.get("MIN_NOTIONAL", {}).get("minNotional"),
                10.0,
            )
            min_qty = safe_float(
                filters.get("LOT_SIZE", {}).get("minQty"), 0.001
            )
            qty_prec = (
                max(0, -int(math.log10(step_size))) if step_size > 0 else 3
            )
            price_prec = (
                max(0, -int(math.log10(tick_size))) if tick_size > 0 else 2
            )

        return ExchangeFilter(
            symbol=symbol,
            step_size=step_size,
            tick_size=tick_size,
            min_notional=min_notional,
            min_qty=min_qty,
            qty_precision=qty_prec,
            price_precision=price_prec,
        )

    # Symbol not found in exchange info → conservative defaults
    return _default_filter(symbol)


def _default_filter(symbol: str) -> ExchangeFilter:
    """Return safe conservative defaults when exchange info is unavailable."""
    return ExchangeFilter(
        symbol=symbol,
        step_size=0.001,
        tick_size=0.01,
        min_notional=5.0,
        min_qty=0.001,
        qty_precision=3,
        price_precision=2,
    )


# ---------------------------------------------------------------------------
# L5-04: Quantize (floor to step size)
# ---------------------------------------------------------------------------

def quantize(value: float, step: float) -> float:
    """Floor ``value`` to the nearest multiple of ``step``.

    Uses ``Decimal`` arithmetic internally to avoid floating-point drift.

    >>> quantize(0.1234, 0.001)
    0.123
    >>> quantize(1.2399, 0.01)
    1.23
    >>> quantize(0.005, 0.001)
    0.005
    """
    if step <= 0:
        return value
    d = Decimal(str(value))
    q = Decimal(str(step))
    return float((d // q) * q)


# ---------------------------------------------------------------------------
# L5-05: Round to tick
# ---------------------------------------------------------------------------

def round_to_tick(price: float, tick_size: float) -> float:
    """Floor ``price`` to the nearest tick (no floating-point drift).

    >>> round_to_tick(29876.543, 0.1)
    29876.5
    >>> round_to_tick(100.0, 0.01)
    100.0
    """
    if tick_size <= 0:
        return price
    steps = math.floor(price / tick_size)
    return steps * tick_size
