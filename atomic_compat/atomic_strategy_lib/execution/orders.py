"""shim — atomic.execution.orders.

Atomic uses ``live=False/True`` (default safe = False).
cyqnt_trd uses ``dry_run=True/False`` (default safe = True).
We translate between them so existing case scripts don't have to change.
"""
from cyqnt_trd.exec_cli import (  # noqa: F401
    market_order as _market_order,
    limit_order as _limit_order,
    stop_market_order as _stop_market_order,
    cancel_all as _cancel_all,
    partial_close as _partial_close,
)


def _live_to_dry(live: bool) -> bool:
    """live=True means dry_run=False (real order); live=False means dry_run=True."""
    return not bool(live)


def market_order(symbol, side, quantity, market="futures", position_side=None,
                reduce_only=False, profile="default", binary="binance-cli", live=False):
    """atomic-style market_order. live=False default = dry-run."""
    return _market_order(
        symbol=symbol, side=side, qty=quantity, market=market,
        position_side=position_side, reduce_only=reduce_only,
        profile=profile, binary=binary, dry_run=_live_to_dry(live),
    )


def limit_order(symbol, side, quantity, price, market="futures", position_side=None,
               time_in_force="GTC", profile="default", binary="binance-cli", live=False):
    """atomic-style limit_order."""
    return _limit_order(
        symbol=symbol, side=side, qty=quantity, price=price, market=market,
        position_side=position_side, time_in_force=time_in_force,
        profile=profile, binary=binary, dry_run=_live_to_dry(live),
    )


def stop_order(symbol, side, stop_price, quantity=0, market="futures",
              position_side=None, stop_type="STOP_MARKET",
              working_type="CONTRACT_PRICE", time_in_force="GTC",
              close_position=False, profile="default", binary="binance-cli", live=False):
    """atomic-style stop_order — maps to cyqnt_trd.stop_market_order."""
    return _stop_market_order(
        symbol=symbol, side=side, stop_price=stop_price,
        qty=float(quantity) if quantity else 0.0,
        market=market, position_side=position_side,
        working_type=working_type, close_position=close_position,
        profile=profile, binary=binary, dry_run=_live_to_dry(live),
    )


def cancel_all(symbol, market="futures", profile="default", binary="binance-cli", live=False):
    """atomic-style cancel_all."""
    return _cancel_all(
        symbol=symbol, market=market, profile=profile,
        binary=binary, dry_run=_live_to_dry(live),
    )


def partial_close(symbol, direction, close_pct, current_qty, market="futures",
                 profile="default", binary="binance-cli", live=False):
    """atomic-style partial_close: takes percentage + direction, computes side & qty."""
    close_qty = round(float(current_qty) * float(close_pct) / 100.0, 8)
    if close_qty <= 0:
        from cyqnt_trd.exec_cli._subprocess import OrderResult
        return OrderResult(success=False, raw_response={"error": "zero close qty"})
    close_side = "SELL" if str(direction).upper() == "LONG" else "BUY"
    position_side = str(direction).upper()
    return _partial_close(
        symbol=symbol, side=close_side, qty=close_qty, market=market,
        position_side=position_side, profile=profile,
        binary=binary, dry_run=_live_to_dry(live),
    )
