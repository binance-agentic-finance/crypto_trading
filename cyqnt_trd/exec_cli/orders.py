"""L5-06 to L5-10: Order placement — market, limit, stop, cancel-all, partial-close.

Port of atomic_strategy_lib.execution.orders.

SAFETY CONTRACT
---------------
All public functions default to ``dry_run=True``.
When dry_run=True, the function:
  1. Prints the command that *would* be executed.
  2. Returns a fake OrderResult(success=True) — NO subprocess is invoked.
To place a real order you MUST pass ``dry_run=False`` explicitly.
"""

from __future__ import annotations

from typing import Optional

from ._subprocess import (
    CLIError,
    OrderResult,
    _parse_order_result,
    run_cli,
)

_VALID_SIDES = frozenset({"BUY", "SELL"})


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_order(symbol: str, side: str, qty: float) -> None:
    if not symbol:
        raise ValueError("symbol must not be empty")
    if side not in _VALID_SIDES:
        raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")
    if qty <= 0:
        raise ValueError(f"qty must be > 0, got {qty}")


# ---------------------------------------------------------------------------
# L5-06: Market order
# ---------------------------------------------------------------------------

def market_order(
    symbol: str,
    side: str,
    qty: float,
    market: str = "futures",
    position_side: Optional[str] = None,
    reduce_only: bool = False,
    profile: str = "default",
    binary: str = "binance-cli",
    dry_run: bool = True,
) -> OrderResult:
    """Place a MARKET order.

    Args:
        symbol:        Trading pair, e.g. "BTCUSDT"
        side:          "BUY" or "SELL"
        qty:           Quantity in base asset (must be > 0)
        market:        "futures" (default) or "spot"
        position_side: "LONG" / "SHORT" for futures hedge mode
        reduce_only:   True → close-only (will not increase position)
        profile:       binance-cli profile name
        binary:        CLI binary name (override for testing)
        dry_run:       **True by default** — print command, no execution.
                       Pass ``dry_run=False`` to place a real order.

    Returns:
        OrderResult with executed_qty=qty and success=True on dry_run.
    """
    _validate_order(symbol, side, qty)

    if market == "futures":
        cmd = [
            binary, "futures-usds", "new-order",
            "--symbol", symbol, "--side", side,
            "--type", "MARKET", "--quantity", str(qty),
        ]
        if position_side:
            cmd.extend(["--positionSide", position_side])
        if reduce_only:
            cmd.extend(["--reduceOnly", "true"])
    else:
        cmd = [
            binary, "spot", "order", "new",
            "--symbol", symbol, "--side", side,
            "--type", "MARKET", "--quantity", str(qty),
        ]

    if dry_run:
        print(f"[DRY RUN] market_order: {' '.join(cmd)}")
        return OrderResult(
            success=True,
            order_id="DRY_RUN",
            executed_qty=qty,
            executed_price=0.0,
            fee=0.0,
            raw_response={
                "dryRun": True, "symbol": symbol, "side": side,
                "type": "MARKET", "quantity": qty,
            },
        )

    try:
        raw = run_cli(cmd, profile=profile)
        return _parse_order_result(raw)
    except CLIError as exc:
        return OrderResult(
            success=False,
            raw_response={"error": str(exc), "command": cmd},
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# L5-07: Limit order
# ---------------------------------------------------------------------------

def limit_order(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    market: str = "futures",
    position_side: Optional[str] = None,
    time_in_force: str = "GTC",
    profile: str = "default",
    binary: str = "binance-cli",
    dry_run: bool = True,
) -> OrderResult:
    """Place a LIMIT order.

    Args:
        price: limit price (must be > 0)
        dry_run: **True by default** — print command, no execution.
    """
    _validate_order(symbol, side, qty)
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    if market == "futures":
        cmd = [
            binary, "futures-usds", "new-order",
            "--symbol", symbol, "--side", side,
            "--type", "LIMIT", "--timeInForce", time_in_force,
            "--quantity", str(qty), "--price", str(price),
        ]
        if position_side:
            cmd.extend(["--positionSide", position_side])
    else:
        cmd = [
            binary, "spot", "order", "new",
            "--symbol", symbol, "--side", side,
            "--type", "LIMIT", "--timeInForce", time_in_force,
            "--quantity", str(qty), "--price", str(price),
        ]

    if dry_run:
        print(f"[DRY RUN] limit_order: {' '.join(cmd)}")
        return OrderResult(
            success=True,
            order_id="DRY_RUN",
            executed_qty=0.0,   # limit orders are not filled immediately
            executed_price=price,
            fee=0.0,
            raw_response={
                "dryRun": True, "symbol": symbol, "side": side,
                "type": "LIMIT", "quantity": qty, "price": price,
            },
        )

    try:
        raw = run_cli(cmd, profile=profile)
        return _parse_order_result(raw)
    except CLIError as exc:
        return OrderResult(
            success=False,
            raw_response={"error": str(exc), "command": cmd},
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# L5-08: Stop market order
# ---------------------------------------------------------------------------

def stop_market_order(
    symbol: str,
    side: str,
    stop_price: float,
    qty: float = 0.0,
    market: str = "futures",
    position_side: Optional[str] = None,
    working_type: str = "CONTRACT_PRICE",
    close_position: bool = False,
    profile: str = "default",
    binary: str = "binance-cli",
    dry_run: bool = True,
) -> OrderResult:
    """Place a STOP_MARKET order (futures stop-loss).

    Args:
        stop_price:     Trigger price (must be > 0)
        qty:            Quantity to close. Ignored if close_position=True.
        close_position: True → closePosition=true (closes entire position)
        dry_run:        **True by default** — print command, no execution.
    """
    if side not in _VALID_SIDES:
        raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")
    if stop_price <= 0:
        raise ValueError(f"stop_price must be > 0, got {stop_price}")
    if not close_position and qty <= 0:
        raise ValueError("qty must be > 0 when close_position=False")

    cmd = [
        binary, "futures-usds", "new-order",
        "--symbol", symbol, "--side", side,
        "--type", "STOP_MARKET",
        "--stopPrice", str(stop_price),
        "--workingType", working_type,
    ]
    if position_side:
        cmd.extend(["--positionSide", position_side])
    if close_position:
        cmd.extend(["--closePosition", "true"])
    elif qty > 0:
        cmd.extend(["--quantity", str(qty)])

    if dry_run:
        print(f"[DRY RUN] stop_market_order: {' '.join(cmd)}")
        return OrderResult(
            success=True,
            order_id="DRY_RUN",
            executed_qty=qty,
            executed_price=stop_price,
            fee=0.0,
            raw_response={
                "dryRun": True, "symbol": symbol, "side": side,
                "type": "STOP_MARKET", "stopPrice": stop_price, "quantity": qty,
            },
        )

    try:
        raw = run_cli(cmd, profile=profile)
        return _parse_order_result(raw)
    except CLIError as exc:
        return OrderResult(
            success=False,
            raw_response={"error": str(exc), "command": cmd},
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# L5-09: Cancel all open orders
# ---------------------------------------------------------------------------

def cancel_all(
    symbol: str,
    market: str = "futures",
    profile: str = "default",
    binary: str = "binance-cli",
    dry_run: bool = True,
) -> OrderResult:
    """Cancel all open orders for a symbol.

    Args:
        dry_run: **True by default** — print command, no execution.
    """
    if not symbol:
        raise ValueError("symbol must not be empty")

    if market == "futures":
        cmd = [binary, "futures-usds", "cancel-all-open-orders", "--symbol", symbol]
    else:
        cmd = [binary, "spot", "cancel-open-orders", "--symbol", symbol]

    if dry_run:
        print(f"[DRY RUN] cancel_all: {' '.join(cmd)}")
        return OrderResult(
            success=True,
            order_id=None,
            raw_response={"dryRun": True, "symbol": symbol, "action": "cancel_all"},
        )

    try:
        raw = run_cli(cmd, profile=profile)
        code = raw.get("code", 0) if isinstance(raw, dict) else 0
        has_error = bool(
            (isinstance(raw, dict) and raw.get("error"))
            or (isinstance(code, int) and code < 0)
        )
        return OrderResult(
            success=not has_error,
            raw_response=raw if isinstance(raw, dict) else {"result": str(raw)},
            error=(raw.get("msg") or raw.get("error"))
            if (has_error and isinstance(raw, dict))
            else None,
        )
    except CLIError as exc:
        return OrderResult(
            success=False,
            raw_response={"error": str(exc), "command": cmd},
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# L5-10: Partial close
# ---------------------------------------------------------------------------

def partial_close(
    symbol: str,
    side: str,
    qty: float,
    market: str = "futures",
    position_side: Optional[str] = None,
    profile: str = "default",
    binary: str = "binance-cli",
    dry_run: bool = True,
) -> OrderResult:
    """Close a partial quantity of an open position via market order.

    Args:
        side:          Closing side — "SELL" to close LONG, "BUY" to close SHORT.
        qty:           Quantity to close (must be > 0).
        position_side: "LONG" or "SHORT" for hedge-mode accounts.
        dry_run:       **True by default** — print command, no execution.
    """
    _validate_order(symbol, side, qty)

    return market_order(
        symbol=symbol,
        side=side,
        qty=qty,
        market=market,
        position_side=position_side,
        reduce_only=True,
        profile=profile,
        binary=binary,
        dry_run=dry_run,
    )
