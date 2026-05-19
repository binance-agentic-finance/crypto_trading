"""Order specification helpers.

Lightweight dataclasses describing orders. The actual execution is the
responsibility of :mod:`cyqnt_trd.standard_bot.execution` — this module
just provides ergonomic constructors so user code can write::

    order = exec.market_order("BTCUSDT", side="long", notional=1000)

without remembering all the fields of ``ExecutionIntent``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "OrderSpec",
    "market_order",
    "limit_order",
    "stop_market_order",
    "stop_limit_order",
    "oco_pair",
]


_TIF = {"GTC", "IOC", "FOK", "GTX"}


@dataclass
class OrderSpec:
    """A simple, framework-agnostic order specification.

    The execution layer maps this to ``ExecutionIntent``.
    """

    symbol: str
    side: str  # "long" or "short" (the trade direction)
    order_type: str  # "MARKET" / "LIMIT" / "STOP_MARKET" / "STOP_LIMIT"
    quantity: Optional[float] = None
    notional: Optional[float] = None
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    client_tag: Optional[str] = None
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper()
        if self.side not in ("long", "short"):
            raise ValueError(f"side must be 'long' or 'short', got {self.side!r}")
        if self.order_type not in ("MARKET", "LIMIT", "STOP_MARKET", "STOP_LIMIT"):
            raise ValueError(f"unsupported order_type: {self.order_type!r}")
        if self.time_in_force not in _TIF:
            raise ValueError(f"unsupported time_in_force: {self.time_in_force!r}")
        if self.quantity is None and self.notional is None:
            raise ValueError("provide either quantity or notional")
        if self.order_type in ("LIMIT", "STOP_LIMIT") and self.price is None:
            raise ValueError(f"{self.order_type} order requires a limit price")
        if self.order_type in ("STOP_MARKET", "STOP_LIMIT") and self.stop_price is None:
            raise ValueError(f"{self.order_type} order requires a stop_price")


def market_order(
    symbol: str,
    side: str,
    quantity: Optional[float] = None,
    *,
    notional: Optional[float] = None,
    reduce_only: bool = False,
    client_tag: Optional[str] = None,
) -> OrderSpec:
    """Build a market-order spec."""
    return OrderSpec(
        symbol=symbol,
        side=side,
        order_type="MARKET",
        quantity=quantity,
        notional=notional,
        reduce_only=reduce_only,
        client_tag=client_tag,
    )


def limit_order(
    symbol: str,
    side: str,
    price: float,
    quantity: Optional[float] = None,
    *,
    notional: Optional[float] = None,
    time_in_force: str = "GTC",
    reduce_only: bool = False,
    client_tag: Optional[str] = None,
) -> OrderSpec:
    """Build a limit-order spec."""
    return OrderSpec(
        symbol=symbol,
        side=side,
        order_type="LIMIT",
        quantity=quantity,
        notional=notional,
        price=price,
        time_in_force=time_in_force,
        reduce_only=reduce_only,
        client_tag=client_tag,
    )


def stop_market_order(
    symbol: str,
    side: str,
    stop_price: float,
    quantity: Optional[float] = None,
    *,
    notional: Optional[float] = None,
    reduce_only: bool = True,
    client_tag: Optional[str] = None,
) -> OrderSpec:
    """Build a stop-market spec — used as a stop-loss order."""
    return OrderSpec(
        symbol=symbol,
        side=side,
        order_type="STOP_MARKET",
        quantity=quantity,
        notional=notional,
        stop_price=stop_price,
        reduce_only=reduce_only,
        client_tag=client_tag,
    )


def stop_limit_order(
    symbol: str,
    side: str,
    stop_price: float,
    limit_price: float,
    quantity: Optional[float] = None,
    *,
    notional: Optional[float] = None,
    time_in_force: str = "GTC",
    reduce_only: bool = True,
    client_tag: Optional[str] = None,
) -> OrderSpec:
    """Build a stop-limit spec."""
    return OrderSpec(
        symbol=symbol,
        side=side,
        order_type="STOP_LIMIT",
        quantity=quantity,
        notional=notional,
        price=limit_price,
        stop_price=stop_price,
        time_in_force=time_in_force,
        reduce_only=reduce_only,
        client_tag=client_tag,
    )


def oco_pair(
    symbol: str,
    side: str,
    take_profit_price: float,
    stop_price: float,
    stop_limit_price: Optional[float] = None,
    quantity: Optional[float] = None,
    *,
    notional: Optional[float] = None,
    client_tag: Optional[str] = None,
) -> tuple:
    """Return ``(limit_tp_order, stop_order)`` — a standard OCO bracket.

    *side* is the side of the **open position** (``"long"`` or ``"short"``).
    Both protective legs (TP and stop) are placed on the *opposite* side
    automatically — for a long position, the TP and stop are sells; for a
    short position, they are buys. They are also marked ``reduce_only=True``
    so that the legs only close, never open new exposure.
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    exit_side = "short" if side == "long" else "long"
    tp = limit_order(
        symbol, exit_side, take_profit_price, quantity=quantity,
        notional=notional, reduce_only=True, client_tag=client_tag,
    )
    if stop_limit_price is None:
        stop = stop_market_order(
            symbol, exit_side, stop_price, quantity=quantity,
            notional=notional, client_tag=client_tag,
        )
    else:
        stop = stop_limit_order(
            symbol, exit_side, stop_price, stop_limit_price,
            quantity=quantity, notional=notional, client_tag=client_tag,
        )
    return tp, stop
