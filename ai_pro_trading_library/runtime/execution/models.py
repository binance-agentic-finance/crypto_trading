"""Fee / slippage / fill model Protocols + concrete implementations.

Mirrors the Protocol surface in `cyqnt_trd.standard_bot.simulation.python_live_paper_session`
(FeeModel, SlippageModel, FillModel) but in the new repo's simpler shape.
Backtest engines accept these optionally; if not provided, the engine
uses inline `commission_bps` / `slippage_bps` parameters as before.

Protocols:
- `FeeModel`: returns total fee for a given side / quantity / price.
- `SlippageModel`: adjusts the requested price up (buy) or down (sell).
- `FillModel`: produces a `Fill` from a requested order against a bar.

Concrete implementations:
- `BpsFeeModel(commission_bps)` — fee = qty * price * commission_bps / 10000.
- `BpsSlippageModel(slippage_bps)` — price adjusted by ±slippage_bps / 10000.
- `ImmediateFillModel(fee_model, slippage_model)` — fills at the
  requested_price adjusted by slippage, with fee from the fee model.
  Matches the bar-close fill semantics in `run_long_only_backtest`.

`PendingOrder` and `Fill` dataclasses cover the standard_bot
`PendingOrder` / `PaperFill` field set so callers writing their own
session loop have a portable shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ai_pro_trading_library.library.core.protocols import (
        Bar,
        ExecutionIntent,
    )


@dataclass(frozen=True)
class Fill:
    timestamp: int
    instrument_id: str
    side: str  # "buy" | "sell"
    quantity: float
    price: float
    fee: float


@dataclass(frozen=True)
class PendingOrder:
    """Order awaiting fill on the next bar (for next-open fill models)."""

    intent: "ExecutionIntent"
    submitted_at: int  # bar index when the order was placed
    fill_when: str = "current_close"  # or "next_open"


class FeeModel(Protocol):
    def fee(self, *, side: str, quantity: float, price: float) -> float: ...


class SlippageModel(Protocol):
    def adjust(self, *, side: str, price: float) -> float: ...


class FillModel(Protocol):
    def fill(
        self,
        *,
        side: str,
        quantity: float,
        bar: "Bar",
    ) -> Fill: ...


@dataclass(frozen=True)
class BpsFeeModel:
    """fee = quantity * price * commission_bps / 10_000."""

    commission_bps: float = 0.0

    def fee(self, *, side: str, quantity: float, price: float) -> float:  # noqa: ARG002
        return quantity * price * self.commission_bps / 10_000.0


@dataclass(frozen=True)
class BpsSlippageModel:
    """price * (1 + slippage_bps/10_000) for buys, price * (1 - …) for sells."""

    slippage_bps: float = 0.0

    def adjust(self, *, side: str, price: float) -> float:
        bps = self.slippage_bps / 10_000.0
        if side.lower() == "buy":
            return price * (1.0 + bps)
        return price * (1.0 - bps)


@dataclass(frozen=True)
class ImmediateFillModel:
    """Fill at the bar's close, adjusted by slippage, with model-provided fee.

    Matches the inner loop semantics in `runtime.backtest.long_only_engine.
    run_long_only_backtest`. Use this when porting code from standard_bot
    that expected `(execution_price, fee)` returned per bar.
    """

    fee_model: FeeModel
    slippage_model: SlippageModel

    def fill(self, *, side: str, quantity: float, bar: "Bar") -> Fill:
        adjusted = self.slippage_model.adjust(side=side, price=bar.close)
        fee = self.fee_model.fee(side=side, quantity=quantity, price=adjusted)
        return Fill(
            timestamp=bar.timestamp,
            instrument_id=bar.instrument_id,
            side=side,
            quantity=quantity,
            price=adjusted,
            fee=fee,
        )


__all__ = [
    "BpsFeeModel",
    "BpsSlippageModel",
    "FeeModel",
    "Fill",
    "FillModel",
    "ImmediateFillModel",
    "PendingOrder",
    "SlippageModel",
]
