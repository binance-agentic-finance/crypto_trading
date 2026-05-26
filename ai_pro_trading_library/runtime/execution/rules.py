"""Risk rules for the execution layer.

Mirrors `cyqnt_trd.standard_bot.execution.rules` adapted to the new
repo's simpler `ExecutionIntent` / `AccountSnapshot` shapes:

- `ExecutionIntent` fields here are `symbol / side / notional / leverage /
  order_type / reduce_only / client_order_id` (see
  `library.core.protocols.ExecutionIntent`).
- `AccountSnapshot` exposes `equity / available_balance / positions` —
  no per-asset `balances` dict, so `MaxPositionFractionRule` uses
  `available_balance` directly.
- `AccountPosition` uses `symbol` (not `instrument_id`) and represents
  long via `notional > 0`.

Each rule is a frozen dataclass with a `.validate(intent, snapshot)`
method that returns `None` when the intent is allowed, or a string
reject reason. `ExecutionPlanner` runs intents through a chain of rules
and short-circuits on the first rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ai_pro_trading_library.library.core.protocols import (
    AccountSnapshot,
    ExecutionIntent,
)


class RiskRule(Protocol):
    """Protocol: return None to allow the intent, str reason to reject."""

    def validate(
        self, intent: ExecutionIntent, snapshot: AccountSnapshot | None
    ) -> str | None: ...


@dataclass(frozen=True)
class MaxPositionFractionRule:
    """Reject directional intents that exceed a fraction of available cash.

    Mirrors `standard_bot.execution.rules.MaxPositionFractionRule` with
    the simplification that our `AccountSnapshot` carries a single
    `available_balance` instead of a `balances[base_currency]` dict.
    """

    max_fraction: float = 0.95

    def validate(
        self, intent: ExecutionIntent, snapshot: AccountSnapshot | None
    ) -> str | None:
        if snapshot is None or intent.reduce_only:
            return None
        if not 0 < self.max_fraction <= 1:
            return "invalid_max_fraction"
        if snapshot.available_balance <= 0:
            return "insufficient_cash"
        if intent.notional > snapshot.available_balance * self.max_fraction + 1e-9:
            return "max_position_fraction_exceeded"
        return None


@dataclass(frozen=True)
class LongOnlySinglePositionRule:
    """Prevent duplicate long entries when the symbol is already held long."""

    def validate(
        self, intent: ExecutionIntent, snapshot: AccountSnapshot | None
    ) -> str | None:
        if intent.side.lower() != "buy" or snapshot is None:
            return None
        for position in snapshot.positions:
            if position.symbol == intent.symbol and position.notional > 0:
                return "position_exists"
        return None


@dataclass(frozen=True)
class InstrumentWhitelistRule:
    """Restrict execution to an explicit symbol allowlist."""

    symbols: tuple[str, ...] = field(default_factory=tuple)

    def validate(
        self, intent: ExecutionIntent, snapshot: AccountSnapshot | None  # noqa: ARG002
    ) -> str | None:
        allowed = {s.upper() for s in self.symbols}
        if not allowed:
            return "empty_instrument_whitelist"
        if intent.symbol.upper() not in allowed:
            return "instrument_not_whitelisted"
        return None


@dataclass(frozen=True)
class MaxAbsoluteNotionalRule:
    """Bound a single intent's notional in quote currency."""

    max_notional: float

    def validate(
        self, intent: ExecutionIntent, snapshot: AccountSnapshot | None  # noqa: ARG002
    ) -> str | None:
        if intent.reduce_only:
            return None
        if self.max_notional <= 0:
            return "invalid_max_notional"
        if intent.notional > self.max_notional + 1e-9:
            return "max_absolute_notional_exceeded"
        return None


@dataclass
class ExecutionPlanner:
    """Run intents through a chain of risk rules.

    Mirrors `standard_bot.execution.interfaces.ExecutionPlanner` but
    operates on `ExecutionIntent` lists directly (the new repo's
    strategies emit intents, not `SignalBatch` objects).

    `plan(intents, snapshot)` returns `(approved, rejections)` where
    `approved` is the list of intents that passed all rules, and
    `rejections` is `[(intent, reason), ...]` for short-circuited ones.
    """

    rules: tuple[RiskRule, ...] = field(default_factory=tuple)

    def plan(
        self,
        intents: Sequence[ExecutionIntent],
        snapshot: AccountSnapshot | None = None,
    ) -> tuple[list[ExecutionIntent], list[tuple[ExecutionIntent, str]]]:
        approved: list[ExecutionIntent] = []
        rejections: list[tuple[ExecutionIntent, str]] = []
        for intent in intents:
            reason: str | None = None
            for rule in self.rules:
                reason = rule.validate(intent, snapshot)
                if reason is not None:
                    break
            if reason is None:
                approved.append(intent)
            else:
                rejections.append((intent, reason))
        return approved, rejections


__all__ = [
    "ExecutionPlanner",
    "InstrumentWhitelistRule",
    "LongOnlySinglePositionRule",
    "MaxAbsoluteNotionalRule",
    "MaxPositionFractionRule",
    "RiskRule",
]
