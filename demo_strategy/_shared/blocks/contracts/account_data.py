"""AccountData — framework-fetched account snapshot.

UI card: "account state" — shows what balances/positions the template
is allowed to read.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AccountData:
    balances:  dict[str, float] = field(default_factory=dict)
    positions: dict[str, float] = field(default_factory=dict)   # signed qty
