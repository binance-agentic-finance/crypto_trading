"""Take-profit target helpers + exit pricing math.

Includes blocks-style single-trade pricing (`fixed_stop_price`,
`fixed_tp_price`, `atr_stop_price`, `risk_reward`, `passes_min_rr`,
`compute_partial_close_levels`) with L1 parity vs `cyqnt_trd.blocks.exit`.
"""

from __future__ import annotations

from dataclasses import dataclass


_LONG = "long"
_SHORT = "short"


def _check_side(side: str) -> str:
    side = str(side).lower()
    if side not in {_LONG, _SHORT}:
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    return side


@dataclass(frozen=True)
class TakeProfitStep:
    target_price: float
    close_fraction: float


def graduated_take_profit(
    entry_price: float,
    direction: str,
    target_returns: tuple[float, ...],
    close_fractions: tuple[float, ...],
) -> tuple[TakeProfitStep, ...]:
    if len(target_returns) != len(close_fractions):
        raise ValueError("target_returns and close_fractions must have the same length")
    if not 0.999 <= sum(close_fractions) <= 1.001:
        raise ValueError("close_fractions must sum to 1.0")
    if direction not in {"long", "short"}:
        raise ValueError("direction must be 'long' or 'short'")
    multiplier = 1.0 if direction == "long" else -1.0
    return tuple(
        TakeProfitStep(entry_price * (1.0 + multiplier * ret), fraction)
        for ret, fraction in zip(target_returns, close_fractions, strict=True)
    )


# ---------------------------------------------------------------------------
# Blocks-style pricing helpers (L1 parity vs cyqnt_trd.blocks.exit)
# ---------------------------------------------------------------------------


def fixed_stop_price(entry: float, pct: float, side: str = _LONG) -> float:
    """Stop price at `pct` (e.g. 0.02 for 2%) below long entry / above short entry."""
    if pct < 0:
        raise ValueError(f"pct must be >= 0, got {pct}")
    side = _check_side(side)
    if side == _LONG:
        return float(entry) * (1.0 - float(pct))
    return float(entry) * (1.0 + float(pct))


def fixed_tp_price(entry: float, pct: float, side: str = _LONG) -> float:
    """Take-profit price at `pct` favourable distance from entry."""
    if pct < 0:
        raise ValueError(f"pct must be >= 0, got {pct}")
    side = _check_side(side)
    if side == _LONG:
        return float(entry) * (1.0 + float(pct))
    return float(entry) * (1.0 - float(pct))


def atr_stop_price(
    entry: float,
    atr_value: float,
    multiplier: float = 2.0,
    side: str = _LONG,
) -> float:
    """ATR-distance stop: entry ± multiplier * atr_value."""
    if atr_value < 0:
        raise ValueError(f"atr_value must be >= 0, got {atr_value}")
    if multiplier < 0:
        raise ValueError(f"multiplier must be >= 0, got {multiplier}")
    side = _check_side(side)
    if side == _LONG:
        return float(entry) - float(multiplier) * float(atr_value)
    return float(entry) + float(multiplier) * float(atr_value)


def risk_reward(entry: float, stop: float, tp: float) -> float:
    """Reward-to-risk ratio: |tp-entry| / |entry-stop|."""
    risk = abs(float(entry) - float(stop))
    reward = abs(float(tp) - float(entry))
    if risk == 0.0:
        return float("inf") if reward > 0 else 0.0
    return reward / risk


def passes_min_rr(entry: float, stop: float, tp: float, min_rr: float = 1.5) -> bool:
    return risk_reward(entry, stop, tp) >= float(min_rr)


def compute_partial_close_levels(
    entry: float,
    targets: list[tuple[float, float]],
    side: str = _LONG,
) -> list[tuple[float, float]]:
    """Convert (gain_pct, close_ratio) pairs to (price, close_ratio) pairs."""
    side = _check_side(side)
    out: list[tuple[float, float]] = []
    for gain_pct, close_ratio in targets:
        if not 0.0 < float(close_ratio) <= 1.0:
            raise ValueError(f"close_ratio must be in (0, 1], got {close_ratio}")
        out.append((fixed_tp_price(entry, gain_pct, side), float(close_ratio)))
    return out


__all__ = [
    "TakeProfitStep",
    "atr_stop_price",
    "compute_partial_close_levels",
    "fixed_stop_price",
    "fixed_tp_price",
    "graduated_take_profit",
    "passes_min_rr",
    "risk_reward",
]

