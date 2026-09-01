"""Position-sizing helpers.

All functions return a notional USD size given the chosen sizing rule
and account state. They never apply leverage themselves — the caller
should multiply by ``leverage`` if needed and convert to base-asset
quantity using the latest mark price.

Examples
--------
>>> from cyqnt_trd.blocks import sizing
>>> notional = sizing.fixed_pct_of_equity(equity=10_000, pct=0.15)
>>> qty = notional * leverage / mark_price
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ._utils import non_negative_float, positive_int

__all__ = [
    "fixed_pct_of_equity",
    "fixed_amount",
    "atr_position_size",
    "risk_based_size",
    "kelly_fraction",
    "grid_levels",
    "pyramid_add",
    "round_step_size",
    # atomic ports (L4-03 … L4-07)
    "fixed_dollar_loss",
    "fixed_risk_pct",
    "kelly_size",
    "atr_inverse_size",
    "adaptive_stop_pct",
    "leverage_cap",
    "compute_stop_price",
]


def fixed_pct_of_equity(equity: float, pct: float) -> float:
    """Notional = ``equity * pct``."""
    pct = non_negative_float(pct, "pct")
    return float(equity) * pct


def fixed_amount(amount_usd: float) -> float:
    """Notional = constant ``amount_usd``."""
    return non_negative_float(amount_usd, "amount_usd")


def atr_position_size(
    equity: float,
    atr_value: float,
    mark_price: float,
    risk_pct: float = 0.01,
    stop_distance_atr_mult: float = 2.0,
) -> float:
    """Compute position size targeting ``risk_pct * equity`` of dollar risk.

    A common ATR-based rule: ``stop = mark_price - atr * mult``.
    Position size in base-asset is ``risk_dollars / stop_distance``.
    Returns the notional value (size * mark_price).
    """
    risk_pct = non_negative_float(risk_pct, "risk_pct")
    stop_distance_atr_mult = non_negative_float(stop_distance_atr_mult, "stop_distance_atr_mult")
    if mark_price <= 0:
        raise ValueError(f"mark_price must be > 0, got {mark_price}")
    if atr_value <= 0:
        return 0.0
    risk_dollars = float(equity) * risk_pct
    stop_distance = atr_value * stop_distance_atr_mult
    qty = risk_dollars / stop_distance
    return qty * mark_price


def risk_based_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.01,
) -> float:
    """Compute position size where loss-at-stop = ``risk_pct * equity``.

    Returns notional (qty * entry_price).
    """
    risk_pct = non_negative_float(risk_pct, "risk_pct")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be > 0, got {entry_price}")
    distance = abs(float(entry_price) - float(stop_price))
    if distance == 0:
        return 0.0
    qty = (float(equity) * risk_pct) / distance
    return qty * float(entry_price)


def kelly_fraction(
    win_rate: float, avg_win: float, avg_loss: float, fractional: float = 0.5
) -> float:
    """Fractional-Kelly position fraction. Returns a value in ``[0, 1]``.

    *fractional* in ``(0, 1]`` is the user's risk-adjustment multiplier
    (defaults to half-Kelly = 0.5, the standard "safety" choice).
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError(f"win_rate must be within [0, 1], got {win_rate}")
    if avg_loss <= 0:
        raise ValueError(f"avg_loss must be > 0, got {avg_loss}")
    if avg_win <= 0:
        return 0.0
    if not 0.0 < fractional <= 1.0:
        raise ValueError(f"fractional must be within (0, 1], got {fractional}")
    b = avg_win / avg_loss
    f = win_rate - (1.0 - win_rate) / b
    return max(0.0, min(1.0, f * fractional))


def grid_levels(
    center_price: float,
    range_pct: float,
    n_grids: int,
    per_grid_notional: float,
) -> List[Tuple[float, float]]:
    """Build an evenly-spaced limit-buy grid around *center_price*.

    Returns ``[(price, notional), ...]`` from lowest to highest.
    """
    if range_pct <= 0:
        raise ValueError(f"range_pct must be > 0, got {range_pct}")
    n_grids = positive_int(n_grids, "n_grids")
    low = float(center_price) * (1.0 - range_pct)
    high = float(center_price) * (1.0 + range_pct)
    if n_grids == 1:
        return [(float(center_price), float(per_grid_notional))]
    step = (high - low) / (n_grids - 1)
    return [(low + i * step, float(per_grid_notional)) for i in range(n_grids)]


def pyramid_add(
    initial_notional: float,
    add_count: int,
    add_ratio: float = 0.5,
    max_adds: int = 2,
) -> float:
    """Compute the additional notional for the *add_count*-th pyramid add.

    Returns 0 if *add_count* exceeds *max_adds*.
    """
    if add_count < 0:
        raise ValueError(f"add_count must be >= 0, got {add_count}")
    if add_count == 0 or add_count > max_adds:
        return 0.0
    return float(initial_notional) * float(add_ratio)


def round_step_size(qty: float, step_size: float) -> float:
    """Round *qty* down to the nearest *step_size* — matches Binance LOT_SIZE filter."""
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    n_steps = int(qty / step_size)
    return n_steps * step_size


# ===========================================================================
# Atomic ports — L4-03 to L4-07
# ===========================================================================
# Ported 1-to-1 from atomic_strategy_lib/decision/sizing.py.
# Where an equivalent already exists in this module, the atomic function is
# added as a complementary API (different signature / dict return value).
# ===========================================================================


def fixed_dollar_loss(
    max_loss: Optional[float] = None,
    stop_pct: float = 5.0,
    balance: float = 0.0,
    max_leverage: float = 5.0,
    min_leverage: float = 1.0,
    max_loss_usd: Optional[float] = None,
) -> dict:
    """Position sizing via fixed-dollar-loss method (L4-03).

    Ported from ``atomic_strategy_lib.decision.sizing.fixed_dollar_loss``.

    Derives the leverage that would risk *max_loss* dollars on a *stop_pct* stop.
    Clamps to ``[min_leverage, max_leverage]`` and recomputes actual risk.

    Parameters
    ----------
    max_loss:
        Maximum dollar loss per trade (use ``max_loss_usd`` as alias if preferred).
    stop_pct:
        Stop distance as a percentage (e.g. 5.0 = 5 %).
    balance:
        Account balance in USD.
    max_leverage / min_leverage:
        Leverage bounds.

    Returns
    -------
    dict with keys: ``notional``, ``leverage``, ``actual_max_loss``, ``stop_distance``.
    """
    if max_loss is None:
        max_loss = max_loss_usd if max_loss_usd is not None else 0.0

    stop_distance = stop_pct / 100.0 if stop_pct > 0 else 0.05
    notional = max_loss / stop_distance if stop_distance > 0 else 0.0

    lev = notional / balance if balance > 0 else 0.0
    lev = max(min_leverage, min(round(lev, 1), max_leverage))

    actual_notional = balance * lev
    actual_max_loss = actual_notional * stop_distance

    return {
        "notional": round(actual_notional, 2),
        "leverage": lev,
        "actual_max_loss": round(actual_max_loss, 2),
        "stop_distance": stop_distance,
    }


def fixed_risk_pct(
    balance: float,
    risk_pct: float = 2.0,
    stop_pct: float = 5.0,
    max_leverage: float = 5.0,
    leverage: Optional[float] = None,
    stop_distance_pct: Optional[float] = None,
) -> dict:
    """Position sizing: risk a fixed percentage of account balance (L4-03b).

    Ported from ``atomic_strategy_lib.decision.sizing.fixed_risk_pct``.

    Distinct from :func:`risk_based_size` which takes absolute entry/stop prices.
    This variant works purely from percentage parameters and returns leverage.

    Parameters
    ----------
    balance:
        Account balance in USD.
    risk_pct:
        Percentage of balance to risk per trade (e.g. 2.0 = 2 %).
    stop_pct:
        Stop distance as a percentage.
    max_leverage:
        Maximum allowed leverage (overridden by *leverage* if provided).
    stop_distance_pct:
        Alias for *stop_pct*.

    Returns
    -------
    dict with keys: ``notional``, ``leverage``, ``risk_amount``, ``risk_pct_actual``.
    """
    if leverage is not None:
        max_leverage = leverage
    if stop_distance_pct is not None:
        stop_pct = stop_distance_pct

    risk_amount = balance * (risk_pct / 100.0)
    stop_distance = stop_pct / 100.0 if stop_pct > 0 else 0.05

    notional = risk_amount / stop_distance if stop_distance > 0 else 0.0
    lev = notional / balance if balance > 0 else 0.0
    lev = min(round(lev, 1), max_leverage)

    actual_notional = balance * lev
    actual_risk = actual_notional * stop_distance

    return {
        "notional": round(actual_notional, 2),
        "leverage": lev,
        "risk_amount": round(actual_risk, 2),
        "risk_pct_actual": round(actual_risk / balance * 100, 2) if balance > 0 else 0.0,
    }


def kelly_size(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    fraction: float = 0.5,
    balance: float = 1000.0,
    max_leverage: float = 5.0,
) -> dict:
    """Kelly-criterion position sizing returning notional + leverage (L4-05).

    Ported from ``atomic_strategy_lib.decision.sizing.kelly_size``.

    Internally uses the same Kelly formula as :func:`kelly_fraction` but
    returns a richer dict including notional and clamped leverage.

    Parameters
    ----------
    win_rate:
        Historical win rate in [0, 1].
    avg_win_pct:
        Average winning trade size as a percentage (positive).
    avg_loss_pct:
        Average losing trade size as a percentage (positive number).
    fraction:
        Kelly fraction multiplier, e.g. 0.5 = half-Kelly.
    balance:
        Account balance in USD.
    max_leverage:
        Maximum allowed leverage.

    Returns
    -------
    dict with keys: ``kelly_pct``, ``kelly_adjusted_pct``, ``notional``,
    ``leverage``, ``fraction``.  On invalid input: ``reason`` key is added.
    """
    if avg_loss_pct <= 0 or win_rate <= 0 or win_rate >= 1:
        return {"kelly_pct": 0, "notional": 0, "leverage": 0, "reason": "Invalid inputs"}

    b = avg_win_pct / avg_loss_pct
    q = 1.0 - win_rate
    kelly_full = (win_rate * b - q) / b

    if kelly_full <= 0:
        return {"kelly_pct": 0, "notional": 0, "leverage": 0, "reason": "Negative edge — no bet"}

    kelly_adj = kelly_full * fraction
    kelly_adj = max(0.0, min(kelly_adj, 1.0))

    notional = balance * kelly_adj
    lev = min(round(kelly_adj, 2), max_leverage)

    return {
        "kelly_pct": round(kelly_full * 100, 2),
        "kelly_adjusted_pct": round(kelly_adj * 100, 2),
        "notional": round(notional, 2),
        "leverage": lev,
        "fraction": fraction,
    }


def atr_inverse_size(
    balance: float,
    atr_pct: float,
    target_risk_pct: float = 2.0,
    max_leverage: float = 5.0,
    min_leverage: float = 1.0,
) -> dict:
    """Size inversely proportional to ATR percentage (L4-06).

    Ported from ``atomic_strategy_lib.decision.sizing.atr_inverse_size``.

    Distinct from :func:`atr_position_size` which takes an absolute ATR value
    and mark price.  This variant expects *atr_pct* (ATR as % of current price)
    and returns leverage rather than raw notional.

    Higher ATR → smaller position. Lower ATR → larger position.

    Parameters
    ----------
    balance:
        Account balance in USD.
    atr_pct:
        ATR expressed as a percentage of current price (e.g. 2.5 = 2.5 %).
    target_risk_pct:
        Desired risk per trade as % of balance.
    max_leverage / min_leverage:
        Leverage bounds.

    Returns
    -------
    dict with keys: ``notional``, ``leverage``, ``atr_pct``, ``actual_risk``.
    On zero/negative ATR: returns ``reason`` key.
    """
    if atr_pct <= 0:
        return {"notional": 0.0, "leverage": min_leverage, "reason": "ATR unavailable"}

    risk_amount = balance * (target_risk_pct / 100.0)
    notional = risk_amount / (atr_pct / 100.0)

    lev = notional / balance if balance > 0 else 0.0
    lev = max(min_leverage, min(round(lev, 1), max_leverage))

    actual_notional = balance * lev
    actual_risk = actual_notional * (atr_pct / 100.0)

    return {
        "notional": round(actual_notional, 2),
        "leverage": lev,
        "atr_pct": atr_pct,
        "actual_risk": round(actual_risk, 2),
    }


def adaptive_stop_pct(
    default_stop_pct: float = 5.0,
    change_7d: Optional[float] = None,
    volatile_threshold_7d: float = 50.0,
    volatile_stop_pct: float = 12.0,
    days_since_listing: Optional[int] = None,
    new_coin_days: int = 30,
    new_coin_stop_pct: float = 8.0,
) -> dict:
    """Return an adaptive stop percentage based on coin characteristics (L4-07a).

    Ported from ``atomic_strategy_lib.decision.sizing.adaptive_stop_pct``.

    Priority:

    1. **New coin** (listed < *new_coin_days* ago) → *new_coin_stop_pct* (widest).
    2. **High volatility** (|7-day change| > *volatile_threshold_7d*) → *volatile_stop_pct*.
    3. **Default** → *default_stop_pct*.

    Returns
    -------
    dict with keys: ``stop_pct``, ``reason``.
    """
    if days_since_listing is not None and days_since_listing < new_coin_days:
        return {
            "stop_pct": new_coin_stop_pct,
            "reason": "Very new coin (%d days) → wider stop %.1f%%" % (
                days_since_listing, new_coin_stop_pct),
        }
    if change_7d is not None and abs(change_7d) > volatile_threshold_7d:
        return {
            "stop_pct": volatile_stop_pct,
            "reason": "Volatile (7d %.1f%%) → wide stop %.1f%%" % (change_7d, volatile_stop_pct),
        }
    return {
        "stop_pct": default_stop_pct,
        "reason": "Default stop %.1f%%" % default_stop_pct,
    }


def leverage_cap(
    proposed_leverage: float,
    max_leverage: float = 5.0,
    min_leverage: float = 1.0,
    days_since_listing: Optional[int] = None,
    new_coin_max_leverage: float = 2.0,
    new_coin_days: int = 30,
    atr_pct: Optional[float] = None,
    high_vol_atr_threshold: float = 8.0,
    high_vol_max_leverage: float = 3.0,
) -> dict:
    """Cap leverage based on market conditions (L4-07b).

    Ported from ``atomic_strategy_lib.decision.sizing.leverage_cap``.

    Reduces leverage for:

    * **New coins** (listed < *new_coin_days* ago) → capped at *new_coin_max_leverage*.
    * **High-volatility** instruments (ATR % > *high_vol_atr_threshold*) → capped at
      *high_vol_max_leverage*.

    Returns
    -------
    dict with keys: ``leverage``, ``proposed``, ``effective_max``, ``capped``, ``reasons``.
    """
    effective_max = max_leverage
    cap_reasons: list[str] = []

    if days_since_listing is not None and days_since_listing < new_coin_days:
        if effective_max > new_coin_max_leverage:
            effective_max = new_coin_max_leverage
            cap_reasons.append(
                "New coin (%d days) → max %sx" % (days_since_listing, new_coin_max_leverage)
            )

    if atr_pct is not None and atr_pct > high_vol_atr_threshold:
        if effective_max > high_vol_max_leverage:
            effective_max = high_vol_max_leverage
            cap_reasons.append(
                "High ATR (%.1f%%) → max %sx" % (atr_pct, high_vol_max_leverage)
            )

    capped = max(min_leverage, min(proposed_leverage, effective_max))

    return {
        "leverage": capped,
        "proposed": proposed_leverage,
        "effective_max": effective_max,
        "capped": capped != proposed_leverage,
        "reasons": cap_reasons if cap_reasons else ["No cap applied"],
    }


def compute_stop_price(
    entry_price: float,
    direction: str,
    stop_pct: Optional[float] = None,
    atr_val: Optional[float] = None,
    multiplier: float = 1.0,
) -> float:
    """Unified stop-price calculator (L4-07c).

    Ported from ``atomic_strategy_lib.decision.sizing.compute_stop_price``.

    If *stop_pct* is ``None`` and *atr_val* is provided, derives the stop
    percentage from ``atr_val * multiplier / entry_price * 100``.

    Parameters
    ----------
    entry_price:
        Trade entry price.
    direction:
        ``"LONG"`` or ``"SHORT"``, matched case-insensitively. It used to be compared with a bare
        ``== "LONG"``, so ``"long"`` fell through to the short branch and put the stop ABOVE the
        entry on a long trade -- a wrong answer that looks like a valid price, with no exception
        and no clue at the call site. ``stop_loss.atr_dynamic_stop`` computes the same thing and
        has always used ``.upper()``; the two now agree.
    stop_pct:
        Stop distance as a percentage (e.g. 3.0 = 3 %).
    atr_val:
        Absolute ATR value; only used when *stop_pct* is None.
    multiplier:
        ATR multiplier when deriving stop from ATR.

    Returns
    -------
    float — the absolute stop price, rounded to 8 decimal places.
    """
    if stop_pct is None:
        if entry_price > 0 and atr_val is not None:
            stop_pct = atr_val * multiplier / entry_price * 100.0
        else:
            stop_pct = 0.0
    distance = stop_pct / 100.0
    if direction.upper() == "LONG":
        return round(entry_price * (1.0 - distance), 8)
    return round(entry_price * (1.0 + distance), 8)
