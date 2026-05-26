"""Numba-friendly execution kernels for bar-based position simulation.

Ported from cyqnt_trd.standard_bot.simulation.execution_kernels.

The kernel simulates a target-position execution model:
- Signal at bar-close t sets a target position (long=+1, short=-1, flat=0).
- Execution fills at bar t+1's open price.
- Taker fees, slippage (fixed + impact), funding, and volume caps are applied.
- Quantity step / min quantity enforce exchange lot-size rules.

Returns 8 aligned arrays: equity, cash, position, trade_action,
trade_quantity, trade_price, trade_fees, funding_fees.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


TARGET_LONG = 1
TARGET_SHORT = -1
TARGET_FLAT = 0
TARGET_KEEP = 2


@njit(cache=True)
def simulate_target_positions_next_open(
    opens: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    quote_volumes: np.ndarray,
    target_updates: np.ndarray,
    target_keep: int,
    initial_capital: float,
    taker_fee_bps: float,
    slippage_bps: float,
    impact_slippage_bps: float,
    funding_rate_per_bar: float,
    max_bar_volume_fraction: float,
    contract_multiplier: float,
    quantity_step: float,
    min_quantity: float,
    fixed_fee_per_contract: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bar_count = closes.shape[0]
    equity_curve = np.zeros(bar_count, dtype=np.float64)
    cash_curve = np.zeros(bar_count, dtype=np.float64)
    position_curve = np.zeros(bar_count, dtype=np.float64)
    trade_action = np.zeros(bar_count, dtype=np.int8)
    trade_quantity = np.zeros(bar_count, dtype=np.float64)
    trade_price = np.zeros(bar_count, dtype=np.float64)
    trade_fees = np.zeros(bar_count, dtype=np.float64)
    funding_fees = np.zeros(bar_count, dtype=np.float64)

    cash = initial_capital
    position_qty = 0.0
    target_position = 0
    target_quantity = 0.0
    fee_rate = taker_fee_bps / 10_000.0
    multiplier = contract_multiplier
    if multiplier <= 0.0:
        multiplier = 1.0

    for index in range(bar_count):
        if index > 0 and target_updates[index - 1] != target_keep:
            target_position = int(target_updates[index - 1])
            equity_reference = cash + position_qty * opens[index] * multiplier
            if equity_reference < 0.0:
                equity_reference = 0.0
            if target_position == 1:
                target_quantity = equity_reference / (opens[index] * multiplier)
            elif target_position == -1:
                target_quantity = -equity_reference / (opens[index] * multiplier)
            else:
                target_quantity = 0.0
            if quantity_step > 0.0:
                target_sign = 1.0
                if target_quantity < 0.0:
                    target_sign = -1.0
                target_abs = np.floor(abs(target_quantity) / quantity_step) * quantity_step
                if target_abs < max(min_quantity, quantity_step):
                    target_quantity = 0.0
                else:
                    target_quantity = target_sign * target_abs

        if index > 0 and max_bar_volume_fraction > 0.0 and opens[index] > 0.0:
            base_cap_qty = max(volumes[index], 0.0) * max_bar_volume_fraction
            quote_cap_qty = base_cap_qty
            if quote_volumes[index] > 0.0:
                quote_cap_qty = (quote_volumes[index] * max_bar_volume_fraction) / (opens[index] * multiplier)
                if quote_cap_qty < base_cap_qty:
                    base_cap_qty = quote_cap_qty
            available_qty = max(base_cap_qty, 0.0)

            delta_qty = target_quantity - position_qty
            if delta_qty > available_qty:
                delta_qty = available_qty
            elif delta_qty < -available_qty:
                delta_qty = -available_qty

            if quantity_step > 0.0:
                delta_sign = 1.0
                if delta_qty < 0.0:
                    delta_sign = -1.0
                delta_abs = np.floor(abs(delta_qty) / quantity_step) * quantity_step
                if delta_abs < max(min_quantity, quantity_step):
                    delta_qty = 0.0
                else:
                    delta_qty = delta_sign * delta_abs

            if abs(delta_qty) > 1e-12:
                participation = abs(delta_qty) / max(volumes[index], 1e-12)
                slip = (slippage_bps + impact_slippage_bps * participation) / 10_000.0
                fill_price = opens[index]
                if delta_qty > 0.0:
                    fill_price = opens[index] * (1.0 + slip)
                else:
                    fill_price = opens[index] * (1.0 - slip)

                fee = abs(delta_qty) * fill_price * multiplier * fee_rate
                if fixed_fee_per_contract > 0.0:
                    fee += abs(delta_qty) * fixed_fee_per_contract
                cash -= delta_qty * fill_price * multiplier + fee
                position_qty += delta_qty
                if abs(position_qty) < 1e-12:
                    position_qty = 0.0
                trade_action[index] = 1 if delta_qty > 0.0 else -1
                trade_quantity[index] = abs(delta_qty)
                trade_price[index] = fill_price
                trade_fees[index] = fee

        if funding_rate_per_bar != 0.0 and position_qty != 0.0:
            funding_fee = abs(position_qty) * closes[index] * multiplier * funding_rate_per_bar
            cash -= funding_fee
            funding_fees[index] = funding_fee

        equity_curve[index] = cash + position_qty * closes[index] * multiplier
        cash_curve[index] = cash
        position_curve[index] = position_qty

    return (
        equity_curve,
        cash_curve,
        position_curve,
        trade_action,
        trade_quantity,
        trade_price,
        trade_fees,
        funding_fees,
    )
