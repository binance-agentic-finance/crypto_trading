"""shim — atomic.risk.graduated_tp.

Atomic provides three stateful helpers (graduated_tp_evaluate,
emergency_exit_check, compute_graduated_stop). cyqnt_trd ships a single
ladder builder (graduated_take_profit) plus dataclass TakeProfitStep.
We expose the atomic names by composing these primitives.
"""
from typing import Optional
from cyqnt_trd.blocks.exit import graduated_take_profit, TakeProfitStep  # noqa: F401


def compute_graduated_stop(
    entry_price: float,
    direction: str,
    current_stage: int,
    stage1_stop_to_breakeven: bool = True,
    stage2_stop_above_entry_pct: float = 3.0,
    stage3_trailing_stop_pct: float = 5.0,
    current_price: Optional[float] = None,
) -> dict:
    """atomic shim: compute the stop for a given graduated-TP stage.

    Stages roughly mean:
      stage 0: pre any TP — original stop applies (returned as None to caller)
      stage 1: first leg sold — move stop to breakeven (entry)
      stage 2: second leg sold — move stop above entry by stage2_stop_above_entry_pct
      stage 3: third leg sold — trailing stop at stage3_trailing_stop_pct from current price
    """
    direction = direction.upper()
    if current_stage <= 0:
        return {"stop_price": None, "stage": 0, "reason": "no graduated stop yet"}
    if current_stage == 1:
        if stage1_stop_to_breakeven:
            return {"stop_price": float(entry_price), "stage": 1, "reason": "breakeven"}
        return {"stop_price": None, "stage": 1, "reason": "no breakeven move"}
    if current_stage == 2:
        offset = stage2_stop_above_entry_pct / 100.0
        if direction == "LONG":
            stop = float(entry_price) * (1 + offset)
        else:
            stop = float(entry_price) * (1 - offset)
        return {"stop_price": round(stop, 8), "stage": 2,
                "reason": f"locked {stage2_stop_above_entry_pct}% above entry"}
    if current_stage >= 3 and current_price is not None:
        offset = stage3_trailing_stop_pct / 100.0
        if direction == "LONG":
            stop = float(current_price) * (1 - offset)
        else:
            stop = float(current_price) * (1 + offset)
        return {"stop_price": round(stop, 8), "stage": current_stage,
                "reason": f"trailing {stage3_trailing_stop_pct}% from current"}
    return {"stop_price": None, "stage": current_stage, "reason": "no current_price"}


def emergency_exit_check(
    pnl_leveraged_pct: float,
    peak_profit_pct: float,
    emergency_drop_threshold_pct: float = 10.0,
) -> dict:
    """atomic shim: detect when pnl drops more than threshold from its peak."""
    drop = float(peak_profit_pct) - float(pnl_leveraged_pct)
    triggered = drop >= float(emergency_drop_threshold_pct) and peak_profit_pct > 0
    return {
        "trigger_emergency_exit": triggered,
        "drop_from_peak_pct": round(drop, 2),
        "current_pnl_pct": float(pnl_leveraged_pct),
        "peak_pnl_pct": float(peak_profit_pct),
        "threshold": float(emergency_drop_threshold_pct),
        "reason": (
            "Profit dropped %.1fpp from peak %.1f%% → exit"
            % (drop, peak_profit_pct)
            if triggered
            else "Drop %.1fpp within tolerance" % drop
        ),
    }


def graduated_tp_evaluate(
    pnl_leveraged_pct: float,
    current_stage: int,
    peak_profit_pct: float = 0.0,
    stage1_profit_pct: float = 5.0,
    stage1_sell_pct: float = 5.0,
    stage2_profit_pct: float = 10.0,
    stage2_sell_pct: float = 10.0,
    stage3_profit_pct: float = 20.0,
    stage3_sell_pct: float = 15.0,
    emergency_profit_drop_threshold_pct: float = 10.0,
    emergency_sell_pct: float = 50.0,
) -> dict:
    """atomic shim: evaluate which TP stage to act on given current PnL.

    Returns a dict describing the action (close/hold/emergency_exit), the
    fraction to close, and the new stage to transition to.
    """
    new_peak = max(float(peak_profit_pct), float(pnl_leveraged_pct))

    # Emergency drop check first
    emerg = emergency_exit_check(
        pnl_leveraged_pct, new_peak, emergency_profit_drop_threshold_pct
    )
    if emerg["trigger_emergency_exit"]:
        return {
            "action": "emergency_exit",
            "close_fraction": float(emergency_sell_pct) / 100.0,
            "new_stage": current_stage,
            "new_peak_pct": new_peak,
            "reason": emerg["reason"],
        }

    # Stage progression
    if current_stage == 0 and pnl_leveraged_pct >= stage1_profit_pct:
        return {
            "action": "partial_close",
            "close_fraction": float(stage1_sell_pct) / 100.0,
            "new_stage": 1,
            "new_peak_pct": new_peak,
            "reason": f"Hit stage1 +{stage1_profit_pct}% — close {stage1_sell_pct}%",
        }
    if current_stage == 1 and pnl_leveraged_pct >= stage2_profit_pct:
        return {
            "action": "partial_close",
            "close_fraction": float(stage2_sell_pct) / 100.0,
            "new_stage": 2,
            "new_peak_pct": new_peak,
            "reason": f"Hit stage2 +{stage2_profit_pct}% — close {stage2_sell_pct}%",
        }
    if current_stage == 2 and pnl_leveraged_pct >= stage3_profit_pct:
        return {
            "action": "partial_close",
            "close_fraction": float(stage3_sell_pct) / 100.0,
            "new_stage": 3,
            "new_peak_pct": new_peak,
            "reason": f"Hit stage3 +{stage3_profit_pct}% — close {stage3_sell_pct}%",
        }

    return {
        "action": "hold",
        "close_fraction": 0.0,
        "new_stage": current_stage,
        "new_peak_pct": new_peak,
        "reason": (
            f"Stage {current_stage}, pnl {pnl_leveraged_pct:.1f}%, "
            f"peak {new_peak:.1f}% — no action"
        ),
    }
