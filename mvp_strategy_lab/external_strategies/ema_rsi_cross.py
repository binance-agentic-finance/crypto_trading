from __future__ import annotations

import numpy as np

from cyqnt_trd.standard_bot.signal.numba_kernels import (
    TARGET_KEEP,
    TARGET_LONG,
    TARGET_SHORT,
    njit,
)
from cyqnt_trd.standard_bot.simulation import NumbaBacktestRunner


@njit(cache=True)
def ema_rsi_cross_target_updates(
    closes: np.ndarray,
    fast_window: int,
    slow_window: int,
    rsi_period: int,
    rsi_mid: float,
    session_start_h: int,
    session_end_h: int,
    timestamps: np.ndarray,
):
    """External EMA/RSI cross target kernel.

    Generates long targets when fast EMA crosses above slow EMA with RSI above
    ``rsi_mid``. Generates short targets when fast EMA crosses below slow EMA
    with RSI below ``rsi_mid``. Optional UTC session filter uses millisecond
    timestamps; set ``session_start_h`` and ``session_end_h`` to the same value
    to disable the session filter.
    """
    n = closes.shape[0]
    target_updates = np.full(n, TARGET_KEEP, dtype=np.int8)
    strengths = np.zeros(n, dtype=np.float64)

    if n == 0:
        return target_updates, strengths
    if fast_window < 1 or slow_window <= fast_window or rsi_period < 2:
        return target_updates, strengths

    alpha_fast = 2.0 / (fast_window + 1.0)
    alpha_slow = 2.0 / (slow_window + 1.0)
    fast_ema = closes[0]
    slow_ema = closes[0]
    prev_fast_ema = fast_ema
    prev_slow_ema = slow_ema

    warmup = slow_window
    if rsi_period > warmup:
        warmup = rsi_period

    previous_target = TARGET_KEEP

    for i in range(1, n):
        prev_fast_ema = fast_ema
        prev_slow_ema = slow_ema
        fast_ema = closes[i] * alpha_fast + fast_ema * (1.0 - alpha_fast)
        slow_ema = closes[i] * alpha_slow + slow_ema * (1.0 - alpha_slow)

        if i < warmup:
            continue

        if session_start_h != session_end_h:
            hour = (timestamps[i] // 3_600_000) % 24
            if session_start_h < session_end_h:
                in_session = session_start_h <= hour < session_end_h
            else:
                in_session = hour >= session_start_h or hour < session_end_h
            if not in_session:
                continue

        gains = 0.0
        losses = 0.0
        for s in range(i - rsi_period + 1, i + 1):
            delta = closes[s] - closes[s - 1]
            if delta > 0.0:
                gains += delta
            else:
                losses += -delta

        avg_gain = gains / rsi_period
        avg_loss = losses / rsi_period
        rsi = 100.0 if avg_loss == 0.0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

        crossed_up = prev_fast_ema <= prev_slow_ema and fast_ema > slow_ema
        crossed_down = prev_fast_ema >= prev_slow_ema and fast_ema < slow_ema

        if crossed_up and rsi >= rsi_mid:
            if previous_target != TARGET_LONG:
                target_updates[i] = TARGET_LONG
                strengths[i] = abs(fast_ema - slow_ema) / max(abs(slow_ema), 1e-9) + abs(rsi - rsi_mid) / 100.0
                previous_target = TARGET_LONG
        elif crossed_down and rsi <= rsi_mid:
            if previous_target != TARGET_SHORT:
                target_updates[i] = TARGET_SHORT
                strengths[i] = abs(fast_ema - slow_ema) / max(abs(slow_ema), 1e-9) + abs(rsi_mid - rsi) / 100.0
                previous_target = TARGET_SHORT

    return target_updates, strengths


NumbaBacktestRunner.register_kernel(
    strategy_id="ema_rsi_cross",
    kernel_fn=ema_rsi_cross_target_updates,
    arg_map=[
        {"source": "series", "field": "closes"},
        {"source": "config", "field": "fast_window", "type": "int", "default": 2},
        {"source": "config", "field": "slow_window", "type": "int", "default": 5},
        {"source": "config", "field": "rsi_period", "type": "int", "default": 3},
        {"source": "config", "field": "rsi_mid", "type": "float", "default": 50.0},
        {"source": "config", "field": "session_start_h", "type": "int", "default": 0},
        {"source": "config", "field": "session_end_h", "type": "int", "default": 0},
        {"source": "series", "field": "timestamps"},
    ],
)
