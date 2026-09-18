"""
Pure-NumPy performance metric helpers for historical simulations.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_equity_statistics(
    equity_values: np.ndarray,
    periods_per_year: float,
) -> Tuple[np.ndarray, float, float, float, float]:
    drawdowns = np.zeros(equity_values.shape[0], dtype=np.float64)
    if equity_values.shape[0] == 0:
        return drawdowns, 0.0, 0.0, 0.0, 0.0

    high_watermark = 0.0
    max_drawdown = 0.0
    mean_return = 0.0
    return_m2 = 0.0
    return_count = 0

    for index in range(equity_values.shape[0]):
        equity = equity_values[index]
        if equity > high_watermark:
            high_watermark = equity
        if high_watermark > 0.0:
            drawdown = (high_watermark - equity) / high_watermark
            if drawdown < 0.0:
                drawdown = 0.0
            drawdowns[index] = drawdown
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        if index == 0:
            continue
        previous_equity = equity_values[index - 1]
        if previous_equity <= 0.0:
            continue
        bar_return = equity / previous_equity - 1.0
        return_count += 1
        delta = bar_return - mean_return
        mean_return += delta / return_count
        return_m2 += delta * (bar_return - mean_return)

    return_volatility = 0.0
    if return_count > 1:
        return_volatility = np.sqrt(return_m2 / (return_count - 1))

    sharpe_ratio = 0.0
    if periods_per_year > 0.0 and return_volatility > 1e-12:
        sharpe_ratio = mean_return / return_volatility * np.sqrt(periods_per_year)

    return drawdowns, max_drawdown, mean_return, return_volatility, sharpe_ratio
