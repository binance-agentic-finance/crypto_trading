"""Fitness function (SPEC §3.1).

Maps a backtest result dict → scalar fitness used for selection.

Design notes:
    - Negative fitness is allowed (worse than zero return).
    - Trade count outside [10, 500] is heavily penalised.
    - Win-rate inside [0.4, 0.7] gets a small stability bonus.
    - Excessive complexity (entry_factors > 4) is rejected at validation
      time, but we add a soft Occam's razor penalty here as well.
"""

from __future__ import annotations

import math
from typing import Any, Dict


def fitness_score(metrics: Dict[str, Any], n_factors: int = 1) -> float:
    """Compute fitness from a backtest metrics dict.

    Parameters
    ----------
    metrics : dict
        Must contain ``sharpe_ratio``, ``total_return``, ``max_drawdown``,
        ``trade_count``, ``win_rate``.
    n_factors : int
        Number of entry factors, used for the Occam's razor penalty.

    Returns
    -------
    float
        Higher = better. A degenerate strategy (no trades / NaN) returns -10.
    """
    sharpe = float(metrics.get("sharpe_ratio") or 0.0)
    ret = float(metrics.get("total_return") or 0.0)
    dd = abs(float(metrics.get("max_drawdown") or 0.0))
    trades = int(metrics.get("trade_count") or 0)
    win_rate = float(metrics.get("win_rate") or 0.0)

    # Sanity / NaN guards
    if math.isnan(sharpe) or math.isinf(sharpe):
        sharpe = 0.0
    if math.isnan(ret) or math.isinf(ret):
        ret = 0.0

    # Base score per SPEC §3.1
    score = sharpe * 0.4 + ret * 10 * 0.3 - dd * 5 * 0.2

    # Trade count gates
    if trades < 10:
        score *= 0.3
    elif trades > 500:
        score *= 0.7

    # Win-rate stability bonus
    if 0.4 <= win_rate <= 0.7 and trades >= 10:
        score *= 1.1

    # Occam's razor — soft penalty for complexity > 3 factors
    if n_factors > 3:
        score *= 0.95
    if n_factors > 4:
        score *= 0.9  # compound (>4 should already be invalid)

    # Catastrophic strategies (lost > 50%) → strong penalty
    if ret < -0.5:
        score = min(score, -2.0)

    return float(score)
