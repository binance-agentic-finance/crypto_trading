"""Backtest performance metrics.

Pure-Python port of `cyqnt_trd.standard_bot.simulation.metrics_kernels.
compute_equity_statistics` plus trade-level metrics that the standard_bot
runner did not expose directly. Use this when you need richer stats than
`LongOnlyResult.max_drawdown` / `sharpe` carry.

Functions:
- `compute_equity_metrics(equity, periods_per_year)` →
  `{max_drawdown, mean_return, return_volatility, sharpe, sortino,
    calmar, total_return}` — equity-curve-driven.
- `compute_trade_metrics(trades)` →
  `{trade_count, win_count, loss_count, win_rate, profit_factor,
    expectancy, avg_win, avg_loss, total_pnl}` — trade-list-driven.
- `compute_full_metrics(equity, trades, periods_per_year)` — union of both.

Sharpe uses the same Welford incremental algorithm as
`metrics_kernels.compute_equity_statistics` so the numbers match at 1e-9
on the same input.
"""

from __future__ import annotations

import math
from typing import Sequence


def compute_equity_metrics(
    equity: Sequence[float],
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Mirror of `compute_equity_statistics` (drawdowns + sharpe) plus
    sortino / calmar / total_return."""
    n = len(equity)
    if n == 0:
        return {
            "max_drawdown": 0.0,
            "mean_return": 0.0,
            "return_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "total_return": 0.0,
        }

    high_watermark = 0.0
    max_dd = 0.0
    mean_return = 0.0
    return_m2 = 0.0
    return_count = 0
    downside_squared_sum = 0.0
    downside_count = 0

    for i, eq in enumerate(equity):
        if eq > high_watermark:
            high_watermark = eq
        if high_watermark > 0.0:
            dd = (high_watermark - eq) / high_watermark
            if dd < 0.0:
                dd = 0.0
            if dd > max_dd:
                max_dd = dd

        if i == 0:
            continue
        prev = equity[i - 1]
        if prev <= 0.0:
            continue
        bar_return = eq / prev - 1.0
        return_count += 1
        delta = bar_return - mean_return
        mean_return += delta / return_count
        return_m2 += delta * (bar_return - mean_return)
        if bar_return < 0.0:
            downside_squared_sum += bar_return * bar_return
            downside_count += 1

    return_volatility = 0.0
    if return_count > 1:
        return_volatility = math.sqrt(return_m2 / (return_count - 1))

    sharpe = 0.0
    if periods_per_year > 0.0 and return_volatility > 1e-12:
        sharpe = mean_return / return_volatility * math.sqrt(periods_per_year)

    sortino = 0.0
    if downside_count > 0 and periods_per_year > 0.0:
        downside_vol = math.sqrt(downside_squared_sum / downside_count)
        if downside_vol > 1e-12:
            sortino = mean_return / downside_vol * math.sqrt(periods_per_year)

    initial = equity[0] if equity[0] > 0 else 1.0
    final = equity[-1]
    total_return = (final - initial) / initial

    annualized_return = 0.0
    if periods_per_year > 0.0 and return_count > 0:
        annualized_return = mean_return * periods_per_year
    calmar = 0.0
    if max_dd > 1e-12:
        calmar = annualized_return / max_dd

    return {
        "max_drawdown": max_dd,
        "mean_return": mean_return,
        "return_volatility": return_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "total_return": total_return,
    }


def compute_trade_metrics(trades: Sequence[object]) -> dict[str, float]:
    """Trade-list-driven metrics. Trades are anything with a `pnl` attribute
    (e.g. `TradeRecord`) — entries with `pnl is None` are skipped (only
    closed trades count toward win/loss/profit-factor stats)."""
    closed_pnls = [
        float(getattr(t, "pnl"))
        for t in trades
        if getattr(t, "pnl", None) is not None
    ]
    trade_count = len(closed_pnls)
    if trade_count == 0:
        return {
            "trade_count": 0.0,
            "win_count": 0.0,
            "loss_count": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_pnl": 0.0,
        }
    wins = [p for p in closed_pnls if p > 0]
    losses = [p for p in closed_pnls if p < 0]
    sum_wins = sum(wins)
    sum_losses = sum(losses)  # negative number
    abs_loss_sum = -sum_losses
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / trade_count
    profit_factor = (sum_wins / abs_loss_sum) if abs_loss_sum > 1e-12 else 0.0
    avg_win = (sum_wins / win_count) if win_count else 0.0
    avg_loss = (sum_losses / loss_count) if loss_count else 0.0
    total_pnl = sum_wins + sum_losses
    expectancy = total_pnl / trade_count
    return {
        "trade_count": float(trade_count),
        "win_count": float(win_count),
        "loss_count": float(loss_count),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_pnl": total_pnl,
    }


def compute_full_metrics(
    equity: Sequence[float],
    trades: Sequence[object],
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Union of equity + trade metrics in one dict."""
    out = compute_equity_metrics(equity, periods_per_year)
    out.update(compute_trade_metrics(trades))
    return out


__all__ = [
    "compute_equity_metrics",
    "compute_full_metrics",
    "compute_trade_metrics",
]
