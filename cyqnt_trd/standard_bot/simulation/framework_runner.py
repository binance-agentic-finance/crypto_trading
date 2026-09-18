"""Backtest a single-instrument ``make_signals`` strategy on the vectorized
``cyqnt_trd.eval`` framework, emitting the standard :class:`BacktestResult`.

This is the convergence runner: instead of the event-driven ``SnapshotBacktestRunner``
loop, it reconstructs the per-symbol OHLCV frame, turns ``make_signals(df) -> (long,
short)`` into target weights, and runs the framework's ``portfolio.simulate`` (net-change
rebalancing, ``entry_lag`` fill, turnover cost, funding). The result carries the same
contract fields the entrypoints read (``total_return``, ``equity_curve``, ``metrics``,
``extras['trades']``), so a caller can swap engines without changing how it reads output.

Execution semantics differ from the event engine (vectorized net-change vs single-position
event loop), so **the numbers differ** — this is the intended, framework-unified口径.
"""
from __future__ import annotations

import math
import uuid
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..core import BacktestResult, EquityPoint

__all__ = ["FrameworkBacktestRunner"]


def _ts_ms(ts) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def _trades_from_weights(fills: pd.Series, instrument_id: str) -> list[dict]:
    """Discrete position-change events from the continuous held-weight series."""
    prev, out = 0.0, []
    for ts, w in fills.items():
        w = 0.0 if not np.isfinite(w) else float(w)
        if w != prev:
            out.append({"timestamp": _ts_ms(ts), "instrument_id": instrument_id,
                        "side": "buy" if w > prev else "sell",
                        "weight_from": prev, "weight_to": w,
                        "action": "entry" if abs(w) > abs(prev) else "exit"})
            prev = w
    return out


class FrameworkBacktestRunner:
    """Run a ``make_signals`` strategy through ``cyqnt_trd.eval`` and return a
    :class:`~cyqnt_trd.standard_bot.core.BacktestResult`."""

    def run(self, make_signals: Callable[[pd.DataFrame], tuple], df: pd.DataFrame, *,
            instrument_id: str = "ASSET", initial_capital: float = 10000.0,
            cost_bps: float = 6.5, entry_lag: int = 2, min_history: int = 20,
            funding: Optional[pd.Series] = None, request_id: Optional[str] = None,
            extras: Optional[dict] = None) -> BacktestResult:
        from cyqnt_trd.eval.adapters import backtest_signals  # lazy: keep standard_bot import light

        book = backtest_signals(make_signals, df, symbol=instrument_id, entry_lag=entry_lag,
                                cost_bps=cost_bps, funding=funding, min_history=min_history)
        equity = book.equity.astype(float)
        curve = [EquityPoint(timestamp=_ts_ms(ts), equity=float(v) * initial_capital)
                 for ts, v in equity.items()]
        final_frac = float(equity.iloc[-1]) if len(equity) else 1.0
        total_return = final_frac - 1.0
        held = book.fills[instrument_id] if instrument_id in book.fills else book.fills.iloc[:, 0]
        trades = _trades_from_weights(held, instrument_id)
        m = dict(book.metrics)
        metrics = {
            "snapshot_count": float(len(df)),
            "trade_count": float(len(trades)),
            "final_equity": final_frac * initial_capital,
            "total_return": total_return,
            "sharpe_ratio": float(m.get("sharpe", float("nan"))),
            "max_drawdown": float(m.get("max_drawdown", float("nan"))),
            **{k: float(v) for k, v in m.items()
               if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))},
        }
        return BacktestResult(
            request_id=request_id or f"framework-{uuid.uuid4().hex[:8]}",
            total_return=total_return,
            equity_curve=curve,
            metrics=metrics,
            signal_batches=[],                       # wiring slice fills these from the plugin run
            extras={"run_id": request_id, "trades": trades, "engine": "framework",
                    **(extras or {})},
        )
