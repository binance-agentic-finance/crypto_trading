"""Long-only single-position backtest engine.

Per `docs/plans/acceptance-plan.md` §E4 (BacktestRunner parity vs
`standard_bot.simulation.runner.SnapshotBacktestRunner`). This engine
mirrors the standard_bot cash / position / fee / slippage bookkeeping
verbatim, but takes a flat `(bars, entry_signal, exit_signal)` tuple
instead of the snapshot+signal_pipeline chain. That keeps the
acceptance contract testable without re-implementing the snapshot
machinery.

Differences from `runtime.backtest.engine.run_vector_backtest`:
- This engine is **stateful** (cash + position) and matches the
  long-only one-position-at-a-time semantic of standard_bot.
- Vector backtest is fully vectorised and assumes always-in-market
  while signal is True. Use it for fast indicator validation.

Use the long-only engine when E4 parity matters; use vector backtest
for fast smoke / signal sanity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ai_pro_trading_library.library.core.protocols import Bar


@dataclass(frozen=True)
class TradeRecord:
    timestamp: int
    instrument_id: str
    side: str  # "buy" | "sell"
    price: float
    quantity: float
    fee: float
    action: str  # "entry" | "exit" | "forced_exit"
    entry_price: float | None = None
    pnl: float | None = None


@dataclass(frozen=True)
class EquityPoint:
    timestamp: int
    equity: float
    cash: float


@dataclass(frozen=True)
class LongOnlyResult:
    """Long-only backtest result. Field shape mirrors the relevant subset
    of `standard_bot.core.contracts.BacktestResult` for E4 parity."""

    total_return: float
    final_equity: float
    initial_capital: float
    trade_count: int
    snapshot_count: int  # = bar count
    equity_curve: list[EquityPoint] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    max_drawdown: float = 0.0


def _compute_max_drawdown(equity_values: list[float]) -> float:
    """Mirror of `standard_bot.simulation.metrics_kernels.compute_equity_statistics`
    drawdown loop. Returns the largest peak-to-trough decline as a fraction
    of the running peak."""
    if not equity_values:
        return 0.0
    high_watermark = 0.0
    max_dd = 0.0
    for equity in equity_values:
        if equity > high_watermark:
            high_watermark = equity
        if high_watermark > 0.0:
            dd = (high_watermark - equity) / high_watermark
            if dd < 0.0:
                dd = 0.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def run_long_only_backtest(
    bars: Sequence[Bar],
    entry_signal: Sequence[bool],
    exit_signal: Sequence[bool] | None = None,
    *,
    initial_capital: float = 10_000.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    force_exit_at_end: bool = True,
    fee_model: object | None = None,
    slippage_model: object | None = None,
) -> LongOnlyResult:
    """Single-position long-only backtest with fee + slippage.

    Per-bar logic (mirrors `SnapshotBacktestRunner.run` inner loop):
    - if `entry_signal[i]` and not in position → BUY at next-bar close
      (or current bar close at the engine's discretion — this engine
      uses **current bar close** to match standard_bot's `latest_bar.close`).
    - if `exit_signal[i]` and in position → SELL at current bar close.
    - if neither: hold.
    - At end, if `force_exit_at_end` and still holding, mark a forced
      exit at the final bar's close.
    - Slippage is applied as `price * (1 ± slippage_bps/10000)` for buys
      and sells respectively.
    - Commission is `qty * price * commission_bps / 10000` per side.

    `exit_signal=None` means "exit when entry_signal is False" — common
    long-only strategies use this convention.

    Returns `LongOnlyResult` with total_return, equity_curve, and trade list.
    """
    n = len(bars)
    if len(entry_signal) != n:
        raise ValueError(f"entry_signal length {len(entry_signal)} != bars length {n}")
    if exit_signal is None:
        exit_signal = [not bool(s) for s in entry_signal]
    elif len(exit_signal) != n:
        raise ValueError(f"exit_signal length {len(exit_signal)} != bars length {n}")

    cash = float(initial_capital)
    position_qty = 0.0
    position_entry = 0.0
    trades: list[TradeRecord] = []
    equity_curve: list[EquityPoint] = []

    bps = 1.0 / 10_000.0

    def _adjust_price(side: str, price: float) -> float:
        if slippage_model is not None:
            return float(slippage_model.adjust(side=side, price=price))  # type: ignore[attr-defined]
        sign = 1.0 if side == "buy" else -1.0
        return price * (1.0 + sign * slippage_bps * bps)

    def _compute_fee(side: str, quantity: float, price: float) -> float:
        if fee_model is not None:
            return float(fee_model.fee(side=side, quantity=quantity, price=price))  # type: ignore[attr-defined]
        return quantity * price * commission_bps * bps

    for i, bar in enumerate(bars):
        timestamp = bar.timestamp
        close = bar.close

        # Entry: long-only single position
        if bool(entry_signal[i]) and position_qty == 0:
            execution_price = _adjust_price("buy", close)
            qty = cash / execution_price if execution_price > 0 else 0.0
            fee = _compute_fee("buy", qty, execution_price)
            if qty > 0:
                cash = cash - qty * execution_price - fee
                position_qty = qty
                position_entry = execution_price
                trades.append(
                    TradeRecord(
                        timestamp=timestamp,
                        instrument_id=bar.instrument_id,
                        side="buy",
                        price=execution_price,
                        quantity=qty,
                        fee=fee,
                        action="entry",
                    )
                )

        # Exit: close out long position
        elif bool(exit_signal[i]) and position_qty > 0:
            execution_price = _adjust_price("sell", close)
            fee = _compute_fee("sell", position_qty, execution_price)
            realized = position_qty * execution_price - fee
            pnl = (execution_price - position_entry) * position_qty - fee
            cash = cash + realized
            trades.append(
                TradeRecord(
                    timestamp=timestamp,
                    instrument_id=bar.instrument_id,
                    side="sell",
                    price=execution_price,
                    quantity=position_qty,
                    fee=fee,
                    action="exit",
                    entry_price=position_entry,
                    pnl=pnl,
                )
            )
            position_qty = 0.0
            position_entry = 0.0

        equity = cash + position_qty * close
        equity_curve.append(
            EquityPoint(timestamp=timestamp, equity=float(equity), cash=float(cash))
        )

    # Forced exit at end if still holding (mirrors SnapshotBacktestRunner)
    if force_exit_at_end and position_qty > 0 and bars:
        final_bar = bars[-1]
        exit_price = final_bar.close  # forced exit uses unadjusted close (mirrors source)
        fee = _compute_fee("sell", position_qty, exit_price)
        pnl = (exit_price - position_entry) * position_qty - fee
        cash = cash + position_qty * exit_price - fee
        trades.append(
            TradeRecord(
                timestamp=final_bar.timestamp,
                instrument_id=final_bar.instrument_id,
                side="sell",
                price=exit_price,
                quantity=position_qty,
                fee=fee,
                action="forced_exit",
                entry_price=position_entry,
                pnl=pnl,
            )
        )
        position_qty = 0.0
        if equity_curve:
            equity_curve[-1] = EquityPoint(
                timestamp=equity_curve[-1].timestamp,
                equity=float(cash),
                cash=float(cash),
            )

    final_equity = equity_curve[-1].equity if equity_curve else float(initial_capital)
    total_return = (
        (final_equity - initial_capital) / initial_capital if initial_capital else 0.0
    )

    max_dd = _compute_max_drawdown([p.equity for p in equity_curve])

    return LongOnlyResult(
        total_return=float(total_return),
        final_equity=float(final_equity),
        initial_capital=float(initial_capital),
        trade_count=len(trades),
        snapshot_count=n,
        equity_curve=equity_curve,
        trades=trades,
        max_drawdown=float(max_dd),
    )


__all__ = [
    "EquityPoint",
    "LongOnlyResult",
    "TradeRecord",
    "run_long_only_backtest",
]
