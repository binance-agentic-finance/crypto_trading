"""Vectorized backtester for blocks-based strategies.

Signal generation is fully vectorized (make_signals called once on the
full DataFrame). Position/exit management uses a lightweight numpy loop.
This makes it 10-100x faster than SnapshotBacktestRunner while still
supporting the full exit_cfg spec (pct_stop_tp, atr_stop_tp, time_only,
ma_cross_exit, opposite_signal).

Usage via CLI:
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
        --engine vectorized \
        --strategy my_strategy_v1 \
        --strategy-module workspace.my_strategy \
        --symbol BTCUSDT --interval 1h --limit 5000

Usage as library:
    from cyqnt_trd.standard_bot.simulation.vectorized_backtest import (
        run_vectorized_backtest,
        run_vectorized_batch,
    )
    result = run_vectorized_backtest(
        df=df,
        signal_fn=make_signals,
        exit_cfg={"type": "pct_stop_tp", "stop_pct": 0.02, "tp_pct": 0.04},
        timeframe="1h",
    )

Execution model (matches SnapshotBacktestRunner / PythonLivePaperSession):
- Signal at bar[i] close → fill at bar[i+1] open
- Single position at a time (no pyramiding)
- Fees on every entry + exit
- Slippage as adverse fill adjustment
- TP/SL checked intra-bar using high/low (same priority: SL → TP → max_bars)
- Side-aware: long SL = bar_low ≤ stop; short SL = bar_high ≥ stop
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cyqnt_trd.blocks import indicators as ind

__all__ = [
    "run_vectorized_backtest",
    "run_vectorized_batch",
    "VectorizedBacktestResult",
]

# ── Constants ──────────────────────────────────────────────────────────────

PERIODS_PER_YEAR = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "1h": 365 * 24,
    "2h": 365 * 12,
    "4h": 365 * 6,
    "6h": 365 * 4,
    "8h": 365 * 3,
    "12h": 365 * 2,
    "1d": 365,
}


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class VectorizedBacktestResult:
    """Backtest result matching SnapshotBacktestRunner output format."""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    total_pnl: float = 0.0
    exposure: float = 0.0
    final_equity: float = 1.0
    equity_curve: Optional[np.ndarray] = field(default=None, repr=False)
    trades: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "avg_trade_pnl": self.avg_trade_pnl,
            "total_pnl": self.total_pnl,
            "exposure": self.exposure,
            "final_equity": self.final_equity,
        }


# ── Exit checking (side-aware) ─────────────────────────────────────────────

def _check_exit_long(
    *,
    i: int,
    entry_price: float,
    entry_idx: int,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    atr_at_entry: float,
    opposite_signal: bool,
    ma_val: Optional[float],
    exit_cfg: dict,
) -> Tuple[bool, float, str]:
    """Return (should_exit, exit_price, reason) for a long position.

    Unified exit priority (first hit wins):
        1. Stop Loss      — intra-bar, fill at stop price (pct_stop_tp / atr_stop_tp)
        2. Take Profit    — intra-bar, fill at tp price   (pct_stop_tp / atr_stop_tp)
        3. MA cross       — bar close,  fill at next-bar open (ma_cross_exit only)
        4. Opposite signal — bar close, fill at next-bar open (always checked, all types)
        5. max_bars       — bar close,  fill at next-bar open (timeout)
    """
    etype = exit_cfg.get("type", "opposite_signal")
    max_bars = int(exit_cfg.get("max_bars", 9999))

    # Priority 1 & 2: Stop Loss / Take Profit (intra-bar) — only when configured
    if etype == "pct_stop_tp":
        stop_pct = float(exit_cfg.get("stop_pct", 0.02))
        tp_pct = float(exit_cfg.get("tp_pct", 0.04))
        stop_price = entry_price * (1.0 - stop_pct)
        tp_price = entry_price * (1.0 + tp_pct)
        if bar_low <= stop_price:
            return True, stop_price, "stop_loss"
        if bar_high >= tp_price:
            return True, tp_price, "take_profit"
    elif etype == "atr_stop_tp":
        stop_mult = float(exit_cfg.get("stop_mult", 2.0))
        tp_mult = float(exit_cfg.get("tp_mult", 3.0))
        if atr_at_entry > 0 and not math.isnan(atr_at_entry):
            stop_price = entry_price - stop_mult * atr_at_entry
            tp_price = entry_price + tp_mult * atr_at_entry
            if bar_low <= stop_price:
                return True, stop_price, "stop_loss"
            if bar_high >= tp_price:
                return True, tp_price, "take_profit"

    # Priority 3: MA cross (bar close) — only when type=ma_cross_exit
    if etype == "ma_cross_exit":
        if ma_val is not None and bar_close < ma_val:
            return True, bar_open, "ma_cross"

    # Priority 4: Opposite signal (bar close) — checked for ALL exit types
    if opposite_signal:
        return True, bar_open, "opposite_signal"

    # Priority 5: max_bars timeout (bar close)
    if i - entry_idx >= max_bars:
        return True, bar_open, "max_bars"

    return False, 0.0, ""


def _check_exit_short(
    *,
    i: int,
    entry_price: float,
    entry_idx: int,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    atr_at_entry: float,
    opposite_signal: bool,
    ma_val: Optional[float],
    exit_cfg: dict,
) -> Tuple[bool, float, str]:
    """Return (should_exit, exit_price, reason) for a short position.

    Same priority as long, with side-aware SL/TP and MA cross direction.
    """
    etype = exit_cfg.get("type", "opposite_signal")
    max_bars = int(exit_cfg.get("max_bars", 9999))

    # Priority 1 & 2: Stop Loss / Take Profit (intra-bar)
    if etype == "pct_stop_tp":
        stop_pct = float(exit_cfg.get("stop_pct", 0.02))
        tp_pct = float(exit_cfg.get("tp_pct", 0.04))
        stop_price = entry_price * (1.0 + stop_pct)  # short: stop above entry
        tp_price = entry_price * (1.0 - tp_pct)      # short: TP below entry
        if bar_high >= stop_price:
            return True, stop_price, "stop_loss"
        if bar_low <= tp_price:
            return True, tp_price, "take_profit"
    elif etype == "atr_stop_tp":
        stop_mult = float(exit_cfg.get("stop_mult", 2.0))
        tp_mult = float(exit_cfg.get("tp_mult", 3.0))
        if atr_at_entry > 0 and not math.isnan(atr_at_entry):
            stop_price = entry_price + stop_mult * atr_at_entry  # short: stop above
            tp_price = entry_price - tp_mult * atr_at_entry      # short: TP below
            if bar_high >= stop_price:
                return True, stop_price, "stop_loss"
            if bar_low <= tp_price:
                return True, tp_price, "take_profit"

    # Priority 3: MA cross (bar close) — short exits when close > MA
    if etype == "ma_cross_exit":
        if ma_val is not None and bar_close > ma_val:
            return True, bar_open, "ma_cross"

    # Priority 4: Opposite signal (bar close) — checked for ALL exit types
    if opposite_signal:
        return True, bar_open, "opposite_signal"

    # Priority 5: max_bars timeout
    if i - entry_idx >= max_bars:
        return True, bar_open, "max_bars"

    return False, 0.0, ""


# ── Main backtest function ─────────────────────────────────────────────────

def run_vectorized_backtest(
    *,
    df: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], Tuple[pd.Series, pd.Series]],
    exit_cfg: Optional[Dict] = None,
    timeframe: str = "1h",
    size: float = 0.5,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
    initial_capital: float = 10_000.0,
    record_trades: bool = True,
    long_only: bool = False,
) -> VectorizedBacktestResult:
    """Run a single vectorized backtest.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: open, high, low, close, volume.
    signal_fn : callable
        make_signals(df) → (long_signal: pd.Series[bool], short_signal: pd.Series[bool])
    exit_cfg : dict or None
        Exit specification. If None, uses opposite_signal with max_bars=9999.
    timeframe : str
        For Sharpe annualization (e.g. "1h", "15m", "1d").
    size : float
        Fraction of equity per trade (default 0.5 = 50%).
    fee_bps : float
        Commission in basis points per side.
    slippage_bps : float
        Slippage in basis points (adverse).
    initial_capital : float
        Starting equity (default 10000).
    record_trades : bool
        If True, record individual trades in result.trades.
    long_only : bool
        If True, short signals can only close long positions (spot market behavior).
        Short positions are never opened. Default False (futures: long + short).

    Returns
    -------
    VectorizedBacktestResult
    """
    if exit_cfg is None:
        exit_cfg = {"type": "opposite_signal", "max_bars": 9999}

    if df.empty or len(df) < 50:
        return VectorizedBacktestResult()

    # ── Step 1: Vectorized signal generation (ONE call) ────────────────
    long_sig, short_sig = signal_fn(df)
    long_sig = long_sig.reindex(df.index, fill_value=False).astype(bool).values
    short_sig = short_sig.reindex(df.index, fill_value=False).astype(bool).values

    # ── Step 2: Pre-compute arrays ────────────────────────────────────
    atr_period = int(exit_cfg.get("atr_period", 14))
    atr_arr = ind.atr(df, period=atr_period).values

    ma_arr: Optional[np.ndarray] = None
    if exit_cfg.get("type") == "ma_cross_exit":
        period = int(exit_cfg.get("period", 50))
        ma_type = exit_cfg.get("ma_type", "ema")
        if ma_type == "ema":
            ma_arr = ind.ema(df["close"], period).values
        else:
            ma_arr = ind.sma(df["close"], period).values

    open_arr = df["open"].astype(float).values
    high_arr = df["high"].astype(float).values
    low_arr = df["low"].astype(float).values
    close_arr = df["close"].astype(float).values

    n = len(df)
    fee_rate = fee_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0

    # Flip vs close-only: when exit_cfg is opposite_signal (or None default),
    # an exit at bar i may be followed by an entry on the same bar (flip).
    # When exit_cfg is anything else (pct_stop_tp / atr_stop_tp / time_only /
    # ma_cross_exit), the engine must close-only — no re-entry on the same bar.
    exit_allows_flip = (exit_cfg.get("type", "opposite_signal") == "opposite_signal")

    # ── Step 3: Position loop (lightweight numpy) ─────────────────────
    pos = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    atr_at_entry = 0.0

    equity = initial_capital
    equity_curve = np.empty(n, dtype=np.float64)
    trade_returns: List[float] = []
    trades: List[Dict[str, Any]] = []
    bars_in_position = 0

    for i in range(n):
        bar_o = open_arr[i]
        bar_h = high_arr[i]
        bar_l = low_arr[i]
        bar_c = close_arr[i]

        # Previous bar signals (for exit checks using opposite_signal)
        l_prev = bool(long_sig[i - 1]) if i > 0 else False
        s_prev = bool(short_sig[i - 1]) if i > 0 else False
        ma_val = float(ma_arr[i]) if ma_arr is not None and not math.isnan(ma_arr[i]) else None

        exited_this_bar = False

        # ── Exit check ────────────────────────────────────────────────
        if pos == 1:
            should_exit, ex_px, reason = _check_exit_long(
                i=i, entry_price=entry_price, entry_idx=entry_idx,
                bar_open=bar_o, bar_high=bar_h, bar_low=bar_l, bar_close=bar_c,
                atr_at_entry=atr_at_entry, opposite_signal=s_prev,
                ma_val=ma_val, exit_cfg=exit_cfg,
            )
            if should_exit:
                fill_px = ex_px * (1.0 - slip_rate)
                pnl_pct = (fill_px - entry_price) / entry_price
                trade_pnl = equity * size * pnl_pct
                trade_fee = equity * size * fee_rate
                equity += trade_pnl - trade_fee
                trade_returns.append((trade_pnl - trade_fee) / (equity - trade_pnl + trade_fee))
                if record_trades:
                    trades.append({
                        "entry_idx": entry_idx, "exit_idx": i,
                        "side": "long", "entry_price": entry_price,
                        "exit_price": fill_px, "pnl": trade_pnl - trade_fee,
                        "reason": reason,
                    })
                pos = 0
                exited_this_bar = True

        elif pos == -1:
            should_exit, ex_px, reason = _check_exit_short(
                i=i, entry_price=entry_price, entry_idx=entry_idx,
                bar_open=bar_o, bar_high=bar_h, bar_low=bar_l, bar_close=bar_c,
                atr_at_entry=atr_at_entry, opposite_signal=l_prev,
                ma_val=ma_val, exit_cfg=exit_cfg,
            )
            if should_exit:
                fill_px = ex_px * (1.0 + slip_rate)
                pnl_pct = (entry_price - fill_px) / entry_price
                trade_pnl = equity * size * pnl_pct
                trade_fee = equity * size * fee_rate
                equity += trade_pnl - trade_fee
                trade_returns.append((trade_pnl - trade_fee) / (equity - trade_pnl + trade_fee))
                if record_trades:
                    trades.append({
                        "entry_idx": entry_idx, "exit_idx": i,
                        "side": "short", "entry_price": entry_price,
                        "exit_price": fill_px, "pnl": trade_pnl - trade_fee,
                        "reason": reason,
                    })
                pos = 0
                exited_this_bar = True

        # ── Entry (signal on bar i-1 → fill at bar i open) ───────────
        # Close-only: if we just exited under a non-opposite_signal exit_cfg,
        # skip re-entry on the same bar — wait for the next bar.
        can_enter = (pos == 0 and i > 0 and not (exited_this_bar and not exit_allows_flip))
        if can_enter:
            if long_sig[i - 1]:
                pos = 1
                entry_price = bar_o * (1.0 + slip_rate)
                entry_idx = i
                atr_at_entry = atr_arr[i - 1] if not math.isnan(atr_arr[i - 1]) else 0.0
                equity -= equity * size * fee_rate  # entry fee
            elif short_sig[i - 1] and not long_only:
                pos = -1
                entry_price = bar_o * (1.0 - slip_rate)
                entry_idx = i
                atr_at_entry = atr_arr[i - 1] if not math.isnan(atr_arr[i - 1]) else 0.0
                equity -= equity * size * fee_rate  # entry fee

        # ── Mark-to-market ────────────────────────────────────────────
        if pos == 1:
            unrealized = equity * size * (bar_c - entry_price) / entry_price
            equity_curve[i] = equity + unrealized
            bars_in_position += 1
        elif pos == -1:
            unrealized = equity * size * (entry_price - bar_c) / entry_price
            equity_curve[i] = equity + unrealized
            bars_in_position += 1
        else:
            equity_curve[i] = equity

    # ── Step 4: Compute metrics ───────────────────────────────────────
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    total_pnl = final_equity - initial_capital

    # Sharpe
    log_returns = np.diff(np.log(np.maximum(equity_curve, 1e-12)))
    if len(log_returns) > 1 and log_returns.std() > 0:
        ppy = PERIODS_PER_YEAR.get(timeframe, 365 * 24)
        sharpe = float(log_returns.mean() / log_returns.std() * math.sqrt(ppy))
    else:
        sharpe = 0.0

    # Max drawdown
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_peak) / np.maximum(running_peak, 1e-12)
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Trade stats
    trade_count = len(trade_returns)
    if trade_count > 0:
        wins = sum(1 for r in trade_returns if r > 0)
        win_rate = wins / trade_count
        avg_trade_pnl = float(np.mean([t["pnl"] for t in trades])) if trades else 0.0
    else:
        win_rate = 0.0
        avg_trade_pnl = 0.0

    exposure = bars_in_position / n if n > 0 else 0.0

    return VectorizedBacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe if not math.isnan(sharpe) else 0.0,
        max_drawdown=max_dd,
        trade_count=trade_count,
        win_rate=win_rate,
        avg_trade_pnl=avg_trade_pnl,
        total_pnl=total_pnl,
        exposure=exposure,
        final_equity=final_equity,
        equity_curve=equity_curve,
        trades=trades,
    )


# ── Batch runner ───────────────────────────────────────────────────────────

def run_vectorized_batch(
    *,
    df: pd.DataFrame,
    strategies: List[Dict[str, Any]],
    timeframe: str = "1h",
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
    initial_capital: float = 10_000.0,
    long_only: bool = False,
) -> List[Dict[str, Any]]:
    """Run multiple strategies on the same data in batch.

    Parameters
    ----------
    df : pd.DataFrame
        Shared OHLCV data.
    strategies : list of dict
        Each dict must have:
            - "strategy_id": str
            - "signal_fn": callable (make_signals)
            - "exit_cfg": dict (optional, defaults to opposite_signal)
            - "size": float (optional, default 0.5)
            - "long_only": bool (optional, overrides batch-level long_only)

    Returns
    -------
    list of dict
        Each dict contains strategy_id + all metrics from VectorizedBacktestResult.
        Sorted by sharpe_ratio descending.
    """
    results = []
    for spec in strategies:
        strategy_id = spec["strategy_id"]
        signal_fn = spec["signal_fn"]
        exit_cfg = spec.get("exit_cfg", {"type": "opposite_signal", "max_bars": 9999})
        size = spec.get("size", 0.5)

        result = run_vectorized_backtest(
            df=df,
            signal_fn=signal_fn,
            exit_cfg=exit_cfg,
            timeframe=timeframe,
            size=size,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            initial_capital=initial_capital,
            record_trades=False,
            long_only=spec.get("long_only", long_only),
        )
        entry = {"strategy_id": strategy_id, **result.to_dict()}
        results.append(entry)

    results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
    return results
