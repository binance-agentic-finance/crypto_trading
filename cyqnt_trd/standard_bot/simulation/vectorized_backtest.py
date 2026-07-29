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
import warnings
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
    running_peak: float = 0.0,
) -> Tuple[bool, float, str, float]:
    """Return (should_exit, exit_price, reason, new_running_peak) for a long position.

    Unified exit priority (first hit wins):
        1. Stop Loss / Trailing Stop — intra-bar, fill at stop price (pct_stop_tp / atr_stop_tp / atr_trailing_stop)
        2. Take Profit    — intra-bar, fill at tp price   (pct_stop_tp / atr_stop_tp)
        3. MA cross       — bar close,  fill at next-bar open (ma_cross_exit only)
        4. Opposite signal — bar close, fill at next-bar open (always checked, all types)
        5. max_bars       — bar close,  fill at next-bar open (timeout)

    The ``running_peak`` parameter is the previous bar's tracked peak (for
    ``atr_trailing_stop``). Returned ``new_running_peak`` is updated to
    ``max(running_peak, bar_high)`` for long. Caller must persist it across
    bars while the position is open and reset it on entry.
    """
    etype = exit_cfg.get("type", "opposite_signal")
    max_bars = int(exit_cfg.get("max_bars", 9999))
    new_peak = running_peak

    # Priority 0: exits that fill at THIS bar's OPEN. Chronologically the open
    # precedes this bar's high/low, so they must be tested before the intra-bar
    # stop/TP — otherwise a stop only touched later in the same bar pre-empts an
    # exit that had already been filled at the open, and this engine disagrees
    # with the event-driven one (which acts on the same information one bar
    # earlier, at that bar's close).
    # ``opposite_signal`` is checked BEFORE ``max_bars``: it is a decision made
    # on bar i-1's CLOSE, whereas max_bars is a bar-i condition. Bar i-1's close
    # happens first, so when both are due at bar i the opposite signal wins —
    # which is also what the event engine does, because it sees the opposite
    # signal a whole bar earlier (at bar i-1) and by then max_bars is not yet due.
    if opposite_signal:
        return True, bar_open, "opposite_signal", new_peak
    if i - entry_idx >= max_bars:
        return True, bar_open, "max_bars", new_peak

    # Priority 1 & 2: Stop Loss / Take Profit (intra-bar) — only when configured
    if etype == "pct_stop_tp":
        stop_pct = float(exit_cfg.get("stop_pct", 0.02))
        tp_pct = float(exit_cfg.get("tp_pct", 0.04))
        stop_price = entry_price * (1.0 - stop_pct)
        tp_price = entry_price * (1.0 + tp_pct)
        if bar_low <= stop_price:
            return True, stop_price, "stop_loss", new_peak
        if bar_high >= tp_price:
            return True, tp_price, "take_profit", new_peak
    elif etype == "atr_stop_tp":
        stop_mult = float(exit_cfg.get("stop_mult", 2.0))
        tp_mult = float(exit_cfg.get("tp_mult", 3.0))
        if atr_at_entry > 0 and not math.isnan(atr_at_entry):
            stop_price = entry_price - stop_mult * atr_at_entry
            tp_price = entry_price + tp_mult * atr_at_entry
            if bar_low <= stop_price:
                return True, stop_price, "stop_loss", new_peak
            if bar_high >= tp_price:
                return True, tp_price, "take_profit", new_peak
    elif etype == "atr_trailing_stop":
        # Update running peak first, then derive dynamic stop from
        # entry-time ATR. Fill at the dynamic stop_price for faithfulness.
        trail_mult = float(exit_cfg.get("trail_mult", 3.0))
        if atr_at_entry > 0 and not math.isnan(atr_at_entry):
            new_peak = max(running_peak, bar_high)
            stop_price = new_peak - trail_mult * atr_at_entry
            if bar_low <= stop_price:
                return True, stop_price, "trailing_stop", new_peak

    # Priority 3: MA cross (bar close) — only when type=ma_cross_exit
    if etype == "ma_cross_exit":
        if ma_val is not None and bar_close < ma_val:
            # Fill at bar close (decision bar) — no lookahead, matches the
            # event-driven SnapshotBacktestRunner. (Was bar_open = lookahead.)
            return True, bar_close, "ma_cross", new_peak

    return False, 0.0, "", new_peak


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
    running_peak: float = 0.0,
) -> Tuple[bool, float, str, float]:
    """Return (should_exit, exit_price, reason, new_running_peak) for a short position.

    Same priority as long, with side-aware SL/TP and MA cross direction.

    For ``atr_trailing_stop``, ``running_peak`` is the *running low*; updated
    to ``min(running_peak, bar_low)`` and stop = peak + trail_mult * ATR.
    """
    etype = exit_cfg.get("type", "opposite_signal")
    max_bars = int(exit_cfg.get("max_bars", 9999))
    new_peak = running_peak

    # Priority 0: open-filled exits first, opposite before max_bars — see
    # _check_exit_long for the rationale.
    if opposite_signal:
        return True, bar_open, "opposite_signal", new_peak
    if i - entry_idx >= max_bars:
        return True, bar_open, "max_bars", new_peak

    # Priority 1 & 2: Stop Loss / Take Profit (intra-bar)
    if etype == "pct_stop_tp":
        stop_pct = float(exit_cfg.get("stop_pct", 0.02))
        tp_pct = float(exit_cfg.get("tp_pct", 0.04))
        stop_price = entry_price * (1.0 + stop_pct)  # short: stop above entry
        tp_price = entry_price * (1.0 - tp_pct)      # short: TP below entry
        if bar_high >= stop_price:
            return True, stop_price, "stop_loss", new_peak
        if bar_low <= tp_price:
            return True, tp_price, "take_profit", new_peak
    elif etype == "atr_stop_tp":
        stop_mult = float(exit_cfg.get("stop_mult", 2.0))
        tp_mult = float(exit_cfg.get("tp_mult", 3.0))
        if atr_at_entry > 0 and not math.isnan(atr_at_entry):
            stop_price = entry_price + stop_mult * atr_at_entry  # short: stop above
            tp_price = entry_price - tp_mult * atr_at_entry      # short: TP below
            if bar_high >= stop_price:
                return True, stop_price, "stop_loss", new_peak
            if bar_low <= tp_price:
                return True, tp_price, "take_profit", new_peak
    elif etype == "atr_trailing_stop":
        trail_mult = float(exit_cfg.get("trail_mult", 3.0))
        if atr_at_entry > 0 and not math.isnan(atr_at_entry):
            new_peak = min(running_peak, bar_low)
            stop_price = new_peak + trail_mult * atr_at_entry
            if bar_high >= stop_price:
                return True, stop_price, "trailing_stop", new_peak

    # Priority 3: MA cross (bar close) — short exits when close > MA
    if etype == "ma_cross_exit":
        if ma_val is not None and bar_close > ma_val:
            # Fill at bar close (decision bar) — no lookahead, matches the
            # event-driven SnapshotBacktestRunner. (Was bar_open = lookahead.)
            return True, bar_close, "ma_cross", new_peak

    return False, 0.0, "", new_peak


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
        # Don't return an all-zero result silently — a caller sweeping many
        # windows would read it as "strategy made no money" instead of
        # "not enough data to backtest".
        warnings.warn(
            f"run_vectorized_backtest: only {len(df)} bars supplied (minimum 50); "
            "returning an empty result with total_return=0.",
            RuntimeWarning,
            stacklevel=2,
        )
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
    # ATR-trailing state: running peak (long: max of bar_highs since entry;
    # short: min of bar_lows since entry). Reset on entry.
    trail_peak = 0.0

    equity = initial_capital
    equity_curve = np.empty(n, dtype=np.float64)
    trade_returns: List[float] = []
    trades: List[Dict[str, Any]] = []
    bars_in_position = 0
    last_exit_reason = ""

    def _try_exit(*, i, bar_o, bar_h, bar_l, bar_c, ma_val, l_prev, s_prev) -> bool:
        """Run the exit check for the open position at bar *i*.

        Returns True when the position was closed. Mutates ``pos`` / ``equity``
        / ``trail_peak`` and appends to ``trade_returns`` / ``trades``.
        Called twice per bar: once before entry (normal path) and once right
        after an entry fill (entry-bar exit risk).
        """
        nonlocal pos, equity, trail_peak, last_exit_reason

        if pos == 1:
            should_exit, ex_px, reason, new_peak = _check_exit_long(
                i=i, entry_price=entry_price, entry_idx=entry_idx,
                bar_open=bar_o, bar_high=bar_h, bar_low=bar_l, bar_close=bar_c,
                atr_at_entry=atr_at_entry, opposite_signal=s_prev,
                ma_val=ma_val, exit_cfg=exit_cfg,
                running_peak=trail_peak,
            )
            trail_peak = new_peak
            if not should_exit:
                return False
            fill_px = ex_px * (1.0 - slip_rate)
            pnl_pct = (fill_px - entry_price) / entry_price
            side = "long"
        elif pos == -1:
            should_exit, ex_px, reason, new_peak = _check_exit_short(
                i=i, entry_price=entry_price, entry_idx=entry_idx,
                bar_open=bar_o, bar_high=bar_h, bar_low=bar_l, bar_close=bar_c,
                atr_at_entry=atr_at_entry, opposite_signal=l_prev,
                ma_val=ma_val, exit_cfg=exit_cfg,
                running_peak=trail_peak,
            )
            trail_peak = new_peak
            if not should_exit:
                return False
            fill_px = ex_px * (1.0 + slip_rate)
            pnl_pct = (entry_price - fill_px) / entry_price
            side = "short"
        else:
            return False

        trade_pnl = equity * size * pnl_pct
        trade_fee = equity * size * fee_rate
        equity += trade_pnl - trade_fee
        trade_returns.append((trade_pnl - trade_fee) / (equity - trade_pnl + trade_fee))
        if record_trades:
            trades.append({
                "entry_idx": entry_idx, "exit_idx": i,
                "side": side, "entry_price": entry_price,
                "exit_price": fill_px, "pnl": trade_pnl - trade_fee,
                "reason": reason,
            })
        pos = 0
        last_exit_reason = reason
        return True

    for i in range(n):
        bar_o = open_arr[i]
        bar_h = high_arr[i]
        bar_l = low_arr[i]
        bar_c = close_arr[i]

        # Previous bar signals (for exit checks using opposite_signal)
        l_prev = bool(long_sig[i - 1]) if i > 0 else False
        s_prev = bool(short_sig[i - 1]) if i > 0 else False
        ma_val = float(ma_arr[i]) if ma_arr is not None and not math.isnan(ma_arr[i]) else None

        # ── Exit check ────────────────────────────────────────────────
        exited_this_bar = _try_exit(
            i=i, bar_o=bar_o, bar_h=bar_h, bar_l=bar_l, bar_c=bar_c,
            ma_val=ma_val, l_prev=l_prev, s_prev=s_prev,
        )

        # ── Entry (signal on bar i-1 → fill at bar i open) ───────────
        # Close-only: if we just exited under a non-opposite_signal exit_cfg,
        # skip re-entry on the same bar — wait for the next bar.
        #
        # Exception: an ``opposite_signal`` exit was decided on bar i-1's close
        # and fills at bar i's OPEN — the very same information and fill point
        # the entry would use. Blocking it here delayed the flip by one bar
        # versus the event-driven engine, which books that exit on bar i-1 and
        # queues bar i-1's signal for a bar-i-open fill. So allow it.
        # ``max_bars`` deliberately does NOT get this exemption: the event
        # engine books it on bar i and then queues *bar i's* signal, so both
        # engines re-enter at bar i+1 and already agree.
        same_bar_reentry_ok = last_exit_reason == "opposite_signal"
        can_enter = (
            pos == 0 and i > 0
            and not (exited_this_bar and not exit_allows_flip and not same_bar_reentry_ok)
        )
        entered_this_bar = False
        if can_enter:
            if long_sig[i - 1]:
                pos = 1
                entry_price = bar_o * (1.0 + slip_rate)
                entry_idx = i
                atr_at_entry = atr_arr[i - 1] if not math.isnan(atr_arr[i - 1]) else 0.0
                # Initialize running peak to current bar high so the first
                # exit check at i+1 sees max(entry_high, ...).
                trail_peak = bar_h
                equity -= equity * size * fee_rate  # entry fee
                entered_this_bar = True
            elif short_sig[i - 1] and not long_only:
                pos = -1
                entry_price = bar_o * (1.0 - slip_rate)
                entry_idx = i
                atr_at_entry = atr_arr[i - 1] if not math.isnan(atr_arr[i - 1]) else 0.0
                trail_peak = bar_l
                equity -= equity * size * fee_rate  # entry fee
                entered_this_bar = True

        # ── Exit risk ON the entry bar ────────────────────────────────
        # A position filled at bar i's open is exposed to bar i's own high/low
        # and to bar i's close. The event-driven SnapshotBacktestRunner fills
        # the entry (Step 1) *before* checking exits (Step 2), so it can stop
        # out on the entry bar; skipping that here understated stop-loss risk
        # and was a main source of engine disagreement.
        #
        # ``opposite_signal`` is deliberately suppressed: on the entry bar the
        # only signal available to this engine is bar i-1's (the one that
        # caused the entry). The event engine sees bar i's signal there, and
        # this engine already picks that up on the next iteration via
        # ``s_prev``/``l_prev`` — filling at bar i+1's open, which equals
        # bar i's close in a continuous market. So the two stay aligned.
        if entered_this_bar:
            _try_exit(
                i=i, bar_o=bar_o, bar_h=bar_h, bar_l=bar_l, bar_c=bar_c,
                ma_val=ma_val, l_prev=False, s_prev=False,
            )

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

    # ── Forced exit of any position still open at the end of the data ──
    # Matches SnapshotBacktestRunner, which force-closes at the final bar's
    # close with NO slippage and records it as a trade. Previously this engine
    # left the position open: its unrealized PnL still landed in equity_curve
    # (so total_return counted it) but it was excluded from trades /
    # trade_count / win_rate, and never paid an exit fee.
    if pos != 0 and n > 0:
        fill_px = close_arr[n - 1]
        pnl_pct = ((fill_px - entry_price) if pos == 1 else (entry_price - fill_px)) / entry_price
        trade_pnl = equity * size * pnl_pct
        trade_fee = equity * size * fee_rate
        equity += trade_pnl - trade_fee
        trade_returns.append((trade_pnl - trade_fee) / (equity - trade_pnl + trade_fee))
        if record_trades:
            trades.append({
                "entry_idx": entry_idx, "exit_idx": n - 1,
                "side": "long" if pos == 1 else "short", "entry_price": entry_price,
                "exit_price": fill_px, "pnl": trade_pnl - trade_fee,
                "reason": "forced_exit",
            })
        pos = 0
        equity_curve[n - 1] = equity

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
