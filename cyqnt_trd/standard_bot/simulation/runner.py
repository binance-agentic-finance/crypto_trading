"""
Snapshot-driven backtest runner for the MVP.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import numpy as np

from ..core import BacktestRequest, BacktestResult, EquityPoint, SignalContext, SignalKind, TradeSide
from ..signal.registry import SignalPluginRegistry
from .metrics_kernels import compute_equity_statistics

SIMULATION_NAMESPACE = uuid.UUID("0bf7f6fd-3ca7-57aa-8266-4f22048d8bf8")


def _periods_per_year(timeframe: str) -> float:
    """Return the number of bars in one year for annualization."""
    if not timeframe or len(timeframe) < 2:
        return 0.0
    unit = timeframe[-1]
    try:
        size = float(timeframe[:-1])
    except ValueError:
        return 0.0
    if size <= 0.0:
        return 0.0
    if unit == "m":
        return 365.0 * 24.0 * 60.0 / size
    if unit == "h":
        return 365.0 * 24.0 / size
    if unit == "d":
        return 365.0 / size
    if unit == "w":
        return 365.0 / (7.0 * size)
    if unit == "M":
        return 12.0 / size
    return 0.0


class SnapshotBacktestRunner:
    def __init__(self, *, signal_registry: SignalPluginRegistry) -> None:
        self.signal_registry = signal_registry

    def run(
        self,
        *,
        request: BacktestRequest,
        snapshots: list,
        signal_context: Optional[SignalContext] = None,
    ) -> BacktestResult:
        ordered = sorted(
            snapshots,
            key=lambda snapshot: snapshot.meta.decision_as_of or snapshot.meta.assembled_at,
        )
        plugin_states: Dict[str, object] = {}
        trade_rows: List[Dict] = []
        equity_curve: List[EquityPoint] = []
        signal_batches = []

        cash = float(request.initial_capital)
        position_qty = 0.0                     # magnitude (>= 0)
        position_side: Optional[str] = None    # "long" | "short" | None
        position_entry = 0.0
        current_instrument = request.instruments[0]
        commission_bps = float(request.fee_model.get("commission_bps", 0.0))
        slippage_bps = float(request.slippage_model.get("slippage_bps", 0.0))
        # long_only: when True, SELL signals only CLOSE longs (never OPEN shorts).
        # Auto-set by the entrypoint (spot=True, futures=False) or explicit --long-only.
        long_only = bool((request.extras or {}).get("long_only", False))
        _comm = commission_bps / 10_000.0
        _slip = slippage_bps / 10_000.0

        # Execution model: "close_fill" (legacy) or "next_bar_open" (realistic)
        execution_model = (request.extras or {}).get("execution_model", "close_fill")

        # --- Position / exit state ---
        position_exit_spec: Optional[Dict] = None
        position_entry_idx: int = 0

        # --- Next-bar-open state: pending orders from previous bar ---
        pending_entry_signal = None   # queued entry signal (BUY->long / SELL->short)
        pending_close: bool = False   # queued close of the current position

        def _open(side, base_px, idx, signal, ts):
            """Open a long or short position (mirrored spot-cash accounting).

            long : pay notional, cash -= qty*fill (+fee); equity = cash + qty*px
            short: receive proceeds, cash += qty*fill (-fee); equity = cash - qty*px
            The long branch is arithmetically identical to the pre-short code, so
            long-only backtests are unchanged.
            """
            nonlocal cash, position_qty, position_side, position_entry
            nonlocal position_exit_spec, position_entry_idx
            fill = base_px * (1.0 + _slip) if side == "long" else base_px * (1.0 - _slip)
            size_frac = float((signal.payload or {}).get("size", 1.0))
            qty = (cash * size_frac) / fill if fill > 0 else 0.0
            if qty <= 0:
                return
            fee = qty * fill * _comm
            if side == "long":
                cash -= qty * fill + fee
            else:
                cash += qty * fill - fee
            position_qty = qty
            position_side = side
            position_entry = fill
            position_entry_idx = idx
            spec = (signal.payload or {}).get("exit_spec")
            position_exit_spec = self._finalize_exit_prices(spec, fill, side=side) if spec else None
            trade_rows.append({
                "timestamp": ts, "instrument_id": signal.instrument_id,
                "side": "buy" if side == "long" else "sell", "price": fill,
                "quantity": qty, "fee": fee, "signal_id": signal.signal_id,
                "action": "entry", "position_side": side,
            })

        def _close(base_px, reason, ts, apply_slip=True):
            """Close the open position (side-aware PnL)."""
            nonlocal cash, position_qty, position_side, position_entry, position_exit_spec
            side = position_side
            if apply_slip:
                fill = base_px * (1.0 - _slip) if side == "long" else base_px * (1.0 + _slip)
            else:
                fill = base_px
            fee = position_qty * fill * _comm
            if side == "long":
                cash += position_qty * fill - fee
                pnl = (fill - position_entry) * position_qty - fee
                close_side = "sell"
            else:
                cash -= position_qty * fill + fee
                pnl = (position_entry - fill) * position_qty - fee
                close_side = "buy"
            trade_rows.append({
                "timestamp": ts, "instrument_id": current_instrument, "side": close_side,
                "price": fill, "quantity": position_qty, "fee": fee, "action": "exit",
                "exit_reason": reason, "entry_price": position_entry, "pnl": pnl,
                "position_side": side,
            })
            position_qty = 0.0
            position_side = None
            position_entry = 0.0
            position_exit_spec = None

        def _equity(mark_px):
            if position_side == "long":
                return cash + position_qty * mark_px
            if position_side == "short":
                return cash - position_qty * mark_px
            return cash

        for snapshot_idx, snapshot in enumerate(ordered):
            step_result = self.signal_registry.run_pipeline_step(
                snapshot,
                request.signal_pipeline.plugin_chain,
                previous_states=plugin_states,
                context=signal_context,
            )
            plugin_states = step_result.states
            signal_batches.append(step_result.batch)
            timestamp = snapshot.meta.decision_as_of or snapshot.meta.assembled_at

            market = snapshot.require_market()
            primary_key = market.key(current_instrument, request.primary_timeframe)
            series = market.bars.get(primary_key, [])
            if not series:
                continue
            latest_bar = series[-1]

            # =============================================================
            # Step 1: Fill pending orders from PREVIOUS bar (next_bar_open)
            # =============================================================
            if execution_model == "next_bar_open":
                # Fill pending close (exit) at this bar's open
                if pending_close and position_side is not None:
                    _close(float(latest_bar.open), "signal", timestamp)
                    pending_close = False

                # Fill pending entry (long or short) at this bar's open
                if pending_entry_signal is not None and position_side is None:
                    sig = pending_entry_signal
                    side = "long" if sig.side == TradeSide.BUY else "short"
                    _open(side, float(latest_bar.open), snapshot_idx, sig, timestamp)
                    pending_entry_signal = None

            # =============================================================
            # Step 2: Check exit_spec (stop/TP/MA cross/opposite/max_bars)
            # =============================================================
            exited_this_bar = False
            # Track exit_cfg type for close-only enforcement in Step 3.
            exit_cfg_type = (position_exit_spec or {}).get("type", "opposite_signal")

            if position_side is not None and position_exit_spec is not None:
                # Opposite signal = a signal against the current side
                # (long → SELL present; short → BUY present).
                if position_side == "long":
                    has_opp = any(
                        s.kind == SignalKind.TRADE and s.side == TradeSide.SELL
                        for s in step_result.batch.signals
                    )
                else:
                    has_opp = any(
                        s.kind == SignalKind.TRADE and s.side == TradeSide.BUY
                        for s in step_result.batch.signals
                    )

                # Pre-compute MA value at this bar for ma_cross_exit — use the
                # canonical blocks indicator library (same as the vectorized
                # engine and all block strategies) instead of a hand-rolled EMA,
                # so the two backtest engines agree on ma_cross exits.
                ma_value: Optional[float] = None
                if exit_cfg_type == "ma_cross_exit":
                    period = int(position_exit_spec.get("period", 50))
                    ma_type = str(position_exit_spec.get("ma_type", "ema"))
                    closes = [float(b.close) for b in series]
                    if len(closes) >= period:
                        import pandas as _pd
                        from ...blocks import indicators as _ind

                        _s = _pd.Series(closes, dtype=float)
                        _ma = _ind.ema(_s, period) if ma_type == "ema" else _ind.sma(_s, period)
                        ma_value = float(_ma.iloc[-1])

                triggered, exit_price, exit_reason = self._check_exit(
                    spec=position_exit_spec,
                    entry_price=position_entry,
                    entry_idx=position_entry_idx,
                    current_idx=snapshot_idx,
                    bar_high=float(latest_bar.high),
                    bar_low=float(latest_bar.low),
                    bar_close=float(latest_bar.close),
                    ma_value=ma_value,
                    opposite_signal=has_opp,
                )
                if triggered:
                    _close(exit_price, exit_reason, timestamp)
                    pending_close = False   # Step 2 just exited; cancel any queued close
                    exited_this_bar = True

            # =============================================================
            # Step 3: Process trade signals
            # =============================================================
            trade_signals = [signal for signal in step_result.batch.signals if signal.kind == SignalKind.TRADE]

            # Close-only: for non-opposite_signal exit types, exit_spec owns the
            # exits — no opposite-side entries/flips on the same bar.
            close_only = exit_cfg_type != "opposite_signal"

            if execution_model == "next_bar_open":
                # Queue signals for next-bar fill
                for signal in trade_signals:
                    want = "long" if signal.side == TradeSide.BUY else "short"
                    if want == "short" and long_only:
                        # SELL only closes an open long; never opens a short.
                        if position_side == "long" and not close_only and pending_entry_signal is None:
                            pending_close = True
                        continue
                    if position_side is None and pending_entry_signal is None:
                        if exited_this_bar and close_only:
                            continue  # close-only: no re-entry on same bar
                        pending_entry_signal = signal
                    elif position_side is not None and want != position_side and not close_only:
                        # Opposite signal → close next bar (flip re-enters once flat).
                        pending_close = True
            else:
                # Legacy close_fill model: execute immediately at bar close
                for signal in trade_signals:
                    want = "long" if signal.side == TradeSide.BUY else "short"
                    base = float(latest_bar.close)
                    if want == "short" and long_only:
                        if position_side == "long" and not close_only:
                            _close(base, "signal_sell", timestamp)
                            exited_this_bar = True
                        continue
                    if position_side is None:
                        if exited_this_bar and close_only:
                            continue  # close-only: no re-entry on same bar
                        _open(want, base, snapshot_idx, signal, timestamp)
                    elif want != position_side and not close_only:
                        # Flip: close current, open opposite on the same bar close.
                        _close(base, "signal_flip", timestamp)
                        _open(want, base, snapshot_idx, signal, timestamp)

            equity = _equity(float(latest_bar.close))
            equity_curve.append(EquityPoint(timestamp=timestamp, equity=float(equity), cash=float(cash)))

        if ordered and position_side is not None:
            final_snapshot = ordered[-1]
            market = final_snapshot.require_market()
            series = market.bars.get(market.key(current_instrument, request.primary_timeframe), [])
            final_bar = series[-1]
            # Close at final bar close, no slippage (matches legacy forced-exit).
            _close(float(final_bar.close), "forced_exit", final_bar.timestamp, apply_slip=False)
            if equity_curve:
                equity_curve[-1] = EquityPoint(
                    timestamp=equity_curve[-1].timestamp,
                    equity=float(cash),
                    cash=float(cash),
                )

        run_id = str(
            uuid.uuid5(
                SIMULATION_NAMESPACE,
                "%s|%s|%s|%s"
                % (request.request_id, request.start_ts, request.end_ts, len(ordered)),
            )
        )
        initial = float(request.initial_capital)
        final_equity = float(equity_curve[-1].equity) if equity_curve else initial
        total_return = (final_equity - initial) / initial if initial else 0.0

        # --- Compute performance metrics (same as NumbaBacktestRunner) ---
        ppy = _periods_per_year(request.primary_timeframe)
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        mean_bar_return = 0.0
        bar_return_volatility = 0.0

        if equity_curve:
            equity_values = np.array([pt.equity for pt in equity_curve], dtype=np.float64)
            drawdowns, max_drawdown, mean_bar_return, bar_return_volatility, sharpe_ratio = (
                compute_equity_statistics(equity_values, ppy)
            )

        # Trade-level statistics
        trade_pnls = [t["pnl"] for t in trade_rows if "pnl" in t]
        trade_count = len(trade_rows)
        entry_count = sum(1 for t in trade_rows if t.get("action") in ("entry",))
        exit_count = sum(1 for t in trade_rows if t.get("action") in ("exit", "forced_exit"))
        win_count = sum(1 for p in trade_pnls if p > 0)
        win_rate = win_count / len(trade_pnls) if trade_pnls else 0.0
        avg_trade_pnl = float(np.mean(trade_pnls)) if trade_pnls else 0.0
        total_pnl = float(np.sum(trade_pnls)) if trade_pnls else 0.0

        return BacktestResult(
            request_id=request.request_id,
            total_return=float(total_return),
            equity_curve=equity_curve,
            metrics={
                "snapshot_count": float(len(ordered)),
                "trade_count": float(len(trade_pnls)),
                "final_equity": float(final_equity),
                "total_return": float(total_return),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "mean_bar_return": float(mean_bar_return),
                "bar_return_volatility": float(bar_return_volatility),
                "win_rate": float(win_rate),
                "win_count": float(win_count),
                "avg_trade_pnl": float(avg_trade_pnl),
                "total_pnl": float(total_pnl),
            },
            signal_batches=signal_batches,
            extras={"run_id": run_id, "trades": trade_rows},
        )

    # --- Phase 2 helpers: exit condition checking ---

    @staticmethod
    def _check_exit(
        *,
        spec: Dict,
        entry_price: float,
        entry_idx: int,
        current_idx: int,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        ma_value: Optional[float] = None,
        opposite_signal: bool = False,
    ) -> tuple:
        """Check if any exit condition triggers on this bar.

        Returns (triggered: bool, exit_price: float, reason: str).

        Unified priority order (first hit wins):
            1. Stop Loss      — intra-bar (side-aware)
            2. Take Profit    — intra-bar (side-aware)
            3. MA cross       — bar close, fill at bar close (only when type=ma_cross_exit)
            4. Opposite signal — bar close, fill at bar close (always checked)
            5. max_bars       — bar close, fill at bar close (timeout)

        Side-aware:
          Long:  SL triggers when bar_low  <= stop (price drops to stop)
                 TP triggers when bar_high >= tp   (price rises to TP)
          Short: SL triggers when bar_high >= stop (price rises to stop)
                 TP triggers when bar_low  <= tp   (price drops to TP)

        ATR trailing (etype == "atr_trailing_stop"):
            Mutates ``spec["running_peak"]`` in place each call:
              long  → max(peak, bar_high), trigger when bar_low <= peak - trail*ATR
              short → min(peak, bar_low),  trigger when bar_high >= peak + trail*ATR
        """
        side = spec.get("side", "long")
        etype = spec.get("type", "opposite_signal")

        # Priority 1 & 2: Stop Loss / Take Profit (intra-bar, side-aware)
        # For atr_trailing_stop we update running_peak then derive a dynamic
        # stop_price; treat that as the SL price for the priority order.
        if etype == "atr_trailing_stop":
            atr = spec.get("atr_at_entry")
            trail_mult = float(spec.get("trail_mult", 3.0))
            if atr is not None and atr > 0:
                peak = spec.get("running_peak", entry_price)
                if side == "long":
                    peak = max(float(peak), bar_high)
                    spec["running_peak"] = peak
                    stop_price_dyn = peak - trail_mult * float(atr)
                    spec["stop_loss_price"] = stop_price_dyn  # for inspection
                    if bar_low <= stop_price_dyn:
                        return True, stop_price_dyn, "trailing_stop"
                else:
                    peak = min(float(peak), bar_low)
                    spec["running_peak"] = peak
                    stop_price_dyn = peak + trail_mult * float(atr)
                    spec["stop_loss_price"] = stop_price_dyn
                    if bar_high >= stop_price_dyn:
                        return True, stop_price_dyn, "trailing_stop"
            # Fall through to opposite_signal / max_bars below.
        else:
            stop_price = spec.get("stop_loss_price")
            tp_price = spec.get("take_profit_price")

            if side == "long":
                if stop_price is not None and bar_low <= stop_price:
                    return True, stop_price, "stop_loss"
                if tp_price is not None and bar_high >= tp_price:
                    return True, tp_price, "take_profit"
            else:
                # Short: stop when price rises, TP when price drops
                if stop_price is not None and bar_high >= stop_price:
                    return True, stop_price, "stop_loss"
                if tp_price is not None and bar_low <= tp_price:
                    return True, tp_price, "take_profit"

        # Priority 3: MA cross (bar close) — only when type=ma_cross_exit
        if etype == "ma_cross_exit" and ma_value is not None:
            if side == "long" and bar_close < ma_value:
                return True, bar_close, "ma_cross"
            if side == "short" and bar_close > ma_value:
                return True, bar_close, "ma_cross"

        # Priority 4: Opposite signal (bar close) — checked for ALL exit types
        if opposite_signal:
            return True, bar_close, "opposite_signal"

        # Priority 5: max_bars timeout (bar close)
        max_bars = spec.get("max_bars", 9999)
        bars_held = current_idx - entry_idx
        if bars_held >= max_bars:
            return True, bar_close, "max_bars"

        return False, 0.0, ""

    @staticmethod
    def _finalize_exit_prices(spec: Dict, fill_price: float, side: str = "long") -> Dict:
        """Recompute absolute stop/TP prices using actual fill price.

        Side-aware:
          Long:  SL = fill × (1 - stop_pct),  TP = fill × (1 + tp_pct)
          Short: SL = fill × (1 + stop_pct),  TP = fill × (1 - tp_pct)

        For atr_trailing_stop, anchor ``running_peak`` to ``fill_price`` so
        the initial trail width is computed from the actual fill.
        """
        spec = dict(spec)  # copy
        spec["side"] = side
        etype = spec.get("type", "")

        if etype == "pct_stop_tp":
            stop_pct = float(spec.get("stop_pct", 0.02))
            tp_pct = float(spec.get("tp_pct", 0.04))
            if side == "long":
                spec["stop_loss_price"] = fill_price * (1.0 - stop_pct)
                spec["take_profit_price"] = fill_price * (1.0 + tp_pct)
            else:  # short
                spec["stop_loss_price"] = fill_price * (1.0 + stop_pct)
                spec["take_profit_price"] = fill_price * (1.0 - tp_pct)

        elif etype == "atr_stop_tp":
            atr = spec.get("atr_at_entry")
            if atr is not None and atr > 0:
                stop_mult = float(spec.get("stop_mult", 2.0))
                tp_mult = float(spec.get("tp_mult", 3.0))
                if side == "long":
                    spec["stop_loss_price"] = fill_price - stop_mult * atr
                    spec["take_profit_price"] = fill_price + tp_mult * atr
                else:  # short
                    spec["stop_loss_price"] = fill_price + stop_mult * atr
                    spec["take_profit_price"] = fill_price - tp_mult * atr

        elif etype == "atr_trailing_stop":
            # Re-anchor the trailing peak to the actual fill price so the
            # initial trail width uses a faithful starting point.
            spec["running_peak"] = float(fill_price)
            atr = spec.get("atr_at_entry")
            if atr is not None and atr > 0:
                trail_mult = float(spec.get("trail_mult", 3.0))
                if side == "long":
                    spec["stop_loss_price"] = fill_price - trail_mult * atr
                else:
                    spec["stop_loss_price"] = fill_price + trail_mult * atr

        # time_only / ma_cross_exit / opposite_signal: no absolute prices needed
        return spec
