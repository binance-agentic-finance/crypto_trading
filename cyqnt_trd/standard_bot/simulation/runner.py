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
        position_qty = 0.0
        position_entry = 0.0
        current_instrument = request.instruments[0]
        commission_bps = float(request.fee_model.get("commission_bps", 0.0))
        slippage_bps = float(request.slippage_model.get("slippage_bps", 0.0))

        # Execution model: "close_fill" (legacy) or "next_bar_open" (realistic)
        execution_model = (request.extras or {}).get("execution_model", "close_fill")

        # --- Phase 2 state: exit management ---
        position_exit_spec: Optional[Dict] = None
        position_entry_idx: int = 0

        # --- Next-bar-open state: pending orders from previous bar ---
        pending_entry_signal = None  # queued BUY signal to fill next bar
        pending_exit_sell: bool = False  # queued SELL to fill next bar

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
                # Fill pending SELL (exit) at this bar's open
                if pending_exit_sell and position_qty > 0:
                    exit_px = float(latest_bar.open) * (1.0 - slippage_bps / 10_000.0)
                    fee = position_qty * exit_px * commission_bps / 10_000.0
                    pnl = (exit_px - position_entry) * position_qty - fee
                    cash = cash + position_qty * exit_px - fee
                    trade_rows.append(
                        {
                            "timestamp": timestamp,
                            "instrument_id": current_instrument,
                            "side": "sell",
                            "price": exit_px,
                            "quantity": position_qty,
                            "fee": fee,
                            "action": "exit",
                            "exit_reason": "signal_sell",
                            "entry_price": position_entry,
                            "pnl": pnl,
                        }
                    )
                    position_qty = 0.0
                    position_entry = 0.0
                    position_exit_spec = None
                    pending_exit_sell = False

                # Fill pending BUY (entry) at this bar's open
                if pending_entry_signal is not None and position_qty == 0:
                    signal = pending_entry_signal
                    execution_price = float(latest_bar.open) * (1.0 + slippage_bps / 10_000.0)
                    size_frac = float((signal.payload or {}).get("size", 1.0))
                    target_notional = cash * size_frac
                    qty = target_notional / execution_price if execution_price > 0 else 0.0
                    fee = qty * execution_price * commission_bps / 10_000.0
                    if qty > 0:
                        cash = cash - qty * execution_price - fee
                        position_qty = qty
                        position_entry = execution_price
                        position_entry_idx = snapshot_idx
                        payload = signal.payload or {}
                        position_exit_spec = payload.get("exit_spec")
                        if position_exit_spec:
                            position_exit_spec = self._finalize_exit_prices(
                                position_exit_spec, execution_price, side="long"
                            )
                        trade_rows.append(
                            {
                                "timestamp": timestamp,
                                "instrument_id": signal.instrument_id,
                                "side": signal.side.value,
                                "price": execution_price,
                                "quantity": qty,
                                "fee": fee,
                                "signal_id": signal.signal_id,
                                "action": "entry",
                            }
                        )
                    pending_entry_signal = None

            # =============================================================
            # Step 2: Check exit_spec (stop/TP/MA cross/opposite/max_bars)
            # =============================================================
            exited_this_bar = False
            # Track exit_cfg type for close-only enforcement in Step 3.
            exit_cfg_type = (position_exit_spec or {}).get("type", "opposite_signal")

            if position_qty > 0 and position_exit_spec is not None:
                # Detect opposite signal in this bar's batch (SELL while long).
                has_sell_signal = any(
                    s.kind == SignalKind.TRADE and s.side == TradeSide.SELL
                    for s in step_result.batch.signals
                )

                # Pre-compute MA value at this bar for ma_cross_exit.
                ma_value: Optional[float] = None
                if exit_cfg_type == "ma_cross_exit":
                    period = int(position_exit_spec.get("period", 50))
                    ma_type = str(position_exit_spec.get("ma_type", "ema"))
                    closes = [float(b.close) for b in series[-max(period * 3, period + 1):]]
                    if len(closes) >= period:
                        if ma_type == "ema":
                            # Simple EMA
                            k = 2.0 / (period + 1.0)
                            ema_v = closes[0]
                            for c in closes[1:]:
                                ema_v = c * k + ema_v * (1.0 - k)
                            ma_value = float(ema_v)
                        else:
                            ma_value = float(sum(closes[-period:]) / period)

                triggered, exit_price, exit_reason = self._check_exit(
                    spec=position_exit_spec,
                    entry_price=position_entry,
                    entry_idx=position_entry_idx,
                    current_idx=snapshot_idx,
                    bar_high=float(latest_bar.high),
                    bar_low=float(latest_bar.low),
                    bar_close=float(latest_bar.close),
                    ma_value=ma_value,
                    opposite_signal=has_sell_signal,
                )
                if triggered:
                    exit_px = exit_price * (1.0 - slippage_bps / 10_000.0)
                    fee = position_qty * exit_px * commission_bps / 10_000.0
                    pnl = (exit_px - position_entry) * position_qty - fee
                    cash = cash + position_qty * exit_px - fee
                    trade_rows.append(
                        {
                            "timestamp": timestamp,
                            "instrument_id": current_instrument,
                            "side": "sell",
                            "price": exit_px,
                            "quantity": position_qty,
                            "fee": fee,
                            "action": "exit",
                            "exit_reason": exit_reason,
                            "entry_price": position_entry,
                            "pnl": pnl,
                        }
                    )
                    position_qty = 0.0
                    position_entry = 0.0
                    position_exit_spec = None
                    # Cancel any queued SELL — Step 2 just exited.
                    pending_exit_sell = False
                    exited_this_bar = True

            # =============================================================
            # Step 3: Process trade signals
            # =============================================================
            trade_signals = [signal for signal in step_result.batch.signals if signal.kind == SignalKind.TRADE]

            # Close-only: when exit_cfg.type != "opposite_signal", we just
            # exited via Step 2 (or have an active spec) — let exit_spec own
            # the exits. SELL signals are ignored, and BUY entries on the
            # same bar as an exit are skipped to enforce close-only behavior.
            close_only = exit_cfg_type != "opposite_signal"

            if execution_model == "next_bar_open":
                # Queue signals for next bar fill
                for signal in trade_signals:
                    if signal.side == TradeSide.BUY and position_qty == 0 and pending_entry_signal is None:
                        if exited_this_bar and close_only:
                            continue  # close-only: no re-entry on same bar
                        pending_entry_signal = signal
                    elif signal.side == TradeSide.SELL and position_qty > 0:
                        if close_only:
                            continue  # exit_spec manages exits, ignore SELL
                        pending_exit_sell = True
            else:
                # Legacy close_fill model: execute immediately at bar close
                for signal in trade_signals:
                    execution_price = latest_bar.close * (
                        1.0 + (slippage_bps / 10_000.0 if signal.side == TradeSide.BUY else -slippage_bps / 10_000.0)
                    )
                    if signal.side == TradeSide.BUY and position_qty == 0:
                        if exited_this_bar and close_only:
                            continue  # close-only: no re-entry on same bar
                        size_frac = float((signal.payload or {}).get("size", 1.0))
                        target_notional = cash * size_frac
                        qty = target_notional / execution_price if execution_price > 0 else 0.0
                        fee = qty * execution_price * commission_bps / 10_000.0
                        if qty > 0:
                            cash = cash - qty * execution_price - fee
                            position_qty = qty
                            position_entry = execution_price
                            position_entry_idx = snapshot_idx
                            payload = signal.payload or {}
                            position_exit_spec = payload.get("exit_spec")
                            if position_exit_spec:
                                position_exit_spec = self._finalize_exit_prices(
                                    position_exit_spec, execution_price, side="long"
                                )
                            trade_rows.append(
                                {
                                    "timestamp": timestamp,
                                    "instrument_id": signal.instrument_id,
                                    "side": signal.side.value,
                                    "price": execution_price,
                                    "quantity": qty,
                                    "fee": fee,
                                    "signal_id": signal.signal_id,
                                    "action": "entry",
                                }
                            )
                    elif signal.side == TradeSide.SELL and position_qty > 0:
                        if close_only:
                            continue  # exit_spec manages exits, ignore SELL
                        fee = position_qty * execution_price * commission_bps / 10_000.0
                        realized = position_qty * execution_price - fee
                        pnl = (execution_price - position_entry) * position_qty - fee
                        cash = cash + realized
                        trade_rows.append(
                            {
                                "timestamp": timestamp,
                                "instrument_id": signal.instrument_id,
                                "side": signal.side.value,
                                "price": execution_price,
                                "quantity": position_qty,
                                "fee": fee,
                                "signal_id": signal.signal_id,
                                "action": "exit",
                                "entry_price": position_entry,
                                "pnl": pnl,
                            }
                        )
                        position_qty = 0.0
                        position_entry = 0.0
                        position_exit_spec = None

            equity = cash + position_qty * latest_bar.close
            equity_curve.append(EquityPoint(timestamp=timestamp, equity=float(equity), cash=float(cash)))

        if ordered and position_qty > 0:
            final_snapshot = ordered[-1]
            market = final_snapshot.require_market()
            series = market.bars.get(market.key(current_instrument, request.primary_timeframe), [])
            final_bar = series[-1]
            exit_price = final_bar.close
            fee = position_qty * exit_price * commission_bps / 10_000.0
            pnl = (exit_price - position_entry) * position_qty - fee
            cash = cash + position_qty * exit_price - fee
            trade_rows.append(
                {
                    "timestamp": final_bar.timestamp,
                    "instrument_id": current_instrument,
                    "side": "sell",
                    "price": exit_price,
                    "quantity": position_qty,
                    "fee": fee,
                    "action": "forced_exit",
                    "entry_price": position_entry,
                    "pnl": pnl,
                }
            )
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
        """
        side = spec.get("side", "long")
        etype = spec.get("type", "opposite_signal")

        # Priority 1 & 2: Stop Loss / Take Profit (intra-bar, side-aware)
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

        # time_only / ma_cross_exit / opposite_signal: no absolute prices needed
        return spec
