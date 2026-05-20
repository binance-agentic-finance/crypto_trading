"""
Persistent Python-engine paper trading session for ``cyqnt_trd.blocks`` strategies.

This is the Python-engine counterpart of :class:`NumbaLivePaperSession`.
It accepts strategies registered via :func:`cyqnt_trd.blocks.strategy.register`
(as opposed to Numba kernels registered via ``NumbaBacktestRunner.register_kernel``)
and drives them through the same next-bar-open execution model so that paper
trading results are consistent with backtest results from
``mvp_backtest --engine python``.

Architecture mirrors ``NumbaLivePaperSession``:

* Maintains a growing in-memory bar history (parallel lists).
* On each ``tick(bar)``: execute pending order at this bar's open, append the
  new bar, recompute the latest target, and queue a new pending order if the
  target differs from the current position.
* Same ``state_snapshot()`` projection so the paper daemon can write the same
  ``state.json`` shape the watcher expects.
* Crash-safe ``checkpoint_state()`` / ``from_checkpoint()`` so a daemon
  restart resumes exactly where it left off.

The only thing that is intentionally different from the Numba session is the
*signal computation site* — instead of dispatching to a ``@njit`` kernel, we
call ``plugin.signal_fn(df)`` on the registered ``BlockStrategyPlugin`` and
read the boolean signals at the latest bar.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# Re-export the same dataclasses so callers can import either session class
# uniformly. We import (don't redefine) to keep the types identical.
from .live_paper_session import (
    PaperFill,
    PaperPosition,
    PendingOrder,
    SESSION_NAMESPACE,
)
from ..signal.numba_kernels import TARGET_KEEP, TARGET_LONG, TARGET_SHORT


__all__ = ["PythonLivePaperSession"]


class PythonLivePaperSession:
    """
    Paper trading session backed by a Python-engine (blocks) strategy.

    Usage::

        # User strategy module registers via cyqnt_trd.blocks.strategy.register(...)
        import importlib
        importlib.import_module("my_strategy_module")

        session = PythonLivePaperSession(
            strategy_id="my_strategy_id",
            symbol="BTCUSDT",
            config={"timeframe": "15m"},
            initial_capital=10_000.0,
        )

        for bar in incoming_bars:
            fill = session.tick(bar)
    """

    def __init__(
        self,
        *,
        strategy_id: str,
        symbol: str,
        config: Dict[str, Any],
        initial_capital: float = 10_000.0,
        fee_bps: float = 10.0,
        slippage_bps: float = 0.0,
        market_type: str = "futures",
        contract_multiplier: float = 1.0,
        max_bar_volume_fraction: float = 0.10,
    ) -> None:
        # Resolve the registered block-strategy plugin.
        # cyqnt_trd.blocks.strategy.register(...) appends to a global pending
        # list keyed by plugin_id; we look it up there *and* in the
        # _KNOWN_BLOCK_STRATEGY_IDS set (which is preserved even after
        # flush_pending_into has drained the pending list).
        from ...blocks.strategy import (  # type: ignore  # noqa: WPS433
            _KNOWN_BLOCK_STRATEGY_IDS,
            _PENDING_REGISTRATIONS,
            BlockStrategyPlugin,
            is_known_block_strategy,
        )

        plugin = self._resolve_plugin(
            strategy_id, _PENDING_REGISTRATIONS, _KNOWN_BLOCK_STRATEGY_IDS
        )
        if plugin is None:
            raise ValueError(
                "block strategy '%s' is not registered; call "
                "cyqnt_trd.blocks.strategy.register() first" % strategy_id
            )
        if not isinstance(plugin, BlockStrategyPlugin):
            raise TypeError(
                "registered plugin for '%s' is not a BlockStrategyPlugin"
                % strategy_id
            )

        self.strategy_id = strategy_id
        self.symbol = symbol
        self.config = dict(config)
        self.market_type = market_type
        self.contract_multiplier = contract_multiplier
        self.max_bar_volume_fraction = max_bar_volume_fraction

        self._plugin: BlockStrategyPlugin = plugin
        self._signal_fn: Callable = plugin.signal_fn

        # --- Financial state ---
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.position: Optional[PaperPosition] = None
        self.position_qty: float = 0.0  # signed: +long, -short

        # --- Bar history (growing) ---
        self._timestamps: List[int] = []
        self._open_times: List[int] = []
        self._opens: List[float] = []
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []
        self._volumes: List[float] = []
        self._quote_volumes: List[float] = []
        # Optional fields kept for parity with NumbaLivePaperSession's bar dict
        self._oi_change_bps: List[float] = []
        self._funding_rate_bps: List[float] = []
        self._long_liq_notional_usd: List[float] = []
        self._short_liq_notional_usd: List[float] = []

        # --- Signal & execution state ---
        self._pending_order: Optional[PendingOrder] = None
        self._last_target: int = 0
        self._target_history: List[int] = []

        # --- Trade log ---
        self.trade_log: List[PaperFill] = []
        self._tick_count: int = 0
        self._session_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Plugin resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_plugin(strategy_id, pending_list, known_ids):
        """Return the BlockStrategyPlugin instance for ``strategy_id`` or None.

        We look in two places:
          1. The pending registrations list (typical case — strategy module
             was imported but ``flush_pending_into`` hasn't been called yet).
          2. We don't have a global plugin registry for blocks, but if the
             strategy_id is in ``_KNOWN_BLOCK_STRATEGY_IDS`` it means it was
             at some point flushed into a SignalPluginRegistry — we can't
             recover the plugin object from that, so the caller must
             re-register before constructing the session.
        """
        for plugin, _factory in pending_list:
            if plugin.plugin_id == strategy_id:
                return plugin
        # Not in pending list. If it's a known id we still fail because we
        # have no way to reach into the registry from here without one being
        # provided. The error message in __init__ guides the caller.
        return None

    # ------------------------------------------------------------------
    # Public surface (mirrors NumbaLivePaperSession)
    # ------------------------------------------------------------------

    def tick(self, bar: Dict[str, Any]) -> Optional[PaperFill]:
        """Process one new confirmed bar (same contract as NumbaLivePaperSession.tick)."""
        fill = None

        # Step 1: Execute pending order at this bar's open
        if self._pending_order is not None:
            open_price = float(bar["open"])
            volume = float(bar.get("volume", 1e12))
            quote_volume = float(bar.get("quote_volume", open_price * volume))
            fill = self._execute_pending(
                open_price=open_price,
                bar_timestamp_ms=int(bar["timestamp"]),
                volume=volume,
                quote_volume=quote_volume,
            )
            self._pending_order = None

        # Step 2: Append bar to history
        self._append_bar(bar)
        self._tick_count += 1

        # Step 3: Compute latest target via the signal function
        target_at_this_bar, strength = self._compute_latest_target()
        self._target_history.append(target_at_this_bar)

        # Step 4: If target changed, create pending order for next bar
        if target_at_this_bar != TARGET_KEEP:
            current_direction = self._position_direction()
            target_direction = int(target_at_this_bar)
            if target_direction != current_direction:
                self._pending_order = PendingOrder(
                    target_position=target_direction,
                    signal_bar_index=len(self._timestamps) - 1,
                    signal_bar_timestamp_ms=int(bar["timestamp"]),
                    signal_strength=strength,
                )
                self._last_target = target_direction

        return fill

    def warm_up(self, bar: Dict[str, Any]) -> None:
        """Load one historical bar without trading (matches NumbaLivePaperSession)."""
        self._append_bar(bar)
        self._tick_count += 1

        target_at_this_bar, _strength = self._compute_latest_target()
        self._target_history.append(target_at_this_bar)

        # Warm-up never schedules an order
        self._pending_order = None
        self._last_target = self._position_direction()

    def has_pending_order(self) -> bool:
        return self._pending_order is not None

    @property
    def equity(self) -> float:
        if not self._closes:
            return self.cash
        last_price = self._closes[-1]
        return self.cash + self.position_qty * last_price * self.contract_multiplier

    @property
    def pnl_usd(self) -> float:
        return self.equity - self.initial_capital

    @property
    def pnl_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return self.pnl_usd / self.initial_capital

    def state_snapshot(self) -> Dict[str, Any]:
        """Read-only state projection for state.json writers."""
        position_dict = None
        if self.position is not None:
            position_dict = {
                "side": self.position.side,
                "entry_price": self.position.entry_price,
                "qty": self.position.quantity,
                "notional": self.position.entry_price * self.position.quantity,
            }

        latest_signal = None
        if self._pending_order is not None:
            side = (
                "long" if self._pending_order.target_position == 1 else
                "short" if self._pending_order.target_position == -1 else
                "close"
            )
            latest_signal = {
                "signal_id": str(uuid.uuid5(
                    SESSION_NAMESPACE,
                    "%s|%s|%d"
                    % (
                        self._session_id,
                        side,
                        self._pending_order.signal_bar_timestamp_ms,
                    ),
                )),
                "side": side,
                "strength": self._pending_order.signal_strength,
                "source": "blocks_python",
                "pending": True,
            }
        elif self.trade_log:
            last_trade = self.trade_log[-1]
            latest_signal = {
                "signal_id": last_trade.fill_id,
                "side": last_trade.side,
                "strength": 0.0,
                "source": "blocks_python",
                "pending": False,
            }

        return {
            "session_id": self._session_id,
            "strategy": self.strategy_id,
            "symbol": self.symbol,
            "market_type": self.market_type,
            "params": self.config,
            "initial_capital": self.initial_capital,
            "session_start_equity": self.initial_capital,
            "current_equity": round(self.equity, 4),
            "pnl_usd": round(self.pnl_usd, 4),
            "pnl": round(self.pnl_pct, 6),
            "position": position_dict,
            "latest_signal": latest_signal,
            "has_pending_order": self._pending_order is not None,
            "tick_count": self._tick_count,
            "bar_count": len(self._timestamps),
            "trade_count": len(self.trade_log),
            "trade_log": [self._fill_to_dict(f) for f in self.trade_log[-50:]],
            "last_bar_timestamp": self._timestamps[-1] if self._timestamps else None,
            "current_price": self._closes[-1] if self._closes else None,
            "last_update_ts": int(time.time()),
        }

    # ------------------------------------------------------------------
    # Checkpoint / restore
    # ------------------------------------------------------------------

    def checkpoint_state(self) -> Dict[str, Any]:
        """Serialize the full in-memory session for daemon restart resume."""
        return {
            "format_version": 1,
            "engine": "python",
            "session_id": self._session_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "config": dict(self.config),
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "market_type": self.market_type,
            "contract_multiplier": self.contract_multiplier,
            "max_bar_volume_fraction": self.max_bar_volume_fraction,
            "position_qty": self.position_qty,
            "position": self._position_to_dict(self.position),
            "pending_order": self._pending_to_dict(self._pending_order),
            "last_target": self._last_target,
            "tick_count": self._tick_count,
            "target_history": list(self._target_history),
            "bars": {
                "timestamps": list(self._timestamps),
                "open_times": list(self._open_times),
                "opens": list(self._opens),
                "highs": list(self._highs),
                "lows": list(self._lows),
                "closes": list(self._closes),
                "volumes": list(self._volumes),
                "quote_volumes": list(self._quote_volumes),
                "oi_change_bps": list(self._oi_change_bps),
                "funding_rate_bps": list(self._funding_rate_bps),
                "long_liq_notional_usd": list(self._long_liq_notional_usd),
                "short_liq_notional_usd": list(self._short_liq_notional_usd),
            },
            "trade_log": [self._fill_to_checkpoint_dict(f) for f in self.trade_log],
        }

    @classmethod
    def from_checkpoint(cls, payload: Dict[str, Any]) -> "PythonLivePaperSession":
        """Restore a session previously created by checkpoint_state().

        IMPORTANT: the user's strategy module must be re-imported (so
        ``cyqnt_trd.blocks.strategy.register(...)`` re-pending the plugin)
        BEFORE calling ``from_checkpoint``.
        """
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError("unsupported live paper checkpoint format")

        session = cls(
            strategy_id=str(payload["strategy_id"]),
            symbol=str(payload["symbol"]),
            config=dict(payload["config"]),
            initial_capital=float(payload["initial_capital"]),
            fee_bps=float(payload["fee_bps"]),
            slippage_bps=float(payload["slippage_bps"]),
            market_type=str(payload.get("market_type", "futures")),
            contract_multiplier=float(payload.get("contract_multiplier", 1.0)),
            max_bar_volume_fraction=float(
                payload.get("max_bar_volume_fraction", 0.10)
            ),
        )

        session._session_id = str(payload["session_id"])
        session.cash = float(payload["cash"])
        session.position_qty = float(payload.get("position_qty", 0.0))
        session.position = cls._position_from_dict(payload.get("position"))
        session._pending_order = cls._pending_from_dict(payload.get("pending_order"))
        session._last_target = int(payload.get("last_target", 0))
        session._tick_count = int(payload.get("tick_count", 0))
        session._target_history = [int(v) for v in payload.get("target_history", [])]

        bars = payload.get("bars", {})
        session._timestamps = [int(v) for v in bars.get("timestamps", [])]
        session._open_times = [int(v) for v in bars.get("open_times", [])]
        session._opens = [float(v) for v in bars.get("opens", [])]
        session._highs = [float(v) for v in bars.get("highs", [])]
        session._lows = [float(v) for v in bars.get("lows", [])]
        session._closes = [float(v) for v in bars.get("closes", [])]
        session._volumes = [float(v) for v in bars.get("volumes", [])]
        session._quote_volumes = [float(v) for v in bars.get("quote_volumes", [])]
        session._oi_change_bps = [float(v) for v in bars.get("oi_change_bps", [])]
        session._funding_rate_bps = [
            float(v) for v in bars.get("funding_rate_bps", [])
        ]
        session._long_liq_notional_usd = [
            float(v) for v in bars.get("long_liq_notional_usd", [])
        ]
        session._short_liq_notional_usd = [
            float(v) for v in bars.get("short_liq_notional_usd", [])
        ]
        session.trade_log = [
            cls._fill_from_checkpoint_dict(item)
            for item in payload.get("trade_log", [])
        ]
        return session

    # ------------------------------------------------------------------
    # Bar / signal helpers
    # ------------------------------------------------------------------

    def _append_bar(self, bar: Dict[str, Any]) -> None:
        self._timestamps.append(int(bar["timestamp"]))
        self._open_times.append(int(bar.get("open_time", bar["timestamp"])))
        self._opens.append(float(bar["open"]))
        self._highs.append(float(bar["high"]))
        self._lows.append(float(bar["low"]))
        self._closes.append(float(bar["close"]))
        self._volumes.append(float(bar.get("volume", 0.0)))
        self._quote_volumes.append(float(bar.get("quote_volume", 0.0)))
        self._oi_change_bps.append(float(bar.get("oi_change_bps", 0.0)))
        self._funding_rate_bps.append(float(bar.get("funding_rate_bps", 0.0)))
        self._long_liq_notional_usd.append(
            float(bar.get("long_liq_notional_usd", 0.0))
        )
        self._short_liq_notional_usd.append(
            float(bar.get("short_liq_notional_usd", 0.0))
        )

    def _build_dataframe(self) -> pd.DataFrame:
        """Build the OHLCV DataFrame the user signal_fn expects.

        Columns include the standard OHLCV plus ``timestamp`` and
        ``close_time`` (mirroring what ``cyqnt_trd.blocks.data.bars_to_df``
        produces from real ``Bar`` objects).
        """
        if not self._timestamps:
            return pd.DataFrame(
                columns=[
                    "open", "high", "low", "close", "volume", "quote_volume",
                    "open_time", "close_time", "timestamp",
                ]
            )
        df = pd.DataFrame(
            {
                "open": self._opens,
                "high": self._highs,
                "low": self._lows,
                "close": self._closes,
                "volume": self._volumes,
                "quote_volume": self._quote_volumes,
                "open_time": self._open_times,
                "close_time": self._timestamps,
                "timestamp": self._timestamps,
            }
        )
        return df

    def _compute_latest_target(self) -> Tuple[int, float]:
        """Run the user signal function over the full history and read the
        latest bar's signal.

        Returns ``(target, strength)``:
          * target=TARGET_LONG (1) if last bar's long signal is True
          * target=TARGET_SHORT (-1) if last bar's short signal is True
          * target=TARGET_KEEP (2) if neither (i.e. hold current position)
        """
        df = self._build_dataframe()
        if df.empty:
            return int(TARGET_KEEP), 0.0

        # Use the plugin's _call_signal_fn for consistent normalization
        # (reindex / fillna / dtype) — it's the same function the Python
        # backtest engine uses.
        long_s, short_s = self._plugin._call_signal_fn(df)

        last_idx = len(df) - 1
        try:
            last_long = bool(long_s.iloc[last_idx])
        except (IndexError, KeyError):
            last_long = False
        if short_s is not None:
            try:
                last_short = bool(short_s.iloc[last_idx])
            except (IndexError, KeyError):
                last_short = False
        else:
            last_short = False

        # Tie-break: prefer long over short (matches BlockStrategyPlugin._envelope_from_signals)
        if last_long and last_short:
            last_short = False

        if last_long:
            return int(TARGET_LONG), 1.0
        if last_short:
            return int(TARGET_SHORT), 1.0
        return int(TARGET_KEEP), 0.0

    # ------------------------------------------------------------------
    # Execution / position helpers (verbatim from NumbaLivePaperSession)
    # ------------------------------------------------------------------

    def _execute_pending(
        self,
        *,
        open_price: float,
        bar_timestamp_ms: int,
        volume: float,
        quote_volume: float,
    ) -> Optional[PaperFill]:
        """Execute pending order at the given bar's open price.

        Same execution model as NumbaLivePaperSession to keep paper-Python
        results comparable with paper-Numba runs.
        """
        pending = self._pending_order
        if pending is None:
            return None

        target_pos = pending.target_position
        multiplier = self.contract_multiplier

        equity_ref = self.cash + self.position_qty * open_price * multiplier
        if equity_ref < 0.0:
            equity_ref = 0.0

        if target_pos == 1:  # TARGET_LONG
            target_qty = equity_ref / (open_price * multiplier)
        elif target_pos == -1:  # TARGET_SHORT
            target_qty = -equity_ref / (open_price * multiplier)
        else:
            target_qty = 0.0

        delta_qty = target_qty - self.position_qty

        # Liquidity cap
        if self.max_bar_volume_fraction > 0.0 and open_price > 0.0:
            base_cap = max(volume, 0.0) * self.max_bar_volume_fraction
            if quote_volume > 0.0:
                quote_cap = (
                    quote_volume * self.max_bar_volume_fraction
                ) / (open_price * multiplier)
                if quote_cap < base_cap:
                    base_cap = quote_cap
            available = max(base_cap, 0.0)
            if delta_qty > available:
                delta_qty = available
            elif delta_qty < -available:
                delta_qty = -available

        if abs(delta_qty) < 1e-12:
            return None

        # Slippage
        slip = self.slippage_bps / 10_000.0
        if delta_qty > 0.0:
            fill_price = open_price * (1.0 + slip)
        else:
            fill_price = open_price * (1.0 - slip)

        # Fee
        fee = abs(delta_qty) * fill_price * multiplier * (self.fee_bps / 10_000.0)

        # Apply
        old_position_qty = self.position_qty
        self.cash -= delta_qty * fill_price * multiplier + fee
        self.position_qty += delta_qty
        if abs(self.position_qty) < 1e-12:
            self.position_qty = 0.0

        action = self._action_label(old_qty=old_position_qty, new_qty=self.position_qty)

        if self.position_qty > 0:
            self.position = PaperPosition(
                side="long",
                entry_price=fill_price,
                quantity=abs(self.position_qty),
                opened_at_ms=bar_timestamp_ms,
            )
        elif self.position_qty < 0:
            self.position = PaperPosition(
                side="short",
                entry_price=fill_price,
                quantity=abs(self.position_qty),
                opened_at_ms=bar_timestamp_ms,
            )
        else:
            self.position = None

        fill_id = str(uuid.uuid5(
            SESSION_NAMESPACE,
            "%s|%d|%s|%.8f"
            % (self._session_id, bar_timestamp_ms, action, fill_price),
        ))
        side = "buy" if delta_qty > 0 else "sell"

        paper_fill = PaperFill(
            fill_id=fill_id,
            timestamp_ms=bar_timestamp_ms,
            side=side,
            price=fill_price,
            quantity=abs(delta_qty),
            fee=fee,
            signal_bar_timestamp_ms=pending.signal_bar_timestamp_ms,
            action=action,
        )
        self.trade_log.append(paper_fill)
        return paper_fill

    def _position_direction(self) -> int:
        if self.position_qty > 1e-12:
            return 1
        if self.position_qty < -1e-12:
            return -1
        return 0

    @staticmethod
    def _action_label(*, old_qty: float, new_qty: float) -> str:
        old_dir = 1 if old_qty > 1e-12 else (-1 if old_qty < -1e-12 else 0)
        new_dir = 1 if new_qty > 1e-12 else (-1 if new_qty < -1e-12 else 0)

        if old_dir == 0 and new_dir == 1:
            return "open_long"
        if old_dir == 0 and new_dir == -1:
            return "open_short"
        if old_dir == 1 and new_dir == 0:
            return "close_long"
        if old_dir == -1 and new_dir == 0:
            return "close_short"
        if old_dir == -1 and new_dir == 1:
            return "flip_to_long"
        if old_dir == 1 and new_dir == -1:
            return "flip_to_short"
        return "rebalance"

    # ------------------------------------------------------------------
    # Dict (de)serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_to_dict(f: PaperFill) -> Dict[str, Any]:
        return {
            "fill_id": f.fill_id,
            "timestamp_ms": f.timestamp_ms,
            "side": f.side,
            "price": round(f.price, 6),
            "quantity": round(f.quantity, 8),
            "fee": round(f.fee, 6),
            "action": f.action,
            "signal_bar_timestamp_ms": f.signal_bar_timestamp_ms,
        }

    @staticmethod
    def _fill_to_checkpoint_dict(f: PaperFill) -> Dict[str, Any]:
        return {
            "fill_id": f.fill_id,
            "timestamp_ms": f.timestamp_ms,
            "side": f.side,
            "price": f.price,
            "quantity": f.quantity,
            "fee": f.fee,
            "signal_bar_timestamp_ms": f.signal_bar_timestamp_ms,
            "action": f.action,
        }

    @staticmethod
    def _fill_from_checkpoint_dict(item: Dict[str, Any]) -> PaperFill:
        return PaperFill(
            fill_id=str(item["fill_id"]),
            timestamp_ms=int(item["timestamp_ms"]),
            side=str(item["side"]),
            price=float(item["price"]),
            quantity=float(item["quantity"]),
            fee=float(item["fee"]),
            signal_bar_timestamp_ms=int(item["signal_bar_timestamp_ms"]),
            action=str(item["action"]),
        )

    @staticmethod
    def _position_to_dict(pos: Optional[PaperPosition]) -> Optional[Dict[str, Any]]:
        if pos is None:
            return None
        return {
            "side": pos.side,
            "entry_price": pos.entry_price,
            "quantity": pos.quantity,
            "opened_at_ms": pos.opened_at_ms,
        }

    @staticmethod
    def _position_from_dict(d: Optional[Dict[str, Any]]) -> Optional[PaperPosition]:
        if not d:
            return None
        return PaperPosition(
            side=str(d["side"]),
            entry_price=float(d["entry_price"]),
            quantity=float(d["quantity"]),
            opened_at_ms=int(d["opened_at_ms"]),
        )

    @staticmethod
    def _pending_to_dict(p: Optional[PendingOrder]) -> Optional[Dict[str, Any]]:
        if p is None:
            return None
        return {
            "target_position": p.target_position,
            "signal_bar_index": p.signal_bar_index,
            "signal_bar_timestamp_ms": p.signal_bar_timestamp_ms,
            "signal_strength": p.signal_strength,
        }

    @staticmethod
    def _pending_from_dict(d: Optional[Dict[str, Any]]) -> Optional[PendingOrder]:
        if not d:
            return None
        return PendingOrder(
            target_position=int(d["target_position"]),
            signal_bar_index=int(d["signal_bar_index"]),
            signal_bar_timestamp_ms=int(d["signal_bar_timestamp_ms"]),
            signal_strength=float(d["signal_strength"]),
        )
