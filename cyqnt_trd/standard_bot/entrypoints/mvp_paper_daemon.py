"""
Long-running paper trade daemon using Numba kernels.

This daemon runs continuously in the OpenClaw workspace, maintaining strategy
state in memory.  It writes ``state.json`` atomically so that external watchers
can read status without triggering a re-run.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  mvp_paper_daemon (this process, long-running)          │
    │  ├─ NumbaLivePaperSession (in-memory state)             │
    │  ├─ Polls Binance REST for new bars at interval         │
    │  ├─ Runs kernel on each new bar → signals               │
    │  ├─ Executes paper fills (next-bar-open model)          │
    │  └─ Writes state.json atomically (tmp + rename)         │
    └─────────────────────────────────────────────────────────┘
                          ↕ state.json (read-only for watcher)
    ┌─────────────────────────────────────────────────────────┐
    │  watcher_template.py                                    │
    │  └─ Reads state.json → risk check → jarvis notify      │
    └─────────────────────────────────────────────────────────┘

Usage:
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \\
        --symbol BTCUSDT --interval 1h --strategy ema_rsi_cross \\
        --strategy-module mvp_strategy_lab.external_strategies.ema_rsi_cross \\
        --extra-params '{"fast_window":2,"slow_window":5,"rsi_period":3}' \\
        --state-dir /path/to/watcher/run_id/ \\
        --poll-interval 3600 \\
        --warm-up-bars 120

Graceful stop:
    Set state.json["status"] = "stopped" → daemon exits on next poll.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..data import timeframe_to_ms
from ..data.adapters import BinanceRestMarketDataAdapter
from ..simulation.live_paper_session import NumbaLivePaperSession, PaperFill
from ..simulation.python_live_paper_session import PythonLivePaperSession


# ── Helpers ────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically: write to temp file then rename."""
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".state_"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_state(path: Path) -> Dict[str, Any]:
    """Load state.json if it exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    """Append one JSON record. Used for trade/event journals."""
    with open(path, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records, skipping damaged partial lines."""
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ── Market data fetching ───────────────────────────────────────────────────

class BarFetcher:
    """Fetches confirmed kline bars from Binance REST API."""

    def __init__(self, *, symbol: str, interval: str, market_type: str = "spot") -> None:
        self.symbol = symbol
        self.interval = interval
        self.market_type = market_type
        self.interval_ms = timeframe_to_ms(interval)
        self._adapter = BinanceRestMarketDataAdapter(market_type=market_type)

    def fetch_bars(
        self,
        limit: Optional[int] = 120,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch latest confirmed bars from Binance.

        Returns list of bar dicts with keys:
        timestamp, open_time, open, high, low, close, volume, quote_volume
        """
        from ..core import MarketQuery, TimeRange

        query = MarketQuery(
            instruments=[self.symbol],
            timeframes=[self.interval],
            time_range=TimeRange(
                start_ts=start_ts,
                end_ts=end_ts,
                tail_bars=limit,
            ),
        )
        bundle = self._adapter.fetch_market(query)
        key = bundle.key(self.symbol, self.interval)
        raw_bars = bundle.bars.get(key, [])

        # Only confirmed bars
        confirmed = [b for b in raw_bars if b.confirmed]
        confirmed.sort(key=lambda b: b.timestamp)

        result = []
        for bar in confirmed:
            result.append({
                "timestamp": int(bar.timestamp),
                "open_time": int(bar.extras.get("open_time", bar.timestamp)),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "quote_volume": float(
                    bar.quote_volume if bar.quote_volume is not None
                    else bar.close * bar.volume
                ),
                "oi_change_bps": float(bar.extras.get("oi_change_bps", 0.0) or 0.0),
                "funding_rate_bps": float(bar.extras.get("funding_rate_bps", 0.0) or 0.0),
                "long_liq_notional_usd": float(bar.extras.get("long_liq_notional_usd", 0.0) or 0.0),
                "short_liq_notional_usd": float(bar.extras.get("short_liq_notional_usd", 0.0) or 0.0),
            })
        return result

    def fetch_new_bars_since(
        self,
        last_timestamp_ms: int,
        limit: int = 1000,
        max_batches: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Fetch every confirmed bar newer than last_timestamp_ms.

        This is intentionally paginated. A daemon that sleeps too long, gets
        rate-limited, or is restarted after downtime must catch up every missed
        bar; otherwise signal transitions and next-bar-open fills are lost.
        """
        new_bars: List[Dict[str, Any]] = []
        seen_timestamps: set[int] = set()
        next_open_ts = int(last_timestamp_ms) + 1

        for _ in range(max_batches):
            batch = self.fetch_bars(limit=limit, start_ts=next_open_ts)
            batch = [
                bar for bar in batch
                if bar["timestamp"] > last_timestamp_ms and bar["timestamp"] not in seen_timestamps
            ]
            if not batch:
                break

            batch.sort(key=lambda bar: bar["timestamp"])
            new_bars.extend(batch)
            seen_timestamps.update(int(bar["timestamp"]) for bar in batch)

            if len(batch) < int(limit):
                break

            last_open_time = int(batch[-1].get("open_time", batch[-1]["timestamp"] - self.interval_ms + 1))
            next_open_ts = last_open_time + self.interval_ms

        return new_bars


# ── Daemon main logic ──────────────────────────────────────────────────────

class PaperDaemon:
    """
    Long-running paper trade daemon.

    Lifecycle:
    1. __init__: validate args, import strategy module
    2. start(): fetch history → warm-up session → main loop
    3. main loop: sleep → fetch new bars → tick session → write state
    4. Stop: external writes status="stopped" to state.json
    """

    def __init__(
        self,
        *,
        symbol: str,
        interval: str,
        strategy: str,
        strategy_module: Optional[str],
        extra_params: Optional[Dict[str, Any]],
        state_dir: str,
        poll_interval: int,
        warm_up_bars: int,
        initial_capital: float,
        fee_bps: float,
        slippage_bps: float,
        market_type: str,
        jarvis_user_id: str = "",
        jarvis_thread_id: str = "",
        session_end_at: str = "",
        engine: str = "numba",
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.strategy = strategy
        self.strategy_module = strategy_module
        self.extra_params = extra_params or {}
        self.state_dir = Path(state_dir)
        self.poll_interval = poll_interval
        self.warm_up_bars = warm_up_bars
        self.initial_capital = initial_capital
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.market_type = market_type
        self.jarvis_user_id = jarvis_user_id
        self.jarvis_thread_id = jarvis_thread_id
        self.session_end_at = session_end_at
        if engine not in ("numba", "python"):
            raise ValueError("engine must be 'numba' or 'python', got %r" % engine)
        self.engine = engine

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        self.pid_path = self.state_dir / "pid"
        self.log_path = self.state_dir / "daemon.log"
        self.trades_path = self.state_dir / "trades.jsonl"
        self.events_path = self.state_dir / "events.jsonl"
        self.checkpoint_path = self.state_dir / "session_checkpoint.json"

        self._session: Optional[Any] = None  # Numba or Python session
        self._fetcher: Optional[BarFetcher] = None
        self._last_bar_ts: int = 0
        self._running = True
        self._poll_count = 0
        self._persisted_trade_log: List[Dict[str, Any]] = []
        self._persisted_trade_ids: set[str] = set()

    def start(self) -> int:
        """Main entry point. Returns exit code."""
        # Write PID
        self.pid_path.write_text(str(os.getpid()))
        self._started_at = _now_iso()
        self._load_persisted_trades()

        # Import external strategy module if specified
        if self.strategy_module:
            import importlib
            self._log("importing strategy module: %s" % self.strategy_module)
            importlib.import_module(self.strategy_module)

        # Build config for the session
        config = {
            "instrument_id": self.symbol,
            "timeframe": self.interval,
        }
        config.update(self.extra_params)

        # Create session — branch on engine type
        if self.engine == "python":
            self._log("using Python engine (cyqnt_trd.blocks)")
            self._session = PythonLivePaperSession(
                strategy_id=self.strategy,
                symbol=self.symbol,
                config=config,
                initial_capital=self.initial_capital,
                fee_bps=self.fee_bps,
                slippage_bps=self.slippage_bps,
                market_type=self.market_type,
            )
        else:
            self._log("using Numba engine")
            self._session = NumbaLivePaperSession(
                strategy_id=self.strategy,
                symbol=self.symbol,
                config=config,
                initial_capital=self.initial_capital,
                fee_bps=self.fee_bps,
                slippage_bps=self.slippage_bps,
                market_type=self.market_type,
            )

        # Create fetcher
        self._fetcher = BarFetcher(
            symbol=self.symbol,
            interval=self.interval,
            market_type=self.market_type,
        )

        restored = self._restore_session_from_checkpoint()
        if restored:
            self._log(
                "restored session from checkpoint. equity=%.2f bars=%d trades=%d last_ts=%d"
                % (
                    self._session.equity,
                    self._session.state_snapshot()["bar_count"],
                    len(self._persisted_trade_log),
                    self._last_bar_ts,
                )
            )
        else:
            # Fetch initial history and warm up
            self._log("fetching %d warm-up bars..." % self.warm_up_bars)
            bars = self._fetcher.fetch_bars(limit=self.warm_up_bars)
            if not bars:
                self._log("ERROR: no bars fetched, cannot start")
                return 1

            self._log("warming up session with %d bars..." % len(bars))
            for bar in bars:
                self._session.warm_up(bar)
            self._last_bar_ts = bars[-1]["timestamp"]
            self._write_checkpoint()

            self._log(
                "warm-up complete. equity=%.2f bars=%d trades=%d last_ts=%d"
                % (
                    self._session.equity,
                    len(bars),
                    len(self._persisted_trade_log),
                    self._last_bar_ts,
                )
            )

        # Write initial state
        self._write_state()

        # Install signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Main loop
        self._log(
            "daemon started. symbol=%s strategy=%s poll=%ds"
            % (self.symbol, self.strategy, self.poll_interval)
        )

        while self._running:
            time.sleep(self.poll_interval)
            try:
                should_continue = self._poll()
            except Exception as e:
                self._log("ERROR in poll: %s\n%s" % (e, traceback.format_exc()))
                should_continue = True  # keep running despite errors

            if not should_continue:
                break

        self._log("daemon stopped. final equity=%.2f trades=%d" % (
            self._session.equity, len(self._persisted_trade_log)
        ))
        self._write_state(status="stopped")
        return 0

    def _poll(self) -> bool:
        """
        One poll cycle.

        Returns True to continue, False to stop.
        """
        self._poll_count += 1

        # Check if external stop requested via state.json
        if self._check_stop_requested():
            self._log("stop requested via state.json")
            return False

        # Fetch new bars since last known timestamp
        new_bars = self._fetcher.fetch_new_bars_since(
            self._last_bar_ts, limit=1000
        )

        if not new_bars:
            # No new bars yet — just update timestamp in state
            self._write_state()
            return True

        # Process each new bar through the session
        fills_this_poll: List[PaperFill] = []
        for bar in new_bars:
            fill = self._session.tick(bar)
            self._last_bar_ts = bar["timestamp"]
            self._write_checkpoint()
            if fill is not None:
                fills_this_poll.append(fill)
                self._record_fill(fill)
                self._log(
                    "FILL: %s price=%.4f qty=%.6f fee=%.4f action=%s"
                    % (fill.side, fill.price, fill.quantity, fill.fee, fill.action)
                )

        # Log status
        self._log(
            "poll #%d: +%d bars, %d fills, equity=%.2f, pos=%s"
            % (
                self._poll_count,
                len(new_bars),
                len(fills_this_poll),
                self._session.equity,
                self._session.position.side if self._session.position else "flat",
            )
        )

        # Write updated state
        self._write_state()
        return True

    def _check_stop_requested(self) -> bool:
        """Check if state.json has been externally modified to request stop."""
        try:
            state = _load_state(self.state_path)
            return state.get("status") in ("stopped", "risk_triggered")
        except Exception:
            return False

    def _restore_session_from_checkpoint(self) -> bool:
        """Resume a previous long-running session exactly where it left off."""
        if not self.checkpoint_path.exists():
            return False

        try:
            current_state = _load_state(self.state_path)
            if current_state.get("status") in ("stopped", "risk_triggered"):
                return False
            payload = _load_state(self.checkpoint_path)
            checkpoint = payload.get("session", payload)
            # Branch on engine for from_checkpoint
            if self.engine == "python":
                restored = PythonLivePaperSession.from_checkpoint(checkpoint)
            else:
                restored = NumbaLivePaperSession.from_checkpoint(checkpoint)
        except Exception as exc:
            self._log("WARN: failed to restore checkpoint: %s" % exc)
            return False

        if restored.symbol.upper() != self.symbol or restored.strategy_id != self.strategy:
            self._log(
                "WARN: checkpoint mismatch symbol/strategy (%s/%s), ignoring"
                % (restored.symbol, restored.strategy_id)
            )
            return False

        self._session = restored
        self._last_bar_ts = int(
            payload.get("last_bar_ts")
            or restored.state_snapshot().get("last_bar_timestamp")
            or 0
        )
        self._poll_count = int(payload.get("poll_count", self._poll_count))
        self._reconcile_persisted_trades_from_session()
        return True

    def _write_checkpoint(self) -> None:
        """Persist exact session runtime state for crash-safe resume."""
        if self._session is None:
            return

        payload = {
            "saved_at": _now_iso(),
            "last_bar_ts": self._last_bar_ts,
            "poll_count": self._poll_count,
            "session": self._session.checkpoint_state(),
        }
        _atomic_write_json(self.checkpoint_path, payload)

    def _load_persisted_trades(self) -> None:
        """Restore append-only trade history across daemon/watcher restarts."""
        trades: List[Dict[str, Any]] = []
        for row in _load_jsonl(self.trades_path):
            trade = row.get("trade") if isinstance(row.get("trade"), dict) else row
            if isinstance(trade, dict):
                trades.append(trade)

        if not trades:
            existing = _load_state(self.state_path)
            existing_trades = existing.get("trade_log", [])
            if isinstance(existing_trades, list):
                trades.extend(t for t in existing_trades if isinstance(t, dict))

        for trade in trades:
            fill_id = str(trade.get("fill_id") or "")
            if not fill_id or fill_id in self._persisted_trade_ids:
                continue
            self._persisted_trade_ids.add(fill_id)
            self._persisted_trade_log.append(trade)

        if self._persisted_trade_log and not self.trades_path.exists():
            for trade in self._persisted_trade_log:
                _append_jsonl(self.trades_path, trade)

    def _reconcile_persisted_trades_from_session(self) -> None:
        """
        Ensure the append-only journal contains every fill already in session memory.

        The checkpoint is the authoritative resume state. If a crash happens
        after checkpoint save but before journal append, this backfills the
        missing journal rows on startup.
        """
        if self._session is None:
            return

        for fill in self._session.trade_log:
            trade = self._fill_to_trade(fill)
            fill_id = str(trade["fill_id"])
            if fill_id in self._persisted_trade_ids:
                continue
            self._persisted_trade_ids.add(fill_id)
            self._persisted_trade_log.append(trade)
            _append_jsonl(self.trades_path, trade)

    def _fill_to_trade(self, fill: PaperFill) -> Dict[str, Any]:
        """Convert a session fill into watcher-compatible persistent JSON."""
        return {
            "fill_id": fill.fill_id,
            "ts": fill.timestamp_ms // 1000,
            "side": fill.side,
            "price": fill.price,
            "qty": fill.quantity,
            "quantity": fill.quantity,
            "fee": fill.fee,
            "action": fill.action,
            "signal_bar_ts": fill.signal_bar_timestamp_ms,
            "live": False,
        }

    def _record_fill(self, fill: PaperFill) -> None:
        """Persist one new fill exactly once."""
        trade = self._fill_to_trade(fill)
        fill_id = str(trade["fill_id"])
        if fill_id in self._persisted_trade_ids:
            return

        self._persisted_trade_ids.add(fill_id)
        self._persisted_trade_log.append(trade)
        _append_jsonl(self.trades_path, trade)
        _append_jsonl(self.events_path, {
            "ts": int(time.time()),
            "type": "paper_fill",
            "trade": trade,
        })

    def _write_state(self, status: str = "running") -> None:
        """Write state.json atomically, merging session snapshot with daemon metadata."""
        snapshot = self._session.state_snapshot()

        # Merge with watcher-compatible format
        state = {
            "run_id": "%s_%s_%s" % (
                self.symbol,
                self.strategy,
                self._session._session_id[:8],
            ),
            "symbol": self.symbol,
            "market_type": self.market_type,
            "mode": "paper",
            "signal_source": "python_daemon" if self.engine == "python" else "numba_daemon",
            "strategy": self.strategy,
            "params": self.extra_params,
            "interval": self.interval,
            "initial_capital": self.initial_capital,
            "session_start_equity": self.initial_capital,
            "current_equity": snapshot["current_equity"],
            "stop_equity": self.initial_capital * 0.9,  # default 10% drawdown stop
            "pnl_usd": snapshot["pnl_usd"],
            "pnl": snapshot["pnl"],
            "trade_log": self._persisted_trade_log,
            "position": snapshot["position"],
            "status": status,
            "latest_signal": snapshot["latest_signal"],
            "has_pending_order": snapshot["has_pending_order"],
            "poll_interval_sec": self.poll_interval,
            "poll_count": self._poll_count,
            "tick_count": snapshot["tick_count"],
            "bar_count": snapshot["bar_count"],
            "trade_count": len(self._persisted_trade_log),
            "last_bar_timestamp": snapshot["last_bar_timestamp"],
            "current_price": snapshot["current_price"],
            "last_update_ts": int(time.time()),
            "last_update_at": _now_iso(),
            "daemon_pid": os.getpid(),
            "daemon_started_at": getattr(self, "_started_at", _now_iso()),
        }

        # Preserve external fields (stop_equity, jarvis_*, session_end_at, etc.)
        try:
            existing = _load_state(self.state_path)
            if existing.get("run_id"):
                state["run_id"] = existing["run_id"]
            for key in ("stop_equity", "jarvis_user_id", "jarvis_thread_id",
                        "session_end_at", "risk_triggered", "risk_reason"):
                if key in existing and key not in state:
                    state[key] = existing[key]
                elif key in existing and key == "stop_equity":
                    state[key] = existing[key]  # respect external override
        except Exception:
            pass

        # Inject daemon-level jarvis / session_end_at (CLI args take precedence)
        if self.jarvis_user_id:
            state["jarvis_user_id"] = self.jarvis_user_id
        if self.jarvis_thread_id:
            state["jarvis_thread_id"] = self.jarvis_thread_id
        if self.session_end_at:
            state["session_end_at"] = self.session_end_at

        _atomic_write_json(self.state_path, state)

    def _handle_signal(self, signum, frame):
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        self._log("received signal %d, shutting down..." % signum)
        self._running = False

    def _log(self, msg: str) -> None:
        """Log to stdout and optionally to file."""
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = "[%s] %s" % (ts, msg)
        print(line, flush=True)
        try:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Long-running paper trade daemon using Numba kernels or Python (blocks) strategies"
    )
    parser.add_argument("--symbol", required=True, help="Trading pair (e.g. BTCUSDT)")
    parser.add_argument("--interval", default="1h", help="Kline interval (default: 1h)")
    parser.add_argument("--strategy", required=True, help="Strategy ID (builtin or registered)")
    parser.add_argument(
        "--strategy-module", default=None,
        help="Python module to import for external strategy registration"
    )
    parser.add_argument(
        "--engine", choices=["numba", "python"], default="numba",
        help="Execution engine: 'numba' for @njit kernels (default), 'python' for cyqnt_trd.blocks strategies"
    )
    parser.add_argument(
        "--extra-params", default=None,
        help="JSON dict of strategy config params"
    )
    parser.add_argument(
        "--state-dir", required=True,
        help="Directory for state.json and daemon files (e.g. watcher/<run_id>/)"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=3600,
        help="Seconds between polls for new bars (default: 3600 for 1h candles)"
    )
    parser.add_argument(
        "--warm-up-bars", type=int, default=120,
        help="Number of historical bars to load on startup (default: 120)"
    )
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument(
        "--market-type", choices=["spot", "futures"], default="futures"
    )
    parser.add_argument(
        "--jarvis-user-id", default="",
        help="Jarvis user ID for trade notifications"
    )
    parser.add_argument(
        "--jarvis-thread-id", default="",
        help="Jarvis thread ID for trade notifications"
    )
    parser.add_argument(
        "--session-end-at", default="",
        help="ISO8601 UTC time to stop the session (e.g. 2026-05-15T12:00:00Z)"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    extra_params = None
    if args.extra_params:
        extra_params = json.loads(args.extra_params)

    daemon = PaperDaemon(
        symbol=args.symbol,
        interval=args.interval,
        strategy=args.strategy,
        strategy_module=args.strategy_module,
        extra_params=extra_params,
        state_dir=args.state_dir,
        poll_interval=args.poll_interval,
        warm_up_bars=args.warm_up_bars,
        initial_capital=args.initial_capital,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        market_type=args.market_type,
        jarvis_user_id=args.jarvis_user_id,
        jarvis_thread_id=args.jarvis_thread_id,
        session_end_at=args.session_end_at,
        engine=args.engine,
    )

    return daemon.start()


if __name__ == "__main__":
    raise SystemExit(main())
