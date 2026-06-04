"""
BinanceCliExecutor — trades.jsonl watcher → binance-cli live orders.

This is a strategy-agnostic execution adapter that bridges the paper daemon
(any blocks strategy registered via ``cyqnt_trd.blocks.strategy.register``)
to real Binance Futures orders placed via ``binance-cli``.

Architecture::

    PythonLivePaperSession (any strategy)
        │
        ▼ writes trades.jsonl
    BinanceCliExecutor.watch_trades()
        │
        ▼ reads action from each fill
    binance-cli futures-usds new-order ...

Supported actions (produced by PythonLivePaperSession._action_label):
    - open_long       → BUY MARKET
    - open_short      → SELL MARKET
    - close_long      → SELL MARKET reduce-only
    - close_short     → BUY MARKET reduce-only
    - flip_to_long    → close_short + open_long (two-step)
    - flip_to_short   → close_long + open_short (two-step)

Features:
    - Retry with exponential backoff (configurable)
    - Position reconciliation before each order
    - Audit trail (executions.jsonl)
    - Kill switch (EMERGENCY_STOP file)
    - Heartbeat logging
    - Strategy-agnostic: works with any strategy that uses PythonLivePaperSession
"""

from __future__ import annotations

import json
import math
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_POLL_SEC = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_SEC = 2.0
DEFAULT_HEARTBEAT_INTERVAL = 300  # 5 minutes
EXECUTOR_STATE_FILENAME = "live_executor_state.json"

ALL_KNOWN_ACTIONS = frozenset({
    "open_long", "open_short",
    "close_long", "close_short",
    "flip_to_long", "flip_to_short",
    "rebalance",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, data: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ── BinanceCliExecutor ────────────────────────────────────────────────────────

class BinanceCliExecutor:
    """
    Strategy-agnostic live trade executor.

    Watches a paper daemon's ``trades.jsonl`` and translates each fill's
    ``action`` into real Binance Futures orders via ``binance-cli``.

    Usage::

        executor = BinanceCliExecutor(
            symbol="BTCUSDT",
            max_notional=200.0,
            notional_fraction=0.95,
            dry_run=True,
        )
        executor.watch_trades(state_dir=Path("./watcher/MY_STRATEGY_BTCUSDT_1h"))
    """

    def __init__(
        self,
        *,
        symbol: str = "BTCUSDT",
        max_notional: float = 200.0,
        notional_fraction: float = 0.95,
        dry_run: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_sec: float = DEFAULT_RETRY_BASE_SEC,
        poll_sec: int = DEFAULT_POLL_SEC,
        heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
    ) -> None:
        self.symbol = symbol.upper()
        self.max_notional = max_notional
        self.notional_fraction = notional_fraction
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.retry_base_sec = retry_base_sec
        self.poll_sec = poll_sec
        self.heartbeat_interval = heartbeat_interval
        self._step_size_cache: Dict[str, float] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def watch_trades(self, state_dir: Path) -> None:
        """
        Main loop: tail trades.jsonl, process new fills, exit when daemon stops.

        This method blocks until:
        - The daemon writes ``status: stopped`` to state.json
        - EMERGENCY_STOP file is created in state_dir
        - KeyboardInterrupt
        """
        trades_path = state_dir / "trades.jsonl"
        executions_path = state_dir / "executions.jsonl"
        kill_switch_path = state_dir / "EMERGENCY_STOP"
        executor_state_path = state_dir / EXECUTOR_STATE_FILENAME
        seen_ids: set = set()
        last_heartbeat = time.time()
        executor_state = self._load_executor_state(executor_state_path)

        self._log("═══════════════════════════════════════════════")
        self._log(f"watching {trades_path}")
        self._log(f"symbol={self.symbol}  max_notional={self.max_notional}")
        self._log(f"dry_run={self.dry_run}")
        self._log(f"executions log → {executions_path}")
        self._log(f"kill switch → {kill_switch_path}")
        self._log("═══════════════════════════════════════════════")

        # Load existing fills to skip (don't re-execute on restart)
        if trades_path.exists():
            for line in trades_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    fill = json.loads(line)
                    seen_ids.add(fill.get("fill_id", ""))
                except Exception:
                    pass
            self._log(f"loaded {len(seen_ids)} existing fills (will not re-execute)")

        while True:
            # Kill switch
            if kill_switch_path.exists():
                self._log("⚠️  EMERGENCY STOP activated!")
                self._emergency_stop(executions_path)
                break

            time.sleep(self.poll_sec)

            # Heartbeat
            now = time.time()
            if now - last_heartbeat > self.heartbeat_interval:
                self._log(f"heartbeat alive at {_now_iso()}")
                last_heartbeat = now

            # Check daemon status
            state_path = state_dir / "state.json"
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text())
                    if state.get("status") in ("stopped", "risk_triggered"):
                        self._log("daemon stopped, exiting")
                        break
                except Exception:
                    pass

            if not trades_path.exists():
                continue

            # If we are mid-transition, finish reconciling before consuming
            # additional paper fills. This keeps live orders sequential and
            # avoids compounding drift after transient API failures.
            if self._has_pending_transition(executor_state):
                executor_state = self._advance_pending_transition(
                    executor_state, executions_path
                )
                self._save_executor_state(executor_state_path, executor_state)
                if self._has_pending_transition(executor_state):
                    continue

            # Process new fills
            for line in trades_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    fill = json.loads(line)
                except Exception:
                    continue

                fill_id = fill.get("fill_id", "")
                if fill_id in seen_ids:
                    continue

                seen_ids.add(fill_id)
                action = fill.get("action", "")
                ts = fill.get("ts", 0)
                self._log(f"──── NEW FILL ────")
                self._log(f"fill_id={fill_id[:12]}  action={action}  ts={ts}")

                try:
                    executor_state = self.handle_fill(
                        fill, executions_path, executor_state
                    )
                    self._save_executor_state(executor_state_path, executor_state)
                    if self._has_pending_transition(executor_state):
                        break
                except Exception as e:
                    self._log(f"UNHANDLED ERROR: {e}")
                    traceback.print_exc()
                    _append_jsonl(executions_path, {
                        "ts": _now_iso(),
                        "fill_id": fill_id,
                        "action": action,
                        "status": "unhandled_error",
                        "error": str(e),
                    })

    def handle_fill(
        self,
        fill: Dict[str, Any],
        executions_path: Path,
        executor_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a single paper fill and update the desired live position.

        Rather than assuming each paper fill maps 1:1 to a fixed list of live
        orders, we persist a target-direction transition and let the reconcile
        loop walk the real account toward that target. This makes transient
        failures (502, timeout, ReduceOnly rejection after drift) recoverable.
        """
        action = fill.get("action", "")

        if action == "rebalance":
            self._log("REBALANCE: not implemented (should not occur in position-flip strategies)")
            return executor_state

        if action not in ALL_KNOWN_ACTIONS:
            self._log(f"UNKNOWN action={action!r}, skip")
            return executor_state

        target_direction = self._target_direction_for_action(action)
        if target_direction is None:
            self._log(f"UNKNOWN target direction for action={action!r}, skip")
            return executor_state

        transition = {
            "origin_fill_id": fill.get("fill_id", ""),
            "source_action": action,
            "target_direction": target_direction,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "attempt_count": 0,
            "last_attempt_action": None,
            "last_attempt_at": None,
            "last_error": None,
            "last_observed_direction": self._get_position_direction(),
        }
        executor_state["pending_transition"] = transition
        self._log(
            "QUEUE transition:"
            f" action={action} target={target_direction}"
            f" actual={transition['last_observed_direction']}"
        )
        return self._advance_pending_transition(
            executor_state,
            executions_path,
            use_cached_position=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Order placement
    # ──────────────────────────────────────────────────────────────────────────

    def _place_single_order(
        self, action: str, executions_path: Path
    ) -> Dict[str, Any]:
        """
        Execute a single atomic action via binance-cli.

        Returns execution record dict with ``status`` field.
        """
        is_close = action in ("close_long", "close_short")
        side = "BUY" if action in ("open_long", "close_short") else "SELL"

        price = self._get_current_price()
        step = self._get_step_size()

        record: Dict[str, Any] = {
            "ts": _now_iso(),
            "action": action,
            "symbol": self.symbol,
            "side": side,
            "price_at_decision": price,
            "dry_run": self.dry_run,
        }

        if is_close:
            pos = self._get_position()
            if pos is None and not self.dry_run:
                record["status"] = "skipped"
                record["reason"] = "no_position_to_close"
                self._log(f"SKIP: {action} but no position found")
                _append_jsonl(executions_path, record)
                return record

            qty = 0.001 if self.dry_run else abs(float(pos["positionAmt"]))  # type: ignore
            qty = self._round_step(qty, step)
            if qty <= 0:
                record["status"] = "skipped"
                record["reason"] = "qty_zero"
                _append_jsonl(executions_path, record)
                return record

            cmd_args = [
                "futures-usds", "new-order",
                "--symbol", self.symbol,
                "--side", side,
                "--type", "MARKET",
                "--quantity", str(qty),
                "--reduce-only", "true",
            ]
            record["quantity"] = qty
            self._log(f"CLOSE {action}  {side} {qty} {self.symbol} @ ~{price:.2f}")

        else:
            balance = self._get_usdt_balance()
            if balance <= 0 and not self.dry_run:
                record["status"] = "skipped"
                record["reason"] = "zero_balance"
                self._log(f"SKIP: USDT balance=0 for {action}")
                _append_jsonl(executions_path, record)
                return record

            notional = min(balance * self.notional_fraction, self.max_notional)
            qty = self._round_step(notional / price, step)
            if qty <= 0:
                record["status"] = "skipped"
                record["reason"] = "qty_rounds_to_zero"
                _append_jsonl(executions_path, record)
                return record

            cmd_args = [
                "futures-usds", "new-order",
                "--symbol", self.symbol,
                "--side", side,
                "--type", "MARKET",
                "--quantity", str(qty),
            ]
            record["quantity"] = qty
            record["notional_target"] = notional
            self._log(
                f"OPEN {action}  {side} {qty} {self.symbol} "
                f"@ ~{price:.2f}  notional≈{qty*price:.2f}"
            )

        # Execute with retry
        try:
            result = self._cli_with_retry(*cmd_args)
            if not self.dry_run and result:
                record["order_id"] = result.get("orderId")
                record["status"] = result.get("status", "unknown")
                record["avg_price"] = result.get("avgPrice")
                self._log(
                    f"ORDER OK  orderId={result.get('orderId')} "
                    f"status={result.get('status')}"
                )
            else:
                record["status"] = "dry_run_ok"
        except RuntimeError as e:
            record["status"] = "failed"
            record["error"] = str(e)
            self._log(f"ERROR: {e}")

        _append_jsonl(executions_path, record)
        return record

    def _advance_pending_transition(
        self,
        executor_state: Dict[str, Any],
        executions_path: Path,
        *,
        use_cached_position: bool = False,
    ) -> Dict[str, Any]:
        """
        Move the real account one or two safe steps toward the pending target.

        The executor never blindly assumes the previous order succeeded. Each
        pass re-reads the real account position and chooses the next atomic
        action needed to reach ``target_direction``.
        """
        transition = executor_state.get("pending_transition")
        if not transition:
            return executor_state

        for _ in range(2):
            transition = executor_state.get("pending_transition")
            if not transition:
                return executor_state

            target_direction = transition["target_direction"]
            if use_cached_position and transition.get("last_observed_direction") in {
                "long",
                "short",
                "flat",
            }:
                actual_direction = str(transition["last_observed_direction"])
            else:
                actual_direction = self._get_position_direction()
            transition["updated_at"] = _now_iso()
            transition["last_observed_direction"] = actual_direction

            if actual_direction == target_direction:
                self._log(
                    "RECONCILE OK:"
                    f" target={target_direction} actual={actual_direction}"
                    f" source={transition.get('source_action')}"
                )
                executor_state["pending_transition"] = None
                return executor_state

            next_action = self._next_action_for(actual_direction, target_direction)
            if next_action is None:
                transition["last_error"] = (
                    f"unsupported transition actual={actual_direction} "
                    f"target={target_direction}"
                )
                executor_state["pending_transition"] = transition
                return executor_state

            transition["attempt_count"] = int(transition.get("attempt_count", 0)) + 1
            transition["last_attempt_action"] = next_action
            transition["last_attempt_at"] = _now_iso()
            self._log(
                "RECONCILE STEP:"
                f" actual={actual_direction} target={target_direction}"
                f" -> {next_action}"
            )
            result = self._place_single_order(next_action, executions_path)
            status = result.get("status")

            if status in ("failed", "skipped"):
                transition["last_error"] = (
                    result.get("error")
                    or result.get("reason")
                    or str(status)
                )
                executor_state["pending_transition"] = transition
                return executor_state

            transition["last_error"] = None
            if self.dry_run:
                actual_after = target_direction
            else:
                time.sleep(1)
                actual_after = self._get_position_direction()

            transition["last_observed_direction"] = actual_after
            executor_state["pending_transition"] = transition

            if actual_after == target_direction:
                self._log(
                    "RECONCILE OK:"
                    f" target={target_direction} actual={actual_after}"
                    f" source={transition.get('source_action')}"
                )
                executor_state["pending_transition"] = None
                return executor_state

            if actual_after == actual_direction:
                transition["last_error"] = (
                    f"{next_action} acknowledged but actual_position stayed "
                    f"{actual_after}"
                )
                executor_state["pending_transition"] = transition
                return executor_state

        return executor_state

    # ──────────────────────────────────────────────────────────────────────────
    # binance-cli interface
    # ──────────────────────────────────────────────────────────────────────────

    def _cli(self, *args: str) -> dict | list:
        """Execute binance-cli and return parsed JSON."""
        cmd = ["binance-cli"] + list(args)
        if self.dry_run:
            print(f"[DRY-RUN] {' '.join(cmd)}")
            return {}
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            raise RuntimeError(
                f"binance-cli error (rc={result.returncode}): {stderr or stdout}"
            )
        if not result.stdout.strip():
            return {}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"binance-cli non-JSON output: {result.stdout[:200]}"
            ) from e

    def _cli_with_retry(self, *args: str) -> dict | list:
        """_cli with exponential backoff retry."""
        last_error: Optional[RuntimeError] = None
        for attempt in range(self.max_retries):
            try:
                return self._cli(*args)
            except RuntimeError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = self.retry_base_sec * (2 ** attempt)
                    self._log(
                        f"retry {attempt+1}/{self.max_retries} in {wait:.1f}s: {e}"
                    )
                    time.sleep(wait)
        raise last_error  # type: ignore

    # ──────────────────────────────────────────────────────────────────────────
    # Market data queries
    # ──────────────────────────────────────────────────────────────────────────

    def _get_usdt_balance(self) -> float:
        if self.dry_run:
            return 10000.0
        data = self._cli_with_retry(
            "futures-usds", "futures-account-balance-v3"
        )
        for item in data:  # type: ignore
            if item.get("asset") == "USDT":
                return float(item.get("availableBalance", 0))
        return 0.0

    def _get_current_price(self) -> float:
        if self.dry_run:
            return 100000.0
        data = self._cli_with_retry(
            "futures-usds", "symbol-price-ticker", "--symbol", self.symbol
        )
        return float(data["price"])  # type: ignore

    def _get_step_size(self) -> float:
        if self.symbol in self._step_size_cache:
            return self._step_size_cache[self.symbol]
        if self.dry_run:
            self._step_size_cache[self.symbol] = 0.001
            return 0.001
        data = self._cli_with_retry("futures-usds", "exchange-information")
        for s in data.get("symbols", []):  # type: ignore
            if s["symbol"] == self.symbol:
                for f in s.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        step = float(f["stepSize"])
                        self._step_size_cache[self.symbol] = step
                        return step
        self._step_size_cache[self.symbol] = 0.001
        return 0.001

    def _get_position(self) -> Optional[Dict[str, Any]]:
        """Return current position dict, or None if flat."""
        if self.dry_run:
            return None
        data = self._cli_with_retry(
            "futures-usds", "position-information-v3", "--symbol", self.symbol
        )
        for p in data:  # type: ignore
            if abs(float(p.get("positionAmt", 0))) > 0:
                return p
        return None

    def _has_position_direction(self, direction: str) -> bool:
        """Check if current position matches expected direction (reconciliation)."""
        if self.dry_run:
            return False
        pos = self._get_position()
        if pos is None:
            return False
        amt = float(pos.get("positionAmt", 0))
        if direction == "long":
            return amt > 0
        if direction == "short":
            return amt < 0
        return False

    def _get_position_direction(self) -> str:
        if self.dry_run:
            return "flat"
        pos = self._get_position()
        if pos is None:
            return "flat"
        amt = float(pos.get("positionAmt", 0))
        if amt > 0:
            return "long"
        if amt < 0:
            return "short"
        return "flat"

    @staticmethod
    def _target_direction_for_action(action: str) -> Optional[str]:
        mapping = {
            "open_long": "long",
            "flip_to_long": "long",
            "open_short": "short",
            "flip_to_short": "short",
            "close_long": "flat",
            "close_short": "flat",
        }
        return mapping.get(action)

    @staticmethod
    def _next_action_for(actual_direction: str, target_direction: str) -> Optional[str]:
        transitions = {
            ("flat", "long"): "open_long",
            ("flat", "short"): "open_short",
            ("long", "flat"): "close_long",
            ("short", "flat"): "close_short",
            ("long", "short"): "close_long",
            ("short", "long"): "close_short",
        }
        return transitions.get((actual_direction, target_direction))

    @staticmethod
    def _default_executor_state() -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "pending_transition": None,
        }

    def _load_executor_state(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return self._default_executor_state()
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return self._default_executor_state()
        state = self._default_executor_state()
        state.update(payload if isinstance(payload, dict) else {})
        return state

    def _save_executor_state(self, path: Path, state: Dict[str, Any]) -> None:
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    @staticmethod
    def _has_pending_transition(state: Dict[str, Any]) -> bool:
        return bool(state.get("pending_transition"))

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _round_step(qty: float, step: float) -> float:
        """Round quantity down to step_size precision."""
        if step <= 0:
            return qty
        decimals = max(0, -int(math.floor(math.log10(step))))
        return round(math.floor(qty / step) * step, decimals)

    def _emergency_stop(self, executions_path: Path) -> None:
        """Cancel all open orders and exit."""
        self._log("Attempting to cancel all open orders...")
        try:
            self._cli(
                "futures-usds", "cancel-all-open-orders",
                "--symbol", self.symbol,
            )
        except Exception as e:
            self._log(f"cancel failed: {e}")
        _append_jsonl(executions_path, {
            "ts": _now_iso(),
            "event": "emergency_stop",
            "symbol": self.symbol,
        })
        self._log("Exiting due to EMERGENCY_STOP.")

    def _log(self, msg: str) -> None:
        print(f"[live-executor] {msg}", flush=True)
