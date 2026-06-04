"""
Live trade executor — watches paper daemon trades.jsonl and places real orders.

This is the live-trade counterpart of ``mvp_paper_daemon``. While the daemon
generates signals and paper fills using any blocks strategy, this executor
translates those fills into real Binance Futures orders via ``binance-cli``.

Strategy-agnostic: works with ANY strategy registered via
``cyqnt_trd.blocks.strategy.register()``.

Architecture::

    mvp_paper_daemon (signal source)
        │ writes trades.jsonl
        ▼
    mvp_live_executor (this process)
        │ reads action from each fill
        ▼
    binance-cli futures-usds new-order ...

Usage::

    # Start paper daemon (signal source) — same as paper trade mode
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \\
        --strategy my_strategy --strategy-module strategies.my_strategy \\
        --symbol BTCUSDT --interval 1h --engine python \\
        --state-dir ./watcher/MY_STRATEGY_BTCUSDT_1h

    # Start live executor (real orders) — in a separate terminal
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor \\
        --state-dir ./watcher/MY_STRATEGY_BTCUSDT_1h \\
        --symbol BTCUSDT --max-notional 200

    # Dry-run first to validate
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor \\
        --state-dir ./watcher/MY_STRATEGY_BTCUSDT_1h \\
        --symbol BTCUSDT --max-notional 200 --dry-run

Emergency stop::

    touch ./watcher/MY_STRATEGY_BTCUSDT_1h/EMERGENCY_STOP
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from ..execution.cli_executor import (
    EXECUTOR_STATE_FILENAME,
    BinanceCliExecutor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Live trade executor: watches paper daemon trades.jsonl "
            "and places real Binance Futures orders via binance-cli. "
            "Strategy-agnostic — works with any blocks strategy."
        )
    )
    parser.add_argument(
        "--state-dir", required=True,
        help="Paper daemon state directory (contains trades.jsonl)",
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT",
        help="Trading pair (default: BTCUSDT)",
    )
    parser.add_argument(
        "--max-notional", type=float, default=200.0,
        help="Maximum notional per order in USDT (default: 200)",
    )
    parser.add_argument(
        "--notional-fraction", type=float, default=0.95,
        help="Fraction of available balance to use per trade (default: 0.95)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print binance-cli commands without executing",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retry attempts per order (default: 3)",
    )
    parser.add_argument(
        "--retry-base-sec", type=float, default=2.0,
        help="Base seconds for exponential retry backoff (default: 2.0)",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=5,
        help="Seconds between trades.jsonl polls (default: 5)",
    )
    parser.add_argument(
        "--heartbeat-interval", type=int, default=300,
        help="Seconds between heartbeat logs while running (default: 300)",
    )
    parser.add_argument(
        "--reconcile-only", action="store_true",
        help=(
            "Do not consume new paper fills. Only resume and reconcile any "
            "pending live transition already stored in live_executor_state.json."
        ),
    )
    parser.add_argument(
        "--max-reconcile-cycles", type=int, default=50,
        help="Maximum reconcile cycles in --reconcile-only mode (default: 50)",
    )
    return parser


def _executor_state_path(state_dir: Path) -> Path:
    return state_dir / EXECUTOR_STATE_FILENAME


def _load_pending_transition(state_dir: Path) -> dict | None:
    state_path = _executor_state_path(state_dir)
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    pending = payload.get("pending_transition")
    return pending if isinstance(pending, dict) else None


def _format_pending_transition(pending: dict | None) -> str:
    if not pending:
        return "none"
    return (
        f"target={pending.get('target_direction')} "
        f"source={pending.get('source_action')} "
        f"attempts={pending.get('attempt_count', 0)} "
        f"last_action={pending.get('last_attempt_action')} "
        f"last_error={pending.get('last_error') or '-'}"
    )


def _run_preflight(state_dir: Path, executor: BinanceCliExecutor) -> dict:
    trades_path = state_dir / "trades.jsonl"
    state_path = state_dir / "state.json"

    if not trades_path.exists():
        raise FileNotFoundError(
            f"missing trades.jsonl in state-dir: {trades_path}. Start paper daemon first."
        )
    if not state_path.exists():
        raise FileNotFoundError(
            f"missing state.json in state-dir: {state_path}. Start paper daemon first."
        )
    if shutil.which("binance-cli") is None:
        raise RuntimeError("binance-cli not found on PATH")

    balance = executor._get_usdt_balance()
    price = executor._get_current_price()
    step = executor._get_step_size()
    notional = min(balance * executor.notional_fraction, executor.max_notional)
    qty = executor._round_step(notional / price, step)
    actual_direction = executor._get_position_direction()
    pending = _load_pending_transition(state_dir)

    if qty <= 0:
        raise RuntimeError(
            "configured max_notional rounds to zero quantity after step-size "
            f"constraints for {executor.symbol}; increase max_notional or use another symbol"
        )

    return {
        "balance": balance,
        "price": price,
        "step_size": step,
        "preview_notional": notional,
        "preview_qty": qty,
        "actual_direction": actual_direction,
        "pending_transition": pending,
    }


def _reconcile_only(
    state_dir: Path,
    executor: BinanceCliExecutor,
    *,
    max_cycles: int,
    sleep_sec: int,
) -> int:
    executions_path = state_dir / "executions.jsonl"
    executor_state_path = _executor_state_path(state_dir)
    executor_state = executor._load_executor_state(executor_state_path)

    if not executor._has_pending_transition(executor_state):
        print("[live-executor] reconcile-only: no pending transition found")
        return 0

    print(
        "[live-executor] reconcile-only: resuming "
        f"{_format_pending_transition(executor_state.get('pending_transition'))}"
    )
    for cycle in range(1, max_cycles + 1):
        executor_state = executor._advance_pending_transition(
            executor_state, executions_path
        )
        executor._save_executor_state(executor_state_path, executor_state)
        if not executor._has_pending_transition(executor_state):
            print(f"[live-executor] reconcile-only: completed in {cycle} cycle(s)")
            return 0
        time.sleep(sleep_sec)

    pending = executor_state.get("pending_transition")
    print(
        "[live-executor] reconcile-only: still pending after "
        f"{max_cycles} cycles -> {_format_pending_transition(pending)}"
    )
    return 2


def main() -> int:
    args = build_parser().parse_args()

    state_dir = Path(args.state_dir)
    if not state_dir.exists():
        print(f"[live-executor] ERROR: state-dir does not exist: {state_dir}")
        print(f"[live-executor] Start the paper daemon first to create it.")
        return 1

    executor = BinanceCliExecutor(
        symbol=args.symbol,
        max_notional=args.max_notional,
        notional_fraction=args.notional_fraction,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        retry_base_sec=args.retry_base_sec,
        poll_sec=args.poll_interval,
        heartbeat_interval=args.heartbeat_interval,
    )

    try:
        summary = _run_preflight(state_dir, executor)
    except Exception as e:
        print(f"[live-executor] PRECHECK FAILED: {e}")
        return 1

    print(
        "[live-executor] preflight ok:"
        f" balance≈{summary['balance']:.4f} USDT"
        f" price≈{summary['price']:.4f}"
        f" step={summary['step_size']}"
        f" preview_qty={summary['preview_qty']}"
        f" preview_notional≈{summary['preview_notional']:.4f}"
        f" actual_position={summary['actual_direction']}"
    )
    print(
        "[live-executor] pending transition:"
        f" {_format_pending_transition(summary['pending_transition'])}"
    )

    if args.dry_run:
        print("[live-executor] ⚠️  DRY-RUN mode: commands printed but not executed")
    else:
        print(f"[live-executor] ⚠️  LIVE mode: real orders on {args.symbol}")
        print(f"[live-executor] ⚠️  max_notional={args.max_notional} USDT")
        print(f"[live-executor] ⚠️  Emergency stop: touch {state_dir}/EMERGENCY_STOP")

    if args.reconcile_only:
        return _reconcile_only(
            state_dir,
            executor,
            max_cycles=args.max_reconcile_cycles,
            sleep_sec=args.poll_interval,
        )

    try:
        executor.watch_trades(state_dir=state_dir)
    except KeyboardInterrupt:
        print("\n[live-executor] Ctrl+C — shutting down")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
