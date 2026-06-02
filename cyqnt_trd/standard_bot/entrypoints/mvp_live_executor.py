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
import sys
from pathlib import Path

from ..execution.cli_executor import BinanceCliExecutor


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
        "--poll-interval", type=int, default=5,
        help="Seconds between trades.jsonl polls (default: 5)",
    )
    return parser


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
        poll_sec=args.poll_interval,
    )

    if args.dry_run:
        print("[live-executor] ⚠️  DRY-RUN mode: commands printed but not executed")
    else:
        print(f"[live-executor] ⚠️  LIVE mode: real orders on {args.symbol}")
        print(f"[live-executor] ⚠️  max_notional={args.max_notional} USDT")
        print(f"[live-executor] ⚠️  Emergency stop: touch {state_dir}/EMERGENCY_STOP")

    try:
        executor.watch_trades(state_dir=state_dir)
    except KeyboardInterrupt:
        print("\n[live-executor] Ctrl+C — shutting down")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
