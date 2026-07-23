"""
signal_executor.py — trades.jsonl watcher → binance-cli 真實下單
=================================================================

使用 cyqnt_trd 框架的 BinanceCliExecutor class。
此 script 是框架 mvp_live_executor 的等價 standalone wrapper。

支援所有 action type：
  - open_long / open_short / close_long / close_short
  - flip_to_long / flip_to_short（拆成 close + open 兩步）

安全機制：
  - Position reconciliation（每次下單前檢查真實倉位）
  - Retry with exponential backoff（最多 3 次）
  - Audit trail（executions.jsonl）
  - Kill switch（EMERGENCY_STOP 檔案）
  - Heartbeat log

用法：
  # Dry-run（驗證，不真實下單）
  python scripts/signal_executor.py \
    --state-dir ./watcher/MA_CROSS_V1_ETHUSDT_1h \
    --symbol ETHUSDT \
    --max-notional 200 \
    --dry-run

  # 真實下單
  python scripts/signal_executor.py \
    --state-dir ./watcher/MA_CROSS_V1_ETHUSDT_1h \
    --symbol ETHUSDT \
    --max-notional 200

等價於：
  python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor \
    --state-dir ./watcher/MA_CROSS_V1_ETHUSDT_1h \
    --symbol ETHUSDT --max-notional 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cyqnt_trd.standard_bot.execution.cli_executor import BinanceCliExecutor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Signal executor: paper fills → binance-cli orders (uses BinanceCliExecutor)"
    )
    parser.add_argument("--state-dir", required=True,
                        help="Paper daemon state dir (含 trades.jsonl)")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--notional-fraction", type=float, default=0.95)
    parser.add_argument("--max-notional", type=float, default=200.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    if not state_dir.exists():
        print(f"[signal_executor] ERROR: state-dir not found: {state_dir}")
        print(f"[signal_executor] Start paper daemon first.")
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
        print("[signal_executor] ⚠️  DRY-RUN mode")
    else:
        print(f"[signal_executor] ⚠️  LIVE: real orders on {args.symbol}")
        print(f"[signal_executor] ⚠️  Emergency stop: touch {state_dir}/EMERGENCY_STOP")

    try:
        executor.watch_trades(state_dir=state_dir)
    except KeyboardInterrupt:
        print("\n[signal_executor] Ctrl+C — shutting down")

    return 0


if __name__ == "__main__":
    sys.exit(main())
