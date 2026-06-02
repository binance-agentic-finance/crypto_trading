"""
run_strategy_v2.py — backtest / paper / live 統一入口（本地可執行版）
=====================================================================

修正內容（相較 v1）：
  1. 移除 /root/.openclaw/workspace hardcoded 路徑
  2. 路徑改為可配置（--workspace 或 STRATEGY_WORKSPACE 環境變數）
  3. Live mode 改用 signal_executor_v2（支援 flip + reconciliation）
  4. 新增 --live-backend 選項（cli / api）為未來擴展準備
  5. 自動偵測 python3.11 或 python3

用法：
  # 回測
  python scripts/run_strategy_v2.py --mode backtest --data-path data/BTCUSDT_1h.parquet

  # Paper trade
  python scripts/run_strategy_v2.py --mode paper

  # Live trade（dry-run 先驗證）
  python scripts/run_strategy_v2.py --mode live --dry-run

  # Live trade（真實下單，50 USDT 上限）
  python scripts/run_strategy_v2.py --mode live --max-notional 50
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _find_python() -> str:
    """找到可用的 Python 3.11+"""
    for candidate in ["python3.11", "python3.12", "python3"]:
        if shutil.which(candidate):
            return candidate
    return sys.executable


def _resolve_workspace(args) -> Path:
    """解析 workspace 路徑"""
    workspace = getattr(args, "workspace", None)
    if workspace:
        return Path(workspace).resolve()
    env_ws = os.environ.get("STRATEGY_WORKSPACE")
    if env_ws:
        return Path(env_ws).resolve()
    # 預設：script 所在目錄的上一層
    return Path(__file__).resolve().parent.parent


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(args) -> int:
    workspace = _resolve_workspace(args)
    python = _find_python()
    data_path = args.data_path

    # 如果 data_path 是相對路徑，基於 workspace 解析
    if not Path(data_path).is_absolute():
        data_path = str(workspace / data_path)

    cmd = [
        python, "-m",
        "cyqnt_trd.standard_bot.entrypoints.mvp_backtest",
        "--engine",          "python",
        "--strategy",        args.strategy,
        "--strategy-module", args.strategy_module,
        "--symbol",          args.symbol,
        "--interval",        args.interval,
        "--data-path",       data_path,
        "--fee-bps",         str(args.fee_bps),
        "--slippage-bps",    str(args.slippage_bps),
        "--initial-capital", str(args.initial_capital),
        "--execution-model", "next_bar_open",
    ]

    env = {**os.environ, "PYTHONPATH": f"{workspace}:{os.environ.get('PYTHONPATH', '')}"}
    print(f"[run_strategy] mode=backtest")
    print(f"[run_strategy] workspace={workspace}")
    print(f"[run_strategy] {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


# ── Paper trade ───────────────────────────────────────────────────────────────

def run_paper(args) -> int:
    workspace = _resolve_workspace(args)
    python = _find_python()
    state_dir = _resolve_state_dir(args, workspace)
    state_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python, "-m",
        "cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon",
        "--symbol",          args.symbol,
        "--interval",        args.interval,
        "--strategy",        args.strategy,
        "--strategy-module", args.strategy_module,
        "--engine",          "python",
        "--state-dir",       str(state_dir),
        "--poll-interval",   str(args.poll_interval),
        "--warm-up-bars",    str(args.warm_up_bars),
        "--initial-capital", str(args.initial_capital),
        "--fee-bps",         str(args.fee_bps),
        "--slippage-bps",    str(args.slippage_bps),
        "--market-type",     args.market_type,
    ]

    env = {**os.environ, "PYTHONPATH": f"{workspace}:{os.environ.get('PYTHONPATH', '')}"}
    print(f"[run_strategy] mode=paper")
    print(f"[run_strategy] workspace={workspace}")
    print(f"[run_strategy] state_dir={state_dir}")
    print(f"[run_strategy] {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


# ── Live trade ────────────────────────────────────────────────────────────────

def run_live(args) -> int:
    """
    啟動兩個 process：
      1. Paper daemon（訊號來源，和 paper mode 完全相同）
      2. Signal executor v2（偵測 trades.jsonl → binance-cli 真實下單）

    關鍵設計：
      - paper daemon 的 make_signals() 和 backtest/paper 完全相同 → 訊號一致
      - executor 只看 action 方向，sizing 獨立計算 → 不會漏掉 position flip
    """
    workspace = _resolve_workspace(args)
    python = _find_python()
    state_dir = _resolve_state_dir(args, workspace)
    state_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = workspace / "scripts"

    env = {**os.environ, "PYTHONPATH": f"{workspace}:{os.environ.get('PYTHONPATH', '')}"}

    # ── Process 1: Paper Daemon ──────────────────────────────────────────────
    daemon_cmd = [
        python, "-m",
        "cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon",
        "--symbol",          args.symbol,
        "--interval",        args.interval,
        "--strategy",        args.strategy,
        "--strategy-module", args.strategy_module,
        "--engine",          "python",
        "--state-dir",       str(state_dir),
        "--poll-interval",   str(args.poll_interval),
        "--warm-up-bars",    str(args.warm_up_bars),
        "--initial-capital", str(args.initial_capital),
        "--fee-bps",         str(args.fee_bps),
        "--slippage-bps",    str(args.slippage_bps),
        "--market-type",     args.market_type,
    ]

    # ── Process 2: Live Executor (framework-level) ───────────────────────────
    executor_cmd = [
        python, "-m",
        "cyqnt_trd.standard_bot.entrypoints.mvp_live_executor",
        "--state-dir",         str(state_dir),
        "--symbol",            args.symbol,
        "--notional-fraction", str(args.notional_fraction),
        "--max-notional",      str(args.max_notional),
    ]
    if args.dry_run:
        executor_cmd.append("--dry-run")

    print(f"[run_strategy] ═══════════════════════════════════════════════")
    print(f"[run_strategy] mode=live  dry_run={args.dry_run}")
    print(f"[run_strategy] workspace={workspace}")
    print(f"[run_strategy] state_dir={state_dir}")
    print(f"[run_strategy] ═══════════════════════════════════════════════")
    print(f"[run_strategy] daemon:   {' '.join(daemon_cmd[:6])}...")
    print(f"[run_strategy] executor: {' '.join(executor_cmd[:4])}...")

    if args.dry_run:
        print(f"[run_strategy] ⚠️  DRY-RUN: executor 只印出指令不下單")
    else:
        print(f"[run_strategy] ⚠️  LIVE: 真實下單！max_notional={args.max_notional} USDT")
        print(f"[run_strategy] ⚠️  要緊急停止：touch {state_dir}/EMERGENCY_STOP")

    daemon_proc = subprocess.Popen(daemon_cmd, env=env)
    # 等 daemon 啟動（確保 state.json 存在）
    time.sleep(3)
    executor_proc = subprocess.Popen(executor_cmd, env=env)

    try:
        while True:
            if daemon_proc.poll() is not None:
                print("[run_strategy] daemon exited, stopping executor")
                executor_proc.terminate()
                break
            if executor_proc.poll() is not None:
                print("[run_strategy] executor exited, stopping daemon")
                daemon_proc.terminate()
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n[run_strategy] Ctrl+C — stopping both processes")
        daemon_proc.terminate()
        executor_proc.terminate()

    daemon_proc.wait()
    executor_proc.wait()
    return 0


# ── 輔助 ──────────────────────────────────────────────────────────────────────

def _resolve_state_dir(args, workspace: Path) -> Path:
    if args.state_dir:
        return Path(args.state_dir).resolve()
    # 自動生成：watcher/<STRATEGY>_<SYMBOL>_<INTERVAL>
    run_id = f"{args.strategy.upper()}_{args.symbol}_{args.interval}"
    return workspace / "watcher" / run_id


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="策略統一入口 — backtest / paper / live"
    )

    # 共用參數
    parser.add_argument("--mode", choices=["backtest", "paper", "live"], default="paper")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--strategy", default="ma_cross_v1")
    parser.add_argument("--strategy-module", default="strategies.ma_cross_v1")
    parser.add_argument("--workspace", default=None,
                        help="策略 workspace 根目錄（預設用 STRATEGY_WORKSPACE 環境變數或 script 上層）")
    parser.add_argument("--state-dir", default=None,
                        help="State 目錄（paper/live mode 用）")

    # 模擬參數
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--market-type", default="futures")
    parser.add_argument("--warm-up-bars", type=int, default=80)
    parser.add_argument("--poll-interval", type=int, default=3570,
                        help="Paper daemon poll 間隔（秒）")

    # Backtest 專用
    parser.add_argument("--data-path", default="data/BTCUSDT_1h.parquet")

    # Live 專用
    parser.add_argument("--max-notional", type=float, default=200.0)
    parser.add_argument("--notional-fraction", type=float, default=0.95)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live-backend", choices=["cli", "api"], default="cli",
                        help="cli=binance-cli subprocess; api=direct REST (未實裝)")

    args = parser.parse_args()

    if args.mode == "backtest":
        return run_backtest(args)
    elif args.mode == "paper":
        return run_paper(args)
    else:
        if args.live_backend == "api":
            print("[run_strategy] ERROR: --live-backend api 尚未實裝")
            return 1
        return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
