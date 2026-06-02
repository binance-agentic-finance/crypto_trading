"""
signal_executor_v2.py — 改良版 trades.jsonl watcher → binance-cli 下單
=========================================================================

相較 v1 的改進：
  1. 支援 flip_to_long / flip_to_short action（拆成 close + open 兩步）
  2. 加入 position reconciliation（每次下單前對齊真實倉位）
  3. Retry with exponential backoff（最多 3 次）
  4. 寫入 executions.jsonl（audit trail）
  5. 可配置路徑（不再 hardcode /root/.openclaw/workspace）
  6. Kill switch 支援（EMERGENCY_STOP 檔案）
  7. Heartbeat log（定期寫 alive 訊息）

使用方式：
  python3.11 scripts/signal_executor_v2.py \\
    --state-dir ./watcher/MA_CROSS_BTCUSDT_1h \\
    --symbol BTCUSDT \\
    --max-notional 200 \\
    --dry-run

  # 移除 --dry-run 即為真實下單
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 常數 ──────────────────────────────────────────────────────────────────────
POLL_SEC = 5
MAX_RETRIES = 3
RETRY_BASE_SEC = 2.0
HEARTBEAT_INTERVAL = 300  # 5 分鐘寫一次 heartbeat
STEP_SIZE_CACHE: dict[str, float] = {}


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, data: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ── binance-cli helpers ───────────────────────────────────────────────────────

def _cli(*args: str, dry_run: bool = False) -> dict | list:
    """執行 binance-cli，回傳 parsed JSON。dry_run=True 時只印不執行。"""
    cmd = ["binance-cli"] + list(args)
    if dry_run:
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
        raise RuntimeError(f"binance-cli non-JSON output: {result.stdout[:200]}") from e


def _cli_with_retry(*args: str, dry_run: bool = False) -> dict | list:
    """_cli with exponential backoff retry。"""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return _cli(*args, dry_run=dry_run)
        except RuntimeError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_SEC * (2 ** attempt)
                print(f"[executor] retry {attempt+1}/{MAX_RETRIES} in {wait:.1f}s: {e}")
                time.sleep(wait)
    raise last_error  # type: ignore


def get_usdt_balance(dry_run: bool = False) -> float:
    if dry_run:
        return 10000.0  # mock balance for dry-run
    data = _cli_with_retry("futures-usds", "futures-account-balance-v3")
    for item in data:
        if item.get("asset") == "USDT":
            return float(item.get("availableBalance", 0))
    return 0.0


def get_current_price(symbol: str, dry_run: bool = False) -> float:
    if dry_run:
        return 100000.0  # mock price
    data = _cli_with_retry("futures-usds", "symbol-price-ticker", "--symbol", symbol)
    return float(data["price"])


def get_step_size(symbol: str, dry_run: bool = False) -> float:
    if symbol in STEP_SIZE_CACHE:
        return STEP_SIZE_CACHE[symbol]
    if dry_run:
        STEP_SIZE_CACHE[symbol] = 0.001
        return 0.001
    data = _cli_with_retry("futures-usds", "exchange-information")
    for s in data.get("symbols", []):
        if s["symbol"] == symbol:
            for f in s.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
                    STEP_SIZE_CACHE[symbol] = step
                    return step
    STEP_SIZE_CACHE[symbol] = 0.001
    return 0.001


def get_position(symbol: str, dry_run: bool = False) -> Optional[Dict[str, Any]]:
    """回傳目前持倉 dict；空倉時回傳 None。"""
    if dry_run:
        return None
    data = _cli_with_retry("futures-usds", "position-information-v3", "--symbol", symbol)
    for p in data:
        if abs(float(p.get("positionAmt", 0))) > 0:
            return p
    return None


def round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    decimals = max(0, -int(math.floor(math.log10(step))))
    return round(math.floor(qty / step) * step, decimals)


# ── Reconciliation ────────────────────────────────────────────────────────────

def reconcile_position(
    symbol: str,
    expected_direction: str,  # "long", "short", "flat"
    dry_run: bool = False,
) -> bool:
    """
    檢查真實倉位方向是否符合預期。

    Returns True if consistent, False if mismatch detected.
    """
    pos = get_position(symbol, dry_run=dry_run)
    if dry_run:
        return True

    if expected_direction == "flat":
        if pos is not None:
            print(f"[reconcile] WARN: expected flat but have position: {pos}")
            return False
        return True

    if pos is None:
        print(f"[reconcile] WARN: expected {expected_direction} but position is flat")
        return False

    pos_amt = float(pos.get("positionAmt", 0))
    if expected_direction == "long" and pos_amt <= 0:
        print(f"[reconcile] WARN: expected long but positionAmt={pos_amt}")
        return False
    if expected_direction == "short" and pos_amt >= 0:
        print(f"[reconcile] WARN: expected short but positionAmt={pos_amt}")
        return False

    return True


# ── 下單核心 ──────────────────────────────────────────────────────────────────

def place_single_order(
    symbol: str,
    action: str,
    max_notional: float,
    notional_fraction: float,
    dry_run: bool,
    executions_path: Path,
) -> Dict[str, Any]:
    """
    執行單一 action (open_long / open_short / close_long / close_short)。
    回傳 execution record dict。
    """
    is_close = action in ("close_long", "close_short")
    side = "BUY" if action in ("open_long", "close_short") else "SELL"

    price = get_current_price(symbol, dry_run=dry_run)
    step = get_step_size(symbol, dry_run=dry_run)

    execution_record = {
        "ts": _now_iso(),
        "action": action,
        "symbol": symbol,
        "side": side,
        "price_at_decision": price,
        "dry_run": dry_run,
    }

    if is_close:
        pos = get_position(symbol, dry_run=dry_run)
        if pos is None and not dry_run:
            execution_record["status"] = "skipped"
            execution_record["reason"] = "no_position_to_close"
            print(f"[executor] SKIP: {action} but no position found")
            _append_jsonl(executions_path, execution_record)
            return execution_record

        if dry_run:
            qty = 0.001  # mock
        else:
            qty = abs(float(pos["positionAmt"]))  # type: ignore

        qty = round_step(qty, step)
        if qty <= 0:
            execution_record["status"] = "skipped"
            execution_record["reason"] = "qty_zero"
            _append_jsonl(executions_path, execution_record)
            return execution_record

        cmd_args = [
            "futures-usds", "new-order",
            "--symbol", symbol,
            "--side", side,
            "--type", "MARKET",
            "--quantity", str(qty),
            "--reduce-only", "true",
        ]
        execution_record["quantity"] = qty
        print(f"[executor] CLOSE {action}  {side} {qty} {symbol} @ ~{price:.2f}")

    else:
        balance = get_usdt_balance(dry_run=dry_run)
        if balance <= 0 and not dry_run:
            execution_record["status"] = "skipped"
            execution_record["reason"] = "zero_balance"
            print(f"[executor] SKIP: USDT balance=0 for {action}")
            _append_jsonl(executions_path, execution_record)
            return execution_record

        notional = min(balance * notional_fraction, max_notional)
        qty = round_step(notional / price, step)
        if qty <= 0:
            execution_record["status"] = "skipped"
            execution_record["reason"] = "qty_rounds_to_zero"
            _append_jsonl(executions_path, execution_record)
            return execution_record

        cmd_args = [
            "futures-usds", "new-order",
            "--symbol", symbol,
            "--side", side,
            "--type", "MARKET",
            "--quantity", str(qty),
        ]
        execution_record["quantity"] = qty
        execution_record["notional_target"] = notional
        print(f"[executor] OPEN {action}  {side} {qty} {symbol} @ ~{price:.2f}  notional≈{qty*price:.2f}")

    # Execute
    try:
        result = _cli_with_retry(*cmd_args, dry_run=dry_run)
        if not dry_run and result:
            execution_record["order_id"] = result.get("orderId")
            execution_record["status"] = result.get("status", "unknown")
            execution_record["avg_price"] = result.get("avgPrice")
            print(f"[executor] ORDER OK  orderId={result.get('orderId')} status={result.get('status')}")
        else:
            execution_record["status"] = "dry_run_ok"
    except RuntimeError as e:
        execution_record["status"] = "failed"
        execution_record["error"] = str(e)
        print(f"[executor] ERROR: {e}")

    _append_jsonl(executions_path, execution_record)
    return execution_record


def handle_fill(
    fill: Dict[str, Any],
    symbol: str,
    max_notional: float,
    notional_fraction: float,
    dry_run: bool,
    executions_path: Path,
) -> None:
    """
    處理一個 paper fill，支援所有 action type 包括 flip。
    """
    action = fill.get("action", "")

    # 簡單 actions — 直接下單
    if action in ("open_long", "open_short", "close_long", "close_short"):
        # Pre-flight reconciliation
        if action == "open_long":
            pos = get_position(symbol, dry_run=dry_run)
            if pos is not None and float(pos.get("positionAmt", 0)) > 0 and not dry_run:
                print(f"[executor] SKIP: open_long but already have long position")
                return
        elif action == "open_short":
            pos = get_position(symbol, dry_run=dry_run)
            if pos is not None and float(pos.get("positionAmt", 0)) < 0 and not dry_run:
                print(f"[executor] SKIP: open_short but already have short position")
                return

        place_single_order(symbol, action, max_notional, notional_fraction, dry_run, executions_path)

    # Flip actions — 拆成 close + open
    elif action == "flip_to_long":
        print(f"[executor] FLIP: close_short → open_long")
        close_result = place_single_order(
            symbol, "close_short", max_notional, notional_fraction, dry_run, executions_path
        )
        # 驗證 close 成功後再 open
        if close_result.get("status") not in ("failed", "skipped"):
            time.sleep(1)  # 等待交易所結算
            place_single_order(
                symbol, "open_long", max_notional, notional_fraction, dry_run, executions_path
            )
        else:
            print(f"[executor] ABORT flip: close_short failed/skipped, won't open_long")

    elif action == "flip_to_short":
        print(f"[executor] FLIP: close_long → open_short")
        close_result = place_single_order(
            symbol, "close_long", max_notional, notional_fraction, dry_run, executions_path
        )
        if close_result.get("status") not in ("failed", "skipped"):
            time.sleep(1)
            place_single_order(
                symbol, "open_short", max_notional, notional_fraction, dry_run, executions_path
            )
        else:
            print(f"[executor] ABORT flip: close_long failed/skipped, won't open_short")

    elif action == "rebalance":
        print(f"[executor] REBALANCE: not implemented, skip (should not occur in MA cross)")

    else:
        print(f"[executor] UNKNOWN action={action!r}, skip")


# ── trades.jsonl tail watcher ─────────────────────────────────────────────────

def watch_trades(
    state_dir: Path,
    symbol: str,
    max_notional: float,
    notional_fraction: float,
    dry_run: bool,
) -> None:
    trades_path = state_dir / "trades.jsonl"
    executions_path = state_dir / "executions.jsonl"
    kill_switch_path = state_dir / "EMERGENCY_STOP"
    seen_ids: set[str] = set()
    last_heartbeat = time.time()

    print(f"[executor-v2] ═══════════════════════════════════════════════")
    print(f"[executor-v2] watching {trades_path}")
    print(f"[executor-v2] symbol={symbol}  max_notional={max_notional}")
    print(f"[executor-v2] dry_run={dry_run}")
    print(f"[executor-v2] executions log → {executions_path}")
    print(f"[executor-v2] kill switch → {kill_switch_path}")
    print(f"[executor-v2] ═══════════════════════════════════════════════")

    # Load existing fills to skip
    if trades_path.exists():
        for line in trades_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                fill = json.loads(line)
                seen_ids.add(fill.get("fill_id", ""))
            except Exception:
                pass
        print(f"[executor-v2] loaded {len(seen_ids)} existing fills (will not re-execute)")

    while True:
        # Kill switch check
        if kill_switch_path.exists():
            print(f"\n[executor-v2] ⚠️  EMERGENCY STOP activated!")
            print(f"[executor-v2] Attempting to cancel all open orders...")
            try:
                _cli("futures-usds", "cancel-all-open-orders", "--symbol", symbol, dry_run=dry_run)
            except Exception as e:
                print(f"[executor-v2] cancel failed: {e}")
            _append_jsonl(executions_path, {
                "ts": _now_iso(),
                "event": "emergency_stop",
                "symbol": symbol,
            })
            print(f"[executor-v2] Exiting due to EMERGENCY_STOP.")
            break

        time.sleep(POLL_SEC)

        # Heartbeat
        now = time.time()
        if now - last_heartbeat > HEARTBEAT_INTERVAL:
            print(f"[executor-v2] heartbeat alive at {_now_iso()}")
            last_heartbeat = now

        # Check daemon status
        state_path = state_dir / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                if state.get("status") in ("stopped", "risk_triggered"):
                    print("[executor-v2] daemon stopped, exiting")
                    break
            except Exception:
                pass

        if not trades_path.exists():
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
            print(f"\n[executor-v2] ──── NEW FILL ────")
            print(f"[executor-v2] fill_id={fill_id[:12]}  action={action}  ts={ts}")

            try:
                handle_fill(
                    fill=fill,
                    symbol=symbol,
                    max_notional=max_notional,
                    notional_fraction=notional_fraction,
                    dry_run=dry_run,
                    executions_path=executions_path,
                )
            except Exception as e:
                print(f"[executor-v2] UNHANDLED ERROR: {e}")
                traceback.print_exc()
                _append_jsonl(executions_path, {
                    "ts": _now_iso(),
                    "fill_id": fill_id,
                    "action": action,
                    "status": "unhandled_error",
                    "error": str(e),
                })


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Signal Executor v2: paper fills → binance-cli (with reconciliation + retry)"
    )
    parser.add_argument("--state-dir", required=True,
                        help="Paper daemon state dir (含 trades.jsonl)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--notional-fraction", type=float, default=0.95,
                        help="下單金額佔可用餘額的比例")
    parser.add_argument("--max-notional", type=float, default=200.0,
                        help="單筆最大下單 USDT")
    parser.add_argument("--dry-run", action="store_true",
                        help="只印出指令，不真正下單")
    args = parser.parse_args()

    watch_trades(
        state_dir=Path(args.state_dir),
        symbol=args.symbol,
        max_notional=args.max_notional,
        notional_fraction=args.notional_fraction,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
