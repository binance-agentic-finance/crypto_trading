"""
session_watcher.py — MA-cross strategy session watcher

Reads state.json and trades.jsonl from the paper daemon state dir.
Responsibilities:
  - Detect new fills (trades.jsonl) and print a fill notification
  - Detect live executor fills (executions.jsonl) and reconcile direction
  - Risk check: equity drawdown > threshold → write stop request to state.json
  - Detect daemon stopped/risk_triggered status → print summary + exit cleanly
  - Final cleanup: summarise session on exit

Usage:
  python3 session_watcher.py \
    --state-dir ./watcher/VAL_ETHUSDT_1m \
    --run-id VAL_ETHUSDT_1m \
    --poll-interval 15 \
    --max-drawdown-pct 20.0 \
    [--session-end-at 2026-06-03T17:45:00Z]

Design:
  - Fully external process — reads files only, no in-process state
  - Tolerant of missing files (daemon may not have written them yet)
  - Idempotent: repeated polls are safe
  - Will stop itself when daemon status is "stopped" or "risk_triggered"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    return rows


def _parse_deadline(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _write_stop_request(state_path: Path, reason: str) -> None:
    """Write a stop request into state.json (daemon will pick it up on next poll)."""
    import tempfile
    data = _load_json(state_path)
    data["status"] = "risk_triggered"
    data["stop_reason"] = reason
    data["stopped_at"] = datetime.now(timezone.utc).isoformat()
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(state_path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(state_path))
    except Exception as e:
        print(f"[watcher] WARNING: could not write stop request: {e}")


# ── Banner helpers ─────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}\n")


def _log(tag: str, msg: str) -> None:
    print(f"[{_now_iso()}] [{tag}] {msg}", flush=True)


# ── Fill notification ──────────────────────────────────────────────────────────

def _format_fill(trade: Dict[str, Any]) -> str:
    action = trade.get("action", "?")
    side = trade.get("side", "?")
    price = trade.get("price", 0)
    qty = trade.get("qty") or trade.get("quantity", 0)
    pnl = trade.get("pnl")
    pnl_str = f"  pnl={pnl:+.4f}" if pnl is not None else ""
    equity = trade.get("equity")
    eq_str = f"  equity={equity:.2f}" if equity is not None else ""
    ts = trade.get("ts") or trade.get("timestamp", "?")
    return (
        f"action={action}  side={side}  price={price:.4f}  qty={qty:.6f}"
        f"{pnl_str}{eq_str}  ts={ts}"
    )


def _format_execution(ex: Dict[str, Any]) -> str:
    event = ex.get("event", "?")
    action = ex.get("action", "?")
    symbol = ex.get("symbol", "?")
    qty = ex.get("qty_placed", ex.get("preview_qty", "?"))
    notional = ex.get("notional", "?")
    dry = "DRY-RUN " if ex.get("dry_run") else ""
    return f"{dry}event={event}  action={action}  symbol={symbol}  qty={qty}  notional={notional}"


# ── Consistency check ──────────────────────────────────────────────────────────

_ACTION_TO_DIRECTION = {
    "open_long": "long",
    "flip_to_long": "long",
    "close_long": "flat",
    "open_short": "short",
    "flip_to_short": "short",
    "close_short": "flat",
}


def _extract_direction(action: str) -> Optional[str]:
    return _ACTION_TO_DIRECTION.get(action)


def _check_consistency(
    paper_trade: Dict[str, Any],
    exec_record: Dict[str, Any],
) -> str:
    """
    Compare the direction implied by a paper fill with the direction of the
    live execution that followed it.
    Returns 'OK', 'MISMATCH', or 'UNKNOWN'.
    """
    paper_action = paper_trade.get("action", "")
    exec_action = exec_record.get("action", "")
    paper_dir = _extract_direction(paper_action)
    exec_dir = _extract_direction(exec_action)
    if paper_dir is None or exec_dir is None:
        return "UNKNOWN"
    # flip_to_* in paper → split into two executions; both directions end up
    # the same target, so just compare target direction
    if paper_dir == exec_dir:
        return "OK"
    return f"MISMATCH (paper→{paper_dir}, exec→{exec_dir})"


# ── Watcher main loop ──────────────────────────────────────────────────────────

class SessionWatcher:
    def __init__(
        self,
        *,
        state_dir: Path,
        run_id: str,
        poll_interval: int,
        max_drawdown_pct: float,
        session_end_at: Optional[str],
    ) -> None:
        self.state_dir = state_dir
        self.run_id = run_id
        self.poll_interval = poll_interval
        self.max_drawdown_pct = max_drawdown_pct
        self.deadline = _parse_deadline(session_end_at)

        self.state_path = state_dir / "state.json"
        self.trades_path = state_dir / "trades.jsonl"
        self.executions_path = state_dir / "executions.jsonl"
        self.daemon_log = state_dir / "daemon.log"

        self._seen_fill_ids: Set[str] = set()
        self._seen_exec_ids: Set[str] = set()
        self._initial_equity: Optional[float] = None
        self._poll_count: int = 0
        self._fill_count: int = 0
        self._exec_count: int = 0
        self._risk_triggered: bool = False

    def run(self) -> int:
        _banner(f"SESSION WATCHER STARTED  run_id={self.run_id}")
        _log("watcher", f"state_dir={self.state_dir}")
        _log("watcher", f"poll_interval={self.poll_interval}s  max_drawdown={self.max_drawdown_pct}%")
        if self.deadline:
            _log("watcher", f"session_end_at={datetime.fromtimestamp(self.deadline, tz=timezone.utc).isoformat()}")

        while True:
            self._poll_count += 1

            # Check deadline first
            if self.deadline and time.time() >= self.deadline:
                _log("watcher", "session_end_at reached → requesting daemon stop")
                _write_stop_request(self.state_path, "session_end_at")
                time.sleep(self.poll_interval)
                # One final poll then exit
                self._poll()
                break

            stop = self._poll()
            if stop:
                break

            time.sleep(self.poll_interval)

        self._print_final_summary()
        return 0

    def _poll(self) -> bool:
        """Returns True when watcher should stop."""
        state = _load_json(self.state_path)
        if not state:
            _log("watcher", f"poll #{self._poll_count}: state.json not yet available, waiting...")
            return False

        # Capture initial equity
        if self._initial_equity is None:
            self._initial_equity = float(
                state.get("session_start_equity") or state.get("initial_capital") or 10000.0
            )

        current_equity = float(state.get("current_equity", self._initial_equity))
        status = state.get("status", "running")
        pos = state.get("position") or "flat"
        price = state.get("current_price", 0.0)
        bar_count = state.get("bar_count", 0)
        trade_count = state.get("trade_count", 0)

        # Process new fills
        new_fills = self._process_new_fills()

        # Process new executions
        new_execs = self._process_new_executions(new_fills)

        # Risk check
        if not self._risk_triggered and self._initial_equity and self._initial_equity > 0:
            drawdown_pct = (self._initial_equity - current_equity) / self._initial_equity * 100.0
            if drawdown_pct >= self.max_drawdown_pct:
                _log("RISK", (
                    f"⛔  MAX DRAWDOWN BREACHED: {drawdown_pct:.2f}% >= {self.max_drawdown_pct}%  "
                    f"equity={current_equity:.2f} initial={self._initial_equity:.2f}"
                ))
                _write_stop_request(self.state_path, f"max_drawdown_{drawdown_pct:.1f}pct")
                self._risk_triggered = True

        # Status log
        _log("watcher", (
            f"poll #{self._poll_count}  status={status}  equity={current_equity:.2f}  "
            f"pos={pos}  price={price:.4f}  bars={bar_count}  trades={trade_count}  "
            f"new_fills={len(new_fills)}  new_execs={len(new_execs)}"
        ))

        # Check for terminal states
        if status in ("stopped", "risk_triggered"):
            reason = state.get("stop_reason", "")
            _log("watcher", f"daemon reached terminal status={status}  reason={reason}")
            return True

        return False

    def _process_new_fills(self) -> List[Dict[str, Any]]:
        """Detect new fills, print notifications, return new ones."""
        rows = _load_jsonl(self.trades_path)
        new_fills: List[Dict[str, Any]] = []

        for row in rows:
            # trades.jsonl rows may be bare trade dicts or {trade: {...}}
            trade = row.get("trade") if isinstance(row.get("trade"), dict) else row
            if not isinstance(trade, dict):
                continue
            fill_id = str(trade.get("fill_id") or trade.get("ts") or id(trade))
            if fill_id in self._seen_fill_ids:
                continue
            self._seen_fill_ids.add(fill_id)
            self._fill_count += 1
            new_fills.append(trade)
            _log("FILL", f"#{self._fill_count} PAPER FILL  {_format_fill(trade)}")

        return new_fills

    def _process_new_executions(self, recent_fills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect new live executions, check consistency against recent paper fills."""
        rows = _load_jsonl(self.executions_path)
        new_execs: List[Dict[str, Any]] = []

        for row in rows:
            ex_id = str(row.get("ts") or row.get("fill_id") or id(row))
            if ex_id in self._seen_exec_ids:
                continue
            self._seen_exec_ids.add(ex_id)
            self._exec_count += 1
            new_execs.append(row)
            _log("EXEC", f"#{self._exec_count} LIVE EXEC  {_format_execution(row)}")

            # Consistency: match this execution to the most recent paper fill by direction
            if recent_fills:
                last_fill = recent_fills[-1]
                consistency = _check_consistency(last_fill, row)
                if consistency == "OK":
                    _log("CONSISTENCY", f"✅  paper action={last_fill.get('action')} → live action={row.get('action')}  ALIGNED")
                elif consistency.startswith("MISMATCH"):
                    _log("CONSISTENCY", f"⚠️   {consistency}  — paper={last_fill.get('action')}  live={row.get('action')}")
                else:
                    _log("CONSISTENCY", f"ℹ️   {consistency}  paper={last_fill.get('action')}  live={row.get('action')}")

        return new_execs

    def _print_final_summary(self) -> None:
        state = _load_json(self.state_path)
        trades = _load_jsonl(self.trades_path)
        execs = _load_jsonl(self.executions_path)

        current_equity = float(state.get("current_equity", self._initial_equity or 10000.0))
        total_return_pct = 0.0
        if self._initial_equity and self._initial_equity > 0:
            total_return_pct = (current_equity - self._initial_equity) / self._initial_equity * 100.0

        # Count paper fills
        paper_fill_count = sum(
            1 for row in trades
            if isinstance(row.get("trade") if isinstance(row.get("trade"), dict) else row, dict)
        )
        # Count live executions (dry-run or real order placed)
        live_order_count = sum(
            1 for ex in execs
            if ex.get("status") in ("dry_run_ok", "order_placed", "order_success")
            or ex.get("event") in ("order_placed", "order_dry_run", "order_success")
        )

        # Direction consistency
        paper_actions = []
        for row in trades:
            trade = row.get("trade") if isinstance(row.get("trade"), dict) else row
            if isinstance(trade, dict) and trade.get("action"):
                paper_actions.append(trade["action"])

        live_actions = [
            ex.get("action", "") for ex in execs
            if ex.get("status") in ("dry_run_ok", "order_placed", "order_success")
            or ex.get("event") in ("order_placed", "order_dry_run", "order_success")
        ]

        _banner("SESSION WATCHER FINAL SUMMARY")
        print(f"  run_id           : {self.run_id}")
        print(f"  status           : {state.get('status', 'unknown')}")
        print(f"  stop_reason      : {state.get('stop_reason', '-')}")
        print(f"  initial_equity   : {self._initial_equity:.2f}")
        print(f"  final_equity     : {current_equity:.2f}")
        print(f"  total_return     : {total_return_pct:+.4f}%")
        print(f"  paper_fills      : {paper_fill_count}")
        print(f"  live_execs       : {live_order_count}")
        print(f"  risk_triggered   : {self._risk_triggered}")
        print(f"  paper_actions    : {paper_actions}")
        print(f"  live_actions     : {live_actions}")

        # Consistency verdict
        if paper_actions and live_actions:
            mismatches = 0
            for p, l in zip(paper_actions, live_actions):
                pd = _extract_direction(p)
                ld = _extract_direction(l)
                if pd and ld and pd != ld:
                    mismatches += 1
            if mismatches == 0:
                print(f"\n  CONSISTENCY VERDICT: ✅  All {len(live_actions)} executions aligned with paper signals")
            else:
                print(f"\n  CONSISTENCY VERDICT: ⚠️   {mismatches} direction mismatches out of {min(len(paper_actions), len(live_actions))} compared")
        elif paper_actions and not live_actions:
            print(f"\n  CONSISTENCY VERDICT: ℹ️   Paper fills present but no live executions recorded")
        else:
            print(f"\n  CONSISTENCY VERDICT: ℹ️   No fills recorded this session")

        print(f"\n  Watcher exiting cleanly.\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="MA-cross session watcher")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--run-id", default="VAL_RUN")
    parser.add_argument("--poll-interval", type=int, default=15,
                        help="Seconds between polls (default 15)")
    parser.add_argument("--max-drawdown-pct", type=float, default=20.0,
                        help="Max equity drawdown %% before triggering risk stop (default 20)")
    parser.add_argument("--session-end-at", default=None,
                        help="ISO8601 UTC deadline; watcher requests daemon stop at this time")
    args = parser.parse_args()

    watcher = SessionWatcher(
        state_dir=Path(args.state_dir),
        run_id=args.run_id,
        poll_interval=args.poll_interval,
        max_drawdown_pct=args.max_drawdown_pct,
        session_end_at=args.session_end_at,
    )
    return watcher.run()


if __name__ == "__main__":
    raise SystemExit(main())
