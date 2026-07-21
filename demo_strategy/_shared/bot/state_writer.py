"""State + JSONL log writer — matches bdp-ai-trading-bot's file contract:

    <bot_dir>/state/state.json     (atomic write via tmp + rename)
    <bot_dir>/logs/events.jsonl    (append; one JSON line per event)
    <bot_dir>/logs/trades.jsonl    (append; one JSON line per fill)

bdp-bot's `supervisor.reader.read_state` and `tail_jsonl` read these
files directly — so the schema below is a hard contract.

state.json required keys:
    status:        "running" | "stopped" | "risk_halted" | "error"
    equity:        float (USDT)
    open_positions: dict[symbol → signed_qty]
    last_bar_ts:   int (ms epoch of last processed bar)
    last_signal_ts: int (ms epoch of last emitted signal)
    updated_at:    int (ms epoch)

events.jsonl each row:
    { ts, kind, ... }
    kind ∈ {started, signal, order_placed, order_filled,
             stopped, risk_halted, error}

trades.jsonl each row:
    { ts, symbol, side, qty, price, pnl_usdt, ... }
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


_STATUS_ENUM = {"running", "stopped", "risk_halted", "error"}


def _now_ms() -> int:
    return int(time.time() * 1000)


class StateBook:
    """Thin, thread-unsafe wrapper — one instance per daemon loop."""

    def __init__(self, bot_dir: str | Path):
        self.bot_dir    = Path(bot_dir).resolve()
        self.state_dir  = self.bot_dir / "state"
        self.logs_dir   = self.bot_dir / "logs"
        self.state_path = self.state_dir / "state.json"
        self.events     = self.logs_dir / "events.jsonl"
        self.trades     = self.logs_dir / "trades.jsonl"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # in-memory copy — write_state() serializes this dict
        self._state: dict[str, Any] = {
            "status":         "running",
            "equity":         0.0,
            "open_positions": {},
            "last_bar_ts":    0,
            "last_signal_ts": 0,
            "updated_at":     _now_ms(),
        }

    # ── State ────────────────────────────────────────────────────────
    def set_state(self, **kv) -> None:
        """Merge kwargs into state; caller decides when to flush."""
        for k, v in kv.items():
            if k == "status" and v not in _STATUS_ENUM:
                raise ValueError(f"status must be in {_STATUS_ENUM}, got {v!r}")
            self._state[k] = v

    def flush_state(self) -> None:
        """Atomic write: state.json.tmp → rename → state.json."""
        self._state["updated_at"] = _now_ms()
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self.state_path)

    def state(self) -> dict[str, Any]:
        return dict(self._state)

    # ── Events / trades ──────────────────────────────────────────────
    def append_event(self, kind: str, **payload) -> None:
        self._append_jsonl(self.events, {"ts": _now_ms(), "kind": kind, **payload})

    def append_trade(self, **payload) -> None:
        self._append_jsonl(self.trades, {"ts": _now_ms(), **payload})

    @staticmethod
    def _append_jsonl(path: Path, obj: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
