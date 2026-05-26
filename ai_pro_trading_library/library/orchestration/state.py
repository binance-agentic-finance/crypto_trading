"""Position record + signal-log persistence — L2 parity vs `atomic.orchestration.state`.

Independent from atomic's `core.run_output.atomic_write_text` — uses a
local atomic-write helper.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically: tmp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Position records
# ---------------------------------------------------------------------------


def position_create(
    symbol: str,
    direction: str,
    entry_price: float,
    stop_price: float,
    stop_pct: float,
    leverage: float,
    notional: float,
    max_loss: float,
    verdict: str = "",
    score: float = 0.0,
    mode: str = "futures",
    positions_file: str = "tmp/positions.json",
) -> dict:
    """Create and persist a new position record."""
    record = {
        "id": "%s_%d" % (symbol, int(time.time())),
        "symbol": symbol,
        "direction": direction,
        "mode": mode,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "stop_pct": stop_pct,
        "leverage": leverage,
        "notional": notional,
        "max_loss": max_loss,
        "verdict": verdict,
        "score": score,
        "status": "open",
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "graduated_tp_stage": 0,
        "peak_profit_pct": 0,
    }
    positions = position_list(positions_file)
    positions.append(record)
    _save_positions(Path(positions_file), positions)
    return record


def position_list(positions_file: str = "tmp/positions.json") -> list[dict]:
    """Load all position records from file."""
    path = Path(positions_file)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return []


def position_update(
    position_id: str,
    updates: dict,
    positions_file: str = "tmp/positions.json",
) -> bool:
    """Update fields of a position record by ID."""
    positions = position_list(positions_file)
    for pos in positions:
        if pos.get("id") == position_id:
            pos.update(updates)
            _save_positions(Path(positions_file), positions)
            return True
    return False


def position_close(
    position_id: str,
    positions_file: str = "tmp/positions.json",
) -> bool:
    """Mark a position as closed."""
    return position_update(
        position_id,
        {"status": "closed", "closed_at": datetime.now(timezone.utc).isoformat()},
        positions_file,
    )


def position_open_list(positions_file: str = "tmp/positions.json") -> list[dict]:
    """Get only open positions."""
    return [p for p in position_list(positions_file) if p.get("status") == "open"]


def _save_positions(path: Path, positions: list[dict]) -> None:
    _atomic_write_text(path, json.dumps(positions, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Signal log CRUD
# ---------------------------------------------------------------------------


def signal_log_append(
    signal: dict,
    signals_file: str = "tmp/signals_log.jsonl",
) -> None:
    """Append a signal dict to the JSONL log."""
    path = Path(signals_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(signal, ensure_ascii=False) + "\n")


def signal_log_load(signals_file: str = "tmp/signals_log.jsonl") -> list[dict]:
    """Load all signals from JSONL file."""
    path = Path(signals_file)
    if not path.exists():
        return []
    signals: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            signals.append(json.loads(line))
    return signals


def signal_log_save(
    signals: list[dict],
    signals_file: str = "tmp/signals_log.jsonl",
) -> None:
    """Rewrite all signals to JSONL atomically."""
    payload = "".join(json.dumps(sig, ensure_ascii=False) + "\n" for sig in signals)
    _atomic_write_text(Path(signals_file), payload)


__all__ = [
    "position_close",
    "position_create",
    "position_list",
    "position_open_list",
    "position_update",
    "signal_log_append",
    "signal_log_load",
    "signal_log_save",
]
