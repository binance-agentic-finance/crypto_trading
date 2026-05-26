"""
cyqnt_trd.standard_bot.orchestration.state
============================================

JSON-based position CRUD and JSONL-based signal log CRUD.

Ported from atomic_strategy_lib.orchestration.state (L8-08, L8-09).

Schema compatibility
--------------------
The position schema emitted by :func:`position_create` is intentionally
identical to the one written by the daemon's existing ``state.json``::

    {
        "id":                 "<symbol>_<unix_ts>",
        "symbol":             "BTCUSDT",
        "direction":          "LONG",
        "mode":               "futures",
        "entry_price":        42000.0,
        "stop_price":         41000.0,
        "stop_pct":           2.4,
        "leverage":           5,
        "notional":           1000.0,
        "max_loss":           20.0,
        "verdict":            "STRONG_CANDIDATE",
        "score":              3.5,
        "status":             "open",
        "opened_at":          "2024-01-01T00:00:00+00:00",
        "graduated_tp_stage": 0,
        "peak_profit_pct":    0
    }

All writes are atomic (write to ``.tmp`` then ``os.replace``) to prevent
corrupt files on daemon crash.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "position_create",
    "position_list",
    "position_update",
    "position_close",
    "position_open_list",
    "signal_log_append",
    "signal_log_load",
    "signal_log_save",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically via a ``.tmp`` side-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def _save_positions(path: Path, positions: list[dict]) -> None:
    _atomic_write(path, json.dumps(positions, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# L8-08  Position record CRUD
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
    """Create and persist a new position record.

    Parameters
    ----------
    symbol:
        Trading pair, e.g. ``"BTCUSDT"``.
    direction:
        ``"LONG"`` or ``"SHORT"``.
    entry_price:
        Entry fill price.
    stop_price:
        Absolute stop-loss price.
    stop_pct:
        Stop-loss distance as a percentage.
    leverage:
        Leverage multiplier.
    notional:
        Position size in quote currency.
    max_loss:
        Maximum allowed loss in quote currency.
    verdict:
        Optional verdict string from scoring layer.
    score:
        Optional composite score.
    mode:
        ``"futures"`` (default) or ``"spot"``.
    positions_file:
        Path to the JSON positions file.

    Returns
    -------
    dict
        The newly created position record.
    """
    record: dict[str, Any] = {
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

    path = Path(positions_file)
    positions = position_list(positions_file)
    positions.append(record)
    _save_positions(path, positions)

    return record


def position_list(positions_file: str = "tmp/positions.json") -> list[dict]:
    """Load all position records (open and closed) from file.

    Returns an empty list if the file does not exist or is corrupt.
    """
    path = Path(positions_file)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        return []


def position_update(
    position_id: str,
    updates: dict,
    positions_file: str = "tmp/positions.json",
) -> bool:
    """Update arbitrary fields of a position record by ID.

    Parameters
    ----------
    position_id:
        The ``"id"`` field of the position to update.
    updates:
        Dict of field:value pairs to merge into the record.
    positions_file:
        Path to the JSON positions file.

    Returns
    -------
    bool
        ``True`` if the record was found and updated, ``False`` otherwise.
    """
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
    """Mark a position as closed (sets ``status="closed"`` and ``closed_at``).

    Returns
    -------
    bool
        ``True`` if the record was found and updated.
    """
    return position_update(
        position_id,
        {
            "status": "closed",
            "closed_at": datetime.now(timezone.utc).isoformat(),
        },
        positions_file,
    )


def position_open_list(positions_file: str = "tmp/positions.json") -> list[dict]:
    """Return only positions with ``status == "open"``."""
    return [p for p in position_list(positions_file) if p.get("status") == "open"]


# ---------------------------------------------------------------------------
# L8-09  Signal log CRUD (JSONL)
# ---------------------------------------------------------------------------

def signal_log_append(
    signal: dict,
    signals_file: str = "tmp/signals_log.jsonl",
) -> None:
    """Append a single signal dict to the JSONL log file.

    Parent directories are created if they do not exist.
    """
    path = Path(signals_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(signal, ensure_ascii=False) + "\n")


def signal_log_load(signals_file: str = "tmp/signals_log.jsonl") -> list[dict]:
    """Load all signals from a JSONL file.

    Returns an empty list if the file does not exist.  Skips blank lines
    silently.
    """
    path = Path(signals_file)
    if not path.exists():
        return []
    signals: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                signals.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip corrupt lines
    return signals


def signal_log_save(
    signals: list[dict],
    signals_file: str = "tmp/signals_log.jsonl",
) -> None:
    """Rewrite the entire JSONL signal log atomically.

    Useful for back-filling ``outcomes`` on existing signal records.
    """
    payload = "".join(json.dumps(sig, ensure_ascii=False) + "\n" for sig in signals)
    _atomic_write(Path(signals_file), payload)
