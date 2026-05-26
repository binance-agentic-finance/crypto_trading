"""Shared subprocess runner and core dataclasses for exec_cli.

If cyqnt_trd.data_cli._subprocess is available, this module can delegate
to it in the future. For now it is self-contained so exec_cli works
independently of the data_cli port (task #33).

Port origin: atomic_strategy_lib.core.cli + atomic_strategy_lib.core.types
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Core result types
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    """Result from any order or position management operation.

    success       : True if the operation completed without error.
    order_id      : Exchange-assigned order ID (None for non-order ops).
    executed_qty  : Quantity actually filled (0 for pending/dry-run).
    executed_price: Average fill price (0 if not yet filled).
    fee           : Trading fee paid (0 if unavailable from CLI output).
    raw_response  : Full parsed binance-cli JSON for debugging / audit.
    error         : Human-readable error message when success=False.
    """
    success: bool
    order_id: Optional[str] = None
    executed_qty: float = 0.0
    executed_price: float = 0.0
    fee: float = 0.0
    raw_response: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ExchangeFilter:
    """Exchange trading rules fetched from binance-cli exchange-information."""
    symbol: str
    step_size: float        # LOT_SIZE stepSize
    tick_size: float        # PRICE_FILTER tickSize
    min_notional: float     # MIN_NOTIONAL
    min_qty: float          # LOT_SIZE minQty
    qty_precision: int
    price_precision: int


# ---------------------------------------------------------------------------
# CLI error
# ---------------------------------------------------------------------------

class CLIError(Exception):
    """Raised when a binance-cli subprocess call fails."""

    def __init__(self, message: str, command: list = None, stderr: str = ""):
        super().__init__(message)
        self.command = command or []
        self.stderr = stderr


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def run_cli(
    command: list,
    profile: Optional[str] = None,
    timeout: int = 10,
    json_output: bool = True,
) -> Union[dict, list]:
    """Execute a binance-cli command and return parsed JSON.

    Args:
        command: full command list, e.g.
            ["binance-cli", "futures-usds", "new-order", "--symbol", "BTCUSDT", ...]
        profile: optional --profile <name> injected after the binary name
        timeout: subprocess timeout in seconds
        json_output: append --json flag if not already present

    Returns:
        Parsed JSON as dict or list.

    Raises:
        CLIError: on non-zero exit, timeout, or unparsable output.
    """
    cmd = list(command)

    if profile:
        cmd.insert(1, "--profile")
        cmd.insert(2, profile)

    if json_output and "--json" not in cmd:
        cmd.append("--json")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise CLIError(
            f"CLI binary not found: {cmd[0]!r}. "
            "Install binance-cli (https://github.com/binance/binance-cli).",
            command=cmd,
        )
    except subprocess.TimeoutExpired:
        raise CLIError(f"CLI command timed out after {timeout}s", command=cmd)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        try:
            err_data = json.loads(stderr)
            msg = err_data.get("msg", err_data.get("message", stderr))
        except (json.JSONDecodeError, AttributeError):
            msg = f"exit {result.returncode}: {stderr}"
        raise CLIError(msg, command=cmd, stderr=stderr)

    stdout = result.stdout.strip()
    if not stdout:
        return {}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise CLIError(
            f"Failed to parse CLI JSON output: {stdout[:200]}",
            command=cmd,
        ) from exc


def run_cli_safe(
    command: list,
    profile: Optional[str] = None,
    timeout: int = 10,
) -> tuple:
    """Non-raising variant of run_cli. Returns (success: bool, data: dict|list)."""
    try:
        data = run_cli(command, profile=profile, timeout=timeout)
        return True, data
    except CLIError as exc:
        return False, {"error": str(exc), "stderr": exc.stderr}


def safe_float(val, default=None) -> Optional[float]:
    """Defensive float conversion — returns default on None / unparsable input."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_order_result(raw: Union[dict, list]) -> OrderResult:
    """Convert a raw binance-cli order response dict into an OrderResult."""
    if not isinstance(raw, dict):
        return OrderResult(
            success=False,
            raw_response={"result": str(raw)},
            error=f"Unexpected response type: {type(raw).__name__}",
        )

    # Detect error: explicit error key, or negative Binance error code
    code = raw.get("code", 0)
    has_error = bool(raw.get("error")) or (isinstance(code, int) and code < 0)

    return OrderResult(
        success=not has_error,
        order_id=str(raw["orderId"]) if "orderId" in raw else None,
        executed_qty=float(raw.get("executedQty", 0) or 0),
        executed_price=float(raw.get("avgPrice", raw.get("price", 0)) or 0),
        fee=0.0,  # binance-cli new-order doesn't include fills inline
        raw_response=raw,
        error=(raw.get("msg") or raw.get("error")) if has_error else None,
    )


def _parse_generic_result(raw: Union[dict, list]) -> OrderResult:
    """Parse generic CLI response (leverage, margin type, cancel-all)."""
    if not isinstance(raw, dict):
        return OrderResult(
            success=True,
            raw_response={"result": str(raw)},
        )
    code = raw.get("code", 0)
    has_error = bool(raw.get("error")) or (isinstance(code, int) and code < 0)
    return OrderResult(
        success=not has_error,
        raw_response=raw,
        error=(raw.get("msg") or raw.get("error")) if has_error else None,
    )
