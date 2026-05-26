"""
Core subprocess runners for binance-cli and binance-pro-cli.

Architecture
------------
All CLI invocations in cyqnt_trd.data_cli go through ``run_binance_cli``
or ``run_binance_pro_cli``.  Each function:

1. Spawns the given binary with the provided args list.
2. Captures stdout (and stderr on failure).
3. Parses JSON from stdout.
4. Raises :class:`CLIError` on timeout, non-zero exit, or malformed JSON.

The binaries are **never actually imported** — they are purely invoked as
child processes so the wrapper works without the CLIs installed (useful in
tests via ``unittest.mock.patch``).
"""

from __future__ import annotations

import json
import subprocess
from typing import Union

# Configurable at module level so tests can override without patching subprocess.
BINANCE_CLI_BINARY = "binance-cli"
BINANCE_PRO_CLI_BINARY = "binance-pro-cli"

DEFAULT_TIMEOUT = 30  # seconds


class CLIError(Exception):
    """Raised when a CLI subprocess call fails."""


def _run(binary: str, args: list[str], timeout: int = DEFAULT_TIMEOUT) -> Union[dict, list]:
    """
    Internal helper — spawns *binary* with *args*, returns parsed JSON.

    Parameters
    ----------
    binary:
        Executable name or full path (e.g. ``"binance-cli"``).
    args:
        Argument list appended after the binary, e.g.
        ``["futures-usds", "kline-candlestick-data", "--symbol", "BTCUSDT"]``.
    timeout:
        Seconds before the process is killed and :class:`CLIError` is raised.

    Returns
    -------
    dict | list
        Parsed JSON from the process's stdout.

    Raises
    ------
    CLIError
        On timeout, non-zero exit code, or JSON parse failure.
    """
    cmd = [binary] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise CLIError(
            f"CLI timeout ({timeout}s): {' '.join(cmd)}"
        ) from exc
    except FileNotFoundError as exc:
        raise CLIError(
            f"CLI binary not found: {binary!r} — is it installed and on PATH?"
        ) from exc

    if result.returncode != 0:
        stderr_snip = (result.stderr or "").strip()[:500]
        raise CLIError(
            f"CLI exited {result.returncode}: {' '.join(cmd)}\nstderr: {stderr_snip}"
        )

    stdout = (result.stdout or "").strip()
    if not stdout:
        # Treat empty stdout as empty JSON object — some commands return nothing
        return {}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        snip = stdout[:300]
        raise CLIError(
            f"CLI returned non-JSON from {' '.join(cmd)!r}: {exc}\nOutput: {snip}"
        ) from exc


def run_binance_cli(args: list[str], timeout: int = DEFAULT_TIMEOUT) -> Union[dict, list]:
    """
    Spawn ``binance-cli`` with *args*, return parsed JSON dict/list.

    Example
    -------
    >>> run_binance_cli(["futures-usds", "kline-candlestick-data",
    ...                  "--symbol", "BTCUSDT", "--interval", "1h", "--limit", "1"])
    [[1700000000000, "35000.0", "35100.0", ...], ...]
    """
    return _run(BINANCE_CLI_BINARY, args, timeout=timeout)


def run_binance_pro_cli(args: list[str], timeout: int = DEFAULT_TIMEOUT) -> Union[dict, list]:
    """
    Spawn ``binance-pro-cli`` with *args*, return parsed JSON dict/list.

    Example
    -------
    >>> run_binance_pro_cli(["search", "hotrank", "futures", "--lang", "en"])
    [{"symbol": "BTCUSDT", "rank": 1, ...}, ...]
    """
    return _run(BINANCE_PRO_CLI_BINARY, args, timeout=timeout)
