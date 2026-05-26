"""shim — atomic.core.cli helpers (run_cli / safe_float / CLIError).

These are subprocess primitives. Re-export from cyqnt_trd's wrappers.
"""
from cyqnt_trd.data_cli._subprocess import run_binance_cli, run_binance_pro_cli  # noqa: F401
from cyqnt_trd.exec_cli._subprocess import CLIError  # noqa: F401


def run_cli(cmd, profile=None, timeout=30):
    """atomic shim: dispatch to run_binance_cli or run_binance_pro_cli based on cmd[0]."""
    if not cmd:
        return None
    binary = cmd[0] if isinstance(cmd, list) else "binance-cli"
    args = cmd[1:] if isinstance(cmd, list) else []
    if "pro" in binary:
        return run_binance_pro_cli(args, timeout=timeout)
    return run_binance_cli(args, timeout=timeout)


def safe_float(val, default=0.0):
    """atomic shim: convert to float, return default on error."""
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default
