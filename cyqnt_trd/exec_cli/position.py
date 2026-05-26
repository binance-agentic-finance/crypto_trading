"""L5-01 to L5-02: Position setup — leverage and margin type.

Port of atomic_strategy_lib.execution.position (set_leverage, set_margin_type).

Filter/quantize functions live in ``filters.py`` (L5-03 to L5-05).

SAFETY CONTRACT
---------------
All public functions default to ``dry_run=True``.
Pass ``dry_run=False`` explicitly to send real requests.
"""

from __future__ import annotations

from ._subprocess import CLIError, OrderResult, _parse_generic_result, run_cli


# ---------------------------------------------------------------------------
# L5-01: Set leverage
# ---------------------------------------------------------------------------

def set_leverage(
    symbol: str,
    leverage: int,
    profile: str = "default",
    binary: str = "binance-cli",
    dry_run: bool = True,
) -> OrderResult:
    """Set futures leverage for a symbol.

    Args:
        symbol:   Trading pair, e.g. "BTCUSDT"
        leverage: Positive integer (e.g. 5, 10, 20, 125)
        dry_run:  **True by default** — print command, no execution.

    Returns:
        OrderResult with raw_response from binance-cli.
    """
    if not symbol:
        raise ValueError("symbol must not be empty")
    if not isinstance(leverage, int) or leverage <= 0:
        raise ValueError(f"leverage must be a positive integer, got {leverage!r}")

    cmd = [
        binary, "futures-usds", "change-leverage",
        "--symbol", symbol, "--leverage", str(int(leverage)),
    ]

    if dry_run:
        print(f"[DRY RUN] set_leverage: {' '.join(cmd)}")
        return OrderResult(
            success=True,
            raw_response={"dryRun": True, "symbol": symbol, "leverage": leverage},
        )

    try:
        raw = run_cli(cmd, profile=profile)
        return _parse_generic_result(raw)
    except CLIError as exc:
        return OrderResult(
            success=False,
            raw_response={"error": str(exc), "command": cmd},
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# L5-02: Set margin type
# ---------------------------------------------------------------------------

def set_margin_type(
    symbol: str,
    margin_type: str = "ISOLATED",
    profile: str = "default",
    binary: str = "binance-cli",
    dry_run: bool = True,
) -> OrderResult:
    """Set margin type (ISOLATED or CROSSED) for a futures symbol.

    Note: Binance returns code -4046 when the margin type is already set to
    the requested value — this is treated as a success (idempotent).

    Args:
        symbol:      Trading pair
        margin_type: "ISOLATED" (default) or "CROSSED"
        dry_run:     **True by default** — print command, no execution.
    """
    if not symbol:
        raise ValueError("symbol must not be empty")
    if margin_type not in ("ISOLATED", "CROSSED"):
        raise ValueError(
            f"margin_type must be 'ISOLATED' or 'CROSSED', got {margin_type!r}"
        )

    cmd = [
        binary, "futures-usds", "change-margin-type",
        "--symbol", symbol, "--marginType", margin_type,
    ]

    if dry_run:
        print(f"[DRY RUN] set_margin_type: {' '.join(cmd)}")
        return OrderResult(
            success=True,
            raw_response={"dryRun": True, "symbol": symbol, "marginType": margin_type},
        )

    try:
        raw = run_cli(cmd, profile=profile)
        # -4046 = "No need to change margin type" → idempotent success
        if isinstance(raw, dict) and raw.get("code") == -4046:
            return OrderResult(success=True, raw_response=raw)
        return _parse_generic_result(raw)
    except CLIError as exc:
        return OrderResult(
            success=False,
            raw_response={"error": str(exc), "command": cmd},
            error=str(exc),
        )
