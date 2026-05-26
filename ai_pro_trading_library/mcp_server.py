"""MCP server for ai_pro_trading_library.

Wraps the same dispatch surface as `cli.py` as MCP tools so an LLM
agent can drive the library over stdio:

- `catalog_list()` → list of case ids
- `catalog_json()` → full case catalog
- `case_spec(case_id)` → StrategySpec dict
- `smoke_backtest(case_id, rows=120)` → smoke backtest result
- `active_case_smoke(rows=120)` → smoke backtest for all 6 priority cases
- `validate()` → 5-check self-test (mirror of `ai-pro-trading validate`)

`fastmcp` is the only runtime dependency, gated behind the `[mcp]`
extra. Without it `import ai_pro_trading_library.mcp_server` works but
calling `main()` raises a clear error instructing the user to
`pip install ai-pro-trading-library[mcp]`.

This file intentionally has no live-trading or network-dependent
tools — those wait until the corresponding deferral notes
(`docs/migration/live-broker-deferred.md`,
`docs/migration/historical-downloaders-deferred.md`) get unblocked.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ai_pro_trading_library.cases import list_case_ids, load_catalog
from ai_pro_trading_library.cases.loader import load_case_strategy
from ai_pro_trading_library.cli import ACTIVE_CASES, run_validate
from ai_pro_trading_library.runtime.smoke import (
    run_all_active_case_smokes,
    run_case_backtest_smoke,
)


def _tool_catalog_list() -> dict[str, Any]:
    return {"cases": list_case_ids()}


def _tool_catalog_json() -> dict[str, Any]:
    return load_catalog()


def _tool_case_spec(case_id: str) -> dict[str, Any]:
    return asdict(load_case_strategy(case_id))


def _tool_smoke_backtest(case_id: str, rows: int = 120) -> dict[str, Any]:
    return run_case_backtest_smoke(case_id, rows=rows)


def _tool_active_case_smoke(rows: int = 120) -> dict[str, Any]:
    return {"results": run_all_active_case_smokes(list(ACTIVE_CASES), rows=rows)}


def _tool_validate() -> dict[str, Any]:
    return run_validate()


# Public mapping the test harness can dispatch through without needing
# fastmcp installed.
TOOL_HANDLERS = {
    "catalog_list": _tool_catalog_list,
    "catalog_json": _tool_catalog_json,
    "case_spec": _tool_case_spec,
    "smoke_backtest": _tool_smoke_backtest,
    "active_case_smoke": _tool_active_case_smoke,
    "validate": _tool_validate,
}


def _build_server() -> Any:
    """Return a configured `fastmcp.FastMCP` instance, importing lazily."""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "fastmcp is required to run the MCP server. "
            "Install with `pip install ai-pro-trading-library[mcp]`."
        ) from exc

    server: Any = FastMCP("ai-pro-trading")

    @server.tool()
    def catalog_list() -> dict[str, Any]:
        """List the case ids available in the catalog."""
        return _tool_catalog_list()

    @server.tool()
    def catalog_json() -> dict[str, Any]:
        """Return the full case catalog as JSON."""
        return _tool_catalog_json()

    @server.tool()
    def case_spec(case_id: str) -> dict[str, Any]:
        """Return the StrategySpec dataclass for a case as a dict."""
        return _tool_case_spec(case_id)

    @server.tool()
    def smoke_backtest(case_id: str, rows: int = 120) -> dict[str, Any]:
        """Run a smoke backtest for one case (default 120 rows)."""
        return _tool_smoke_backtest(case_id, rows=rows)

    @server.tool()
    def active_case_smoke(rows: int = 120) -> dict[str, Any]:
        """Run smoke backtests for all 6 priority cases."""
        return _tool_active_case_smoke(rows=rows)

    @server.tool()
    def validate() -> dict[str, Any]:
        """Run the 5-check self-test (mirror of `ai-pro-trading validate`)."""
        return _tool_validate()

    return server


def main() -> int:
    """Console entry point for `ai-pro-trading-mcp`. Runs the server over stdio."""
    server = _build_server()
    server.run()
    return 0


__all__ = ["TOOL_HANDLERS", "main"]
