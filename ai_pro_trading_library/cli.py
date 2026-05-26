"""Command line interface for ai_pro_trading_library."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any

from ai_pro_trading_library.cases import (
    list_case_ids,
    list_report_case_ids,
    load_case_report,
    load_catalog,
    runnable_report_case_ids,
)
from ai_pro_trading_library.cases.loader import load_case_strategy
from ai_pro_trading_library.runtime.smoke import run_all_active_case_smokes, run_case_backtest_smoke


ACTIVE_CASES = (
    "rsi-mean-reversion",
    "bb-squeeze-momentum",
    "btc-multi-factor-trend",
    "structure-breakout-priceaction",
    "stochrsi-mdi-trend-v1",
    "funding-rate-scanner",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-pro-trading", description="AI Pro trading library CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)

    catalog = subcommands.add_parser("catalog", help="Inspect the case catalog")
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_sub.add_parser("list", help="List case ids")
    catalog_sub.add_parser("json", help="Print catalog JSON")

    case = subcommands.add_parser("case", help="Inspect migrated cases")
    case_sub = case.add_subparsers(dest="case_command", required=True)
    case_spec = case_sub.add_parser("spec", help="Print a case StrategySpec as JSON")
    case_spec.add_argument("case_id")
    case_report = case_sub.add_parser(
        "report",
        help="Run a report case (scanner_report / trade_plan_report) and print JSON",
    )
    case_report.add_argument("case_id")
    case_report.add_argument(
        "--fixture",
        action="store_true",
        help="Load fixtures/input.json instead of requiring --inputs",
    )
    case_report.add_argument(
        "--inputs",
        default=None,
        help="Path to a JSON file with external inputs (earnings/macro/etc.)",
    )
    case_list_reports = case_sub.add_parser(
        "list-reports",
        help="List report-type case ids (with --runnable to filter to those with runner.py)",
    )
    case_list_reports.add_argument("--runnable", action="store_true")

    smoke = subcommands.add_parser("smoke", help="Run deterministic local smoke checks")
    smoke_sub = smoke.add_subparsers(dest="smoke_command", required=True)
    backtest = smoke_sub.add_parser("backtest", help="Run one case through BacktestRunner")
    backtest.add_argument("case_id")
    backtest.add_argument("--rows", type=int, default=120)
    all_cases = smoke_sub.add_parser("active-cases", help="Run all active cases through BacktestRunner")
    all_cases.add_argument("--rows", type=int, default=120)

    subcommands.add_parser(
        "validate",
        help="Self-check: registry has 6 priority cases; schema validates; smoke passes",
    )

    paper = subcommands.add_parser("paper", help="Run a case through the paper daemon")
    paper_sub = paper.add_subparsers(dest="paper_command", required=True)
    paper_start = paper_sub.add_parser("start", help="Start a bounded paper-daemon run for a case")
    paper_start.add_argument("case_id")
    paper_start.add_argument("--output-dir", default=".ai_pro_trading_runs")
    paper_start.add_argument("--run-id", default=None)
    paper_start.add_argument("--max-iterations", type=int, default=1)

    runs = subcommands.add_parser("runs", help="Inspect / control runs managed by LocalRunManager")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_sub.add_parser("list", help="List managed runs")
    runs_list.add_argument("--root-dir", default=".ai_pro_trading_runs")
    runs_get = runs_sub.add_parser("get", help="Show one managed run record")
    runs_get.add_argument("run_id")
    runs_get.add_argument("--root-dir", default=".ai_pro_trading_runs")
    runs_stop = runs_sub.add_parser("stop", help="SIGTERM a managed run")
    runs_stop.add_argument("run_id")
    runs_stop.add_argument("--root-dir", default=".ai_pro_trading_runs")
    runs_stop.add_argument("--timeout-seconds", type=float, default=5.0)
    runs_kill = runs_sub.add_parser("kill", help="SIGKILL a managed run")
    runs_kill.add_argument("run_id")
    runs_kill.add_argument("--root-dir", default=".ai_pro_trading_runs")
    runs_kill.add_argument("--timeout-seconds", type=float, default=2.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = dispatch(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    if payload is not None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def dispatch(args: argparse.Namespace) -> Any:
    if args.command == "catalog" and args.catalog_command == "list":
        return {"cases": list_case_ids()}
    if args.command == "catalog" and args.catalog_command == "json":
        return load_catalog()
    if args.command == "case" and args.case_command == "spec":
        return asdict(load_case_strategy(args.case_id))
    if args.command == "case" and args.case_command == "report":
        external_inputs: dict[str, Any] | None = None
        if args.inputs:
            external_inputs = json.loads(__import__("pathlib").Path(args.inputs).read_text())
        return load_case_report(
            args.case_id,
            adapter=None,
            inputs=external_inputs,
            fixture=args.fixture,
        )
    if args.command == "case" and args.case_command == "list-reports":
        if args.runnable:
            return {"cases": runnable_report_case_ids()}
        return {"cases": list_report_case_ids()}
    if args.command == "smoke" and args.smoke_command == "backtest":
        return run_case_backtest_smoke(args.case_id, rows=args.rows)
    if args.command == "smoke" and args.smoke_command == "active-cases":
        return {"results": run_all_active_case_smokes(list(ACTIVE_CASES), rows=args.rows)}
    if args.command == "validate":
        return run_validate()
    if args.command == "paper" and args.paper_command == "start":
        return run_paper_start(
            args.case_id,
            output_dir=args.output_dir,
            run_id=args.run_id,
            max_iterations=args.max_iterations,
        )
    if args.command == "runs":
        from ai_pro_trading_library.runtime.run_manager import LocalRunManager

        manager = LocalRunManager(root_dir=args.root_dir)
        if args.runs_command == "list":
            return {"runs": [r.to_dict() for r in manager.list_runs()]}
        if args.runs_command == "get":
            return manager.get_run(args.run_id).to_dict()
        if args.runs_command == "stop":
            return manager.stop(args.run_id, timeout_seconds=args.timeout_seconds).to_dict()
        if args.runs_command == "kill":
            return manager.kill(args.run_id, timeout_seconds=args.timeout_seconds).to_dict()
    raise ValueError(f"unsupported command: {args.command}")


def run_paper_start(
    case_id: str,
    *,
    output_dir: str = ".ai_pro_trading_runs",
    run_id: str | None = None,
    max_iterations: int = 1,
) -> dict[str, Any]:
    """Boot a `PaperDaemon` for `case_id` and return the final state dict."""
    from pathlib import Path

    from ai_pro_trading_library.runtime.daemon import PaperDaemon

    spec = load_case_strategy(case_id)
    daemon = PaperDaemon(output_dir=Path(output_dir), run_id=run_id)
    state = daemon.start(spec, max_iterations=max_iterations)
    return state.to_dict()


def run_validate() -> dict[str, Any]:
    """Self-check: confirm package is installed correctly and core invariants hold."""
    from ai_pro_trading_library.cases import load_catalog as _load, validate_case
    from ai_pro_trading_library.library.strategy import default_registry

    checks: list[dict[str, Any]] = []

    # 1. Catalog has all 6 priority cases
    catalog_ids = {c["case_id"] for c in _load()["cases"]}
    missing = [c for c in ACTIVE_CASES if c not in catalog_ids]
    checks.append(
        {
            "name": "catalog_has_priority_cases",
            "passed": not missing,
            "missing": missing,
        }
    )

    # 2. Each case validates against case.schema.json
    schema_failures: list[dict[str, str]] = []
    for c in _load()["cases"]:
        try:
            validate_case(c)
        except Exception as exc:  # noqa: BLE001
            schema_failures.append({"case_id": c["case_id"], "error": str(exc)})
    checks.append(
        {"name": "case_schema_validation", "passed": not schema_failures,
         "failures": schema_failures}
    )

    # 3. All priority cases registered with default_registry
    registered = set(default_registry.list_ids())
    unregistered = [c for c in ACTIVE_CASES if c not in registered]
    checks.append(
        {
            "name": "priority_cases_registered",
            "passed": not unregistered,
            "unregistered": unregistered,
        }
    )

    # 4. Smoke backtest each priority case
    smoke_failures: list[dict[str, str]] = []
    for case_id in ACTIVE_CASES:
        try:
            r = run_case_backtest_smoke(case_id, rows=60)
            if r["bars"] != 60:
                smoke_failures.append({"case_id": case_id, "error": f"expected 60 bars got {r['bars']}"})
        except Exception as exc:  # noqa: BLE001
            smoke_failures.append({"case_id": case_id, "error": str(exc)})
    checks.append(
        {"name": "smoke_backtest_runs", "passed": not smoke_failures,
         "failures": smoke_failures}
    )

    # 5. Package data accessible (catalog.json + schema)
    from importlib.resources import files as _files
    pkg_root = _files("ai_pro_trading_library.cases")
    package_data_ok = (
        pkg_root.joinpath("catalog.json").is_file()
        and pkg_root.joinpath("case.schema.json").is_file()
    )
    checks.append(
        {"name": "package_data_present", "passed": package_data_ok}
    )

    all_ok = all(c["passed"] for c in checks)
    return {
        "ok": all_ok,
        "checks": checks,
        "summary": (
            f"{sum(c['passed'] for c in checks)} / {len(checks)} checks passed"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())

