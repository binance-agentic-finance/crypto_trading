"""Report-case runner — Phase 3B foundation.

Strategy cases (case_type=strategy_backtest) emit a `StrategySpec` and run
through `runtime.backtest`. Report cases (case_type in {scanner_report,
trade_plan_report}) emit a structured JSON report instead. They have a
different bundle shape:

    case.json
    preset.json
    runner.py        # build_report(spec, adapter, inputs=None) -> dict
    README.md
    fixtures/
      input.json     # optional external events/earnings input
      expected.json  # golden report payload for tests

This module provides the canonical loader and a fixture-mode helper so
that report cases can be exercised in unit tests and the CLI without any
network access. The actual report logic lives in each case's `runner.py`
under `cases/migrated/<case_id>/`.

The returned report payload always carries the following envelope fields:

    case_id            — case identifier (matches case.json)
    case_type          — copied from catalog
    generated_at       — ISO-8601 UTC timestamp
    inputs             — preset + optional external inputs that shaped the run
    signals            — list of {symbol, side, score, reason} (may be empty)
    summary            — human-readable summary string
    recommended_next_step — what the user / agent should do next

Individual cases may add extra fields (e.g. `tiers` for trade_plan_report).
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from ai_pro_trading_library.cases.catalog import load_catalog
from ai_pro_trading_library.cases.loader import cases_root


REPORT_CASE_TYPES = ("scanner_report", "trade_plan_report")


class CaseReportError(RuntimeError):
    """Raised when a case cannot be loaded or its report runner fails."""


@dataclass
class CaseReport:
    """Canonical report envelope returned by every report-case runner."""

    case_id: str
    case_type: str
    generated_at: str
    inputs: dict[str, Any]
    signals: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    recommended_next_step: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Flatten extras into the top-level payload for stable JSON shape.
        extras = payload.pop("extras", {})
        payload.update(extras)
        return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _catalog_entry(case_id: str) -> dict[str, Any]:
    catalog = load_catalog()
    for entry in catalog["cases"]:
        if entry["case_id"] == case_id:
            return entry
    raise CaseReportError(f"unknown case_id in catalog: {case_id}")


def _import_runner(case_dir: Path, case_id: str) -> Callable[..., Any]:
    runner_file = case_dir / "runner.py"
    if not runner_file.exists():
        raise CaseReportError(
            f"case {case_id} has no runner.py at {runner_file}; "
            "report cases require a build_report(spec, adapter, inputs=None) entrypoint"
        )
    module_name = f"_case_runner_{case_id.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, runner_file)
    if spec is None or spec.loader is None:
        raise CaseReportError(f"could not import runner module for case {case_id}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_report"):
        raise CaseReportError(
            f"case {case_id} runner.py is missing build_report(spec, adapter, inputs=None)"
        )
    return module.build_report


def _load_preset(case_dir: Path) -> dict[str, Any]:
    preset_file = case_dir / "preset.json"
    if not preset_file.exists():
        return {}
    return json.loads(preset_file.read_text())


def _load_fixture_inputs(case_dir: Path) -> dict[str, Any] | None:
    fixture = case_dir / "fixtures" / "input.json"
    if not fixture.exists():
        return None
    return json.loads(fixture.read_text())


def load_case_report(
    case_id: str,
    *,
    adapter: Any | None = None,
    inputs: Mapping[str, Any] | None = None,
    fixture: bool = False,
) -> dict[str, Any]:
    """Run a report case and return its payload as a dict.

    Parameters
    ----------
    case_id:
        Catalog case identifier.
    adapter:
        Optional data-source adapter passed verbatim to ``build_report``.
        Cases that need klines / funding / tickers should accept an
        adapter object; fixture-mode cases may ignore it.
    inputs:
        Optional external inputs (e.g. earnings calendar, macro events).
        When ``fixture=True`` and ``inputs`` is None, the case's
        ``fixtures/input.json`` is loaded if present.
    fixture:
        When True, no adapter is required and inputs default to the
        case's fixture file. Used by tests and the CLI ``--fixture`` flag.
    """
    entry = _catalog_entry(case_id)
    case_type = entry.get("case_type")
    if case_type not in REPORT_CASE_TYPES:
        raise CaseReportError(
            f"case {case_id} has case_type={case_type!r}; "
            f"load_case_report only handles {REPORT_CASE_TYPES}"
        )

    case_dir = cases_root() / case_id
    if not case_dir.exists():
        raise CaseReportError(f"case directory missing: {case_dir}")

    build_report = _import_runner(case_dir, case_id)
    preset = _load_preset(case_dir)

    resolved_inputs: dict[str, Any] = {}
    if inputs is not None:
        resolved_inputs.update(dict(inputs))
    elif fixture:
        fixture_inputs = _load_fixture_inputs(case_dir)
        if fixture_inputs:
            resolved_inputs.update(fixture_inputs)

    payload = build_report(preset=preset, adapter=adapter, inputs=resolved_inputs)
    if not isinstance(payload, dict):
        raise CaseReportError(
            f"case {case_id} build_report() must return dict, got {type(payload).__name__}"
        )

    # Wrap in canonical envelope while preserving any extra keys the case added.
    envelope = CaseReport(
        case_id=case_id,
        case_type=case_type,
        generated_at=_utc_now_iso(),
        inputs={"preset": preset, "external": resolved_inputs},
        signals=list(payload.pop("signals", [])),
        summary=str(payload.pop("summary", "")),
        recommended_next_step=str(payload.pop("recommended_next_step", "")),
        extras=payload,
    )
    return envelope.to_dict()


def list_report_case_ids() -> tuple[str, ...]:
    """Return all catalog case_ids whose case_type is a report type."""
    catalog = load_catalog()
    return tuple(
        c["case_id"]
        for c in catalog["cases"]
        if c.get("case_type") in REPORT_CASE_TYPES
    )


def runnable_report_case_ids() -> tuple[str, ...]:
    """Return report cases that have a runner.py on disk (i.e. executable)."""
    out = []
    for cid in list_report_case_ids():
        if (cases_root() / cid / "runner.py").exists():
            out.append(cid)
    return tuple(out)
