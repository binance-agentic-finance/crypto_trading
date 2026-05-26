"""Migrated case bundles — kept as library-migration regression tests.

These 6 cases (rsi-mean-reversion, bb-squeeze-momentum,
btc-multi-factor-trend, structure-breakout-priceaction,
stochrsi-mdi-trend-v1, funding-rate-scanner) bootstrapped the
ai_pro_trading_library migration. They live here so the smoke /
acceptance / parity tests still exercise the library end-to-end.

New end-to-end pipeline cases (square-buzz-screener, volume-surge-
daily-report, kline-indicator-checkup, etc.) ship as a separate skill
package, not bundled into the library — see
`docs/migration/cases-relocation.md`.

Case directories use hyphens (e.g. `rsi-mean-reversion`), which are not
valid Python identifiers. Importing this package triggers
`register_all_signals()` which loads every `<case>/signal.py` via importlib
and registers its `build_signal` factory with `default_registry`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ai_pro_trading_library.library.strategy import default_registry


_REGISTERED = False


def register_all_signals() -> tuple[str, ...]:
    """Import every migrated case's signal.py and register its factory.

    Idempotent: subsequent calls are no-ops.
    """
    global _REGISTERED
    if _REGISTERED:
        return default_registry.list_ids()

    cases_root = Path(__file__).resolve().parent
    registered: list[str] = []
    for case_dir in sorted(cases_root.iterdir()):
        if not case_dir.is_dir():
            continue
        signal_file = case_dir / "signal.py"
        if not signal_file.exists():
            continue
        module_name = f"_case_signal_{case_dir.name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, signal_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "build_signal"):
            continue
        factory = module.build_signal
        if not default_registry.has(case_dir.name):
            default_registry.register(case_dir.name, factory)
        registered.append(case_dir.name)
    _REGISTERED = True
    return tuple(registered)


register_all_signals()
