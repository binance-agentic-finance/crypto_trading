"""Standardized strategy case catalog."""

from ai_pro_trading_library.cases.catalog import list_case_ids, load_catalog
from ai_pro_trading_library.cases.loader import load_case_strategy
from ai_pro_trading_library.cases.runner import (
    CaseReport,
    CaseReportError,
    list_report_case_ids,
    load_case_report,
    runnable_report_case_ids,
)
from ai_pro_trading_library.cases.schema import CaseValidationError, validate_case

# Importing `migrated` triggers signal-builder registration for the 6
# library-migration regression cases. Production cases live as a
# separate skill — see docs/migration/cases-relocation.md.
from ai_pro_trading_library.cases import migrated  # noqa: F401

__all__ = [
    "CaseReport",
    "CaseReportError",
    "CaseValidationError",
    "list_case_ids",
    "list_report_case_ids",
    "load_case_report",
    "load_case_strategy",
    "load_catalog",
    "runnable_report_case_ids",
    "validate_case",
]
