"""
cyqnt_trd.standard_bot.orchestration
======================================

Runtime orchestration utilities — workflow pipelines, scheduler loops, and
position/signal state CRUD.

These are runtime-level glue components (scheduler loops, state files, scan
pipelines) that belong in ``standard_bot`` rather than in ``blocks`` (which
is purely for pandas strategy logic).

Modules
-------
pipeline    Sequential workflow, two-stage scan/validate, scan-rank-select,
            candidate filter/rank.
scheduler   Fixed-interval loop, rescan pass, rescan loop.
state       JSON-based position CRUD and signal-log CRUD (JSONL).
"""

from .pipeline import (
    sequential_workflow,
    two_stage_pipeline,
    scan_rank_select,
    candidate_filter_rank,
    Verdict,
)
from .scheduler import (
    fixed_interval_loop,
    rescan_pass,
    rescan_loop,
)
from .state import (
    position_create,
    position_list,
    position_update,
    position_close,
    position_open_list,
    signal_log_append,
    signal_log_load,
    signal_log_save,
)

__all__ = [
    # pipeline
    "Verdict",
    "sequential_workflow",
    "two_stage_pipeline",
    "scan_rank_select",
    "candidate_filter_rank",
    # scheduler
    "fixed_interval_loop",
    "rescan_pass",
    "rescan_loop",
    # state
    "position_create",
    "position_list",
    "position_update",
    "position_close",
    "position_open_list",
    "signal_log_append",
    "signal_log_load",
    "signal_log_save",
]
