"""
Runtime layer exports for the standard bot architecture.

Also hosts the two modules a compiled DAG strategy imports —
``data`` (catalog-backed fetch surface) and ``action`` (signal emission /
execution boundary) — so generated code targets the standard bot instead of
re-implementing fetching and order placement::

    from cyqnt_trd.standard_bot.runtime import data, action
"""

from . import action, data
from .interfaces import RunContext, Runner, TriggerAdapter
from .run_manager import LocalRunManager, ManagedRunRecord
from .runner import MarketOnlyPaperRunner

__all__ = [
    "LocalRunManager",
    "ManagedRunRecord",
    "MarketOnlyPaperRunner",
    "RunContext",
    "Runner",
    "TriggerAdapter",
    "action",
    "data",
]
