"""atomic_strategy_lib shim — thin re-export layer over cyqnt_trd.

This is NOT the original atomic_strategy_lib. It is a backward-compatibility
shim that lives next to cyqnt_trd and exposes the atomic public API surface
by delegating to the matching cyqnt_trd implementations.

Usage from a usage_project_case::

    # _paths.py only needs to add this directory's parent
    sys.path.insert(0, "<repo>/atomic_compat")

    # Then existing code works unchanged:
    from atomic_strategy_lib.signals.momentum import rsi_compute, rsi_current
    from atomic_strategy_lib.scoring.gates import verdict_with_gate
    # …etc

Internally these route to cyqnt_trd.compat.atomic_signals,
cyqnt_trd.blocks.verdicts, etc.
"""

__shim_for__ = "atomic_strategy_lib"
__shim_target__ = "cyqnt_trd"
