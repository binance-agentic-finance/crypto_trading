"""Daemon variants — split by strategy kind so each has its own UI card.

    base.py             — shared lifecycle + signal handlers + state I/O
    signal_daemon.py    — single-symbol templates (calculate_signal)
    selection_daemon.py — cross-sectional templates (calculate_selection)

Import shortcut: `from demo_strategy._shared.bot import Daemon` still
returns SignalDaemon (backward-compat alias).
"""
from .base              import DaemonBase
from .signal_daemon     import SignalDaemon
from .selection_daemon  import SelectionDaemon

# Historical alias — some scripts import `Daemon` and expect the
# single-symbol variant. Keep this stable.
Daemon = SignalDaemon

__all__ = ["DaemonBase", "SignalDaemon", "SelectionDaemon", "Daemon"]
