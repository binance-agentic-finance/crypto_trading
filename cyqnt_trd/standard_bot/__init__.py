"""
Scaffold package for the standard trading bot architecture.

This package intentionally starts as a non-breaking sidecar to the existing
``cyqnt_trd`` modules. The current repository can keep using the legacy
modules while new functionality migrates into the protocol-first layers here.
"""

from . import core
from . import data
from . import execution
from . import monitoring
from . import orchestration
from . import runtime
from . import signal
from . import simulation

__all__ = [
    "core",
    "data",
    "execution",
    "monitoring",
    "orchestration",
    "runtime",
    "signal",
    "simulation",
]
