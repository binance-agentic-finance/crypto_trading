"""Strategy-module loader for a separate process (paper daemon / backtest CLI).

The standard_bot entrypoints register external strategies by importing a
``--strategy-module``. A YAML strategy isn't a static module, so this loader
bridges the gap: on import it reads the spec path from the ``CYQNT_YAML_SPEC``
environment variable and registers it, exactly as if a hand-written strategy
module had called ``strategy.register(...)``.

Usage (what ``yaml_pipeline run`` prints for paper/live)::

    CYQNT_YAML_SPEC=/abs/path/strategy.yaml \\
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \\
        --engine python --strategy <id> \\
        --strategy-module cyqnt_trd.standard_bot.yaml_pipeline._daemon_loader ...
"""

from __future__ import annotations

import os

from .spec import register_from_yaml

_spec_path = os.environ.get("CYQNT_YAML_SPEC")
if _spec_path:
    register_from_yaml(_spec_path)
