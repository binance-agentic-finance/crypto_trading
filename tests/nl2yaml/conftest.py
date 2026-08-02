"""Make ``tools.nl2yaml`` importable.

``cyqnt_trd`` is pip-installed editable in ``.venv-standard-bot``; ``tools`` is
not a distribution and never will be, so the repo root has to go on the path.
pytest does not add its rootdir by itself, and ``tests/`` has no ``__init__.py``,
so without this the test module's own directory is what lands on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
