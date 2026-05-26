"""shim — atomic.core.template_hint.

Verbatim copy of atomic's template-hint emitter (pure stdlib utility,
no atomic / cyqnt_trd dependencies).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

_HEADER = "=" * 100
_TEMPLATE_DIR = "usage_project_cases/_templates"


def emit_template_hint(
    template: str,
    data_path,
    summary_lines: Iterable[str],
    *,
    secondary_template: Optional[str] = None,
    stream=sys.stdout,
) -> None:
    """Print a structured agent-facing template hint.

    The block contains the primary template path, optional secondary template,
    and a compact summary. *data_path* is intentionally not printed because
    local result paths are internal runtime details, not user-facing content.
    """
    stream.write(_HEADER + "\n")
    stream.write("TEMPLATE HINT FOR AGENT\n")
    stream.write(_HEADER + "\n")
    stream.write(f"  primary_template:   {_TEMPLATE_DIR}/{template}\n")
    if secondary_template:
        stream.write(f"  secondary_template: {_TEMPLATE_DIR}/{secondary_template}\n")
    stream.write("  usage: render the user-facing report by filling the sections\n")
    stream.write("         of primary_template from the structured result data.\n")
    stream.write("  visibility: do not mention local JSON/Markdown paths in the final answer\n")
    stream.write("              unless the user explicitly asks for saved artifact paths.\n")
    stream.write("\nSUMMARY (for quick scan, not the final answer):\n")
    for line in summary_lines:
        stream.write(f"  {line}\n")
    stream.write(_HEADER + "\n")


__all__ = ["emit_template_hint"]
