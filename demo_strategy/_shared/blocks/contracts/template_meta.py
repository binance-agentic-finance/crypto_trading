"""TemplateMeta — metadata the control plane reads.

UI card: "strategy identity" — strategy_id, display_name, version,
author, description, config_schema (renders into a form).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TemplateMeta:
    strategy_id:   str                                # stable id; don't change once shipped
    display_name:  str
    description:   str = ""
    config_schema: dict[str, Any] = field(default_factory=dict)
    version:       str = "0.1.0"
    author:        str = ""
