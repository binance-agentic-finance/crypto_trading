"""Declarative YAML → standard_bot strategy pipeline.

A frontend (or bdp-ai-trading-bot) hands this a YAML spec that composes
``cyqnt_trd.blocks.*`` primitives into a strategy. The pipeline validates it
(static + dry-run) and registers it as a block strategy that runs identically
across backtest / paper / live via the existing standard_bot entrypoints.

Public API::

    from cyqnt_trd.standard_bot.yaml_pipeline import (
        load_spec, validate_spec, register_from_yaml, build_make_signals,
    )
"""

from __future__ import annotations

from .interpreter import SpecError, build_make_signals
from .spec import load_spec, register_from_yaml, validate_spec

__all__ = [
    "load_spec",
    "validate_spec",
    "register_from_yaml",
    "build_make_signals",
    "SpecError",
]
