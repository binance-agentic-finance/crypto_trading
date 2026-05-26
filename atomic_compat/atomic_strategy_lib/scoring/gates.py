"""shim — atomic.scoring.gates (8 gates / verdict helpers)"""
from cyqnt_trd.blocks.verdicts import (  # noqa: F401
    hard_gate,
    enum_gate,
    soft_factor,
    verdict_classify,
    verdict_with_gate,
    cross_validate,
    conflict_detect,
    normalize_score,
)
