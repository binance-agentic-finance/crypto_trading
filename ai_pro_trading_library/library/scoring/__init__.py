"""Score combinators, gates, verdict helpers, and atomic-shape compat layer."""

from ai_pro_trading_library.library.scoring import atomic_compat
from ai_pro_trading_library.library.scoring.combinators import (
    additive_combine,
    envelope_score,
    weighted_composite,
    weighted_score,
)
from ai_pro_trading_library.library.scoring.gates import Gate, verdict_classify
from ai_pro_trading_library.library.scoring.rules import Rule, ScoringSystem, Verdict
from ai_pro_trading_library.library.scoring.signalize import signalize, signalize_series

__all__ = [
    "Gate",
    "Rule",
    "ScoringSystem",
    "Verdict",
    "additive_combine",
    "atomic_compat",
    "envelope_score",
    "signalize",
    "signalize_series",
    "verdict_classify",
    "weighted_composite",
    "weighted_score",
]
