"""Direction voters — one file per voting source.

Each voter is a function `vote_<source>(signal_out) -> Optional[str]`
returning "long" / "short" / None (abstain). Template collects them all
and hands to `decision.direction_vote`.
"""
from .ema_direction  import vote_ema_direction
from .macd_histogram import vote_macd_histogram
from .resonance_dir  import vote_resonance_direction

__all__ = [
    "vote_ema_direction",
    "vote_macd_histogram",
    "vote_resonance_direction",
]
