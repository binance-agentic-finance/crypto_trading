"""Tier scorers — one file per tier so a future UI shows one card per
row in the strategy's "score card" table.

Each scorer function:  `score_<name>(signal_out, thr_cfg) -> tier_dict`
where tier_dict = `{'name': str, 'score': int, 'reason': str}`.
"""
from .ema_trend      import score_ema_trend
from .rsi_zone       import score_rsi_zone
from .macd_momentum  import score_macd_momentum
from .volatility     import score_volatility
from .resonance      import score_resonance

__all__ = [
    "score_ema_trend", "score_rsi_zone", "score_macd_momentum",
    "score_volatility", "score_resonance",
]
