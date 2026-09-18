"""Ultimate Oscillator 因子 — 委托 blocks.factor_votes（去重）。"""
from __future__ import annotations

from cyqnt_trd.blocks import factor_votes as fv

from ._delegate import last_vote
from ._delegate import to_blocks_df as _df


def uo_factor(data_slice, period1: int = 7, period2: int = 14, period3: int = 28, oversold: float = 30.0, overbought: float = 70.0) -> float:
    """委托 cyqnt_trd.blocks.ultimate_oscillator_vote，返回切片最后一根的方向票(1/-1/0)。"""
    return last_vote(fv.ultimate_oscillator_vote(_df(data_slice), period1, period2, period3, oversold, overbought))
