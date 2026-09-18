"""Stochastic %K 因子 — 委托 blocks.factor_votes（去重）。"""
from __future__ import annotations

from cyqnt_trd.blocks import factor_votes as fv

from ._delegate import last_vote
from ._delegate import to_blocks_df as _df


def stochastic_k_factor(data_slice, period: int = 14, k_smooth: int = 3, d_smooth: int = 3, oversold: float = 20.0, overbought: float = 80.0) -> float:
    """委托 cyqnt_trd.blocks.stochastic_k_vote，返回切片最后一根的方向票(1/-1/0)。"""
    return last_vote(fv.stochastic_k_vote(_df(data_slice), period, k_smooth, oversold, overbought))
