"""动量因子 — 委托 blocks.factor_votes（去重）。"""
from __future__ import annotations

from cyqnt_trd.blocks import factor_votes as fv

from ._delegate import last_vote
from ._delegate import to_blocks_df as _df


def momentum_factor(data_slice, period: int = 10) -> float:
    """委托 cyqnt_trd.blocks.momentum_vote，返回切片最后一根的方向票(1/-1/0)。"""
    return last_vote(fv.momentum_vote(_df(data_slice), period))
