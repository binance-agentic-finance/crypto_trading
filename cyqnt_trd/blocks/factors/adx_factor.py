"""ADX 因子 — 委托 blocks.factor_votes（去重，Wilder 口径）。"""
from __future__ import annotations

from cyqnt_trd.blocks import factor_votes as fv

from ._delegate import last_vote
from ._delegate import to_blocks_df as _df


def adx_factor(data_slice, period: int = 14, adx_threshold: float = 25.0) -> float:
    """委托 cyqnt_trd.blocks.adx_vote，返回切片最后一根的方向票(1/-1/0)。"""
    return last_vote(fv.adx_vote(_df(data_slice), period, adx_threshold))
