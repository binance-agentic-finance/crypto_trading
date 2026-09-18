"""Awesome Oscillator 因子 — 委托 blocks.factor_votes（去重）。"""
from __future__ import annotations

from cyqnt_trd.blocks import factor_votes as fv

from ._delegate import last_vote
from ._delegate import to_blocks_df as _df


def ao_factor(data_slice) -> float:
    """委托 cyqnt_trd.blocks.awesome_oscillator_vote，返回切片最后一根的方向票(1/-1/0)。"""
    return last_vote(fv.awesome_oscillator_vote(_df(data_slice)))
