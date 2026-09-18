"""经典 TA 因子委托到 cyqnt_trd.blocks.factor_votes 的共用辅助。

去重后这些因子不再自算指标：把 `_price` 列名转成 blocks 惯用的小写，调用向量化的
`*_vote`，取切片最后一根的票。数值口径改用 blocks.indicators（如 RSI 用 Wilder 平滑），
与旧的手写实现可能有细微差异；票是 {-1, 0, 1} 粗粒度。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_RENAME = {"open_price": "open", "high_price": "high",
           "low_price": "low", "close_price": "close"}


def to_blocks_df(data_slice: pd.DataFrame) -> pd.DataFrame:
    """把 *_price 列改成 blocks 惯用的小写 OHLC（volume/quote_volume 本就小写）。"""
    return data_slice.rename(columns=_RENAME)


def last_vote(series: pd.Series) -> float:
    """取整段票的最后一根；缺失/预热记为 0.0（对齐旧因子的"数据不足→0"语义）。"""
    if series is None or len(series) == 0:
        return 0.0
    v = series.iloc[-1]
    return 0.0 if pd.isna(v) or not np.isfinite(v) else float(v)
