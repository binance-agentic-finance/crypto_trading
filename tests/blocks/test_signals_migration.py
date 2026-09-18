"""blocks.signals：从（已删除的）trading_signal.signal 迁入的有状态信号层。

签名 signal_func(data_slice, position, entry_price, entry_index, take_profit,
stop_loss, ...) -> 'buy' | 'sell' | 'hold'。这里只锁定迁移完整 + 可调用返回三态。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cyqnt_trd.blocks.factors import rsi_factor
from cyqnt_trd.blocks.signals import (
    factor_based_signal,
    ma_cross_signal,
    ma_signal,
    multi_factor_signal,
    normalized_factor_signal,
)


def _df(n=60):
    close = np.linspace(100.0, 120.0, n)
    return pd.DataFrame({"open_price": close, "high_price": close * 1.01,
                         "low_price": close * 0.99, "close_price": close,
                         "volume": np.full(n, 1e3), "quote_volume": close * 1e3})


def test_all_signal_functions_are_importable_from_blocks():
    for fn in (ma_signal, ma_cross_signal, factor_based_signal,
               multi_factor_signal, normalized_factor_signal):
        assert callable(fn)


def test_ma_signal_returns_a_three_valued_action():
    out = ma_signal(_df(), position=0.0, entry_price=0.0, entry_index=0,
                    take_profit=0.1, stop_loss=0.05, period=5)
    assert out in ("buy", "sell", "hold")


def test_factor_based_signal_drives_off_a_blocks_factor():
    out = factor_based_signal(_df(), position=0.0, entry_price=0.0, entry_index=0,
                              take_profit=0.1, stop_loss=0.05, check_periods=1,
                              factor_func=rsi_factor)
    assert out in ("buy", "sell", "hold")
