"""
Alpha#23 因子

公式: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)

说明：
此alpha因子基于WorldQuant的101个alpha因子公式实现。
适配crypto交易数据格式（open_price, high_price, low_price, close_price, volume, quote_volume）。

标签：待补充
"""

import pandas as pd
import numpy as np
from typing import Optional
from .alpha_utils import (
    ts_sum, sma, stddev, correlation, covariance,
    ts_rank, product, ts_min, ts_max, delta, delay,
    rank, scale, ts_argmax, ts_argmin, decay_linear,
    sign, abs, log, signed_power
)


def alpha23_factor(
    data_slice: pd.DataFrame,
    **kwargs
) -> float:
    """
    Alpha#23 因子计算
    
    Args:
        data_slice: 数据切片，必须包含以下列：
                   - open_price: 开盘价
                   - high_price: 最高价
                   - low_price: 最低价
                   - close_price: 收盘价
                   - volume: 成交量
                   - quote_volume: 成交额（用于计算vwap）
        **kwargs: 其他可选参数
    
    Returns:
        因子值（最后一个时间点的值）
    """
    try:
        if len(data_slice) < 2:
            return 0.0
        
        # 提取数据列
        open_price = data_slice['open_price']
        high_price = data_slice['high_price']
        low_price = data_slice['low_price']
        close_price = data_slice['close_price']
        volume = data_slice['volume']
        quote_volume = data_slice.get('quote_volume', volume * close_price)  # 如果没有quote_volume，使用volume*close_price估算
        
        # 计算收益率
        returns = close_price.pct_change().fillna(0)
        
        # 计算VWAP (Volume Weighted Average Price)
        # vwap = quote_volume / volume，如果volume为0则使用close_price
        vwap = (quote_volume / (volume + 1e-10)).fillna(close_price)
        
        # 计算adv20 (20日平均成交量)
        adv20 = sma(volume, 20)
        
        # 实现Alpha因子逻辑
        cond = sma(high_price, 20) < high_price
        alpha = pd.Series(np.zeros_like(close_price), index=close_price.index)
        alpha[cond] = -1 * delta(high_price, 2).fillna(value=0)
        result = alpha
        
        # 返回最后一个值（如果是Series）或直接返回值
        if isinstance(result, pd.Series):
            result_value = result.iloc[-1] if len(result) > 0 else 0.0
        elif isinstance(result, (int, float, np.number)):
            result_value = float(result)
        else:
            result_value = 0.0
        
        # 处理NaN和无穷大
        if pd.isna(result_value) or np.isinf(result_value):
            return 0.0
        
        return float(result_value)
        
    except Exception as e:
        # 如果计算出错，返回0
        return 0.0
