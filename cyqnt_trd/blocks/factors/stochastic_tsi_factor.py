"""
Stochastic TSI（随机TSI）因子

基于Stochastic TSI指标的交易因子
"""

import pandas as pd
import numpy as np


def _ema(data: np.ndarray, period: int) -> float:
    """计算指数移动平均"""
    if len(data) < period:
        return np.mean(data)
    
    alpha = 2.0 / (period + 1.0)
    ema = data[-period]
    
    for i in range(-period + 1, 0):
        ema = alpha * data[i] + (1 - alpha) * ema
    
    return ema


def stochastic_tsi_fast_factor(
    data_slice: 'pd.DataFrame',
    r_period: int = 3,
    s_period: int = 3,
    tsi_period1: int = 14,
    tsi_period2: int = 14,
    oversold: float = 20.0,
    overbought: float = 80.0
) -> float:
    """
    Stochastic TSI Fast因子
    
    当Stochastic TSI低于oversold时看多，高于overbought时看空
    
    Args:
        data_slice: 数据切片，必须包含足够的历史数据
                   最后一行是当前数据点
        r_period: %K平滑周期（默认3）
        s_period: %D平滑周期（默认3）
        tsi_period1: TSI第一个周期（默认14）
        tsi_period2: TSI第二个周期（默认14）
        oversold: 超卖阈值（默认20）
        overbought: 超买阈值（默认80）
    
    Returns:
        因子值：1.0（看多）、-1.0（看空）或 0（数据不足或中性）
    """
    # 需要足够的数据来计算TSI和Stochastic
    min_period = tsi_period1 + tsi_period2 + max(r_period, s_period) + 10
    
    if len(data_slice) < min_period + 1:
        return 0.0
    
    # 计算价格变化
    close_prices = data_slice['close_price'].values
    price_changes = np.diff(close_prices)
    
    # 计算TSI
    # TSI = 100 * (EMA(EMA(price_change, period1), period2) / EMA(EMA(abs(price_change), period1), period2))
    
    # 计算第一个EMA
    ema1_values = []
    ema1_abs_values = []
    
    for i in range(len(price_changes) - tsi_period1 + 1):
        subset = price_changes[i:i+tsi_period1]
        ema1 = _ema(subset, tsi_period1)
        ema1_values.append(ema1)
        
        abs_subset = np.abs(subset)
        ema1_abs = _ema(abs_subset, tsi_period1)
        ema1_abs_values.append(ema1_abs)
    
    if len(ema1_values) < tsi_period2:
        return 0.0
    
    # 计算第二个EMA
    ema2 = _ema(np.array(ema1_values[-tsi_period2:]), tsi_period2)
    ema2_abs = _ema(np.array(ema1_abs_values[-tsi_period2:]), tsi_period2)
    
    # 计算TSI
    if ema2_abs == 0:
        tsi = 0.0
    else:
        tsi = 100.0 * ema2 / ema2_abs
    
    # 计算Stochastic TSI
    # 需要r_period个TSI值来计算%K
    tsi_history = []
    for i in range(r_period):
        if len(ema1_values) < tsi_period2 + i:
            return 0.0
        
        ema2_val = _ema(np.array(ema1_values[-(tsi_period2 + i):-i if i > 0 else None]), tsi_period2)
        ema2_abs_val = _ema(np.array(ema1_abs_values[-(tsi_period2 + i):-i if i > 0 else None]), tsi_period2)
        
        if ema2_abs_val == 0:
            tsi_val = 0.0
        else:
            tsi_val = 100.0 * ema2_val / ema2_abs_val
        
        tsi_history.append(tsi_val)
    
    if len(tsi_history) < r_period:
        return 0.0
    
    # 计算%K（Stochastic of TSI）
    tsi_array = np.array(tsi_history)
    highest_tsi = np.max(tsi_array)
    lowest_tsi = np.min(tsi_array)
    current_tsi = tsi_history[-1]
    
    if highest_tsi == lowest_tsi:
        stoch_tsi = 50.0
    else:
        stoch_tsi = 100.0 * (current_tsi - lowest_tsi) / (highest_tsi - lowest_tsi)
    
    # 根据Stochastic TSI值返回因子
    if stoch_tsi < oversold:
        return 1.0  # 超卖，看多
    elif stoch_tsi > overbought:
        return -1.0  # 超买，看空
    else:
        return 0.0  # 中性

