"""
基于因子的交易信号策略

展示如何在信号策略中使用factor中的因子
"""

from typing import Callable, Optional
import pandas as pd
import numpy as np

# 导入因子
from ..factors.ma_factor import ma_factor, ma_cross_factor
from ..factors.rsi_factor import rsi_factor


def factor_based_signal(
    data_slice: 'pd.DataFrame', 
    position: float, 
    entry_price: float, 
    entry_index: int,
    take_profit: float,
    stop_loss: float,
    check_periods: int,
    factor_func: Optional[Callable] = None,
    factor_period: int = 3
) -> str:
    """
    基于因子的交易信号策略，带止盈止损
    
    使用指定的因子函数生成交易信号
    当因子值从负转正时买入，从正转负时卖出
    
    Args:
        data_slice: 数据切片，必须包含足够的历史数据用于计算因子
                   以及未来check_periods个周期用于检查止盈止损
                   最后一行是当前数据点，前面是历史数据，后面是未来数据（如果有）
        position: 当前持仓数量（如果没有持仓则为0）
        entry_price: 入场价格（如果没有持仓则为0）
        entry_index: 入场索引（如果没有持仓则为-1，保留用于兼容性）
        take_profit: 止盈比例（例如：0.1 表示 10%）
        stop_loss: 止损比例（例如：0.1 表示 10%）
        check_periods: 检查未来多少个周期（用于止盈止损检查）
        factor_func: 因子函数，接受数据切片作为参数（如果为None，默认使用ma_factor）
        factor_period: 因子计算周期（用于某些因子）
    
    Returns:
        'buy', 'sell' 或 'hold'
    """
    # 默认使用ma_factor
    if factor_func is None:
        factor_func = lambda d: ma_factor(d, period=factor_period)
        # 当使用默认factor_func时，使用factor_period
        min_required = factor_period
    else:
        # 当传入自定义factor_func时，使用保守估计值（30）
        # 这样可以确保有足够的数据用于大多数因子计算（如MA5需要6行，RSI14需要15行，alpha1需要25+行）
        min_required = 30
    
    # 需要至少min_required+1行数据来计算当前因子，以及额外一行来计算上一周期的因子
    if len(data_slice) < min_required + 2:
        return 'hold'
    
    try:
        # 当前数据切片：最后一行是当前数据点，前面是历史数据
        # 使用足够的数据切片（min_required+1行）以确保因子函数有足够的数据
        current_slice = data_slice.iloc[-(min_required+1):]
        current_factor = factor_func(current_slice)
        
        # 如果因子返回0且数据足够，可能是数据不足导致的
        if current_factor == 0 and len(data_slice) > min_required + 1:
            # 不直接返回hold，继续尝试计算上一周期的因子
            pass
    except Exception:
        return 'hold'
    
    # 计算上一周期的因子值
    prev_factor = 0.0
    if len(data_slice) > min_required + 2:
        try:
            # 上一周期的数据切片：倒数第二行是上一周期的数据点
            # 使用足够的数据切片（min_required+1行）
            prev_slice = data_slice.iloc[-(min_required+2):-1]
            prev_factor = factor_func(prev_slice)
        except Exception:
            prev_factor = 0.0
    
    # 如果有持仓，先检查止盈止损
    if position > 0 and entry_price > 0:
        # 计算止盈和止损价格
        take_profit_price = entry_price * (1 + take_profit) if take_profit is not None else None
        stop_loss_price = entry_price * (1 - stop_loss) if stop_loss is not None else None
        
        # 检查当前周期和未来 check_periods 个周期
        # 当前周期是最后一行，未来周期在data_slice中（如果有）
        # 注意：如果check_periods > 1，data_slice应该包含未来数据
        check_end = min(check_periods, len(data_slice))
        for i in range(check_end):
            # 从最后一行开始，向前检查（如果data_slice包含未来数据）
            # 或者只检查当前周期（最后一行）
            if i == 0:
                # 当前周期（最后一行）
                check_low = data_slice.iloc[-1]['low_price']
                check_high = data_slice.iloc[-1]['high_price']
            else:
                # 未来周期（如果data_slice包含未来数据，应该从最后一行之后开始）
                # 但根据设计，data_slice应该只包含历史+当前，不包含未来
                # 所以这里只检查当前周期
                continue
            
            # 优先检查止损
            if stop_loss_price is not None and check_low <= stop_loss_price:
                return 'sell'  # 触发止损
            
            # 检查止盈
            if take_profit_price is not None and check_high >= take_profit_price:
                return 'sell'  # 触发止盈
        
        # 如果没有触发止盈止损，检查策略信号
        # 因子从正转负：卖出
        if prev_factor > 0 and current_factor < 0:
            return 'sell'
        else:
            return 'hold'
    
    # 如果没有持仓，检查买入信号
    else:
        # 因子从负转正：买入
        if prev_factor < 0 and current_factor > 0:
            return 'buy'
        else:
            return 'hold'


def multi_factor_signal(
    data_slice: 'pd.DataFrame', 
    position: float, 
    entry_price: float, 
    entry_index: int,
    take_profit: float,
    stop_loss: float,
    check_periods: int,
    factor_funcs: list = None,
    weights: list = None
) -> str:
    """
    多因子组合交易信号策略，带止盈止损
    
    组合多个因子，加权求和后生成交易信号
    
    Args:
        data_slice: 数据切片，必须包含足够的历史数据用于计算因子
                   以及未来check_periods个周期用于检查止盈止损
                   最后一行是当前数据点，前面是历史数据，后面是未来数据（如果有）
        position: 当前持仓数量（如果没有持仓则为0）
        entry_price: 入场价格（如果没有持仓则为0）
        entry_index: 入场索引（如果没有持仓则为-1，保留用于兼容性）
        take_profit: 止盈比例（例如：0.1 表示 10%）
        stop_loss: 止损比例（例如：0.1 表示 10%）
        check_periods: 检查未来多少个周期（用于止盈止损检查）
        factor_funcs: 因子函数列表，每个函数接受数据切片作为参数
                     （如果为None，默认使用[ma_factor, rsi_factor]）
        weights: 因子权重列表（如果为None，默认等权重）
    
    Returns:
        'buy', 'sell' 或 'hold'
    """
    # 默认使用ma_factor和rsi_factor
    if factor_funcs is None:
        factor_funcs = [
            lambda d: ma_factor(d, period=5),
            lambda d: rsi_factor(d, period=14)
        ]
    
    # 默认等权重
    if weights is None:
        weights = [1.0 / len(factor_funcs)] * len(factor_funcs)
    
    # 确保权重和因子数量一致
    if len(weights) != len(factor_funcs):
        weights = [1.0 / len(factor_funcs)] * len(factor_funcs)
    
    # 需要至少max_period+1行数据来计算因子
    # 假设最长的因子周期是20（ma_cross_factor的long_period）
    max_period = 20
    if len(data_slice) < max_period + 1:
        return 'hold'
    
    # 计算加权因子值
    try:
        # 当前数据切片：最后一行是当前数据点，前面是历史数据
        current_slice = data_slice.iloc[-(max_period+1):]
        combined_factor = 0.0
        for factor_func, weight in zip(factor_funcs, weights):
            factor_value = factor_func(current_slice)
            if factor_value is not None and not (isinstance(factor_value, float) and factor_value == 0 and len(data_slice) == 1):
                combined_factor += factor_value * weight
    except Exception:
        return 'hold'
    
    # 计算上一周期的组合因子值
    prev_combined_factor = 0.0
    if len(data_slice) > max_period + 1:
        try:
            # 上一周期的数据切片：倒数第二行是上一周期的数据点
            prev_slice = data_slice.iloc[-(max_period+2):-1]
            for factor_func, weight in zip(factor_funcs, weights):
                factor_value = factor_func(prev_slice)
                if factor_value is not None:
                    prev_combined_factor += factor_value * weight
        except Exception:
            prev_combined_factor = 0.0
    
    # 如果有持仓，先检查止盈止损
    if position > 0 and entry_price > 0:
        # 计算止盈和止损价格
        take_profit_price = entry_price * (1 + take_profit) if take_profit is not None else None
        stop_loss_price = entry_price * (1 - stop_loss) if stop_loss is not None else None
        
        # 检查当前周期（最后一行）
        check_low = data_slice.iloc[-1]['low_price']
        check_high = data_slice.iloc[-1]['high_price']
        
        # 优先检查止损
        if stop_loss_price is not None and check_low <= stop_loss_price:
            return 'sell'  # 触发止损
        
        # 检查止盈
        if take_profit_price is not None and check_high >= take_profit_price:
            return 'sell'  # 触发止盈
        
        # 如果没有触发止盈止损，检查策略信号
        # 组合因子从正转负：卖出
        if prev_combined_factor > 0 and combined_factor < 0:
            return 'sell'
        else:
            return 'hold'
    
    # 如果没有持仓，检查买入信号
    else:
        # 组合因子从负转正：买入
        if prev_combined_factor < 0 and combined_factor > 0:
            return 'buy'
        else:
            return 'hold'


import concurrent.futures

def normalized_factor_signal(
    data_slice: 'pd.DataFrame', 
    position: float, 
    entry_price: float, 
    entry_index: int,
    take_profit: float,
    stop_loss: float,
    check_periods: int,
    factor_func: Optional[Callable] = None,
    factor_period: int = 3,
    lookback_periods: int = 30
) -> str:
    """
    基于归一化因子的交易信号策略，带止盈止损
    
    计算当前周期和之前lookback_periods个周期的因子值，进行归一化后生成交易信号
    当归一化后的因子值从负转正时买入，从正转负时卖出
    
    Args:
        data_slice: 数据切片，必须包含足够的历史数据用于计算因子
                   以及未来check_periods个周期用于检查止盈止损
                   最后一行是当前数据点，前面是历史数据，后面是未来数据（如果有）
        position: 当前持仓数量（如果没有持仓则为0）
        entry_price: 入场价格（如果没有持仓则为0）
        entry_index: 入场索引（如果没有持仓则为-1，保留用于兼容性）
        take_profit: 止盈比例（例如：0.1 表示 10%）
        stop_loss: 止损比例（例如：0.1 表示 10%）
        check_periods: 检查未来多少个周期（用于止盈止损检查）
        factor_func: 因子函数，接受数据切片作为参数（如果为None，默认使用ma_factor）
        factor_period: 因子计算周期（用于某些因子）
        lookback_periods: 回看周期数，计算当前周期和之前多少个周期的因子值（默认30）
    
    Returns:
        'buy', 'sell' 或 'hold'
    """
    # 默认使用ma_factor
    if factor_func is None:
        factor_func = lambda d: ma_factor(d, period=factor_period)
        # 当使用默认factor_func时，使用factor_period
        min_required = factor_period
    else:
        # 当传入自定义factor_func时，使用保守估计值（30）
        # 这样可以确保有足够的数据用于大多数因子计算
        min_required = 30

    # 需要至少min_required+lookback_periods+1行数据
    # min_required用于计算因子，lookback_periods用于回看，+1用于当前周期
    # 如果数据不足，自适应调整回看周期
    available_periods = len(data_slice) - min_required - 1
    if available_periods < 2:
        return 'hold'  # 至少需要2个周期才能计算prev_factor和current_factor

    # 自适应调整回看周期，但不能小于2
    actual_lookback = min(lookback_periods, max(2, available_periods))

    try:
        # 利用多线程并行计算因子值
        def compute_factor_value(i):
            end_idx = len(data_slice) - i
            start_idx = max(0, end_idx - min_required - 1)
            if end_idx <= start_idx:
                return 0.0

            period_slice = data_slice.iloc[start_idx:end_idx]
            try:
                factor_value = factor_func(period_slice)
                if factor_value is not None:
                    return factor_value
                else:
                    return 0.0
            except Exception:
                return 0.0

        indices = list(range(actual_lookback + 1))
        factor_values = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 返回顺序保证和原for循环一致
            factor_values = list(executor.map(compute_factor_value, indices))

        # 如果收集到的因子值不足，返回hold
        if len(factor_values) < 2:
            return 'hold'

        # 将因子值转换为numpy数组进行归一化
        factor_array = np.array(factor_values)

        # Min-Max归一化：将值映射到[-1, 1]区间
        # 如果所有值都相同，则归一化后都为0
        factor_min = factor_array.min()
        factor_max = factor_array.max()

        if factor_max == factor_min:
            # 所有值相同，归一化后都为0
            normalized_factors = np.zeros_like(factor_array)
        else:
            # Min-Max归一化到[-1, 1]区间
            normalized_factors = 2 * (factor_array - factor_min) / (factor_max - factor_min) - 1

        # 当前归一化因子值（第一个，即当前周期）
        current_factor = normalized_factors[0]
        # 上一周期的归一化因子值（第二个，即前1个周期）
        prev_factor = normalized_factors[1] if len(normalized_factors) > 1 else 0.0

    except Exception:
        return 'hold'

    # 如果有持仓，先检查止盈止损
    if position > 0 and entry_price > 0:
        # 计算止盈和止损价格
        take_profit_price = entry_price * (1 + take_profit) if take_profit is not None else None
        stop_loss_price = entry_price * (1 - stop_loss) if stop_loss is not None else None

        # 检查当前周期（最后一行）
        check_low = data_slice.iloc[-1]['low_price']
        check_high = data_slice.iloc[-1]['high_price']

        # 优先检查止损
        if stop_loss_price is not None and check_low <= stop_loss_price:
            return 'sell'  # 触发止损

        # 检查止盈
        if take_profit_price is not None and check_high >= take_profit_price:
            return 'sell'  # 触发止盈

        # 如果没有触发止盈止损，检查策略信号
        # 归一化因子从正转负：卖出
        if prev_factor > 0 and current_factor < 0:
            return 'sell'
        else:
            return 'hold'

    # 如果没有持仓，检查买入信号
    else:
        # 归一化因子从负转正：买入
        if prev_factor < 0 and current_factor > 0:
            return 'buy'
        else:
            return 'hold'

