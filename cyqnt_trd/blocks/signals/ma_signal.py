"""
基于移动平均线的交易信号策略

包含基于移动平均线的各种交易策略
"""

import pandas as pd


def ma_signal(
    data_slice: 'pd.DataFrame', 
    position: float, 
    entry_price: float, 
    entry_index: int,
    take_profit: float,
    stop_loss: float,
    period: int = 5
) -> str:
    """
    基于移动平均线的交易信号策略，带止盈止损
    
    当价格上穿MA时买入，下穿MA时卖出
    同时检查止盈止损条件
    
    注意：只使用当前周期的数据生成信号，不使用未来数据（无预知功能）
    
    Args:
        data_slice: 数据切片，必须包含至少 period+1 行数据
                   最后一行是当前数据点，前面 period+1 行是历史数据
        position: 当前持仓数量（如果没有持仓则为0）
        entry_price: 入场价格（如果没有持仓则为0）
        entry_index: 入场索引（如果没有持仓则为-1，保留用于兼容性）
        take_profit: 止盈比例（例如：0.1 表示 10%）
        stop_loss: 止损比例（例如：0.1 表示 10%）
        period: 移动平均周期（默认5）
    
    Returns:
        'buy', 'sell' 或 'hold'
    """
    if len(data_slice) < period + 1:
        return 'hold'
    
    # 最后一行是当前数据点
    current_price = data_slice.iloc[-1]['close_price']
    # 前period行是历史数据（不包括最后一行）
    ma = data_slice.iloc[-period-1:-1]['close_price'].mean()
    # 倒数第二行是上一周期的价格
    prev_price = data_slice.iloc[-2]['close_price']
    # 上一周期的MA（使用倒数第二行之前的数据）
    prev_ma = data_slice.iloc[-period-2:-2]['close_price'].mean()
    
    # 如果有持仓，先检查止盈止损
    if position > 0 and entry_price > 0:
        # 计算止盈和止损价格
        take_profit_price = entry_price * (1 + take_profit) if take_profit is not None else None
        stop_loss_price = entry_price * (1 - stop_loss) if stop_loss is not None else None
        
        # 只检查当前周期的数据，不检查未来数据
        current_low = data_slice.iloc[-1]['low_price']
        current_high = data_slice.iloc[-1]['high_price']
        
        # 优先检查止损
        if stop_loss_price is not None and current_low <= stop_loss_price:
            return 'sell'  # 触发止损
        
        # 检查止盈
        if take_profit_price is not None and current_high >= take_profit_price:
            return 'sell'  # 触发止盈
        
        # 如果没有触发止盈止损，检查策略信号
        # 下穿：之前价格在MA上方，现在在MA下方
        if prev_price >= prev_ma and current_price < ma:
            return 'sell'
        else:
            return 'hold'
    
    # 如果没有持仓，检查买入信号
    else:
        # 上穿：之前价格在MA下方，现在在MA上方
        if prev_price <= prev_ma and current_price > ma:
            return 'buy'
        else:
            return 'hold'


def ma_cross_signal(
    data_slice: 'pd.DataFrame', 
    position: float, 
    entry_price: float, 
    entry_index: int,
    take_profit: float,
    stop_loss: float,
    check_periods: int,
    short_period: int = 5,
    long_period: int = 20
) -> str:
    """
    基于移动平均线交叉的交易信号策略，带止盈止损
    
    当短期均线上穿长期均线时买入，下穿时卖出
    同时检查止盈止损条件
    
    注意：只使用当前周期的数据生成信号，不使用未来数据（无预知功能）
    
    Args:
        data_slice: 数据切片，必须包含至少 long_period+2 行数据
                   最后一行是当前数据点，前面 long_period+1 行是历史数据
        position: 当前持仓数量（如果没有持仓则为0）
        entry_price: 入场价格（如果没有持仓则为0）
        entry_index: 入场索引（如果没有持仓则为-1，保留用于兼容性）
        take_profit: 止盈比例（例如：0.1 表示 10%）
        stop_loss: 止损比例（例如：0.1 表示 10%）
        check_periods: 已弃用，不再使用（保留以保持向后兼容性）
        short_period: 短期均线周期（默认5）
        long_period: 长期均线周期（默认20）
    
    Returns:
        'buy', 'sell' 或 'hold'
    """
    if len(data_slice) < long_period + 2:
        return 'hold'
    
    # 计算当前周期的均线（使用最后period行数据，不包括最后一行）
    short_ma_current = data_slice.iloc[-short_period-1:-1]['close_price'].mean()
    long_ma_current = data_slice.iloc[-long_period-1:-1]['close_price'].mean()
    
    # 计算上一周期的均线（使用倒数第二行之前的数据）
    short_ma_prev = data_slice.iloc[-short_period-2:-2]['close_price'].mean()
    long_ma_prev = data_slice.iloc[-long_period-2:-2]['close_price'].mean()
    
    # 如果有持仓，先检查止盈止损
    if position > 0 and entry_price > 0:
        # 计算止盈和止损价格
        take_profit_price = entry_price * (1 + take_profit) if take_profit is not None else None
        stop_loss_price = entry_price * (1 - stop_loss) if stop_loss is not None else None
        
        # 只检查当前周期的数据，不检查未来数据
        current_low = data_slice.iloc[-1]['low_price']
        current_high = data_slice.iloc[-1]['high_price']
        
        # 优先检查止损
        if stop_loss_price is not None and current_low <= stop_loss_price:
            return 'sell'  # 触发止损
        
        # 检查止盈
        if take_profit_price is not None and current_high >= take_profit_price:
            return 'sell'  # 触发止盈
        
        # 如果没有触发止盈止损，检查策略信号
        # 下穿：之前短期均线在长期均线上方，现在在下方
        if short_ma_prev >= long_ma_prev and short_ma_current < long_ma_current:
            return 'sell'
        else:
            return 'hold'
    
    # 如果没有持仓，检查买入信号
    else:
        # 上穿：之前短期均线在长期均线下方，现在在上方
        if short_ma_prev <= long_ma_prev and short_ma_current > long_ma_current:
            return 'buy'
        else:
            return 'hold'

