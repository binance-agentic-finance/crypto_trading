"""
交易信号策略模块

定义各种交易信号策略，可以组合使用factor中的因子
信号函数签名: signal_func(data, index, position, entry_price, entry_index, take_profit, stop_loss, check_periods) -> str
- 返回: 'buy'（买入）, 'sell'（卖出）, 'hold'（持有）
"""

# 导入所有信号策略
from .ma_signal import ma_signal, ma_cross_signal
from .factor_based_signal import factor_based_signal, multi_factor_signal, normalized_factor_signal

__all__ = [
    'ma_signal',
    'ma_cross_signal',
    'factor_based_signal',
    'multi_factor_signal',
    'normalized_factor_signal',
]

