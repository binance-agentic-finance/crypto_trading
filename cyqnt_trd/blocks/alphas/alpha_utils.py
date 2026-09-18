"""
Alpha因子计算辅助函数模块

提供所有alpha因子计算所需的辅助函数，包括时间序列函数、统计函数等。
这些函数适配crypto交易数据格式（open_price, high_price, low_price, close_price, volume）。
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Union


def ts_sum(series: pd.Series, window: int = 10) -> pd.Series:
    """
    时间序列滚动求和
    
    Args:
        series: pandas Series
        window: 滚动窗口大小
    
    Returns:
        滚动求和结果
    """
    return series.rolling(window).sum()


def sma(series: pd.Series, window: int = 10) -> pd.Series:
    """
    简单移动平均
    
    Args:
        series: pandas Series
        window: 滚动窗口大小
    
    Returns:
        移动平均结果
    """
    return series.rolling(window).mean()


def stddev(series: pd.Series, window: int = 10) -> pd.Series:
    """
    滚动标准差
    
    Args:
        series: pandas Series
        window: 滚动窗口大小
    
    Returns:
        滚动标准差结果
    """
    return series.rolling(window).std()


def correlation(x: pd.Series, y: pd.Series, window: int = 10) -> pd.Series:
    """
    滚动相关系数
    
    Args:
        x: 第一个序列
        y: 第二个序列
        window: 滚动窗口大小
    
    Returns:
        滚动相关系数结果
    """
    return x.rolling(window).corr(y)


def covariance(x: pd.Series, y: pd.Series, window: int = 10) -> pd.Series:
    """
    滚动协方差
    
    Args:
        x: 第一个序列
        y: 第二个序列
        window: 滚动窗口大小
    
    Returns:
        滚动协方差结果
    """
    return x.rolling(window).cov(y)


def rolling_rank(na: np.ndarray) -> float:
    """
    辅助函数：用于rolling apply计算rank
    
    Args:
        na: numpy数组
    
    Returns:
        最后一个值的rank
    """
    if len(na) == 0:
        return np.nan
    return float(rankdata(na)[-1])


def ts_rank(series: pd.Series, window: int = 10) -> pd.Series:
    """
    时间序列滚动rank
    
    Args:
        series: pandas Series
        window: 滚动窗口大小
    
    Returns:
        滚动rank结果
    """
    return series.rolling(window).apply(rolling_rank, raw=True)


def rolling_prod(na: np.ndarray) -> float:
    """
    辅助函数：计算数组的乘积
    
    Args:
        na: numpy数组
    
    Returns:
        数组元素的乘积
    """
    return float(np.prod(na))


def product(series: pd.Series, window: int = 10) -> pd.Series:
    """
    滚动乘积
    
    Args:
        series: pandas Series
        window: 滚动窗口大小
    
    Returns:
        滚动乘积结果
    """
    return series.rolling(window).apply(rolling_prod, raw=True)


def ts_min(series: pd.Series, window: int = 10) -> pd.Series:
    """
    滚动最小值
    
    Args:
        series: pandas Series
        window: 滚动窗口大小
    
    Returns:
        滚动最小值结果
    """
    return series.rolling(window).min()


def ts_max(series: pd.Series, window: int = 10) -> pd.Series:
    """
    滚动最大值
    
    Args:
        series: pandas Series
        window: 滚动窗口大小
    
    Returns:
        滚动最大值结果
    """
    return series.rolling(window).max()


def delta(series: pd.Series, period: int = 1) -> pd.Series:
    """
    差分（当前值减去period期前的值）
    
    Args:
        series: pandas Series
        period: 差分周期
    
    Returns:
        差分结果
    """
    return series.diff(period)


def delay(series: pd.Series, period: int = 1) -> pd.Series:
    """
    滞后（shift）
    
    Args:
        series: pandas Series
        period: 滞后周期
    
    Returns:
        滞后结果
    """
    return series.shift(period)


def rank(series: pd.Series) -> pd.Series:
    """
    截面rank（百分比rank）
    
    Args:
        series: pandas Series
    
    Returns:
        百分比rank结果（0-1之间）
    """
    return series.rank(pct=True)


def scale(series: pd.Series, k: float = 1.0) -> pd.Series:
    """
    缩放时间序列，使得sum(abs(series)) = k
    
    Args:
        series: pandas Series
        k: 缩放因子
    
    Returns:
        缩放后的序列
    """
    abs_sum = series.abs().sum()
    if abs_sum == 0 or np.isnan(abs_sum):
        return series * 0
    return series.mul(k).div(abs_sum)


def ts_argmax(series: pd.Series, window: int = 10) -> pd.Series:
    """
    滚动窗口内最大值的索引位置（从1开始）
    
    Args:
        series: pandas Series
        window: 滚动窗口大小
    
    Returns:
        最大值索引位置（从1开始）
    """
    return series.rolling(window).apply(lambda x: np.argmax(x) + 1 if len(x) > 0 else np.nan, raw=True)


def ts_argmin(series: pd.Series, window: int = 10) -> pd.Series:
    """
    滚动窗口内最小值的索引位置（从1开始）
    
    Args:
        series: pandas Series
        window: 滚动窗口大小
    
    Returns:
        最小值索引位置（从1开始）
    """
    return series.rolling(window).apply(lambda x: np.argmin(x) + 1 if len(x) > 0 else np.nan, raw=True)


def decay_linear(series: pd.Series, period: int = 10) -> pd.Series:
    """
    线性加权移动平均（LWMA）
    
    Args:
        series: pandas Series
        period: LWMA周期
    
    Returns:
        LWMA结果
    """
    # 清理数据
    series_clean = series.ffill().bfill().fillna(0)
    
    result = pd.Series(index=series.index, dtype=float)
    
    divisor = period * (period + 1) / 2.0
    weights = np.arange(1, period + 1) / divisor
    
    for i in range(period - 1, len(series_clean)):
        window_data = series_clean.iloc[i - period + 1:i + 1].values
        if len(window_data) == period:
            result.iloc[i] = np.dot(window_data, weights)
        else:
            result.iloc[i] = np.nan
    
    return result


def sign(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    符号函数
    
    Args:
        x: 输入值或序列
    
    Returns:
        符号（-1, 0, 或 1）
    """
    return np.sign(x)


def abs(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    绝对值
    
    Args:
        x: 输入值或序列
    
    Returns:
        绝对值
    """
    return np.abs(x)


def log(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    自然对数
    
    Args:
        x: 输入值或序列
    
    Returns:
        自然对数
    """
    if isinstance(x, pd.Series):
        return x.apply(lambda v: np.log(v) if v > 0 else np.nan)
    else:
        return np.log(x) if x > 0 else np.nan


def signed_power(x: Union[float, pd.Series], power: float) -> Union[float, pd.Series]:
    """
    带符号的幂运算
    
    Args:
        x: 输入值或序列
        power: 幂次
    
    Returns:
        带符号的幂运算结果
    """
    if isinstance(x, pd.Series):
        return x.apply(lambda v: np.sign(v) * (np.abs(v) ** power))
    else:
        return np.sign(x) * (np.abs(x) ** power)

