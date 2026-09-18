"""
风险类因子（源自 JoinQuant 聚宽因子库 · 风险因子）

将个股收益的统计矩（方差 / 偏度 / 峰度 / 夏普）适配到 crypto K 线，
转换为方向性交易因子。因子函数签名:
    factor_func(data_slice: pd.DataFrame, period: int = ...) -> float
返回: 1.0（看多）、-1.0（看空）、0.0（数据不足或中性）

对应 JoinQuant 因子 code:
    Variance20/60/120, Skewness20/60/120, Kurtosis20/60/120,
    sharpe_ratio_20/60/120
数据列: close_price
"""

import pandas as pd
import numpy as np


def _log_returns(data_slice: 'pd.DataFrame', period: int) -> 'np.ndarray':
    """取最近 period+1 根收盘价，算 period 个对数收益率。"""
    prices = data_slice.iloc[-period - 1:]['close_price'].values.astype(float)
    prices = prices[prices > 0]
    if len(prices) < 3:
        return np.array([])
    return np.diff(np.log(prices))


def variance_factor(data_slice: 'pd.DataFrame', period: int = 20,
                    calm_pct: float = 0.30, risk_pct: float = 0.70) -> float:
    """
    年化收益方差因子（Variance20/60/120）

    方差本身无方向，这里按"波动率状态"给方向：当前窗口方差落在自身历史的
    低分位（市场平静）看多，高分位（风险上升）看空。

    Args:
        data_slice: 至少需要 2*period+1 行
        period: 方差窗口（20/60/120）
        calm_pct: 低于该分位视为平静（看多）
        risk_pct: 高于该分位视为高风险（看空）
    """
    if len(data_slice) < 2 * period + 1:
        return 0.0
    rets = _log_returns(data_slice, 2 * period)
    if len(rets) < period:
        return 0.0
    # 滚动 period 方差序列，取分位
    var_series = pd.Series(rets).rolling(period).var().dropna().values
    if len(var_series) < 5:
        return 0.0
    cur = var_series[-1]
    lo = np.quantile(var_series, calm_pct)
    hi = np.quantile(var_series, risk_pct)
    if cur <= lo:
        return 1.0   # 波动收敛，风险偏好
    if cur >= hi:
        return -1.0  # 波动放大，规避
    return 0.0


def skewness_factor(data_slice: 'pd.DataFrame', period: int = 20,
                    threshold: float = 0.3) -> float:
    """
    收益偏度因子（Skewness20/60/120）

    右偏（正偏度）代表大涨尾部概率高，看多；左偏看空。
    """
    if len(data_slice) < period + 1:
        return 0.0
    rets = _log_returns(data_slice, period)
    if len(rets) < 3:
        return 0.0
    skew = pd.Series(rets).skew()
    if np.isnan(skew):
        return 0.0
    if skew > threshold:
        return 1.0
    if skew < -threshold:
        return -1.0
    return 0.0


def kurtosis_factor(data_slice: 'pd.DataFrame', period: int = 20,
                    threshold: float = 3.0) -> float:
    """
    收益峰度因子（Kurtosis20/60/120）

    高峰度=肥尾=极端行情概率高=尾部风险，看空；接近正态则中性偏多。
    （pandas.kurt 为超额峰度，正态≈0，这里以 threshold 判定肥尾。）
    """
    if len(data_slice) < period + 1:
        return 0.0
    rets = _log_returns(data_slice, period)
    if len(rets) < 4:
        return 0.0
    kurt = pd.Series(rets).kurt()
    if np.isnan(kurt):
        return 0.0
    if kurt > threshold:
        return -1.0  # 肥尾，尾部风险
    if kurt < 0.0:
        return 1.0   # 分布平坦，行情温和
    return 0.0


def sharpe_ratio_factor(data_slice: 'pd.DataFrame', period: int = 20,
                        rf: float = 0.04, bars_per_year: int = 365) -> float:
    """
    夏普比率因子（sharpe_ratio_20/60/120）

    (Rp - Rf) / Sigma_p。年化夏普为正看多、为负看空。

    Args:
        rf: 无风险利率（JoinQuant 设 0.04）
        bars_per_year: 年化系数（日线 365；换周期请调整）
    """
    if len(data_slice) < period + 1:
        return 0.0
    rets = _log_returns(data_slice, period)
    if len(rets) < 3:
        return 0.0
    mu = np.mean(rets) * bars_per_year
    sigma = np.std(rets, ddof=1) * np.sqrt(bars_per_year)
    if sigma == 0:
        return 0.0
    sharpe = (mu - rf) / sigma
    if sharpe > 0:
        return 1.0
    if sharpe < 0:
        return -1.0
    return 0.0
