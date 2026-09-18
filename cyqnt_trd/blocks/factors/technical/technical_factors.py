"""
技术类因子（源自 JoinQuant 聚宽因子库 · 技术因子）

将均线 / 布林 / MACD / MFI 类量价指标适配到 crypto K 线，转换为方向性因子。
返回: 1.0（看多）、-1.0（看空）、0.0（数据不足或中性）

对应 JoinQuant 因子 code:
    boll_up, boll_down, EMA5, EMAC10/12/20/26/120, MAC5/10/20/60/120,
    MACDC, MFI14
数据列: high_price, low_price, close_price, volume
"""

import pandas as pd
import numpy as np


def _ema(arr: 'np.ndarray', period: int) -> float:
    """标准 EMA，返回最后一个值。"""
    if len(arr) == 0:
        return float('nan')
    k = 2.0 / (period + 1.0)
    ema = arr[0]
    for x in arr[1:]:
        ema = x * k + ema * (1 - k)
    return ema


def boll_factor(data_slice: 'pd.DataFrame', period: int = 20,
                num_std: float = 2.0) -> float:
    """
    布林带因子（boll_up / boll_down，M=20）

    均值回归口径：收盘价跌破下轨看多，突破上轨看空。
    """
    if len(data_slice) < period:
        return 0.0
    closes = data_slice.iloc[-period:]['close_price'].values.astype(float)
    ma = np.mean(closes)
    std = np.std(closes, ddof=0)
    if std == 0:
        return 0.0
    upper = ma + num_std * std
    lower = ma - num_std * std
    price = closes[-1]
    if price < lower:
        return 1.0
    if price > upper:
        return -1.0
    return 0.0


def emac_factor(data_slice: 'pd.DataFrame', period: int = 10) -> float:
    """
    指数均线因子（EMA5, EMAC10/12/20/26/120）

    JoinQuant 定义 EMAC = EMA(period) / close。价格站上 EMA（close>EMA）看多，
    跌破看空。
    """
    if len(data_slice) < period:
        return 0.0
    closes = data_slice.iloc[-period:]['close_price'].values.astype(float)
    ema = _ema(closes, period)
    price = closes[-1]
    if np.isnan(ema) or ema == 0:
        return 0.0
    if price > ema:
        return 1.0
    if price < ema:
        return -1.0
    return 0.0


def mac_factor(data_slice: 'pd.DataFrame', period: int = 20) -> float:
    """
    简单均线因子（MAC5/10/20/60/120）

    JoinQuant 定义 MAC = MA(period) / close。价格站上 MA 看多，跌破看空。
    """
    if len(data_slice) < period:
        return 0.0
    closes = data_slice.iloc[-period:]['close_price'].values.astype(float)
    ma = np.mean(closes)
    price = closes[-1]
    if ma == 0:
        return 0.0
    if price > ma:
        return 1.0
    if price < ma:
        return -1.0
    return 0.0


def macdc_factor(data_slice: 'pd.DataFrame', short: int = 12,
                 long: int = 26, mid: int = 9) -> float:
    """
    MACD 因子（MACDC = MACD(12,26,9) / close）

    DIFF-DEA 得到的 MACD 柱 > 0 看多、< 0 看空。
    """
    need = long + mid
    if len(data_slice) < need:
        return 0.0
    closes = data_slice.iloc[-need:]['close_price'].values.astype(float)
    # 逐点 DIFF 序列
    diffs = []
    for i in range(long, len(closes) + 1):
        window = closes[:i]
        diffs.append(_ema(window[-short:], short) - _ema(window[-long:], long))
    diffs = np.array(diffs)
    if len(diffs) < mid:
        return 0.0
    dea = _ema(diffs[-mid:], mid)
    macd_hist = diffs[-1] - dea
    if macd_hist > 0:
        return 1.0
    if macd_hist < 0:
        return -1.0
    return 0.0


def mfi_factor(data_slice: 'pd.DataFrame', period: int = 14,
               oversold: float = 20.0, overbought: float = 80.0) -> float:
    """
    资金流量指标因子（MFI14）

    典型价 TP=(H+L+C)/3，资金流=TP*volume，按 TP 涨跌分正负流，
    MFI=100-100/(1+正/负)。低于 oversold 看多、高于 overbought 看空。
    """
    if len(data_slice) < period + 1:
        return 0.0
    seg = data_slice.iloc[-period - 1:]
    high = seg['high_price'].values.astype(float)
    low = seg['low_price'].values.astype(float)
    close = seg['close_price'].values.astype(float)
    vol = seg['volume'].values.astype(float)
    tp = (high + low + close) / 3.0
    mf = tp * vol
    pos, neg = 0.0, 0.0
    for i in range(1, len(tp)):
        if tp[i] > tp[i - 1]:
            pos += mf[i]
        elif tp[i] < tp[i - 1]:
            neg += mf[i]
    if neg == 0:
        mfi = 100.0
    else:
        mfi = 100.0 - 100.0 / (1.0 + pos / neg)
    if mfi < oversold:
        return 1.0
    if mfi > overbought:
        return -1.0
    return 0.0
