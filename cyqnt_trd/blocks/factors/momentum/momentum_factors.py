"""
动量类因子（源自 JoinQuant 聚宽因子库 · 动量因子）

将 Aroon / BBI / Elder-Ray / 乖离 / CCI / CR / ROC / TRIX / 线性回归斜率 /
价格位置 / 价量趋势 / 梅斯线等动量指标适配到 crypto K 线，转换为方向性因子。
返回: 1.0（看多）、-1.0（看空）、0.0（数据不足或中性）

对应 JoinQuant 因子 code:
    arron_up_25/arron_down_25, BBIC, bull_power/bear_power,
    BIAS5/10/20/60, CCI10/15/20/88, CR20, ROC6/12/20/60/120,
    TRIX5/10, PLRC6/12/24, Price1M/3M/1Y, fifty_two_week_close_rank,
    single_day_VPT(_6/_12), MASS
数据列: open_price, high_price, low_price, close_price, volume
"""

import pandas as pd
import numpy as np


def _ema_series(arr: 'np.ndarray', period: int) -> 'np.ndarray':
    k = 2.0 / (period + 1.0)
    out = np.empty(len(arr))
    if len(arr) == 0:
        return out
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def aroon_factor(data_slice: 'pd.DataFrame', period: int = 25) -> float:
    """
    Aroon 因子（arron_up_25 / arron_down_25）

    Aroon_up=(period-最高价距今天数)/period*100，Aroon_down 同理用最低价。
    up>down 上升动能占优看多，反之看空。
    """
    if len(data_slice) < period + 1:
        return 0.0
    seg = data_slice.iloc[-period - 1:]
    highs = seg['high_price'].values.astype(float)
    lows = seg['low_price'].values.astype(float)
    since_high = len(highs) - 1 - int(np.argmax(highs))
    since_low = len(lows) - 1 - int(np.argmin(lows))
    aroon_up = (period - since_high) / period * 100.0
    aroon_down = (period - since_low) / period * 100.0
    if aroon_up > aroon_down:
        return 1.0
    if aroon_up < aroon_down:
        return -1.0
    return 0.0


def bbi_factor(data_slice: 'pd.DataFrame') -> float:
    """
    BBI 动量因子（BBIC = BBI(3,6,12,24) / close）

    BBI=MA3/6/12/24 的均值（多空均线）。收盘价站上 BBI 看多，跌破看空。
    """
    if len(data_slice) < 24:
        return 0.0
    closes = data_slice['close_price'].values.astype(float)
    bbi = np.mean([np.mean(closes[-n:]) for n in (3, 6, 12, 24)])
    price = closes[-1]
    if price > bbi:
        return 1.0
    if price < bbi:
        return -1.0
    return 0.0


def bull_power_factor(data_slice: 'pd.DataFrame', period: int = 13) -> float:
    """
    多头力道因子（bull_power = (high - EMA(close,13)) / close）> 0 看多。
    """
    if len(data_slice) < period:
        return 0.0
    seg = data_slice.iloc[-period:]
    ema = _ema_series(seg['close_price'].values.astype(float), period)[-1]
    high = seg['high_price'].values.astype(float)[-1]
    close = seg['close_price'].values.astype(float)[-1]
    if close == 0:
        return 0.0
    bp = (high - ema) / close
    return 1.0 if bp > 0 else (-1.0 if bp < 0 else 0.0)


def bear_power_factor(data_slice: 'pd.DataFrame', period: int = 13) -> float:
    """
    空头力道因子（bear_power = (low - EMA(close,13)) / close）< 0 看空、> 0 看多。
    """
    if len(data_slice) < period:
        return 0.0
    seg = data_slice.iloc[-period:]
    ema = _ema_series(seg['close_price'].values.astype(float), period)[-1]
    low = seg['low_price'].values.astype(float)[-1]
    close = seg['close_price'].values.astype(float)[-1]
    if close == 0:
        return 0.0
    bp = (low - ema) / close
    return 1.0 if bp > 0 else (-1.0 if bp < 0 else 0.0)


def bias_factor(data_slice: 'pd.DataFrame', period: int = 20,
                threshold: float = 6.0) -> float:
    """
    乖离率因子（BIAS5/10/20/60）

    BIAS=(close-MA)/MA*100。均值回归口径：负乖离过大（超跌）看多，
    正乖离过大（超买）看空。threshold 随周期可调（短周期宜小）。
    """
    if len(data_slice) < period:
        return 0.0
    closes = data_slice.iloc[-period:]['close_price'].values.astype(float)
    ma = np.mean(closes)
    if ma == 0:
        return 0.0
    bias = (closes[-1] - ma) / ma * 100.0
    if bias < -threshold:
        return 1.0
    if bias > threshold:
        return -1.0
    return 0.0


def cci_period_factor(data_slice: 'pd.DataFrame', period: int = 20,
                      oversold: float = -100.0, overbought: float = 100.0) -> float:
    """
    顺势指标因子（CCI10/15/20/88）

    CCI=(TYP-MA(TYP,N))/(0.015*平均绝对偏差)，TYP=(H+L+C)/3。
    低于 oversold 看多、高于 overbought 看空。
    （与顶层 cci_factor.py 同族，这里提供多周期版本。）
    """
    if len(data_slice) < period:
        return 0.0
    seg = data_slice.iloc[-period:]
    tp = (seg['high_price'].values + seg['low_price'].values
          + seg['close_price'].values).astype(float) / 3.0
    ma = np.mean(tp)
    mad = np.mean(np.abs(tp - ma))
    if mad == 0:
        return 0.0
    cci = (tp[-1] - ma) / (0.015 * mad)
    if cci < oversold:
        return 1.0
    if cci > overbought:
        return -1.0
    return 0.0


def cr_factor(data_slice: 'pd.DataFrame', period: int = 20,
              strong: float = 200.0, weak: float = 50.0) -> float:
    """
    CR 指标因子（CR20）

    中间价=前一日(H+L)/2；上升值=今日H-昨中间价(负记0)，下跌值=昨中间价-今日L(负记0)；
    CR=多方强度/空方强度*100。CR 高（>strong）动能强看多，低（<weak）看空。
    """
    if len(data_slice) < period + 1:
        return 0.0
    seg = data_slice.iloc[-period - 1:]
    high = seg['high_price'].values.astype(float)
    low = seg['low_price'].values.astype(float)
    mid = (high + low) / 2.0
    up, dn = 0.0, 0.0
    for i in range(1, len(seg)):
        up += max(high[i] - mid[i - 1], 0.0)
        dn += max(mid[i - 1] - low[i], 0.0)
    if dn == 0:
        return 1.0
    cr = up / dn * 100.0
    if cr > strong:
        return 1.0
    if cr < weak:
        return -1.0
    return 0.0


def roc_factor(data_slice: 'pd.DataFrame', period: int = 12) -> float:
    """
    变动速率因子（ROC6/12/20/60/120）

    ROC=(close-close_n)/close_n*100。> 0 看多、< 0 看空。
    """
    if len(data_slice) < period + 1:
        return 0.0
    closes = data_slice['close_price'].values.astype(float)
    past = closes[-period - 1]
    if past == 0:
        return 0.0
    roc = (closes[-1] - past) / past * 100.0
    if roc > 0:
        return 1.0
    if roc < 0:
        return -1.0
    return 0.0


def trix_factor(data_slice: 'pd.DataFrame', period: int = 12) -> float:
    """
    TRIX 因子（TRIX5/10）

    对 close 求三次 EMA(period) 得 MTR，TRIX=(MTR-前一日MTR)/前一日MTR*100。
    > 0 看多、< 0 看空。
    """
    need = period * 3 + 2
    if len(data_slice) < need:
        return 0.0
    closes = data_slice['close_price'].values.astype(float)
    e1 = _ema_series(closes, period)
    e2 = _ema_series(e1, period)
    e3 = _ema_series(e2, period)
    if e3[-2] == 0:
        return 0.0
    trix = (e3[-1] - e3[-2]) / e3[-2] * 100.0
    if trix > 0:
        return 1.0
    if trix < 0:
        return -1.0
    return 0.0


def plrc_factor(data_slice: 'pd.DataFrame', period: int = 12) -> float:
    """
    收盘价-日期线性回归斜率因子（PLRC6/12/24）

    对 (close/mean(close)) 与序号 t(1..period) 做线性回归，斜率 beta > 0
    （上行趋势）看多、< 0 看空。
    """
    if len(data_slice) < period:
        return 0.0
    closes = data_slice.iloc[-period:]['close_price'].values.astype(float)
    m = np.mean(closes)
    if m == 0:
        return 0.0
    y = closes / m
    t = np.arange(1, period + 1, dtype=float)
    beta = np.polyfit(t, y, 1)[0]
    if beta > 0:
        return 1.0
    if beta < 0:
        return -1.0
    return 0.0


def price_position_factor(data_slice: 'pd.DataFrame', period: int = 21,
                          threshold: float = 0.0) -> float:
    """
    价格位置动量因子（Price1M=21 / Price3M=61 / Price1Y=250）

    Price = close / mean(close, period) - 1。> threshold（站上均值，趋势向上）
    看多、< -threshold 看空。
    """
    if len(data_slice) < period:
        return 0.0
    closes = data_slice.iloc[-period:]['close_price'].values.astype(float)
    m = np.mean(closes)
    if m == 0:
        return 0.0
    val = closes[-1] / m - 1.0
    if val > threshold:
        return 1.0
    if val < -threshold:
        return -1.0
    return 0.0


def price_rank_factor(data_slice: 'pd.DataFrame', period: int = 250,
                      high_pct: float = 0.8, low_pct: float = 0.2) -> float:
    """
    价格分位因子（fifty_two_week_close_rank，默认过去 250 根）

    当前收盘价在过去 period 根收盘价中的分位。处高位（≥high_pct，强势动量）看多，
    处低位（≤low_pct）看空。
    """
    if len(data_slice) < period:
        return 0.0
    closes = data_slice.iloc[-period:]['close_price'].values.astype(float)
    rank = float(np.mean(closes <= closes[-1]))  # 0~1，当前价所处分位
    if rank >= high_pct:
        return 1.0
    if rank <= low_pct:
        return -1.0
    return 0.0


def single_day_vpt_factor(data_slice: 'pd.DataFrame', period: int = 1) -> float:
    """
    单日价量趋势因子（single_day_VPT / _6 / _12）

    VPT=(close-昨close)/昨close*volume；period>1 取其 MA。均值 > 0（量价配合上行）
    看多、< 0 看空。
    """
    if len(data_slice) < period + 1:
        return 0.0
    seg = data_slice.iloc[-period - 1:]
    close = seg['close_price'].values.astype(float)
    vol = seg['volume'].values.astype(float)
    vpt = []
    for i in range(1, len(close)):
        if close[i - 1] == 0:
            vpt.append(0.0)
        else:
            vpt.append((close[i] - close[i - 1]) / close[i - 1] * vol[i])
    val = np.mean(vpt) if period > 1 else vpt[-1]
    if val > 0:
        return 1.0
    if val < 0:
        return -1.0
    return 0.0


def mass_factor(data_slice: 'pd.DataFrame', n1: int = 9, n2: int = 25,
                bulge: float = 27.0, trigger: float = 26.5) -> float:
    """
    梅斯线因子（MASS(N1=9,N2=25,M=6)）

    MASS=SUM(EMA(H-L,N1)/EMA(EMA(H-L,N1),N1), N2)。梅斯线形成"鼓包"（升破 bulge
    再跌回 trigger）预示趋势反转 → 看空反转警示；处低位则中性。
    """
    need = n2 + 2 * n1
    if len(data_slice) < need:
        return 0.0
    seg = data_slice.iloc[-need:]
    rng = (seg['high_price'].values - seg['low_price'].values).astype(float)
    ema1 = _ema_series(rng, n1)
    ema2 = _ema_series(ema1, n1)
    ratio = np.where(ema2 == 0, 0.0, ema1 / ema2)
    if len(ratio) < n2 + 1:
        return 0.0
    mass_now = np.sum(ratio[-n2:])
    mass_prev = np.sum(ratio[-n2 - 1:-1])
    # 鼓包回落：上一根在 bulge 上方、当前跌回 trigger 下方 → 反转看空
    if mass_prev >= bulge and mass_now < trigger:
        return -1.0
    return 0.0
