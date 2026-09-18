"""
情绪类因子（源自 JoinQuant 聚宽因子库 · 情绪因子 · 成交量/成交额子集）

情绪因子中大量为"换手率"类（VOL5/10/20…、DAVOL、turnover_volatility），
换手率 = 成交量 / 流通股本，crypto 无流通股本口径，故这些不实现。
本模块仅实现可由 OHLCV + 成交额(quote_volume) 计算的成交量/情绪因子。
返回: 1.0（看多）、-1.0（看空）、0.0（数据不足或中性）

对应 JoinQuant 因子 code:
    VROC6/12, VOSC, VSTD10/20, TVMA6/20, TVSTD6/20, VMACD, VR, PSY,
    AR, BR, ARBR, WVAD, MAWVAD, money_flow_20, ATR6/14
数据列: open_price, high_price, low_price, close_price, volume, quote_volume
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


def vroc_factor(data_slice: 'pd.DataFrame', period: int = 12,
                threshold: float = 0.0) -> float:
    """
    量变动速率因子（VROC6/12）

    VROC=(vol-vol_n)/vol_n*100，配合价格方向：放量且价升看多、放量且价跌看空。
    单纯缩量给中性。
    """
    if len(data_slice) < period + 1:
        return 0.0
    seg = data_slice.iloc[-period - 1:]
    vol = seg['volume'].values.astype(float)
    close = seg['close_price'].values.astype(float)
    if vol[0] == 0:
        return 0.0
    vroc = (vol[-1] - vol[0]) / vol[0] * 100.0
    price_up = close[-1] > close[0]
    if vroc > threshold:
        return 1.0 if price_up else -1.0
    return 0.0


def vosc_factor(data_slice: 'pd.DataFrame', short: int = 12, long: int = 26) -> float:
    """
    成交量震荡因子（VOSC = (VEMA_short - VEMA_long)/VEMA_short * 100）

    短期量能均值高于长期（量能扩张）配合价格方向给信号。
    """
    if len(data_slice) < long:
        return 0.0
    seg = data_slice.iloc[-long:]
    vol = seg['volume'].values.astype(float)
    close = seg['close_price'].values.astype(float)
    vshort = _ema_series(vol, short)[-1]
    vlong = _ema_series(vol, long)[-1]
    if vshort == 0:
        return 0.0
    vosc = (vshort - vlong) / vshort * 100.0
    if vosc > 0:  # 量能扩张
        return 1.0 if close[-1] >= close[0] else -1.0
    return 0.0


def vstd_factor(data_slice: 'pd.DataFrame', period: int = 10,
                risk_pct: float = 0.8) -> float:
    """
    成交量标准差因子（VSTD10/20）

    量能波动放大代表情绪不稳/风险上升：当前量标准差处于自身高分位看空。
    """
    if len(data_slice) < 2 * period:
        return 0.0
    vol = data_slice['volume'].values.astype(float)
    vstd = pd.Series(vol).rolling(period).std().dropna().values
    if len(vstd) < 5:
        return 0.0
    if vstd[-1] >= np.quantile(vstd, risk_pct):
        return -1.0
    return 0.0


def tvma_factor(data_slice: 'pd.DataFrame', period: int = 6) -> float:
    """
    成交金额移动平均因子（TVMA6/20，用 quote_volume 作成交额）

    当日成交额高于其 period 均值（放量）配合价格方向给信号。
    """
    if len(data_slice) < period:
        return 0.0
    seg = data_slice.iloc[-period:]
    amt = seg['quote_volume'].values.astype(float)
    close = seg['close_price'].values.astype(float)
    ma = np.mean(amt)
    if ma == 0:
        return 0.0
    if amt[-1] > ma:
        return 1.0 if close[-1] >= close[0] else -1.0
    return 0.0


def tvstd_factor(data_slice: 'pd.DataFrame', period: int = 6,
                 risk_pct: float = 0.8) -> float:
    """
    成交金额标准差因子（TVSTD6/20）

    成交额波动放大代表情绪不稳：处自身高分位看空。
    """
    if len(data_slice) < 2 * period:
        return 0.0
    amt = data_slice['quote_volume'].values.astype(float)
    tvstd = pd.Series(amt).rolling(period).std().dropna().values
    if len(tvstd) < 5:
        return 0.0
    if tvstd[-1] >= np.quantile(tvstd, risk_pct):
        return -1.0
    return 0.0


def vmacd_factor(data_slice: 'pd.DataFrame', short: int = 12,
                 long: int = 26, mid: int = 9) -> float:
    """
    量能 MACD 因子（VMACD）

    对成交量求 EMA12-EMA26 得 VDIFF，VDIFF 的 M 日 EMA 得 VDEA，
    VMACD=VDIFF-VDEA。量能动能 > 0 配合价格方向给信号。
    """
    need = long + mid
    if len(data_slice) < need:
        return 0.0
    seg = data_slice.iloc[-need:]
    vol = seg['volume'].values.astype(float)
    close = seg['close_price'].values.astype(float)
    vdiff = []
    for i in range(long, len(vol) + 1):
        w = vol[:i]
        vdiff.append(_ema_series(w[-short:], short)[-1] - _ema_series(w[-long:], long)[-1])
    vdiff = np.array(vdiff)
    if len(vdiff) < mid:
        return 0.0
    vdea = _ema_series(vdiff[-mid:], mid)[-1]
    vmacd = vdiff[-1] - vdea
    if vmacd > 0:
        return 1.0 if close[-1] >= close[0] else -1.0
    return 0.0


def vr_factor(data_slice: 'pd.DataFrame', period: int = 26,
              strong: float = 160.0, weak: float = 40.0) -> float:
    """
    成交量比率因子（VR = (AVS+0.5*CVS)/(BVS+0.5*CVS)*100）

    AVS/BVS/CVS 分别为上涨/下跌/平盘日成交量之和。VR 高（>strong）人气旺看多，
    低（<weak）看空。
    """
    if len(data_slice) < period + 1:
        return 0.0
    seg = data_slice.iloc[-period - 1:]
    close = seg['close_price'].values.astype(float)
    vol = seg['volume'].values.astype(float)
    avs = bvs = cvs = 0.0
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            avs += vol[i]
        elif close[i] < close[i - 1]:
            bvs += vol[i]
        else:
            cvs += vol[i]
    denom = bvs + 0.5 * cvs
    if denom == 0:
        return 1.0
    vr = (avs + 0.5 * cvs) / denom * 100.0
    if vr > strong:
        return 1.0
    if vr < weak:
        return -1.0
    return 0.0


def psy_factor(data_slice: 'pd.DataFrame', period: int = 12,
               high: float = 75.0, low: float = 25.0) -> float:
    """
    心理线因子（PSY = period 内上涨天数/period*100）

    均值回归口径：PSY 过高（超买）看空，过低（超卖）看多。
    """
    if len(data_slice) < period + 1:
        return 0.0
    closes = data_slice.iloc[-period - 1:]['close_price'].values.astype(float)
    up_days = int(np.sum(np.diff(closes) > 0))
    psy = up_days / period * 100.0
    if psy <= low:
        return 1.0
    if psy >= high:
        return -1.0
    return 0.0


def ar_factor(data_slice: 'pd.DataFrame', period: int = 26,
              strong: float = 180.0, weak: float = 70.0) -> float:
    """
    人气指标因子（AR = SUM(H-O)/SUM(O-L)*100，N=26）

    AR 高人气过热看空，AR 低人气低迷看多（均值回归）。
    """
    if len(data_slice) < period:
        return 0.0
    seg = data_slice.iloc[-period:]
    high = seg['high_price'].values.astype(float)
    low = seg['low_price'].values.astype(float)
    open_ = seg['open_price'].values.astype(float)
    up = np.sum(high - open_)
    dn = np.sum(open_ - low)
    if dn == 0:
        return -1.0
    ar = up / dn * 100.0
    if ar >= strong:
        return -1.0
    if ar <= weak:
        return 1.0
    return 0.0


def br_factor(data_slice: 'pd.DataFrame', period: int = 26,
              strong: float = 300.0, weak: float = 50.0) -> float:
    """
    意愿指标因子（BR = SUM(H-昨C)/SUM(昨C-L)*100，N=26）

    BR 过高看空、过低看多（均值回归）。
    """
    if len(data_slice) < period + 1:
        return 0.0
    seg = data_slice.iloc[-period - 1:]
    high = seg['high_price'].values.astype(float)
    low = seg['low_price'].values.astype(float)
    close = seg['close_price'].values.astype(float)
    up = dn = 0.0
    for i in range(1, len(seg)):
        up += max(high[i] - close[i - 1], 0.0)
        dn += max(close[i - 1] - low[i], 0.0)
    if dn == 0:
        return -1.0
    br = up / dn * 100.0
    if br >= strong:
        return -1.0
    if br <= weak:
        return 1.0
    return 0.0


def arbr_factor(data_slice: 'pd.DataFrame', period: int = 26,
                threshold: float = 30.0) -> float:
    """
    ARBR 因子（AR - BR）

    AR、BR 背离度。AR 明显高于 BR（AR-BR>threshold）人气过热看空，
    明显低于（<-threshold）看多。
    """
    if len(data_slice) < period + 1:
        return 0.0
    seg = data_slice.iloc[-period - 1:]
    high = seg['high_price'].values.astype(float)
    low = seg['low_price'].values.astype(float)
    open_ = seg['open_price'].values.astype(float)
    close = seg['close_price'].values.astype(float)
    ar_up = np.sum(high[1:] - open_[1:])
    ar_dn = np.sum(open_[1:] - low[1:])
    br_up = br_dn = 0.0
    for i in range(1, len(seg)):
        br_up += max(high[i] - close[i - 1], 0.0)
        br_dn += max(close[i - 1] - low[i], 0.0)
    ar = ar_up / ar_dn * 100.0 if ar_dn else 0.0
    br = br_up / br_dn * 100.0 if br_dn else 0.0
    diff = ar - br
    if diff > threshold:
        return -1.0
    if diff < -threshold:
        return 1.0
    return 0.0


def wvad_factor(data_slice: 'pd.DataFrame', period: int = 6) -> float:
    """
    威廉变异离散量因子（WVAD / MAWVAD）

    WVAD=SUM((close-open)/(high-low)*volume, period)。> 0 资金净流入看多、< 0 看空。
    period 增大即取更长均值（近似 MAWVAD）。
    """
    if len(data_slice) < period:
        return 0.0
    seg = data_slice.iloc[-period:]
    open_ = seg['open_price'].values.astype(float)
    close = seg['close_price'].values.astype(float)
    high = seg['high_price'].values.astype(float)
    low = seg['low_price'].values.astype(float)
    vol = seg['volume'].values.astype(float)
    rng = high - low
    wvad = np.sum(np.where(rng == 0, 0.0, (close - open_) / rng * vol))
    if wvad > 0:
        return 1.0
    if wvad < 0:
        return -1.0
    return 0.0


def money_flow_factor(data_slice: 'pd.DataFrame', period: int = 20) -> float:
    """
    资金流量因子（money_flow_20）

    资金流=典型价(H+L+C)/3 * volume。近半窗资金流均值高于前半窗（资金流入增强）
    配合价格方向给信号。
    """
    if len(data_slice) < period:
        return 0.0
    seg = data_slice.iloc[-period:]
    tp = (seg['high_price'].values + seg['low_price'].values
          + seg['close_price'].values).astype(float) / 3.0
    vol = seg['volume'].values.astype(float)
    mf = tp * vol
    half = period // 2
    recent = np.mean(mf[-half:])
    prev = np.mean(mf[:half])
    close = seg['close_price'].values.astype(float)
    if recent > prev:
        return 1.0 if close[-1] >= close[0] else -1.0
    return 0.0


def atr_factor(data_slice: 'pd.DataFrame', period: int = 14,
               risk_pct: float = 0.8) -> float:
    """
    均幅指标因子（ATR6/14）

    ATR=真实振幅的 period 日均值。振幅处自身高分位代表波动/风险抬升看空，
    低分位（行情平稳）看多。
    """
    if len(data_slice) < 2 * period + 1:
        return 0.0
    seg = data_slice
    high = seg['high_price'].values.astype(float)
    low = seg['low_price'].values.astype(float)
    close = seg['close_price'].values.astype(float)
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])
    atr = pd.Series(tr).rolling(period).mean().dropna().values
    if len(atr) < 5:
        return 0.0
    hi = np.quantile(atr, risk_pct)
    lo = np.quantile(atr, 1 - risk_pct)
    if atr[-1] >= hi:
        return -1.0
    if atr[-1] <= lo:
        return 1.0
    return 0.0
