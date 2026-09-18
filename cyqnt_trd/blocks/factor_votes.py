"""向量化的方向票因子（``cyqnt_trd.blocks.factor_votes``）。

把"指标 + 阈值 → {-1, 0, 1} 方向票"这一常见模式做成**整段 Series** 的可复用 block：
``1.0`` 看多、``-1.0`` 看空、``0.0`` 中性，预热段为 ``NaN``。指标数学统一取自
:mod:`cyqnt_trd.blocks.indicators`（如 RSI 用 Wilder 平滑），因此这些票与 blocks 的其余
指标同源、且构造上无未来函数。

``trading_signal/factor`` 里的经典 TA 因子（``rsi_factor`` 等逐点、返回单值）现在委托到
这里：它们等价于 ``<name>_vote(df).iloc[-1]``。列名用 blocks 惯例的小写 OHLCV。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

__all__ = [
    "bounded_vote", "sign_vote",
    "ma_vote", "ma_cross_vote", "ema_vote", "ema_cross_vote", "momentum_vote",
    "rsi_vote", "stochastic_k_vote", "cci_vote", "williams_r_vote",
    "ultimate_oscillator_vote", "adx_vote", "awesome_oscillator_vote",
    "macd_level_vote", "bull_bear_power_vote",
]


def _close(data) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.astype(float)
    return data["close"].astype(float)


def bounded_vote(value: pd.Series, lower: float, upper: float) -> pd.Series:
    """超卖/超买型票：``value < lower`` → +1（看多），``value > upper`` → -1，其余 0；
    ``value`` 为 NaN 处保持 NaN。"""
    out = pd.Series(np.where(value < lower, 1.0, np.where(value > upper, -1.0, 0.0)),
                    index=value.index)
    return out.where(value.notna())


def sign_vote(value: pd.Series) -> pd.Series:
    """符号票：>0 → +1，<0 → -1，==0 → 0，NaN 保持 NaN。"""
    return pd.Series(np.sign(value), index=value.index).where(value.notna())


# --------------------------------------------------------------- 均线 / 动量
def ma_vote(data, period: int = 5) -> pd.Series:
    """收盘价 vs 前一根为止的 ``period`` 日 SMA：在上 +1，否则 -1。"""
    close = _close(data)
    ma = ind.sma(close, period).shift(1)
    return pd.Series(np.where(close > ma, 1.0, -1.0), index=close.index).where(ma.notna())


def _cross_vote(fast: pd.Series, slow: pd.Series) -> pd.Series:
    up = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
    down = (fast.shift(1) >= slow.shift(1)) & (fast < slow)
    out = pd.Series(np.where(up, 1.0, np.where(down, -1.0, 0.0)), index=fast.index)
    return out.where(fast.notna() & slow.notna() & fast.shift(1).notna() & slow.shift(1).notna())


def ma_cross_vote(data, short_period: int = 5, long_period: int = 20) -> pd.Series:
    """短/长 SMA 金叉 +1、死叉 -1、其余 0（均线取到前一根为止）。"""
    close = _close(data)
    return _cross_vote(ind.sma(close, short_period).shift(1), ind.sma(close, long_period).shift(1))


def ema_vote(data, period: int = 10) -> pd.Series:
    """收盘价 vs 前一根为止的 ``period`` 日 EMA：在上 +1，否则 -1。"""
    close = _close(data)
    e = ind.ema(close, period).shift(1)
    return pd.Series(np.where(close > e, 1.0, -1.0), index=close.index).where(e.notna())


def ema_cross_vote(data, short_period: int = 10, long_period: int = 20) -> pd.Series:
    close = _close(data)
    return _cross_vote(ind.ema(close, short_period).shift(1), ind.ema(close, long_period).shift(1))


def momentum_vote(data, period: int = 10) -> pd.Series:
    """动量 ``close - close[period]`` 的符号票。"""
    close = _close(data)
    return sign_vote(close - close.shift(period))


# ------------------------------------------------------------- 震荡类（超卖/超买）
def rsi_vote(data, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> pd.Series:
    return bounded_vote(ind.rsi(_close(data), period), oversold, overbought)


def stochastic_k_vote(df: pd.DataFrame, period: int = 14, k_smooth: int = 3,
                      oversold: float = 20.0, overbought: float = 80.0) -> pd.Series:
    k, _ = ind.stochastic(df, k_period=period, d_period=k_smooth, smooth_k=k_smooth)
    return bounded_vote(k, oversold, overbought)


def cci_vote(df: pd.DataFrame, period: int = 20,
             oversold: float = -100.0, overbought: float = 100.0) -> pd.Series:
    return bounded_vote(ind.cci(df, period), oversold, overbought)


def williams_r_vote(df: pd.DataFrame, period: int = 14,
                    oversold: float = -80.0, overbought: float = -20.0) -> pd.Series:
    return bounded_vote(ind.williams_r(df, period), oversold, overbought)


def ultimate_oscillator_vote(df: pd.DataFrame, period1: int = 7, period2: int = 14,
                             period3: int = 28, oversold: float = 30.0,
                             overbought: float = 70.0) -> pd.Series:
    high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_close = close.shift(1)
    true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
    true_high = pd.concat([high, prev_close], axis=1).max(axis=1)
    bp = close - true_low
    tr = (true_high - true_low).replace(0.0, np.nan)
    avg = {p: bp.rolling(p).sum() / tr.rolling(p).sum() for p in (period1, period2, period3)}
    uo = 100.0 * (4 * avg[period1] + 2 * avg[period2] + avg[period3]) / 7.0
    return bounded_vote(uo, oversold, overbought)


# ------------------------------------------------------------- 趋势 / 强弱
def adx_vote(df: pd.DataFrame, period: int = 14, adx_threshold: float = 25.0) -> pd.Series:
    adx, plus_di, minus_di = ind.adx(df, period)
    trend = np.where(plus_di > minus_di, 1.0, np.where(minus_di > plus_di, -1.0, 0.0))
    out = pd.Series(np.where(adx < adx_threshold, 0.0, trend), index=df.index)
    return out.where(adx.notna() & plus_di.notna() & minus_di.notna())


def awesome_oscillator_vote(df: pd.DataFrame) -> pd.Series:
    """AO 符号票（原逐点逻辑里的 rising/falling 精修项是冗余的，净结果等于 sign(AO)）。"""
    return sign_vote(ind.awesome_oscillator(df))


def macd_level_vote(data, fast_period: int = 12, slow_period: int = 26,
                    signal_period: int = 9) -> pd.Series:
    """MACD 相对信号线：柱状（macd − signal）的符号票。"""
    _, _, hist = ind.macd(_close(data), fast=fast_period, slow=slow_period, signal=signal_period)
    return sign_vote(hist)


def bull_bear_power_vote(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """Bull+Bear Power = (high-EMA) + (low-EMA) 的符号票。"""
    close, high, low = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    e = ind.ema(close, period)
    return sign_vote((high - e) + (low - e))
