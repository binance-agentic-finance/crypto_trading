"""
ETH 突破回踩策略 - 第二轮优化
核心逻辑：等待突破关键位 → 回踩确认 → 不破再入场
结合：市场状态过滤 + 时间过滤
"""
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, regime, strategy

def make_signals(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # ===== 核心指标 =====
    # 20 日高低点作为关键位
    donchian_upper, donchian_lower, donchian_mid = ind.donchian(df, 20)
    
    # MA20/60 趋势过滤
    ma20 = ind.sma(close, 20)
    ma60 = ind.sma(close, 60)
    
    # ADX 趋势强度
    adx, plus_di, minus_di = ind.adx(df, 14)
    
    # 布林带波动率
    bb_upper, bb_mid, bb_lower = ind.bollinger(close, 20, 2.0)
    
    # 成交量
    vol_ma20 = ind.volume_ma(df, 20)
    
    # RSI 超买超卖
    rsi14 = ind.rsi(close, 14)
    
    # ===== 市场状态过滤 =====
    is_trending = cond.adx_trending(adx, threshold=25.0)
    bb_width = (bb_upper - bb_lower) / bb_mid
    is_volatile = bb_width > 0.02
    
    # 时间过滤（UTC 13:00-21:00）
    timestamps_ms = df["close_time"]
    is_liquid_hours = cond.time_filter(timestamps_ms, start_hour=13, end_hour=21, tz_offset_hours=0)
    
    # ===== 突破信号 =====
    # 向上突破：收盘价 > Donchian 上轨
    breakout_long = cond.breakout_high(df, lookback=20)
    
    # 向下跌破：收盘价 < Donchian 下轨
    breakout_short = cond.breakout_low(df, lookback=20)
    
    # ===== 回踩确认信号 =====
    # 回踩多头：突破后，价格回踩到 donchian_upper 附近但不跌破
    # 简化实现：突破后 1-3 根 K 线，最低价触及上轨区域，收盘价仍在上轨之上
    retest_long = cond.price_bounce_ma(df, donchian_upper, direction="long")
    
    # 回踩空头：跌破后，价格反弹到 donchian_lower 附近但不突破
    retest_short = cond.price_bounce_ma(df, donchian_lower, direction="short")
    
    # ===== 趋势确认 =====
    ma_bullish = (ma20 > ma60) & cond.price_above_ma(df, ma20, bars=1)
    ma_bearish = (ma20 < ma60) & cond.price_below_ma(df, ma20, bars=1)
    
    # ===== 成交量确认 =====
    is_volume_surge = cond.volume_surge(df, vol_ma20, multiplier=1.5)
    
    # ===== 入场条件 =====
    # 多头：突破 + 回踩成功 + 趋势多头 + 趋势市 + 波动率足够
    long_breakout = entry.all_of([
        breakout_long,
        ma_bullish,
        is_trending,
        is_volatile,
        is_liquid_hours,
    ])
    
    long_retest = entry.all_of([
        retest_long,
        ma_bullish,
        is_trending,
        is_volatile,
        is_liquid_hours,
        is_volume_surge,
    ])
    
    # 空头：跌破 + 回踩成功 + 趋势空头 + 趋势市 + 波动率足够
    short_breakout = entry.all_of([
        breakout_short,
        ma_bearish,
        is_trending,
        is_volatile,
        is_liquid_hours,
    ])
    
    short_retest = entry.all_of([
        retest_short,
        ma_bearish,
        is_trending,
        is_volatile,
        is_liquid_hours,
        is_volume_surge,
    ])
    
    # ===== 返回信号 =====
    # 合并突破和回踩信号
    long_signal = long_breakout | long_retest
    short_signal = short_breakout | short_retest
    
    return long_signal, short_signal

strategy.register("eth_breakout_retest_v2", make_signals)
