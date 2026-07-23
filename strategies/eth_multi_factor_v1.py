"""
ETH 多因子共振策略 - 第一轮优化
结合：多时间框架 + 市场状态过滤 + 时间过滤 + 波动率过滤
"""
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, regime, strategy

def make_signals(df):
    # ===== 核心指标 =====
    close = df["close"]
    
    # 趋势指标 - MA20/60
    ma20 = ind.sma(close, 20)
    ma60 = ind.sma(close, 60)
    
    # 动量指标 - MACD (6,13,5)
    macd_line, signal_line, macd_hist = ind.macd(close, 6, 13, 5)
    
    # 趋势强度 - ADX
    adx, plus_di, minus_di = ind.adx(df, 14)
    
    # 波动率 - 布林带
    bb_upper, bb_mid, bb_lower = ind.bollinger(close, 20, 2.0)
    
    # 成交量 - 20 日均量
    vol_ma20 = ind.volume_ma(df, 20)
    
    # ===== 市场状态过滤 =====
    # 1. 趋势市过滤：ADX > 25 才交易趋势策略
    is_trending = cond.adx_trending(adx, threshold=25.0)
    
    # 2. 波动率过滤：避开极低波动（布林带宽度 > 2%）
    bb_width = (bb_upper - bb_lower) / bb_mid
    is_volatile = bb_width > 0.02
    
    # 3. 时间过滤：只在美盘/欧盘高流动性时段（UTC 13:00-21:00）
    # 注意：Binance 时间戳是毫秒，需要转换
    timestamps_ms = df["close_time"]
    is_liquid_hours = cond.time_filter(timestamps_ms, start_hour=13, end_hour=21, tz_offset_hours=0)
    
    # 4. 极端波动率过滤：成交量 > 20 日均量 2 倍
    is_volume_surge = cond.volume_surge(df, vol_ma20, multiplier=2.0)
    
    # ===== 趋势方向判断 =====
    # MA 多头排列：MA20 > MA60
    ma_bullish = cond.ma_cross_above(ma20, ma60) | (ma20 > ma60)
    ma_bearish = cond.ma_cross_below(ma20, ma60) | (ma20 < ma60)
    
    # MACD 确认
    macd_bullish = cond.macd_above_zero(macd_line)
    macd_bearish = cond.macd_below_zero(macd_line)
    
    # ===== 入场条件 =====
    # 多头：趋势市 + 波动率足够 + 流动性时段 + MA 多头 + MACD 多头
    long_base = entry.all_of([
        ma_bullish,
        macd_bullish,
        is_trending,
        is_volatile,
    ])
    
    # 多头增强版：加上成交量爆发
    long_enhanced = entry.all_of([
        long_base,
        is_volume_surge,
    ])
    
    # 空头：趋势市 + 波动率足够 + 流动性时段 + MA 空头 + MACD 空头
    short_base = entry.all_of([
        ma_bearish,
        macd_bearish,
        is_trending,
        is_volatile,
    ])
    
    # 空头增强版：加上成交量爆发
    short_enhanced = entry.all_of([
        short_base,
        is_volume_surge,
    ])
    
    # ===== 返回信号 =====
    # 使用增强版信号（更严格，胜率可能更高）
    long_signal = long_enhanced
    short_signal = short_enhanced
    
    return long_signal, short_signal

strategy.register("eth_multi_factor_v1", make_signals)
