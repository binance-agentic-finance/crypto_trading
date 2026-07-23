"""
ETH 市场状态自适应策略 - 第三轮优化
核心逻辑：根据市场状态（趋势/震荡）切换不同策略
- 趋势市：使用 MA 交叉 + MACD 确认
- 震荡市：使用布林带 + RSI 均值回归
结合：ADX 状态判断 + 时间过滤 + 波动率过滤
"""
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, regime, strategy

def make_signals(df):
    close = df["close"]
    
    # ===== 核心指标 =====
    # 趋势指标
    ma20 = ind.sma(close, 20)
    ma60 = ind.sma(close, 60)
    macd_line, signal_line, macd_hist = ind.macd(close, 6, 13, 5)
    
    # 震荡指标
    bb_upper, bb_mid, bb_lower = ind.bollinger(close, 20, 2.0)
    rsi14 = ind.rsi(close, 14)
    
    # 市场状态判断
    adx, plus_di, minus_di = ind.adx(df, 14)
    
    # 成交量
    vol_ma20 = ind.volume_ma(df, 20)
    
    # ===== 市场状态过滤 =====
    # ADX 判断趋势/震荡
    is_trending = cond.adx_trending(adx, threshold=25.0)
    is_ranging = cond.adx_ranging(adx, threshold=20.0)
    
    # 波动率过滤
    bb_width = (bb_upper - bb_lower) / bb_mid
    is_volatile = bb_width > 0.02
    
    # 时间过滤（UTC 13:00-21:00）
    timestamps_ms = df["close_time"]
    is_liquid_hours = cond.time_filter(timestamps_ms, start_hour=13, end_hour=21, tz_offset_hours=0)
    
    # ===== 趋势策略信号 =====
    # 趋势多头：MA 金叉 + MACD>0 + 趋势市
    trend_long = entry.all_of([
        cond.ma_cross_above(ma20, ma60),
        cond.macd_above_zero(macd_line),
        is_trending,
        is_volatile,
        is_liquid_hours,
    ])
    
    # 趋势空头：MA 死叉 + MACD<0 + 趋势市
    trend_short = entry.all_of([
        cond.ma_cross_below(ma20, ma60),
        cond.macd_below_zero(macd_line),
        is_trending,
        is_volatile,
        is_liquid_hours,
    ])
    
    # ===== 震荡策略信号 =====
    # 震荡多头：价格触及布林带下轨 + RSI 超卖
    range_long = entry.all_of([
        close <= bb_lower,
        cond.rsi_oversold(rsi14, threshold=30.0),
        is_ranging,
        is_volatile,
        is_liquid_hours,
    ])
    
    # 震荡空头：价格触及布林带上轨 + RSI 超买
    range_short = entry.all_of([
        close >= bb_upper,
        cond.rsi_overbought(rsi14, threshold=70.0),
        is_ranging,
        is_volatile,
        is_liquid_hours,
    ])
    
    # ===== 成交量确认（可选增强）=====
    is_volume_surge = cond.volume_surge(df, vol_ma20, multiplier=1.5)
    
    # ===== 自适应切换 =====
    # 使用 regime_switch 根据市场状态选择信号
    # 但这里我们简化：直接 OR 两种策略的信号
    long_signal = entry.any_of([
        entry.all_of([trend_long, is_volume_surge]),  # 趋势策略需要成交量确认
        range_long,  # 震荡策略不需要成交量确认（均值回归）
    ])
    
    short_signal = entry.any_of([
        entry.all_of([trend_short, is_volume_surge]),
        range_short,
    ])
    
    return long_signal, short_signal

strategy.register("eth_regime_adaptive_v3", make_signals)
