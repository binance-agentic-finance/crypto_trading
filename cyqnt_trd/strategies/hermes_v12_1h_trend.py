"""
爱马仕 v12.0 - 1 小时趋势策略

策略逻辑:
1. ADX ≥ 25 (强趋势确认)
2. MA20 > MA50 > MA200 (多头排列)
3. SuperTrend + SAR 同向确认
4. 仓位 10% (信号少可重仓)
5. 最小持仓 6 小时

风控规则 (由框架处理):
- 止损: 5ATR
- 止盈: 移动止盈 3ATR 启动
- 持仓时间: 最小 6 小时
"""

from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, strategy

def make_signals(df):
    # 趋势确认指标
    ma20 = ind.sma(df["close"], 20)
    ma50 = ind.sma(df["close"], 50)
    ma200 = ind.sma(df["close"], 200)
    
    # ADX 趋势强度
    adx, plus_di, minus_di = ind.adx(df, period=14)
    
    # SuperTrend
    supertrend_value, supertrend_dir = ind.supertrend(df, period=10, multiplier=3.0)
    
    # Parabolic SAR (用 Donchian 通道近似)
    donchian_upper, donchian_lower, donchian_mid = ind.donchian(df, period=20)
    
    # 多头排列条件
    ma_bullish = entry.all_of([
        df["close"] > ma20,
        ma20 > ma50,
        ma50 > ma200,
    ])
    
    # 强趋势条件
    strong_trend = cond.adx_trending(adx, threshold=25.0)
    
    # SuperTrend 向上
    supertrend_bullish = supertrend_dir == 1
    
    # 价格在 Donchian 通道中上部 (近似 SAR 支撑)
    = cond.price_above_ma(df, donchian_mid, bars=1)
    
    # 多头进场: 所有条件 AND
    long = entry.all_of([
        ma_bullish,
        strong_trend,
        supertrend_bullish,
        price_above_donchian_mid,
    ])
    
    # 空头进场: MA 空头排列 + 强趋势
    ma_bearish = entry.all_of([
        df["close"] < ma20,
        ma20 < ma50,
        ma50 < ma200,
    ])
    
    supertrend_bearish = supertrend_dir == -1
    
    short = entry.all_of([
        ma_bearish,
        strong_trend,
        supertrend_bearish,
    ])
    
    return long, short

strategy.register("hermes_v12_1h_trend", make_signals)
