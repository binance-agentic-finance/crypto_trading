"""
上升通道突破策略 — 零前视版本

策略逻辑：
1. 使用 Donchian Channel 画出 N 日高低点通道
2. 通道低点附近开多，通道高点附近开空
3. 完全零前视：只用当前 bar 及之前的数据
4. 风控：固定止损 + 时间退出

风控规则（由框架统一处理）：
- 止损：ATR × 2.0
- 止盈：ATR × 4.0 (R:R = 2:1)
- 时间退出：10 根 K 线后平仓
- 最大仓位：15% 权益
"""
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, exit as ex, strategy

def make_signals(df):
    # ========== 通道参数 ==========
    channel_period = 20  # 20 日通道(可调整)

    # ========== 计算 Donchian Channel ==========
    upper, lower, mid = ind.donchian(df, period=channel_period)

    # ========== 定义"接近通道边界"的阈值 ==========
    # 价格在通道下轨附近(距离 <= 1%)→ 准备开多
    # 价格在通道上轨附近(距离 <= 1%)→ 准备开空
    close = df["close"]
    near_lower = (close - lower) / lower <= 0.01  # 距离下轨 <= 1%
    near_upper = (upper - close) / upper <= 0.01  # 距离上轨 <= 1%

    # ========== 确认趋势方向(避免在下降通道中做多)==========
    # 使用 MA20 作为趋势过滤器
    ma20 = ind.sma(close, 20)
    uptrend = close > ma20
    downtrend = close < ma20

    # ========== 成交量确认(避免假突破)==========
    vol_ma20 = ind.volume_ma(df, 20)
    vol_surge = cond.volume_surge(df, vol_ma20, multiplier=1.2)

    # ========== 进场条件 ==========
    # 做多:价格接近下轨 +  uptrend + 成交量放大
    long = entry.all_of([
        near_lower,
        uptrend,
        vol_surge,
    ])

    # 做空:价格接近上轨 + downtrend + 成交量放大
    short = entry.all_of([
        near_upper,
        downtrend,
        vol_surge,
    ])

    return long, short

# ========== 注册策略 ==========
strategy.register("channel_breakout_v1", make_signals)
