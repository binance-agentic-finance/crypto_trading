"""
拉娜風格動能策略 - 低勝率高賠率打法

核心邏輯：
1. 捕捉趨勢突破（高動能）
2. 放寬止損範圍（承受波動）
3. 多因子評分進場（提高質量）
4. 分批止盈（讓利潤奔跑）

適用：新幣/高波動標的
時間框架：15m / 1h
"""
from cyqnt_trd.blocks import (
    indicators as ind,
    conditions as cond,
    entry,
    exit as ex,
    scoring,
    regime,
    strategy,
)


def make_signals(df):
    """
    生成多空头信号
    
    Args:
        df: OHLCV DataFrame，包含 close_time（毫秒時間戳）
    
    Returns:
        (long_signal, short_signal) - 兩個 pd.Series[bool]
    """
    close = df["close"]
    
    # ========== 指標計算 ==========
    # 趨勢指標
    ma20 = ind.sma(close, 20)
    ma60 = ind.sma(close, 60)
    ema12 = ind.ema(close, 12)
    ema26 = ind.ema(close, 26)
    
    # 動能指標
    macd_line, signal_line, macd_hist = ind.macd(close, 6, 13, 5)
    rsi14 = ind.rsi(close, 14)
    
    # 波動率指標
    atr14 = ind.atr(df, 14)
    upper, mid, lower = ind.bollinger(close, 20, 2.0)
    
    # 成交量指標
    vol_ma20 = ind.volume_ma(df, 20)
    
    # 市場體制判斷
    adx_val, plus_di, minus_di = ind.adx(df, 14)
    is_trending = cond.adx_trending(adx_val, threshold=25.0)
    is_ranging = cond.adx_ranging(adx_val, threshold=20.0)
    
    # ========== 多因子評分系統 ==========
    sys = scoring.ScoringSystem()
    
    # 趨勢因子（正權重）
    sys.add_rule("ma_bullish", cond.price_above_ma(df, ma60, bars=1), weight=2.0)
    sys.add_rule("ema_alignment", cond.ma_cross_above(ema12, ema26), weight=1.5)
    sys.add_rule("macd_positive", cond.macd_above_zero(macd_line), weight=1.5)
    
    # 動能因子
    sys.add_rule("breakout_high", cond.breakout_high(df, lookback=20), weight=2.0)
    sys.add_rule("volume_surge", cond.volume_surge(df, vol_ma20, multiplier=1.5), weight=1.0)
    
    # RSI 因子（避免過熱）
    sys.add_rule("rsi_zone", cond.rsi_in_range(rsi14, 40, 75), weight=1.0)
    sys.add_rule("rsi_not_overbought", ~cond.rsi_overbought(rsi14, 70), weight=0.5)
    
    # 波動率因子（高波動優先）
    vol_regime = regime.volatility_regime(df, period=20, high_quantile=0.7, low_quantile=0.3)
    high_vol = (vol_regime == "high")
    sys.add_rule("high_volatility", high_vol, weight=1.0)
    
    # 反向因子（負權重）
    sys.add_rule("ma_bearish", cond.price_below_ma(df, ma60, bars=1), weight=-2.0)
    sys.add_rule("macd_negative", cond.macd_below_zero(macd_line), weight=-1.5)
    
    # ========== 進場條件 ==========
    # 長線：評分 >= 5.0 且在趨勢體制
    long_score = sys.evaluate() >= 5.0
    long = entry.all_of([
        long_score,
        is_trending | high_vol,  # 趨勢或高波動
        cond.ma_cross_above(ma20, ma60),  # MA 金叉確認
    ])
    
    # 短線：評分 <= -3.0（反向做空）
    short_score = sys.evaluate() <= -3.0
    short = entry.all_of([
        short_score,
        cond.ma_cross_below(ma20, ma60),
    ])
    
    # ========== 止盈止損規則（註解形式，由框架處理） ==========
    # 止損：ATR 2.5 倍（放寬止損）
    # stop_loss = ex.AtrTrailingStop(atr14, multiplier=2.5)
    
    # 止盈：分批止盈
    # take_profit = ex.MultiTp(targets=[
    #     (0.05, 0.3),   # 5% 漲幅賣出 30%
    #     (0.10, 0.3),   # 10% 漲幅賣出 30%
    #     (0.20, 0.4),   # 20% 漲幅賣出 40%
    # ])
    
    # 時間止盈：持有超過 N 根 K 棒自動止盈
    # time_exit = ex.TimeBasedExit(max_bars=48)  # 15m 框架下 = 12 小時
    
    return long, short


# 註冊策略
strategy.register("lana_momentum_v1", make_signals)
