"""
綜合趨勢動能策略 - Comprehensive Trend Momentum Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

策略邏輯：
  做多時機 (Long Entry)：
  1. 趨勢濾網：價格 > EMA50 > EMA200 (多頭排列)
  2. 動能確認：MACD 黃金交叉 + MACD > 0
  3. RSI 處於強勢區 (50-70)，不過熱
  4. 成交量放大 (> 20 日均量 1.5 倍)
  5. 價格突破 20 日高點 或 回測 EMA20 後反彈

  做空時機 (Short Entry)：
  1. 趨勢濾網：價格 < EMA50 < EMA200 (空頭排列)
  2. 動能確認：MACD 死亡交叉 + MACD < 0
  3. RSI 處於弱勢區 (30-50)，不過度超賣
  4. 成交量放大 (> 20 日均量 1.5 倍)
  5. 價格跌破 20 日低點 或 反彈至 EMA20 後回落

  出場規則：
  - 止損：ATR 的 2 倍距離
  - 止盈：1) ATR 的 3 倍距離 2) 反向信號出現
  - 時間止損：持有超過 20 根 K 棒無獲利

  風控規則：
  - 單一品種最大倉位：總資金的 15%
  - 每日最大虧損：總資金的 5%
  - 避免在資金費率結算前後 15 分鐘開新倉
  - 不做空頭排列中的做多，不做多頭排列中的做空

策略編號：CYQNT-001
適用週期：15m, 1h, 4h
適用品種：BTCUSDT, ETHUSDT 等高流動性合約
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
from cyqnt_trd.blocks import (
    indicators as ind,
    conditions as cond,
    entry,
    exit as ex,
    patterns as pat,
    regime,
    strategy,
)


def make_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    生成做多/做空信號。
    
    Args:
        df: OHLCV DataFrame，包含 open, high, low, close, volume, close_time 等欄位
    
    Returns:
        (long_signal, short_signal): 兩個布林值 Series
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # ═══════════════════════════════════════════════════════
    # 核心指標計算
    # ═══════════════════════════════════════════════════════
    
    # 移動平均線組 (多頭/空頭排列判斷)
    ema20 = ind.ema(close, 20)
    ema50 = ind.ema(close, 50)
    ema200 = ind.ema(close, 200)
    sma20_vol = ind.volume_ma(df, 20)
    
    # MACD (標準參數 12, 26, 9)
    macd_line, signal_line, macd_hist = ind.macd(close, 12, 26, 9)
    
    # RSI (14 週期)
    rsi = ind.rsi(close, 14)
    
    # ATR (14 週期，用於止損計算)
    atr = ind.atr(df, 14)
    
    # 布林帶 (用於區間判斷)
    bb_upper, bb_mid, bb_lower = ind.bollinger(close, 20, 2.0)
    
    # 20 日高低點 (用於突破判斷)
    donchian_upper, donchian_lower, _ = ind.donchian(df, 20)
    
    # ADX (趨勢強度)
    adx, plus_di, minus_di = ind.adx(df, 14)
    
    # 市場狀態 (趨勢市/震盪市)
    is_trending = cond.adx_trending(adx, threshold=25.0)
    is_ranging = cond.adx_ranging(adx, threshold=20.0)
    
    # ═══════════════════════════════════════════════════════
    # 做多條件 (Long Conditions)
    # ═══════════════════════════════════════════════════════
    
    # 1. 趨勢濾網：多頭排列 (價格 > EMA20 > EMA50 > EMA200)
    bullish_alignment = entry.all_of([
        cond.price_above_ma(df, ema20, bars=1),
        cond.price_above_ma(df, ema50, bars=1),
        cond.price_above_ma(df, ema200, bars=1),
        ema20 > ema50,
        ema50 > ema200,
    ])
    
    # 2. 動能確認：MACD 黃金交叉 + MACD 在零軸上方
    macd_bullish = entry.all_of([
        cond.macd_golden_cross(macd_line, signal_line),
        cond.macd_above_zero(macd_line),
    ])
    
    # 3. RSI 強勢但不過熱 (50-70)
    rsi_bullish_zone = cond.rsi_in_range(rsi, low=50.0, high=70.0)
    
    # 4. 成交量放大 (大於 20 日均量 1.5 倍)
    volume_surge_long = cond.volume_surge(df, sma20_vol, multiplier=1.5)
    
    # 5. 價格突破 20 日高點 或 回測 EMA20 後反彈
    breakout_long = cond.breakout_high(df, lookback=20)
    pullback_long = cond.price_bounce_ma(df, ema20, direction="long")
    entry_trigger_long = entry.any_of([breakout_long, pullback_long])
    
    # 6. K 棒確認：多頭吞噬 或 錘子線
    candle_bullish = entry.any_of([
        pat.bullish_engulfing(df),
        pat.hammer(df),
        pat.morning_star(df),
    ])
    
    # === 做多信號組合 ===
    # 核心條件：趨勢 + 動能 + RSI 區間
    long_core = entry.all_of([
        bullish_alignment,
        macd_bullish,
        rsi_bullish_zone,
    ])
    
    # 進場觸發：成交量 + 突破/回測 + K 棒確認
    long_trigger = entry.all_of([
        volume_surge_long,
        entry_trigger_long,
        candle_bullish,
    ])
    
    # 最終做多信號
    long_signal = entry.all_of([
        long_core,
        long_trigger,
        is_trending,  # 只在趨勢市進場
    ])
    
    # ═══════════════════════════════════════════════════════
    # 做空條件 (Short Conditions)
    # ═══════════════════════════════════════════════════════
    
    # 1. 趨勢濾網：空頭排列 (價格 < EMA20 < EMA50 < EMA200)
    bearish_alignment = entry.all_of([
        cond.price_below_ma(df, ema20, bars=1),
        cond.price_below_ma(df, ema50, bars=1),
        cond.price_below_ma(df, ema200, bars=1),
        ema20 < ema50,
        ema50 < ema200,
    ])
    
    # 2. 動能確認：MACD 死亡交叉 + MACD 在零軸下方
    macd_bearish = entry.all_of([
        cond.macd_death_cross(macd_line, signal_line),
        cond.macd_below_zero(macd_line),
    ])
    
    # 3. RSI 弱勢但不過度超賣 (30-50)
    rsi_bearish_zone = cond.rsi_in_range(rsi, low=30.0, high=50.0)
    
    # 4. 成交量放大
    volume_surge_short = cond.volume_surge(df, sma20_vol, multiplier=1.5)
    
    # 5. 價格跌破 20 日低點 或 反彈至 EMA20 後回落
    breakout_short = cond.breakout_low(df, lookback=20)
    pullback_short = cond.price_bounce_ma(df, ema20, direction="short")
    entry_token_short = entry.any_of([breakout_short, pullback_short])
    
    # 6. K 反轉 K 棒：空頭吞噬 或 流星線
    candle_bearish = entry.any_of([
        pat.bearish_engulfing(df),
        pat.shooting_star(df),
        pat.evening_star(df),
    ])
    
    # === 做空信號組合 ===
    bearish_core = entry.all_of([
        bearish_alignment,
        macd_bearish,
        rsi_bearish_zone,
    ])
    
    short_trigger = entry.all_of([
        volume_surge_short,
        breakout_short,
        candle_bearish,
    ])
    
    short_signal = entry.all_of([
        bearish_core,
        short_trigger,
        is_trending,
    ])
    
    # ═══════════════════════════════════════════════════════
    # 出場規則 (Exit Rules) - 供框架參考
    # ═══════════════════════════════════════════════════════
    # 止損：ATR 2 倍
    #   long_sl_price = entry_price - 2 * ATR
    #   short_sl_price = entry_price + 2 * ATR
    #
    # 止盈：ATR 3 倍
    #   long_tp_price = entry_price + 3 * ATR
    #   short_tp_price = entry_price - 3 * ATR
    #
    # 時間止損：20 根 K 棒
    #   ex.TimeBasedExit(max_bars=20)
    #
    # 移動止損：ATR 追蹤止損
    #   ex.AtrTrailingStop(atr, multiplier=2.0)
    
    # ═══════════════════════════════════════════════════════
    # 風控規則 (Risk Management) - 供框架參考
    # ═══════════════════════════════════════════════════════
    # - 單一品種最大倉位：15% 總資金
    # - 每日最大虧損：5% 總資金
    # - 避免資金費率結算前後 15 分鐘開倉
    # - 不做逆勢單 (多頭排列不做空，空頭排列不做多)
    # - 震盪市 (ADX < 20) 減少倉位或暫停交易
    
    return long_signal, short_signal


# 註冊策略
strategy.register("comprehensive_trend_momentum", make_signals)
