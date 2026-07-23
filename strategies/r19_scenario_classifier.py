"""
R19 小周期場景判斷策略 - Multi-Scenario Market Regime Classifier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

策略目標：
  識別市場當前行駛的場景（Regime），並根據場景切換交易策略。
  
  原始場景（A/B/C）擴展為更全面的分類系統：
  
  【趨勢場景】
  - A1: 強多頭趨勢 (Strong Bull Trend)
  - A2: 弱多頭趨勢 (Weak Bull Trend)
  - B1: 強空頭趨勢 (Strong Bear Trend)
  - B2: 弱空頭趨勢 (Weak Bear Trend)
  
  【震盪場景】
  - C1: 高位震盪 (High Range)
  - C2: 中位震盪 (Mid Range)
  - C3: 低位震盪 (Low Range)
  
  【過渡/特殊場景】
  - D: 趨勢轉換期 (Transition/Choppy)
  - E: 極端波動/突破前夕 (High Volatility/Pre-Breakout)
  - F: 低波動壓縮 (Low Volatility Compression)

判斷依據：
  1. ADX - 趨勢強度指標
  2. MA 排列 - 趨勢方向
  3. 價格相對於 MA/布林帶的位置
  4. 波動率（布林帶寬度、ATR）
  5. 成交量變化
  6. RSI 動能
  7. MACD 動能確認

適用週期：5m, 15m, 1h
適用品種：BTCUSDT, ETHUSDT 等高流動性合約
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
from cyqnt_trd.blocks import (
    indicators as ind,
    conditions as cond,
    entry,
    regime,
    strategy,
)


def classify_market_scenario(df: pd.DataFrame) -> pd.Series:
    """
    多場景分類器 - 識別市場當前行駛的場景。
    
    返回一個 string Series，每個值代表該 bar 的場景標籤。
    
    場景分類邏輯：
    ┌──────────────────────────────────────────────────────────┐
    │                    ADX >= 25 (趨勢市)                    │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │  MA 多頭排列 + MACD>0 + RSI>50 → A1 (強多頭)        │ │
    │  │  MA 多頭排列 + MACD<0 或 RSI<50 → A2 (弱多頭)       │ │
    │  │  MA 空頭排列 + MACD<0 + RSI<50 → B1 (強空頭)        │ │
    │  │  MA 空頭排列 + MACD>0 或 RSI>50 → B2 (弱空頭)       │ │
    │  └─────────────────────────────────────────────────────┘ │
    ├──────────────────────────────────────────────────────────┤
    │                    ADX < 25 (震盪市)                     │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │  價格 > BB 中軌 + RSI>50 → C1 (高位震盪)            │ │
    │  │  價格 ≈ BB 中軌 ±1% + RSI 40-60 → C2 (中位震盪)    │ │
    │  │  價格 < BB 中軌 + RSI<50 → C3 (低位震盪)            │ │
    │  └─────────────────────────────────────────────────────┘ │
    ├──────────────────────────────────────────────────────────┤
    │                    特殊場景                              │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │  ADX 20-30 + MA 糾結 → D (趨勢轉換期)               │ │
    │  │  BB 寬度 > 5% + ATR 飆升 → E (極端波動)             │ │
    │  │  BB 寬度 < 2% + 成交量萎縮 → F (低波動壓縮)         │ │
    │  └─────────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────────┘
    """
    close = df["close"]
    timestamps = df["close_time"]
    
    # ═══════════════════════════════════════════════════════
    # 核心指標計算
    # ═══════════════════════════════════════════════════════
    
    # 移動平均線組
    ma20 = ind.sma(close, 20)
    ma50 = ind.sma(close, 50)
    ma200 = ind.sma(close, 200)
    ema20 = ind.ema(close, 20)
    ema50 = ind.ema(close, 50)
    
    # MACD
    macd_line, signal_line, macd_hist = ind.macd(close, 12, 26, 9)
    
    # RSI
    rsi = ind.rsi(close, 14)
    
    # 布林帶
    bb_upper, bb_mid, bb_lower = ind.bollinger(close, 20, 2.0)
    
    # ATR
    atr = ind.atr(df, 14)
    
    # ADX
    adx, plus_di, minus_di = ind.adx(df, 14)
    
    # 成交量
    vol_ma20 = ind.volume_ma(df, 20)
    
    # 波動率（布林帶寬度百分比）
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    # ═══════════════════════════════════════════════════════
    # 趨勢判斷條件
    # ═══════════════════════════════════════════════════════
    
    # MA 多頭排列：短 > 中 > 長
    ma_bullish = (ma20 > ma50) & (ma50 > ma200)
    
    # MA 空頭排列：短 < 中 < 長
    ma_bearish = (ma20 < ma50) & (ma50 < ma200)
    
    # MA 糾結（過渡期特徵）
    ma_tangled = (
        (abs(ma20 - ma50) / ma50 < 0.01) &
        (abs(ma50 - ma200) / ma200 < 0.02)
    )
    
    # 趨勢市 vs 震盪市
    is_trending = adx >= 25.0
    is_ranging = adx < 25.0
    is_transition = (adx >= 20.0) & (adx < 30.0) & ma_tangled
    
    # ═══════════════════════════════════════════════════════
    # 場景 A: 多頭趨勢 (A1 強多頭，A2 弱多頭)
    # ═══════════════════════════════════════════════════════
    
    # A1: 強多頭 - 趨勢市 + MA 多頭 + MACD>0 + RSI>50
    scenario_a1 = entry.all_of([
        is_trending,
        ma_bullish,
        macd_line > 0,
        rsi > 50,
    ])
    
    # A2: 弱多頭 - 趨勢市 + MA 多頭 + 但動能不足
    scenario_a2 = entry.all_of([
        is_trending,
        ma_bullish,
        entry.any_of([
            macd_line < 0,
            rsi < 50,
        ]),
    ])
    
    # ═══════════════════════════════════════════════════════
    # 場景 B: 空頭趨勢 (B1 強空頭，B2 弱空頭)
    # ═══════════════════════════════════════════════════════
    
    # B1: 強空頭 - 趨勢市 + MA 空頭 + MACD<0 + RSI<50
    scenario_b1 = entry.all_of([
        is_trending,
        ma_bearish,
        macd_line < 0,
        rsi < 50,
    ])
    
    # B2: 弱空頭 - 趨勢市 + MA 空頭 + 但動能不足
    scenario_b2 = entry.all_of([
        is_trending,
        ma_bearish,
        entry.any_of([
            macd_line > 0,
            rsi > 50,
        ]),
    ])
    
    # ═══════════════════════════════════════════════════════
    # 場景 C: 震盪 (C1 高位，C2 中位，C3 低位)
    # ═══════════════════════════════════════════════════════
    
    # 價格相對於布林帶的位置
    price_at_upper = close >= bb_upper * 0.98
    price_at_lower = close <= bb_lower * 1.02
    price_at_mid = (close > bb_lower * 1.02) & (close < < bb_upper * 0.98)
    
    # C1: 高位震盪 - 震盪市 + 價格靠近上軌 + RSI>50
    scenario_c1 = entry.all_of([
        is_ranging,
        price_at_upper,
        rsi > 50,
    ])
    
    # C2: 中位震盪 - 震盪市 + 價格在中間 + RSI 40-60
    scenario_c2 = entry.all_of([
        is_ranging,
        price_at_mid,
        (rsi >= 40) & (rsi <= 60),
    ])
    
    # # C3: 低位震盪 - 震盪市 + 價格靠近下軌 + RSI<50
    scenario_c3 = entry.all_of([
        is_ranging,
        price_at_lower,
        rsi < 50,
    ])
    
    # ═══════════════════════════════════════════════════════
    # 場景 D: 趨勢轉換期
    # ═══════════════════════════════════════════════════════
    
    scenario_d = is_transition
    
    # ═══════════════════════════════════════════════════════
    # 場景 E: 極端波動/突破前夕
    # ═══════════════════════════════════════════════════════
    
    # 高波動率 + ATR 飆升 + 成交量放大
    high_volatility = bb_width > 0.05
    atr_spike = atr > ind.ema(attr, 20) * 1.5
    volume_surge = cond.volume_surge(df, vol_ma20, multiplier=2.0)
    
    scenario_e = entry.all_of([
        entry.any_of([high_volatility, atr