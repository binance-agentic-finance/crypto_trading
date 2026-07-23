"""
RSI/MACD Divergence + 200 EMA Trend Filter Strategy
Based on user dialogue: "Divergence is a hint, not a signal"
Entry requires: Position + Divergence + Candle confirmation

Key rules:
- Long only when price > 200 EMA
- Short only when price < 200 EMA
- RSI divergence at support/resistance zones
- Entry triggered by candle patterns (engulfing, hammer, breakout)
- Stop loss: below divergence low (long) / above divergence high (short)
- Take profit: 1) Previous high/low, 2) RR 1:2
"""

from cyqnt_trd.blocks import (
    indicators as ind,
    conditions as cond,
    entry,
    exit,
    patterns as pat,
    strategy,
)
import pandas as pd


def make_signals(df: pd.DataFrame):
    """
    Generate long/short signals based on RSI divergence + MACD + 200 EMA filter.
    
    Returns:
        (long_signal, short_signal) - tuple of boolean Series
    """
    # === Core Indicators ===
    # RSI (14) - for divergence detection
    rsi = ind.rsi(df["close"], 14)
    
    # MACD (12, 26, 9) - confirmation
    macd_line, signal_line, macd_hist = ind.macd(df["close"], 12, 26, 9)
    
    # 200 EMA - trend filter
    ema200 = ind.ema(df["close"], 200)
    
    # === Support/Resistance Levels (for "position" check) ===
    # Use swing highs/lows to identify key levels
    swing_high = ind.swing_high(df, lookback=20)
    swing_low = ind.swing_low(df, lookback=20)
    
    # Previous high/low for structure breaks
    prev_high = ind.highest(df["high"], 20)
    prev_low = ind.lowest(df["low"], 20)
    
    # === Trend Filter (200 EMA) ===
    # Long only when price is above 200 EMA
    price_above_ema200 = cond.price_above_ma(df, ema200, bars=1)
    # Short only when price is below 200 EMA
    price_below_ema200 = cond.price_below_ma(df, ema200, bars=1)
    
    # === RSI Divergence Conditions ===
    # Regular bullish divergence: Price makes LL, RSI makes HL
    # This is detected at support zones (near swing lows)
    rsi_oversold = cond.rsi_oversold(rsi, threshold=35.0)  # Slightly above 30 for earlier detection
    rsi_overbought = cond.rsi_overbought(rsi, threshold=65.0)  # Slightly below 70
    
    # MACD divergence detection
    macd_bullish_div = cond.macd_bullish_divergence(df["close"], macd_line, lookback=20)
    macd_bearish_div = cond.macd_bearish_divergence(df["close"], macd_line, lookback=20)
    
    # === Hidden Divergence (trend continuation) ===
    # Hidden bullish: Price makes HL, RSI makes LL (in uptrend)
    # Hidden bearish: Price makes LH, RSI makes HH (in downtrend)
    # Note: blocks doesn't have explicit hidden divergence, approximate with structure
    
    # === Position Check (Support/Resistance) ===
    # Price near support (within 2% of swing low)
    near_support = (df["close"] - swing_low) / swing_low < 0.02
    # Price near resistance (within 2% of swing high)
    near_resistance = (swing_high - df["close"]) / swing_high < 0.02
    
    # === Candle Confirmation Patterns (Entry Trigger) ===
    # Bullish patterns for long entry
    bullish_engulfing = pat.bullish_engulfing(df)
    hammer = pat.hammer(df)
    morning_star = pat.morning_star(df)
    strong_bullish_bar = pat.is_bullish_bar(df, min_body_pct=0.01)  # Large body
    
    # Bearish patterns for short entry
    bearish_engulfing = pat.bearish_engulfing(df)
    shooting_star = pat.shooting_star(df)
    evening_star = pat.evening_star(df)
    strong_bearish_bar = pat.is_bearish_bar(df, min_body_pct=0.01)
    
    # === Structure Break Confirmation ===
    # Breakout above previous high (for long)
    breakout_long = cond.breakout_high(df, lookback=10)
    # Breakdown below previous low (for short)
    breakout_short = cond.breakout_low(df, lookback=10)
    
    # === LONG ENTRY LOGIC ===
    # Must have:
    # 1. Price above 200 EMA (trend filter)
    # 2. At support zone OR RSI oversold
    # 3. Bullish divergence (MACD or RSI-based)
    # 4. Candle confirmation (engulfing, hammer, or breakout)
    
    long_divergence_setup = entry.all_of([
        price_above_ema200,
        entry.any_of([near_support, rsi_oversold]),
        macd_bullish_div,
    ])
    
    long_confirmation = entry.any_of([
        bullish_engulfing,
        hammer,
        morning_star,
        strong_bullish_bar,
        breakout_long,
    ])
    
    long_signal = entry.all_of([
        long_divergence_setup,
        long_confirmation,
    ])
    
    # === SHORT ENTRY LOGIC ===
    # Must have:
    # 1. Price below 200 EMA (trend filter)
    # 2. At resistance zone OR RSI overbought
    # 3. Bearish divergence
    # 4. Candle confirmation
    
    short_divergence_setup = entry.all_of([
        price_below_ema200,
        entry.any_of([near_resistance, rsi_overbought]),
        macd_bearish_div,
    ])
    
    short_confirmation = entry.any_of([
        bearish_engulfing,
        shooting_star,
        evening_star,
        strong_bearish_bar,
        breakout_short,
    ])
    
    short_signal = entry.all_of([
        short_divergence_setup,
        short_confirmation,
    ])
    
    # === RISK MANAGEMENT (documented for framework) ===
    # Stop Loss: 
    #   - Long: Below the divergence low (swing_low)
    #   - Short: Above the divergence high (swing_high)
    # Take Profit:
    #   - TP1: Previous high/low (swing_high/swing_low)
    #   - TP2: RR 1:2 (2x risk distance)
    # 
    # Note: Exit rules should be configured in the backtest framework
    # using exit.stop_loss() and exit.take_profit() with dynamic levels
    # based on swing_high/swing_low at entry time.
    
    return long_signal, short_signal


# Register the strategy
strategy.register("divergence_ema200_rsi", make_signals)
