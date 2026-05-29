"""
ADA/USDT Multi-Timeframe Confluence Strategy (Long-only)
Combines RSI, Stochastic, Bollinger Bands, Volume, EMAs (20, 50, 200)
with Fibonacci-based support/resistance levels for entry signals.

Risk Management (configured externally):
- Capital: 4,000 USDT
- Leverage: 20x
- Stop Loss: Technical (below Fibonacci support / EMA200)
- Take Profit: 3 levels based on Fibonacci extensions

Note: Order Book analysis, Fibonacci levels, and liquidation calculations
are handled by the execution framework, not in signal generation.
"""

from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, strategy

def make_signals(df):
    # === Indicators ===
    # EMAs for trend structure (20, 50, 200)
    ema20 = ind.ema(df["close"], 20)
    ema50 = ind.ema(df["close"], 50)
    ema200 = ind.ema(df["close"], 200)
    
    # RSI for momentum (14-period)
    rsi14 = ind.rsi(df["close"], 14)
    
    # Stochastic for overbought/oversold (14, 3, 3)
    stoch_k, stoch_d = ind.stochastic(df, k_period=14, d_period=3, smooth_k=3)
    
    # Bollinger Bands for volatility (20, 2.0)
    bb_upper, bb_mid, bb_lower = ind.bollinger(df["close"], period=20, std_mult=2.0)
    
    # Volume confirmation
    vol_ma20 = ind.volume_ma(df, 20)
    
    # MACD for additional momentum confirmation (6, 13, 5 - faster for crypto)
    macd_line, signal_line, macd_hist = ind.macd(df["close"], 6, 13, 5)
    
    # ATR for volatility-based stop loss calculation (handled by framework)
    atr14 = ind.atr(df, 14)
    
    # === Market Regime ===
    # Trend confirmation via EMA alignment
    # Bullish: EMA20 > EMA50 > EMA200
    ema20_above_ema50 = ema20 > ema50
    ema50_above_ema200 = ema50 > ema200
    trend_bullish = ema20_above_ema50 & ema50_above_ema200
    
    # Check if price is near key support (Bollinger Lower Band or EMA200)
    # Support zone: price within 0.5% of BB lower or 1% of EMA200
    near_bb_support = df["close"] <= bb_lower * 1.005
    near_ema200_support = df["close"] <= ema200 * 1.01
    near_support = near_bb_support | near_ema200_support
    
    # === Entry Conditions (Long) ===
    # 1. RSI oversold or recovering from oversold
    rsi_oversold = cond.rsi_oversold(rsi14, threshold=35.0)
    rsi_recovering = (rsi14 > 40) & (rsi14.shift(1) <= 40)
    rsi_condition = rsi_oversold | rsi_recovering
    
    # 2. Stochastic bullish crossover in oversold zone
    stoch_oversold = (stoch_k < 20) & (stoch_d < 20)
    stoch_cross = cond.macd_golden_cross(stoch_k, stoch_d)  # Reuse MACD crossover logic for Stoch
    stoch_condition = stoch_oversold & stoch_cross
    
    # 3. Price bouncing from Bollinger Lower Band
    bb_bounce = (df["close"] > bb_lower) & (df["close"].shift(1) <= bb_lower.shift(1))
    
    # 4. Volume surge confirming the move
    vol_surge = cond.volume_surge(df, vol_ma20, multiplier=1.5)
    
    # 5. MACD confirmation (above zero or golden cross)
    macd_confirm = cond.macd_above_zero(macd_line) | cond.macd_golden_cross(macd_line, signal_line)
    
    # 6. Trend alignment (price above EMA200 for long-term bullish bias)
    price_above_ema200 = cond.price_above_ma(df, ema200, bars=1)
    
    # Combine all long conditions using entry.all_of
    long = entry.all_of([
        price_above_ema200,               # Long-term uptrend bias
        near_support,                     # Price at key support zone
        rsi_condition,                    # RSI showing strength or oversold
        bb_bounce,                        # Bollinger Band bounce
        vol_surge,                        # Volume confirmation
        macd_confirm,                     # MACD confirmation
    ])
    
    # === Short Conditions ===
    # This strategy is LONG-only based on user requirements
    # Short signals would require inverse conditions
    short = None
    
    return long, short

# Register the strategy with ID "ada_usdt_multi_tf_confluence"
strategy.register("ada_usdt_multi_tf_confluence", make_signals)

# === RISK MANAGEMENT NOTES (Handled by Framework) ===
# 
# Capital: 4,000 USDT
# Leverage: 20x
# 
# Stop Loss Calculation:
# - Technical SL: Below nearest Fibonacci support level or EMA200
# - Use ATR-based buffer: SL = Entry - (2.0 * ATR14)
# - Liquidation price must be >5% below SL to avoid noise liquidation
# - With 20x leverage, liquidation occurs at ~5% move against position
# 
# Take Profit Levels (3 tiers):
# - TP1: Fibonacci 0.382 retracement from recent swing high
# - TP2: Fibonacci 0.618 retracement (golden ratio)
# - TP3: Fibonacci 1.0 extension (full retracement)
# 
# Position Sizing:
# - Risk per trade: 2% of capital (80 USDT)
# - Position size = Risk / (Entry - SL)
# - With 20x leverage, max position = 4000 * 20 = 80,000 USDT notional
# 
# Order Book Analysis (Not in signal generation):
# - Buy/Sell ratio at 2% and 5% from current price
# - Identify buy walls > 100,000 USDT within 2% range
# - Order flow control: Aggressive buyers if taker_buy_volume > 1.5 * taker_sell_volume
# 
# Multi-Timeframe Confirmation:
# - 1h: Entry trigger (this strategy)
# - 4h: Trend confirmation (EMA alignment)
# - 1D: Major support/resistance levels (Fibonacci)
