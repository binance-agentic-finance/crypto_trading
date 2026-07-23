"""Turtle Breakout Strategy v3 - 5m USDT Futures with ATR-based stops.

Based on user dialogue analysis:
- Long trigger: close > $82.40 (recent high breakout)
- Short trigger: close < $82.11 (recent low breakdown)
- ATR(N) = $0.0789 for stop calculations
- Long exit: $82.11 (10-period low)
- Short exit: $82.25 (10-period high)
- Stop loss: 2N from entry
"""

from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, exit as ex, strategy

def make_signals(df):
    """Generate long/short signals based on Turtle breakout rules.
    
    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume, ...]
    
    Returns:
        (long_signal, short_signal) - both pd.Series[bool]
    """
    # Core indicators
    atr = ind.atr(df, period=14)  # ATR for stop calculation
    donchian_upper, donchian_lower, _ = ind.donchian(df, period=20)  # 20-period breakout levels
    swing_high_10 = ind.swing_high(df, lookback=10)  # 10-period high for short exit
    swing_low_10 = ind.swing_low(df, lookback=10)   # 10-period low for long exit
    
    # Volume confirmation
    vol_ma20 = ind.volume_ma(df, period=20)
    
    # Long conditions: breakout above 20-period high with volume surge
    long_breakout = cond.breakout_high(df, lookback=20)
    long_volume = cond.volume_surge(df, vol_ma20, multiplier=1.2)
    long = entry.all_of([
        long_breakout,
        long_volume,
    ])
    
    # Short conditions: breakdown below 20-period low with volume surge
    short_breakout = cond.breakout_low(df, lookback=20)
    short_volume = cond.volume_surge(df, vol_ma20, multiplier=1.2)
    short = entry.all_of([
        short_breakout,
        short_volume,
    ])
    
    return long, short

# Register strategy for backtesting
strategy.register("turtle_breakout_v3", make_signals)
