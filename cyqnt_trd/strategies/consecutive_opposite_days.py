"""
Strategy to detect consecutive opposite-direction days (1 up, 1 down, 1 up, 1 down...).
Returns signals when this pattern is broken or when it reaches a certain streak length.
"""
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, strategy
import pandas as pd
import numpy as np

def make_signals(df):
    """
    Detect consecutive opposite-direction days pattern.
    
    A "red" day: close < open
    A "green" day: close > open
    
    We want to find streaks where:
    - Day 1: up, Day 2: down, Day 3: up, Day 4: down, ...
    - OR: Day 1: down, Day 2: up, Day 3: down, Day 4: up, ...
    
    Returns:
    - long_signal: True when a long streak of alternating days ends (potential reversal)
    - short_signal: True when a long streak of alternating days ends (potential reversal)
    """
    close = df["close"]
    open_price = df["open"]
    
    # Determine day direction: 1 = green (up), -1 = red (down), 0 = doji
    day_direction = pd.Series(0, index=df.index)
    day_direction[close > open_price] = 1   # green
    day_direction[close < open_price] = -1  # red
    
    # Check if current day is opposite to previous day
    is_opposite = day_direction.shift(1) * day_direction == -1
    
    # Count consecutive opposite days
    # We need to track streaks manually since blocks doesn't have this exact function
    streak = pd.Series(0, index=df.index)
    
    for i in range(1, len(df)):
        if is_opposite.iloc[i]:
            streak.iloc[i] = streak.iloc[i-1] + 1
        else:
            streak.iloc[i] = 0
    
    # Signal when streak reaches certain length (e.g., 5+ consecutive opposite days)
    # This indicates an unusually long alternating pattern
    long_streak_threshold = 5
    
    # Long signal: alternating streak ended after being long (potential continuation up)
    # Short signal: alternating streak ended after being long (potential continuation down)
    
    # Detect when the alternating pattern breaks
    pattern_broke = ~is_opposite & (streak.shift(1) >= long_streak_threshold)
    
    # For simplicity, we'll signal on both sides when a long alternating streak breaks
    # The actual direction would need more context
    long = pattern_broke & (day_direction == 1)  # Broke after an up day
    short = pattern_broke & (day_direction == -1)  # Broke after a down day
    
    return long, short

strategy.register("consecutive_opposite_days", make_signals)
