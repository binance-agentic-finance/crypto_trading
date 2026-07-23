"""SMC Institutional Trader Strategy - 4H Framework.
Smart Money Concepts: Order Blocks, FVG, CHoCH, BOS.
Designed for BTC/USDT and ETH/BTC on 4H timeframe.

Risk Parameters:
- Risk per trade: 25 USDT (~5.5% of 450 USDT balance)
- Target R:R = 1:3 (minimum 1:2.5 at structural resistance)
- Leverage: x5 (adjustable in execution layer)
- Max 1 open position at a time

Note: Multi-timeframe confirmation (1H/15m) must be done manually
or via separate execution layer. This strategy provides 4H signals.
"""
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, exit as ex, risk, sizing, strategy

# ========== Configuration ==========
RISK_PER_TRADE_USDT = 25.0
ACCOUNT_BALANCE_USDT = 450.0
RISK_PCT = RISK_PER_TRADE_USDT / ACCOUNT_BALANCE_USDT  # ~5.5%
LEVERAGE = 5
MIN_RR_RATIO = 2.5
TARGET_RR_RATIO = 3.0
CONFIDENCE_THRESHOLD = 80  # Score-based confidence

def make_signals(df):
    """
    SMC Institutional Strategy Signal Generator.
    
    Step 1: 4H Context - Identify trend, swing highs/lows, FVG
    Step 2: Entry triggers - CHoCH + BOS confirmation (approximated)
    Step 3: Risk management - SL behind structure, TP at liquidity pools
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    open_price = df["open"]
    
    # ========== Step 1: 4H Market Context ==========
    
    # 1.1 Identify Swing Highs/Lows (20-bar lookback)
    swing_high = ind.swing_high(df, lookback=20)
    swing_low = ind.swing_low(df, lookback=20)
    
    # 1.2 Trend Detection - MA Alignment (20/50/200)
    ma20 = ind.sma(close, 20)
    ma50 = ind.sma(close, 50)
    ma200 = ind.sma(close, 200)
    
    # Bullish trend: MA20 > MA50 > MA200
    bullish_alignment = cond.ma_cross_above(ma20, ma50) & (ma20 > ma200)
    # Bearish trend: MA20 < MA50 < MA200
    bearish_alignment = cond.ma_cross_below(ma20, ma50) & (ma20 < ma200)
    
    # 1.3 Fair Value Gaps (FVG) - Approximated
    # FVG: Large candle with wicks not filling the body range
    candle_range = ind.candle_range(df) if hasattr(ind, 'candle_range') else (high - low)
    candle_body = abs(close - open_price)
    fvg_bullish = (candle_body > candle_range * 0.7) & (close > open_price) & (low > open_price.shift(1))
    fvg_bearish = (candle_body > candle_range * 0.7) & (close < open_price) & (high < open_price.shift(1))
    
    # 1.4 Order Blocks - Recent consolidation zones
    # Approximated as areas where price consolidated before a strong move
    ma_deviation = (close - ma20) / ma20
    order_block_long = (ma_deviation.abs() < 0.02).rolling(5).sum() >= 3  # 3 of last 5 bars near MA
    order_block_short = order_block_long  # Same logic for shorts
    
    # ========== Step 2: Entry Triggers (CHoCH + BOS) ==========
    
    # 2.1 Change of Character (CHoCH) - Break of recent swing
    # Long CHoCH: Price breaks above recent swing high after being below
    choch_long = cond.breakout_high(df, lookback=10) & bullish_alignment
    # Short CHoCH: Price breaks below recent swing low after being above
    choch_short = cond.breakout_low(df, lookback=10) & bearish_alignment
    
    # 2.2 Break of Structure (BOS) Confirmation
    # BOS: Higher high + higher low sequence (long) or lower high + lower low (short)
    hh = cond.higher_high(df, lookback=10)
    hl = cond.higher_low(df, lookback=10)
    lh = cond.lower_high(df, lookback=10)
    ll = cond.lower_low(df, lookback=10)
    
    bos_long = hh & hl
    bos_short = lh & ll
    
    # 2.3 Volume Confirmation - Institutional interest
    vol_ma20 = ind.volume_ma(df, 20)
    volume_surge = cond.volume_surge(df, vol_ma20, multiplier=1.5)
    
    # 2.4 RSI Filter - Avoid overbought/oversold entries
    rsi14 = ind.rsi(close, 14)
    rsi_long_ok = cond.rsi_in_range(rsi14, low=40, high=70)  # Not overbought
    rsi_short_ok = cond.rsi_in_range(rsi14, low=30, high=60)  # Not oversold
    
    # 2.5 ATR Volatility Filter - Ensure sufficient movement
    atr14 = ind.atr(df, 14)
    atr_expansion = atr14 > atr14.rolling(20).mean()  # ATR above its average
    
    # ========== Step 3: Signal Construction ==========
    
    # Long Entry: CHoCH + BOS + Volume + RSI OK + in Order Block zone
    long_conditions = [
        choch_long,
        bos_long,
        volume_surge,
        rsi_long_ok,
        order_block_long,
    ]
    long_signal = entry.all_of(long_conditions)
    
    # Short Entry: CHoCH + BOS + Volume + RSI OK + in Order Block zone
    short_conditions = [
        choch_short,
        bos_short,
        volume_surge,
        rsi_short_ok,
        order_block_short,
    ]
    short_signal = entry.all_of(short_conditions)
    
    # ========== Confidence Scoring (for >80% threshold) ==========
    # Score each signal component to estimate confidence
    score_components_long = (
        choch_long.astype(int) * 20 +
        bos_long.astype(int) * 25 +
        volume_surge.astype(int) * 20 +
        rsi_long_ok.astype(int) * 15 +
        order_block_long.astype(int) * 20
    )
    
    score_components_short = (
        choch_short.astype(int) * 20 +
        bos_short.astype(int) * 25 +
        volume_surge.astype(int) * 20 +
        rsi_short_ok.astype(int) * 15 +
        order_block_short.astype(int) * 20
    )
    
    # Filter signals below 80% confidence
    long_signal = long_signal & (score_components_long >= CONFIDENCE_THRESHOLD)
    short_signal = short_signal & (score_components_short >= CONFIDENCE_THRESHOLD)
    
    # ========== Risk Management Notes (for execution layer) ==========
    # Stop Loss: Place behind the swing low/high that confirmed the CHoCH
    #   - Long SL: Below recent swing_low (use ind.swing_low)
    #   - Short SL: Above recent swing_high (use ind.swing_high)
    # Take Profit: Next 4H liquidity pool (prior swing high/low", or use ATR-based targets
    #   - TP1: 1:2.5 R:R
    #   - TP2: 1:3 R:R or at structural resistance
    # Position Size: 25 USDT per trade (5.5% of 450 USDT)
    #   - Use sizing.risk_based_size(equity, entry_price, stop_price, RISK_PCT)
    
    return long_signal, short_signal

# Register the strategy
strategy.register("smc_institutional_4h", make_signals)

# ========== Execution Layer Notes (for live_executor.py) ==========
"""
To implement the full Telegram + Binance automation:

1. live_executor.py:
   - Use Binance Futures API (ccxt or python-binance)
   - Calculate position size: sizing.risk_based_size(450, entry, sl, 0.055)
   - Place OCO orders: execution.oco_pair(symbol, side, tp_price, sl_price)
   - Leverage: Set via Binance API (user_data.modify_leverage)

2. telegram_bot.py:
   - Use python-telegram-bot library
   - Send signal cards with inline buttons:
     Keyboard: [[("✅ ОТКРЫТЬ", "open_trade"), ("❌ ПРОПУСТИТЬ", "skip_trade")]]
   - Handle callbacks and call live_executor.py

3. trade_manager.py:
   - Monitor open positions via Binance API
   - Implement trailing SL: ex.AtrTrailingStop(atr14, multiplier=2.0)
   - Send alerts on TP1/TP2 hit
   - TTL close: Track entry time, force close after N hours

4. Risk Guard Configuration:
   risk_config = risk.RiskConfig(
       max_loss_per_trade_pct=0.055,  # 25/450
       leverage=5,
       max_positions=1,
       per_symbol_cooldown=(1, 3600000),  # 1 hour cooldown
   )
   guard = risk.RiskGuard(risk_config)

Run backtest:
   python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \\
     --engine python \\
     --strategy smc_institutional_4h \\
     --strategy-module smc_institutional_strategy \\
     --symbol BTCUSDT --interval 4h --limit 1000
"""
