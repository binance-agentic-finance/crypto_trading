"""
Scalping strategy for futures markets - excludes large-cap coins.
Focuses on fast momentum + volume surge + RSI extremes for quick entries.
"""
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, regime, universe, strategy

def make_signals(df):
    # Fast indicators for scalping (5m-15m timeframe optimized)
    ma9 = ind.sma(df["close"], 9)
    ma21 = ind.sma(df["close"], 21)
    rsi7 = ind.rsi(df["close"], 7)
    vol_ma20 = ind.volume_ma(df, 20)
    atr14 = ind.atr(df, 14)
    
    # Bollinger Bands for mean-reversion scalps
    bb_upper, bb_mid, bb_lower = ind.bollinger(df["close"], period=20, std_mult=1.5)
    
    # Volume surge detection (critical for scalping liquidity)
    volume_surge = cond.volume_surge(df, vol_ma20, multiplier=1.8)
    
    # RSI extremes for quick reversals
    rsi_oversold = cond.rsi_oversold(rsi7, threshold=25.0)
    rsi_overbought = cond.rsi_overbought(rsi7, threshold=75.0)
    
    # Fast MA cross for momentum
    ma_cross_up = cond.ma_cross_above(ma9, ma21)
    ma_cross_down = cond.ma_cross_below(ma9, ma21)
    
    # Price position relative to MA
    price_above_ma9 = cond.price_above_ma(df, ma9, bars=1)
    price_below_ma9 = cond.price_below_ma(df, ma9, bars=1)
    
    # Bollinger bounce/breakout
    price_near_lower = df["close"] <= bb_lower
    price_near_upper = df["close"] >= bb_upper
    
    # Range regime detection (better for mean-reversion scalps)
    in_range = regime.is_range_regime(df, period=20, max_range_pct=0.05)
    
    # LONG: RSI oversold + volume surge + price bouncing or crossing up
    long = entry.all_of([
        rsi_oversold,
        volume_surge,
        entry.any_of([
            price_near_lower,  # BB bounce play
            ma_cross_up,       # Momentum cross
        ]),
    ])
    
    # SHORT: RSI overbought + volume surge + price rejecting or crossing down
    short = entry.all_of([
        rsi_overbought,
        volume_surge,
        entry.any_of([
            price_near_upper,  # BB rejection play
            ma_cross_down,     # Momentum cross
        ]),
    ])
    
    return long, short


# Universe scanner helper - call this separately to find candidates
def scan_scalp_candidates():
    """
    Scan futures universe for scalping opportunities.
    Excludes large-cap (BTC, ETH, BNB) and filters for volume + volatility.
    Returns list of symbol candidates.
    """
    # Fetch 24h ticker data
    tickers = universe.fetch_perpetual_universe(market_type="futures")
    
    # Filter and scan
    candidates = (
        universe.UniverseFilter(tickers)
            .filter_quote_suffix("USDT")
            .filter_quote_volume(min_quote_volume=50_000_000)  # Min $50M daily volume
            .filter_change_pct(max_abs_pct=15.0)  # Exclude extreme pumps/dumps
            .exclude_symbols(["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"])  # Exclude large caps
            .top_gainers(n=10)  # Get top 10 gainers for momentum scalps
            .symbols()
    )
    
    return candidates


# Register the strategy
strategy.register("scalp_opportunity", make_signals)
