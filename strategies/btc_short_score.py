"""BTC Short Scoring Strategy — 枯竭 + 回落 + 资金费率 + 趋势分

Scoring-based short strategy that accumulates points from:
① BTC 趋势分：BTC < MA20 且 MA20 向下 +30 分
② 枯竭信号：24h 涨>15% + 4h 停滞 + 量缩 +30 分
③ 回落结构：距近期高点已回落>5% +20 分
④ 资金费率：fundingRate > 0.06% +10 分

Short signal fires when score >= 60 (adjustable threshold).
"""
import pandas as pd  # used for the pd.Series fallbacks below (was missing -> NameError)

from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, scoring, strategy


def make_signals(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    close_time = df["close_time"]

    # === ① BTC 趋势分：BTC < MA20 且 MA20 向下 ===
    ma20 = ind.sma(close, 20)
    ma20_prev = ma20.shift(1)
    ma20_declining = ma20 < ma20_prev
    price_below_ma20 = close < ma20
    trend_score_cond = price_below_ma20 & ma20_declining

    # === ② 枯竭信号：24h 涨>15% + 4h 停滞 + 量缩 ===
    # 24h 涨>15%: on 1h timeframe, lookback 24; on 15m, lookback 96
    # We approximate with price_change_pct over ~24 bars (adjust for your timeframe)
    price_24h_ago = close.shift(24)
    gain_24h = (close - price_24h_ago) / price_24h_ago
    surge_24h = gain_24h > 0.15

    # 4h 停滞：price range tight over last ~16 bars (4h on 15m) or ~4 bars on 1h
    # We use consolidation_range with 3% max range over 16 bars
    stalled = cond.consolidation_range(df, period=16, max_range_pct=0.03)

    # 量缩：volume < MA(volume, 20) for last 3 bars
    vol_ma20 = ind.volume_ma(df, 20)
    volume_shrinking = cond.volume_shrink(df, vol_ma20, bars=3, multiplier=1.0)

    exhaustion_cond = surge_24h & stalled & volume_shrinking

    # === ③ 回落结构：距近期高点已回落>5% ===
    swing_hi = ind.swing_high(df, lookback=20)
    drop_from_high = (swing_hi - close) / swing_hi
    pullback_cond = drop_from_high > 0.05

    # === ④ 资金费率：fundingRate > 0.06% ===
    # NOTE: funding rate is NOT in OHLCV DataFrame.
    # This must be injected by the framework or fetched separately.
    # For now, we approximate with a placeholder that ALWAYS evaluates False.
    # To enable: add 'funding_rate' column to df or use derivatives.funding_rate_state().
    # Here we assume funding_rate column exists (in basis points, 0.06% = 6 bps = 0.0006)
    if "funding_rate" in df.columns:
        funding_high = df["funding_rate"] > 0.0006
    else:
        # Fallback: skip this condition (score reduced by 10)
        funding_high = pd.Series(False, index=df.index)

    # === Build scoring system ===
    sys = scoring.ScoringSystem()
    sys.add_rule("trend_bearish", trend_score_cond, weight=30)
    sys.add_rule("exhaustion", exhaustion_cond, weight=30)
    sys.add_rule("pullback", pullback_cond, weight=20)
    sys.add_rule("funding_high", funding_high, weight=10)

    # Short fires when score >= 60
    short_signal = sys.signal(threshold=60)

    # Long signal: none (this is a short-only strategy)
    long_signal = pd.Series(False, index=df.index)

    return long_signal, short_signal


# Register the strategy
strategy.register("btc_short_score", make_signals)
