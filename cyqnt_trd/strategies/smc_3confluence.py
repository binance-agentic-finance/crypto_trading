"""SMC 3-confluence — relaxed version of smc_5confluence.

Long when:
  1. Bullish sweep in last 10 bars (key trigger)
  2. trend_state == UP OR NEUTRAL (allow start of uptrend)
  3. price in DISCOUNT zone

Short = mirror.

This is the more *realistic* SMC framework: 3 conditions = ~5x more
signals than 5-confluence, but each carries less conviction.
"""

from cyqnt_trd.blocks import strategy
from cyqnt_trd.blocks.smc_structure import bos_choch_detect
from cyqnt_trd.blocks.smc_liquidity import (
    liquidity_sweep_detect,
    premium_discount_zone,
)


def _rolling_any(series, window: int, value: str):
    return (series == value).rolling(window=window, min_periods=1).sum().fillna(0).gt(0)


def make_signals(df):
    bos = bos_choch_detect(df, swing_lookback=5)
    sweep = liquidity_sweep_detect(df, swing_lookback=5)
    pd_zone = premium_discount_zone(df, swing_lookback=5)

    # Long
    bull_sweep_recent = _rolling_any(sweep["sweep_direction"], 10, "BULL")
    not_downtrend = bos["trend_state"] != "DOWN"
    in_discount = pd_zone["current_zone"] == "DISCOUNT"

    long_signal = bull_sweep_recent & not_downtrend & in_discount

    # Short
    bear_sweep_recent = _rolling_any(sweep["sweep_direction"], 10, "BEAR")
    not_uptrend = bos["trend_state"] != "UP"
    in_premium = pd_zone["current_zone"] == "PREMIUM"

    short_signal = bear_sweep_recent & not_uptrend & in_premium

    return long_signal, short_signal


strategy.register("smc_3confluence_v1", make_signals)
