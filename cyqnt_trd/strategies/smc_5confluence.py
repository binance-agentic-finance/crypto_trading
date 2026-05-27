"""SMC 5-confluence — Smart Money Concepts long/short strategy.

Long when 5 SMC conditions align:
  1. Bullish liquidity sweep in last 10 bars (institutions cleared retail stops)
  2. trend_state == UP (BOS confirms uptrend after CHoCH)
  3. Bullish Order Block nearby (institutional buy zone)
  4. Price in DISCOUNT zone (lower half of swing range — buyer's advantage)
  5. Bullish FVG in last 5 bars (gap confirming bullish momentum)

Short = mirror conditions for bearish.

This is a *high-conviction, low-frequency* strategy. Expected:
  - 5-15 setups per month on 1h timeframe
  - Win rate 40-50%
  - Risk:Reward 1:3 to 1:5
  - Net edge from RR, not from win rate

Risk management (handled by framework):
  - Stop loss: structure-based (Order Block low for long, OB high for short)
  - Take profit: opposing liquidity zone
  - Max position: 15% equity
"""

from cyqnt_trd.blocks import strategy
from cyqnt_trd.blocks.smc_structure import (
    bos_choch_detect,
    fair_value_gap,
    order_block_detect,
)
from cyqnt_trd.blocks.smc_liquidity import (
    liquidity_sweep_detect,
    premium_discount_zone,
)


def _rolling_any(series, window: int, value: str):
    """Returns boolean Series: True if `value` appeared in last `window` bars."""
    return (series == value).rolling(window=window, min_periods=1).sum().fillna(0).gt(0)


def make_signals(df):
    # === Detect SMC components ===
    fvg = fair_value_gap(df)
    ob = order_block_detect(df, swing_lookback=5)
    bos = bos_choch_detect(df, swing_lookback=5)
    sweep = liquidity_sweep_detect(df, swing_lookback=5)
    pd_zone = premium_discount_zone(df, swing_lookback=5)

    # === LONG conditions ===
    bull_sweep_recent = _rolling_any(sweep["sweep_direction"], 10, "BULL")
    in_uptrend = bos["trend_state"] == "UP"
    bull_ob_present = ob["ob_direction"] == "BULL"
    in_discount = pd_zone["current_zone"] == "DISCOUNT"
    bull_fvg_recent = _rolling_any(fvg["fvg_direction"], 5, "BULL")

    long_signal = (
        bull_sweep_recent
        & in_uptrend
        & bull_ob_present
        & in_discount
        & bull_fvg_recent
    )

    # === SHORT conditions (mirror) ===
    bear_sweep_recent = _rolling_any(sweep["sweep_direction"], 10, "BEAR")
    in_downtrend = bos["trend_state"] == "DOWN"
    bear_ob_present = ob["ob_direction"] == "BEAR"
    in_premium = pd_zone["current_zone"] == "PREMIUM"
    bear_fvg_recent = _rolling_any(fvg["fvg_direction"], 5, "BEAR")

    short_signal = (
        bear_sweep_recent
        & in_downtrend
        & bear_ob_present
        & in_premium
        & bear_fvg_recent
    )

    return long_signal, short_signal


# Register with framework
strategy.register("smc_5confluence_v1", make_signals)
