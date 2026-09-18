"""
Pure-NumPy signal kernels for market-only strategies.

The kernels in this module only consume historical arrays that are visible at
decision time. They intentionally avoid any look-ahead fields so the same math
can be reused by conservative backtests and, later, incremental runtimes.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

TARGET_SHORT = np.int8(-1)
TARGET_FLAT = np.int8(0)
TARGET_LONG = np.int8(1)
TARGET_KEEP = np.int8(2)
SIGNAL_SELL = np.int8(-1)
SIGNAL_NONE = np.int8(0)
SIGNAL_BUY = np.int8(1)


def _ema_series(values: np.ndarray, period: int) -> np.ndarray:
    ema_values = np.zeros(values.shape[0], dtype=np.float64)
    if values.shape[0] == 0 or period < 1:
        return ema_values

    alpha = 2.0 / (period + 1.0)
    ema = values[0]
    ema_values[0] = ema
    for index in range(1, values.shape[0]):
        ema = alpha * values[index] + (1.0 - alpha) * ema
        ema_values[index] = ema
    return ema_values


def _true_range_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    true_ranges = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0:
        return true_ranges

    true_ranges[0] = highs[0] - lows[0]
    for index in range(1, closes.shape[0]):
        high_low = highs[index] - lows[index]
        high_close = abs(highs[index] - closes[index - 1])
        low_close = abs(lows[index] - closes[index - 1])
        true_range = high_low
        if high_close > true_range:
            true_range = high_close
        if low_close > true_range:
            true_range = low_close
        true_ranges[index] = true_range
    return true_ranges


def _adx_components(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    adx_values = np.zeros(closes.shape[0], dtype=np.float64)
    plus_di_values = np.zeros(closes.shape[0], dtype=np.float64)
    minus_di_values = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0 or period < 2:
        return adx_values, plus_di_values, minus_di_values

    plus_dm = np.zeros(closes.shape[0], dtype=np.float64)
    minus_dm = np.zeros(closes.shape[0], dtype=np.float64)
    true_ranges = _true_range_series(highs, lows, closes)
    dx_values = np.zeros(closes.shape[0], dtype=np.float64)

    for index in range(1, closes.shape[0]):
        up_move = highs[index] - highs[index - 1]
        down_move = lows[index - 1] - lows[index]
        if up_move > down_move and up_move > 0.0:
            plus_dm[index] = up_move
        elif down_move > up_move and down_move > 0.0:
            minus_dm[index] = down_move

    tr_prefix = np.zeros(closes.shape[0] + 1, dtype=np.float64)
    plus_prefix = np.zeros(closes.shape[0] + 1, dtype=np.float64)
    minus_prefix = np.zeros(closes.shape[0] + 1, dtype=np.float64)
    dx_prefix = np.zeros(closes.shape[0] + 1, dtype=np.float64)
    for index in range(closes.shape[0]):
        tr_prefix[index + 1] = tr_prefix[index] + true_ranges[index]
        plus_prefix[index + 1] = plus_prefix[index] + plus_dm[index]
        minus_prefix[index + 1] = minus_prefix[index] + minus_dm[index]

    for index in range(closes.shape[0]):
        if index + 1 < period:
            continue
        tr_sum = tr_prefix[index + 1] - tr_prefix[index + 1 - period]
        if tr_sum <= 0.0:
            continue
        plus_sum = plus_prefix[index + 1] - plus_prefix[index + 1 - period]
        minus_sum = minus_prefix[index + 1] - minus_prefix[index + 1 - period]
        plus_di = 100.0 * plus_sum / tr_sum
        minus_di = 100.0 * minus_sum / tr_sum
        plus_di_values[index] = plus_di
        minus_di_values[index] = minus_di
        denominator = plus_di + minus_di
        if denominator > 0.0:
            dx_values[index] = 100.0 * abs(plus_di - minus_di) / denominator
        dx_prefix[index + 1] = dx_prefix[index] + dx_values[index]

    for index in range(closes.shape[0]):
        if index + 1 < 2 * period - 1:
            continue
        adx_values[index] = (dx_prefix[index + 1] - dx_prefix[index + 1 - period]) / period

    return adx_values, plus_di_values, minus_di_values


def moving_average_cross_signal_rows(
    closes: np.ndarray,
    fast_window: int,
    slow_window: int,
    entry_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = np.full(closes.shape[0], SIGNAL_NONE, dtype=np.int8)
    strengths = np.zeros(closes.shape[0], dtype=np.float64)
    fast_values = np.zeros(closes.shape[0], dtype=np.float64)
    slow_values = np.zeros(closes.shape[0], dtype=np.float64)
    spreads = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0 or fast_window < 1 or slow_window <= fast_window:
        return directions, strengths, fast_values, slow_values, spreads

    fast_sum = 0.0
    slow_sum = 0.0
    for index in range(closes.shape[0]):
        close_value = closes[index]
        fast_sum += close_value
        slow_sum += close_value
        if index >= fast_window:
            fast_sum -= closes[index - fast_window]
        if index >= slow_window:
            slow_sum -= closes[index - slow_window]
        if index + 1 < slow_window:
            continue

        slow_sma = slow_sum / slow_window
        if slow_sma == 0.0:
            continue

        fast_sma = fast_sum / fast_window
        fast_values[index] = fast_sma
        slow_values[index] = slow_sma
        spread = (fast_sma - slow_sma) / slow_sma
        spreads[index] = spread
        if spread > entry_threshold:
            directions[index] = SIGNAL_BUY
            strengths[index] = abs(spread)
        elif spread < -entry_threshold:
            directions[index] = SIGNAL_SELL
            strengths[index] = abs(spread)
    return directions, strengths, fast_values, slow_values, spreads


def moving_average_cross_target_updates(
    closes: np.ndarray,
    fast_window: int,
    slow_window: int,
    entry_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions, strengths, _, _, _ = moving_average_cross_signal_rows(
        closes,
        fast_window,
        slow_window,
        entry_threshold,
    )
    target_updates = np.full(closes.shape[0], TARGET_KEEP, dtype=np.int8)
    for index in range(closes.shape[0]):
        if directions[index] == SIGNAL_BUY:
            target_updates[index] = TARGET_LONG
        elif directions[index] == SIGNAL_SELL:
            target_updates[index] = TARGET_FLAT
    return target_updates, strengths


def price_moving_average_signal_rows(
    closes: np.ndarray,
    period: int,
    entry_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = np.full(closes.shape[0], SIGNAL_NONE, dtype=np.int8)
    strengths = np.zeros(closes.shape[0], dtype=np.float64)
    current_ma_values = np.zeros(closes.shape[0], dtype=np.float64)
    prev_ma_values = np.zeros(closes.shape[0], dtype=np.float64)
    spreads = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0 or period < 2:
        return directions, strengths, current_ma_values, prev_ma_values, spreads

    prefix = np.zeros(closes.shape[0] + 1, dtype=np.float64)
    for index in range(closes.shape[0]):
        prefix[index + 1] = prefix[index] + closes[index]

    for index in range(closes.shape[0]):
        if index < period + 1:
            continue

        current_price = closes[index]
        prev_price = closes[index - 1]
        current_ma = (prefix[index] - prefix[index - period]) / period
        prev_ma = (prefix[index - 1] - prefix[index - 1 - period]) / period
        current_ma_values[index] = current_ma
        prev_ma_values[index] = prev_ma
        if current_ma == 0.0 or prev_ma == 0.0:
            continue

        upward_cross = prev_price <= prev_ma and current_price > current_ma
        downward_cross = prev_price >= prev_ma and current_price < current_ma
        if not upward_cross and not downward_cross:
            continue

        spread = (current_price - current_ma) / current_ma
        spreads[index] = spread
        if abs(spread) <= entry_threshold:
            continue

        directions[index] = SIGNAL_BUY if upward_cross else SIGNAL_SELL
        strengths[index] = abs(spread)
    return directions, strengths, current_ma_values, prev_ma_values, spreads


def price_moving_average_target_updates(
    closes: np.ndarray,
    period: int,
    entry_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions, strengths, _, _, _ = price_moving_average_signal_rows(
        closes,
        period,
        entry_threshold,
    )
    target_updates = np.full(closes.shape[0], TARGET_KEEP, dtype=np.int8)
    for index in range(closes.shape[0]):
        if directions[index] == SIGNAL_BUY:
            target_updates[index] = TARGET_LONG
        elif directions[index] == SIGNAL_SELL:
            target_updates[index] = TARGET_FLAT
    return target_updates, strengths


def rsi_reversion_signal_rows(
    closes: np.ndarray,
    period: int,
    oversold: float,
    overbought: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    directions = np.full(closes.shape[0], SIGNAL_NONE, dtype=np.int8)
    strengths = np.zeros(closes.shape[0], dtype=np.float64)
    rsi_values = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0 or period < 2:
        return directions, strengths, rsi_values

    for index in range(period, closes.shape[0]):
        gains = 0.0
        losses = 0.0
        for step in range(index - period + 1, index + 1):
            delta = closes[step] - closes[step - 1]
            if delta > 0.0:
                gains += delta
            elif delta < 0.0:
                losses += -delta

        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0.0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_values[index] = rsi

        if rsi < oversold:
            directions[index] = SIGNAL_BUY
            strengths[index] = (oversold - rsi) / max(oversold, 1e-9)
        elif rsi > overbought:
            directions[index] = SIGNAL_SELL
            strengths[index] = (rsi - overbought) / max(100.0 - overbought, 1e-9)
    return directions, strengths, rsi_values


def rsi_reversion_target_updates(
    closes: np.ndarray,
    period: int,
    oversold: float,
    overbought: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions, strengths, _ = rsi_reversion_signal_rows(
        closes,
        period,
        oversold,
        overbought,
    )
    target_updates = np.full(closes.shape[0], TARGET_KEEP, dtype=np.int8)
    for index in range(closes.shape[0]):
        if directions[index] == SIGNAL_BUY:
            target_updates[index] = TARGET_LONG
        elif directions[index] == SIGNAL_SELL:
            target_updates[index] = TARGET_FLAT
    return target_updates, strengths


def donchian_breakout_signal_rows(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    lookback_window: int,
    breakout_buffer_bps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = np.full(closes.shape[0], SIGNAL_NONE, dtype=np.int8)
    strengths = np.zeros(closes.shape[0], dtype=np.float64)
    upper_band_values = np.zeros(closes.shape[0], dtype=np.float64)
    lower_band_values = np.zeros(closes.shape[0], dtype=np.float64)
    breakout_bps_values = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0 or lookback_window < 1:
        return directions, strengths, upper_band_values, lower_band_values, breakout_bps_values

    upper_multiplier = 1.0 + breakout_buffer_bps / 10_000.0
    lower_multiplier = 1.0 - breakout_buffer_bps / 10_000.0

    for index in range(closes.shape[0]):
        if index < lookback_window:
            continue

        upper_band = highs[index - lookback_window]
        lower_band = lows[index - lookback_window]
        for window_index in range(index - lookback_window + 1, index):
            if highs[window_index] > upper_band:
                upper_band = highs[window_index]
            if lows[window_index] < lower_band:
                lower_band = lows[window_index]

        upper_band_values[index] = upper_band
        lower_band_values[index] = lower_band

        upper_trigger = upper_band * upper_multiplier
        lower_trigger = lower_band * lower_multiplier
        close_value = closes[index]

        if upper_trigger > 0.0 and close_value > upper_trigger:
            breakout_bps = ((close_value - upper_trigger) / upper_trigger) * 10_000.0
            directions[index] = SIGNAL_BUY
            strengths[index] = abs(breakout_bps)
            breakout_bps_values[index] = breakout_bps
        elif lower_trigger > 0.0 and close_value < lower_trigger:
            breakout_bps = ((lower_trigger - close_value) / lower_trigger) * 10_000.0
            directions[index] = SIGNAL_SELL
            strengths[index] = abs(breakout_bps)
            breakout_bps_values[index] = -breakout_bps
    return directions, strengths, upper_band_values, lower_band_values, breakout_bps_values


def donchian_breakout_target_updates(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    lookback_window: int,
    breakout_buffer_bps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions, strengths, _, _, _ = donchian_breakout_signal_rows(
        closes,
        highs,
        lows,
        lookback_window,
        breakout_buffer_bps,
    )
    target_updates = np.full(closes.shape[0], TARGET_KEEP, dtype=np.int8)
    previous_direction = SIGNAL_NONE
    for index in range(closes.shape[0]):
        current_direction = directions[index]
        if current_direction == SIGNAL_NONE:
            previous_direction = SIGNAL_NONE
            continue
        if current_direction == previous_direction:
            continue
        if current_direction == SIGNAL_BUY:
            target_updates[index] = TARGET_LONG
        elif current_direction == SIGNAL_SELL:
            target_updates[index] = TARGET_SHORT
        previous_direction = current_direction
    return target_updates, strengths


def oi_funding_breakout_signal_rows(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    oi_change_bps: np.ndarray,
    funding_rate_bps: np.ndarray,
    lookback_window: int,
    breakout_buffer_bps: float,
    oi_threshold_bps: float,
    max_funding_rate_bps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = np.full(closes.shape[0], SIGNAL_NONE, dtype=np.int8)
    strengths = np.zeros(closes.shape[0], dtype=np.float64)
    upper_band_values = np.zeros(closes.shape[0], dtype=np.float64)
    lower_band_values = np.zeros(closes.shape[0], dtype=np.float64)
    breakout_bps_values = np.zeros(closes.shape[0], dtype=np.float64)
    oi_values = np.zeros(closes.shape[0], dtype=np.float64)
    funding_values = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0 or lookback_window < 1:
        return (
            directions,
            strengths,
            upper_band_values,
            lower_band_values,
            breakout_bps_values,
            oi_values,
            funding_values,
        )

    upper_multiplier = 1.0 + breakout_buffer_bps / 10_000.0
    lower_multiplier = 1.0 - breakout_buffer_bps / 10_000.0

    for index in range(closes.shape[0]):
        if index < lookback_window:
            continue

        oi_value = oi_change_bps[index]
        funding_value = funding_rate_bps[index]
        if np.isnan(oi_value) or np.isnan(funding_value):
            continue

        oi_values[index] = oi_value
        funding_values[index] = funding_value
        if oi_value < oi_threshold_bps:
            continue

        upper_band = highs[index - lookback_window]
        lower_band = lows[index - lookback_window]
        for window_index in range(index - lookback_window + 1, index):
            if highs[window_index] > upper_band:
                upper_band = highs[window_index]
            if lows[window_index] < lower_band:
                lower_band = lows[window_index]

        upper_band_values[index] = upper_band
        lower_band_values[index] = lower_band

        upper_trigger = upper_band * upper_multiplier
        lower_trigger = lower_band * lower_multiplier
        close_value = closes[index]

        if upper_trigger > 0.0 and close_value > upper_trigger and funding_value <= max_funding_rate_bps:
            breakout_bps = ((close_value - upper_trigger) / upper_trigger) * 10_000.0
            directions[index] = SIGNAL_BUY
            breakout_bps_values[index] = breakout_bps
            strengths[index] = abs(breakout_bps) + max(oi_value - oi_threshold_bps, 0.0) / 100.0
        elif lower_trigger > 0.0 and close_value < lower_trigger and funding_value >= -max_funding_rate_bps:
            breakout_bps = ((lower_trigger - close_value) / lower_trigger) * 10_000.0
            directions[index] = SIGNAL_SELL
            breakout_bps_values[index] = -breakout_bps
            strengths[index] = abs(breakout_bps) + max(oi_value - oi_threshold_bps, 0.0) / 100.0
    return (
        directions,
        strengths,
        upper_band_values,
        lower_band_values,
        breakout_bps_values,
        oi_values,
        funding_values,
    )


def oi_funding_breakout_target_updates(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    oi_change_bps: np.ndarray,
    funding_rate_bps: np.ndarray,
    lookback_window: int,
    breakout_buffer_bps: float,
    oi_threshold_bps: float,
    max_funding_rate_bps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions, strengths, _, _, _, _, _ = oi_funding_breakout_signal_rows(
        closes,
        highs,
        lows,
        oi_change_bps,
        funding_rate_bps,
        lookback_window,
        breakout_buffer_bps,
        oi_threshold_bps,
        max_funding_rate_bps,
    )
    target_updates = np.full(closes.shape[0], TARGET_KEEP, dtype=np.int8)
    previous_direction = SIGNAL_NONE
    for index in range(closes.shape[0]):
        current_direction = directions[index]
        if current_direction == SIGNAL_NONE:
            previous_direction = SIGNAL_NONE
            continue
        if current_direction == previous_direction:
            continue
        if current_direction == SIGNAL_BUY:
            target_updates[index] = TARGET_LONG
        elif current_direction == SIGNAL_SELL:
            target_updates[index] = TARGET_SHORT
        previous_direction = current_direction
    return target_updates, strengths


def liquidation_reversal_signal_rows(
    long_liq_notional_usd: np.ndarray,
    short_liq_notional_usd: np.ndarray,
    long_liquidation_threshold_usd: float,
    short_liquidation_threshold_usd: float,
    liquidation_imbalance_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = np.full(long_liq_notional_usd.shape[0], SIGNAL_NONE, dtype=np.int8)
    strengths = np.zeros(long_liq_notional_usd.shape[0], dtype=np.float64)
    total_notional_values = np.zeros(long_liq_notional_usd.shape[0], dtype=np.float64)
    imbalance_values = np.zeros(long_liq_notional_usd.shape[0], dtype=np.float64)
    if long_liq_notional_usd.shape[0] == 0:
        return directions, strengths, total_notional_values, imbalance_values

    for index in range(long_liq_notional_usd.shape[0]):
        long_notional = long_liq_notional_usd[index]
        short_notional = short_liq_notional_usd[index]
        if np.isnan(long_notional):
            long_notional = 0.0
        if np.isnan(short_notional):
            short_notional = 0.0
        total_notional = long_notional + short_notional
        total_notional_values[index] = total_notional
        if total_notional <= 0.0:
            continue

        imbalance = (short_notional - long_notional) / total_notional
        imbalance_values[index] = imbalance
        long_ratio = long_notional / total_notional
        short_ratio = short_notional / total_notional
        if long_notional >= long_liquidation_threshold_usd and long_ratio >= liquidation_imbalance_ratio:
            directions[index] = SIGNAL_BUY
            strengths[index] = (long_notional / max(long_liquidation_threshold_usd, 1.0)) * long_ratio
        elif short_notional >= short_liquidation_threshold_usd and short_ratio >= liquidation_imbalance_ratio:
            directions[index] = SIGNAL_SELL
            strengths[index] = (short_notional / max(short_liquidation_threshold_usd, 1.0)) * short_ratio
    return directions, strengths, total_notional_values, imbalance_values


def liquidation_reversal_target_updates(
    long_liq_notional_usd: np.ndarray,
    short_liq_notional_usd: np.ndarray,
    long_liquidation_threshold_usd: float,
    short_liquidation_threshold_usd: float,
    liquidation_imbalance_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions, strengths, _, _ = liquidation_reversal_signal_rows(
        long_liq_notional_usd,
        short_liq_notional_usd,
        long_liquidation_threshold_usd,
        short_liquidation_threshold_usd,
        liquidation_imbalance_ratio,
    )
    target_updates = np.full(long_liq_notional_usd.shape[0], TARGET_KEEP, dtype=np.int8)
    previous_direction = SIGNAL_NONE
    for index in range(long_liq_notional_usd.shape[0]):
        current_direction = directions[index]
        if current_direction == SIGNAL_NONE:
            previous_direction = SIGNAL_NONE
            continue
        if current_direction == previous_direction:
            continue
        if current_direction == SIGNAL_BUY:
            target_updates[index] = TARGET_LONG
        elif current_direction == SIGNAL_SELL:
            target_updates[index] = TARGET_SHORT
        previous_direction = current_direction
    return target_updates, strengths


def adx_trend_strength_signal_rows(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    period: int,
    adx_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = np.full(closes.shape[0], SIGNAL_NONE, dtype=np.int8)
    strengths = np.zeros(closes.shape[0], dtype=np.float64)
    adx_values, plus_di_values, minus_di_values = _adx_components(highs, lows, closes, period)
    di_spread_values = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0 or period < 2:
        return directions, strengths, adx_values, plus_di_values, minus_di_values

    for index in range(closes.shape[0]):
        adx_value = adx_values[index]
        if adx_value <= adx_threshold:
            continue
        di_spread = plus_di_values[index] - minus_di_values[index]
        di_spread_values[index] = di_spread
        if di_spread > 0.0:
            directions[index] = SIGNAL_BUY
            strengths[index] = ((adx_value - adx_threshold) / max(adx_threshold, 1e-9)) + abs(di_spread) / 100.0
        elif di_spread < 0.0:
            directions[index] = SIGNAL_SELL
            strengths[index] = ((adx_value - adx_threshold) / max(adx_threshold, 1e-9)) + abs(di_spread) / 100.0
    return directions, strengths, adx_values, plus_di_values, minus_di_values


def adx_trend_strength_target_updates(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    period: int,
    adx_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions, strengths, _, _, _ = adx_trend_strength_signal_rows(
        closes,
        highs,
        lows,
        period,
        adx_threshold,
    )
    target_updates = np.full(closes.shape[0], TARGET_KEEP, dtype=np.int8)
    previous_direction = SIGNAL_NONE
    for index in range(closes.shape[0]):
        current_direction = directions[index]
        if current_direction == SIGNAL_NONE:
            previous_direction = SIGNAL_NONE
            continue
        if current_direction == previous_direction:
            continue
        if current_direction == SIGNAL_BUY:
            target_updates[index] = TARGET_LONG
        elif current_direction == SIGNAL_SELL:
            target_updates[index] = TARGET_SHORT
        previous_direction = current_direction
    return target_updates, strengths


def atr_breakout_signal_rows(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    ma_period: int,
    atr_period: int,
    atr_multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = np.full(closes.shape[0], SIGNAL_NONE, dtype=np.int8)
    strengths = np.zeros(closes.shape[0], dtype=np.float64)
    ma_values = np.zeros(closes.shape[0], dtype=np.float64)
    atr_values = np.zeros(closes.shape[0], dtype=np.float64)
    upper_channel = np.zeros(closes.shape[0], dtype=np.float64)
    lower_channel = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0 or ma_period < 1 or atr_period < 1 or atr_multiplier < 0.0:
        return directions, strengths, ma_values, atr_values, upper_channel, lower_channel

    tr_values = _true_range_series(highs, lows, closes)
    close_prefix = np.zeros(closes.shape[0] + 1, dtype=np.float64)
    tr_prefix = np.zeros(closes.shape[0] + 1, dtype=np.float64)
    for index in range(closes.shape[0]):
        close_prefix[index + 1] = close_prefix[index] + closes[index]
        tr_prefix[index + 1] = tr_prefix[index] + tr_values[index]

    min_index = ma_period
    if atr_period > min_index:
        min_index = atr_period
    for index in range(closes.shape[0]):
        if index + 1 < min_index:
            continue

        ma_value = (close_prefix[index + 1] - close_prefix[index + 1 - ma_period]) / ma_period
        atr_value = (tr_prefix[index + 1] - tr_prefix[index + 1 - atr_period]) / atr_period
        ma_values[index] = ma_value
        atr_values[index] = atr_value
        upper = ma_value + atr_multiplier * atr_value
        lower = ma_value - atr_multiplier * atr_value
        upper_channel[index] = upper
        lower_channel[index] = lower

        if atr_value <= 0.0:
            continue
        if closes[index] > upper:
            directions[index] = SIGNAL_BUY
            strengths[index] = (closes[index] - upper) / atr_value
        elif closes[index] < lower:
            directions[index] = SIGNAL_SELL
            strengths[index] = (lower - closes[index]) / atr_value
    return directions, strengths, ma_values, atr_values, upper_channel, lower_channel


def atr_breakout_target_updates(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    ma_period: int,
    atr_period: int,
    atr_multiplier: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions, strengths, _, _, _, _ = atr_breakout_signal_rows(
        closes,
        highs,
        lows,
        ma_period,
        atr_period,
        atr_multiplier,
    )
    target_updates = np.full(closes.shape[0], TARGET_KEEP, dtype=np.int8)
    previous_direction = SIGNAL_NONE
    for index in range(closes.shape[0]):
        current_direction = directions[index]
        if current_direction == SIGNAL_NONE:
            previous_direction = SIGNAL_NONE
            continue
        if current_direction == previous_direction:
            continue
        if current_direction == SIGNAL_BUY:
            target_updates[index] = TARGET_LONG
        elif current_direction == SIGNAL_SELL:
            target_updates[index] = TARGET_SHORT
        previous_direction = current_direction
    return target_updates, strengths


def bollinger_mean_reversion_signal_rows(
    closes: np.ndarray,
    period: int,
    stddev_multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = np.full(closes.shape[0], SIGNAL_NONE, dtype=np.int8)
    strengths = np.zeros(closes.shape[0], dtype=np.float64)
    mid_values = np.zeros(closes.shape[0], dtype=np.float64)
    upper_values = np.zeros(closes.shape[0], dtype=np.float64)
    lower_values = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0 or period < 2 or stddev_multiplier <= 0.0:
        return directions, strengths, mid_values, upper_values, lower_values

    prefix = np.zeros(closes.shape[0] + 1, dtype=np.float64)
    prefix_sq = np.zeros(closes.shape[0] + 1, dtype=np.float64)
    for index in range(closes.shape[0]):
        prefix[index + 1] = prefix[index] + closes[index]
        prefix_sq[index + 1] = prefix_sq[index] + closes[index] * closes[index]

    for index in range(closes.shape[0]):
        if index + 1 < period:
            continue
        total = prefix[index + 1] - prefix[index + 1 - period]
        total_sq = prefix_sq[index + 1] - prefix_sq[index + 1 - period]
        mean = total / period
        variance = (total_sq / period) - mean * mean
        if variance < 0.0:
            variance = 0.0
        stddev = np.sqrt(variance)
        mid_values[index] = mean
        upper_values[index] = mean + stddev_multiplier * stddev
        lower_values[index] = mean - stddev_multiplier * stddev
        if stddev <= 0.0:
            continue
        if closes[index] < lower_values[index]:
            directions[index] = SIGNAL_BUY
            strengths[index] = abs((closes[index] - mean) / stddev)
        elif closes[index] > upper_values[index]:
            directions[index] = SIGNAL_SELL
            strengths[index] = abs((closes[index] - mean) / stddev)
    return directions, strengths, mid_values, upper_values, lower_values


def bollinger_mean_reversion_target_updates(
    closes: np.ndarray,
    period: int,
    stddev_multiplier: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions, strengths, _, _, _ = bollinger_mean_reversion_signal_rows(
        closes,
        period,
        stddev_multiplier,
    )
    target_updates = np.full(closes.shape[0], TARGET_KEEP, dtype=np.int8)
    previous_direction = SIGNAL_NONE
    for index in range(closes.shape[0]):
        current_direction = directions[index]
        if current_direction == SIGNAL_NONE:
            previous_direction = SIGNAL_NONE
            continue
        if current_direction == previous_direction:
            continue
        if current_direction == SIGNAL_BUY:
            target_updates[index] = TARGET_LONG
        elif current_direction == SIGNAL_SELL:
            target_updates[index] = TARGET_SHORT
        previous_direction = current_direction
    return target_updates, strengths


def macd_trend_follow_signal_rows(
    closes: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    histogram_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = np.full(closes.shape[0], SIGNAL_NONE, dtype=np.int8)
    strengths = np.zeros(closes.shape[0], dtype=np.float64)
    macd_line = np.zeros(closes.shape[0], dtype=np.float64)
    signal_line = np.zeros(closes.shape[0], dtype=np.float64)
    histogram = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0 or fast_period < 1 or slow_period <= fast_period or signal_period < 1:
        return directions, strengths, macd_line, signal_line, histogram

    fast_ema = _ema_series(closes, fast_period)
    slow_ema = _ema_series(closes, slow_period)
    for index in range(closes.shape[0]):
        macd_line[index] = fast_ema[index] - slow_ema[index]
    signal_line = _ema_series(macd_line, signal_period)

    warmup = slow_period + signal_period - 1
    for index in range(closes.shape[0]):
        histogram[index] = macd_line[index] - signal_line[index]
        if index + 1 < warmup:
            continue
        histogram_ratio = histogram[index] / max(abs(closes[index]), 1e-9)
        if histogram_ratio > histogram_threshold:
            directions[index] = SIGNAL_BUY
            strengths[index] = abs(histogram_ratio)
        elif histogram_ratio < -histogram_threshold:
            directions[index] = SIGNAL_SELL
            strengths[index] = abs(histogram_ratio)
    return directions, strengths, macd_line, signal_line, histogram


def macd_trend_follow_target_updates(
    closes: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    histogram_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions, strengths, _, _, _ = macd_trend_follow_signal_rows(
        closes,
        fast_period,
        slow_period,
        signal_period,
        histogram_threshold,
    )
    target_updates = np.full(closes.shape[0], TARGET_KEEP, dtype=np.int8)
    previous_direction = SIGNAL_NONE
    for index in range(closes.shape[0]):
        current_direction = directions[index]
        if current_direction == SIGNAL_NONE:
            previous_direction = SIGNAL_NONE
            continue
        if current_direction == previous_direction:
            continue
        if current_direction == SIGNAL_BUY:
            target_updates[index] = TARGET_LONG
        elif current_direction == SIGNAL_SELL:
            target_updates[index] = TARGET_SHORT
        previous_direction = current_direction
    return target_updates, strengths


def multi_timeframe_ma_spread_signal_rows(
    primary_timestamps: np.ndarray,
    primary_closes: np.ndarray,
    secondary_timestamps: np.ndarray,
    secondary_closes: np.ndarray,
    primary_period: int,
    secondary_period: int,
    threshold_bps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = np.full(primary_closes.shape[0], SIGNAL_NONE, dtype=np.int8)
    strengths = np.zeros(primary_closes.shape[0], dtype=np.float64)
    primary_ma_values = np.zeros(primary_closes.shape[0], dtype=np.float64)
    secondary_ma_aligned = np.zeros(primary_closes.shape[0], dtype=np.float64)
    spread_bps_values = np.zeros(primary_closes.shape[0], dtype=np.float64)
    if primary_closes.shape[0] == 0 or secondary_closes.shape[0] == 0:
        return directions, strengths, primary_ma_values, secondary_ma_aligned, spread_bps_values
    if primary_period < 1 or secondary_period < 1:
        return directions, strengths, primary_ma_values, secondary_ma_aligned, spread_bps_values

    primary_prefix = np.zeros(primary_closes.shape[0] + 1, dtype=np.float64)
    secondary_prefix = np.zeros(secondary_closes.shape[0] + 1, dtype=np.float64)
    for index in range(primary_closes.shape[0]):
        primary_prefix[index + 1] = primary_prefix[index] + primary_closes[index]
    for index in range(secondary_closes.shape[0]):
        secondary_prefix[index + 1] = secondary_prefix[index] + secondary_closes[index]

    secondary_cursor = -1
    threshold_ratio = threshold_bps / 10_000.0

    for primary_index in range(primary_closes.shape[0]):
        while secondary_cursor + 1 < secondary_timestamps.shape[0] and secondary_timestamps[secondary_cursor + 1] <= primary_timestamps[primary_index]:
            secondary_cursor += 1
        if primary_index + 1 < primary_period:
            continue
        if secondary_cursor + 1 < secondary_period:
            continue

        primary_ma = (primary_prefix[primary_index + 1] - primary_prefix[primary_index + 1 - primary_period]) / primary_period
        secondary_ma = (
            secondary_prefix[secondary_cursor + 1] - secondary_prefix[secondary_cursor + 1 - secondary_period]
        ) / secondary_period
        primary_ma_values[primary_index] = primary_ma
        secondary_ma_aligned[primary_index] = secondary_ma
        if secondary_ma == 0.0:
            continue

        spread_ratio = (primary_ma - secondary_ma) / secondary_ma
        spread_bps = spread_ratio * 10_000.0
        spread_bps_values[primary_index] = spread_bps
        if spread_ratio > threshold_ratio:
            directions[primary_index] = SIGNAL_BUY
            strengths[primary_index] = abs(spread_ratio)
        elif spread_ratio < -threshold_ratio:
            directions[primary_index] = SIGNAL_SELL
            strengths[primary_index] = abs(spread_ratio)
    return directions, strengths, primary_ma_values, secondary_ma_aligned, spread_bps_values


def multi_timeframe_ma_spread_target_updates(
    primary_timestamps: np.ndarray,
    primary_closes: np.ndarray,
    secondary_timestamps: np.ndarray,
    secondary_closes: np.ndarray,
    primary_period: int,
    secondary_period: int,
    threshold_bps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions, strengths, _, _, _ = multi_timeframe_ma_spread_signal_rows(
        primary_timestamps,
        primary_closes,
        secondary_timestamps,
        secondary_closes,
        primary_period,
        secondary_period,
        threshold_bps,
    )
    target_updates = np.full(primary_closes.shape[0], TARGET_KEEP, dtype=np.int8)
    previous_direction = SIGNAL_NONE
    for index in range(primary_closes.shape[0]):
        current_direction = directions[index]
        if current_direction == SIGNAL_NONE:
            previous_direction = SIGNAL_NONE
            continue
        if current_direction == previous_direction:
            continue
        if current_direction == SIGNAL_BUY:
            target_updates[index] = TARGET_LONG
        elif current_direction == SIGNAL_SELL:
            target_updates[index] = TARGET_SHORT
        previous_direction = current_direction
    return target_updates, strengths
