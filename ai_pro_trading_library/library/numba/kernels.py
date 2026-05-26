"""Numba kernels for the most-used indicator math.

Selected from `cyqnt_trd.standard_bot.signal.numba_kernels` — only the
foundational helpers that show up in 80%+ of strategy paths:
- EMA (also feeds MACD, ADX trend, etc.)
- RSI (mean-reversion family + many composite signals)
- True Range / ATR (volatility filter, stops sizing)
- MACD (trend-follow family)

Decorator falls through to plain Python when numba isn't installed, so
calling code does not need to gate on `[numba]` extra availability —
it just runs slower. `NUMBA_AVAILABLE` exposes the runtime status.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on numba-less envs
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator


# ---------- EMA ----------

@njit(cache=True)
def ema_series(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. alpha = 2/(period+1).

    Mirrors `cyqnt_trd.standard_bot.signal.numba_kernels._ema_series` —
    seeds with values[0] and runs the recursive form bar-by-bar.
    """
    out = np.zeros(values.shape[0], dtype=np.float64)
    if values.shape[0] == 0 or period < 1:
        return out
    alpha = 2.0 / (period + 1.0)
    ema = values[0]
    out[0] = ema
    for i in range(1, values.shape[0]):
        ema = alpha * values[i] + (1.0 - alpha) * ema
        out[i] = ema
    return out


# ---------- RSI ----------

@njit(cache=True)
def rsi_series(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling-window RSI as in standard_bot.signal.numba_kernels.
    rsi_reversion_signal_rows.

    Bars before `period` are 0 (insufficient data sentinel; the canonical
    pandas indicator returns NaN — switch via NaN if you need that
    semantic).
    """
    out = np.zeros(values.shape[0], dtype=np.float64)
    if values.shape[0] == 0 or period < 2:
        return out
    for i in range(period, values.shape[0]):
        gains = 0.0
        losses = 0.0
        for step in range(i - period + 1, i + 1):
            delta = values[step] - values[step - 1]
            if delta > 0.0:
                gains += delta
            elif delta < 0.0:
                losses += -delta
        if losses == 0.0:
            out[i] = 100.0
        elif gains == 0.0:
            out[i] = 0.0
        else:
            rs = gains / losses
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


# ---------- True Range / ATR ----------

@njit(cache=True)
def true_range_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """Per-bar true range. tr[0] = high[0] - low[0]. From standard_bot
    `_true_range_series`."""
    out = np.zeros(closes.shape[0], dtype=np.float64)
    if closes.shape[0] == 0:
        return out
    out[0] = highs[0] - lows[0]
    for i in range(1, closes.shape[0]):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr = hl
        if hc > tr:
            tr = hc
        if lc > tr:
            tr = lc
        out[i] = tr
    return out


@njit(cache=True)
def atr_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    """Wilder ATR — EMA of true_range with alpha=1/period.

    First `period` bars are 0 (insufficient data); from bar `period`
    onward we run the Wilder smoother:
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    seeded with the simple mean of the first `period` true ranges.
    """
    n = closes.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or period < 1:
        return out
    tr = true_range_series(highs, lows, closes)
    if n < period:
        return out
    seed_sum = 0.0
    for i in range(period):
        seed_sum += tr[i]
    seed = seed_sum / period
    out[period - 1] = seed
    atr = seed
    for i in range(period, n):
        atr = (atr * (period - 1) + tr[i]) / period
        out[i] = atr
    return out


# ---------- MACD ----------

@njit(cache=True)
def macd_series(
    values: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> np.ndarray:
    """MACD as a 3-row 2D array: (macd_line, signal_line, histogram).

    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    """
    n = values.shape[0]
    out = np.zeros((3, n), dtype=np.float64)
    if n == 0:
        return out
    fast_ema = ema_series(values, fast)
    slow_ema = ema_series(values, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema_series(macd_line, signal)
    histogram = macd_line - signal_line
    out[0, :] = macd_line
    out[1, :] = signal_line
    out[2, :] = histogram
    return out


# ---------- SMA ----------

@njit(cache=True)
def sma_series(values: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average. Bars before `period-1` return 0."""
    n = values.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or period < 1:
        return out
    if period == 1:
        for i in range(n):
            out[i] = values[i]
        return out
    window_sum = 0.0
    for i in range(min(period, n)):
        window_sum += values[i]
    if n >= period:
        out[period - 1] = window_sum / period
    for i in range(period, n):
        window_sum += values[i] - values[i - period]
        out[i] = window_sum / period
    return out


# ---------- WMA ----------

@njit(cache=True)
def wma_series(values: np.ndarray, period: int) -> np.ndarray:
    """Linearly weighted moving average. weight[k] = k+1, k=0..period-1."""
    n = values.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or period < 1 or n < period:
        return out
    weight_sum = period * (period + 1) / 2.0
    for i in range(period - 1, n):
        acc = 0.0
        for k in range(period):
            acc += values[i - period + 1 + k] * (k + 1)
        out[i] = acc / weight_sum
    return out


# ---------- Bollinger ----------

@njit(cache=True)
def bollinger_series(
    values: np.ndarray,
    period: int,
    stddev: float,
) -> np.ndarray:
    """Bollinger bands as 3-row 2D array: (upper, middle, lower).

    middle = SMA(period); upper/lower = middle ± stddev * sample_std.
    """
    n = values.shape[0]
    out = np.zeros((3, n), dtype=np.float64)
    if n == 0 or period < 2 or n < period:
        return out
    for i in range(period - 1, n):
        s = 0.0
        for k in range(period):
            s += values[i - period + 1 + k]
        mid = s / period
        var = 0.0
        for k in range(period):
            d = values[i - period + 1 + k] - mid
            var += d * d
        sd = (var / period) ** 0.5
        out[0, i] = mid + stddev * sd
        out[1, i] = mid
        out[2, i] = mid - stddev * sd
    return out


# ---------- Donchian ----------

@njit(cache=True)
def donchian_series(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int,
) -> np.ndarray:
    """Donchian channels: (upper, middle, lower) over `period` bars."""
    n = highs.shape[0]
    out = np.zeros((3, n), dtype=np.float64)
    if n == 0 or period < 1 or n < period:
        return out
    for i in range(period - 1, n):
        hi = highs[i - period + 1]
        lo = lows[i - period + 1]
        for k in range(1, period):
            v = highs[i - period + 1 + k]
            if v > hi:
                hi = v
            v2 = lows[i - period + 1 + k]
            if v2 < lo:
                lo = v2
        out[0, i] = hi
        out[2, i] = lo
        out[1, i] = (hi + lo) / 2.0
    return out


# ---------- Stochastic ----------

@njit(cache=True)
def stochastic_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    k_period: int,
    d_period: int,
) -> np.ndarray:
    """Slow stochastic. Returns (k, d).

    %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = SMA(%K, d_period)
    """
    n = closes.shape[0]
    out = np.zeros((2, n), dtype=np.float64)
    if n == 0 or k_period < 1 or d_period < 1 or n < k_period:
        return out
    raw_k = np.zeros(n, dtype=np.float64)
    for i in range(k_period - 1, n):
        hi = highs[i - k_period + 1]
        lo = lows[i - k_period + 1]
        for k in range(1, k_period):
            v = highs[i - k_period + 1 + k]
            if v > hi:
                hi = v
            v2 = lows[i - k_period + 1 + k]
            if v2 < lo:
                lo = v2
        rng = hi - lo
        if rng <= 0.0:
            raw_k[i] = 50.0
        else:
            raw_k[i] = (closes[i] - lo) / rng * 100.0
    out[0, :] = raw_k
    out[1, :] = sma_series(raw_k, d_period)
    return out


# ---------- StochRSI ----------

@njit(cache=True)
def stochrsi_series(
    values: np.ndarray,
    rsi_period: int,
    k_period: int,
    smooth_k: int,
    smooth_d: int,
) -> np.ndarray:
    """StochRSI: stochastic oscillator over the RSI series.

    Returns (k, d). Range [0, 1].
    """
    n = values.shape[0]
    out = np.zeros((2, n), dtype=np.float64)
    if n == 0 or rsi_period < 2 or k_period < 1:
        return out
    rsi = rsi_series(values, rsi_period)
    raw_k = np.zeros(n, dtype=np.float64)
    for i in range(rsi_period + k_period - 1, n):
        hi = rsi[i - k_period + 1]
        lo = rsi[i - k_period + 1]
        for k in range(1, k_period):
            v = rsi[i - k_period + 1 + k]
            if v > hi:
                hi = v
            if v < lo:
                lo = v
        rng = hi - lo
        if rng <= 0.0:
            raw_k[i] = 0.5
        else:
            raw_k[i] = (rsi[i] - lo) / rng
    if smooth_k > 1:
        raw_k = sma_series(raw_k, smooth_k)
    out[0, :] = raw_k
    out[1, :] = sma_series(raw_k, smooth_d) if smooth_d > 1 else raw_k
    return out


# ---------- ADX (+DI / -DI) ----------

@njit(cache=True)
def adx_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    """ADX with +DI and -DI. Returns 3-row 2D array (adx, plus_di, minus_di).

    Wilder smoothing for TR / +DM / -DM. ADX is the EMA-1/period of DX.
    """
    n = closes.shape[0]
    out = np.zeros((3, n), dtype=np.float64)
    if n == 0 or period < 1 or n < period + 1:
        return out
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        if up > dn and up > 0.0:
            plus_dm[i] = up
        if dn > up and dn > 0.0:
            minus_dm[i] = dn
        h_l = highs[i] - lows[i]
        h_c = abs(highs[i] - closes[i - 1])
        l_c = abs(lows[i] - closes[i - 1])
        max_v = h_l
        if h_c > max_v:
            max_v = h_c
        if l_c > max_v:
            max_v = l_c
        tr[i] = max_v

    smoothed_plus = 0.0
    smoothed_minus = 0.0
    smoothed_tr = 0.0
    for i in range(1, period + 1):
        smoothed_plus += plus_dm[i]
        smoothed_minus += minus_dm[i]
        smoothed_tr += tr[i]

    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    if smoothed_tr > 0.0:
        plus_di[period] = 100.0 * smoothed_plus / smoothed_tr
        minus_di[period] = 100.0 * smoothed_minus / smoothed_tr
    pi_mi = plus_di[period] + minus_di[period]
    if pi_mi > 0.0:
        dx[period] = 100.0 * abs(plus_di[period] - minus_di[period]) / pi_mi

    for i in range(period + 1, n):
        smoothed_plus = smoothed_plus * (period - 1) / period + plus_dm[i]
        smoothed_minus = smoothed_minus * (period - 1) / period + minus_dm[i]
        smoothed_tr = smoothed_tr * (period - 1) / period + tr[i]
        if smoothed_tr > 0.0:
            plus_di[i] = 100.0 * smoothed_plus / smoothed_tr
            minus_di[i] = 100.0 * smoothed_minus / smoothed_tr
        pi_mi = plus_di[i] + minus_di[i]
        if pi_mi > 0.0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / pi_mi

    # ADX = Wilder smooth of DX over `period`
    if 2 * period < n:
        seed_sum = 0.0
        for i in range(period, 2 * period):
            seed_sum += dx[i]
        adx_val = seed_sum / period
        out[0, 2 * period - 1] = adx_val
        for i in range(2 * period, n):
            adx_val = (adx_val * (period - 1) + dx[i]) / period
            out[0, i] = adx_val
    out[1, :] = plus_di
    out[2, :] = minus_di
    return out


# ---------- SuperTrend ----------

@njit(cache=True)
def supertrend_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
    multiplier: float,
) -> np.ndarray:
    """SuperTrend trailing line + direction.

    Returns 2-row 2D array (line, direction). direction = +1 (uptrend) /
    -1 (downtrend).
    """
    n = closes.shape[0]
    out = np.zeros((2, n), dtype=np.float64)
    if n == 0 or period < 1 or n <= period:
        return out
    atr = atr_series(highs, lows, closes, period)
    line = np.zeros(n, dtype=np.float64)
    direction = np.ones(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        median = (highs[i] + lows[i]) / 2.0
        u = median + multiplier * atr[i]
        l = median - multiplier * atr[i]
        if i == period:
            upper[i] = u
            lower[i] = l
            line[i] = l if closes[i] >= median else u
            direction[i] = 1.0 if closes[i] >= median else -1.0
            continue
        # Adjust upper/lower to be ratcheting in the right direction.
        if u < upper[i - 1] or closes[i - 1] > upper[i - 1]:
            upper[i] = u
        else:
            upper[i] = upper[i - 1]
        if l > lower[i - 1] or closes[i - 1] < lower[i - 1]:
            lower[i] = l
        else:
            lower[i] = lower[i - 1]
        # Direction flip rules.
        if direction[i - 1] > 0:
            if closes[i] < lower[i]:
                direction[i] = -1.0
                line[i] = upper[i]
            else:
                direction[i] = 1.0
                line[i] = lower[i]
        else:
            if closes[i] > upper[i]:
                direction[i] = 1.0
                line[i] = lower[i]
            else:
                direction[i] = -1.0
                line[i] = upper[i]
    out[0, :] = line
    out[1, :] = direction
    return out


# ---------- OBV ----------

@njit(cache=True)
def obv_series(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """On-balance volume. Cumulative volume signed by close direction."""
    n = closes.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    obv = 0.0
    out[0] = 0.0
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
        out[i] = obv
    return out


# ---------- VWAP ----------

@njit(cache=True)
def vwap_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Cumulative VWAP from start of series.

    typical_price = (high + low + close) / 3
    vwap[i] = sum(typical * volume) / sum(volume)
    """
    n = closes.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        typical = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_pv += typical * volumes[i]
        cum_v += volumes[i]
        out[i] = cum_pv / cum_v if cum_v > 0.0 else 0.0
    return out


# ---------- Swing high / low ----------

@njit(cache=True)
def swing_high_series(
    highs: np.ndarray,
    period: int,
) -> np.ndarray:
    """Detected swing-high points. Returns 1.0 where a swing high is
    confirmed (highs[i] > all neighbors within ±period), else 0.0."""
    n = highs.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or period < 1:
        return out
    for i in range(period, n - period):
        is_swing = True
        for k in range(1, period + 1):
            if highs[i - k] >= highs[i] or highs[i + k] >= highs[i]:
                is_swing = False
                break
        if is_swing:
            out[i] = 1.0
    return out


@njit(cache=True)
def swing_low_series(
    lows: np.ndarray,
    period: int,
) -> np.ndarray:
    """Detected swing-low points. 1.0 at confirmed swing-low bars."""
    n = lows.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or period < 1:
        return out
    for i in range(period, n - period):
        is_swing = True
        for k in range(1, period + 1):
            if lows[i - k] <= lows[i] or lows[i + k] <= lows[i]:
                is_swing = False
                break
        if is_swing:
            out[i] = 1.0
    return out


# ---------- Volume z-score ----------

@njit(cache=True)
def volume_zscore_series(
    volumes: np.ndarray,
    period: int,
) -> np.ndarray:
    """Rolling z-score of volume over `period` bars.

    Bars before `period-1` return 0.
    """
    n = volumes.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or period < 2 or n < period:
        return out
    for i in range(period - 1, n):
        s = 0.0
        for k in range(period):
            s += volumes[i - period + 1 + k]
        mean = s / period
        var = 0.0
        for k in range(period):
            d = volumes[i - period + 1 + k] - mean
            var += d * d
        sd = (var / period) ** 0.5
        if sd > 0.0:
            out[i] = (volumes[i] - mean) / sd
    return out


# ---------- CCI ----------

@njit(cache=True)
def cci_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    """Commodity Channel Index.

    typical_price = (high + low + close) / 3
    CCI = (tp - SMA(tp, period)) / (0.015 * mean_abs_dev)
    """
    n = closes.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or period < 1 or n < period:
        return out
    tp = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tp[i] = (highs[i] + lows[i] + closes[i]) / 3.0
    for i in range(period - 1, n):
        s = 0.0
        for k in range(period):
            s += tp[i - period + 1 + k]
        mean = s / period
        mad = 0.0
        for k in range(period):
            mad += abs(tp[i - period + 1 + k] - mean)
        mad /= period
        if mad > 0.0:
            out[i] = (tp[i] - mean) / (0.015 * mad)
    return out


# ---------- Williams %R ----------

@njit(cache=True)
def williams_r_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    """Williams %R = (highest_high - close) / (highest_high - lowest_low) * -100.

    Range [-100, 0]. Bars before `period-1` return 0.
    """
    n = closes.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or period < 1 or n < period:
        return out
    for i in range(period - 1, n):
        hi = highs[i - period + 1]
        lo = lows[i - period + 1]
        for k in range(1, period):
            v = highs[i - period + 1 + k]
            if v > hi:
                hi = v
            v2 = lows[i - period + 1 + k]
            if v2 < lo:
                lo = v2
        rng = hi - lo
        if rng > 0.0:
            out[i] = (hi - closes[i]) / rng * -100.0
    return out


# ---------- Money Flow Index ----------

@njit(cache=True)
def mfi_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int,
) -> np.ndarray:
    """Money Flow Index — RSI-style oscillator weighted by volume."""
    n = closes.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or period < 1 or n < period + 1:
        return out
    tp = np.zeros(n, dtype=np.float64)
    rmf = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tp[i] = (highs[i] + lows[i] + closes[i]) / 3.0
        rmf[i] = tp[i] * volumes[i]
    for i in range(period, n):
        pos = 0.0
        neg = 0.0
        for k in range(i - period + 1, i + 1):
            if tp[k] > tp[k - 1]:
                pos += rmf[k]
            elif tp[k] < tp[k - 1]:
                neg += rmf[k]
        if neg == 0.0:
            out[i] = 100.0
        elif pos == 0.0:
            out[i] = 0.0
        else:
            mr = pos / neg
            out[i] = 100.0 - (100.0 / (1.0 + mr))
    return out


# ---------- Rate of Change ----------

@njit(cache=True)
def roc_series(values: np.ndarray, period: int) -> np.ndarray:
    """Rate of Change = (value[i] - value[i-period]) / value[i-period] * 100."""
    n = values.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or period < 1 or n <= period:
        return out
    for i in range(period, n):
        if values[i - period] != 0.0:
            out[i] = (values[i] - values[i - period]) / values[i - period] * 100.0
    return out


# ---------- Parabolic SAR ----------

@njit(cache=True)
def parabolic_sar_series(
    highs: np.ndarray,
    lows: np.ndarray,
    af_step: float,
    af_max: float,
) -> np.ndarray:
    """Parabolic SAR. Returns 2-row 2D array (sar, direction).

    direction: +1 = uptrend, -1 = downtrend. Standard Wilder algorithm.
    """
    n = highs.shape[0]
    out = np.zeros((2, n), dtype=np.float64)
    if n < 2:
        return out
    direction = 1.0  # start uptrend
    sar = lows[0]
    ep = highs[0]
    af = af_step
    out[0, 0] = sar
    out[1, 0] = direction
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if direction > 0:
            if i >= 2 and sar > lows[i - 1]:
                sar = lows[i - 1]
            if i >= 1 and sar > lows[i - 1]:
                sar = lows[i - 1]
            if lows[i] < sar:
                direction = -1.0
                sar = ep
                ep = lows[i]
                af = af_step
            elif highs[i] > ep:
                ep = highs[i]
                af = af + af_step if (af + af_step) <= af_max else af_max
        else:
            if i >= 2 and sar < highs[i - 1]:
                sar = highs[i - 1]
            if i >= 1 and sar < highs[i - 1]:
                sar = highs[i - 1]
            if highs[i] > sar:
                direction = 1.0
                sar = ep
                ep = highs[i]
                af = af_step
            elif lows[i] < ep:
                ep = lows[i]
                af = af + af_step if (af + af_step) <= af_max else af_max
        out[0, i] = sar
        out[1, i] = direction
    return out


__all__ = [
    "NUMBA_AVAILABLE",
    "adx_series",
    "atr_series",
    "bollinger_series",
    "cci_series",
    "donchian_series",
    "ema_series",
    "macd_series",
    "mfi_series",
    "obv_series",
    "parabolic_sar_series",
    "roc_series",
    "rsi_series",
    "sma_series",
    "stochastic_series",
    "stochrsi_series",
    "supertrend_series",
    "swing_high_series",
    "swing_low_series",
    "true_range_series",
    "volume_zscore_series",
    "vwap_series",
    "williams_r_series",
    "wma_series",
]
