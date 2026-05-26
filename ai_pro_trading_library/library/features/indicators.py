"""Look-ahead-safe pandas indicators selected as canonical bootstrap functions.

Source attribution: rewritten from `cyqnt_trd.blocks.indicators`, keeping the
same vectorized semantics for common indicators used by active cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _series(values: pd.Series | list[float]) -> pd.Series:
    return values if isinstance(values, pd.Series) else pd.Series(values)


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"dataframe is missing required columns: {missing}")
    return df


def _safe_divide(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


def sma(values: pd.Series | list[float], period: int) -> pd.Series:
    period = _positive_int(period, "period")
    return _series(values).rolling(window=period, min_periods=period).mean()


def ema(values: pd.Series | list[float], period: int) -> pd.Series:
    period = _positive_int(period, "period")
    return _series(values).ewm(span=period, adjust=False, min_periods=period).mean()


def rma(values: pd.Series | list[float], period: int) -> pd.Series:
    period = _positive_int(period, "period")
    return _series(values).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(values: pd.Series | list[float], period: int = 14) -> pd.Series:
    period = _positive_int(period, "period")
    close = _series(values).astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    valid = avg_gain.notna() & avg_loss.notna()
    rs = _safe_divide(avg_gain, avg_loss, fill=0.0)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(~(valid & (avg_loss == 0) & (avg_gain > 0)), 100.0)
    out = out.where(~(valid & (avg_gain == 0) & (avg_loss > 0)), 0.0)
    return out.where(valid, np.nan)


def macd(
    values: pd.Series | list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = _positive_int(fast, "fast")
    slow = _positive_int(slow, "slow")
    signal = _positive_int(signal, "signal")
    if slow <= fast:
        raise ValueError("slow must be greater than fast")
    close = _series(values).astype(float)
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    return line, signal_line, line - signal_line


def true_range(df: pd.DataFrame) -> pd.Series:
    df = _require_columns(df, ("high", "low", "close"))
    previous_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return rma(true_range(df), _positive_int(period, "period"))


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    period = _positive_int(period, "period")
    df = _require_columns(df, ("high", "low", "close"))
    high = df["high"]
    low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)
    atr_value = atr(df, period)
    plus_di = 100.0 * _safe_divide(rma(plus_dm, period), atr_value, fill=0.0)
    minus_di = 100.0 * _safe_divide(rma(minus_dm, period), atr_value, fill=0.0)
    dx = 100.0 * _safe_divide((plus_di - minus_di).abs(), plus_di + minus_di, fill=0.0)
    return rma(dx, period), plus_di, minus_di


def bollinger_bands(
    values: pd.Series | list[float],
    period: int = 20,
    stddev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    period = _positive_int(period, "period")
    close = _series(values).astype(float)
    mid = sma(close, period)
    spread = close.rolling(window=period, min_periods=period).std(ddof=0) * float(stddev)
    return mid - spread, mid, mid + spread


bollinger = bollinger_bands


def wma(values: pd.Series | list[float], period: int) -> pd.Series:
    period = _positive_int(period, "period")
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    s = _series(values)
    return s.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights), raw=True
    )


def donchian(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian channel (upper, lower, middle) — matches blocks tuple order."""
    period = _positive_int(period, "period")
    df = _require_columns(df, ("high", "low"))
    upper = df["high"].rolling(window=period, min_periods=period).max()
    lower = df["low"].rolling(window=period, min_periods=period).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator slow (%K, %D)."""
    k_period = _positive_int(k_period, "k_period")
    d_period = _positive_int(d_period, "d_period")
    smooth_k = _positive_int(smooth_k, "smooth_k")
    df = _require_columns(df, ("high", "low", "close"))
    hh = df["high"].rolling(window=k_period, min_periods=k_period).max()
    ll = df["low"].rolling(window=k_period, min_periods=k_period).min()
    raw_k = 100.0 * _safe_divide(df["close"] - ll, hh - ll, fill=50.0)
    k = sma(raw_k, smooth_k)
    d = sma(k, d_period)
    return k, d


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP (running, not session-reset)."""
    df = _require_columns(df, ("high", "low", "close", "volume"))
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv_cum = (typical * df["volume"]).cumsum()
    vol_cum = df["volume"].cumsum()
    return _safe_divide(pv_cum, vol_cum, fill=0.0)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    df = _require_columns(df, ("close", "volume"))
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    df = _require_columns(df, ("volume",))
    return sma(df["volume"], period)


def rolling_zscore(values: pd.Series | list[float], period: int) -> pd.Series:
    period = _positive_int(period, "period")
    s = _series(values).astype(float)
    mu = s.rolling(window=period, min_periods=period).mean()
    sd = s.rolling(window=period, min_periods=period).std(ddof=0)
    return _safe_divide(s - mu, sd, fill=0.0)


def rolling_quantile(values: pd.Series | list[float], period: int, q: float) -> pd.Series:
    period = _positive_int(period, "period")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0,1], got {q}")
    return _series(values).rolling(window=period, min_periods=period).quantile(q)


def volume_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    df = _require_columns(df, ("volume",))
    return rolling_zscore(df["volume"], period)


def highest(values: pd.Series | list[float], period: int) -> pd.Series:
    period = _positive_int(period, "period")
    return _series(values).rolling(window=period, min_periods=period).max()


def lowest(values: pd.Series | list[float], period: int) -> pd.Series:
    period = _positive_int(period, "period")
    return _series(values).rolling(window=period, min_periods=period).min()


def swing_high(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    return highest(_require_columns(df, ("high",))["high"], lookback)


def swing_low(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    return lowest(_require_columns(df, ("low",))["low"], lookback)


def price_change_pct(values: pd.Series | list[float], periods: int = 1) -> pd.Series:
    return _series(values).pct_change(periods=periods)


def ma_direction(
    ma_series: pd.Series | list[float],
    lookback: int = 5,
    flat_threshold_bps: float = 5.0,
) -> pd.Series:
    """Categorical MA slope: `up` / `down` / `flat`."""
    lookback = _positive_int(lookback, "lookback")
    s = _series(ma_series).astype(float)
    pct = s.pct_change(periods=lookback) * 10_000.0
    flat = pct.abs() <= float(flat_threshold_bps)
    direction = pd.Series(np.where(pct > 0, "up", "down"), index=s.index)
    direction = direction.where(~flat, "flat")
    direction = direction.where(s.notna(), other=np.nan)
    return direction


def ma_alignment(*ma_series: pd.Series) -> pd.Series:
    """Categorical MA stack: `bullish` / `bearish` / `mixed`.

    Bullish = ma1 > ma2 > ... > maN.
    """
    if len(ma_series) < 2:
        raise ValueError("need at least two MA series")
    df = pd.concat(list(ma_series), axis=1)
    diffs = df.diff(axis=1).iloc[:, 1:]
    bullish = (diffs < 0).all(axis=1)
    bearish = (diffs > 0).all(axis=1)
    out = pd.Series("mixed", index=df.index)
    out = out.where(~bullish, "bullish")
    out = out.where(~bearish, "bearish")
    return out


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """SuperTrend indicator. Returns (supertrend_value, direction ∈ {+1, -1})."""
    period = _positive_int(period, "period")
    df = _require_columns(df, ("high", "low", "close"))
    n = len(df)
    atr_arr = atr(df, period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy()
    close = df["close"].to_numpy()
    upper_basic = hl2 + multiplier * atr_arr
    lower_basic = hl2 - multiplier * atr_arr

    final_upper = upper_basic.copy()
    final_lower = lower_basic.copy()
    direction_arr = np.zeros(n, dtype=int)
    st_arr = np.full(n, np.nan)
    started = False

    for i in range(1, n):
        if np.isnan(upper_basic[i]) or np.isnan(lower_basic[i]):
            continue
        if started:
            if not (
                upper_basic[i] < final_upper[i - 1]
                or close[i - 1] > final_upper[i - 1]
            ):
                final_upper[i] = final_upper[i - 1]
            if not (
                lower_basic[i] > final_lower[i - 1]
                or close[i - 1] < final_lower[i - 1]
            ):
                final_lower[i] = final_lower[i - 1]
            prev_dir = direction_arr[i - 1]
            if prev_dir == 1 and close[i] < final_lower[i]:
                direction_arr[i] = -1
            elif prev_dir == -1 and close[i] > final_upper[i]:
                direction_arr[i] = 1
            else:
                direction_arr[i] = prev_dir
        else:
            direction_arr[i] = 1 if close[i] > final_upper[i] else -1
            started = True
        st_arr[i] = final_lower[i] if direction_arr[i] == 1 else final_upper[i]

    return (
        pd.Series(st_arr, index=df.index),
        pd.Series(direction_arr, index=df.index),
    )


def ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """Ichimoku cloud — returns DataFrame with 5 components."""
    df = _require_columns(df, ("high", "low", "close"))
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tenkan_sen = (highest(high, tenkan) + lowest(low, tenkan)) / 2.0
    kijun_sen = (highest(high, kijun) + lowest(low, kijun)) / 2.0
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0).shift(displacement)
    senkou_b_ = ((highest(high, senkou_b) + lowest(low, senkou_b)) / 2.0).shift(displacement)
    chikou = close.shift(-displacement)
    return pd.DataFrame(
        {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_a,
            "senkou_span_b": senkou_b_,
            "chikou_span": chikou,
        }
    )


def parabolic_sar(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> tuple[pd.Series, pd.Series]:
    """Parabolic SAR (Wilder). Returns (sar, direction)."""
    df = _require_columns(df, ("high", "low", "close"))
    n = len(df)
    if n < 2:
        return (
            pd.Series(np.nan, index=df.index),
            pd.Series(0, index=df.index, dtype=int),
        )

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    sar = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    direction[0] = 1 if close[1] >= close[0] else -1
    if direction[0] == 1:
        sar[0] = low[0]
        ep = high[0]
    else:
        sar[0] = high[0]
        ep = low[0]
    af = af_start

    for i in range(1, n):
        prev_dir = direction[i - 1]
        prev_sar = sar[i - 1]
        cand = prev_sar + af * (ep - prev_sar)

        if prev_dir == 1:
            cand = min(cand, low[i - 1], low[max(i - 2, 0)])
            if low[i] < cand:
                direction[i] = -1
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                direction[i] = 1
                sar[i] = cand
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            cand = max(cand, high[i - 1], high[max(i - 2, 0)])
            if high[i] > cand:
                direction[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                direction[i] = -1
                sar[i] = cand
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return (
        pd.Series(sar, index=df.index),
        pd.Series(direction, index=df.index),
    )
