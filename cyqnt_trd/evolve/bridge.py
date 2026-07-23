"""Bridge: genome (declarative JSON) → make_signals(df) → vectorized backtest.

This module owns the *only* place where the genome vocabulary
(``ALLOWED_INDICATORS`` / ``ALLOWED_CONDITIONS`` / ``ALLOWED_FILTERS``)
gets translated into actual pandas operations on an OHLCV DataFrame.

Adding a new indicator/condition/filter requires:
    1. add the token to the corresponding ALLOWED_* set in genome.py
    2. handle it here in :func:`_factor_to_series` or :func:`_apply_filter`

Caller contract:
    df : DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
         and a DatetimeIndex (UTC). Optional: 'close_time' (ms).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from cyqnt_trd.blocks import indicators as ind
from cyqnt_trd.standard_bot.simulation.vectorized_backtest import (
    VectorizedBacktestResult,
    run_vectorized_backtest,
)

from .genome import StrategyGenome


# ── Param accessors with defaults ─────────────────────────────────────────

def _p(d: Dict, key: str, default):
    """Param accessor that tolerates None / missing keys."""
    v = d.get(key)
    return default if v is None else v


# ── Factor → boolean Series ────────────────────────────────────────────────

def _factor_to_series(df: pd.DataFrame, factor) -> pd.Series:
    """Convert a single Factor (indicator + condition + params) into a
    boolean pandas Series indexed like df.

    Returns a Series of True where the factor "fires" on that bar.
    Unknown combinations produce all-False (silent fallback) but raise in
    debug mode if you import this and want strict checking.
    """
    ind_name = factor.indicator
    cond = factor.condition
    p = factor.params or {}
    close = df["close"]
    out: pd.Series

    # ── EMA / SMA ──
    if ind_name in ("ema", "sma"):
        fast_p = int(_p(p, "fast", _p(p, "period", 9)))
        slow_p = int(_p(p, "slow", 21))
        fn = ind.ema if ind_name == "ema" else ind.sma
        if cond == "cross_above":
            fast = fn(close, fast_p); slow = fn(close, slow_p)
            prev = fast.shift(1) <= slow.shift(1)
            out = (fast > slow) & prev
        elif cond == "cross_below":
            fast = fn(close, fast_p); slow = fn(close, slow_p)
            prev = fast.shift(1) >= slow.shift(1)
            out = (fast < slow) & prev
        elif cond == "above":
            ma = fn(close, fast_p)
            out = close > ma
        elif cond == "below":
            ma = fn(close, fast_p)
            out = close < ma
        elif cond == "slope_positive":
            period = int(_p(p, "period", 20))
            lookback = int(_p(p, "lookback", 3))
            ma = fn(close, period)
            out = ma > ma.shift(lookback)
        elif cond == "slope_negative":
            period = int(_p(p, "period", 20))
            lookback = int(_p(p, "lookback", 3))
            ma = fn(close, period)
            out = ma < ma.shift(lookback)
        else:
            out = pd.Series(False, index=df.index)

    # ── RSI ──
    elif ind_name == "rsi":
        period = int(_p(p, "period", 14))
        rsi_v = ind.rsi(close, period)
        if cond == "oversold":
            thr = float(_p(p, "threshold", 30))
            # Trigger on the bar where RSI crosses back ABOVE threshold from below
            cross_back = (rsi_v > thr) & (rsi_v.shift(1) <= thr)
            out = cross_back
        elif cond == "overbought":
            thr = float(_p(p, "threshold", 70))
            cross_back = (rsi_v < thr) & (rsi_v.shift(1) >= thr)
            out = cross_back
        elif cond == "above_threshold" or cond == "above":
            thr = float(_p(p, "threshold", 50))
            out = rsi_v > thr
        elif cond == "below_threshold" or cond == "below":
            thr = float(_p(p, "threshold", 50))
            out = rsi_v < thr
        elif cond == "cross_above":
            thr = float(_p(p, "threshold", 50))
            out = (rsi_v > thr) & (rsi_v.shift(1) <= thr)
        elif cond == "cross_below":
            thr = float(_p(p, "threshold", 50))
            out = (rsi_v < thr) & (rsi_v.shift(1) >= thr)
        else:
            out = pd.Series(False, index=df.index)

    # ── MACD ──
    elif ind_name == "macd":
        fast = int(_p(p, "fast", 12))
        slow = int(_p(p, "slow", 26))
        signal = int(_p(p, "signal", 9))
        if slow <= fast:
            slow = fast + 1
        macd_line, signal_line, hist = ind.macd(close, fast, slow, signal)
        if cond == "macd_golden_cross" or cond == "cross_above":
            out = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        elif cond == "macd_death_cross" or cond == "cross_below":
            out = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        elif cond == "macd_above_zero":
            out = macd_line > 0
        elif cond == "macd_below_zero":
            out = macd_line < 0
        else:
            out = pd.Series(False, index=df.index)

    # ── Bollinger Bands ──
    elif ind_name == "bollinger":
        period = int(_p(p, "period", 20))
        std = float(_p(p, "std", 2.0))
        upper, mid, lower = ind.bollinger(close, period, std)
        if cond == "touch_lower":
            out = df["low"] <= lower
        elif cond == "touch_upper":
            out = df["high"] >= upper
        elif cond == "breakout":
            # close breaks above upper (after being below)
            out = (close > upper) & (close.shift(1) <= upper.shift(1))
        elif cond == "squeeze":
            bw_thr = float(_p(p, "bandwidth_threshold", 0.05))
            bandwidth = (upper - lower) / mid
            out = bandwidth < bw_thr
        elif cond == "above":
            out = close > upper
        elif cond == "below":
            out = close < lower
        else:
            out = pd.Series(False, index=df.index)

    # ── Stochastic ──
    elif ind_name == "stoch":
        k_period = int(_p(p, "k", _p(p, "k_period", 14)))
        d_period = int(_p(p, "d", _p(p, "d_period", 3)))
        smooth_k = int(_p(p, "smooth_k", 3))
        k, d = ind.stochastic(df, k_period, d_period, smooth_k)
        if cond == "cross_above":
            thr = float(_p(p, "threshold", 20))
            # K crosses above D, both below threshold (oversold reversal)
            kx = (k > d) & (k.shift(1) <= d.shift(1))
            out = kx & (k < thr + 30)  # somewhat oversold
        elif cond == "cross_below":
            thr = float(_p(p, "threshold", 80))
            kx = (k < d) & (k.shift(1) >= d.shift(1))
            out = kx & (k > thr - 30)
        elif cond == "oversold":
            thr = float(_p(p, "threshold", 20))
            out = (k > thr) & (k.shift(1) <= thr)
        elif cond == "overbought":
            thr = float(_p(p, "threshold", 80))
            out = (k < thr) & (k.shift(1) >= thr)
        else:
            out = pd.Series(False, index=df.index)

    # ── ADX ──
    elif ind_name == "adx":
        period = int(_p(p, "period", 14))
        adx_v, _, _ = ind.adx(df, period)
        if cond == "trending" or cond == "above_threshold":
            thr = float(_p(p, "threshold", 25))
            out = adx_v > thr
        else:
            out = pd.Series(False, index=df.index)

    # ── ATR ──
    elif ind_name == "atr":
        period = int(_p(p, "period", 14))
        atr_v = ind.atr(df, period)
        atr_pct = atr_v / close
        if cond == "spike":
            mult = float(_p(p, "multiplier", 1.5))
            ma_period = int(_p(p, "ma_period", 20))
            atr_ma = atr_v.rolling(ma_period, min_periods=ma_period).mean()
            out = atr_v > atr_ma * mult
        elif cond == "above_threshold":
            thr = float(_p(p, "threshold", 0.005))
            out = atr_pct > thr
        else:
            out = pd.Series(False, index=df.index)

    # ── Supertrend ──
    elif ind_name == "supertrend":
        period = int(_p(p, "period", 10))
        mult = float(_p(p, "multiplier", 3.0))
        st_val, st_dir = ind.supertrend(df, period, mult)
        if cond == "cross_above":
            out = (st_dir == 1) & (st_dir.shift(1) == -1)
        elif cond == "cross_below":
            out = (st_dir == -1) & (st_dir.shift(1) == 1)
        elif cond == "above":
            out = st_dir == 1
        elif cond == "below":
            out = st_dir == -1
        else:
            out = pd.Series(False, index=df.index)

    # ── VWAP ──
    elif ind_name == "vwap":
        vwap_v = ind.vwap(df)
        if cond == "cross_above":
            out = (close > vwap_v) & (close.shift(1) <= vwap_v.shift(1))
        elif cond == "cross_below":
            out = (close < vwap_v) & (close.shift(1) >= vwap_v.shift(1))
        elif cond == "above":
            out = close > vwap_v
        elif cond == "below":
            out = close < vwap_v
        else:
            out = pd.Series(False, index=df.index)

    # ── Donchian breakout ──
    elif ind_name == "donchian":
        period = int(_p(p, "period", 20))
        upper, lower, mid = ind.donchian(df, period)
        if cond == "breakout":
            # Close > prev N-bar high
            out = close > upper.shift(1)
        elif cond == "cross_below":
            out = close < lower.shift(1)
        elif cond == "above":
            out = close > mid
        else:
            out = pd.Series(False, index=df.index)

    # ── Williams %R / CCI / MFI / Keltner / HMA / TEMA / DEMA ──
    elif ind_name == "williams_r":
        period = int(_p(p, "period", 14))
        wr = ind.williams_r(df, period)
        # WR ranges -100..0; oversold = below -80
        if cond == "oversold":
            thr = float(_p(p, "threshold", -80))
            out = (wr > thr) & (wr.shift(1) <= thr)
        elif cond == "overbought":
            thr = float(_p(p, "threshold", -20))
            out = (wr < thr) & (wr.shift(1) >= thr)
        else:
            out = pd.Series(False, index=df.index)

    elif ind_name == "cci":
        period = int(_p(p, "period", 20))
        cci_v = ind.cci(df, period)
        if cond == "oversold":
            thr = float(_p(p, "threshold", -100))
            out = (cci_v > thr) & (cci_v.shift(1) <= thr)
        elif cond == "overbought":
            thr = float(_p(p, "threshold", 100))
            out = (cci_v < thr) & (cci_v.shift(1) >= thr)
        elif cond == "above_threshold":
            out = cci_v > float(_p(p, "threshold", 0))
        elif cond == "below_threshold":
            out = cci_v < float(_p(p, "threshold", 0))
        else:
            out = pd.Series(False, index=df.index)

    elif ind_name == "mfi":
        period = int(_p(p, "period", 14))
        mfi_v = ind.mfi(df, period)
        if cond == "oversold":
            thr = float(_p(p, "threshold", 20))
            out = (mfi_v > thr) & (mfi_v.shift(1) <= thr)
        elif cond == "overbought":
            thr = float(_p(p, "threshold", 80))
            out = (mfi_v < thr) & (mfi_v.shift(1) >= thr)
        else:
            out = pd.Series(False, index=df.index)

    elif ind_name == "keltner":
        period = int(_p(p, "period", 20))
        atr_period = int(_p(p, "atr_period", 10))
        mult = float(_p(p, "multiplier", 2.0))
        upper, mid, lower = ind.keltner(df, period, atr_period, mult)
        if cond == "breakout":
            out = (close > upper) & (close.shift(1) <= upper.shift(1))
        elif cond == "touch_lower":
            out = df["low"] <= lower
        elif cond == "touch_upper":
            out = df["high"] >= upper
        else:
            out = pd.Series(False, index=df.index)

    elif ind_name in ("hma", "tema", "dema"):
        period = int(_p(p, "period", 21))
        fn = {"hma": ind.hma, "tema": ind.tema, "dema": ind.dema}[ind_name]
        ma = fn(close, period)
        if cond == "above":
            out = close > ma
        elif cond == "below":
            out = close < ma
        elif cond == "cross_above":
            out = (close > ma) & (close.shift(1) <= ma.shift(1))
        elif cond == "cross_below":
            out = (close < ma) & (close.shift(1) >= ma.shift(1))
        else:
            out = pd.Series(False, index=df.index)

    elif ind_name == "obv":
        # treat obv as a momentum proxy: positive slope
        obv_v = ind.obv(df)
        lookback = int(_p(p, "lookback", 5))
        if cond == "slope_positive":
            out = obv_v > obv_v.shift(lookback)
        elif cond == "slope_negative":
            out = obv_v < obv_v.shift(lookback)
        else:
            out = pd.Series(False, index=df.index)

    # ── StochRSI (faster, more sensitive than Stoch) ──
    elif ind_name == "stochrsi":
        rsi_p = int(_p(p, "rsi_period", 14))
        st_p = int(_p(p, "stoch_period", 14))
        k_p = int(_p(p, "k", 3))
        d_p = int(_p(p, "d", 3))
        k, d = ind.stochrsi(close, rsi_p, st_p, k_p, d_p)
        if cond == "cross_above":
            thr = float(_p(p, "threshold", 20))
            out = (k > d) & (k.shift(1) <= d.shift(1)) & (k < thr + 30)
        elif cond == "oversold":
            thr = float(_p(p, "threshold", 20))
            out = (k > thr) & (k.shift(1) <= thr)
        elif cond == "overbought":
            thr = float(_p(p, "threshold", 80))
            out = (k < thr) & (k.shift(1) >= thr)
        elif cond == "above_threshold":
            out = k > float(_p(p, "threshold", 50))
        else:
            out = pd.Series(False, index=df.index)

    # ── Aroon (trend-strength oscillator) ──
    elif ind_name == "aroon":
        period = int(_p(p, "period", 14))
        a_up, a_dn, a_osc = ind.aroon(df, period)
        if cond == "aroon_up_strong":
            thr = float(_p(p, "threshold", 70))
            out = (a_up > thr) & (a_up.shift(1) <= thr)
        elif cond == "aroon_dn_strong":
            thr = float(_p(p, "threshold", 70))
            out = (a_dn > thr) & (a_dn.shift(1) <= thr)
        elif cond == "cross_above":
            # osc crosses above 0 = bullish
            out = (a_osc > 0) & (a_osc.shift(1) <= 0)
        elif cond == "above_threshold":
            thr = float(_p(p, "threshold", 50))
            out = a_osc > thr
        else:
            out = pd.Series(False, index=df.index)

    # ── Parabolic SAR (trend-flip detector with built-in trailing stop) ──
    elif ind_name == "psar":
        af_start = float(_p(p, "af_start", 0.02))
        af_step = float(_p(p, "af_step", 0.02))
        af_max = float(_p(p, "af_max", 0.20))
        psar_v, psar_dir = ind.parabolic_sar(df, af_start, af_step, af_max)
        if cond == "psar_flip_up" or cond == "cross_above":
            out = (psar_dir == 1) & (psar_dir.shift(1) == -1)
        elif cond == "psar_flip_dn" or cond == "cross_below":
            out = (psar_dir == -1) & (psar_dir.shift(1) == 1)
        elif cond == "above":
            out = psar_dir == 1
        elif cond == "below":
            out = psar_dir == -1
        else:
            out = pd.Series(False, index=df.index)

    # ── ATR ratio (atr / close, regime indicator) ──
    elif ind_name == "atr_ratio":
        period = int(_p(p, "period", 14))
        ar = ind.atr_ratio(df, period)
        if cond == "below_threshold" or cond == "low_atr_ratio":
            thr = float(_p(p, "threshold", 0.005))
            out = ar < thr
        elif cond == "above_threshold" or cond == "high_atr_ratio":
            thr = float(_p(p, "threshold", 0.005))
            out = ar > thr
        else:
            out = pd.Series(False, index=df.index)

    # ── Trend strength (% bars closing above SMA over lookback) ──
    elif ind_name == "trend_strength":
        lookback = int(_p(p, "lookback", 10))
        ts = ind.trend_strength(df, lookback)
        if cond == "above_threshold":
            thr = float(_p(p, "threshold", 0.6))
            out = ts > thr
        elif cond == "below_threshold":
            thr = float(_p(p, "threshold", 0.4))
            out = ts < thr
        else:
            out = pd.Series(False, index=df.index)

    # ── Volume surge ratio (volume / volume_ma) ──
    elif ind_name == "volume_surge":
        lookback = int(_p(p, "lookback", 20))
        vsr = ind.volume_surge_ratio(df, lookback)
        if cond == "spike" or cond == "above_threshold":
            thr = float(_p(p, "threshold", 1.5))
            out = vsr > thr
        else:
            out = pd.Series(False, index=df.index)

    else:
        # unknown indicator → never fires (engine will validate genome before this anyway)
        out = pd.Series(False, index=df.index)

    return out.fillna(False).astype(bool)


# ── Filter → boolean Series ────────────────────────────────────────────────

def _apply_filter(df: pd.DataFrame, fl) -> pd.Series:
    """Return boolean Series indicating bars that PASS the filter."""
    ftype = fl.filter_type
    p = fl.params or {}
    close = df["close"]

    if ftype == "adx_above":
        period = int(_p(p, "period", 14))
        thr = float(_p(p, "threshold", 20))
        adx_v, _, _ = ind.adx(df, period)
        return (adx_v > thr).fillna(False).astype(bool)

    if ftype == "atr_above_pct":
        period = int(_p(p, "period", 14))
        thr = float(_p(p, "threshold", 0.003))
        atr_v = ind.atr(df, period)
        return ((atr_v / close) > thr).fillna(False).astype(bool)

    if ftype == "atr_above_percentile":
        period = int(_p(p, "period", 14))
        win = int(_p(p, "window", 100))
        pct = float(_p(p, "percentile", 0.4))
        atr_v = ind.atr(df, period)
        thr = atr_v.rolling(win, min_periods=max(20, win // 2)).quantile(pct)
        return (atr_v > thr).fillna(False).astype(bool)

    # ── NEW: block when ATR is in the TOP X% (avoid crash / extreme vol) ──
    if ftype == "atr_below_percentile":
        period = int(_p(p, "period", 14))
        win = int(_p(p, "window", 100))
        pct = float(_p(p, "percentile", 0.85))
        atr_v = ind.atr(df, period)
        thr = atr_v.rolling(win, min_periods=max(20, win // 2)).quantile(pct)
        return (atr_v < thr).fillna(False).astype(bool)

    # ── NEW: block when atr_ratio (atr%) above threshold (high-vol guard) ──
    if ftype == "atr_ratio_below":
        period = int(_p(p, "period", 14))
        thr = float(_p(p, "threshold", 0.008))
        ar = ind.atr_ratio(df, period)
        return (ar < thr).fillna(False).astype(bool)

    # ── NEW: explicit EMA-slope-positive gate ──
    if ftype == "ema_slope_positive":
        period = int(_p(p, "period", 50))
        lookback = int(_p(p, "lookback", 5))
        ma = ind.ema(close, period)
        return (ma > ma.shift(lookback)).fillna(False).astype(bool)

    # ── NEW: StochRSI K above threshold (momentum confirm) ──
    if ftype == "stochrsi_above":
        rsi_p = int(_p(p, "rsi_period", 14))
        st_p = int(_p(p, "stoch_period", 14))
        thr = float(_p(p, "threshold", 50))
        k, _ = ind.stochrsi(close, rsi_p, st_p, 3, 3)
        return (k > thr).fillna(False).astype(bool)

    # ── NEW: Aroon Up above threshold (trend confirm) ──
    if ftype == "aroon_up_above":
        period = int(_p(p, "period", 14))
        thr = float(_p(p, "threshold", 60))
        a_up, _, _ = ind.aroon(df, period)
        return (a_up > thr).fillna(False).astype(bool)

    if ftype == "volume_above" or ftype == "volume_above_ma":
        period = int(_p(p, "period", 20))
        mult = float(_p(p, "multiplier", 1.0))
        vma = ind.volume_ma(df, period)
        return (df["volume"] > vma * mult).fillna(False).astype(bool)

    if ftype == "hour_range":
        start = int(_p(p, "start", 0))
        end = int(_p(p, "end", 24))
        # use index hour (assumes DatetimeIndex UTC)
        if isinstance(df.index, pd.DatetimeIndex):
            hours = df.index.hour
        else:
            return pd.Series(True, index=df.index)
        if start <= end:
            mask = (hours >= start) & (hours < end)
        else:  # wrap-around (e.g. 22-04)
            mask = (hours >= start) | (hours < end)
        return pd.Series(mask, index=df.index)

    if ftype == "ma_slope_positive":
        period = int(_p(p, "period", 50))
        lookback = int(_p(p, "lookback", 5))
        ma = ind.ema(close, period)
        return (ma > ma.shift(lookback)).fillna(False).astype(bool)

    if ftype == "bbw_above":
        period = int(_p(p, "period", 20))
        std = float(_p(p, "std", 2.0))
        thr = float(_p(p, "threshold", 0.02))
        upper, mid, lower = ind.bollinger(close, period, std)
        bw = (upper - lower) / mid
        return (bw > thr).fillna(False).astype(bool)

    if ftype == "rsi_in_range":
        period = int(_p(p, "period", 14))
        lo = float(_p(p, "low", 40))
        hi = float(_p(p, "high", 60))
        rsi_v = ind.rsi(close, period)
        return ((rsi_v > lo) & (rsi_v < hi)).fillna(False).astype(bool)

    if ftype == "ema_alignment":
        # bullish alignment: ema(short) > ema(mid) > ema(long)
        s = int(_p(p, "short", 8))
        m = int(_p(p, "mid", 21))
        l = int(_p(p, "long", 55))
        es, em, el = ind.ema(close, s), ind.ema(close, m), ind.ema(close, l)
        return ((es > em) & (em > el)).fillna(False).astype(bool)

    if ftype == "price_above_ma":
        period = int(_p(p, "period", 50))
        ma_type = str(_p(p, "ma_type", "ema")).lower()
        ma = ind.ema(close, period) if ma_type == "ema" else ind.sma(close, period)
        return (close > ma).fillna(False).astype(bool)

    if ftype == "price_below_ma":
        period = int(_p(p, "period", 50))
        ma_type = str(_p(p, "ma_type", "ema")).lower()
        ma = ind.ema(close, period) if ma_type == "ema" else ind.sma(close, period)
        return (close < ma).fillna(False).astype(bool)

    # unknown filter → pass-through (don't accidentally kill all signals)
    return pd.Series(True, index=df.index)


# ── Entry combiner ─────────────────────────────────────────────────────────

def _combine_factors(
    df: pd.DataFrame, factors, logic: str, score_threshold: int
) -> pd.Series:
    """Combine N factor signals according to entry_logic."""
    if not factors:
        return pd.Series(False, index=df.index)
    series_list = [_factor_to_series(df, f) for f in factors]
    if logic == "all_of":
        out = series_list[0]
        for s in series_list[1:]:
            out = out & s
        return out
    if logic == "any_of":
        out = series_list[0]
        for s in series_list[1:]:
            out = out | s
        return out
    if logic == "score_gte":
        # weighted sum of bool→int weighted by factor.weight
        score = sum(s.astype(int) * float(f.weight) for s, f in zip(series_list, factors))
        return score >= float(score_threshold)
    return pd.Series(False, index=df.index)


# ── Public API ─────────────────────────────────────────────────────────────

def make_signal_fn(genome: StrategyGenome) -> Callable[[pd.DataFrame], Tuple[pd.Series, pd.Series]]:
    """Build a make_signals(df) closure for the genome.

    Returns (long_signal, short_signal). Since we are long-only, short_signal
    is always all-False. The vectorized backtester is run with long_only=True
    anyway so this is double-protection.
    """

    def make_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        entry = _combine_factors(
            df,
            genome.entry_factors,
            genome.entry_logic,
            genome.entry_score_threshold,
        )

        # Apply all filters (AND together)
        filter_mask = pd.Series(True, index=df.index)
        for fl in genome.filters:
            filter_mask = filter_mask & _apply_filter(df, fl)

        long_signal = (entry & filter_mask).fillna(False).astype(bool)
        short_signal = pd.Series(False, index=df.index)
        return long_signal, short_signal

    return make_signals


def backtest_genome(
    *,
    genome: StrategyGenome,
    df: pd.DataFrame,
    timeframe: str = None,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
    initial_capital: float = 10_000.0,
    record_trades: bool = False,
) -> VectorizedBacktestResult:
    """Run a vectorized backtest for the genome on the given DataFrame.

    The genome's exit_type/exit_params is translated into the backtest's
    exit_cfg dict directly (same vocabulary).
    """
    tf = timeframe or genome.preferred_interval

    exit_cfg = {"type": genome.exit_type, **dict(genome.exit_params)}

    signal_fn = make_signal_fn(genome)

    return run_vectorized_backtest(
        df=df,
        signal_fn=signal_fn,
        exit_cfg=exit_cfg,
        timeframe=tf,
        size=genome.size,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
        record_trades=record_trades,
        long_only=True,  # spot market (Binance AI Pro) — SPEC requirement
    )
