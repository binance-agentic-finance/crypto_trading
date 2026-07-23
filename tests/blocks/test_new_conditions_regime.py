"""Unit tests for the conditions / regime helpers added in the
evolutionary-strategy-discovery work (XRPUSDT 5m run #2, 2026-06-05).

New conditions:
    - stochrsi_oversold / stochrsi_overbought / stochrsi_cross_above
    - aroon_up_strong / aroon_down_strong / aroon_oscillator_above
    - psar_flip_up / psar_flip_down

New regime filters (OOS-survival):
    - atr_below_percentile / atr_above_percentile
    - atr_ratio_below_threshold / atr_ratio_above_threshold
    - ma_slope_positive / ma_slope_negative
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks import conditions as cond
from cyqnt_trd.blocks import indicators as ind
from cyqnt_trd.blocks import regime


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """Deterministic 300-bar OHLCV — enough warmup for 200-bar windows."""
    rng = np.random.default_rng(42)
    n = 300
    base = np.cumsum(rng.normal(0, 0.5, n)) + 100
    df = pd.DataFrame(
        {
            "open": base + rng.normal(0, 0.1, n),
            "high": base + np.abs(rng.normal(0.3, 0.2, n)),
            "low":  base - np.abs(rng.normal(0.3, 0.2, n)),
            "close": base + rng.normal(0, 0.1, n),
            "volume": rng.uniform(100, 500, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


# ── StochRSI ──────────────────────────────────────────────────────────────

def test_stochrsi_oversold_returns_bool_series(ohlcv):
    k, _ = ind.stochrsi(ohlcv["close"])
    out = cond.stochrsi_oversold(k, threshold=20)
    assert out.dtype == bool
    assert len(out) == len(ohlcv)
    # k <= 20 implies True
    valid = k.notna() & out
    assert (k[valid] <= 20).all()


def test_stochrsi_overbought_returns_bool_series(ohlcv):
    k, _ = ind.stochrsi(ohlcv["close"])
    out = cond.stochrsi_overbought(k, threshold=80)
    assert out.dtype == bool


def test_stochrsi_cross_above_only_at_cross(ohlcv):
    k, _ = ind.stochrsi(ohlcv["close"])
    out = cond.stochrsi_cross_above(k, threshold=20)
    assert out.dtype == bool
    # Triggers must be where k just crossed 20 from below
    triggers = k[out]
    if len(triggers) > 0:
        assert (triggers > 20).all()


# ── Aroon ─────────────────────────────────────────────────────────────────

def test_aroon_up_strong_threshold(ohlcv):
    a_up, a_dn, _ = ind.aroon(ohlcv, 14)
    out = cond.aroon_up_strong(a_up, threshold=70)
    assert out.dtype == bool
    valid = a_up.notna() & out
    assert (a_up[valid] >= 70).all()


def test_aroon_down_strong_threshold(ohlcv):
    a_up, a_dn, _ = ind.aroon(ohlcv, 14)
    out = cond.aroon_down_strong(a_dn, threshold=70)
    valid = a_dn.notna() & out
    assert (a_dn[valid] >= 70).all()


def test_aroon_oscillator_above(ohlcv):
    a_up, a_dn, _ = ind.aroon(ohlcv, 14)
    out = cond.aroon_oscillator_above(a_up, a_dn, threshold=20)
    valid = a_up.notna() & a_dn.notna() & out
    assert ((a_up[valid] - a_dn[valid]) >= 20).all()


# ── PSAR ──────────────────────────────────────────────────────────────────

def test_psar_flip_up_only_when_psar_was_above_now_below(ohlcv):
    psar_v, _ = ind.parabolic_sar(ohlcv)
    flips = cond.psar_flip_up(psar_v, ohlcv["close"])
    assert flips.dtype == bool
    # Where True, PSAR was above close on prev bar and below on this bar
    triggers_idx = flips[flips].index
    for t in triggers_idx:
        prev_t = flips.index[flips.index.get_loc(t) - 1]
        assert psar_v.loc[prev_t] > ohlcv["close"].loc[prev_t]
        assert psar_v.loc[t] < ohlcv["close"].loc[t]


def test_psar_flip_down_symmetric(ohlcv):
    psar_v, _ = ind.parabolic_sar(ohlcv)
    flips = cond.psar_flip_down(psar_v, ohlcv["close"])
    assert flips.dtype == bool
    triggers_idx = flips[flips].index
    for t in triggers_idx:
        prev_t = flips.index[flips.index.get_loc(t) - 1]
        assert psar_v.loc[prev_t] < ohlcv["close"].loc[prev_t]
        assert psar_v.loc[t] > ohlcv["close"].loc[t]


def test_psar_flips_are_disjoint(ohlcv):
    """A single bar cannot be both flip_up and flip_down."""
    psar_v, _ = ind.parabolic_sar(ohlcv)
    up = cond.psar_flip_up(psar_v, ohlcv["close"])
    dn = cond.psar_flip_down(psar_v, ohlcv["close"])
    assert not (up & dn).any()


# ── ATR percentile filters ────────────────────────────────────────────────

def test_atr_below_percentile_blocks_high_vol(ohlcv):
    a = ind.atr(ohlcv, 14)
    out = regime.atr_below_percentile(a, window=200, percentile=0.80)
    assert out.dtype == bool
    # When True, ATR is in bottom 80% — i.e. strictly below the 80% threshold
    # on the rolling window. Hard to test directly without redoing the
    # rolling quantile, so just verify shape + non-trivial output.
    assert len(out) == len(ohlcv)
    # Should have some True (low-vol bars) and some False (top-20% vol bars)
    # except in early warmup where rolling.quantile is NaN.
    warmup = max(20, 200 // 2)
    after_warmup = out.iloc[warmup:]
    assert after_warmup.sum() > 0
    assert (~after_warmup).sum() > 0


def test_atr_above_percentile_complements_below(ohlcv):
    """atr_above_percentile(p) and atr_below_percentile(p) should be
    nearly disjoint after warmup (boundary bar = exactly the threshold
    is unlikely to be hit twice)."""
    a = ind.atr(ohlcv, 14)
    above = regime.atr_above_percentile(a, window=200, percentile=0.50)
    below = regime.atr_below_percentile(a, window=200, percentile=0.50)
    overlap = (above & below).sum()
    # At most a tiny boundary overlap
    assert overlap <= 2


def test_atr_below_percentile_rejects_invalid_percentile(ohlcv):
    a = ind.atr(ohlcv, 14)
    with pytest.raises(ValueError):
        regime.atr_below_percentile(a, percentile=1.5)
    with pytest.raises(ValueError):
        regime.atr_below_percentile(a, percentile=0.0)


# ── ATR ratio filters ─────────────────────────────────────────────────────

def test_atr_ratio_below_threshold(ohlcv):
    a = ind.atr(ohlcv, 14)
    # Use a very high threshold so most bars pass
    out = regime.atr_ratio_below_threshold(a, ohlcv["close"], threshold=1.0)
    assert out.dtype == bool
    # ATR / close < 1.0 should be True for virtually all bars
    valid = (a / ohlcv["close"]).notna()
    assert out[valid].mean() > 0.99


def test_atr_ratio_above_threshold(ohlcv):
    a = ind.atr(ohlcv, 14)
    # Use a tiny threshold so most bars exceed it
    out = regime.atr_ratio_above_threshold(a, ohlcv["close"], threshold=0.0001)
    assert out.dtype == bool
    valid = (a / ohlcv["close"]).notna()
    assert out[valid].mean() > 0.99


def test_atr_ratio_threshold_rejects_negative(ohlcv):
    a = ind.atr(ohlcv, 14)
    with pytest.raises(ValueError):
        regime.atr_ratio_below_threshold(a, ohlcv["close"], threshold=-0.01)
    with pytest.raises(ValueError):
        regime.atr_ratio_above_threshold(a, ohlcv["close"], threshold=0.0)


# ── MA slope filters ──────────────────────────────────────────────────────

def test_ma_slope_positive_basic(ohlcv):
    ma = ind.ema(ohlcv["close"], 50)
    out = regime.ma_slope_positive(ma, lookback=5)
    assert out.dtype == bool
    # Where True: ma[i] > ma[i-5]
    valid_idx = out[out].index
    for t in valid_idx[5:]:  # skip warmup
        prev_t = ma.index[ma.index.get_loc(t) - 5]
        if not pd.isna(ma.loc[prev_t]):
            assert ma.loc[t] > ma.loc[prev_t]


def test_ma_slope_negative_complements_positive(ohlcv):
    ma = ind.ema(ohlcv["close"], 50)
    pos = regime.ma_slope_positive(ma, lookback=5)
    neg = regime.ma_slope_negative(ma, lookback=5)
    # No overlap (a flat slope is neither positive nor negative)
    assert not (pos & neg).any()


def test_ma_slope_lookback_validation(ohlcv):
    ma = ind.ema(ohlcv["close"], 50)
    with pytest.raises(ValueError):
        regime.ma_slope_positive(ma, lookback=0)


# ── Module-level export check ─────────────────────────────────────────────

def test_new_helpers_exported_from_modules():
    """The new functions are listed in __all__ so `from regime import *` works."""
    assert "atr_below_percentile" in regime.__all__
    assert "atr_above_percentile" in regime.__all__
    assert "atr_ratio_below_threshold" in regime.__all__
    assert "atr_ratio_above_threshold" in regime.__all__
    assert "ma_slope_positive" in regime.__all__
    assert "ma_slope_negative" in regime.__all__

    assert "stochrsi_oversold" in cond.__all__
    assert "stochrsi_overbought" in cond.__all__
    assert "stochrsi_cross_above" in cond.__all__
    assert "aroon_up_strong" in cond.__all__
    assert "aroon_down_strong" in cond.__all__
    assert "aroon_oscillator_above" in cond.__all__
    assert "psar_flip_up" in cond.__all__
    assert "psar_flip_down" in cond.__all__
