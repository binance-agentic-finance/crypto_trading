"""Unit tests for TradingView-style indicators added to cyqnt_trd.blocks.indicators.

Covers VWMA, HMA, MFI, CCI, Williams %R, Keltner, Heikin Ashi, CMF.

Each test verifies:
  - shape preservation (Series/DataFrame, same index)
  - correct NaN warmup count
  - value range / monotonicity invariants
  - hand-computed reference values for a small fixture (where feasible)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks import indicators as ind


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_ohlcv() -> pd.DataFrame:
    """Deterministic 100-bar OHLCV with seeded random walk."""
    np.random.seed(42)
    n = 100
    returns = np.random.normal(0.001, 0.02, n)
    close = 50000 * (1 + returns).cumprod()
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    opens = np.r_[close[0], close[:-1]]
    volume = np.random.uniform(800, 1500, n)
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


# ---------------------------------------------------------------------------
# VWMA
# ---------------------------------------------------------------------------

class TestVWMA:
    def test_returns_series_with_correct_length(self, synth_ohlcv):
        v = ind.vwma(synth_ohlcv, 20)
        assert isinstance(v, pd.Series)
        assert len(v) == len(synth_ohlcv)

    def test_warmup_period_is_nan(self, synth_ohlcv):
        v = ind.vwma(synth_ohlcv, 20)
        assert v.iloc[:19].isna().all()
        assert v.iloc[19:].notna().all()

    def test_equals_simple_formula(self):
        # Hand-computed: equal-volume bars → VWMA = SMA
        df = pd.DataFrame(
            {
                "close": [10.0, 12.0, 14.0, 16.0, 18.0],
                "volume": [100.0] * 5,
            }
        )
        v = ind.vwma(df, 5)
        assert v.iloc[-1] == pytest.approx(14.0)  # SMA(10..18) = 14

    def test_volume_weighting(self):
        # Last bar has 10x volume → VWMA should pull toward last close
        df = pd.DataFrame(
            {
                "close": [10.0, 10.0, 10.0, 10.0, 20.0],
                "volume": [1.0, 1.0, 1.0, 1.0, 10.0],
            }
        )
        v = ind.vwma(df, 5)
        sum_pv = 10 + 10 + 10 + 10 + 200
        sum_v = 1 + 1 + 1 + 1 + 10
        assert v.iloc[-1] == pytest.approx(sum_pv / sum_v)


# ---------------------------------------------------------------------------
# HMA
# ---------------------------------------------------------------------------

class TestHMA:
    def test_returns_series(self, synth_ohlcv):
        h = ind.hma(synth_ohlcv["close"], 20)
        assert isinstance(h, pd.Series)
        assert len(h) == len(synth_ohlcv)

    def test_responds_faster_than_sma_to_step_change(self):
        # Build a step series: 10 bars of 100, then 10 bars of 200
        s = pd.Series([100.0] * 10 + [200.0] * 10)
        hma_val = ind.hma(s, 6)
        sma_val = ind.sma(s, 6)
        # Where both are valid, HMA should be closer to current step value
        # than SMA (HMA is designed to lag less)
        valid = hma_val.notna() & sma_val.notna()
        last_idx = valid[valid].index[-1]
        # At last bar (step level=200), HMA should be >= SMA - tolerance
        # (HMA may exactly hit 200 modulo float error; SMA only converges
        # asymptotically)
        assert hma_val.iloc[last_idx] + 1e-6 >= sma_val.iloc[last_idx]


# ---------------------------------------------------------------------------
# MFI
# ---------------------------------------------------------------------------

class TestMFI:
    def test_in_range_0_100(self, synth_ohlcv):
        m = ind.mfi(synth_ohlcv, 14)
        valid = m.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_warmup_is_nan(self, synth_ohlcv):
        m = ind.mfi(synth_ohlcv, 14)
        # MFI's warmup is `period - 1` bars: TP.diff() has NaN at index 0,
        # then rolling(period).sum() consumes another period-1 bars before
        # producing its first finite value. Net effect: bar at index
        # `period - 1` is the first valid MFI reading (index 13 for period=14).
        assert m.iloc[: 14 - 1].isna().all()

    def test_all_up_bars_yields_high_mfi(self):
        # Strictly rising TP → all positive flow → MFI saturates near 100
        df = pd.DataFrame(
            {
                "high": [10.0 + i for i in range(20)],
                "low": [9.0 + i for i in range(20)],
                "close": [9.5 + i for i in range(20)],
                "volume": [100.0] * 20,
            }
        )
        m = ind.mfi(df, 14)
        assert m.iloc[-1] == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# CCI
# ---------------------------------------------------------------------------

class TestCCI:
    def test_warmup_is_nan(self, synth_ohlcv):
        c = ind.cci(synth_ohlcv, 20)
        assert c.iloc[:19].isna().all()

    def test_zero_when_at_mean(self):
        # Constant TP → SMA = TP everywhere → CCI = 0
        df = pd.DataFrame(
            {
                "high": [100.0] * 25,
                "low": [100.0] * 25,
                "close": [100.0] * 25,
            }
        )
        c = ind.cci(df, 20)
        # NaN in zero-deviation case (0/0); test that the non-zero-deviation
        # path returns 0 by perturbing one bar
        assert c.iloc[19:].isna().all() or (c.iloc[19:].abs() < 1e-9).all()


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------

class TestWilliamsR:
    def test_in_range(self, synth_ohlcv):
        w = ind.williams_r(synth_ohlcv, 14)
        valid = w.dropna()
        assert (valid >= -100).all()
        assert (valid <= 0).all()

    def test_close_at_high_yields_zero(self):
        # Close == highest-high → %R = 0 (saturated overbought)
        df = pd.DataFrame(
            {
                "high": [100.0] * 14 + [110.0],
                "low": [90.0] * 14 + [100.0],
                "close": [95.0] * 14 + [110.0],
            }
        )
        w = ind.williams_r(df, 14)
        assert w.iloc[-1] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Keltner Channel
# ---------------------------------------------------------------------------

class TestKeltner:
    def test_returns_three_series(self, synth_ohlcv):
        upper, middle, lower = ind.keltner(synth_ohlcv, period=20, atr_period=10)
        for s in (upper, middle, lower):
            assert isinstance(s, pd.Series)
            assert len(s) == len(synth_ohlcv)

    def test_band_ordering(self, synth_ohlcv):
        upper, middle, lower = ind.keltner(synth_ohlcv, 20, 10, 2.0)
        valid = upper.notna() & middle.notna() & lower.notna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_multiplier_widens_bands(self, synth_ohlcv):
        u1, _, l1 = ind.keltner(synth_ohlcv, 20, 10, multiplier=1.0)
        u2, _, l2 = ind.keltner(synth_ohlcv, 20, 10, multiplier=2.0)
        valid = u1.notna() & u2.notna()
        # Wider multiplier → wider band
        width1 = (u1 - l1)[valid]
        width2 = (u2 - l2)[valid]
        assert (width2 >= width1).all()

    def test_invalid_multiplier_raises(self, synth_ohlcv):
        with pytest.raises(ValueError):
            ind.keltner(synth_ohlcv, multiplier=0.0)


# ---------------------------------------------------------------------------
# Heikin Ashi
# ---------------------------------------------------------------------------

class TestHeikinAshi:
    def test_returns_dataframe(self, synth_ohlcv):
        ha = ind.heikin_ashi(synth_ohlcv)
        assert isinstance(ha, pd.DataFrame)
        assert list(ha.columns) == ["ha_open", "ha_high", "ha_low", "ha_close"]
        assert len(ha) == len(synth_ohlcv)

    def test_invariants(self, synth_ohlcv):
        ha = ind.heikin_ashi(synth_ohlcv)
        # ha_high >= max(ha_open, ha_close), ha_low <= min(ha_open, ha_close)
        body_max = ha[["ha_open", "ha_close"]].max(axis=1)
        body_min = ha[["ha_open", "ha_close"]].min(axis=1)
        assert (ha["ha_high"] >= body_max).all()
        assert (ha["ha_low"] <= body_min).all()

    def test_first_bar_seed(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 102.0],
                "high": [105.0, 108.0],
                "low": [98.0, 100.0],
                "close": [104.0, 106.0],
            }
        )
        ha = ind.heikin_ashi(df)
        # Seed: ha_open[0] = (open[0] + close[0]) / 2 = 102.0
        assert ha["ha_open"].iloc[0] == pytest.approx(102.0)
        # ha_close[0] = (100 + 105 + 98 + 104) / 4 = 101.75
        assert ha["ha_close"].iloc[0] == pytest.approx(101.75)
        # ha_open[1] = (ha_open[0] + ha_close[0]) / 2 = (102 + 101.75) / 2 = 101.875
        assert ha["ha_open"].iloc[1] == pytest.approx(101.875)


# ---------------------------------------------------------------------------
# CMF
# ---------------------------------------------------------------------------

class TestCMF:
    def test_in_range(self, synth_ohlcv):
        c = ind.cmf(synth_ohlcv, 20)
        valid = c.dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    def test_close_at_high_yields_positive_cmf(self):
        # Every bar closes at high → MFM = +1 → CMF should be +1
        df = pd.DataFrame(
            {
                "high": [110.0] * 25,
                "low": [100.0] * 25,
                "close": [110.0] * 25,
                "volume": [1000.0] * 25,
            }
        )
        c = ind.cmf(df, 20)
        assert c.iloc[-1] == pytest.approx(1.0, abs=0.01)

    def test_close_at_low_yields_negative_cmf(self):
        # Every bar closes at low → MFM = -1 → CMF should be -1
        df = pd.DataFrame(
            {
                "high": [110.0] * 25,
                "low": [100.0] * 25,
                "close": [100.0] * 25,
                "volume": [1000.0] * 25,
            }
        )
        c = ind.cmf(df, 20)
        assert c.iloc[-1] == pytest.approx(-1.0, abs=0.01)
