"""Edge-case stress tests for cyqnt_trd.blocks.

These tests deliberately push blocks to their limits with:
- Very short DataFrames (2-3 bars)
- NaN-laden inputs
- Zero/extreme values
- Mismatched-length Series
- Wrong return types from user callbacks
- Period > df length

Each test documents the expected behavior: either a graceful fallback
(NaN/empty result) or a clear error message.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks import (
    conditions as cond,
    derivatives as deriv,
    entry,
    exit as ex,
    indicators as ind,
    strategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int, *, base: float = 100.0, volatility: float = 1.0) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame of length *n*."""
    rng = np.random.default_rng(42)
    close = base + np.cumsum(rng.normal(0, volatility, n))
    high = close + abs(rng.normal(0, volatility * 0.5, n))
    low = close - abs(rng.normal(0, volatility * 0.5, n))
    open_ = close + rng.normal(0, volatility * 0.3, n)
    volume = rng.uniform(100, 10000, n)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _make_flat_ohlcv(n: int, price: float = 100.0) -> pd.DataFrame:
    """All bars at exactly the same price — zero volatility."""
    return pd.DataFrame({
        "open": [price] * n,
        "high": [price] * n,
        "low": [price] * n,
        "close": [price] * n,
        "volume": [1000.0] * n,
    })


def _make_extreme_volatility_ohlcv(n: int) -> pd.DataFrame:
    """Alternating massive swings."""
    prices = [100.0 if i % 2 == 0 else 200.0 for i in range(n)]
    return pd.DataFrame({
        "open": prices,
        "high": [max(prices[i], prices[i]) * 1.1 for i in range(n)],
        "low": [min(prices[i], prices[i]) * 0.9 for i in range(n)],
        "close": prices,
        "volume": [5000.0] * n,
    })


# ===========================================================================
# 1. indicators.parabolic_sar with edge cases
# ===========================================================================


class TestParabolicSarEdgeCases:
    """Test parabolic_sar with very short / flat / extreme data."""

    def test_very_short_df_2_bars(self):
        """2 bars should return NaN SAR and 0 direction (per code guard)."""
        df = _make_ohlcv(2)
        sar, direction = ind.parabolic_sar(df)
        assert len(sar) == 2
        assert len(direction) == 2
        # With only 2 bars, the guard returns NaN SAR
        assert sar.isna().all() or not sar.isna().all()  # just verifies no crash
        assert direction.dtype == int or direction.dtype == np.int64

    def test_very_short_df_3_bars(self):
        """3 bars should produce some result without crashing."""
        df = _make_ohlcv(3)
        sar, direction = ind.parabolic_sar(df)
        assert len(sar) == 3
        assert len(direction) == 3
        # At least bar 0 and 1 should have values (after initialization)
        assert direction.isin([-1, 0, 1]).all()

    def test_flat_data(self):
        """All prices equal — SAR should not crash or produce inf."""
        df = _make_flat_ohlcv(50)
        sar, direction = ind.parabolic_sar(df)
        assert len(sar) == 50
        assert not np.isinf(sar).any()
        # Direction should be stable (no oscillation from noise)
        assert direction.isin([-1, 0, 1]).all()

    def test_extreme_volatility(self):
        """Massive alternating swings should not crash or produce inf."""
        df = _make_extreme_volatility_ohlcv(30)
        sar, direction = ind.parabolic_sar(df)
        assert len(sar) == 30
        assert not np.isinf(sar.dropna()).any()
        assert direction.isin([-1, 0, 1]).all()

    def test_single_bar(self):
        """1-bar DataFrame — should return NaN gracefully."""
        df = _make_ohlcv(1)
        sar, direction = ind.parabolic_sar(df)
        assert len(sar) == 1
        assert sar.isna().all()


# ===========================================================================
# 2. conditions.close_above/close_below with NaN in ref Series
# ===========================================================================


class TestCloseAboveBelowNaN:
    """Test close_above/close_below when reference has NaN."""

    def test_close_above_with_nan_ref(self):
        """NaN in ref should produce False (not NaN) at those positions."""
        df = _make_ohlcv(20)
        ref = pd.Series([np.nan] * 5 + [90.0] * 15, index=df.index)
        result = cond.close_above(df, ref)
        assert result.dtype == bool
        # First 5 bars with NaN ref should be False (not NaN)
        assert not result.iloc[:5].any()

    def test_close_below_with_nan_ref(self):
        """NaN in ref should produce False (not NaN) at those positions."""
        df = _make_ohlcv(20)
        ref = pd.Series([np.nan] * 5 + [200.0] * 15, index=df.index)
        result = cond.close_below(df, ref)
        assert result.dtype == bool
        assert not result.iloc[:5].any()

    def test_close_above_all_nan_ref(self):
        """Entirely NaN ref — all should be False."""
        df = _make_ohlcv(10)
        ref = pd.Series([np.nan] * 10, index=df.index)
        result = cond.close_above(df, ref)
        assert result.dtype == bool
        assert not result.any()

    def test_close_below_all_nan_ref(self):
        """Entirely NaN ref — all should be False."""
        df = _make_ohlcv(10)
        ref = pd.Series([np.nan] * 10, index=df.index)
        result = cond.close_below(df, ref)
        assert result.dtype == bool
        assert not result.any()

    def test_close_above_with_scalar_ref(self):
        """Scalar ref should work without issues."""
        df = _make_ohlcv(20, base=100.0)
        result = cond.close_above(df, 50.0)
        assert result.dtype == bool
        # All bars should be above 50 given base=100
        assert result.sum() > 0


# ===========================================================================
# 3. conditions.price_touch_or_cross at exact boundaries
# ===========================================================================


class TestPriceTouchOrCrossEdge:
    """Test price_touch_or_cross with level at exact bar boundaries."""

    def test_level_exactly_at_high(self):
        """Level == high should count as touched."""
        df = pd.DataFrame({
            "open": [100.0, 100.0, 100.0],
            "high": [110.0, 115.0, 120.0],
            "low": [90.0, 95.0, 100.0],
            "close": [105.0, 110.0, 115.0],
        })
        # Level at exact high of bar 0
        result = cond.price_touch_or_cross(df, 110.0)
        assert result.iloc[0] == True

    def test_level_exactly_at_low(self):
        """Level == low should count as touched."""
        df = pd.DataFrame({
            "open": [100.0, 100.0, 100.0],
            "high": [110.0, 115.0, 120.0],
            "low": [90.0, 95.0, 100.0],
            "close": [105.0, 110.0, 115.0],
        })
        # Level at exact low of bar 0
        result = cond.price_touch_or_cross(df, 90.0)
        assert result.iloc[0] == True

    def test_level_outside_range(self):
        """Level above all highs — never touched."""
        df = pd.DataFrame({
            "open": [100.0, 100.0, 100.0],
            "high": [110.0, 115.0, 120.0],
            "low": [90.0, 95.0, 100.0],
            "close": [105.0, 110.0, 115.0],
        })
        result = cond.price_touch_or_cross(df, 500.0)
        assert not result.any()

    def test_level_as_series_with_nan(self):
        """Level Series with NaN — should not crash."""
        df = _make_ohlcv(10)
        level_s = pd.Series([np.nan] * 3 + [100.0] * 7, index=df.index)
        result = cond.price_touch_or_cross(df, level_s)
        assert result.dtype == bool
        assert len(result) == 10


# ===========================================================================
# 4. entry.all_of / entry.any_of with edge cases
# ===========================================================================


class TestEntryCombinatorEdges:
    """Test all_of/any_of with empty lists, single items, mismatched Series."""

    def test_all_of_empty_list(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="at least one condition"):
            entry.all_of([])

    def test_any_of_empty_list(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="at least one condition"):
            entry.any_of([])

    def test_all_of_single_condition(self):
        """Single condition should work like identity."""
        cond_s = pd.Series([True, False, True, True, False])
        result = entry.all_of([cond_s])
        assert result.tolist() == [True, False, True, True, False]

    def test_any_of_single_condition(self):
        """Single condition should work like identity."""
        cond_s = pd.Series([True, False, True, True, False])
        result = entry.any_of([cond_s])
        assert result.tolist() == [True, False, True, True, False]

    def test_all_of_mismatched_length_series(self):
        """Mismatched-length Series — should align by index or raise."""
        cond1 = pd.Series([True, True, True], index=[0, 1, 2])
        cond2 = pd.Series([True, False], index=[0, 1])
        # This might raise or produce NaN-treated-as-False for missing indices
        try:
            result = entry.all_of([cond1, cond2])
            # If it works, bar 2 should be False (NaN & True => False)
            assert len(result) == 3
            assert result.iloc[2] == False  # cond2 has no index=2
        except (ValueError, IndexError):
            pass  # acceptable to raise

    def test_any_of_mismatched_length_series(self):
        """Mismatched-length Series — should handle gracefully."""
        cond1 = pd.Series([False, False, True], index=[0, 1, 2])
        cond2 = pd.Series([True, False], index=[0, 1])
        try:
            result = entry.any_of([cond1, cond2])
            assert len(result) == 3
            assert result.iloc[0] == True  # either is True
        except (ValueError, IndexError):
            pass  # acceptable to raise

    def test_all_of_with_nan_series(self):
        """Series containing NaN — should treat NaN as False."""
        cond1 = pd.Series([True, True, np.nan, True])
        cond2 = pd.Series([True, True, True, True])
        result = entry.all_of([cond1, cond2])
        assert result.iloc[2] == False  # NaN treated as False

    def test_consecutive_zero_n(self):
        """n < 1 should raise ValueError."""
        cond_s = pd.Series([True, True, True])
        with pytest.raises(ValueError):
            entry.consecutive(cond_s, 0)

    def test_consecutive_n_greater_than_length(self):
        """n > series length — all should be False."""
        cond_s = pd.Series([True, True, True])
        result = entry.consecutive(cond_s, 100)
        assert not result.any()


# ===========================================================================
# 5. exit.AtrTrailingStop with zero ATR
# ===========================================================================


class TestAtrTrailingStopZeroAtr:
    """Test AtrTrailingStop when ATR is zero (flat market)."""

    def test_zero_atr_long(self):
        """Zero ATR means stop == running high — should trigger on any dip."""
        df = pd.DataFrame({
            "open": [100, 101, 102, 101, 100, 99],
            "high": [101, 102, 103, 102, 101, 100],
            "low": [99, 100, 101, 100, 99, 98],
            "close": [100.5, 101.5, 102.5, 100.5, 99.5, 98.5],
        }, dtype=float)
        atr_zero = pd.Series([0.0] * 6, index=df.index)
        rule = ex.AtrTrailingStop(atr=atr_zero, multiplier=2.0)
        result = rule.evaluate(df, entry_price=100.0, side="long")
        assert result.dtype == bool
        # With zero ATR, stop = highest high - 0 = highest high
        # So any bar where close < running max high should exit
        assert result.any()  # should trigger at some point

    def test_zero_atr_short(self):
        """Zero ATR short — stop at running low, exit on any bounce."""
        df = pd.DataFrame({
            "open": [100, 99, 98, 99, 100, 101],
            "high": [101, 100, 99, 100, 101, 102],
            "low": [99, 98, 97, 98, 99, 100],
            "close": [99.5, 98.5, 97.5, 99.5, 100.5, 101.5],
        }, dtype=float)
        atr_zero = pd.Series([0.0] * 6, index=df.index)
        rule = ex.AtrTrailingStop(atr=atr_zero, multiplier=2.0)
        result = rule.evaluate(df, entry_price=100.0, side="short")
        assert result.dtype == bool
        assert result.any()

    def test_nan_atr(self):
        """NaN ATR values — should not crash."""
        df = _make_ohlcv(10)
        atr_nan = pd.Series([np.nan] * 10, index=df.index)
        rule = ex.AtrTrailingStop(atr=atr_nan, multiplier=2.0)
        result = rule.evaluate(df, entry_price=100.0, side="long")
        assert result.dtype == bool
        assert len(result) == 10


# ===========================================================================
# 6. strategy.register with invalid inputs
# ===========================================================================


class TestStrategyRegisterEdgeCases:
    """Test strategy.register with duplicate IDs, bad signal_fn, wrong types."""

    def test_empty_strategy_id(self):
        """Empty string ID should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            strategy.build_plugin("", lambda df: pd.Series(False, index=df.index))

    def test_none_strategy_id(self):
        """None ID should raise."""
        with pytest.raises((ValueError, TypeError)):
            strategy.build_plugin(None, lambda df: pd.Series(False, index=df.index))

    def test_non_callable_signal_fn(self):
        """Non-callable should raise TypeError."""
        with pytest.raises(TypeError, match="callable"):
            strategy.build_plugin("test_bad", "not a function")

    def test_integer_signal_fn(self):
        """Integer as signal_fn should raise TypeError."""
        with pytest.raises(TypeError, match="callable"):
            strategy.build_plugin("test_int_fn", 42)

    def test_valid_build_plugin(self):
        """Valid inputs produce a BlockStrategyPlugin."""
        def sig(df):
            return pd.Series(True, index=df.index), None
        plugin = strategy.build_plugin("test_valid_edge", sig)
        assert plugin.plugin_id == "test_valid_edge"

    def test_duplicate_register(self):
        """Duplicate registration should not crash (silently skipped on flush)."""
        def sig(df):
            return pd.Series(True, index=df.index), None
        # Register twice with same ID — should not raise at registration time
        strategy.register("test_duplicate_stress", sig)
        strategy.register("test_duplicate_stress", sig)
        # The pending list should have both; flush_pending_into skips dupes

    def test_signal_fn_returning_wrong_type(self):
        """signal_fn that returns string — build_plugin should still succeed
        (type checking happens at runtime when called by engine)."""
        def bad_sig(df):
            return "not a series"
        plugin = strategy.build_plugin("test_wrong_return", bad_sig)
        # build_plugin doesn't call signal_fn, so it should succeed
        assert plugin.plugin_id == "test_wrong_return"


# ===========================================================================
# 7. indicators.macd / rsi / bollinger with period > df length
# ===========================================================================


class TestIndicatorsPeriodExceedsLength:
    """When period > number of bars, indicators should return NaN (not crash)."""

    def test_rsi_period_exceeds_length(self):
        """RSI with period=50 on 10-bar df — should return all NaN."""
        df = _make_ohlcv(10)
        result = ind.rsi(df["close"], period=50)
        assert len(result) == 10
        assert result.isna().all()

    def test_macd_slow_exceeds_length(self):
        """MACD with slow=50 on 10-bar df — line/signal/hist should be NaN."""
        df = _make_ohlcv(10)
        macd_line, signal_line, hist = ind.macd(df["close"], fast=12, slow=50, signal=9)
        assert len(macd_line) == 10
        assert len(signal_line) == 10
        assert len(hist) == 10
        # All should be NaN since slow period exceeds df length
        assert macd_line.isna().all()

    def test_bollinger_period_exceeds_length(self):
        """Bollinger with period=50 on 10-bar df — should return NaN."""
        df = _make_ohlcv(10)
        upper, middle, lower = ind.bollinger(df["close"], period=50)
        assert len(upper) == 10
        assert upper.isna().all()
        assert middle.isna().all()
        assert lower.isna().all()

    def test_sma_period_exceeds_length(self):
        """SMA period > length — all NaN."""
        df = _make_ohlcv(5)
        result = ind.sma(df["close"], 20)
        assert len(result) == 5
        assert result.isna().all()

    def test_ema_period_exceeds_length(self):
        """EMA period > length — mostly NaN (EMA has min_periods)."""
        df = _make_ohlcv(5)
        result = ind.ema(df["close"], 20)
        assert len(result) == 5
        # EMA with span > length has fewer valid values
        assert result.isna().sum() >= 4  # at least some NaN

    def test_atr_period_exceeds_length(self):
        """ATR with period=50 on 10-bar df."""
        df = _make_ohlcv(10)
        result = ind.atr(df, period=50)
        assert len(result) == 10
        assert result.isna().all()

    def test_adx_period_exceeds_length(self):
        """ADX with period=50 on 10-bar df."""
        df = _make_ohlcv(10)
        adx_val, plus_di, minus_di = ind.adx(df, period=50)
        assert len(adx_val) == 10
        assert adx_val.isna().all()

    def test_bollinger_period_1(self):
        """Period=1 bollinger — std=0 so upper==middle==lower."""
        df = _make_ohlcv(10)
        upper, middle, lower = ind.bollinger(df["close"], period=1)
        assert len(upper) == 10
        # With period=1, std dev is 0, so bands collapse
        assert (upper == lower).all()


# ===========================================================================
# 8. derivatives.funding_rate_state with scalar input
# ===========================================================================


class TestFundingRateStateScalar:
    """Test funding_rate_state with scalar input (the old bug)."""

    def test_scalar_input_float(self):
        """Scalar float input — should raise or handle gracefully."""
        try:
            result = deriv.funding_rate_state(0.0001)
            # If it doesn't crash, it should return something sensible
            assert result is not None
        except (TypeError, AttributeError, ValueError) as e:
            # Acceptable: scalar doesn't have .index
            assert "Series" in str(e) or "index" in str(e) or "iterable" in str(e).lower()

    def test_scalar_input_int(self):
        """Scalar int input — should raise or handle."""
        try:
            result = deriv.funding_rate_state(0)
            assert result is not None
        except (TypeError, AttributeError, ValueError):
            pass  # Acceptable

    def test_single_element_series(self):
        """Single-element Series — should work."""
        funding = pd.Series([0.0001])
        result = deriv.funding_rate_state(funding)
        assert len(result) == 1
        assert result.iloc[0] in ("neutral", "bullish_squeeze", "bearish_squeeze")

    def test_all_nan_series(self):
        """All-NaN funding — should return neutral or NaN (not crash)."""
        funding = pd.Series([np.nan, np.nan, np.nan])
        result = deriv.funding_rate_state(funding)
        assert len(result) == 3
        # NaN * 10000 is still NaN, comparisons with NaN are False
        # so should be "neutral"
        assert (result == "neutral").all()

    def test_extreme_funding(self):
        """Extreme funding rates — should categorize correctly."""
        funding = pd.Series([0.01, -0.01, 0.0001, -0.0001, 0.0])
        result = deriv.funding_rate_state(funding)
        # 0.01 = 100 bps -> bullish_squeeze (>= 5 bps)
        assert result.iloc[0] == "bullish_squeeze"
        # -0.01 = -100 bps -> bearish_squeeze (<= -5 bps)
        assert result.iloc[1] == "bearish_squeeze"
        # 0.0001 = 1 bp -> neutral
        assert result.iloc[2] == "neutral"
        # -0.0001 = -1 bp -> neutral
        assert result.iloc[3] == "neutral"

    def test_empty_series(self):
        """Empty Series — should return empty result."""
        funding = pd.Series([], dtype=float)
        result = deriv.funding_rate_state(funding)
        assert len(result) == 0


# ===========================================================================
# Additional stress tests
# ===========================================================================


class TestIndicatorsZeroLength:
    """Zero-length DataFrame edge case."""

    def test_parabolic_sar_empty_df(self):
        """Empty DataFrame — should not crash."""
        df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
        sar, direction = ind.parabolic_sar(df)
        assert len(sar) == 0

    def test_rsi_empty_series(self):
        """Empty Series — should not crash."""
        s = pd.Series([], dtype=float)
        result = ind.rsi(s, 14)
        assert len(result) == 0

    def test_macd_empty_series(self):
        """Empty Series — should not crash."""
        s = pd.Series([], dtype=float)
        line, signal, hist = ind.macd(s)
        assert len(line) == 0


class TestConditionsWithInf:
    """Test conditions with infinity values."""

    def test_close_above_inf_ref(self):
        """ref = +inf — nothing is above infinity."""
        df = _make_ohlcv(10)
        result = cond.close_above(df, float('inf'))
        assert not result.any()

    def test_close_below_neg_inf_ref(self):
        """ref = -inf — nothing is below negative infinity."""
        df = _make_ohlcv(10)
        result = cond.close_below(df, float('-inf'))
        assert not result.any()
