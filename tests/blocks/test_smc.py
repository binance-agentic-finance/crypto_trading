"""Unit tests for SMC (Smart Money Concepts) block functions.

Covers:
  - Fractal pivot detection (high / low)
  - Fair Value Gap identification
  - Order Block detection
  - Break of Structure / Change of Character (BOS/CHoCH)
  - Liquidity Sweep detection
  - Equal Highs / Equal Lows clustering
  - Premium / Discount Zone labelling
  - Full integration pipeline across four real-market fixtures

Fixture legend
--------------
btc_1h  : BTCUSDT 1-hour, 500 bars
eth_1h  : ETHUSDT 1-hour, 500 bars
btc_4h  : BTCUSDT 4-hour, 300 bars
btc_15m : BTCUSDT 15-minute, 500 bars
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks.smc_structure import (
    bos_choch_detect,
    fair_value_gap,
    fractal_pivot_high,
    fractal_pivot_low,
    order_block_detect,
)
from cyqnt_trd.blocks.smc_liquidity import (
    equal_highs_lows,
    liquidity_sweep_detect,
    premium_discount_zone,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FIXTURE_DIR = "tests/blocks/fixtures"


@pytest.fixture
def btc_1h() -> pd.DataFrame:
    """BTCUSDT 1-hour, 500 bars."""
    return pd.read_parquet(f"{_FIXTURE_DIR}/BTCUSDT_1h_500bars.parquet")


@pytest.fixture
def eth_1h() -> pd.DataFrame:
    """ETHUSDT 1-hour, 500 bars."""
    return pd.read_parquet(f"{_FIXTURE_DIR}/ETHUSDT_1h_500bars.parquet")


@pytest.fixture
def btc_4h() -> pd.DataFrame:
    """BTCUSDT 4-hour, 300 bars."""
    return pd.read_parquet(f"{_FIXTURE_DIR}/BTCUSDT_4h_300bars.parquet")


@pytest.fixture
def btc_15m() -> pd.DataFrame:
    """BTCUSDT 15-minute, 500 bars."""
    return pd.read_parquet(f"{_FIXTURE_DIR}/BTCUSDT_15m_500bars.parquet")


# ---------------------------------------------------------------------------
# TestFractalPivot
# ---------------------------------------------------------------------------


class TestFractalPivot:
    """Tests for fractal_pivot_high and fractal_pivot_low."""

    def test_high_returns_series_same_length(self, btc_1h):
        ph = fractal_pivot_high(btc_1h)
        assert isinstance(ph, pd.Series)
        assert len(ph) == len(btc_1h)

    def test_high_pivots_are_local_maxima(self, btc_1h):
        """Every confirmed pivot high must be >= all neighbor highs in ±lookback bars."""
        lookback = 5
        ph = fractal_pivot_high(btc_1h, lookback)
        high = btc_1h["high"].values
        n = len(high)

        for i, val in enumerate(ph.values):
            if np.isnan(val):
                continue
            start = max(0, i - lookback)
            end = min(n - 1, i + lookback)
            # Collect neighbors (exclude position i itself)
            window = np.concatenate([high[start:i], high[i + 1 : end + 1]])
            if len(window) > 0:
                assert high[i] >= window.max() - 1e-9, (
                    f"Pivot at bar {i}: high={high[i]:.2f}, "
                    f"neighbor max={window.max():.2f}"
                )

    def test_low_pivots_mirror_high(self, btc_1h):
        """Pivot-high count and pivot-low count should be within 30 % of each other."""
        n_high = fractal_pivot_high(btc_1h).notna().sum()
        n_low = fractal_pivot_low(btc_1h).notna().sum()
        assert n_high > 0 and n_low > 0
        ratio = abs(n_high - n_low) / max(n_high, n_low)
        assert ratio <= 0.30, (
            f"High/low pivot imbalance ratio={ratio:.2f} > 0.30 "
            f"(high={n_high}, low={n_low})"
        )

    def test_increasing_lookback_decreases_pivot_count(self, btc_1h):
        """lookback=10 should produce no more pivot highs than lookback=5."""
        n5 = fractal_pivot_high(btc_1h, lookback=5).notna().sum()
        n10 = fractal_pivot_high(btc_1h, lookback=10).notna().sum()
        assert n10 <= n5, (
            f"lookback=10 gave {n10} pivots but lookback=5 gave only {n5}"
        )


# ---------------------------------------------------------------------------
# TestFairValueGap
# ---------------------------------------------------------------------------


class TestFairValueGap:
    """Tests for fair_value_gap."""

    def test_returns_dataframe_with_expected_columns(self, btc_1h):
        fvg = fair_value_gap(btc_1h)
        assert isinstance(fvg, pd.DataFrame)
        assert list(fvg.columns) == [
            "fvg_top",
            "fvg_bottom",
            "fvg_direction",
            "fvg_size_pct",
        ]
        assert len(fvg) == len(btc_1h)

    def test_first_two_bars_no_fvg(self, btc_1h):
        """Bars 0 and 1 need bar i-2 data, so all FVG columns must be NaN there."""
        fvg = fair_value_gap(btc_1h)
        assert fvg.iloc[:2].isna().all().all()

    def test_size_pct_consistent(self, btc_1h):
        """fvg_size_pct == (fvg_top - fvg_bottom) / close * 100 for all active FVGs."""
        fvg = fair_value_gap(btc_1h)
        close = btc_1h["close"]
        mask = fvg["fvg_direction"].notna()
        expected = (
            (fvg.loc[mask, "fvg_top"] - fvg.loc[mask, "fvg_bottom"])
            / close.loc[mask]
            * 100.0
        )
        np.testing.assert_allclose(
            fvg.loc[mask, "fvg_size_pct"].values,
            expected.values,
            rtol=1e-6,
            err_msg="fvg_size_pct != (top-bottom)/close*100",
        )

    def test_count_reasonable_btc_1h(self, btc_1h):
        """FVG event count on 500 bars should be between 30 and 200."""
        count = fair_value_gap(btc_1h)["fvg_direction"].notna().sum()
        assert 30 <= count <= 200, f"FVG count={count} outside [30, 200]"


# ---------------------------------------------------------------------------
# TestOrderBlock
# ---------------------------------------------------------------------------


class TestOrderBlock:
    """Tests for order_block_detect."""

    def test_returns_dataframe(self, btc_1h):
        ob = order_block_detect(btc_1h)
        assert isinstance(ob, pd.DataFrame)
        assert {"ob_top", "ob_bottom", "ob_direction"}.issubset(ob.columns)
        assert len(ob) == len(btc_1h)

    def test_ob_top_greater_than_bottom(self, btc_1h):
        """Each detected Order Block must have ob_top strictly above ob_bottom."""
        ob = order_block_detect(btc_1h)
        mask = ob["ob_direction"].notna()
        if mask.any():
            assert (ob.loc[mask, "ob_top"] > ob.loc[mask, "ob_bottom"]).all(), (
                "Found OB row(s) where ob_top <= ob_bottom"
            )

    def test_count_reasonable_btc_1h(self, btc_1h):
        """OB event count on 500 bars should be between 5 and 40."""
        count = order_block_detect(btc_1h)["ob_direction"].notna().sum()
        assert 5 <= count <= 40, f"OB count={count} outside [5, 40]"


# ---------------------------------------------------------------------------
# TestBOSCHoCH
# ---------------------------------------------------------------------------


class TestBOSCHoCH:
    """Tests for bos_choch_detect."""

    def test_returns_dataframe(self, btc_1h):
        bc = bos_choch_detect(btc_1h)
        assert isinstance(bc, pd.DataFrame)
        assert {"structure_event", "trend_state"}.issubset(bc.columns)
        assert len(bc) == len(btc_1h)

    def test_event_classes_only_4(self, btc_1h):
        """structure_event values must be a subset of the 4 valid event strings."""
        bc = bos_choch_detect(btc_1h)
        valid = {"BOS_BULL", "BOS_BEAR", "CHOCH_BULL", "CHOCH_BEAR"}
        actual = set(bc["structure_event"].dropna().unique())
        assert actual.issubset(valid), f"Unexpected events found: {actual - valid}"

    def test_trend_state_in_set(self, btc_1h):
        """trend_state must always be 'UP', 'DOWN', or 'NEUTRAL' — nothing else."""
        bc = bos_choch_detect(btc_1h)
        valid = {"UP", "DOWN", "NEUTRAL"}
        actual = set(bc["trend_state"].unique())
        assert actual.issubset(valid), (
            f"Unexpected trend states: {actual - valid}"
        )

    def test_choch_changes_trend(self, btc_1h):
        """CHOCH_BULL rows must have trend_state==UP; CHOCH_BEAR must have DOWN."""
        bc = bos_choch_detect(btc_1h)
        bull_rows = bc[bc["structure_event"] == "CHOCH_BULL"]
        bear_rows = bc[bc["structure_event"] == "CHOCH_BEAR"]
        if not bull_rows.empty:
            assert (bull_rows["trend_state"] == "UP").all(), (
                "CHOCH_BULL rows with trend_state != UP"
            )
        if not bear_rows.empty:
            assert (bear_rows["trend_state"] == "DOWN").all(), (
                "CHOCH_BEAR rows with trend_state != DOWN"
            )


# ---------------------------------------------------------------------------
# TestLiquiditySweep
# ---------------------------------------------------------------------------


class TestLiquiditySweep:
    """Tests for liquidity_sweep_detect."""

    def test_returns_dataframe(self, btc_1h):
        ls = liquidity_sweep_detect(btc_1h)
        assert isinstance(ls, pd.DataFrame)
        assert {"sweep_direction", "sweep_level", "sweep_wick_pct", "sweep_strength"}.issubset(
            ls.columns
        )
        assert len(ls) == len(btc_1h)

    def test_sweep_wick_consistency(self, btc_1h):
        """BEAR sweep: bar high must exceed sweep_level; BULL sweep: bar low must be below it."""
        ls = liquidity_sweep_detect(btc_1h)
        high = btc_1h["high"]
        low = btc_1h["low"]

        bear = ls[ls["sweep_direction"] == "BEAR"]
        if not bear.empty:
            assert (high.loc[bear.index] > bear["sweep_level"]).all(), (
                "BEAR sweep found where high <= sweep_level"
            )

        bull = ls[ls["sweep_direction"] == "BULL"]
        if not bull.empty:
            assert (low.loc[bull.index] < bull["sweep_level"]).all(), (
                "BULL sweep found where low >= sweep_level"
            )

    def test_count_reasonable_btc_1h(self, btc_1h):
        """Sweep count on 500 bars should be between 10 and 80."""
        count = liquidity_sweep_detect(btc_1h)["sweep_direction"].notna().sum()
        assert 10 <= count <= 80, f"Sweep count={count} outside [10, 80]"


# ---------------------------------------------------------------------------
# TestEqualHighsLows
# ---------------------------------------------------------------------------


class TestEqualHighsLows:
    """Tests for equal_highs_lows."""

    def test_returns_dataframe(self, btc_1h):
        eq = equal_highs_lows(btc_1h)
        assert isinstance(eq, pd.DataFrame)
        assert {"eqh_level", "eqh_count", "eql_level", "eql_count"}.issubset(eq.columns)
        assert len(eq) == len(btc_1h)

    def test_count_zero_or_at_least_2(self, btc_1h):
        """_best_cluster only emits count 0 (no cluster) or >= 2 (by definition)."""
        eq = equal_highs_lows(btc_1h)
        eqh = eq["eqh_count"]
        eql = eq["eql_count"]
        assert ((eqh == 0) | (eqh >= 2)).all(), (
            "eqh_count has values of 1, which violates the equal-levels definition"
        )
        assert ((eql == 0) | (eql >= 2)).all(), (
            "eql_count has values of 1, which violates the equal-levels definition"
        )

    def test_levels_consistency(self, btc_1h):
        """When eqh_count > 0 the corresponding eqh_level must not be NaN."""
        eq = equal_highs_lows(btc_1h)
        active_high = eq[eq["eqh_count"] > 0]
        if not active_high.empty:
            assert active_high["eqh_level"].notna().all(), (
                "eqh_level is NaN despite eqh_count > 0"
            )
        active_low = eq[eq["eql_count"] > 0]
        if not active_low.empty:
            assert active_low["eql_level"].notna().all(), (
                "eql_level is NaN despite eql_count > 0"
            )


# ---------------------------------------------------------------------------
# TestPremiumDiscountZone
# ---------------------------------------------------------------------------


class TestPremiumDiscountZone:
    """Tests for premium_discount_zone."""

    def test_returns_dataframe(self, btc_1h):
        pdz = premium_discount_zone(btc_1h)
        assert isinstance(pdz, pd.DataFrame)
        assert {
            "premium_top",
            "discount_bottom",
            "equilibrium",
            "current_zone",
        }.issubset(pdz.columns)
        assert len(pdz) == len(btc_1h)

    def test_zone_in_set(self, btc_1h):
        """current_zone must be PREMIUM, DISCOUNT, EQUILIBRIUM, or NaN."""
        pdz = premium_discount_zone(btc_1h)
        valid = {"PREMIUM", "DISCOUNT", "EQUILIBRIUM"}
        actual = set(pdz["current_zone"].dropna().unique())
        assert actual.issubset(valid), f"Unexpected zone labels: {actual - valid}"

    def test_premium_above_discount(self, btc_1h):
        """When both swing endpoints are known, premium_top must exceed discount_bottom."""
        pdz = premium_discount_zone(btc_1h)
        mask = pdz["premium_top"].notna() & pdz["discount_bottom"].notna()
        subset = pdz[mask]
        if not subset.empty:
            assert (subset["premium_top"] > subset["discount_bottom"]).all(), (
                "Found rows where premium_top <= discount_bottom"
            )

    def test_equilibrium_is_midpoint(self, btc_1h):
        """equilibrium must equal (premium_top + discount_bottom) / 2 exactly."""
        pdz = premium_discount_zone(btc_1h)
        mask = pdz["premium_top"].notna() & pdz["discount_bottom"].notna()
        subset = pdz[mask]
        if not subset.empty:
            expected = (subset["premium_top"] + subset["discount_bottom"]) / 2.0
            np.testing.assert_allclose(
                subset["equilibrium"].values,
                expected.values,
                rtol=1e-10,
                err_msg="equilibrium is not the midpoint of [discount_bottom, premium_top]",
            )


# ---------------------------------------------------------------------------
# TestSMCIntegration
# ---------------------------------------------------------------------------

_ALL_SMC_FUNCTIONS = [
    fractal_pivot_high,
    fractal_pivot_low,
    fair_value_gap,
    order_block_detect,
    bos_choch_detect,
    liquidity_sweep_detect,
    equal_highs_lows,
    premium_discount_zone,
]


def _run_full_pipeline(df: pd.DataFrame, expected_len: int) -> None:
    """Run all 8 SMC functions and assert correct type + length for each result."""
    for fn in _ALL_SMC_FUNCTIONS:
        result = fn(df)
        assert isinstance(result, (pd.DataFrame, pd.Series)), (
            f"{fn.__name__} returned {type(result).__name__}; "
            "expected pd.DataFrame or pd.Series"
        )
        assert len(result) == expected_len, (
            f"{fn.__name__} length={len(result)}, expected {expected_len}"
        )


class TestSMCIntegration:
    """End-to-end pipeline tests across all four market fixtures."""

    def test_full_pipeline_btc_1h(self, btc_1h):
        """All 8 functions run without error on BTCUSDT 1h (500 bars)."""
        _run_full_pipeline(btc_1h, 500)

    def test_full_pipeline_eth_1h(self, eth_1h):
        """All 8 functions run without error on ETHUSDT 1h (500 bars)."""
        _run_full_pipeline(eth_1h, 500)

    def test_full_pipeline_btc_4h(self, btc_4h):
        """All 8 functions run without error on BTCUSDT 4h (300 bars)."""
        _run_full_pipeline(btc_4h, 300)

    def test_full_pipeline_btc_15m(self, btc_15m):
        """All 8 functions run without error on BTCUSDT 15m (500 bars)."""
        _run_full_pipeline(btc_15m, 500)
