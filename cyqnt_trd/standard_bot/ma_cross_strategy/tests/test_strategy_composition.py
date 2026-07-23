"""
Goal A: Verify blocks can compose all three strategies.

Tests:
  1. import + register (no errors)
  2. signal_fn identity via plugin registry
  3. output types are bool Series
  4. warm-up period produces all-False (no lookahead)
  5. golden cross fires at correct bar
  6. death cross fires at correct bar
  7. no simultaneous long+short signal
  8. bar_direction_validation: up bar → long, down bar → short
  9. minimum-data guard (< SLOW_PERIOD+1 rows returns False)

Design notes:
  - make_registry() drains the global _PENDING_REGISTRATIONS queue.
    All tests that need the registry use a single session-scoped fixture
    that imports the strategy modules (triggering register()) then calls
    make_registry() exactly once per test process.
  - Synthetic cross data must include a phase where fast_ma < slow_ma
    BEFORE the crossover so the (prev_fast < prev_slow) condition fires.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

# ── import all three strategies (side-effect: queues them for registration) ──
import strategies.ma_cross_v1 as _v1_mod
import strategies.ma_cross_validation_fast as _fast_mod
import strategies.bar_direction_validation as _bar_mod

from cyqnt_trd.standard_bot.entrypoints.common import make_registry


# ── session-scoped registry (flush happens exactly once per process) ──────────

@pytest.fixture(scope="session")
def registry():
    """
    Build one registry for the whole test session.
    make_registry() drains _PENDING_REGISTRATIONS; calling it multiple times
    would find an empty queue on the 2nd+ call. One shared instance is correct.
    """
    return make_registry()


# ── helpers ───────────────────────────────────────────────────────────────────

def _flat_df(n: int, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({"close": [price] * n})


def _df_with_golden_cross(fast: int, slow: int) -> pd.DataFrame:
    """
    Produce a series with a genuine golden cross.

    Phase 1 (indices 0..slow+fast): declining close → fast_ma < slow_ma
    Phase 2 (indices slow+fast+1..end): sharp rebound → fast_ma > slow_ma
    The cross (prev_fast < prev_slow AND fast_ma > slow_ma) must fire.
    """
    phase1 = slow + fast + 5
    phase2 = slow + fast + 30
    n = phase1 + phase2
    close = np.empty(n)
    # Phase 1: gentle decline so fast MA settles below slow MA
    for i in range(phase1):
        close[i] = 200.0 - i * 1.5
    # Phase 2: sharp rise so fast MA crosses above slow MA
    base = close[phase1 - 1]
    for i in range(phase2):
        close[phase1 + i] = base + (i + 1) * 4.0
    return pd.DataFrame({"close": close})


def _df_with_death_cross(fast: int, slow: int) -> pd.DataFrame:
    """Mirror of _df_with_golden_cross but reversed."""
    phase1 = slow + fast + 5
    phase2 = slow + fast + 30
    n = phase1 + phase2
    close = np.empty(n)
    # Phase 1: gentle rise so fast MA settles above slow MA
    for i in range(phase1):
        close[i] = 100.0 + i * 1.5
    # Phase 2: sharp drop so fast MA crosses below slow MA
    base = close[phase1 - 1]
    for i in range(phase2):
        close[phase1 + i] = base - (i + 1) * 4.0
    return pd.DataFrame({"close": close})


# ── Test 1: import + register ─────────────────────────────────────────────────

class TestImportAndRegister:
    def test_ma_cross_v1_registered(self, registry):
        plugin = registry.get("ma_cross_v1")
        assert plugin is not None, "ma_cross_v1 not in registry"

    def test_ma_cross_validation_fast_registered(self, registry):
        plugin = registry.get("ma_cross_validation_fast")
        assert plugin is not None, "ma_cross_validation_fast not in registry"

    def test_bar_direction_validation_registered(self, registry):
        plugin = registry.get("bar_direction_validation")
        assert plugin is not None, "bar_direction_validation not in registry"


# ── Test 2: signal_fn identity ────────────────────────────────────────────────

class TestSignalFnIdentity:
    def test_v1_fn_is_make_signals(self, registry):
        plugin = registry.get("ma_cross_v1")
        df = _flat_df(30)
        long_s, short_s = plugin.signal_fn(df)
        long_d, short_d = _v1_mod.make_signals(df)
        pd.testing.assert_series_equal(long_s, long_d)
        pd.testing.assert_series_equal(short_s, short_d)

    def test_fast_fn_is_make_signals(self, registry):
        plugin = registry.get("ma_cross_validation_fast")
        df = _flat_df(20)
        long_s, short_s = plugin.signal_fn(df)
        long_d, short_d = _fast_mod.make_signals(df)
        pd.testing.assert_series_equal(long_s, long_d)
        pd.testing.assert_series_equal(short_s, short_d)


# ── Test 3: output types ──────────────────────────────────────────────────────

class TestOutputTypes:
    @pytest.mark.parametrize("fn,fast,slow", [
        (_v1_mod.make_signals, _v1_mod.FAST_PERIOD, _v1_mod.SLOW_PERIOD),
        (_fast_mod.make_signals, _fast_mod.FAST_PERIOD, _fast_mod.SLOW_PERIOD),
    ])
    def test_returns_bool_series(self, fn, fast, slow):
        df = _flat_df(slow + 10)
        long_s, short_s = fn(df)
        assert isinstance(long_s, pd.Series), "long_signal must be pd.Series"
        assert isinstance(short_s, pd.Series), "short_signal must be pd.Series"
        assert long_s.dtype == bool, f"long dtype={long_s.dtype}"
        assert short_s.dtype == bool, f"short dtype={short_s.dtype}"

    def test_bar_dir_returns_bool_series(self):
        df = pd.DataFrame({"close": [100.0, 101.0, 99.0]})
        long_s, short_s = _bar_mod.make_signals(df)
        assert long_s.dtype == bool
        assert short_s.dtype == bool


# ── Test 4: warm-up period all-False ─────────────────────────────────────────

class TestNoLookahead:
    def test_v1_warmup_all_false(self):
        df = _flat_df(_v1_mod.SLOW_PERIOD)
        long_s, short_s = _v1_mod.make_signals(df)
        assert not long_s.any(), "long signal during warm-up"
        assert not short_s.any(), "short signal during warm-up"

    def test_fast_warmup_all_false(self):
        df = _flat_df(_fast_mod.SLOW_PERIOD)
        long_s, short_s = _fast_mod.make_signals(df)
        assert not long_s.any()
        assert not short_s.any()

    def test_minimum_data_guard_v1(self):
        df = _flat_df(3)
        long_s, short_s = _v1_mod.make_signals(df)
        assert not long_s.any()
        assert not short_s.any()


# ── Test 5+6: golden cross and death cross ────────────────────────────────────

class TestCrossSignals:
    def test_golden_cross_fires_v1(self):
        df = _df_with_golden_cross(_v1_mod.FAST_PERIOD, _v1_mod.SLOW_PERIOD)
        long_s, _ = _v1_mod.make_signals(df)
        assert long_s.any(), (
            f"expected golden cross signal. fast={_v1_mod.FAST_PERIOD} slow={_v1_mod.SLOW_PERIOD} "
            f"df_len={len(df)}"
        )

    def test_death_cross_fires_v1(self):
        df = _df_with_death_cross(_v1_mod.FAST_PERIOD, _v1_mod.SLOW_PERIOD)
        _, short_s = _v1_mod.make_signals(df)
        assert short_s.any(), "expected death cross signal"

    def test_golden_cross_fires_fast(self):
        df = _df_with_golden_cross(_fast_mod.FAST_PERIOD, _fast_mod.SLOW_PERIOD)
        long_s, _ = _fast_mod.make_signals(df)
        assert long_s.any()

    def test_death_cross_fires_fast(self):
        df = _df_with_death_cross(_fast_mod.FAST_PERIOD, _fast_mod.SLOW_PERIOD)
        _, short_s = _fast_mod.make_signals(df)
        assert short_s.any()


# ── Test 7: no simultaneous long+short ───────────────────────────────────────

class TestNoSimultaneousSignals:
    def test_v1_no_overlap(self):
        df = _df_with_golden_cross(_v1_mod.FAST_PERIOD, _v1_mod.SLOW_PERIOD)
        long_s, short_s = _v1_mod.make_signals(df)
        assert not (long_s & short_s).any(), "simultaneous long+short signal"

    def test_fast_no_overlap(self):
        df = _df_with_golden_cross(_fast_mod.FAST_PERIOD, _fast_mod.SLOW_PERIOD)
        long_s, short_s = _fast_mod.make_signals(df)
        assert not (long_s & short_s).any()


# ── Test 8: bar_direction_validation logic ────────────────────────────────────

class TestBarDirectionValidation:
    def test_up_bar_yields_long(self):
        df = pd.DataFrame({"close": [100.0, 100.0, 101.0]})
        long_s, short_s = _bar_mod.make_signals(df)
        assert long_s.iloc[-1] == True
        assert short_s.iloc[-1] == False

    def test_down_bar_yields_short(self):
        df = pd.DataFrame({"close": [100.0, 100.0, 99.0]})
        long_s, short_s = _bar_mod.make_signals(df)
        assert short_s.iloc[-1] == True
        assert long_s.iloc[-1] == False

    def test_equal_bar_yields_neither(self):
        df = pd.DataFrame({"close": [100.0, 100.0, 100.0]})
        long_s, short_s = _bar_mod.make_signals(df)
        assert long_s.iloc[-1] == False
        assert short_s.iloc[-1] == False

    def test_insufficient_data_returns_false(self):
        df = pd.DataFrame({"close": [100.0]})
        long_s, short_s = _bar_mod.make_signals(df)
        assert not long_s.any()
        assert not short_s.any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
