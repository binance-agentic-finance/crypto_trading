"""Regression tests for ``sizing.compute_stop_price``.

The bug: ``direction`` was compared with a bare ``== "LONG"``, so ``"long"`` fell through to the
short branch and returned ``entry + distance`` -- a stop ABOVE the entry price on a long trade.
Nothing raised and the number looks like a valid price, so the only way to notice was to eyeball
the output. It was the one place in ``cyqnt_trd.blocks`` still doing a case-sensitive compare;
``stop_loss.atr_dynamic_stop`` / ``fixed_pct_stop`` / ``limits`` all use ``.upper()``.

The function had no tests at all, which is why it survived. These pin the two properties that
matter: the stop lands on the correct side of entry for every spelling, and the two functions
that compute an ATR stop agree on the number.
"""
from __future__ import annotations

import pytest

from cyqnt_trd.blocks.sizing import compute_stop_price
from cyqnt_trd.blocks.stop_loss import atr_dynamic_stop

ENTRY = 50_000.0


@pytest.mark.parametrize("direction", ["LONG", "long", "Long"])
def test_long_stop_is_below_entry_whatever_the_spelling(direction):
    assert compute_stop_price(ENTRY, direction, stop_pct=3.0) == 48_500.0


@pytest.mark.parametrize("direction", ["SHORT", "short", "Short"])
def test_short_stop_is_above_entry_whatever_the_spelling(direction):
    assert compute_stop_price(ENTRY, direction, stop_pct=3.0) == 51_500.0


@pytest.mark.parametrize("direction", ["LONG", "long", "SHORT", "short"])
def test_atr_derived_stop_matches_the_other_atr_stop_helper(direction):
    """Two functions, one definition -- they must not drift apart again.

    ``compute_stop_price`` derives the distance as ``atr * multiplier / entry * 100`` percent and
    then applies it, so it must land exactly where ``atr_dynamic_stop`` puts it. Before the fix
    the lowercase cases disagreed by ``2 * atr * multiplier``.
    """
    atr, mult = 750.0, 2.0
    mine = compute_stop_price(ENTRY, direction, atr_val=atr, multiplier=mult)
    theirs = atr_dynamic_stop(ENTRY, direction, atr, multiplier=mult)["stop_price"]
    assert mine == pytest.approx(theirs, abs=1e-8)


def test_atr_stop_distance_is_the_atr_multiple():
    # long: 50000 - 2*750 = 48500 ; short: 50000 + 2*750 = 51500
    assert compute_stop_price(ENTRY, "long", atr_val=750.0, multiplier=2.0) == 48_500.0
    assert compute_stop_price(ENTRY, "short", atr_val=750.0, multiplier=2.0) == 51_500.0


def test_explicit_stop_pct_wins_over_atr():
    assert compute_stop_price(ENTRY, "LONG", stop_pct=1.0, atr_val=9_999.0) == 49_500.0


def test_without_stop_pct_or_atr_the_stop_collapses_onto_entry():
    """Documented fallback: ``stop_pct`` defaults to 0, so the stop IS the entry price.

    Worth pinning because it is a silent no-protection outcome, not an error -- a caller that
    forgets to pass either argument gets a stop that triggers immediately, and it is the caller's
    job to notice.
    """
    assert compute_stop_price(ENTRY, "LONG") == ENTRY
    assert compute_stop_price(ENTRY, "SHORT") == ENTRY
    # entry_price <= 0 takes the same path even when an ATR is supplied
    assert compute_stop_price(0.0, "LONG", atr_val=750.0) == 0.0
