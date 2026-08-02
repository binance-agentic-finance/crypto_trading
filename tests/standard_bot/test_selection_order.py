"""Which END of the score column a basket is taken from.

``selection.score`` only ever ranked descending, so every screen that lives at
the bottom of a column — "the five most negative funding rates", "the biggest
losers" — had to be written as a detour: narrow the universe with a step that
happens to sort that way, then rank on a different column that happens to be
descending. The basket that came back was ranked by something other than what
the author asked for, and nothing in the output said so.

``order: asc`` states it directly. Two things about it are load-bearing and are
pinned here:

* the turnover tie-break inside ``dedupe_by: base_asset`` does NOT flip with the
  score (flipping it would silently pick the least tradable quote pair of every
  asset), and
* ``min_score`` / ``max_score`` are absolute bounds, so adding ``order:`` to an
  existing spec cannot invert a filter that spec already had.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (SpecError,
                                                              build_selection_fn)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import validate_spec

DT = 1_776_322_799_999

#: Funding straddling zero: the interesting end depends entirely on ``order``.
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
FUNDING_PCT = [-0.05, 0.01, -0.02, 0.03]
TURNOVER = [9e8, 7e8, 4e8, 3e8]


def _universe():
    return pd.DataFrame({"symbol": SYMBOLS,
                         "fundingRatePct": FUNDING_PCT,
                         "quoteVolume": TURNOVER,
                         "available_time": [DT] * len(SYMBOLS)})


def _spec(**selection_overrides):
    selection = {"score": "fundingRatePct", "top_k": 5}
    selection.update(selection_overrides)
    return {
        "strategy": {"id": "sel_order"},
        "run": {"mode": "backtest"},
        "data": {"symbol": "BTCUSDT", "primary": {"interval": "1h"}},
        "selection": selection,
    }


def _run(spec, frame=None):
    return build_selection_fn(spec)(_universe() if frame is None else frame)


def _symbols(spec, frame=None):
    return [candidate["symbol"] for candidate in _run(spec, frame)]


# --------------------------------------------------------------------------- #
# the two directions                                                          #
# --------------------------------------------------------------------------- #


def test_order_asc_ranks_from_the_bottom_of_the_column():
    """"The most negative funding" is rank 1, which is the whole point."""
    candidates = _run(_spec(order="asc"))
    assert [c["symbol"] for c in candidates] == ["BTCUSDT", "SOLUSDT",
                                                 "ETHUSDT", "BNBUSDT"]
    assert [c["score"] for c in candidates] == [-0.05, -0.02, 0.01, 0.03]
    assert [c["rank"] for c in candidates] == [1, 2, 3, 4]


def test_order_desc_is_the_default_and_the_pre_existing_behaviour():
    """Omitting ``order`` must not change what any existing spec returns."""
    explicit = _symbols(_spec(order="desc"))
    assert explicit == ["BNBUSDT", "ETHUSDT", "SOLUSDT", "BTCUSDT"]
    assert _symbols(_spec()) == explicit


def test_the_order_value_is_case_insensitive_like_dedupe_by():
    assert _symbols(_spec(order="ASC")) == _symbols(_spec(order="asc"))


def test_top_k_takes_the_head_of_the_requested_end():
    assert _symbols(_spec(order="asc", top_k=2)) == ["BTCUSDT", "SOLUSDT"]
    assert _symbols(_spec(order="desc", top_k=2)) == ["BNBUSDT", "ETHUSDT"]


def test_direction_screen_filters_before_dedupe_and_top_k():
    """A non-qualifying top score cannot consume a basket slot."""
    candidates = _run(_spec(
        order="desc",
        top_k=2,
        long_when={"cond": "conditions.value_below", "args": ["fundingRatePct", 0]},
    ))

    # BNB/ETH rank first by score but fail the requested long condition. The
    # actual top two *qualified* names are SOL and BTC, with contiguous ranks.
    assert [candidate["symbol"] for candidate in candidates] == ["SOLUSDT", "BTCUSDT"]
    assert [candidate["side"] for candidate in candidates] == ["long", "long"]
    assert [candidate["rank"] for candidate in candidates] == [1, 2]
    assert {candidate["scored_pool"] for candidate in candidates} == {2}


# --------------------------------------------------------------------------- #
# the tie-break must not flip with the score                                  #
# --------------------------------------------------------------------------- #


def _quote_pairs():
    """Two quote pairs per base, same score, very different turnover.

    ETH's *thin* leg is listed first and BTC's *deep* leg is listed first, so a
    dedupe that merely kept the first row would disagree with a dedupe that
    ranked on turnover — the assertions below can tell them apart.
    """
    return pd.DataFrame({
        "symbol": ["BTCUSDT", "BTCUSDC", "ETHUSDC", "ETHUSDT"],
        "fundingRatePct": [-0.05, -0.05, 0.02, 0.02],
        "quoteVolume": [9e8, 1e7, 5e6, 6e8],
        "available_time": [DT] * 4,
    })


@pytest.mark.parametrize("order,expected", [
    ("asc", ["BTCUSDT", "ETHUSDT"]),
    ("desc", ["ETHUSDT", "BTCUSDT"]),
])
def test_dedupe_keeps_the_deeper_quote_pair_in_both_orders(order, expected):
    """The regression that motivated splitting the sort keys.

    ``sort_values(["score", "__turnover__"], ascending=asc)`` would flip the
    tie-break along with the score, so ``order: asc`` would have returned
    BTCUSDC (1e7) and ETHUSDC (5e6) — the least tradable pair of each asset —
    while reporting the same score, rank and reason as the deep pair. Nothing
    downstream could have noticed.
    """
    spec = _spec(order=order, dedupe_by="base_asset")
    assert _symbols(spec, _quote_pairs()) == expected


def test_dedupe_by_base_asset_without_a_turnover_column_still_honours_asc():
    """The no-turnover fallback has no tie-break to protect, only a direction.

    Scores are distinct here on purpose: with no turnover column the order of
    tied rows is whatever pandas' sort does with them, and pinning that would
    be testing numpy rather than this module.
    """
    frame = _universe().drop(columns=["quoteVolume"])
    assert _symbols(_spec(order="asc"), frame) == ["BTCUSDT", "SOLUSDT",
                                                   "ETHUSDT", "BNBUSDT"]


def test_dedupe_none_honours_the_order_too():
    """``dedupe_by: none`` is a third sort site and was missed once already."""
    assert _symbols(_spec(order="asc", dedupe_by="none")) == [
        "BTCUSDT", "SOLUSDT", "ETHUSDT", "BNBUSDT"]
    assert _symbols(_spec(order="desc", dedupe_by="none")) == [
        "BNBUSDT", "ETHUSDT", "SOLUSDT", "BTCUSDT"]


# --------------------------------------------------------------------------- #
# min_score / max_score are absolute bounds                                   #
# --------------------------------------------------------------------------- #


def test_the_score_bounds_mean_the_same_thing_in_both_orders():
    """``min_score`` is a floor and ``max_score`` a ceiling, always.

    If ``min_score`` flipped to "the cutoff on the winning side" under
    ``order: asc`` then ``min_score: 1.0`` would mean two different things in
    two specs — and example_selection.yaml already ships ``min_score: 1.0``, so
    someone later adding ``order: asc`` to it would invert a filter they never
    touched.
    """
    bounded = {"min_score": -0.02, "max_score": 0.01}
    ascending = _run(_spec(order="asc", **bounded))
    descending = _run(_spec(order="desc", **bounded))

    # Both bounds sit exactly on a real score, which also pins inclusivity.
    assert [c["score"] for c in ascending] == [-0.02, 0.01]
    assert [c["score"] for c in descending] == [0.01, -0.02]
    assert {c["symbol"] for c in ascending} == {c["symbol"] for c in descending}


def test_max_score_alone_is_a_ceiling_under_desc():
    """Not a "winning side" cutoff: desc still starts from the biggest score
    that survives the ceiling."""
    assert _symbols(_spec(order="desc", max_score=-0.01)) == ["SOLUSDT", "BTCUSDT"]


def test_asc_plus_max_score_states_the_funding_screen_directly():
    """The motivating request: "funding more negative than -1bp", ranked most
    negative first — no proxy column, no narrowing step."""
    candidates = _run(_spec(order="asc", max_score=-0.01))
    assert [c["symbol"] for c in candidates] == ["BTCUSDT", "SOLUSDT"]
    assert all(c["score"] <= -0.01 for c in candidates)


def test_min_score_still_excludes_the_bottom_under_asc():
    assert _symbols(_spec(order="asc", min_score=0.0)) == ["ETHUSDT", "BNBUSDT"]


# --------------------------------------------------------------------------- #
# the basket says which end it came from                                      #
# --------------------------------------------------------------------------- #


def test_the_reason_states_the_ranking_direction():
    """"rank 1 of 4" reads identically both ways, so an operator holding a
    basket could not tell a short screen from a long one."""
    ascending = _run(_spec(order="asc"))[0]["reason"]
    descending = _run(_spec())[0]["reason"]

    assert "order=asc" in ascending and "lowest first" in ascending
    assert "order=desc" in descending and "highest first" in descending
    assert ascending != descending
    # The existing contract (tests/standard_bot/test_yaml_selection.py) is that
    # the reason carries "of <basket size>"; adding the direction must not move it.
    assert "rank 1 of 4" in ascending and "rank 1 of 4" in descending
    assert "fundingRatePct=-0.05" in ascending


# --------------------------------------------------------------------------- #
# validation                                                                  #
# --------------------------------------------------------------------------- #


def test_an_order_aware_spec_validates_and_dry_runs():
    spec = _spec(order="asc", score="priceChangePercent", max_score=-1.0)
    errors, _warnings = validate_spec(spec)
    assert errors == [], errors


def test_an_unknown_order_value_is_refused():
    with pytest.raises(SpecError, match="selection.order must be one of"):
        build_selection_fn(_spec(order="ascending"))

    errors, _warnings = validate_spec(_spec(order="ascending"))
    assert any("selection.order must be one of" in e for e in errors), errors


def test_min_score_above_max_score_is_refused_statically():
    """An inverted pair can only ever produce an empty basket, and an empty
    basket is otherwise a legitimate outcome — so the dry-run reports it as the
    generic "no candidates" warning and names nothing."""
    errors, _warnings = validate_spec(_spec(min_score=1.0, max_score=0.5))
    assert any("min_score (1) is above selection.max_score (0.5)" in e
               for e in errors), errors


def test_equal_bounds_are_allowed():
    """``min_score == max_score`` is an exact-value filter, not a mistake."""
    errors, _warnings = validate_spec(_spec(min_score=1.0, max_score=1.0))
    assert not any("max_score" in e for e in errors), errors


def test_a_non_numeric_bound_is_named():
    errors, _warnings = validate_spec(_spec(max_score="cheap"))
    assert any("selection.max_score must be numeric" in e for e in errors), errors


def test_order_and_max_score_are_known_selection_keys():
    """They have to be in SELECTION_KEYS or the closed-key check rejects them,
    which is how a new key silently does nothing."""
    errors, _warnings = validate_spec(_spec(order="asc", max_score=1e12))
    assert not any("unknown selection." in e for e in errors), errors
