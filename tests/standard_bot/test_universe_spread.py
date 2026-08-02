"""Real liquidity: the top of the book, not yesterday's turnover.

The gap this closes
-------------------
"剔除低流動性的空氣幣" is the single most common line in the selection requests,
and the repo could only express it as ``universe.filter_quote_volume``. That is a
PROXY, and it fails in the direction that costs money: turnover says how much
traded over 24 h, liquidity says whether an order can be filled now. On the
committed capture (2026-08-02):

======================  ==============  ==========  ===================
instrument              24h turnover    spread      size at the touch
======================  ==============  ==========  ===================
SNDKUSDT                $618m            0.083 bps  $839
SNXXUSDT                $103m           11.03 bps   $18,475
MMTUSDT                 $113m            5.23 bps   $72
IDOLUSDT                $165m            4.19 bps   $6.01
======================  ==============  ==========  ===================

The last three all clear the ``min_quote_volume: 100000000`` floor that
``example_selection.yaml`` ships with, and nothing in the emitted basket says the
order will not fill. The venue's median spread is 5.5 bps and 395 of 727
instruments are above 5 bps, so this is the ordinary case rather than a tail.

Three properties get tests below because each of them fails silently if removed:

* **``top_of_book_usd`` is the MIN of the two sides, not the mean.** A book with
  $500k of bids and $30 of asks is a $30 book for anything that has to get out.
* **A book that cannot be quoted is NaN, never a number.** One-sided, crossed and
  locked books give an infinite, a negative and an exactly-0.0 bps spread
  respectively, and all three CLEAR a ``max_spread_bps`` ceiling — so the worst
  books in the market would sort as the tightest.
* **A resting size of zero is NOT missing.** ``top_of_book_usd == 0.0`` is a true
  statement about the book and a floor is right to drop it for being small; only
  the PRICE has to be usable, because that is what the ratio divides by.

Why a third fixture file
------------------------
``bookTicker`` is whole-market in one request, so unlike the derivatives it is
free — but it is the fastest-moving quantity in the catalog, and
``freeze_selection_fixture.ADDABLE_NODES`` is right to refuse back-filling it into
a four-day-old bundle. Re-capturing either existing fixture would move golden
baskets in three other modules for a reason unrelated to this change. So the book
went into its own bundle alongside the funding schedule it was collected with —
see ``test_funding_apr.py`` for the other half — and
``test_the_earlier_fixtures_did_not_move`` asserts the other two are untouched.
"""

from __future__ import annotations

import json
import urllib.request
import warnings
from pathlib import Path

import pandas as pd
import pytest

from cyqnt_trd.blocks import data as blocks_data
from cyqnt_trd.blocks import universe as ub
from cyqnt_trd.data_cli import _subprocess as data_cli_subprocess
from cyqnt_trd.standard_bot.data.catalog import Availability, FrameKind, get_node
from cyqnt_trd.standard_bot.data.live_snapshot import (
    FAN_OUT_SECTIONS,
    SECTION_NODES,
    requests_for_sections,
)
from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import (
    live_sections_for_spec,
    required_bundle_nodes,
    run_bundle,
)
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import FETCHES_WITHOUT_SOURCE
from cyqnt_trd.standard_bot.yaml_pipeline.spec import (
    _synthetic_book_ticker,
    _synthetic_universe,
    load_spec,
    validate_spec,
)

REPO = Path(__file__).parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "universe_liquidity.json"
CROSS_SECTION_FIXTURE = FIXTURES / "universe_cross_section.json"
DERIVATIVES_FIXTURE = FIXTURES / "universe_derivatives.json"
SPEC = REPO / "docs" / "strategy_yaml_spec" / "example_liquidity_screen.yaml"

FROZEN_FRAMES = {"universe", "funding", "funding_info", "book_ticker"}

NODE = "book_ticker"

#: The spec's own thresholds, named once so the assertions cannot drift from it.
MIN_TURNOVER = 20_000_000.0
MAX_SPREAD_BPS = 5.0
MIN_TOP_OF_BOOK_USD = 10_000.0

# ---------------------------------------------------------------------------
# The golden basket. It lives HERE, beside the assertion, because a reviewer has
# to be able to see WHICH instruments the screen keeps without opening the spec.
#
# example_liquidity_screen.yaml on the committed capture: turnover >= $20m, then
# spread <= 5 bps, then >= $10k at the touch, ranked by depth. Seven of the
# declared top_k=10 slots fill — top_k is a ceiling, not a quota.
LIQUIDITY_BASKET = [
    (1, "BTCUSDT"),
    (2, "SOLUSDT"),
    (3, "ETHUSDT"),
    (4, "DOGEUSDT"),
    (5, "XRPUSDT"),
    (6, "XAUUSDT"),
    (7, "BNBUSDT"),
]

#: Instruments on the committed capture that clear a $100m TURNOVER floor and are
#: still untradable, with the number that says so. This is the whole argument for
#: the two columns, so it is pinned rather than described.
#:
#: ``spread_bps`` / ``top_of_book_usd``, measured. MMTUSDT's spread is entirely
#: respectable, which is why the depth column is not optional.
TURNOVER_LIARS = {
    "SNXXUSDT": (11.03, 18_474.70),
    "MMTUSDT": (5.23, 72.35),
    "IDOLUSDT": (4.19, 6.01),
}


def _fixture() -> dict:
    """A fresh parse per call — a test must not inherit another's mutations."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _frame(name: str, bundle: dict | None = None) -> pd.DataFrame:
    bundle = bundle if bundle is not None else _fixture()
    return pd.DataFrame((bundle["frames"][name].get("rows") or []))


def _book(**overrides) -> pd.DataFrame:
    """A three-row vendor-shaped book, tuned so every number is checkable by eye.

    ETHUSDT is the ASYMMETRIC one: 10 coins bid at 100 ($1,000) against 1 coin
    offered at 101 ($101). That asymmetry is what tells ``min`` from ``mean``.
    """
    frame = pd.DataFrame({
        "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "bidPrice": [99.99, 100.0, 50.0],
        "bidQty": [10.0, 10.0, 4.0],
        "askPrice": [100.01, 101.0, 50.005],
        "askQty": [20.0, 1.0, 100.0],
        "time": [1_700_000_000_000] * 3,
    })
    for column, values in overrides.items():
        frame[column] = values
    return frame


def _universe(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT")) -> pd.DataFrame:
    return pd.DataFrame({
        "instrument_id": list(symbols),
        # Descending, so a turnover tie-break is deterministic; the values are
        # not otherwise read.
        "quoteVolume": [9e8 - index * 1e7 for index in range(len(symbols))],
        "available_time": [1_700_000_000_000] * len(symbols),
    })


def _joined(book=None, universe=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ub.augment_with_spread(
            _universe() if universe is None else universe,
            _book() if book is None else book)


def _row(frame: pd.DataFrame, symbol: str):
    return frame.loc[frame["symbol"] == symbol].iloc[0]


# --------------------------------------------------------------------------- #
# the arithmetic                                                              #
# --------------------------------------------------------------------------- #


def test_the_spread_is_basis_points_of_the_mid():
    """bps and not a fraction, because the same threshold has three spellings.

    0.0005 / 0.05 % / 5 bps are all "five basis points" in the wild and two of
    them are 100x off. The column carries the unit an execution desk quotes.
    """
    joined = _joined()

    # 100.01 - 99.99 = 0.02 over a mid of 100.00 -> 2 bps exactly.
    assert _row(joined, "BTCUSDT")["spread_bps"] == pytest.approx(2.0)
    # 101 - 100 = 1 over a mid of 100.5 -> 99.5 bps.
    assert _row(joined, "ETHUSDT")["spread_bps"] == pytest.approx(
        1.0 / 100.5 * 10_000.0)


def test_the_size_at_the_touch_is_the_smaller_side_not_the_average():
    """A basket is entered AND exited, so the harder direction is the answer.

    ETHUSDT has $1,000 of bids against $101 of offers. ``min`` says $101; the
    mean would say $550 and would be describing a position nobody can close, and
    ``max`` would say $1,000 and would be describing the easy half only.
    """
    joined = _joined()
    eth = _row(joined, "ETHUSDT")

    assert eth["top_of_book_usd"] == pytest.approx(101.0)
    assert eth["top_of_book_usd"] != pytest.approx((1000.0 + 101.0) / 2.0)
    # And the symmetric-ish instrument takes its own smaller side: 10 x 99.99
    # against 20 x 100.01.
    assert _row(joined, "BTCUSDT")["top_of_book_usd"] == pytest.approx(999.9)


def test_a_resting_size_of_zero_is_a_real_zero_and_not_a_missing_reading():
    """"Nothing at the touch" is a fact; the floor is right to drop it for size.

    The mirror image of the NaN tests below, and the distinction matters both
    ways: only the PRICE has to be usable, because that is what the ratio divides
    by. Turning a zero quantity into NaN would make a genuinely empty book
    indistinguishable from a failed read, and it is the empty book that a
    ``min_top_of_book_usd`` floor exists to catch.
    """
    joined = _joined(_book(askQty=[20.0, 0.0, 100.0]))
    eth = _row(joined, "ETHUSDT")

    assert eth["top_of_book_usd"] == 0.0
    assert pd.notna(eth["top_of_book_usd"])
    # The spread is unaffected: both prices are still real.
    assert eth["spread_bps"] == pytest.approx(1.0 / 100.5 * 10_000.0)

    kept = ub.filter_top_of_book(joined, min_top_of_book_usd=1.0)
    assert "ETHUSDT" not in set(kept["symbol"])


# --------------------------------------------------------------------------- #
# the books that cannot be quoted                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label, overrides, arithmetic_value", [
    # bid absent: mid becomes 50.005 and the "spread" is the whole ask side.
    ("one_sided", {"bidPrice": [99.99, 0.0, 50.0]}, "huge"),
    ("unpriced_ask", {"askPrice": [100.01, float("nan"), 50.05]}, "nan"),
    # crossed: bid above ask. Arithmetically NEGATIVE, so it clears every ceiling.
    ("crossed", {"bidPrice": [99.99, 102.0, 50.0]}, "negative"),
    # locked: bid == ask. Arithmetically 0.0 bps, the tightest possible number.
    ("locked", {"bidPrice": [99.99, 101.0, 50.0]}, "zero"),
])
def test_an_unquotable_book_is_nan_and_never_a_number(
    label, overrides, arithmetic_value
):
    """The reason this block exists rather than a subtraction in the spec.

    Each of these four shapes produces a spread that PASSES ``max_spread_bps``:
    a one-sided book's is enormous but its mid is halved so it can land anywhere,
    a crossed one's is negative and a locked one's is exactly 0.0. Under a plain
    subtraction the least tradable instruments on the venue sort as the most
    tradable, and the basket gives no sign of it.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joined = ub.augment_with_spread(_universe(), _book(**overrides))

    eth = _row(joined, "ETHUSDT")
    assert pd.isna(eth["spread_bps"]), (label, eth["spread_bps"])
    # The depth goes with it: a price that cannot be trusted cannot value a size.
    assert pd.isna(eth["top_of_book_usd"]), label
    # The other two instruments are untouched — one bad row does not void the frame.
    assert pd.notna(_row(joined, "BTCUSDT")["spread_bps"]), label
    assert pd.notna(_row(joined, "SOLUSDT")["spread_bps"]), label

    messages = [str(entry.message) for entry in caught
                if issubclass(entry.category, RuntimeWarning)]
    assert any("ETHUSDT" in message for message in messages), (label, messages)
    assert any("max_spread_bps" in message for message in messages), label


def test_a_locked_book_would_otherwise_be_the_tightest_market_on_the_venue():
    """The specific case a reviewer is most likely to think is harmless.

    ``bid == ask`` is 0.0 bps, which is not a tight market — it is two sides read
    at different instants, or an instrument that is halted. 0.0 beats every real
    spread, so it takes rank 1 in a ``score: spread_bps, order: asc`` basket.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        joined = ub.augment_with_spread(
            _universe(), _book(bidPrice=[99.99, 101.0, 50.0]))

    # Rank 1 is the tightest REAL book in the frame (SOLUSDT, 1.0 bps) and not the
    # locked one, which arithmetic would have put ahead of every instrument that
    # actually has a market.
    ranked = joined.sort_values("spread_bps").dropna(subset=["spread_bps"])
    assert list(ranked["symbol"]) == ["SOLUSDT", "BTCUSDT"]
    kept = ub.filter_spread(joined, max_spread_bps=1000.0)
    assert "ETHUSDT" not in set(kept["symbol"])


def test_the_filter_says_how_many_it_dropped_for_being_unknown():
    """"We could not price this" is not "this is expensive to trade"."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joined = ub.augment_with_spread(
            _universe(), _book(bidPrice=[99.99, 0.0, 50.0]))
        kept = ub.filter_spread(joined, max_spread_bps=1000.0)

    assert set(kept["symbol"]) == {"BTCUSDT", "SOLUSDT"}
    assert any("1 of 3" in str(entry.message) and "spread_bps" in str(entry.message)
               for entry in caught), [str(e.message) for e in caught]


# --------------------------------------------------------------------------- #
# the filters                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("call, bounds", [
    (ub.filter_spread, "max_spread_bps, min_spread_bps"),
    (ub.filter_top_of_book, "min_top_of_book_usd, max_top_of_book_usd"),
])
def test_a_filter_with_no_bound_is_refused_rather_than_a_no_op(call, bounds):
    """A step that returns the frame unchanged is indistinguishable from a step
    that is not there, which is the whole reason it raises."""
    with pytest.raises(ValueError, match="needs at least one bound"):
        call(_joined())
    assert bounds  # the message names them; see _bounded_filter


@pytest.mark.parametrize("call, kwargs", [
    (ub.filter_spread, {"max_spread_bps": 5.0}),
    (ub.filter_top_of_book, {"min_top_of_book_usd": 1.0}),
])
def test_a_filter_without_the_augment_step_names_the_step_to_add(call, kwargs):
    with pytest.raises(ValueError, match="augment_with_spread"):
        call(_universe(), **kwargs)


def test_the_spread_filter_bounds_both_ends():
    joined = _joined()

    tight = ub.filter_spread(joined, max_spread_bps=5.0)
    assert set(tight["symbol"]) == {"BTCUSDT", "SOLUSDT"}
    # min_spread_bps is not padding: a market-making screen wants the wide ones.
    wide = ub.filter_spread(joined, min_spread_bps=5.0)
    assert set(wide["symbol"]) == {"ETHUSDT"}


def test_a_filter_hands_the_frame_back_in_the_vocabulary_it_arrived_in():
    """The augment widens with two columns; the filters must not widen at all."""
    joined = _joined()
    kept = ub.filter_spread(joined, max_spread_bps=1_000.0)

    assert list(kept.columns) == list(joined.columns)
    assert len(kept) == len(joined)


# --------------------------------------------------------------------------- #
# the two source vocabularies                                                 #
# --------------------------------------------------------------------------- #


def test_the_vendor_and_canonical_spellings_give_the_same_answer():
    """A direct fetcher call is camelCase; a bundle frame is snake_case.

    The node's ``column_map`` renames on the way into a bundle, so a block that
    understood only one of the two could be called from Python or replayed from a
    bundle but not both.
    """
    vendor = _book()
    canonical = vendor.rename(columns={
        "symbol": "instrument_id", "bidPrice": "bid_price", "bidQty": "bid_qty",
        "askPrice": "ask_price", "askQty": "ask_qty", "time": "event_time"})

    from_vendor = _joined(vendor)
    from_canonical = _joined(canonical)

    pd.testing.assert_frame_equal(from_vendor, from_canonical)


def test_the_canonical_names_are_exactly_what_the_node_promises():
    """Asserted against the catalog rather than a hand-copied list.

    The alias table and the ``column_map`` are two statements of the same fact in
    two files; without this the node could be renamed and the block would keep
    accepting only the vendor spelling, which fails at replay and nowhere else.
    """
    mapped = set(get_node(NODE).column_map.values())
    accepted = {alias for aliases in ub._BOOK_TICKER_COLUMNS.values()
                for alias in aliases}

    assert {"bid_price", "bid_qty", "ask_price", "ask_qty"} <= accepted
    assert {"bid_price", "bid_qty", "ask_price", "ask_qty"} <= mapped
    # instrument_id / event_time are handled by the shared key + PIT machinery.
    assert mapped - accepted == {"instrument_id", "event_time"}


# --------------------------------------------------------------------------- #
# fails closed                                                                #
# --------------------------------------------------------------------------- #


def test_an_empty_book_frame_is_refused_and_not_nan_filled():
    """取不到 != 取到是空的. An all-NaN column makes a failed capture look like a
    strict screen: every threshold returns an empty basket."""
    with pytest.raises(ValueError, match="source is empty"):
        ub.augment_with_spread(_universe(), pd.DataFrame())


def test_the_empty_message_blames_the_single_request_not_a_fan_out():
    """The next step differs, so the message must not borrow the other one.

    A fan-out that collected nothing is a roster / rate-budget question. This read
    is one request for the whole market, so there is no roster and no
    per-instrument hole to investigate — there is one request to retry.
    """
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_spread(_universe(), pd.DataFrame())

    message = str(excinfo.value)
    assert "WHOLE market in ONE request" in message
    assert "no all-market endpoint" not in message


def test_a_missing_price_column_is_refused_rather_than_becoming_nan():
    book = _book().drop(columns=["bidQty"])
    with pytest.raises(ValueError, match="bid_qty"):
        ub.augment_with_spread(_universe(), book)


def test_a_partial_book_raises_and_does_not_advise_reordering_the_steps():
    """The coverage floor, and the advice it must NOT give.

    ``augment_with_open_interest`` tells a caller to move the step after the
    narrowing steps, because its capture fanned out over a narrowed roster. This
    read is whole-market, so step order is irrelevant and that advice would send
    someone to rewrite a correct pipeline. The real causes are a frame from
    another venue or another snapshot.
    """
    universe = _universe(("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
                          "DOGEUSDT", "ADAUSDT"))
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_spread(universe, _book())

    message = str(excinfo.value)
    assert "3 of 6" in message
    assert "DIFFERENT venue" in message
    assert "narrow" not in message


def test_the_fetcher_refuses_spot_because_that_response_carries_no_timestamp():
    """Not "spot has no book" — it has one, and it arrives without ``time``.

    A spread with no timestamp cannot be PIT-gated or checked for staleness, and
    stamping it with "now" would fabricate the one fact that makes a snapshot
    replayable. The message has to say that, or someone adds the URL and a
    silently-undated frame enters the bundle.
    """
    with pytest.raises(ValueError) as excinfo:
        blocks_data.fetch_book_ticker_cross_section("spot")

    message = str(excinfo.value)
    assert "time" in message
    assert "PIT-gated" in message or "staleness" in message


# --------------------------------------------------------------------------- #
# the catalog node                                                            #
# --------------------------------------------------------------------------- #


def test_the_node_is_forward_only_and_says_why_it_cannot_be_replayed():
    node = get_node(NODE)

    assert node.emits is FrameKind.RANK
    assert node.availability is Availability.FORWARD_ONLY
    assert node.pit_hazard.strip(), "a FORWARD_ONLY node must state its hazard"
    assert "history" in node.pit_hazard
    assert node.fetcher == "cyqnt_trd.blocks.data.fetch_book_ticker_cross_section"


def test_the_node_is_not_a_fan_out_and_takes_no_roster():
    """One request for 727 instruments, so it must not be in the expensive table.

    Being listed in ``FAN_OUT_SECTIONS`` would make every caller supply a
    ``fan_out_symbols`` roster it does not need, and would make the joining block
    demand that the step come after the narrowing steps.
    """
    assert NODE not in {node for node, _key, _extra in FAN_OUT_SECTIONS.values()}
    assert SECTION_NODES["selection_book_ticker"] == (NODE,)
    assert [param.key for param in get_node(NODE).params] == ["market_type"]


# --------------------------------------------------------------------------- #
# YAML wiring                                                                 #
# --------------------------------------------------------------------------- #


def test_the_block_may_not_fetch_its_own_source_from_a_spec():
    """Without ``with:``, validate would fire a live request on a spec a frontend
    handed us, and a backtest would read today's book."""
    assert FETCHES_WITHOUT_SOURCE["universe.augment_with_spread"] == NODE

    spec = load_spec(str(SPEC))
    for step in spec["selection"]["universe"]:
        if step["block"] == "universe.augment_with_spread":
            step.pop("with")
    errors, _warnings = validate_spec(spec)

    assert any("with: [book_ticker]" in error for error in errors), errors


def test_the_spec_declares_the_section_the_live_collector_would_read():
    """``with: [book_ticker]`` -> the section -> the request, end to end.

    Derived from the spec rather than restated, so a rename cannot leave the spec
    validating while the live plan collects nothing.
    """
    spec = load_spec(str(SPEC))

    assert NODE in required_bundle_nodes(spec)
    assert "selection_book_ticker" in live_sections_for_spec(spec)

    plan = requests_for_sections(live_sections_for_spec(spec))
    assert (NODE, {"market_type": "futures"}, NODE) in plan
    # And nothing in this plan needs a fan-out roster, which is why the call above
    # does not pass one.
    assert not (set(live_sections_for_spec(spec)) & set(FAN_OUT_SECTIONS))


def test_validate_never_touches_the_network(monkeypatch):
    def deny(*_args, **_kwargs):
        raise AssertionError("validate must use the synthetic frame")

    monkeypatch.setattr(blocks_data, "_request_json", deny)
    monkeypatch.setattr(data_cli_subprocess, "_run", deny)
    monkeypatch.setattr(urllib.request, "urlopen", deny)

    errors, _warnings = validate_spec(load_spec(str(SPEC)))
    assert errors == []


# --------------------------------------------------------------------------- #
# the dry-run stand-in                                                        #
# --------------------------------------------------------------------------- #


def test_the_stand_in_book_offers_exactly_the_columns_the_real_source_has():
    """``validate`` must be neither more permissive nor more restrictive than
    ``run``.

    Set equality in both directions against the FETCHER's own output shape: an
    extra column here buys a spec a green validate followed by a hard failure on
    every real bundle, and a missing one makes a correct spec unvalidatable.
    """
    stand_in = set(_synthetic_book_ticker(_synthetic_universe()).columns)

    assert stand_in == set(blocks_data._BOOK_TICKER_FIELDS)


def test_the_stand_in_book_covers_the_whole_stand_in_universe():
    """Unlike the fan-out stand-ins, and for a reason.

    A partial book is not a state the real source can be in — it answers for
    every instrument in one request — so a stand-in with holes would fail specs
    for something production cannot reproduce.
    """
    universe = _synthetic_universe()
    stand_in = _synthetic_book_ticker(universe)

    assert set(stand_in["symbol"]) == set(universe["instrument_id"])


def test_the_stand_in_book_puts_rows_on_both_sides_of_both_thresholds():
    """Otherwise a dry-run proves only "it did not raise".

    Both screens have to narrow AND leave rows, and they have to DISAGREE about
    which instruments they keep — that disagreement is why ``filter_spread`` and
    ``filter_top_of_book`` are two blocks, and a stand-in where they agreed would
    let a spec screening the wrong one validate as if it screened the right one.
    """
    universe = _synthetic_universe()
    joined = ub.augment_with_spread(universe, _synthetic_book_ticker(universe))

    tight = set(ub.filter_spread(joined, max_spread_bps=MAX_SPREAD_BPS)["symbol"])
    deep = set(ub.filter_top_of_book(
        joined, min_top_of_book_usd=MIN_TOP_OF_BOOK_USD)["symbol"])

    for label, kept in (("spread", tight), ("depth", deep)):
        assert 0 < len(kept) < len(joined), label
    assert tight - deep, "every tight book is also deep: the two screens agree"


def test_the_stand_in_book_is_well_formed_so_no_warning_fires_on_a_good_spec():
    """A defect baked into the stand-in is a warning about the STAND-IN.

    It would fire on every validate of every correct spec, dressed as a warning
    about the user's, on a channel whose value is that it is rare. The
    unquotable-book shapes are pinned above against hand-built frames instead.
    """
    universe = _synthetic_universe()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joined = ub.augment_with_spread(universe, _synthetic_book_ticker(universe))

    assert joined["spread_bps"].notna().all()
    assert [str(entry.message) for entry in caught
            if issubclass(entry.category, RuntimeWarning)] == []


# --------------------------------------------------------------------------- #
# the frozen capture                                                          #
# --------------------------------------------------------------------------- #


def test_the_fixture_is_a_market_only_whole_market_bundle():
    bundle = _fixture()

    assert bundle["schema"] == "cyqnt.input/v1"
    assert set(bundle["frames"]) == FROZEN_FRAMES
    # A status line for a frame that is not here would be unfalsifiable: it is
    # copied onto every emitted signal, where a reader cannot check it.
    assert set(bundle["source_status"]) == FROZEN_FRAMES
    # Committed to a public repo: market data only, never an account snapshot.
    assert bundle["positions"] == {}
    assert bundle["equity"] is None
    # Whole-market means whole-market: the book covers every quoted instrument,
    # which is the property that lets the augment step sit anywhere.
    universe = _frame("universe", bundle)
    assert len(universe) > 200
    assert set(_frame(NODE, bundle)["instrument_id"]) >= set(
        universe["instrument_id"])


def test_the_earlier_fixtures_did_not_move():
    """This stage added a fixture; it did not disturb the two already pinned.

    ``universe_cross_section.json`` carries the golden baskets of two other test
    modules and numbers quoted in three documents; ``universe_derivatives.json``
    carries the E5 funnel. Re-capturing either to make room for the book would
    have moved all of that for a reason unrelated to this change.
    """
    cross = json.loads(CROSS_SECTION_FIXTURE.read_text(encoding="utf-8"))
    derivatives = json.loads(DERIVATIVES_FIXTURE.read_text(encoding="utf-8"))

    assert cross["decision_time"] == 1_785_591_229_856
    assert set(cross["frames"]) == {"universe", "ticker_rank", "funding",
                                    "contract_meta"}
    assert len(cross["frames"]["universe"]["rows"]) == 727

    assert derivatives["decision_time"] == 1_785_608_174_524
    # ``universe_bars`` was ADDED to this bundle by a later stage, at its own
    # decision_time and without touching a byte of the five frames above — klines
    # take an ``endTime``, so that back-fill is checkable rather than asserted (see
    # ``freeze_selection_fixture.ADDABLE_NODES``). The point this test makes is
    # unchanged: nothing that carries a golden basket moved.
    assert set(derivatives["frames"]) == {
        "universe", "contract_meta", "open_interest_snapshot",
        "oi_change_snapshot", "long_short_ratio_snapshot", "universe_bars"}

    # Three captures, three instants — mixing them would be mixing markets.
    assert len({cross["decision_time"], derivatives["decision_time"],
                _fixture()["decision_time"]}) == 3


# --------------------------------------------------------------------------- #
# the acceptance test                                                         #
# --------------------------------------------------------------------------- #


def test_the_liquidity_spec_emits_its_basket_on_the_frozen_capture():
    """The stage's acceptance test: a spread-filtered spec, really run.

    Every step of the funnel is asserted rather than only the basket, because the
    basket could be right for the wrong reason — a coverage guard that silently
    dropped instruments would also shrink it, and the intermediate counts are what
    tell the two apart.
    """
    bundle = _fixture()
    output = run_bundle(str(SPEC), bundle)

    joined = ub.augment_with_spread(_frame("universe", bundle),
                                   _frame(NODE, bundle))
    liquid = ub.filter_quote_volume(joined, MIN_TURNOVER)
    tight = ub.filter_spread(liquid, max_spread_bps=MAX_SPREAD_BPS)
    deep = ub.filter_top_of_book(tight, min_top_of_book_usd=MIN_TOP_OF_BOOK_USD)
    assert (len(joined), len(liquid), len(tight)) == (727, 92, 75)
    assert len(deep) == 8

    signal = output["signals"][0]
    assert signal["kind"] == "selection"
    assert signal["universe_size"] == 727
    basket = [(item["rank"], item["symbol"]) for item in signal["candidates"]]
    assert basket == LIQUIDITY_BASKET

    # 8 rows survive the filters and 7 candidates come back: dedupe_by:
    # base_asset collapsed one token's second quote pair, so the basket is
    # top_k DISTINCT bets rather than top_k rows.
    assert len(basket) == len(deep) - 1
    bases = [item.get("base_asset") or item["symbol"]
             for item in signal["candidates"]]
    assert len(set(bases)) == len(bases)


def test_the_dropped_quote_pair_was_the_thinner_one():
    """What ``dedupe_by: base_asset`` chose, and that it was not arbitrary.

    A token listed against both USDT and USDC carries a DIFFERENT book on each,
    so which pair survives is a real decision about where the order will fill.
    The interpreter breaks the tie on turnover, so the thin quote pair is the one
    that goes — asserted here because the basket above cannot show it.
    """
    bundle = _fixture()
    joined = ub.augment_with_spread(_frame("universe", bundle),
                                   _frame(NODE, bundle))
    deep = ub.filter_top_of_book(
        ub.filter_spread(ub.filter_quote_volume(joined, MIN_TURNOVER),
                         max_spread_bps=MAX_SPREAD_BPS),
        min_top_of_book_usd=MIN_TOP_OF_BOOK_USD)

    survivors = set(deep["symbol"])
    basket = {item[1] for item in LIQUIDITY_BASKET}
    dropped = survivors - basket
    assert len(dropped) == 1, survivors
    dropped_symbol = dropped.pop()
    assert dropped_symbol.endswith("USDC"), dropped_symbol

    kept = dropped_symbol.replace("USDC", "USDT")
    assert kept in basket
    kept_row, dropped_row = _row(deep, kept), _row(deep, dropped_symbol)
    assert kept_row["quoteVolume"] > dropped_row["quoteVolume"]


def test_turnover_alone_keeps_instruments_that_cannot_be_entered():
    """The reason the two columns exist, on captured market data.

    Each of these clears the $100m turnover floor ``example_selection.yaml``
    ships with. MMTUSDT's spread is entirely respectable, which is why the depth
    column is not optional: 5.23 bps and $72 at the touch.
    """
    bundle = _fixture()
    joined = ub.augment_with_spread(_frame("universe", bundle),
                                   _frame(NODE, bundle))
    by_turnover = ub.filter_quote_volume(joined, 1e8)

    assert set(TURNOVER_LIARS) <= set(by_turnover["symbol"]), (
        "recapture moved these out of the $100m basket; the claim below is then "
        "vacuous and the table in this module's docstring is stale")
    for symbol, (spread_bps, top_of_book_usd) in TURNOVER_LIARS.items():
        row = _row(by_turnover, symbol)
        assert row["spread_bps"] == pytest.approx(spread_bps, abs=0.01), symbol
        assert row["top_of_book_usd"] == pytest.approx(
            top_of_book_usd, rel=1e-3), symbol

    # And the screen that DOES express the request removes all three.
    screened = ub.filter_top_of_book(
        ub.filter_spread(by_turnover, max_spread_bps=MAX_SPREAD_BPS),
        min_top_of_book_usd=MIN_TOP_OF_BOOK_USD)
    assert set(TURNOVER_LIARS).isdisjoint(set(screened["symbol"]))


def test_the_screen_is_not_vacuous_on_the_frozen_capture():
    """Both thresholds must split this market, or the acceptance test above
    passes by keeping everything."""
    bundle = _fixture()
    joined = ub.augment_with_spread(_frame("universe", bundle),
                                   _frame(NODE, bundle))
    liquid = ub.filter_quote_volume(joined, MIN_TURNOVER)

    for label, mask in (
        ("spread", liquid["spread_bps"] <= MAX_SPREAD_BPS),
        ("depth", liquid["top_of_book_usd"] >= MIN_TOP_OF_BOOK_USD),
    ):
        assert bool(mask.any()) and not bool(mask.all()), label


def test_replaying_the_liquidity_spec_touches_no_network(monkeypatch):
    calls = []

    def deny(name):
        def blocked(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError("replay must not fetch: %s" % name)
        return blocked

    monkeypatch.setattr(blocks_data, "_request_json", deny("blocks.data"))
    monkeypatch.setattr(data_cli_subprocess, "_run", deny("data_cli"))
    monkeypatch.setattr(urllib.request, "urlopen", deny("urlopen"))

    output = run_bundle(str(SPEC), _fixture())

    assert [(item["rank"], item["symbol"])
            for item in output["signals"][0]["candidates"]] == LIQUIDITY_BASKET
    assert calls == []


def test_replay_is_byte_identical():
    """A selection decision reads no clock, so the same frozen input must
    serialise to the same bytes — ``signal_id`` included."""
    first = run_bundle(str(SPEC), _fixture())
    second = run_bundle(str(SPEC), _fixture())

    assert first["signals"][0]["signal_id"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
