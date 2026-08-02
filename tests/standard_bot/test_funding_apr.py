"""Funding, annualised: fixing a mis-ranking that was already happening.

The bug
-------
``lastFundingRate`` is the rate for **one settlement**, and Binance does not settle
every perpetual on the same schedule. Of 743 contracts (measured 2026-08-02) 443
settle every 4 hours, 296 every 8 and 4 every hour. So ``fundingRatePct`` is a
column whose UNIT differs from row to row, and ``score: fundingRatePct`` ranks
0.01%@1h (87.6 %/yr) level with 0.01%@8h (10.95 %/yr) — eight times apart.

Every "highest funding" and "most negative funding" basket in this repo was
therefore silently mis-ordered, and nothing in the output could reveal it: five
symbols, five plausible rates. On the committed capture the screen "the five most
negative funding rates" over instruments above $20m turnover gives

    by fundingRatePct   SYNUSDT  SNXXUSDT  MMTUSDT   AEVOUSDT  CRCLUSDT
    by fundingRateApr   SYNUSDT  MMTUSDT   DEXEUSDT  AEVOUSDT  SNXXUSDT

DEXEUSDT is the whole argument. Its per-settlement rate is -0.0303 %, which is
14th on the raw column — nowhere near a top five. It settles HOURLY, so its carry
is -265.85 %/yr, the 3rd most negative on the venue. The raw screen misses it and
takes CRCLUSDT (-125 %/yr) instead.

What is deliberately NOT changed
--------------------------------
``fundingRatePct`` keeps its exact meaning and its exact values. Existing specs,
the demo templates and the frozen golden baskets in two other modules all pin it,
and redefining a column in place would move all of that for a reason a reader
could not attribute. The fix is a NEW column with its unit in its name, plus the
``funding_info`` frame that supplies the divisor.

Absent schedule -> NaN, never 8
-------------------------------
The tempting default is the worst of the three outcomes. Assuming 8 hours would
produce numbers for the 447 of 743 contracts that do not settle 8-hourly, and
being wrong in a way that LOOKS repaired is worse than being NaN, because nobody
investigates it. So an absent frame warns and leaves NaN, an EMPTY frame is
refused outright (取不到 != 取到是空的), and a spec that ranks on the annualised
column without declaring the source is rejected statically rather than returning
an empty basket.
"""

from __future__ import annotations

import json
import urllib.request
import warnings
from pathlib import Path

import numpy as np
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
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (
    COLUMN_REQUIRES_SOURCE,
    build_selection_fn,
)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import (
    _synthetic_funding,
    _synthetic_funding_info,
    _synthetic_universe,
    validate_spec,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "universe_liquidity.json"

NODE = "funding_info"

#: The turnover floor the cross-capture claims below are measured over.
#:
#: The claim is about instruments a real screen would keep. Over all 727 rows the
#: venue's untraded tail would satisfy anything.
MIN_TURNOVER = 20_000_000.0

# ---------------------------------------------------------------------------
# The golden numbers, beside the assertion because they ARE the argument.
#
# The five most negative funding rates on the committed capture, over instruments
# above $20m turnover, taken from each column. One name differs and that is the
# point; the third block below says which and by how much.
MOST_NEGATIVE_BY_RATE = ["SYNUSDT", "SNXXUSDT", "MMTUSDT", "AEVOUSDT", "CRCLUSDT"]
MOST_NEGATIVE_BY_APR = ["SYNUSDT", "MMTUSDT", "DEXEUSDT", "AEVOUSDT", "SNXXUSDT"]

#: The instrument the raw column hides: hourly settlement turns a rate that is
#: 14th by magnitude into the 3rd biggest carry on the venue.
#: ``(rate %, interval hours, apr %, rank by rate, rank by apr)``
HOURLY_OUTLIER = ("DEXEUSDT", -0.030348, 1.0, -265.84848, 14, 3)

#: The instruments the raw screen keeps INSTEAD, and their real annual carry.
CROWDED_OUT = {"CRCLUSDT": -125.172735}

#: Dated DELIVERY contracts. ``premiumIndex`` reports a funding rate of exactly
#: 0.0 for these, which is Binance itself spelling "no funding" as a number a
#: near-zero-carry screen would keep. They have no ``fundingInfo`` row because
#: they have no schedule, so the annualised column is NaN for them.
DATED_CONTRACTS = ["BTCUSDT_260925", "BTCUSDT_261225",
                   "ETHUSDT_260925", "ETHUSDT_261225"]


def _fixture() -> dict:
    """A fresh parse per call — a test must not inherit another's mutations."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _frame(name: str, bundle: dict | None = None) -> pd.DataFrame:
    bundle = bundle if bundle is not None else _fixture()
    return pd.DataFrame((bundle["frames"][name].get("rows") or []))


def _joined(bundle: dict | None = None) -> pd.DataFrame:
    bundle = bundle if bundle is not None else _fixture()
    return ub.augment_with_funding(_frame("universe", bundle),
                                   _frame("funding", bundle),
                                   _frame(NODE, bundle))


def _liquid(bundle: dict | None = None) -> pd.DataFrame:
    return ub.filter_quote_volume(_joined(bundle), MIN_TURNOVER)


def _row(frame: pd.DataFrame, symbol: str):
    return frame.loc[frame["symbol"] == symbol].iloc[0]


# --------------------------------------------------------------------------- #
# helpers for the hand-built frames                                           #
# --------------------------------------------------------------------------- #

SYMBOLS = ("EIGHTH", "FOURTH", "HOURLY")


def _universe(symbols=SYMBOLS) -> pd.DataFrame:
    return pd.DataFrame({
        "instrument_id": list(symbols),
        "quoteVolume": [9e8 - index * 1e7 for index in range(len(symbols))],
        "available_time": [1_700_000_000_000] * len(symbols),
    })


def _funding(rate: float = 0.0001, symbols=SYMBOLS) -> pd.DataFrame:
    """The SAME per-settlement rate for every instrument, as a ratio.

    Identical on purpose: it is what makes the interval the only thing that can
    reorder the basket, so a test on that ordering cannot be passing for another
    reason.
    """
    return pd.DataFrame({
        "instrument_id": list(symbols),
        "lastFundingRate": [rate] * len(symbols),
        "time": [1_700_000_000_000] * len(symbols),
    })


def _info(hours=(8, 4, 1), symbols=SYMBOLS, **overrides) -> pd.DataFrame:
    frame = pd.DataFrame({
        "symbol": list(symbols),
        "fundingIntervalHours": list(hours),
        "adjustedFundingRateCap": [0.02] * len(symbols),
        "adjustedFundingRateFloor": [-0.02] * len(symbols),
    })
    for column, values in overrides.items():
        frame[column] = values
    return frame


# --------------------------------------------------------------------------- #
# THE ACCEPTANCE TEST                                                         #
# --------------------------------------------------------------------------- #


def test_the_same_rate_on_different_schedules_ranks_differently():
    """The reason ``fundingRateApr`` exists, stated as an inequality.

    Three instruments, ONE identical per-settlement rate of 0.01 %, three
    settlement intervals. ``fundingRatePct`` cannot separate them at all — it is
    the same number three times — so a "highest funding" basket over it is
    ordered by whatever the sort is stable on, i.e. by nothing. Annualised they
    are 10.95 %, 21.9 % and 87.6 % a year: the hourly contract pays EIGHT TIMES
    the 8-hourly one, and that is the number a carry trade earns.
    """
    joined = ub.augment_with_funding(_universe(), _funding(), _info())

    # The raw column: three identical values. Nothing to rank.
    assert joined["fundingRatePct"].tolist() == [0.01, 0.01, 0.01]
    assert joined["fundingIntervalHours"].tolist() == [8.0, 4.0, 1.0]

    # 0.01 % x (24/h) x 365, computed here from the definition rather than copied
    # from the implementation.
    expected = [0.01 * (24.0 / hours) * 365.0 for hours in (8, 4, 1)]
    assert joined["fundingRateApr"].tolist() == pytest.approx(expected)
    assert joined["fundingRateApr"].tolist() == pytest.approx([10.95, 21.9, 87.6])

    ranked = list(joined.nlargest(3, "fundingRateApr")["symbol"])
    assert ranked == ["HOURLY", "FOURTH", "EIGHTH"]
    # And the eight-fold spread is the size of the error the raw column made.
    assert (_row(joined, "HOURLY")["fundingRateApr"]
            == pytest.approx(8.0 * _row(joined, "EIGHTH")["fundingRateApr"]))


def test_the_annualised_column_reorders_a_real_market():
    """The same claim on captured market data, where the rates are not equal.

    A constructed frame proves the arithmetic; this proves the arithmetic MATTERS
    at the top of a basket someone would actually trade. DEXEUSDT is 14th by
    per-settlement rate and 3rd by annual carry, so a top-five screen on the raw
    column does not merely mis-order the basket — it omits the instrument with the
    third biggest carry on the venue and takes CRCLUSDT instead.
    """
    liquid = _liquid()
    assert len(liquid) == 92

    by_rate = list(liquid.nsmallest(5, "fundingRatePct")["symbol"])
    by_apr = list(liquid.nsmallest(5, "fundingRateApr")["symbol"])
    assert by_rate == MOST_NEGATIVE_BY_RATE
    assert by_apr == MOST_NEGATIVE_BY_APR
    assert by_rate != by_apr

    symbol, rate, hours, apr, rank_by_rate, rank_by_apr = HOURLY_OUTLIER
    row = _row(liquid, symbol)
    assert row["fundingRatePct"] == pytest.approx(rate, rel=1e-4)
    assert row["fundingIntervalHours"] == hours
    assert row["fundingRateApr"] == pytest.approx(apr, rel=1e-4)
    assert int((liquid["fundingRatePct"] < row["fundingRatePct"]).sum()) + 1 \
        == rank_by_rate
    assert int((liquid["fundingRateApr"] < row["fundingRateApr"]).sum()) + 1 \
        == rank_by_apr
    assert symbol in by_apr and symbol not in by_rate

    for crowded, its_apr in CROWDED_OUT.items():
        assert crowded in by_rate and crowded not in by_apr
        assert _row(liquid, crowded)["fundingRateApr"] == pytest.approx(
            its_apr, rel=1e-4)
        # The raw screen preferred a contract paying less than half as much.
        assert its_apr > row["fundingRateApr"]


def test_the_capture_is_not_vacuous_about_settlement_intervals():
    """A capture where every liquid instrument settled 8-hourly would make the
    multiplier a constant, and every test above would pass proving nothing."""
    intervals = sorted(_liquid()["fundingIntervalHours"].dropna().unique())

    assert len(intervals) >= 2, intervals
    assert set(intervals) <= {1.0, 4.0, 8.0}, intervals
    # The venue's plurality is 4-hourly, which is the fact that makes an 8h
    # default wrong for most of the market rather than for a corner of it.
    counts = _joined()["fundingIntervalHours"].value_counts()
    assert counts.idxmax() == 4.0
    assert counts[4.0] > counts.get(8.0, 0)


# --------------------------------------------------------------------------- #
# the raw column is untouched                                                 #
# --------------------------------------------------------------------------- #


def test_adding_the_schedule_does_not_change_the_per_settlement_column():
    """``fundingRatePct`` is pinned by two other modules' golden baskets.

    Compared value by value against the same call WITHOUT the schedule frame, so
    a future change to the annualisation cannot quietly move the column every
    existing spec ranks on.
    """
    bundle = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        without = ub.augment_with_funding(_frame("universe", bundle),
                                          _frame("funding", bundle))
    with_schedule = _joined(bundle)

    pd.testing.assert_series_equal(without["fundingRatePct"],
                                   with_schedule["fundingRatePct"])
    # And the new columns are additions, not replacements.
    assert set(with_schedule.columns) - set(without.columns) == set()
    assert list(without.columns[-3:]) == ["fundingRatePct",
                                          "fundingIntervalHours",
                                          "fundingRateApr"]


def test_annualising_never_changes_a_sign_so_order_keeps_its_meaning():
    """``order: asc`` must still mean "the most negative".

    The multiplier is ``(24/hours) * 365``, strictly positive, so the two columns
    agree on sign for every row. That is what lets ``intent.reconcile_intent``
    enforce a direction phrase matched against ``fundingRatePct`` on a spec that
    ranks ``fundingRateApr`` — see ``_METRIC_COLUMN_ALIASES``.
    """
    joined = _joined()
    both = joined.dropna(subset=["fundingRatePct", "fundingRateApr"])
    assert len(both) > 500

    assert (np.sign(both["fundingRatePct"])
            == np.sign(both["fundingRateApr"])).all()
    # Stated the way a spec would meet it: the bottom of one column is the bottom
    # of the other, in sign if not in order.
    assert (both.nsmallest(20, "fundingRateApr")["fundingRatePct"] < 0).all()


# --------------------------------------------------------------------------- #
# a missing schedule is missing, not eight hours                              #
# --------------------------------------------------------------------------- #


def test_an_absent_schedule_frame_warns_and_leaves_nan():
    """The default that must not exist, asserted against the number it would be.

    An 8-hour assumption is checked for explicitly rather than only asserting
    NaN: ``fillna(8)`` would still satisfy "the column exists", and 10.95 is
    exactly what it would have produced here.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joined = ub.augment_with_funding(_universe(), _funding())

    assert joined["fundingIntervalHours"].isna().all()
    assert joined["fundingRateApr"].isna().all()
    # The number an 8h default would have written, absent from every row.
    assert not (joined["fundingRateApr"] == pytest.approx(10.95)).any()
    # The per-settlement column is unaffected and correct.
    assert joined["fundingRatePct"].tolist() == [0.01, 0.01, 0.01]

    messages = [str(entry.message) for entry in caught
                if issubclass(entry.category, RuntimeWarning)]
    assert len(messages) == 1, messages
    assert "NOT defaulted to 8 hours" in messages[0]
    assert "with: [funding, funding_info]" in messages[0]


def test_an_empty_schedule_frame_is_refused_rather_than_nan_filled():
    """取不到 != 取到是空的.

    Absent means the spec did not ask for the frame, and the honest answer is a
    NaN column plus a warning. EMPTY means it asked and the collection returned
    nothing, and NaN-filling that would present a failed capture as a market in
    which no perpetual settles funding.
    """
    with pytest.raises(ValueError, match="source is empty"):
        ub.augment_with_funding(_universe(), _funding(), pd.DataFrame())


def test_the_schedule_frame_is_never_fetched_behind_the_callers_back(monkeypatch):
    """No live fallback for this argument, unlike ``funding_df``.

    ``augment_with_funding(frame)`` with no funding source keeps its historical
    convenience behaviour and fetches ``premiumIndex``. The schedule deliberately
    does not: a fetch here would fire during ``validate`` on a frontend-supplied
    spec, and the frame is cheap to declare from the same bundle.
    """
    def deny(*_args, **_kwargs):
        raise AssertionError("the schedule must not be fetched implicitly")

    monkeypatch.setattr(blocks_data, "fetch_funding_info", deny)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        joined = ub.augment_with_funding(_universe(), _funding())

    assert joined["fundingRateApr"].isna().all()


@pytest.mark.parametrize("hours, label", [
    ((8, 4, 0), "zero"),
    ((8, 4, -1), "negative"),
    ((8, 4, float("nan")), "missing"),
])
def test_an_unusable_interval_is_nan_and_not_an_infinite_carry(hours, label):
    """Dividing 24 by zero gives an infinity that sorts FIRST in every basket.

    So the row is NaN and counted. An interval outside 8 / 4 / 1 means the
    response changed shape, which the warning says.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joined = ub.augment_with_funding(_universe(), _funding(),
                                         _info(hours=hours))

    hourly = _row(joined, "HOURLY")
    assert pd.isna(hourly["fundingRateApr"]), label
    assert pd.isna(hourly["fundingIntervalHours"]), label
    assert not np.isinf(joined["fundingRateApr"].fillna(0.0)).any(), label
    # The other two are unaffected: one bad schedule row does not void the frame.
    assert _row(joined, "EIGHTH")["fundingRateApr"] == pytest.approx(10.95)

    messages = [str(entry.message) for entry in caught
                if issubclass(entry.category, RuntimeWarning)]
    assert any("HOURLY" in message and "infinite APR" in message
               for message in messages), (label, messages)


def test_a_dated_delivery_contract_reports_zero_funding_and_gets_no_apr():
    """The venue itself spells "no funding" as the number 0.0.

    ``premiumIndex`` returns ``lastFundingRate: 0.00000000`` and
    ``nextFundingTime: 0`` for the quarterly contracts, so a screen for
    "near-zero carry" over ``fundingRatePct`` keeps four instruments that pay no
    funding at all. They have no ``fundingInfo`` row because they have no
    schedule, so the annualised column is NaN and the ranker drops them — which is
    the correct answer arrived at from the correct fact.
    """
    joined = _joined()
    dated = joined[joined["symbol"].isin(DATED_CONTRACTS)]
    assert sorted(dated["symbol"]) == sorted(DATED_CONTRACTS)

    assert (dated["fundingRatePct"] == 0.0).all()
    assert dated["fundingIntervalHours"].isna().all()
    assert dated["fundingRateApr"].isna().all()
    # A near-zero-carry screen on the raw column keeps them; on the annualised
    # one they cannot appear, because a NaN score is dropped rather than ranked.
    near_zero_raw = joined[joined["fundingRatePct"].abs() <= 1e-9]
    assert set(DATED_CONTRACTS) <= set(near_zero_raw["symbol"])
    ranked = build_selection_fn({
        "selection": {
            "universe": [{"block": "universe.augment_with_funding",
                          "with": ["funding", "funding_info"]}],
            "score": "fundingRateApr", "order": "asc", "top_k": 700,
            "dedupe_by": "none",
        },
    })(_frame("universe"), None,
       frames={"funding": _frame("funding"), "funding_info": _frame(NODE)})
    assert set(DATED_CONTRACTS).isdisjoint({item["symbol"] for item in ranked})


def test_a_partial_schedule_raises_and_names_the_dated_contract_case():
    """The coverage floor. The message has to name the one legitimate hole.

    A stale or single-symbol schedule frame leaves most of the cross-section
    un-annualised, and a carry screen then ranks whichever handful survived. The
    message distinguishes that from the honest case a caller can act on: a
    universe narrowed to delivery contracts has no schedule to annualise with at
    all.
    """
    wide = _universe(SYMBOLS + ("D", "E", "F", "G", "H", "I", "J", "K"))
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_funding(_universe(SYMBOLS + ("D",)),
                                _funding(symbols=SYMBOLS + ("D",)),
                                _info(hours=(8,), symbols=("EIGHTH",)))

    message = str(excinfo.value)
    assert "1 of 4" in message
    assert "DATED DELIVERY" in message
    assert "BTCUSDT_260925" in message
    assert len(wide) == 11  # the wide frame is only here to document the shape


def test_the_real_capture_clears_the_floor_with_only_the_dated_contracts_missing():
    """The floor must not be tripped by a correct whole-market capture.

    723 of 727 covered, and the four absentees are exactly the delivery
    contracts — so the floor is doing its job rather than being permanently near
    its limit for an unexamined reason.
    """
    joined = _joined()
    missing = set(joined.loc[joined["fundingIntervalHours"].isna(), "symbol"])

    assert missing == set(DATED_CONTRACTS)
    coverage = joined["fundingIntervalHours"].notna().mean()
    assert coverage == pytest.approx(723 / 727)
    assert coverage > ub._FUNDING_INFO_MIN_COVERAGE


# --------------------------------------------------------------------------- #
# the two source vocabularies                                                 #
# --------------------------------------------------------------------------- #


def test_the_vendor_and_canonical_spellings_give_the_same_answer():
    vendor = _info()
    canonical = vendor.rename(columns={
        "symbol": "instrument_id",
        "fundingIntervalHours": "funding_interval_hours",
        "adjustedFundingRateCap": "funding_rate_cap",
        "adjustedFundingRateFloor": "funding_rate_floor"})

    from_vendor = ub.augment_with_funding(_universe(), _funding(), vendor)
    from_canonical = ub.augment_with_funding(_universe(), _funding(), canonical)

    pd.testing.assert_frame_equal(from_vendor, from_canonical)


def test_the_canonical_name_is_exactly_what_the_node_promises():
    """The alias table and the node's ``column_map`` are one fact in two files."""
    mapped = get_node(NODE).column_map
    accepted = ub._FUNDING_INFO_COLUMNS["fundingIntervalHours"]

    assert mapped["fundingIntervalHours"] in accepted
    assert "fundingIntervalHours" in accepted, "the vendor spelling too"


def test_a_schedule_frame_with_no_interval_column_is_refused():
    with pytest.raises(ValueError, match="fundingIntervalHours"):
        ub.augment_with_funding(
            _universe(), _funding(),
            _info().drop(columns=["fundingIntervalHours"]))


# --------------------------------------------------------------------------- #
# the fetcher and the catalog node                                            #
# --------------------------------------------------------------------------- #


def test_the_node_is_forward_only_and_says_the_venue_rewrites_the_schedule():
    """A schedule looks replayable and is not, so the hazard has to say why.

    ``fundingInfo`` states a contract TERM, which is what makes it tempting to
    back-fill into an older bundle — and Binance has moved large batches of
    perpetuals from 8h to 4h settlement, so a back-fill annualises past rates with
    a divisor those contracts did not have. That is a wrong number, not a missing
    one.
    """
    node = get_node(NODE)

    assert node.emits is FrameKind.RANK
    assert node.availability is Availability.FORWARD_ONLY
    assert "443 of 743" in node.pit_hazard
    assert node.fetcher == "cyqnt_trd.blocks.data.fetch_funding_info"
    assert "updateTime" in node.notes, (
        "the notes must record why updateTime is not the event time: it is null "
        "for BTCUSDT and ETHUSDT, so gating on it drops the two rows that matter "
        "most")


def test_the_node_is_not_a_fan_out_and_takes_no_roster():
    assert NODE not in {node for node, _key, _extra in FAN_OUT_SECTIONS.values()}
    assert SECTION_NODES["selection_funding_info"] == (NODE,)
    assert [param.key for param in get_node(NODE).params] == ["market_type"]


def test_the_fetcher_refuses_spot_because_spot_settles_no_funding():
    with pytest.raises(ValueError, match="futures-only"):
        blocks_data.fetch_funding_info("spot")


def test_the_fetcher_declares_its_columns_instead_of_trusting_the_response():
    """A silently missing ``fundingIntervalHours`` is the whole bug coming back.

    So the field list is declared and a response without it raises at the
    fetcher, where the message can say "do not paper over this with 8".
    """
    assert blocks_data._FUNDING_INFO_FIELDS == (
        "symbol", "fundingIntervalHours", "adjustedFundingRateCap",
        "adjustedFundingRateFloor")


# --------------------------------------------------------------------------- #
# YAML wiring                                                                 #
# --------------------------------------------------------------------------- #


def _apr_spec(sources=("funding", "funding_info"), score="fundingRateApr") -> dict:
    return {
        "spec_version": "1.0",
        "target": "standard_bot",
        "strategy": {"id": "funding_carry_selector"},
        "run": {"mode": "backtest"},
        "data": {"symbol": "BTCUSDT", "market_type": "futures",
                 "primary": {"interval": "1h"}},
        "selection": {
            "universe": [
                {"block": "universe.filter_quote_volume",
                 "params": {"min_quote_volume": MIN_TURNOVER}},
                {"block": "universe.augment_with_funding",
                 "with": list(sources)},
            ],
            "score": score,
            "order": "asc",
            "top_k": 5,
            "dedupe_by": "base_asset",
        },
    }


def test_a_spec_ranking_on_the_annualised_column_validates_and_runs():
    errors, warnings_out = validate_spec(_apr_spec())

    assert errors == []
    assert warnings_out == [], warnings_out


def test_ranking_on_the_annualised_column_without_the_schedule_is_refused():
    """The green-validate / empty-basket asymmetry, closed statically.

    With ``with: [funding]`` alone the block warns and NaN-fills, the ranker drops
    every row, and ``validate`` comes back ``errors=[]`` with the generic "no
    candidates" warning. So the spec ships and returns nothing, forever. The error
    names the key to add.
    """
    errors, _warnings = validate_spec(_apr_spec(sources=("funding",)))

    assert errors, "a spec that cannot produce its own score column validated"
    assert any("funding_info" in error and "fundingRateApr" in error
               for error in errors), errors
    assert COLUMN_REQUIRES_SOURCE["fundingRateApr"] == (
        "universe.augment_with_funding", "funding_info")


def test_the_refusal_does_not_fire_for_a_spec_that_ranks_the_raw_rate():
    """A spec asking for the per-settlement rate needs no schedule and must not
    be told to declare one — that would be an error the author cannot act on."""
    errors, warnings_out = validate_spec(
        _apr_spec(sources=("funding",), score="fundingRatePct"))

    assert errors == []
    # It does still get the annualisation notice, which is the nudge and not a
    # blocker: ranking the raw rate mixes units across 8h / 4h / 1h contracts.
    assert len(warnings_out) == 1
    assert "funding_info" in warnings_out[0]


def test_a_params_value_is_not_read_as_a_column_reference():
    """Only a RANKED / CONDITIONED column counts as "the spec uses this".

    A param value is an arbitrary string. If the checker scanned those too, a
    threshold or a note that happened to contain the column name would demand a
    source for a column nothing ranks on — an error whose author cannot act on it,
    on a spec that is correct.
    """
    spec = _apr_spec(sources=("funding",), score="quoteVolume")
    spec["selection"]["universe"][0]["params"] = {
        "min_quote_volume": MIN_TURNOVER, "note": "fundingRateApr"}

    errors, _warnings = validate_spec(spec)

    assert not any("fundingRateApr" in error for error in errors), errors


def test_the_spec_declares_the_section_the_live_collector_would_read():
    spec = _apr_spec()

    nodes = required_bundle_nodes(spec)
    assert {"funding", NODE} <= nodes
    sections = live_sections_for_spec(spec)
    assert "selection_funding" in sections
    assert "selection_funding_info" in sections

    plan = requests_for_sections(sections)
    assert (NODE, {"market_type": "futures"}, NODE) in plan
    # The per-symbol funding HISTORY request must not also be in the plan under
    # the same key — that is what `selection_funding` displaces.
    assert [key for _node, _params, key in plan].count("funding") == 1


def test_a_spec_that_only_wants_the_raw_rate_does_not_pay_for_the_schedule():
    """Collecting an unasked-for frame puts a source in the bundle that nothing
    reads, and every frame in a bundle is a status line a reader has to check."""
    sections = live_sections_for_spec(_apr_spec(sources=("funding",),
                                                score="fundingRatePct"))

    assert "selection_funding" in sections
    assert "selection_funding_info" not in sections


def test_validate_never_touches_the_network(monkeypatch):
    def deny(*_args, **_kwargs):
        raise AssertionError("validate must use the synthetic frames")

    monkeypatch.setattr(blocks_data, "_request_json", deny)
    monkeypatch.setattr(data_cli_subprocess, "_run", deny)
    monkeypatch.setattr(urllib.request, "urlopen", deny)

    errors, _warnings = validate_spec(_apr_spec())
    assert errors == []


# --------------------------------------------------------------------------- #
# the dry-run stand-in                                                        #
# --------------------------------------------------------------------------- #


def test_the_stand_in_schedule_offers_exactly_the_columns_the_real_source_has():
    stand_in = set(_synthetic_funding_info(_synthetic_universe()).columns)

    assert stand_in == set(blocks_data._FUNDING_INFO_FIELDS)


def test_the_stand_in_schedule_mixes_intervals_so_the_dry_run_can_tell_them_apart():
    """One interval would make annualising a constant multiplier.

    ``score: fundingRateApr`` and ``score: fundingRatePct`` would then dry-run to
    the SAME basket, so a spec that meant to annualise and a spec that forgot
    would validate identically — and validate is the only gate before a frontend
    ships one.
    """
    universe = _synthetic_universe()
    info = _synthetic_funding_info(universe)
    assert sorted(info["fundingIntervalHours"].unique()) == [1, 4, 8]

    frames = {"funding": _synthetic_funding(universe), "funding_info": info}

    def basket(column):
        spec = {"selection": {
            "universe": [{"block": "universe.augment_with_funding",
                          "with": ["funding", "funding_info"]}],
            "score": column, "order": "desc", "top_k": 5, "dedupe_by": "none"}}
        return [item["symbol"]
                for item in build_selection_fn(spec)(universe, None, frames=frames)]

    assert basket("fundingRatePct") != basket("fundingRateApr")


def test_the_stand_in_schedule_covers_the_stand_in_universe():
    """No hole, because the stand-in universe has no dated delivery contract.

    A hole here would model a row the stand-in universe does not contain and would
    drop a live instrument from every dry-run's ranking for a reason no real
    capture reproduces. The coverage floor is pinned above against a hand-built
    frame instead.
    """
    universe = _synthetic_universe()
    info = _synthetic_funding_info(universe)

    assert set(info["symbol"]) == set(universe["instrument_id"])


# --------------------------------------------------------------------------- #
# the demo template                                                           #
# --------------------------------------------------------------------------- #


def test_the_demo_funding_template_ranks_the_annualised_column():
    """The demo is the only funding spec a user never writes by hand.

    It is what the model is shown, so if it ranked ``fundingRatePct`` every
    generated funding basket would keep the mis-ordering no matter what the blocks
    can do.
    """
    import sys

    sys.path.insert(0, str(FIXTURES.parents[2] / "docs" / "strategy_yaml_spec"
                           / "demo"))
    try:
        import server as demo_server
    finally:
        sys.path.pop(0)
    import yaml

    spec = yaml.safe_load(demo_server.FUNDING_SELECTION_EXAMPLE_YAML)
    step = {item["block"]: item.get("with")
            for item in spec["selection"]["universe"]}

    assert spec["selection"]["score"] == "fundingRateApr"
    assert step["universe.augment_with_funding"] == ["funding", "funding_info"]
    # And the template itself validates, so the demo cannot ship an example that
    # its own gate would reject.
    errors, _warnings = validate_spec(spec)
    assert errors == []


def test_the_asc_variant_differs_from_the_desc_one_by_exactly_the_order_key():
    """The two demo examples are one substitution apart on purpose.

    If they drifted in any other line the model would have to guess which
    difference expressed "the most negative", which is the one thing the pair is
    there to teach.
    """
    import sys

    sys.path.insert(0, str(FIXTURES.parents[2] / "docs" / "strategy_yaml_spec"
                           / "demo"))
    try:
        import server as demo_server
    finally:
        sys.path.pop(0)

    desc = demo_server.FUNDING_SELECTION_EXAMPLE_YAML.splitlines()
    asc = demo_server.FUNDING_ASC_SELECTION_EXAMPLE_YAML.splitlines()

    differing = [(left, right) for left, right in zip(desc, asc) if left != right]
    assert len(differing) == 2, differing
    assert any("order:" in left for left, _right in differing)
    assert any("id:" in left for left, _right in differing)
    for line in asc:
        if line.strip().startswith("score:"):
            assert line.strip() == "score: fundingRateApr"
