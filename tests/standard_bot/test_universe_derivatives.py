"""Derivatives fan-out: position inventory, its weekly change, crowd skew.

The gap this closes
-------------------
``example_from_user_chat.yaml``'s gap #4 said it plainly: Binance publishes
``futures/data/globalLongShortAccountRatio`` and ``fapi/v1/openInterest``, and
``blocks.universe`` had no function that could reach either, so "合約持倉 ≥ 500 萬
美元" and "近一週持倉異動 ≥ 20%" could not be stated at all.

What makes this different from every earlier universe source
------------------------------------------------------------
There is **no all-market endpoint**. Omitting ``symbol`` from
``/fapi/v1/openInterest`` or from ``/futures/data/globalLongShortAccountRatio``
answers HTTP 400, so a cross-section of these fields exists only as a FAN-OUT:
one request per instrument. Three consequences, and each one gets tests below
because each fails silently if the guard is removed:

* **Cost is the roster length.** Narrowing 727 instruments to 127 with the free
  cross-sectional filters first turns a 2181-request plan into a 381-request one.
  So the roster ceiling raises rather than truncating — a truncated roster
  silently decides which instruments can be selected — and the joins refuse a
  frame that does not cover the frame it is joined onto, which is what a spec
  that put the augment step BEFORE its narrowing steps produces.
* **Open interest is denominated in coins.** 2.79e9 DOGEUSDT and 1.09e5 BTCUSDT
  are $193m and $6.8b. A "≥ 5 million" screen against the base column keeps every
  sub-dollar coin and drops BTC, and every number in the resulting basket looks
  plausible.
* **"持倉異動" has two answers.** The dollar change and the coin change disagree
  on WHICH instruments moved 20 % — on the committed capture, 5 instruments
  versus 3, sharing one name. Both columns are therefore joined and
  ``filter_oi_change`` takes an explicit ``basis``.

The acceptance test
-------------------
``test_the_frozen_capture_reproduces_the_hand_computed_answer`` replays the E5
screen (sector → $5m inventory → ±20 % weekly change) on the committed fan-out
fixture and pins 727 → 127 → 41 → 5 with the five names. Those numbers were
produced independently in hand-written Python against the live venue before any
of this code existed; reproducing them exactly, through the YAML pipeline, is
what the stage is for.

Why a second fixture file
-------------------------
``universe_cross_section.json`` is untouched — see
``test_the_earlier_frozen_cross_section_did_not_move``. These three sources
MEASURE a market, so back-filling them into a bundle whose ``decision_time`` is
four days old would be a fabrication (``freeze_selection_fixture.ADDABLE_NODES``
refuses exactly that, correctly), and re-capturing that bundle would move golden
baskets in two other test modules for a reason unrelated to this change. So the
fan-out capture is its own bundle, collected in one pass with its own decision
time.
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
    BundleRunError,
    live_sections_for_spec,
    required_bundle_nodes,
    run_bundle,
)
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (
    BARS_BLOCK,
    FAN_OUT_AUGMENTS,
    FETCHES_WITHOUT_SOURCE,
    narrows_the_universe,
)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import (
    _synthetic_contract_meta,
    _synthetic_fan_out_roster,
    _synthetic_long_short_ratio,
    _synthetic_oi_history,
    _synthetic_open_interest,
    _synthetic_universe,
    load_spec,
    validate_spec,
)

REPO = Path(__file__).parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "universe_derivatives.json"
EARLIER_FIXTURE = FIXTURES / "universe_cross_section.json"
SPEC_OI = REPO / "docs" / "strategy_yaml_spec" / "example_open_interest_screen.yaml"

#: ``universe_bars`` was ADDED to this bundle by the kline-fan-out stage, without
#: touching a byte of the five frames the funnel below is pinned against (its
#: ``--bars`` mode is an ``--add-frame``, and klines take an ``endTime``, so the
#: back-fill is verified rather than assumed). Listed here so the bundle's frame
#: set and its ``source_status`` keys stay exactly equal — a status line for a
#: frame that is not present is unfalsifiable, and a frame with no status line is
#: unattributable.
FROZEN_FRAMES = {"universe", "contract_meta", "open_interest_snapshot",
                 "oi_change_snapshot", "long_short_ratio_snapshot",
                 "universe_bars"}

#: The subset of :data:`FROZEN_FRAMES` this module's own claims rest on — the
#: fan-out frames. Kept separate so the assertion below still says "the earlier
#: cross-section has none of the fan-out frames" rather than accidentally
#: asserting something about bars, which that bundle never had either.
FAN_OUT_FRAMES = {"open_interest_snapshot", "oi_change_snapshot",
                  "long_short_ratio_snapshot"}

#: The sector narrowing the capture's roster was chosen with, and the spec's own.
SECTOR_TAGS = ["Alpha", "AI"]

#: The E5 thresholds, named once so the assertions and the spec cannot drift.
MIN_NOTIONAL_USD = 5_000_000.0
MIN_ABS_CHANGE_PCT = 20.0
LOOKBACK_DAYS = 7

# ---------------------------------------------------------------------------
# The golden numbers. They live HERE, beside the assertion, because the whole
# claim of this stage is that a YAML spec reproduces a hand-written answer — and
# a reviewer has to be able to read both without opening a second file.
#
# Produced independently, in hand-written Python against the live venue, before
# this code existed: 727 instruments -> 127 in the Alpha/AI sectors -> 41 with at
# least $5m of open interest -> 5 whose dollar open interest moved 20 % or more
# over the week. The committed capture reproduces all four numbers and all five
# names.
FUNNEL = [("sector", 127), ("inventory", 41), ("change", 5)]

OI_SCREEN_BASKET = [
    (1, "UAIUSDT"),
    (2, "TAGUSDT"),
    (3, "UBUSDT"),
    (4, "USUSDT"),
    (5, "ALLOUSDT"),
]

#: The same screen with ``basis: base`` — a DIFFERENT answer, which is why the
#: keyword is explicit and has no silent default. One name overlaps.
BASE_BASIS_SURVIVORS = ["AKEUSDT", "ESPUSDT", "UAIUSDT"]


def _fixture() -> dict:
    """A fresh parse per call — a test must not inherit another's mutations."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _frame(name: str, bundle: dict | None = None) -> pd.DataFrame:
    bundle = bundle if bundle is not None else _fixture()
    return pd.DataFrame((bundle["frames"][name].get("rows") or []))


def _narrowed(bundle: dict | None = None) -> pd.DataFrame:
    """The frozen universe after the three FREE cross-sectional steps.

    This is the roster the capture fanned out over, re-derived from the frame
    rather than hard-coded, so the fixture and the roster cannot drift apart
    without a test noticing.
    """
    bundle = bundle if bundle is not None else _fixture()
    joined = ub.augment_with_contract_meta(_frame("universe", bundle),
                                           _frame("contract_meta", bundle))
    return ub.filter_sub_type(ub.filter_crypto_only(joined), include=SECTOR_TAGS)


def _screened(bundle: dict | None = None) -> pd.DataFrame:
    """The narrowed frame with all three derivative joins applied."""
    bundle = bundle if bundle is not None else _fixture()
    frame = ub.augment_with_open_interest(
        _narrowed(bundle), _frame("open_interest_snapshot", bundle))
    frame = ub.augment_with_oi_change(
        frame, _frame("oi_change_snapshot", bundle), lookback_days=LOOKBACK_DAYS)
    return ub.augment_with_long_short_ratio(
        frame, _frame("long_short_ratio_snapshot", bundle))


def _symbols(frame: pd.DataFrame) -> list:
    return sorted(frame["instrument_id"].tolist())


def _spec_reading_all_three() -> dict:
    """The shipped spec plus the crowd-positioning step it does not need.

    The shipped spec reads two of the three fan-out sources on purpose — nothing
    should pay to fan out for a field it will not screen on — so the third's
    wiring is exercised through this variant rather than by widening the example.
    """
    spec = load_spec(str(SPEC_OI))
    spec["selection"]["universe"].append(
        {"block": "universe.augment_with_long_short_ratio",
         "with": ["long_short_ratio_snapshot"]})
    return spec


# --------------------------------------------------------------------------- #
# the fan-out fixture                                                         #
# --------------------------------------------------------------------------- #


def test_the_fixture_is_a_market_only_fan_out_bundle():
    bundle = _fixture()
    assert bundle["schema"] == "cyqnt.input/v1"
    assert set(bundle["frames"]) == FROZEN_FRAMES
    # A status line for a frame that is not here would be unfalsifiable: it is
    # copied onto every emitted signal, where a reader cannot check it.
    assert set(bundle["source_status"]) == FROZEN_FRAMES
    # Committed to a public repo: market data only, never an account snapshot.
    assert bundle["positions"] == {}
    assert bundle["equity"] is None

    # The live collector does not truncate, so the universe is the whole market
    # even though only a slice of it was fanned out over.
    assert len(_frame("universe", bundle)) > 200


def test_the_three_fan_out_frames_cover_exactly_the_narrowed_roster():
    """The capture's shape IS the ordering claim, so it is asserted, not assumed.

    ``open_interest_snapshot`` has one row per instrument in the roster and the
    roster is the sector-narrowed slice — not the whole 727-row market. If a
    future capture fanned out over everything, this is where the bill shows up.
    """
    bundle = _fixture()
    roster = set(_narrowed(bundle)["instrument_id"])
    assert len(roster) == dict(FUNNEL)["sector"]

    oi = _frame("open_interest_snapshot", bundle)
    assert set(oi["instrument_id"]) == roster
    assert len(oi) == len(roster), "one current reading per instrument, no series"

    ratio = _frame("long_short_ratio_snapshot", bundle)
    assert set(ratio["instrument_id"]) <= roster
    assert len(ratio) == len(set(ratio["instrument_id"]))

    history = _frame("oi_change_snapshot", bundle)
    assert set(history["instrument_id"]) <= roster
    # Two metrics x (lookback + 1) daily readings x roster. The +1 is the latest
    # reading the baseline is compared against.
    assert len(history) == len(roster) * (LOOKBACK_DAYS + 1) * 2
    assert set(history["metric"]) == {"oi_base", "oi_value"}


def test_the_open_interest_history_really_is_daily():
    """``lookback_days`` is a claim about wall-clock time, not about row count."""
    stamps = sorted(_frame("oi_change_snapshot")["event_time"].unique())
    gaps = {stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)}

    assert len(stamps) == LOOKBACK_DAYS + 1
    assert gaps == {86_400_000}, gaps
    # And every reading is a completed day, knowable before the decision.
    assert max(stamps) < _fixture()["decision_time"]


def test_the_earlier_frozen_cross_section_did_not_move():
    """This stage added a fixture; it did not disturb the one already pinned.

    ``universe_cross_section.json`` carries the golden baskets of two other test
    modules and the numbers quoted in three documents. Re-capturing it to make
    room for the derivative frames would have moved all of that for a reason
    unrelated to the change — so the fan-out went into its own bundle, and this
    asserts the old one still says what those modules expect.
    """
    earlier = json.loads(EARLIER_FIXTURE.read_text(encoding="utf-8"))

    assert earlier["decision_time"] == 1_785_591_229_856
    assert set(earlier["frames"]) == {"universe", "ticker_rank", "funding",
                                      "contract_meta"}
    assert len(earlier["frames"]["universe"]["rows"]) == 727
    assert FAN_OUT_FRAMES - set(earlier["frames"]) == FAN_OUT_FRAMES


# --------------------------------------------------------------------------- #
# the acceptance test                                                         #
# --------------------------------------------------------------------------- #


def test_the_frozen_capture_reproduces_the_hand_computed_answer():
    """727 -> 127 -> 41 -> 5, the funnel a human produced by hand.

    Every step is asserted, not only the end, because the end could be right for
    the wrong reason: a coverage guard that silently dropped instruments would
    also shrink the basket, and the intermediate counts are what tell the two
    apart.
    """
    universe = _frame("universe")
    narrowed = _narrowed()
    assert len(universe) == 727
    assert len(narrowed) == dict(FUNNEL)["sector"]

    with_oi = ub.augment_with_open_interest(narrowed,
                                            _frame("open_interest_snapshot"))
    liquid = ub.filter_open_interest(with_oi, min_notional_usd=MIN_NOTIONAL_USD)
    assert len(liquid) == dict(FUNNEL)["inventory"]

    with_change = ub.augment_with_oi_change(liquid, _frame("oi_change_snapshot"),
                                             lookback_days=LOOKBACK_DAYS)
    moved = ub.filter_oi_change(with_change, min_abs_pct=MIN_ABS_CHANGE_PCT)

    assert len(moved) == dict(FUNNEL)["change"]
    assert _symbols(moved) == sorted(name for _rank, name in OI_SCREEN_BASKET)


def test_the_spec_emits_that_funnel_as_one_basket():
    output = run_bundle(str(SPEC_OI), _fixture())

    assert output["signal_count"] == 1
    signal = output["signals"][0]
    assert signal["schema"] == "cyqnt.signal/v2"
    assert signal["kind"] == "selection"
    assert signal["universe_size"] == 727

    basket = [(item["rank"], item["symbol"]) for item in signal["candidates"]]
    assert basket == OI_SCREEN_BASKET
    # top_k is a ceiling, not a quota: the spec declares 20 and the filters leave
    # five, so five is the honest answer.
    assert len(basket) < int(load_spec(str(SPEC_OI))["selection"]["top_k"])


def test_every_candidate_carries_both_change_columns_whichever_one_filtered():
    """The audit trail for the ``basis`` decision, in the emitted contract.

    ``filter_oi_change(basis="notional")`` decided this basket, and the coin-count
    change is a materially different number for four of the five. Both are in
    ``features``, so a reader of the signal can see the number that was NOT used —
    which is the difference between a documented choice and an invisible one.
    """
    candidates = run_bundle(str(SPEC_OI), _fixture())["signals"][0]["candidates"]

    for candidate in candidates:
        features = candidate["features"]
        assert "oi_change_pct" in features
        assert "oi_base_change_pct" in features
        assert "oi_notional_usd" in features
        # The mark price that produced the notional, so it can be re-derived.
        assert features["oi_notional_usd"] == pytest.approx(
            features["oi_base"] * features["oi_mark_price"], rel=1e-9)

    disagreeing = [c for c in candidates
                   if abs(c["features"]["oi_change_pct"]) >= MIN_ABS_CHANGE_PCT
                   > abs(c["features"]["oi_base_change_pct"])]
    assert len(disagreeing) >= 3, (
        "on this capture most of the basket clears the threshold on dollars only; "
        "if that stopped being true the basis keyword would look decorative")


def test_the_two_bases_select_different_instruments():
    """Why ``basis`` is explicit: it is not a rounding difference.

    The same threshold over the same instruments keeps five on the dollar column
    and three on the coin column, overlapping in one name. A default that was not
    stated in the YAML would silently pick one of two different screens.
    """
    frame = ub.augment_with_oi_change(
        ub.filter_open_interest(
            ub.augment_with_open_interest(_narrowed(),
                                          _frame("open_interest_snapshot")),
            min_notional_usd=MIN_NOTIONAL_USD),
        _frame("oi_change_snapshot"), lookback_days=LOOKBACK_DAYS)

    notional = ub.filter_oi_change(frame, min_abs_pct=MIN_ABS_CHANGE_PCT)
    coins = ub.filter_oi_change(frame, min_abs_pct=MIN_ABS_CHANGE_PCT,
                                basis="base")

    assert _symbols(notional) == sorted(name for _rank, name in OI_SCREEN_BASKET)
    assert _symbols(coins) == sorted(BASE_BASIS_SURVIVORS)
    assert set(_symbols(notional)) != set(_symbols(coins))


def test_replay_is_byte_identical_including_the_signal_id():
    first = run_bundle(str(SPEC_OI), _fixture())
    second = run_bundle(str(SPEC_OI), _fixture())

    assert first["signals"][0]["signal_id"]
    assert (json.dumps(first, sort_keys=True)
            == json.dumps(second, sort_keys=True))


@pytest.fixture()
def no_network(monkeypatch):
    """Make every data transport record-then-raise, and hand back the log."""
    calls = []

    def _blocked(name):
        def deny(*args, **kwargs):
            calls.append(name)
            raise AssertionError("replay must not fetch: %s was called" % name)
        return deny

    monkeypatch.setattr(blocks_data, "_request_json",
                        _blocked("blocks.data._request_json"))
    monkeypatch.setattr(data_cli_subprocess, "_run",
                        _blocked("data_cli._subprocess._run"))
    monkeypatch.setattr(urllib.request, "urlopen",
                        _blocked("urllib.request.urlopen"))
    return calls


def test_replaying_the_fan_out_fixture_touches_no_network(no_network):
    """A fan-out that leaked would leak 381 times, not once."""
    output = run_bundle(str(SPEC_OI), _fixture())

    assert [(c["rank"], c["symbol"])
            for c in output["signals"][0]["candidates"]] == OI_SCREEN_BASKET
    assert no_network == []


def test_validating_the_spec_touches_no_network(no_network):
    errors, _warnings = validate_spec(load_spec(str(SPEC_OI)))

    assert errors == []
    assert no_network == []


# --------------------------------------------------------------------------- #
# coins are not dollars                                                       #
# --------------------------------------------------------------------------- #


def test_a_base_quantity_screen_and_a_dollar_screen_are_different_baskets():
    """The mistake ``oi_notional_usd`` exists to prevent, measured.

    "Open interest above 5 million" against the coin count is a real sentence
    that produces a plausible-looking basket of entirely different instruments —
    the sub-dollar coins, whose position count is large precisely because each
    coin is worth nothing.
    """
    frame = ub.augment_with_open_interest(_narrowed(),
                                          _frame("open_interest_snapshot"))

    dollars = set(ub.filter_open_interest(
        frame, min_notional_usd=MIN_NOTIONAL_USD)["instrument_id"])
    coins = set(frame[frame["oi_base"] >= MIN_NOTIONAL_USD]["instrument_id"])

    assert dollars and coins
    assert dollars != coins
    only_coins = coins - dollars
    assert only_coins, "no instrument passes on coins alone; the point is unproven"
    # And the ones a coin-count screen adds really are the cheap ones.
    cheap = frame[frame["instrument_id"].isin(only_coins)]["oi_mark_price"]
    assert cheap.max() < 1.0, cheap.tolist()


def test_the_notional_agrees_with_the_venues_own_dollar_figure():
    """Cross-check the multiplication against arithmetic we did not do.

    ``openInterestHist`` computes ``sumOpenInterestValue`` itself, and it is in
    the same bundle. Agreement is asserted on the MEDIAN across the roster, not
    per instrument: the snapshot is a live read and the history's newest point is
    that day's 00:00 reading, so they are hours apart and individual instruments
    legitimately differ. What the median catches is a units error — a forgotten
    multiplication lands at ~1e-5, not at 1.0.

    The left-hand side is the BLOCK's ``oi_notional_usd``, deliberately not a
    re-multiplication of the two source columns. The first version of this test
    did the multiplication itself and therefore passed while the block returned
    the bare coin count — it was checking this file's arithmetic, not the code's.
    """
    joined = ub.augment_with_open_interest(_narrowed(),
                                          _frame("open_interest_snapshot"))
    history = _frame("oi_change_snapshot")
    venue = (history[history["metric"] == "oi_value"]
             .sort_values("event_time").groupby("instrument_id")["value"].last())

    ratio = (joined.set_index("instrument_id")["oi_notional_usd"] / venue).dropna()

    assert len(ratio) == dict(FUNNEL)["sector"]
    assert ratio.median() == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("price", [0.0, -1.0, None],
                         ids=["zero", "negative", "missing"])
def test_an_unconvertible_mark_price_is_refused_rather_than_priced_at_nothing(price):
    """A zero or absent price makes the notional zero or NaN, and a floor then
    drops the instrument for being SMALL rather than for being unknown."""
    oi = _frame("open_interest_snapshot")
    oi.loc[oi.index[0], "mark_price"] = price

    with pytest.raises(ValueError, match="zero, negative or missing"):
        ub.augment_with_open_interest(_narrowed(), oi)


def test_the_join_reads_the_vendor_and_the_canonical_vocabulary_alike():
    """A bundle delivers ``oi_base`` / ``mark_price``; the fetcher returns
    ``openInterest`` / ``markPrice``. Both must reach the same numbers, or the
    block can be replayed or called but not both."""
    canonical = _frame("open_interest_snapshot")
    vendor = canonical.rename(columns={"instrument_id": "symbol",
                                       "oi_base": "openInterest",
                                       "mark_price": "markPrice"})

    from_canonical = ub.augment_with_open_interest(_narrowed(), canonical)
    from_vendor = ub.augment_with_open_interest(_narrowed(), vendor)

    pd.testing.assert_frame_equal(from_canonical, from_vendor)


# --------------------------------------------------------------------------- #
# the change: definition, and what it refuses to guess                        #
# --------------------------------------------------------------------------- #


def _history(values, base_values=None, *, symbol: str = "AAAUSDT",
             start: int = 1_700_000_000_000) -> pd.DataFrame:
    """A hand-built daily series, oldest first, in the long bundle vocabulary.

    BOTH magnitudes are emitted because the block requires both: it promises two
    columns, and a source carrying one would leave the other an all-NaN column
    that makes ``filter_oi_change(basis=...)`` empty the basket while looking like
    a strict screen. ``base_values`` defaults to the dollar series, so a test that
    only cares about one number does not have to invent the other.
    """
    base_values = values if base_values is None else base_values
    rows = []
    for metric, series in (("oi_value", values), ("oi_base", base_values)):
        rows += [
            {"instrument_id": symbol, "event_time": start + index * 86_400_000,
             "metric": metric, "value": float(value)}
            for index, value in enumerate(series)
        ]
    return pd.DataFrame(rows)


def _one_row_universe(symbol: str = "AAAUSDT") -> pd.DataFrame:
    return pd.DataFrame({"instrument_id": [symbol], "quoteVolume": [1e9]})


def test_the_change_is_the_latest_reading_against_the_mean_of_the_prior_ones():
    """The definition, stated as arithmetic a reader can check by eye.

    Baseline = mean(100, 100, 100, 100, 100, 100, 200) = 114.2857; latest = 200;
    change = (200 - 114.2857) / 114.2857 = +75 %. A single-point baseline would
    say 0 %, and a mean over all eight would say +64.6 % — three different
    answers, so the one in use is pinned rather than described.
    """
    history = _history([100, 100, 100, 100, 100, 100, 200, 200])

    joined = ub.augment_with_oi_change(_one_row_universe(), history,
                                       lookback_days=7)

    assert joined["oi_change_pct"].iloc[0] == pytest.approx(75.0, abs=1e-9)


def test_a_series_shorter_than_the_lookback_is_nan_and_reported():
    """Never averaged over the days it has.

    A perpetual listed three days ago has open interest growing from zero, so a
    short baseline manufactures a spectacular change for exactly the newest and
    thinnest instruments — the ones this kind of screen surfaces.
    """
    history = _history([100, 100, 300])            # 3 readings, lookback wants 8

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joined = ub.augment_with_oi_change(_one_row_universe(), history,
                                           lookback_days=7)

    assert pd.isna(joined["oi_change_pct"].iloc[0])
    messages = [str(entry.message) for entry in caught
                if issubclass(entry.category, RuntimeWarning)]
    assert any("fewer than 7+1" in message and "AAAUSDT" in message
               for message in messages), messages
    # And the filter drops it, in both directions, saying so.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert len(ub.filter_oi_change(joined, min_abs_pct=20.0)) == 0
        assert len(ub.filter_oi_change(joined, max_pct=1e9)) == 0
    assert any("missing" in str(entry.message) for entry in caught)


def test_a_non_daily_cadence_is_refused_instead_of_relabelled():
    """``period=5m`` with ``lookback_days=7`` would measure 35 minutes.

    Same column name, same YAML, no error anywhere — so the cadence is measured
    from the timestamps that actually arrived rather than taken on trust.
    """
    five_minutes = pd.DataFrame([
        {"instrument_id": "AAAUSDT", "event_time": 1_700_000_000_000 + i * 300_000,
         "metric": metric, "value": 100.0 + i}
        for i in range(9) for metric in ("oi_value", "oi_base")
    ])

    with pytest.raises(ValueError, match="spaced .* hours apart"):
        ub.augment_with_oi_change(_one_row_universe(), five_minutes,
                                  lookback_days=7)


def test_a_baseline_of_zero_is_nan_and_not_an_infinite_change():
    """Zero open interest a week ago is an undefined change, not a huge one.

    An infinity would sort first under ``order: desc`` and become rank 1.
    """
    history = _history([0, 0, 0, 0, 0, 0, 0, 500])

    joined = ub.augment_with_oi_change(_one_row_universe(), history,
                                       lookback_days=7)

    assert pd.isna(joined["oi_change_pct"].iloc[0])


def test_the_long_and_the_wide_history_vocabularies_agree():
    """A bundle carries a melted ``MetricFrame``; the fetcher returns a wide
    vendor frame. Both must produce the same change."""
    wide = pd.DataFrame([
        {"symbol": "AAAUSDT", "timestamp": 1_700_000_000_000 + i * 86_400_000,
         "sumOpenInterest": 10.0 + i, "sumOpenInterestValue": 100.0 * (1 + i)}
        for i in range(8)
    ])
    long = _history([100.0 * (1 + i) for i in range(8)],
                    [10.0 + i for i in range(8)])

    from_wide = ub.augment_with_oi_change(_one_row_universe(), wide,
                                          lookback_days=7)
    from_long = ub.augment_with_oi_change(_one_row_universe(), long,
                                          lookback_days=7)

    pd.testing.assert_frame_equal(from_wide, from_long)


def test_a_lookback_of_zero_is_refused():
    with pytest.raises(ValueError, match="lookback_days must be at least 1"):
        ub.augment_with_oi_change(_one_row_universe(), _history([1, 2]),
                                  lookback_days=0)


# --------------------------------------------------------------------------- #
# long/short: the unit                                                        #
# --------------------------------------------------------------------------- #


def test_shares_of_one_become_percentage_points():
    """67.28, not 0.6728 — the unit every other column in a universe frame uses.

    ``fundingRatePct`` and ``priceChangePercent`` are percentage points, so a
    share left as a fraction makes "retail above 60" and "above 0.6" both look
    like reasonable YAML while only one of them can be right.
    """
    frame = ub.augment_with_long_short_ratio(
        _narrowed(), _frame("long_short_ratio_snapshot"))
    known = frame.dropna(subset=["long_account_pct"])

    assert len(known) == dict(FUNNEL)["sector"]
    assert known["long_account_pct"].between(0.0, 100.0).all()
    assert known["long_account_pct"].max() > 1.0, "still a fraction"
    # The ratio keeps its own scale, where 1.0 is balanced.
    leaning_long = known[known["long_account_pct"] > 50.0]
    assert (leaning_long["long_short_ratio"] > 1.0).all()


def test_a_source_that_switched_to_percentages_is_refused():
    """Verified, not assumed: shares must still sum to 1.

    If the venue started sending 67.28 instead of 0.6728, multiplying by 100
    would give a ``long_account_pct`` of 6728 and every "retail above 60 %" screen
    would match every instrument — a silent inversion of the whole condition.
    """
    ratio = _frame("long_short_ratio_snapshot")
    ratio[["longAccount", "shortAccount"]] *= 100.0

    with pytest.raises(ValueError, match="unit changed"):
        ub.augment_with_long_short_ratio(_narrowed(), ratio)


def test_the_crowd_filter_splits_the_frozen_roster():
    frame = ub.augment_with_long_short_ratio(
        _narrowed(), _frame("long_short_ratio_snapshot"))

    long_leaning = ub.filter_long_short_ratio(frame,
                                              min_long_account_pct=60.0)

    assert 0 < len(long_leaning) < len(frame), (
        "the threshold holds for all or for none of this capture, so a test "
        "using it could not fail")


def test_strict_crowd_long_threshold_excludes_the_exact_boundary():
    """`> 60%` and `>= 60%` are different user requests."""
    frame = pd.DataFrame({
        "instrument_id": ["EQUALUSDT", "ABOVEUSDT"],
        "long_account_pct": [60.0, 60.0001],
    })

    inclusive = ub.filter_long_short_ratio(frame, min_long_account_pct=60.0)
    strict = ub.filter_long_short_ratio(
        frame, min_long_account_pct_exclusive=60.0)

    assert inclusive["instrument_id"].tolist() == ["EQUALUSDT", "ABOVEUSDT"]
    assert strict["instrument_id"].tolist() == ["ABOVEUSDT"]
    with pytest.raises(ValueError, match="not both"):
        ub.filter_long_short_ratio(
            frame, min_long_account_pct=60.0,
            min_long_account_pct_exclusive=60.0,
        )


# --------------------------------------------------------------------------- #
# the fan-out constraints: the point of this stage                            #
# --------------------------------------------------------------------------- #


FAN_OUT_FETCHERS = [
    pytest.param(blocks_data.fetch_open_interest_cross_section, id="open_interest"),
    pytest.param(blocks_data.fetch_oi_history_cross_section, id="oi_history"),
    pytest.param(blocks_data.fetch_long_short_ratio_cross_section, id="long_short"),
]


@pytest.fixture()
def count_requests(monkeypatch):
    """Count outbound calls without making any."""
    calls = []

    def record(url, params):
        calls.append((url, dict(params)))
        raise AssertionError("this test must not reach the network")

    monkeypatch.setattr(blocks_data, "_request_json", record)
    return calls


@pytest.mark.parametrize("fetcher", FAN_OUT_FETCHERS)
def test_an_oversized_roster_raises_before_a_single_request(fetcher, count_requests):
    """The ceiling is a refusal, not a slice.

    Truncating would silently decide which instruments can be selected — the ones
    cut are the tail of whatever order the roster arrived in — and nothing in the
    output would record that the screen saw a smaller market than it claims.
    """
    roster = ["SYM%04dUSDT" % index
              for index in range(blocks_data.FAN_OUT_MAX_SYMBOLS + 1)]

    with pytest.raises(ValueError, match="above the .* ceiling"):
        fetcher(roster)

    assert count_requests == [], "the ceiling must be checked before fetching"


@pytest.mark.parametrize("fetcher", FAN_OUT_FETCHERS)
@pytest.mark.parametrize("roster", [None, [], ["", "  "]],
                         ids=["absent", "empty", "blank"])
def test_a_roster_with_nothing_in_it_is_refused(fetcher, roster, count_requests):
    """An empty cross-section becomes an all-NaN column, and a threshold over
    that returns an empty basket — the same output a working strict screen gives."""
    with pytest.raises(ValueError):
        fetcher(roster)
    assert count_requests == []


@pytest.mark.parametrize("fetcher", FAN_OUT_FETCHERS)
def test_a_bare_string_roster_is_refused_rather_than_read_letter_by_letter(
    fetcher, count_requests
):
    with pytest.raises(ValueError, match="one character at a time"):
        fetcher("BTCUSDT")
    assert count_requests == []


@pytest.mark.parametrize("fetcher", FAN_OUT_FETCHERS)
def test_spot_is_refused_rather_than_served_futures_numbers(fetcher, count_requests):
    """A spot market has no perpetual position inventory. Dropping the argument
    would answer a spot screen with real futures figures."""
    with pytest.raises(ValueError, match="futures-only field"):
        fetcher(["BTCUSDT"], market_type="spot")
    assert count_requests == []


def test_a_duplicated_roster_is_paid_for_once(monkeypatch):
    """Concatenating two filters can name the same instrument twice, and the
    ceiling is a request budget — paying twice for one answer buys nothing."""
    seen = []

    def fake(url, params):
        seen.append(params["symbol"])
        return [{"symbol": params["symbol"], "longAccount": 0.6,
                 "shortAccount": 0.4, "longShortRatio": 1.5,
                 "timestamp": 1_700_000_000_000}]

    monkeypatch.setattr(blocks_data, "_request_json", fake)

    frame = blocks_data.fetch_long_short_ratio_cross_section(
        ["BTCUSDT", "ethusdt", "BTCUSDT", "ETHUSDT"])

    assert seen == ["BTCUSDT", "ETHUSDT"]
    assert frame["symbol"].tolist() == ["BTCUSDT", "ETHUSDT"]


def test_an_unknown_long_short_mode_is_refused_naming_what_each_measures():
    with pytest.raises(ValueError, match="different populations"):
        blocks_data.fetch_long_short_ratio_cross_section(["BTCUSDT"],
                                                         mode="retail")


def test_the_open_interest_fetcher_refuses_a_roster_the_venue_does_not_price(
    monkeypatch
):
    """One missing mark price is one missing dollar figure, which a floor then
    drops for looking small rather than for being unknown."""
    monkeypatch.setattr(
        blocks_data, "fetch_premium_index",
        lambda *a, **k: pd.DataFrame([{"symbol": "BTCUSDT", "markPrice": 60000.0}]))

    with pytest.raises(RuntimeError, match="roster is stale"):
        blocks_data.fetch_open_interest_cross_section(["BTCUSDT", "GONEUSDT"])


def test_the_history_fetcher_keeps_a_symbol_with_no_readings_out_without_failing(
    monkeypatch
):
    """"Read it and it was empty" is a market state for a statistics series —
    the aggregate starts some time after a perpetual is listed — unlike a
    transport error, which raises."""
    def fake(url, params):
        if params["symbol"] == "NEWUSDT":
            return []
        return [{"symbol": params["symbol"], "sumOpenInterest": "10",
                 "sumOpenInterestValue": "1000", "CMCCirculatingSupply": "1",
                 "timestamp": 1_700_000_000_000}]

    monkeypatch.setattr(blocks_data, "_request_json", fake)

    frame = blocks_data.fetch_oi_history_cross_section(["BTCUSDT", "NEWUSDT"])

    assert frame["symbol"].tolist() == ["BTCUSDT"]
    # The third-party supply figure is dropped: it is not open interest, and it
    # would repeat on every row of every series.
    assert "CMCCirculatingSupply" not in frame.columns


# --------------------------------------------------------------------------- #
# the ordering requirement, as a runtime refusal                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("block, frame_name", [
    (ub.augment_with_open_interest, "open_interest_snapshot"),
    (ub.augment_with_oi_change, "oi_change_snapshot"),
    (ub.augment_with_long_short_ratio, "long_short_ratio_snapshot"),
])
def test_joining_a_narrow_roster_onto_the_whole_market_is_refused(block, frame_name):
    """The step-order mistake, and it is the expensive one to get wrong.

    A spec that puts the augment step BEFORE its narrowing steps joins a
    127-instrument roster onto 727 rows: 600 instruments get a NaN reading, every
    threshold below drops them, and the basket is short for a reason nothing in
    the output records. So the join refuses, and the message names step order as
    the first thing to check.
    """
    with pytest.raises(ValueError) as raised:
        block(_frame("universe"), _frame(frame_name))

    message = str(raised.value)
    assert "covers only 127 of 727" in message, message
    assert "AFTER the steps that narrow the universe" in message


def test_an_empty_derivative_frame_is_a_failed_capture_and_not_an_empty_market():
    """These fields have no all-market endpoint, so nothing to fall back on."""
    for block, frame_name in ((ub.augment_with_open_interest, "open_interest"),
                              (ub.augment_with_oi_change, "oi_change"),
                              (ub.augment_with_long_short_ratio, "long_short")):
        with pytest.raises(ValueError, match="source is empty"):
            block(_narrowed(), pd.DataFrame())


def test_a_partial_derivative_frame_is_refused_instead_of_nan_filled():
    """A NaN column reads downstream as "this instrument holds a small position",
    which is a reason to trade it."""
    oi = _frame("open_interest_snapshot").drop(columns=["mark_price"])

    with pytest.raises(ValueError, match="no column for 'oi_mark_price'"):
        ub.augment_with_open_interest(_narrowed(), oi)


# --------------------------------------------------------------------------- #
# the filters                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("call, label", [
    (lambda frame: ub.filter_open_interest(frame), "filter_open_interest"),
    (lambda frame: ub.filter_oi_change(frame), "filter_oi_change"),
    (lambda frame: ub.filter_long_short_ratio(frame), "filter_long_short_ratio"),
])
def test_a_filter_with_no_bound_is_refused(call, label):
    """With no bound it returns the frame unchanged, which is indistinguishable
    from the step not being in the spec at all."""
    with pytest.raises(ValueError, match="needs at least one bound"):
        call(_screened())


@pytest.mark.parametrize("call, step, source", [
    (lambda frame: ub.filter_open_interest(frame, min_notional_usd=1.0),
     "augment_with_open_interest", "open_interest_snapshot"),
    (lambda frame: ub.filter_oi_change(frame, min_abs_pct=1.0),
     "augment_with_oi_change", "oi_change_snapshot"),
    (lambda frame: ub.filter_long_short_ratio(frame, min_long_account_pct=1.0),
     "augment_with_long_short_ratio", "long_short_ratio_snapshot"),
])
def test_the_filter_names_the_missing_step_its_source_and_the_ordering(
    call, step, source
):
    with pytest.raises(ValueError) as raised:
        call(_narrowed())

    message = str(raised.value)
    assert step in message
    assert "with: [%s]" % source in message
    assert "AFTER the steps that narrow the universe" in message


def test_an_unknown_reading_is_dropped_in_both_directions_and_counted():
    """"We could not read this instrument's open interest" is not "this
    instrument holds little open interest", and only one is a reason to skip it.

    ``conditions.value_above`` cannot make this distinction — NaN yields False,
    which is the same answer a small number gives — which is the main reason
    these filters are blocks.
    """
    frame = _screened()
    frame.loc[frame.index[0], "oi_notional_usd"] = None
    victim = frame["instrument_id"].iloc[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kept = ub.filter_open_interest(frame, min_notional_usd=0.0)
        dropped_by_ceiling = ub.filter_open_interest(frame, max_notional_usd=1e30)

    assert victim not in set(kept["instrument_id"])
    assert victim not in set(dropped_by_ceiling["instrument_id"])
    messages = [str(entry.message) for entry in caught
                if issubclass(entry.category, RuntimeWarning)]
    assert any("filter_open_interest" in message and "missing" in message
               for message in messages), messages


def test_an_unknown_basis_is_refused_naming_both_meanings():
    with pytest.raises(ValueError, match="basis must be one of"):
        ub.filter_oi_change(_screened(), min_abs_pct=20.0, basis="dollars")


def test_the_signed_bounds_and_the_magnitude_bound_are_different_screens():
    """``min_abs_pct`` is not ``min_pct``: the capture contains a -26.5 % mover."""
    frame = _screened()

    magnitude = ub.filter_oi_change(frame, min_abs_pct=MIN_ABS_CHANGE_PCT)
    added_only = ub.filter_oi_change(frame, min_pct=MIN_ABS_CHANGE_PCT)
    closed_only = ub.filter_oi_change(frame, max_pct=-MIN_ABS_CHANGE_PCT)

    assert set(_symbols(added_only)) | set(_symbols(closed_only)) == set(
        _symbols(magnitude))
    assert _symbols(closed_only), "no instrument shed inventory; test is vacuous"
    assert set(_symbols(added_only)).isdisjoint(_symbols(closed_only))


def test_the_fluent_builder_reaches_the_same_answer_as_the_blocks():
    """``UniverseFilter`` is the Python-facing surface of the same steps."""
    chained = (ub.UniverseFilter(_narrowed())
               .with_open_interest(_frame("open_interest_snapshot"))
               .filter_open_interest(min_notional_usd=MIN_NOTIONAL_USD)
               .with_oi_change(_frame("oi_change_snapshot"),
                               lookback_days=LOOKBACK_DAYS)
               .filter_oi_change(min_abs_pct=MIN_ABS_CHANGE_PCT)
               .symbols())

    assert sorted(chained) == sorted(name for _rank, name in OI_SCREEN_BASKET)


# --------------------------------------------------------------------------- #
# catalog + collection plan                                                   #
# --------------------------------------------------------------------------- #


FAN_OUT_NODES = ["open_interest_snapshot", "oi_change_snapshot",
                 "long_short_ratio_snapshot"]


@pytest.mark.parametrize("name", FAN_OUT_NODES)
def test_the_catalog_node_declares_a_forward_only_fan_out_with_its_hazard(name):
    node = get_node(name)

    assert node.availability is Availability.FORWARD_ONLY
    # FORWARD_ONLY without a hazard would be a claim with no reason attached.
    assert node.pit_hazard
    assert node.fetcher
    # The roster is required and has no default: a default would be "everything".
    symbols = next(param for param in node.params if param.key == "symbols")
    assert symbols.required is True
    assert symbols.default is None
    assert "ONE REQUEST PER SYMBOL" in node.endpoint.upper()


@pytest.mark.parametrize("name", FAN_OUT_NODES)
def test_the_declared_return_columns_are_what_the_fetcher_actually_returns(name):
    """The catalog is the public contract the docs and codegen read, so it must
    describe the frame that arrives — including for a node no test can fetch."""
    node = get_node(name)
    synthetic = {
        "open_interest_snapshot": _synthetic_open_interest,
        "oi_change_snapshot": _synthetic_oi_history,
        "long_short_ratio_snapshot": _synthetic_long_short_ratio,
    }[name](_synthetic_universe())

    assert set(synthetic.columns) == set(node.returns.columns), (
        "the dry-run stand-in and the declared contract disagree; a stand-in "
        "column the real source lacks buys a spec a green validate and a hard "
        "failure on every real bundle")


def test_the_history_node_declares_a_metric_frame_and_the_others_a_rank_frame():
    """The shape has to state the grain honestly.

    Open interest and the ratio are one reading per instrument at one as-of — a
    ``RankFrame``. The history is several readings per instrument, which under a
    RankFrame would be a claim about the grain that is not true.
    """
    assert get_node("oi_change_snapshot").emits is FrameKind.METRIC
    assert get_node("open_interest_snapshot").emits is FrameKind.RANK
    assert get_node("long_short_ratio_snapshot").emits is FrameKind.RANK


def test_the_fan_out_sections_are_declared_once_and_reachable_by_name():
    for section, (node, key, _extra) in FAN_OUT_SECTIONS.items():
        assert SECTION_NODES[section] == (node,)
        # The bundle key IS the node name, unlike ``funding_snapshot`` -> ``funding``,
        # so a spec's ``with:`` name maps straight onto the section.
        assert key == node


def test_a_fan_out_section_without_a_roster_is_refused():
    with pytest.raises(ValueError, match="explicit fan_out_symbols roster"):
        requests_for_sections(["universe", "selection_open_interest"])


def test_a_fan_out_section_with_an_empty_roster_is_refused():
    with pytest.raises(ValueError, match="EMPTY fan_out_symbols"):
        requests_for_sections(["selection_oi_change"], fan_out_symbols=[])


def test_the_plan_passes_the_roster_and_the_period_the_block_needs():
    plan = requests_for_sections(
        ["universe", "contract_meta", "selection_open_interest",
         "selection_oi_change", "selection_long_short_ratio"],
        fan_out_symbols=["btcusdt", "ETHUSDT"])
    by_key = {key: params for _node, params, key in plan}

    assert by_key["open_interest_snapshot"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
    # A 7-day lookback needs 8 daily readings; the block refuses to average fewer.
    assert by_key["oi_change_snapshot"]["period"] == "1d"
    assert by_key["oi_change_snapshot"]["limit"] == LOOKBACK_DAYS + 1
    assert by_key["long_short_ratio_snapshot"]["mode"] == "global"


def test_the_plan_refuses_two_different_nodes_under_one_bundle_key():
    """``build_live_bundle`` writes ``frames[key]`` as it goes, so a repeated key
    means the later request silently overwrites the earlier one and the strategy
    reads a frame it did not ask for — with a status line that looks fine."""
    plan = list(requests_for_sections(["universe"]))
    plan.append(("open_interest_snapshot", {"symbols": ["BTCUSDT"]}, "universe"))

    from cyqnt_trd.standard_bot.data.live_snapshot import _refuse_duplicate_keys

    with pytest.raises(ValueError, match="same bundle key"):
        _refuse_duplicate_keys(plan)


def test_the_spec_pulls_exactly_the_sections_it_uses_into_the_live_plan():
    spec = load_spec(str(SPEC_OI))
    nodes = required_bundle_nodes(spec)
    sections = live_sections_for_spec(spec)

    assert {"universe", "contract_meta", "open_interest_snapshot",
            "oi_change_snapshot"} <= nodes
    assert "selection_open_interest" in sections
    assert "selection_oi_change" in sections
    # The spec reads no crowd positioning, so nothing pays to fan out for it.
    assert "long_short_ratio_snapshot" not in nodes
    assert "selection_long_short_ratio" not in sections
    assert "derivatives" not in sections, (
        "the per-symbol derivatives section would collect a BTCUSDT history and "
        "land it under a key a cross-sectional block cannot read")


@pytest.mark.parametrize("block, source", [
    ("universe.augment_with_open_interest", "open_interest_snapshot"),
    ("universe.augment_with_oi_change", "oi_change_snapshot"),
    ("universe.augment_with_long_short_ratio", "long_short_ratio_snapshot"),
])
def test_the_block_may_not_fetch_its_own_source_from_a_spec(block, source):
    """Without ``with:`` these blocks fall back to a live fan-out over the frame
    they are handed — so a forgotten declaration would not touch the network
    once during validate, it would touch it once per instrument."""
    assert FETCHES_WITHOUT_SOURCE[block] == source

    spec = _spec_reading_all_three()
    stripped = [step for step in spec["selection"]["universe"]
                if step["block"] == block]
    assert stripped, block
    for step in stripped:
        del step["with"]
    errors, _warnings = validate_spec(spec)

    assert any("with: [%s]" % source in error for error in errors), errors


@pytest.mark.parametrize("node", FAN_OUT_NODES)
def test_an_empty_frame_stops_the_run_and_names_the_source(node):
    """The runner names the node that could not be read, instead of surfacing it
    as a ValueError from inside a universe step three frames down.

    "I could not read it" and "I read it and it was empty" stay different facts
    everywhere else in the bundle contract; for these three the second is also a
    failure, because there is no all-market endpoint to have read.
    """
    bundle = _fixture()
    bundle["frames"][node]["rows"] = []
    bundle["source_status"][node] = "empty"

    with pytest.raises(BundleRunError, match=node):
        run_bundle(_spec_reading_all_three(), bundle)


def test_a_spec_step_without_its_frame_refuses_rather_than_fetching():
    """``with:`` never falls back to live network data."""
    bundle = _fixture()
    del bundle["frames"]["open_interest_snapshot"]
    del bundle["source_status"]["open_interest_snapshot"]

    with pytest.raises(BundleRunError, match="open_interest_snapshot"):
        run_bundle(str(SPEC_OI), bundle)


# --------------------------------------------------------------------------- #
# the dry-run stand-ins                                                       #
# --------------------------------------------------------------------------- #


def test_the_dry_run_exercises_both_sides_of_every_derivative_filter():
    """Otherwise ``validate`` proves only that nothing raised.

    A stand-in where every instrument clears every threshold makes each filter
    dry-run as a step that drops nothing — the same output a missing step gives —
    and one where none clears empties the universe before the later steps are
    reached, which is how this spec first came back "produced no candidates".
    So both sides are asserted, and asserted INSIDE the sector slice the screen
    narrows to, because that is the only place the filters actually run.
    """
    universe = _synthetic_universe()
    narrowed = ub.filter_sub_type(
        ub.filter_crypto_only(
            ub.augment_with_contract_meta(universe,
                                          _synthetic_contract_meta(universe))),
        include=SECTOR_TAGS)
    assert len(narrowed) > 3, "the sector filter left too little to screen"

    frame = ub.augment_with_open_interest(narrowed,
                                          _synthetic_open_interest(universe))
    frame = ub.augment_with_oi_change(frame, _synthetic_oi_history(universe))
    frame = ub.augment_with_long_short_ratio(
        frame, _synthetic_long_short_ratio(universe))

    checks = {
        "filter_open_interest": ub.filter_open_interest(
            frame, min_notional_usd=MIN_NOTIONAL_USD),
        "filter_oi_change(notional)": ub.filter_oi_change(
            frame, min_abs_pct=MIN_ABS_CHANGE_PCT),
        "filter_oi_change(base)": ub.filter_oi_change(
            frame, min_abs_pct=MIN_ABS_CHANGE_PCT, basis="base"),
        "filter_long_short_ratio": ub.filter_long_short_ratio(
            frame, min_long_account_pct=60.0),
    }
    for label, kept in checks.items():
        assert 0 < len(kept) < len(frame), "%s: %d of %d" % (label, len(kept),
                                                             len(frame))
    # And the two bases disagree here too, so a spec with the wrong one cannot
    # validate as though it were the right one.
    assert set(checks["filter_oi_change(notional)"]["instrument_id"]) != set(
        checks["filter_oi_change(base)"]["instrument_id"])


def test_the_stand_in_roster_is_narrower_than_the_stand_in_universe():
    """Because a real one is, and that is what makes the ordering mistake
    catchable at validate time rather than on the first real bundle."""
    universe = _synthetic_universe()
    roster = _synthetic_fan_out_roster(universe)

    assert 0 < len(roster) < len(universe)


@pytest.mark.parametrize("block_ref", sorted(FAN_OUT_AUGMENTS))
def test_validate_catches_every_fan_out_augment_placed_before_a_narrowing_step(
        block_ref):
    """The ordering mistake fails ``validate``, offline, with the fix in the text.

    Parametrised over the whole set because the version of this test that
    covered only ``augment_with_open_interest`` was worse than no test: it
    passed, and the three blocks it did not name validated green when misordered
    and raised in production. They were caught by the dry-run's coverage
    arithmetic only for open interest, and only by accident — the synthetic
    roster lands at 87%, under that join's 0.95 floor but over the 0.50 floor the
    other two use, and the bars join has no coverage floor at all.

    So the ordering is checked statically now. Any block added to
    ``FAN_OUT_AUGMENTS`` gets a case here for free, which is the property the
    hand-written single-block version did not have.
    """
    spec = load_spec(str(SPEC_OI))
    steps = spec["selection"]["universe"]
    augment = {"block": block_ref, "with": ["universe_bars"],
               "params": {"indicator": "rsi", "timeframe": "1d", "as": "probe"}}
    if block_ref != BARS_BLOCK:
        augment = next((step for step in steps if step["block"] == block_ref),
                       {"block": block_ref})
        steps.remove(augment) if augment in steps else None
    steps.insert(0, augment)

    errors, _warnings = validate_spec(spec)

    assert any("must come AFTER at least one step that narrows" in error
               for error in errors), errors
    assert any(block_ref in error for error in errors), errors


def test_a_narrowing_step_above_the_augment_is_what_clears_it():
    """The mutation test for the check above: it must accept the fixed spec.

    A check that rejects the misordered spec AND the correct one would pass the
    test above while making the block unusable, so the accept side is asserted
    on the same spec with only the order changed back.
    """
    spec = load_spec(str(SPEC_OI))
    assert validate_spec(spec)[0] == []

    steps = spec["selection"]["universe"]
    assert any(narrows_the_universe(step["block"]) for step in steps), (
        "SPEC_OI must contain a narrowing step or this asserts nothing")

    # ...and an augment_* is not one, however much it looks like a step.
    assert not narrows_the_universe("universe.augment_with_open_interest")
    assert narrows_the_universe("universe.filter_crypto_only")
    assert narrows_the_universe("universe.top_gainers")


def test_the_dry_run_catches_a_spec_that_filters_without_augmenting():
    spec = load_spec(str(SPEC_OI))
    spec["selection"]["universe"] = [
        step for step in spec["selection"]["universe"]
        if step["block"] != "universe.augment_with_open_interest"
    ]

    errors, _warnings = validate_spec(spec)

    assert any("augment_with_open_interest" in error for error in errors), errors


def test_the_shipped_spec_validates_and_warns_about_nothing():
    """A warning in a shipped example trains its readers to skim the report."""
    errors, spec_warnings = validate_spec(load_spec(str(SPEC_OI)))

    assert errors == []
    assert spec_warnings == []
