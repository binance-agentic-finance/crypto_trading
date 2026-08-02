"""Per-candidate K-lines: a technical indicator inside the selection layer.

The gap this closes
-------------------
The cross-sectional frame has one row per instrument and **no bars**, so
"Supertrend(10,3) bearish on H4 and H1 and M15" and "up 100% from its 3-month
low" — between them the most common shape in the selection corpus — could not be
written at all. What shipped instead was::

    - block: universe.top_losers
      params: { n: 30 }

A 24-hour percentage wearing an indicator's words. It validated, it ran, the
basket held thirty plausible names, and no field of the output said a proxy had
been substituted. ``tools/nl2yaml/capability.py`` records that as
``GAP-PER-SYMBOL-INDICATOR``.

The design decision, and why it is the one under test
-----------------------------------------------------
The bundle carries **bars**; the indicator is computed in the block, from the
spec's own parameters. The rejected alternative was to pre-compute indicator
VALUES during capture, and it fails twice:

* ``params: {period: 10}`` against a bundle that baked in 14 makes the spec a
  LIE — it validates, it runs, it reports Supertrend(10,3), and it screened on
  something else. ``test_the_spec_not_the_bundle_decides_the_indicator_period``
  is that claim: changing only the YAML changes the answer, on identical bars.
* point-in-time correctness stops being checkable. An unfinished candle has a
  ``close_time`` in the future and the bundle's own gate drops it
  (``test_the_unfinished_candle_is_dropped_by_the_bundles_own_pit_gate``); a
  pre-computed number carries no evidence of which bars went into it.

Three independent PIT guards, one test each
-------------------------------------------
1. ``end_ms = decision_time`` on the request — which is why this needs
   ``blocks.data.fetch_klines`` rather than the ``klines`` catalog node, whose
   binance-cli fetcher has no ``endTime`` at all.
2. the unfinished candle is dropped by the bundle, not by the fetcher, so the
   property lives in the artifact where a reviewer can check it.
3. a series too short for its indicator's warm-up **raises**, naming the
   instrument. This is the guard whose absence is invisible: NaN in an indicator
   column is read by ``conditions.value_below(col, 0)`` as "this coin is not
   bearish" AND by ``value_above`` as "not bullish", so the freshest listings —
   exactly what a momentum screen is hunting — leave both screens without
   appearing in either.

Why the bars went INTO the existing derivatives fixture
-------------------------------------------------------
Every other cross-sectional node in the catalog serves only "now", so
``freeze_selection_fixture.ADDABLE_NODES`` is right to refuse back-filling them.
Klines are different in a way that is *checkable rather than argued*: the
endpoint takes an ``endTime``, so ``end_ms = decision_time`` returns exactly the
bars a capture at that instant would have seen, and the freeze script refuses the
addition unless every captured bar closed at or before that instant. So
``universe_bars`` was added to ``universe_derivatives.json`` at its own
decision_time, the five existing frames stayed byte-identical (asserted in
``test_the_frames_the_earlier_stage_pinned_are_byte_identical``), and no golden
basket moved.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks import data as blocks_data
from cyqnt_trd.blocks import indicators as ind
from cyqnt_trd.blocks import universe as ub
from cyqnt_trd.blocks._utils import (
    IndicatorShapeError,
    first_param_is_df,
    select_indicator_component,
)
from cyqnt_trd.standard_bot.data.catalog import Availability, FrameKind, get_node
from cyqnt_trd.standard_bot.data.input_bundle import (
    FRAME_SHAPES,
    build_input_bundle,
    load_input_bundle,
)
from cyqnt_trd.standard_bot.data.live_snapshot import (
    BARS_SECTION,
    SECTION_NODES,
    requests_for_sections,
)
from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import (
    BundleRunError,
    live_sections_for_spec,
    plan_bars_capture,
    required_bundle_nodes,
    run_bundle,
)
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (
    BARS_BLOCK,
    FETCHES_WITHOUT_SOURCE,
    SpecError,
    bar_timeframes_for_spec,
    run_universe_steps,
    universe_steps_before_bars,
)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import (
    _synthetic_universe,
    _synthetic_universe_bars,
    load_spec,
    validate_spec,
)

REPO = Path(__file__).parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "universe_derivatives.json"
SPEC_RUNUP = REPO / "docs" / "strategy_yaml_spec" / "example_three_month_runup_screen.yaml"
SPEC_RESONANCE = (REPO / "docs" / "strategy_yaml_spec"
                  / "example_multi_timeframe_supertrend.yaml")

NODE = "universe_bars"

# ---------------------------------------------------------------------------
# The golden numbers. They live HERE, beside the assertions, because the claim of
# this stage is that a YAML spec reproduces a hand-written answer and a reviewer
# has to be able to read both without opening a second file.
#
# Produced independently in hand-written Python against the live venue, cut at the
# frozen bundle's own decision_time (1785608174524): the low-to-high range of each
# candidate's last 90 CLOSED daily bars.
RUNUP_BASKET = [
    (1, "USUSDT", 1300.2),
    (2, "ALLOUSDT", 594.3),
    (3, "TAGUSDT", 589.3),
    (4, "UBUSDT", 329.5),
    (5, "UAIUSDT", 125.8),
]

#: Supertrend(10,3) direction per timeframe on the same frozen bars.
#:
#: This matrix is the content of the resonance claim, and it is pinned instead of
#: the basket because the basket is EMPTY on this capture — every one of the five
#: had just gained between 126 % and 1300 %, so none is bearish on all three. An
#: empty basket alone is indistinguishable from a broken screen; the matrix shows
#: the screen read three genuinely different timeframes and answered honestly.
#:
#: It is also the number that condemns the proxy: three of the five ARE bearish on
#: at least one timeframe, so ``any_of`` and ``all_of`` give 3 and 0 — and a "top
#: 30 losers" list is neither of them.
RESONANCE_MATRIX = {
    "USUSDT": (+1, -1, +1),
    "UAIUSDT": (-1, +1, +1),
    "ALLOUSDT": (-1, +1, -1),
    "TAGUSDT": (+1, +1, +1),
    "UBUSDT": (+1, +1, +1),
}

#: The mirrored screen — bullish on all three — on the same bars. Non-empty, which
#: is what shows the machinery selects when the market agrees rather than merely
#: never selecting.
RESONANCE_BASKET_LONG = ["TAGUSDT", "UBUSDT"]

#: Every (instrument, timeframe) pair in the frozen capture holds exactly this
#: many CLOSED bars: 100 asked for, minus the one unfinished candle ``endTime``
#: includes and the PIT gate drops.
BARS_PER_PAIR = 99
CAPTURED_TIMEFRAMES = ["1d", "4h", "1h", "15m"]
DECISION_TIME = 1_785_608_174_524

#: The five frames the derivatives stage pinned, with the md5 of each frame entry
#: as it stood BEFORE ``universe_bars`` was added. Recomputed from the committed
#: file by the test below: if a later capture disturbs any of them, the golden
#: baskets in ``test_universe_derivatives.py`` have already moved and this says so
#: first.
FRAMES_PINNED_EARLIER = {
    "universe": "e7e306d053635e4ed1db8baa9dc04dea",
    "contract_meta": "d48d5872f5375389427900443ec1d27a",
    "open_interest_snapshot": "6bc523772d0e144ff3008a401032d481",
    "oi_change_snapshot": "2e3aff67dbecc99be41e73df7fac61ca",
    "long_short_ratio_snapshot": "91687a29a7a356a39f967f3fcdef737d",
}


def _fixture() -> dict:
    """A fresh parse per call — a test must not inherit another's mutations."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _frame(name: str, bundle: dict | None = None) -> pd.DataFrame:
    bundle = bundle if bundle is not None else _fixture()
    return pd.DataFrame((bundle["frames"][name].get("rows") or []))


def _extras(bundle: dict | None = None) -> dict:
    bundle = bundle if bundle is not None else _fixture()
    extras = {key: _frame(key, bundle) for key in bundle["frames"]}
    extras["ticker_rank"] = None
    return extras


def _survivors(bundle: dict | None = None) -> pd.DataFrame:
    """The frozen universe after the runup spec's Pass-1 prefix."""
    bundle = bundle if bundle is not None else _fixture()
    spec = load_spec(str(SPEC_RUNUP))
    return run_universe_steps(universe_steps_before_bars(spec),
                              _frame("universe", bundle), _extras(bundle))


def _bars(**over):
    """A hand-built two-instrument, one-timeframe bar frame.

    ``AAA`` trends up and ``BBB`` trends down, both monotonically, so a Supertrend
    direction is deterministic and the two rows disagree — a fixture whose rows
    cannot disagree cannot show a filter working.
    """
    n = over.pop("n", 120)
    timeframe = over.pop("timeframe", "4h")
    index = np.arange(n)
    out = []
    for symbol, drift in (("AAAUSDT", +0.004), ("BBBUSDT", -0.004)):
        close = 100.0 * (1.0 + drift) ** index
        open_ = np.concatenate([[close[0]], close[:-1]])
        out.append(pd.DataFrame({
            "instrument_id": symbol,
            "timeframe": timeframe,
            "open_time": 1_700_000_000_000 + index * 3_600_000,
            "close_time": 1_700_000_000_000 + (index + 1) * 3_600_000 - 1,
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": 1_000.0 + index,
            **over,
        }))
    return pd.concat(out, ignore_index=True)


def _universe(*symbols):
    names = list(symbols) or ["AAAUSDT", "BBBUSDT"]
    return pd.DataFrame({"instrument_id": names,
                         "quoteVolume": [1e9] * len(names)})


# --------------------------------------------------------------------------- #
# the acceptance test: E5 condition 6                                         #
# --------------------------------------------------------------------------- #


def test_the_frozen_capture_reproduces_the_hand_computed_three_month_runup():
    """+1300% / +594% / +589% / +330% / +126%, through the YAML pipeline.

    Those five numbers were produced in hand-written Python against the live venue
    before this code existed. Reproducing them through ``run_bundle`` — spec ->
    interpreter -> block -> indicator -> signal contract — is what the stage is
    for, and the funnel is asserted as well as the answer because the answer could
    be right for the wrong reason.
    """
    batch = run_bundle(str(SPEC_RUNUP), str(FIXTURE))

    assert batch["signal_count"] == 1
    candidates = batch["signals"][0]["candidates"]
    assert [(c["rank"], c["symbol"]) for c in candidates] == [
        (rank, symbol) for rank, symbol, _gain in RUNUP_BASKET]
    for candidate, (_rank, _symbol, gain) in zip(candidates, RUNUP_BASKET):
        assert candidate["score"] == pytest.approx(gain, abs=0.1)
        # The column is left ON the frame, so an operator reading the emitted
        # candidate can see the number the ranking used rather than trusting the
        # score field alone.
        assert candidate["features"]["gain_3m"] == pytest.approx(gain, abs=0.1)


def test_the_indicator_reproduces_the_arithmetic_bar_for_bar():
    """The block's number IS (window high - window low) / window low, checked
    against the frozen bars directly rather than against itself."""
    bars = _frame(NODE)
    daily = bars[bars["timeframe"] == "1d"]

    for _rank, symbol, gain in RUNUP_BASKET:
        series = daily[daily["instrument_id"] == symbol].sort_values("open_time")
        window = series.tail(90)
        assert len(window) == 90
        low, high = window["low"].min(), window["high"].max()
        assert (high - low) / low * 100.0 == pytest.approx(gain, abs=0.1)


def test_a_higher_floor_actually_drops_rows():
    """The shipped 100% floor keeps all five, so on its own it proves nothing.

    A threshold that happens to be satisfied by every row is indistinguishable
    from a threshold that is not applied. So the same spec is run with a floor
    this market splits on.
    """
    spec = load_spec(str(SPEC_RUNUP))
    spec["selection"]["min_score"] = 600.0
    kept = [c["symbol"] for c in
            run_bundle(spec, str(FIXTURE))["signals"][0]["candidates"]]

    assert kept == ["USUSDT"], "600% should keep only the +1300% name"


# --------------------------------------------------------------------------- #
# the acceptance test: three-timeframe resonance                              #
# --------------------------------------------------------------------------- #


def test_three_timeframes_are_three_different_answers_on_the_frozen_bars():
    """The matrix in :data:`RESONANCE_MATRIX`, and the all_of / any_of split.

    This is the assertion that condemns ``top_losers(n=30)``: the same instrument
    is bearish on one timeframe and bullish on another, so a single 24h percentage
    cannot stand in for "bearish on H4 and H1 and M15" — and 3-versus-0 is how far
    apart ``any_of`` and ``all_of`` are on real data.
    """
    spec = load_spec(str(SPEC_RESONANCE))
    frame = run_universe_steps(spec["selection"]["universe"],
                              _frame("universe"), _extras())
    columns = ["st_dir_4h", "st_dir_1h", "st_dir_15m"]
    observed = {row["symbol"]: tuple(int(row[column]) for column in columns)
                for _, row in frame.iterrows()}

    assert observed == RESONANCE_MATRIX

    bearish = frame[columns] < 0
    assert int(bearish.all(axis=1).sum()) == 0, "all_of"
    assert int(bearish.any(axis=1).sum()) == 3, "any_of"


def test_the_resonance_spec_returns_an_empty_basket_rather_than_a_plausible_one():
    """Nothing is bearish on all three here, and the honest answer is nothing.

    The proxy's failure was returning thirty plausible names for a condition it
    had not evaluated. An empty basket is the correct output of a screen whose
    condition no instrument meets, and it is reported as such — with the universe
    size in the summary, so a reader can tell "screened 727, none matched" from
    "the source was missing".
    """
    batch = run_bundle(str(SPEC_RESONANCE), str(FIXTURE))

    assert batch["signal_count"] == 1
    signal = batch["signals"][0]
    assert signal["candidates"] == []
    assert "727" in signal["summary"]


def test_the_mirrored_bullish_resonance_does_select():
    """Same bars, same three steps, ``value_above`` instead of ``value_below``.

    Without this the suite would only ever have seen the resonance screen return
    nothing, which a broken screen also does.
    """
    spec = load_spec(str(SPEC_RESONANCE))
    spec["selection"].pop("short_when")
    spec["selection"]["long_when"] = {"all_of": [
        {"cond": "conditions.value_above", "args": [column, 0]}
        for column in ("st_dir_4h", "st_dir_1h", "st_dir_15m")]}
    candidates = run_bundle(spec, str(FIXTURE))["signals"][0]["candidates"]

    assert sorted(c["symbol"] for c in candidates) == sorted(RESONANCE_BASKET_LONG)
    assert {c["direction"] for c in candidates} == {"long"}


def test_the_spec_not_the_bundle_decides_the_indicator_period():
    """Change only the YAML, on byte-identical bars, and the answer changes.

    This is the whole argument for shipping BARS in the bundle rather than
    pre-computed indicator values. If the capture had baked in one period, this
    test could not exist: both specs would report their own period and return the
    same basket, and nothing in either output would reveal it.
    """
    # 25 and not 40: the frozen capture holds 99 bars per pair, and the block's own
    # warm-up guard needs 3 x period. A test that tripped that guard would be
    # asserting the guard, not the period.
    long_period = load_spec(str(SPEC_RESONANCE))
    for step in long_period["selection"]["universe"]:
        if step["block"] == BARS_BLOCK:
            step["params"]["period"] = 25
    frame_10 = run_universe_steps(
        load_spec(str(SPEC_RESONANCE))["selection"]["universe"],
        _frame("universe"), _extras())
    frame_25 = run_universe_steps(long_period["selection"]["universe"],
                                  _frame("universe"), _extras())

    columns = ["st_dir_4h", "st_dir_1h", "st_dir_15m"]
    assert not frame_10[columns].equals(frame_25[columns]), (
        "Supertrend(10) and Supertrend(25) agreed on every instrument and every "
        "timeframe, so this test cannot tell the period was read from the spec")


# --------------------------------------------------------------------------- #
# PIT guard 1: endTime = decision_time                                        #
# --------------------------------------------------------------------------- #


def test_the_bars_node_takes_an_end_ms_and_the_klines_node_cannot():
    """Guard 1 is a property of the WIRING, so it is checked there.

    ``klines`` is the obvious node to reuse and it is the wrong one: its fetcher is
    a binance-cli subprocess with no ``endTime``, so a capture through it can only
    mean "as of now" and a replay would pair a past universe with present prices.
    """
    bars = get_node(NODE)
    assert bars.emits is FrameKind.BAR
    assert "end_ms" in {param.key for param in bars.params}
    assert bars.fetcher == "cyqnt_trd.blocks.data.fetch_klines_cross_section"

    klines = get_node("klines")
    assert "end_ms" not in {param.key for param in klines.params}
    assert "data_cli" in klines.fetcher


def test_a_replay_plans_bars_at_the_bundles_own_decision_time():
    """``plan_bars_capture`` returns the bundle's instant, never "now"."""
    plan = plan_bars_capture(load_spec(str(SPEC_RUNUP)), _fixture())

    assert plan.end_ms == DECISION_TIME
    assert plan.timeframes == ["1d"]
    assert plan.symbols == sorted(symbol for _r, symbol, _g in RUNUP_BASKET)


def test_the_collection_plan_passes_decision_time_through_as_the_bars_end_ms():
    """The section wiring, not the caller, decides ``end_ms``.

    Leaving it to the caller is how a replay ends up asking for today's prices
    beside a frozen universe — and the request would look right.
    """
    section, node, key = BARS_SECTION

    def plan(**over):
        requests = requests_for_sections(
            [section], bar_symbols=["BTCUSDT"], bar_timeframes=["4h"], **over)
        return {request[0]: request for request in requests}

    replay = plan(bars_end_ms=DECISION_TIME)[node]
    assert replay[1]["end_ms"] == DECISION_TIME
    assert replay[1]["symbols"] == ["BTCUSDT"]
    assert replay[1]["timeframes"] == ["4h"]
    assert replay[2] == key

    # None means "as of now", which is the honest request when the decision IS now:
    # the still-moving candle is dropped by the gate either way.
    assert plan()[node][1]["end_ms"] is None


def test_the_bars_section_refuses_a_missing_roster_or_timeframe_set():
    """Neither has a default: guessing either is a silent wrong answer."""
    section = BARS_SECTION[0]

    with pytest.raises(ValueError) as no_roster:
        requests_for_sections([section], bar_timeframes=["4h"])
    assert "fan out over one request PER INSTRUMENT" in str(no_roster.value)

    with pytest.raises(ValueError) as no_timeframes:
        requests_for_sections([section], bar_symbols=["BTCUSDT"])
    assert "bar_timeframes" in str(no_timeframes.value)

    with pytest.raises(ValueError) as empty:
        requests_for_sections([section], bar_symbols=["BTCUSDT"],
                              bar_timeframes=[])
    assert "EMPTY bar_timeframes" in str(empty.value)


# --------------------------------------------------------------------------- #
# PIT guard 2: the unfinished candle                                          #
# --------------------------------------------------------------------------- #


def test_the_unfinished_candle_is_dropped_by_the_bundles_own_pit_gate():
    """Guard 2, read straight out of the artifact.

    ``endTime`` is INCLUSIVE of the candle containing it, so the newest bar of
    every series is still moving when it is fetched. The fetcher does NOT drop it —
    it carries its real ``close_time``, which is after the decision, and the
    bundle's gate drops it for that reason. That is the difference the "bars, not
    indicator values" decision buys: the property is visible in the file.
    """
    entry = _fixture()["frames"][NODE]
    bars = pd.DataFrame(entry["rows"])

    # One dropped row per (instrument, timeframe) pair, recorded by the gate.
    pairs = bars.groupby(["instrument_id", "timeframe"]).size()
    assert len(pairs) == len(RUNUP_BASKET) * len(CAPTURED_TIMEFRAMES)
    assert entry["rows_after_decision_time_dropped"] == len(pairs)
    assert set(pairs) == {BARS_PER_PAIR}

    assert bars["close_time"].max() <= DECISION_TIME
    # And the drop was not merely a row count: the newest surviving 15m bar closes
    # within one interval of the decision, so nothing older was thrown away.
    newest = bars[bars["timeframe"] == "15m"]["close_time"].max()
    assert 0 < DECISION_TIME - newest < 900_000


def test_the_fetcher_leaves_the_unfinished_bar_in_place():
    """The complement of the test above: the property has to be the GATE's.

    If the fetcher silently dropped it, "the bundle is point-in-time" would be a
    claim about capture code instead of about data — and a bundle assembled by any
    other path would lose the guarantee with no diagnostic.
    """
    frame = blocks_data.fetch_klines_cross_section(
        ["BTCUSDT"], timeframes=["4h"], limit=3, end_ms=DECISION_TIME)

    assert frame["close_time"].max() > DECISION_TIME


def test_a_bar_frame_is_windowed_per_timeframe_not_per_metric():
    """A BarFrame has no ``metric`` column, so the default series key collapses it.

    Keyed on ``(instrument_id, metric)`` every timeframe of one instrument shares a
    single 240-row bucket, so a four-timeframe capture keeps one of them. The
    symptom is not an error — it is "the 15m indicator is always NaN", which the
    joining block reports as a warm-up failure pointing at the capture's ``limit``.
    """
    bars = pd.concat([_bars(timeframe=timeframe, n=50)
                      for timeframe in ("4h", "1h", "15m")], ignore_index=True)
    bundle = build_input_bundle(
        symbol="AAAUSDT", interval="4h", decision_time=1_800_000_000_000,
        extra_frames={NODE: bars}, metric_lookback=40)

    rows = pd.DataFrame(bundle["frames"][NODE]["rows"])
    kept = rows.groupby(["instrument_id", "timeframe"]).size()
    assert set(kept) == {40}, kept.to_dict()
    assert len(kept) == 6, "2 instruments x 3 timeframes, each windowed separately"


def test_a_bar_frame_is_not_truncated_by_the_event_row_cap():
    """``max_event_rows`` caps EVENTS. A flat cap over a multi-symbol bar frame
    keeps whole series for whichever instrument sorts last and none for the rest —
    and the joining block then refuses for want of warm-up, naming the wrong cause.
    """
    bars = _bars(n=120)
    bundle = build_input_bundle(
        symbol="AAAUSDT", interval="4h", decision_time=1_800_000_000_000,
        extra_frames={NODE: bars}, metric_lookback=None, max_event_rows=50)

    assert len(bundle["frames"][NODE]["rows"]) == len(bars)


# --------------------------------------------------------------------------- #
# PIT guard 3: warm-up, and the NaN that reads as "condition not met"          #
# --------------------------------------------------------------------------- #


def test_a_series_short_of_its_warm_up_raises_and_names_the_instrument():
    """Guard 3. The message has to name the coin, because the capture is usually
    complete for the majors and short for exactly one new listing."""
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(
            _universe(), _bars(n=25), indicator="supertrend", timeframe="4h",
            output=1, period=10)

    message = str(excinfo.value)
    assert "fewer than 30 4h bars" in message
    assert "AAAUSDT=25" in message
    assert "BEARISH screen and a BULLISH one alike" in message


def test_the_warm_up_multiple_is_read_from_the_indicators_effective_params():
    """Signature defaults count, not only what the spec wrote.

    ``ichimoku`` looks back 52 bars by default. A warm-up derived from the spec
    alone would pass a 40-bar frame merely because the spec named no period, and
    the block would then join a column that has not settled.
    """
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(_universe(), _bars(n=140),
                                  indicator="ichimoku", timeframe="4h",
                                  column="tenkan")

    assert "fewer than 156 4h bars" in str(excinfo.value)


def test_a_pure_rolling_window_may_declare_min_bars_multiple_one():
    """And it has to be able to, or the screen inverts its own question.

    ``range_gain_pct(period=90)`` has no seed to settle, so 90 bars is the exact
    answer. At the default multiple of 3 it would demand 270 daily bars — nine
    months — and refuse every instrument listed more recently, which is precisely
    the population a "gained 100%" screen exists to find. On the frozen capture
    that is all five of them.
    """
    bars = _bars(n=95)

    with pytest.raises(ValueError, match="fewer than 270"):
        ub.augment_with_indicator(_universe(), bars, indicator="range_gain_pct",
                                  timeframe="4h", period=90)

    out = ub.augment_with_indicator(_universe(), bars, indicator="range_gain_pct",
                                    timeframe="4h", period=90,
                                    min_bars_multiple=1)
    assert out["range_gain_pct_4h"].notna().all()


def test_an_indicator_that_returns_nan_is_refused_rather_than_joined():
    """The exact guard that needs no heuristic, and cannot be switched off.

    A NaN that reaches the cross-section is read by ``value_below(col, 0)`` as
    "not bearish" and by ``value_above`` as "not bullish", so the instrument leaves
    both screens without appearing in either. Here the price is zeroed, which makes
    ``range_gain_pct``'s denominator unusable while the LENGTH check still passes —
    the case a length-only guard misses.
    """
    bars = _bars(n=95)
    bars.loc[bars["instrument_id"] == "BBBUSDT", ["low", "high"]] = 0.0

    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(_universe(), bars, indicator="range_gain_pct",
                                  timeframe="4h", period=90, min_bars_multiple=1)

    message = str(excinfo.value)
    assert "came out NaN" in message
    assert "BBBUSDT=95" in message
    assert "The length check passed" in message


def test_a_nan_inside_the_aggregation_window_propagates():
    """``Series.min()`` skips NaN and ``NaN < 0`` is False, so pandas' defaults
    would answer "not negative" for a window it could not read."""
    from cyqnt_trd.blocks.universe import _aggregate_indicator_window

    window = pd.Series([-1.0, float("nan"), -1.0])
    for agg in ("last", "min", "max", "mean", "any_negative", "all_negative"):
        value = _aggregate_indicator_window(window, agg, 3)
        assert value != value, "%s swallowed the NaN" % agg

    clean = pd.Series([-1.0, -1.0, 1.0])
    assert _aggregate_indicator_window(clean, "min", 3) == -1.0
    assert _aggregate_indicator_window(clean, "last", 3) == 1.0
    assert _aggregate_indicator_window(clean, "any_negative", 3) == 1.0
    assert _aggregate_indicator_window(clean, "all_negative", 3) == 0.0


# --------------------------------------------------------------------------- #
# aggregation over a window                                                   #
# --------------------------------------------------------------------------- #


def test_last_and_min_over_a_window_select_different_instruments():
    """"in the last two hours" on 15m bars is EIGHT bars, and reading only the
    final one answers a different question."""
    index = np.arange(120)
    # Bullish now, bearish five bars ago: last=+1 while min over 8 bars=-1.
    close = 100.0 * (1.0 - 0.02) ** np.minimum(index, 110)
    close[-6:] = close[-7] * (1.0 + 0.06) ** np.arange(1, 7)
    open_ = np.concatenate([[close[0]], close[:-1]])
    bars = pd.DataFrame({
        "instrument_id": "AAAUSDT", "timeframe": "15m",
        "open_time": 1_700_000_000_000 + index * 900_000,
        "close_time": 1_700_000_000_000 + (index + 1) * 900_000 - 1,
        "open": open_, "high": np.maximum(open_, close) * 1.002,
        "low": np.minimum(open_, close) * 0.998, "close": close,
        "volume": 1_000.0,
    })
    universe = _universe("AAAUSDT")

    def direction(**over):
        frame = ub.augment_with_indicator(
            universe, bars, indicator="supertrend", timeframe="15m", output=1,
            period=10, multiplier=3.0, **over)
        return float(frame["supertrend_15m"].iloc[0])

    assert direction(agg="last", window_bars=1) == 1.0
    assert direction(agg="min", window_bars=8) == -1.0
    assert direction(agg="any_negative", window_bars=8) == 1.0
    assert direction(agg="all_negative", window_bars=8) == 0.0


def test_an_unknown_agg_is_refused_with_the_closed_list():
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(_universe(), _bars(), indicator="supertrend",
                                  timeframe="4h", output=1, agg="median")

    assert "agg must be one of" in str(excinfo.value)
    assert "'last' is now" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# indicator resolution stays inside blocks.indicators                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["data.fetch_klines", "strategy.register",
                                  "_resolve_indicator", "indicators.supertrend"])
def test_a_dotted_or_private_indicator_reference_is_refused(name):
    """One namespace, and it is the right one.

    A dotted ref would make this block a SECOND dispatch surface into the whole
    blocks package — one that is not behind the interpreter's denylist, so
    ``data.fetch_klines`` would fetch during validate and ``strategy.register``
    would mutate the process-wide plugin registry. It would also invert the
    layering: a block importing the interpreter that calls it.
    """
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(_universe(), _bars(), indicator=name,
                                  timeframe="4h")

    assert "BARE name" in str(excinfo.value)


def test_a_name_imported_into_the_indicators_module_is_not_an_indicator():
    """``pd`` and ``np`` are attributes of the module and are callable-adjacent;
    the ``__module__`` check is what keeps them out."""
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(_universe(), _bars(), indicator="positive_int",
                                  timeframe="4h")

    assert "imported into" in str(excinfo.value)
    assert "cyqnt_trd.blocks._utils" in str(excinfo.value)


def test_every_indicator_the_block_can_reach_is_defined_in_indicators():
    """The reachable surface, stated as a set rather than trusted.

    If ``blocks.indicators`` ever re-exports a fetcher or a registrar, this fails
    before a spec can reach it.
    """
    from cyqnt_trd.blocks.universe import _resolve_indicator

    for name in ind.__all__:
        try:
            fn = _resolve_indicator(name)
        except ValueError:
            # The two lazy re-exports from blocks.patterns are legitimately not
            # indicators defined here; the resolver says so and that is the point.
            assert name in ("candle_lower_shadow", "candle_upper_shadow"), name
            continue
        assert fn.__module__ == "cyqnt_trd.blocks.indicators"


# --------------------------------------------------------------------------- #
# the shared output/column reduction                                          #
# --------------------------------------------------------------------------- #


def test_the_block_and_the_interpreter_share_one_answer_about_output():
    """``output: 1`` has to mean the same thing in both, or a spec validates and
    then screens the wrong column.

    ``supertrend`` returns ``(level, direction)``: index 0 is a PRICE and never
    negative, so a spec that meant the direction and got the level would find zero
    instruments below zero — an empty basket with nothing to point at.
    """
    from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import eval_indicator

    bars = _bars(n=120)
    series = bars[bars["instrument_id"] == "BBBUSDT"].reset_index(drop=True)
    raw = ind.supertrend(series, period=10, multiplier=3.0)

    via_interpreter = eval_indicator(
        series, {"block": "indicators.supertrend", "input": "df",
                 "params": {"period": 10, "multiplier": 3.0}, "output": 1})
    via_utils = select_indicator_component(raw, ref="indicators.supertrend",
                                          output=1)
    assert via_interpreter.equals(via_utils)

    level = select_indicator_component(raw, ref="indicators.supertrend", output=0)
    assert (level.dropna() > 0).all(), "index 0 is a price; only index 1 is signed"


def test_omitting_output_on_a_tuple_indicator_is_refused_in_the_block_too():
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(_universe(), _bars(), indicator="supertrend",
                                  timeframe="4h")

    assert "add 'output: <0..1>'" in str(excinfo.value)


def test_first_param_detection_is_the_same_function_for_both_callers():
    assert first_param_is_df(ind.supertrend) is True
    assert first_param_is_df(ind.ema) is False
    assert select_indicator_component(pd.Series([1.0]), ref="x").iloc[0] == 1.0
    with pytest.raises(IndicatorShapeError):
        select_indicator_component(3.0, ref="x")


def test_input_names_a_bar_column_because_a_three_month_low_is_a_low():
    """``indicators.lowest`` takes a SERIES, and the auto-detected default is
    ``close``. The low of the last 90 bars is not the lowest close."""
    bars = _bars(n=95)

    lows = ub.augment_with_indicator(_universe(), bars, indicator="lowest",
                                    input="low", timeframe="4h", period=90,
                                    min_bars_multiple=1, **{"as": "low_90"})
    closes = ub.augment_with_indicator(_universe(), bars, indicator="lowest",
                                       timeframe="4h", period=90,
                                       min_bars_multiple=1, **{"as": "close_90"})
    assert (lows["low_90"] < closes["close_90"]).all()

    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(_universe(), bars, indicator="lowest",
                                  input="lowest_price", timeframe="4h", period=90,
                                  min_bars_multiple=1)
    assert "is not a column of the bars frame" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the bars frame's own shape refusals                                         #
# --------------------------------------------------------------------------- #


def test_a_timeframe_the_capture_does_not_carry_is_refused_with_what_is_there():
    """An all-NaN column would read downstream as "no instrument matched"."""
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(_universe(), _bars(timeframe="4h"),
                                  indicator="supertrend", timeframe="15m",
                                  output=1, period=10)

    message = str(excinfo.value)
    assert "carries no '15m' bars" in message
    assert "['4h']" in message


def test_a_bars_frame_with_no_timeframe_column_is_refused():
    """Otherwise every interval is concatenated into one series and the indicator
    is computed over a mixture — with no error anywhere."""
    bars = _bars().drop(columns=["timeframe"])

    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(_universe(), bars, indicator="supertrend",
                                  timeframe="4h", output=1, period=10)

    assert "no 'timeframe' column" in str(excinfo.value)


def test_bars_are_sorted_by_open_time_not_by_arrival_order():
    """A bundle's rows are ordered by the PIT gate's sort key, and a rolling
    indicator over unordered bars returns numbers rather than an error."""
    ordered = _bars(n=120)
    shuffled = ordered.sample(frac=1.0, random_state=7).reset_index(drop=True)

    def direction(bars):
        frame = ub.augment_with_indicator(_universe(), bars,
                                         indicator="supertrend", timeframe="4h",
                                         output=1, period=10, multiplier=3.0)
        return frame.set_index("symbol")["supertrend_4h"].to_dict()

    assert direction(shuffled) == direction(ordered)


def test_an_empty_bars_frame_is_a_failed_capture_and_not_a_quiet_market():
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(_universe(), _bars().iloc[0:0],
                                  indicator="supertrend", timeframe="4h",
                                  output=1, period=10)

    assert "is empty" in str(excinfo.value)
    assert "no instrument has price history" in str(excinfo.value)


def test_an_uncovered_instrument_raises_because_the_roster_was_derived_from_it():
    """Coverage is TOTAL here, unlike every other join in the module.

    A bars roster is planned from the surviving prefix of the same pipeline, so a
    hole is never a market fact — it means the frame being screened is not the
    frame the capture was planned from.
    """
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(_universe("AAAUSDT", "BBBUSDT", "CCCUSDT"),
                                  _bars(), indicator="supertrend",
                                  timeframe="4h", output=1, period=10)

    message = str(excinfo.value)
    assert "covers only 2 of 3" in message
    assert "['CCCUSDT']" in message
    assert "DERIVED from the surviving prefix" in message


def test_two_steps_may_not_write_the_same_column():
    """The same indicator and timeframe at two periods is a normal thing to want,
    and silently keeping the second while the spec claims both is not."""
    frame = ub.augment_with_indicator(_universe(), _bars(), indicator="supertrend",
                                      timeframe="4h", output=1, period=10,
                                      **{"as": "st"})
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_indicator(frame, _bars(), indicator="supertrend",
                                  timeframe="4h", output=1, period=20,
                                  **{"as": "st"})

    assert "already has a 'st' column" in str(excinfo.value)
    assert "distinct `as:`" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# cost: the fan-out ceiling, in weight                                        #
# --------------------------------------------------------------------------- #


def test_the_measured_weight_table():
    """Weight per klines call, by ``limit`` band.

    Measured on 2026-08-02 from ``X-MBX-USED-WEIGHT-1M`` deltas over bursts of six
    identical calls. It is the input to the ceiling below, so a wrong table would
    put the refusal in the wrong place.
    """
    assert [blocks_data.kline_request_weight(limit)
            for limit in (1, 100, 101, 500, 501, 1000, 1001, 1500)] == [
        1, 1, 2, 2, 5, 5, 10, 10]
    with pytest.raises(ValueError, match="above the 1500"):
        blocks_data.kline_request_weight(1501)


def test_fanning_out_over_the_whole_venue_before_narrowing_cannot_run():
    """The ordering requirement, as arithmetic rather than as advice.

    727 instruments x 3 timeframes at limit 200 is 4362 request weight against a
    2400/min budget for the whole ``/fapi/v1`` group — 182 %. The refusal names
    both escapes (narrow first, or ask for fewer bars) because both are real.
    """
    with pytest.raises(ValueError) as excinfo:
        blocks_data.fetch_klines_cross_section(
            ["SYM%dUSDT" % n for n in range(727)],
            timeframes=["4h", "1h", "15m"], limit=200)

    message = str(excinfo.value)
    # The roster ceiling bites first at 727; both refusals point the same way.
    assert "above the 200 ceiling" in message
    assert "NOT truncated" in message

    with pytest.raises(ValueError) as excinfo:
        blocks_data.fetch_klines_cross_section(
            ["SYM%dUSDT" % n for n in range(150)],
            timeframes=["4h", "1h", "15m"], limit=200)

    message = str(excinfo.value)
    assert "costs 900 request weight, above the 600 ceiling" in message
    assert "cutting the timeframes would answer a three-timeframe" in message


def test_the_measured_plan_is_a_rounding_error_of_the_budget():
    """5 survivors x 4 timeframes at limit 100 = 20 weight, 0.8 % of a minute."""
    plan_weight = (len(RUNUP_BASKET) * len(CAPTURED_TIMEFRAMES)
                   * blocks_data.kline_request_weight(100))

    assert plan_weight == 20
    assert plan_weight < blocks_data.KLINE_FAN_OUT_MAX_WEIGHT / 10


def test_a_bare_string_timeframe_is_refused_rather_than_wrapped():
    """``timeframes="4h"`` would be iterated one character at a time, and the
    venue would reject ``"4"`` — four requests spent to learn that."""
    with pytest.raises(ValueError, match="one character at a time"):
        blocks_data.fetch_klines_cross_section(["BTCUSDT"], timeframes="4h")


# --------------------------------------------------------------------------- #
# the three-stage bridge: node, section, bundle key                           #
# --------------------------------------------------------------------------- #


def test_the_node_is_the_only_backtestable_cross_section_and_says_why():
    """Availability is the field that decides whether a frame may be back-filled.

    Every other cross-section here serves only "now" and carries a ``pit_hazard``;
    this one takes an ``endTime``, so it does not.
    """
    node = get_node(NODE)

    assert node.availability is Availability.BACKTESTABLE
    assert node.pit_hazard == ""
    for other in ("universe", "funding_snapshot", "book_ticker",
                  "open_interest_snapshot", "oi_change_snapshot"):
        assert get_node(other).availability is Availability.FORWARD_ONLY
        assert get_node(other).pit_hazard


def test_the_frame_shape_is_a_barframe_under_its_own_key():
    """Never under ``klines``: that key is special-cased into a ``MarketBundle``
    keyed on the FIRST row's instrument, so a multi-symbol frame landed there would
    file every instrument's bars under one name."""
    assert FRAME_SHAPES[NODE] == "BarFrame@1.0"
    assert FRAME_SHAPES[NODE] == FRAME_SHAPES["klines"]
    assert NODE != "klines"

    snapshot = load_input_bundle(_fixture())
    assert NODE in snapshot.frames
    bars = snapshot.frames[NODE]
    assert set(bars["instrument_id"]) == {symbol for _r, symbol, _g in RUNUP_BASKET}
    assert set(bars["timeframe"]) == set(CAPTURED_TIMEFRAMES)


def test_the_spec_drives_the_section_plan_and_the_required_nodes():
    section, node, key = BARS_SECTION
    spec = load_spec(str(SPEC_RESONANCE))

    assert SECTION_NODES[section] == (node,)
    assert node == key == NODE
    assert NODE in required_bundle_nodes(spec)
    sections = live_sections_for_spec(spec)
    assert section in sections
    # Bars last: their roster is what survives every step above, including the
    # derivative filters.
    assert sections[-1] == section


def test_a_step_that_omits_with_universe_bars_is_refused_statically():
    """The block fetches when the source is absent, so validate would hit the
    network and a backtest would read live data."""
    assert FETCHES_WITHOUT_SOURCE[BARS_BLOCK] == NODE

    spec = load_spec(str(SPEC_RUNUP))
    for step in spec["selection"]["universe"]:
        if step["block"] == BARS_BLOCK:
            step.pop("with")
    errors, _warnings = validate_spec(spec)

    assert any("with: [universe_bars]" in error for error in errors), errors


def test_an_empty_bars_frame_in_the_bundle_fails_the_run_by_name():
    """"I could not read it" must not arrive as an empty basket.

    A capture whose ``end_ms`` was never set has every row dropped by the gate and
    lands an empty frame — which read as "no instrument has price history" would
    empty the basket with nothing to point at.
    """
    bundle = _fixture()
    bundle["frames"][NODE]["rows"] = []

    with pytest.raises(BundleRunError) as excinfo:
        run_bundle(str(SPEC_RUNUP), bundle)

    assert NODE in str(excinfo.value)
    assert "required input source unavailable" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# two-pass capture                                                            #
# --------------------------------------------------------------------------- #


def test_pass_one_is_the_prefix_before_the_first_indicator_step():
    """And it is necessarily cross-sectional, which is why Pass 1 is free."""
    spec = load_spec(str(SPEC_RESONANCE))
    prefix = universe_steps_before_bars(spec)
    all_steps = spec["selection"]["universe"]

    assert len(prefix) == 7
    assert prefix == all_steps[:7]
    assert all(step["block"] != BARS_BLOCK for step in prefix)
    # Everything the prefix reads is already in the bundle: no node it names is a
    # fan-out that Pass 1 would have to pay for.
    assert {name for step in prefix for name in step.get("with", ())} <= set(
        _fixture()["frames"])


def test_the_timeframe_set_is_the_union_of_every_indicator_step():
    """Collecting only the first would leave two columns NaN, which reads
    downstream as "no instrument matched" rather than as a short capture."""
    assert bar_timeframes_for_spec(load_spec(str(SPEC_RESONANCE))) == [
        "4h", "1h", "15m"]
    assert bar_timeframes_for_spec(load_spec(str(SPEC_RUNUP))) == ["1d"]
    assert bar_timeframes_for_spec({}) == []


def test_an_indicator_step_with_no_timeframe_is_refused_while_planning():
    spec = load_spec(str(SPEC_RUNUP))
    for step in spec["selection"]["universe"]:
        if step["block"] == BARS_BLOCK:
            step["params"].pop("timeframe")

    with pytest.raises(SpecError, match="declares no `timeframe:`"):
        bar_timeframes_for_spec(spec)


def test_pass_one_narrowing_to_nothing_is_a_refusal_not_an_empty_capture():
    spec = load_spec(str(SPEC_RUNUP))
    for step in spec["selection"]["universe"]:
        if step["block"] == "universe.filter_open_interest":
            step["params"]["min_notional_usd"] = 1e15

    with pytest.raises(BundleRunError) as excinfo:
        plan_bars_capture(spec, _fixture())

    assert "left no instrument" in str(excinfo.value)
    assert "read by the joining block as a failed capture" in str(excinfo.value)


def test_pass_one_needs_a_universe_frame_to_narrow():
    bundle = _fixture()
    bundle["frames"].pop("universe")

    with pytest.raises(BundleRunError, match="no universe frame"):
        plan_bars_capture(load_spec(str(SPEC_RUNUP)), bundle)


def test_pass_one_runs_the_specs_own_steps_and_not_a_copy_of_them():
    """The roster and the run's own funnel are the same computation.

    A capture that re-implemented "apply the narrowing filters" would derive a
    slightly different roster, and the difference surfaces as a coverage refusal
    inside the join — pointing at the one place that is not at fault.
    """
    plan = plan_bars_capture(load_spec(str(SPEC_RUNUP)), _fixture())
    survivors = sorted(_survivors()["instrument_id"].str.upper())

    assert plan.symbols == survivors
    assert set(_frame(NODE)["instrument_id"]) == set(survivors)


# --------------------------------------------------------------------------- #
# the fixture, and what it must not have disturbed                            #
# --------------------------------------------------------------------------- #


def test_the_frames_the_earlier_stage_pinned_are_byte_identical():
    """Adding bars must not move the E5 funnel or any golden basket.

    Hashing each frame ENTRY rather than the file, because the file necessarily
    changed: the claim is that the five frames the derivatives stage pinned did
    not.
    """
    import hashlib

    frames = _fixture()["frames"]
    for key, expected in FRAMES_PINNED_EARLIER.items():
        blob = json.dumps(frames[key], sort_keys=True,
                          separators=(",", ":")).encode()
        assert hashlib.md5(blob).hexdigest() == expected, key

    assert _fixture()["decision_time"] == DECISION_TIME
    assert set(frames) == set(FRAMES_PINNED_EARLIER) | {NODE}


def test_the_earlier_frozen_cross_section_still_did_not_move():
    """The bundle three other modules pin, untouched by this stage as well."""
    earlier = json.loads(
        (FIXTURES / "universe_cross_section.json").read_text(encoding="utf-8"))

    assert earlier["decision_time"] == 1_785_591_229_856
    assert set(earlier["frames"]) == {"universe", "ticker_rank", "funding",
                                     "contract_meta"}
    assert len(earlier["frames"]["universe"]["rows"]) == 727


def test_the_freeze_script_refuses_a_bars_addition_that_is_not_point_in_time():
    """The claim in ``ADDABLE_NODES`` is CHECKED, which is what allows it.

    Back-filling is permitted here because ``end_ms`` makes the frame replayable.
    If the PIT gate did not run, a surviving future bar is exactly the fabrication
    the allowlist exists to prevent — so the freeze script refuses rather than
    trusting its own request.
    """
    module = _freeze_script()
    rows = _frame(NODE).to_dict("records")
    rows[0]["close_time"] = DECISION_TIME + 1

    with pytest.raises(module.FreezeError) as excinfo:
        module._verify_bars_addition(_fixture(), rows)

    assert "close AFTER the target's decision_time" in str(excinfo.value)
    assert "the gate did not run" in str(excinfo.value)


def test_the_freeze_script_refuses_a_single_series_bars_addition():
    """One (instrument, timeframe) pair lets a resonance test pass by computing
    one column three times."""
    module = _freeze_script()
    rows = _frame(NODE)
    one = rows[(rows["instrument_id"] == "USUSDT")
               & (rows["timeframe"] == "1d")].to_dict("records")

    with pytest.raises(module.FreezeError) as excinfo:
        module._verify_bars_addition(_fixture(), one)

    assert "only 1 (instrument, timeframe) pair" in str(excinfo.value)


def _freeze_script():
    """Import ``scripts/freeze_selection_fixture.py`` as a module."""
    import importlib.util

    path = REPO / "scripts" / "freeze_selection_fixture.py"
    spec = importlib.util.spec_from_file_location("_freeze_bars", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# the synthetic dry-run frame                                                 #
# --------------------------------------------------------------------------- #


def test_both_specs_validate_without_touching_the_network():
    """``validate`` compiles and RUNS the selection on stand-in data, so a spec
    that validates is structurally runnable."""
    for path in (SPEC_RUNUP, SPEC_RESONANCE):
        errors, warnings = validate_spec(load_spec(str(path)))
        assert errors == [], (path.name, errors)
        assert warnings == [], (path.name, warnings)


def test_the_dry_run_produces_candidates_and_drops_a_row_for_the_right_reason():
    """The trap this guards: "produced no candidates" is a WARNING, so a stand-in
    on which nothing can match degrades the dry-run to a no-op that nobody sees.

    Both engineered instruments have to be present and on opposite sides.
    """
    from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import build_selection_fn
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import (
        _synthetic_contract_meta,
        _synthetic_oi_history,
        _synthetic_open_interest,
        _synthetic_ticker_rank,
    )

    universe = _synthetic_universe()
    for path, kept, dropped in ((SPEC_RESONANCE, "TAGUSDT", "USUSDT"),
                                (SPEC_RUNUP, "TAGUSDT", "USUSDT")):
        spec = load_spec(str(path))
        frames = {
            "contract_meta": _synthetic_contract_meta(universe),
            "open_interest_snapshot": _synthetic_open_interest(universe),
            "oi_change_snapshot": _synthetic_oi_history(universe),
            NODE: _synthetic_universe_bars(universe, spec),
        }
        candidates = build_selection_fn(spec)(
            universe, _synthetic_ticker_rank(universe), frames=frames)
        assert [c["symbol"] for c in candidates] == [kept], path.name
        assert dropped not in {c["symbol"] for c in candidates}


def test_the_stand_in_bars_carry_exactly_a_real_captures_columns():
    """One extra column here makes ``validate`` strictly more permissive than every
    real run — the scar ``_synthetic_universe`` records for ``symbol`` and
    ``quote_volume``."""
    spec = load_spec(str(SPEC_RESONANCE))
    stand_in = _synthetic_universe_bars(_synthetic_universe(), spec)
    real = _frame(NODE)

    assert set(stand_in.columns) <= set(real.columns), (
        "the stand-in offers %s, which a real universe_bars frame does not have"
        % sorted(set(stand_in.columns) - set(real.columns)))
    # And it carries every column any indicator reads off the frame.
    assert {"open", "high", "low", "close", "volume"} <= set(stand_in.columns)


def test_the_stand_in_timeframes_come_from_the_spec():
    universe = _synthetic_universe()

    assert sorted(set(_synthetic_universe_bars(
        universe, load_spec(str(SPEC_RESONANCE)))["timeframe"])) == sorted(
            ["4h", "1h", "15m"])
    assert sorted(set(_synthetic_universe_bars(
        universe, load_spec(str(SPEC_RUNUP)))["timeframe"])) == ["1d"]
    assert _synthetic_universe_bars(universe, {}) is None


def test_most_stand_in_instruments_disagree_across_timeframes():
    """Otherwise ``all_of`` and ``any_of`` are the same screen on the stand-in, and
    a spec that named one timeframe three times would dry-run as if it were right.
    """
    universe = _synthetic_universe()
    bars = _synthetic_universe_bars(universe, load_spec(str(SPEC_RESONANCE)))
    frame = universe
    for timeframe in ("4h", "1h", "15m"):
        frame = ub.augment_with_indicator(
            frame, bars, indicator="supertrend", timeframe=timeframe, output=1,
            period=10, multiplier=3.0, **{"as": "d_" + timeframe})

    columns = ["d_4h", "d_1h", "d_15m"]
    agree = frame[columns].nunique(axis=1) == 1
    assert int((~agree).sum()) >= 10, "the timeframes are near-duplicates"
    # And the two engineered rows DO agree, or the resonance dry-run has no row it
    # can keep — the case that turns validate into a no-op.
    assert set(frame.loc[agree, "symbol"]) == {"TAGUSDT", "USUSDT"}


def test_the_stand_in_bars_cover_the_whole_stand_in_universe():
    """Unlike the three derivative fan-outs, whose roster is deliberately partial.

    A bars roster is derived from the surviving prefix of the same pipeline, so it
    covers the frame by construction; a partial stand-in would fail specs for a
    reason no real capture can produce.
    """
    universe = _synthetic_universe()
    bars = _synthetic_universe_bars(universe, load_spec(str(SPEC_RUNUP)))

    assert set(bars["instrument_id"]) == set(universe["instrument_id"])


def test_range_gain_pct_is_nan_before_its_window_fills():
    """A range measured over 12 of the 90 bars asked for is a smaller range
    reported under the wider name, which reads as "this coin was quiet"."""
    index = np.arange(30)
    frame = pd.DataFrame({"high": 100.0 + index, "low": 90.0 + index,
                          "close": 95.0 + index, "open": 95.0 + index,
                          "volume": 1.0})
    out = ind.range_gain_pct(frame, period=20)

    assert out.iloc[:19].isna().all()
    assert out.iloc[19] == pytest.approx((119.0 - 90.0) / 90.0 * 100.0)
    with pytest.raises(ValueError):
        ind.range_gain_pct(frame, period=0)


def test_range_gain_pct_refuses_a_non_positive_denominator():
    """An infinity would sort to the top of a "biggest gainer" ranking."""
    index = np.arange(30)
    frame = pd.DataFrame({"high": 100.0 + index, "low": 0.0,
                          "close": 95.0 + index, "open": 95.0 + index,
                          "volume": 1.0})

    assert ind.range_gain_pct(frame, period=20).isna().all()
