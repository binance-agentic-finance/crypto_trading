"""One frozen cross-section, one basket: the selection specs' golden file.

Why a fixture at all
-------------------
``yaml_pipeline run <selection spec>`` fetched the universe live, so its basket
changed every minute and there was nothing to regress against — a different
basket could mean the code broke or that BNB simply out-traded SOL that hour, and
no test could tell the two apart. But a selection decision reads no clock: the
basket's ``signal_id`` is
``uuid5(snapshot_id | plugin_id | "selection" | as_of)``, all four of which come
out of the input bundle. So freezing the *input* makes the *output* a golden file,
and ``test_replay_is_byte_identical_including_the_signal_id`` is the assertion
that earns the fixture its 570 KB.

The fixture is market data only (24h tickers, Square ticker ranks, funding
snapshot, contract registry) — public, no account state — and is produced by
``scripts/freeze_selection_fixture.py``. Recapture with::

    python scripts/freeze_selection_fixture.py

and expect the golden baskets below to change: they are a real market at a real
instant, not synthetic rows. Update them in the same commit as the fixture.

``contract_meta`` arrived later and by a different route: ``--add-frame
contract_meta`` collected that one node and left every other frame and the
``decision_time`` byte-identical, so the baskets below did NOT move. That route
exists for exactly this — see ``ADDABLE_NODES`` in the freeze script for which
nodes may take it and why a price snapshot may not.
"""

from __future__ import annotations

import copy
import json
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from cyqnt_trd.blocks import data as blocks_data
from cyqnt_trd.blocks import universe as universe_blocks
from cyqnt_trd.blocks.news_feed import base_token
from cyqnt_trd.data_cli import _subprocess as data_cli_subprocess
from cyqnt_trd.standard_bot.data.input_bundle import build_input_bundle
from cyqnt_trd.standard_bot.entrypoints import mvp_input_bundle
from cyqnt_trd.standard_bot.yaml_pipeline import cli as yaml_cli
from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import run_bundle
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import build_selection_fn
from cyqnt_trd.standard_bot.yaml_pipeline.spec import _synthetic_universe, load_spec

REPO = Path(__file__).parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "universe_cross_section.json"
SPEC_NEWS = REPO / "docs" / "strategy_yaml_spec" / "example_selection.yaml"
SPEC_USER_CHAT = REPO / "docs" / "strategy_yaml_spec" / "example_from_user_chat.yaml"

FROZEN_FRAMES = {"universe", "ticker_rank", "funding", "contract_meta"}

# ---------------------------------------------------------------------------
# The golden baskets. They live HERE, beside the assertion, and not in a sidecar
# JSON: a reviewer has to be able to see WHICH symbols this fixture is supposed
# to rank without opening a second file, and a diff on this list is the whole
# point of the exercise.
#
# example_selection.yaml — liquidity floor $100m, then ranked by Square mention
# count, long above bull_ratio 0.55 / short below 0.45. Directional screening
# happens before top_k, so five qualifying rows fill the declared five slots;
# neutral rows cannot consume a slot and silently push out a lower-ranked but
# valid long/short candidate.
NEWS_BUZZ_BASKET = [
    (1, "BNBUSDT", "long"),
    (2, "BTCUSDT", "long"),
    (3, "GIGGLEUSDT", "long"),
    (4, "KOMAUSDT", "long"),
    (5, "DOGEUSDT", "long"),
]

# example_from_user_chat.yaml — the 30 biggest 24h losers over a $2m floor,
# ranked by turnover. Every one of these five is a TradFi perpetual, which is the
# category the originating user asked to EXCLUDE; the spec's own comment says so
# and calls it gap #3. That is a vocabulary gap, not a pipeline bug, and pinning
# the wrong-but-correct basket here is what will show the fix landing.
USER_CHAT_BASKET = [
    (1, "SNDKUSDT", "short"),
    (2, "SOXLUSDT", "short"),
    (3, "MUUSDT", "short"),
    (4, "SKHYUSDT", "short"),
    (5, "KORUUSDT", "short"),
]


SPECS = [
    pytest.param(SPEC_NEWS, NEWS_BUZZ_BASKET, id="example_selection"),
    pytest.param(SPEC_USER_CHAT, USER_CHAT_BASKET, id="example_from_user_chat"),
]

# The eight majors ``example_from_user_chat.yaml`` excludes, in its own order.
USER_CHAT_EXCLUDED = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
                      "BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC"]


def _fixture() -> dict:
    """A fresh parse per call — a test must not inherit another's mutations."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _basket(output: dict) -> list:
    return [(item["rank"], item["symbol"], item["direction"])
            for item in output["signals"][0]["candidates"]]


# --------------------------------------------------------------------------- #
# the fixture itself                                                          #
# --------------------------------------------------------------------------- #


def test_fixture_is_a_market_only_cross_sectional_input_bundle():
    bundle = _fixture()
    assert bundle["schema"] == "cyqnt.input/v1"
    assert set(bundle["frames"]) == FROZEN_FRAMES
    # A status line for a frame that is not here would be unfalsifiable: it is
    # copied onto every emitted signal, where a reader cannot check it against
    # anything. The freeze script drops the two together.
    assert set(bundle["source_status"]) == FROZEN_FRAMES
    # Committed to a public repo: market data only, never an account snapshot.
    assert bundle["positions"] == {}
    assert bundle["equity"] is None

    universe_rows = bundle["frames"]["universe"]["rows"]
    funding_rows = bundle["frames"]["funding"]["rows"]
    # The live collector does not truncate (see the max_event_rows test below),
    # and a 200-row universe would silently be a different, smaller market.
    assert len(universe_rows) > 200
    symbols = {row["instrument_id"] for row in universe_rows}
    covered = {row["instrument_id"] for row in funding_rows} & symbols
    # Under 2 and the runtime refuses the bundle as "not a cross-section".
    assert len(covered) >= 2


# --------------------------------------------------------------------------- #
# the decision                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec_path, golden", SPECS)
def test_selection_spec_emits_one_basket_on_the_frozen_cross_section(
    spec_path, golden
):
    spec = load_spec(str(spec_path))
    bundle = _fixture()
    output = run_bundle(str(spec_path), bundle)

    assert output["signal_count"] == 1
    signal = output["signals"][0]
    assert signal["schema"] == "cyqnt.signal/v2"
    assert signal["kind"] == "selection"
    assert signal["universe_size"] == len(bundle["frames"]["universe"]["rows"])

    top_k = int(spec["selection"]["top_k"])
    candidates = signal["candidates"]
    # top_k is a ceiling, not a quota: a spec whose thresholds only 3 symbols
    # clear must return 3, never 5 padded with rows that failed the filter.
    assert 0 < len(candidates) <= top_k
    assert _basket(output) == golden

    # dedupe_by: base_asset — top_k distinct bets, not top_k rows.
    bases = [item.get("base_asset") or item["symbol"] for item in candidates]
    assert len(set(bases)) == len(bases)


def test_the_candidate_carries_no_second_copy_of_its_own_instrument():
    """``features`` is every frame column except the one used as the instrument.

    So a universe block that left a derived ``symbol`` behind changed this dict:
    the interpreter then keyed the candidate on ``symbol`` and ``instrument_id``
    fell through into ``features`` as a duplicate of it. Nothing failed — the
    basket was right — but the emitted contract differed depending on which steps
    a spec happened to list, which is a difference no reader could account for.
    """
    features = run_bundle(str(SPEC_USER_CHAT),
                          _fixture())["signals"][0]["candidates"][0]["features"]

    assert "symbol" not in features
    assert "instrument_id" not in features
    assert "quoteVolume" in features, features


def test_the_exclusion_step_is_load_bearing_even_where_the_basket_cannot_show_it():
    """What ``example_from_user_chat.yaml``'s golden basket does NOT prove.

    All eight excluded majors are in this cross-section, but none of them is among
    its 30 biggest losers, so the basket above is identical with and without the
    ``exclude_symbols`` step. The golden list therefore pins "the spec runs on a
    bundle at all" — which is what was broken — and nothing about the exclusion.
    Assert that separately, and against the frozen frame's own vocabulary
    (``instrument_id``), which is the shape the block used to refuse.
    """
    universe = pd.DataFrame(_fixture()["frames"]["universe"]["rows"])
    assert set(USER_CHAT_EXCLUDED) <= set(universe["instrument_id"]), (
        "recapture left the fixture without these majors; the exclusion below "
        "would then be vacuous")

    kept = universe_blocks.exclude_symbols(universe, USER_CHAT_EXCLUDED)

    assert set(kept["instrument_id"]).isdisjoint(USER_CHAT_EXCLUDED)
    assert len(kept) == len(universe) - len(USER_CHAT_EXCLUDED)
    # A filter narrows rows and hands the frame back as it received it. Widening
    # it with a derived ``symbol`` is what made the same steps in a different
    # order behave differently.
    assert list(kept.columns) == list(universe.columns)


def test_the_exclusion_reaches_the_basket_when_the_ranking_would_have_chosen_them():
    """The end-to-end half: rank the same universe by turnover with no loser
    screen, where BTCUSDT and ETHUSDT are #1 and #2 by a wide margin."""
    spec = load_spec(str(SPEC_USER_CHAT))
    spec["selection"]["universe"] = [
        step for step in spec["selection"]["universe"]
        if step["block"] != "universe.top_losers"
    ]
    del spec["selection"]["short_when"]
    spec["selection"]["long_when"] = {"cond": "conditions.value_above",
                                      "args": ["quoteVolume", 0.0]}

    basket = [item["symbol"] for item in
              build_selection_fn(spec)(
                  pd.DataFrame(_fixture()["frames"]["universe"]["rows"]), None)]

    assert basket, "the variant spec produced nothing to assert on"
    assert set(basket).isdisjoint(USER_CHAT_EXCLUDED), basket


@pytest.mark.parametrize("spec_path, _golden", SPECS)
def test_replay_is_byte_identical_including_the_signal_id(spec_path, _golden):
    """The reason the fixture exists.

    Two runs over the same frozen input must serialise to the same bytes —
    ``signal_id`` and ``batch``-level provenance included. If any of that were
    derived from the wall clock, no golden-file assertion above could hold, and
    the only way to find out is to compare the whole contract rather than the
    handful of fields a test happens to name.
    """
    first = run_bundle(str(spec_path), _fixture())
    second = run_bundle(str(spec_path), _fixture())

    assert first["signals"][0]["signal_id"]
    assert (json.dumps(first, sort_keys=True)
            == json.dumps(second, sort_keys=True))


def test_replay_refuses_a_candidate_not_known_at_the_decision_time():
    """A hand-edited replay cannot smuggle a future universe row past the gate.

    The selected symbols are only a subset of the frozen cross-section.  Mutating
    one candidate before `run_bundle` proves the ingress gate checks the input
    artifact itself, rather than relying on whether a later selection filter
    happens to touch that row.
    """
    bundle = copy.deepcopy(_fixture())
    candidate = next(
        row for row in bundle["frames"]["universe"]["rows"]
        if row["instrument_id"] == "BNBUSDT"
    )
    candidate["available_time"] = bundle["decision_time"] + 1

    with pytest.raises(
        ValueError,
        match=r"frame 'universe' row \d+ available_time=.*after decision_time",
    ):
        run_bundle(str(SPEC_NEWS), bundle)


# --------------------------------------------------------------------------- #
# offline                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def no_network(monkeypatch):
    """Make every data transport record-then-raise, and hand back the log.

    Recording as well as raising is deliberate: ``data_cli.rest_source`` catches
    ``OSError``/``ValueError`` around its ``urlopen`` and returns ``None``, so a
    transport that only raised would be swallowed there and the test would pass
    while a request went out. The assertion is on the log, not on the absence of
    an exception.
    """
    calls = []

    def _blocked(name):
        def deny(*args, **kwargs):
            calls.append(name)
            raise AssertionError("replay must not fetch: %s was called" % name)
        return deny

    # Binance public REST (universe 24h tickers, premiumIndex funding snapshot).
    monkeypatch.setattr(blocks_data, "_request_json",
                        _blocked("blocks.data._request_json"))
    # The single chokepoint under run_binance_cli / run_binance_pro_cli.
    monkeypatch.setattr(data_cli_subprocess, "_run",
                        _blocked("data_cli._subprocess._run"))
    # The Square / internal REST transport behind data_cli.rest_source.
    monkeypatch.setattr(urllib.request, "urlopen",
                        _blocked("urllib.request.urlopen"))
    return calls


def test_replaying_the_fixture_touches_no_network(no_network):
    output = run_bundle(str(SPEC_NEWS), _fixture())

    assert _basket(output) == NEWS_BUZZ_BASKET
    assert no_network == []


def test_cli_input_json_routes_a_selection_spec_through_the_bundle_runner(
    no_network, tmp_path, capsys
):
    """``run --input-json <bundle>`` on a selection spec.

    ``cmd_run`` detects ``schema == "cyqnt.input/v1"`` and short-circuits to
    ``run_bundle`` before it ever looks at ``run.mode``. Without that branch a
    selection spec fell through to ``_run_selection``, which collects LIVE — so
    ``--input-json`` was quietly ignored and the "offline" run hit the network and
    ranked a different market.
    """
    out_path = tmp_path / "batch.json"
    code = yaml_cli.main(["run", str(SPEC_NEWS),
                          "--input-json", str(FIXTURE),
                          "--output-json", str(out_path)])
    assert code == 0
    assert no_network == []

    expected = run_bundle(str(SPEC_NEWS), _fixture())
    printed = capsys.readouterr().out
    start = printed.index("{")
    assert json.loads(printed[start:]) == expected
    assert json.loads(out_path.read_text(encoding="utf-8")) == expected


def test_mvp_input_bundle_replay_agrees_with_the_cli_path(no_network, tmp_path):
    """The two documented replay routes must not be two different strategies."""
    out_path = tmp_path / "batch.json"
    code = mvp_input_bundle.main(["--replay", str(FIXTURE),
                                 "--strategy-yaml", str(SPEC_NEWS),
                                 "--signal-out", str(out_path)])
    assert code == 0
    assert no_network == []
    assert (json.loads(out_path.read_text(encoding="utf-8"))
            == run_bundle(str(SPEC_NEWS), _fixture()))


# --------------------------------------------------------------------------- #
# the truncation the frozen path does NOT apply                               #
# --------------------------------------------------------------------------- #


SYNTHETIC_DT = 1_785_591_229_856


def _wide_universe(rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "instrument_id": ["SYM%04dUSDT" % index for index in range(rows)],
        "quoteVolume": [1e9 - index for index in range(rows)],
        "available_time": [SYNTHETIC_DT] * rows,
    })


def test_build_input_bundle_caps_event_rows_at_200_unless_told_otherwise():
    """Pin ``max_event_rows=200``, because the two paths disagree on purpose.

    ``build_input_bundle`` (hand-built, from files) keeps only the newest 200 rows
    of a universe/news/rank frame; ``build_live_bundle`` (what the fixture was
    captured with) keeps everything. A test that hand-builds a cross-section and
    forgets ``max_event_rows=None`` therefore ranks the tail of the market and
    still passes, which is why the default is pinned here rather than left to be
    rediscovered.
    """
    frame = _wide_universe(700)

    capped = build_input_bundle(symbol="BTCUSDT", interval="1h",
                               decision_time=SYNTHETIC_DT, universe_frame=frame)
    assert len(capped["frames"]["universe"]["rows"]) == 200

    whole = build_input_bundle(symbol="BTCUSDT", interval="1h",
                              decision_time=SYNTHETIC_DT, universe_frame=frame,
                              max_event_rows=None)
    assert len(whole["frames"]["universe"]["rows"]) == 700

    # And the frozen fixture came through the live path, which does not cap.
    assert len(_fixture()["frames"]["universe"]["rows"]) > 200


# --------------------------------------------------------------------------- #
# the stand-in universe validate uses, measured against this real one          #
# --------------------------------------------------------------------------- #


def test_the_dry_run_universe_offers_exactly_the_columns_a_real_one_has():
    """``validate`` must not be more permissive than ``run``.

    ``spec._synthetic_universe`` is the frame the selection dry-run ranks, so every
    column name it offers is a name a spec may reference and still validate. It
    offered two the real cross-section does not have — ``symbol`` and
    ``quote_volume`` — and each bought a spec a green ``validate`` followed by a
    hard failure on every ``cyqnt.input/v1`` bundle, live or replayed:
    ``score: quote_volume`` raised ``cannot resolve reference``, and
    ``universe.exclude_symbols`` raised ``missing 'symbol' column``.

    Set equality in BOTH directions, against captured market data rather than a
    hand-copied list: a missing name is the opposite failure — the block that
    needs it cannot be validated at all, which reads as a bug in the author's
    spec (that is how ``priceChangePercent`` came to be in the stand-in).
    """
    real = set(pd.DataFrame(_fixture()["frames"]["universe"]["rows"]).columns)

    assert set(_synthetic_universe().columns) == real


def test_the_dry_run_universe_lists_more_than_one_quote_per_token():
    """Because the real one does, and two blocks exist only for that situation.

    ``dedupe_by: base_asset`` and ``universe.filter_quote_suffix`` are both about
    a token listed against several quotes. A stand-in of USDT-only rows exercised
    neither, so "exclude the USDC pairs" dry-ran as a filter matching no row —
    indistinguishable from the typo ``USDCC``, which is the one thing validate is
    there to catch.
    """
    real = pd.DataFrame(_fixture()["frames"]["universe"]["rows"])
    assert real["instrument_id"].str.endswith("USDC").any(), (
        "the captured market has no USDC pair; the stand-in should not invent one")

    synthetic = _synthetic_universe()["instrument_id"]
    assert synthetic.str.endswith("USDC").any()
    assert synthetic.str.endswith("USDT").any()
    bases = synthetic.map(base_token)
    assert len(set(bases)) < len(bases), "no token is listed twice"


def test_every_symbol_keyed_universe_block_reads_the_real_frame_as_it_arrives():
    """One vocabulary rule for the whole module, checked on the real frame.

    A universe out of a bundle is keyed on ``instrument_id``; only some of these
    blocks tolerated that, so a spec's steps had to be ordered so that one of the
    tolerant ones ran first and quietly injected ``symbol`` for the others. Both
    orders validated, one of them raised, and the difference was invisible in the
    spec. Every block reads either key now, and none of them leaves the frame in a
    vocabulary its caller did not pass in.
    """
    universe = pd.DataFrame(_fixture()["frames"]["universe"]["rows"])
    calls = {
        "filter_quote_volume": lambda frame: universe_blocks.filter_quote_volume(
            frame, 2e6),
        "filter_change_pct": lambda frame: universe_blocks.filter_change_pct(
            frame, 50.0),
        "top_gainers": lambda frame: universe_blocks.top_gainers(frame, 5),
        "top_losers": lambda frame: universe_blocks.top_losers(frame, 5),
        "exclude_symbols": lambda frame: universe_blocks.exclude_symbols(
            frame, USER_CHAT_EXCLUDED),
        "only_symbols": lambda frame: universe_blocks.only_symbols(
            frame, USER_CHAT_EXCLUDED),
        "filter_quote_suffix": lambda frame: universe_blocks.filter_quote_suffix(
            frame, "USDC", exclude=True),
    }

    for name, call in calls.items():
        out = call(universe)
        assert len(out), "%s emptied the real cross-section" % name
        assert list(out.columns) == list(universe.columns), name


def test_the_two_exclusions_commute_on_the_real_frame():
    """The order dependence the injected column created, stated as an equality."""
    universe = pd.DataFrame(_fixture()["frames"]["universe"]["rows"])

    suffix_first = universe_blocks.exclude_symbols(
        universe_blocks.filter_quote_suffix(universe, "USDC", exclude=True),
        USER_CHAT_EXCLUDED)
    exclude_first = universe_blocks.filter_quote_suffix(
        universe_blocks.exclude_symbols(universe, USER_CHAT_EXCLUDED),
        "USDC", exclude=True)

    pd.testing.assert_frame_equal(suffix_first, exclude_first)
