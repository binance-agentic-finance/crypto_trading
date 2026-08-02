"""Contract metadata: what an instrument IS, joined onto a cross-section.

The gap this closes
-------------------
A 24h ticker cross-section cannot tell ``BTCUSDT`` from ``SNDKUSDT``, which is a
tokenised SanDisk perpetual. On the frozen Binance snapshot this repo tests
against, 21 of the 30 biggest 24h losers above a $2m turnover floor are stock /
ETF / index perpetuals, and because tokenised equities out-trade mid-cap coins by
one to two orders of magnitude, the five biggest by turnover are **all** TradFi.
``docs/strategy_yaml_spec/example_from_user_chat.yaml`` asks for short candidates
"excluding TradFi" and its golden basket is those exact five: the pipeline was
right and the vocabulary could not say it.

Every number above is asserted below against the committed fixture rather than
quoted, because an assertion that cannot fail is the failure mode this file is
guarding against.

What is worth testing here, and why
-----------------------------------
Three things break silently and only in production, so each gets a test that
fails loudly if the guard is removed:

* ``underlyingSubType`` is a JSON **array**. A 2-element array reaches
  ``interpreter.selection_fn``'s ``if pd.notna(value)`` and raises "truth value of
  an array with more than one element is ambiguous"; a 1-element array does not.
  722 of 727 live contracts carry one tag, so this appears only for a multi-tag
  coin, only after the filters have run.
* "exclude TradFi" has three encodings that give three different answers (577 /
  577 / 575 contracts). Nothing in a basket reveals which one was used.
* A partial join hands the filters NaN metadata, and a NaN read as "no sector"
  lets an un-joined equity perpetual through ``exclude=['TradFi']``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from cyqnt_trd.blocks import data as blocks_data
from cyqnt_trd.blocks import universe as ub
from cyqnt_trd.standard_bot.data.catalog import Availability, FrameKind, get_node
from cyqnt_trd.standard_bot.data.live_snapshot import (
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
    FETCHES_WITHOUT_SOURCE,
    SpecError,
    build_selection_fn,
)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import load_spec, validate_spec

REPO = Path(__file__).parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "universe_cross_section.json"
SPEC_USER_CHAT = REPO / "docs" / "strategy_yaml_spec" / "example_from_user_chat.yaml"

META_COLUMNS = ["contract_type", "underlying_type", "underlying_sub_type",
                "base_asset", "quote_asset", "contract_status"]


def _fixture() -> dict:
    """A fresh parse per call — a test must not inherit another's mutations."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _frozen(name: str) -> pd.DataFrame:
    return pd.DataFrame(_fixture()["frames"][name]["rows"])


# --------------------------------------------------------------------------- #
# a hand-built cross-section                                                  #
# --------------------------------------------------------------------------- #
#
# Hand-built and not sampled, because the assertions below need a shape the
# market only sometimes has: a multi-tag coin, an untagged coin, an INDEX, and a
# TradFi row that WINS the turnover ranking. The frozen fixture happens to
# provide all four today; a recapture might not, and a test that quietly stops
# discriminating is worse than one that fails.
#
# The x100 turnover on the TradFi rows is the whole point: it reproduces the real
# distortion (tokenised equities dominate a liquidity ranking), so "rank by
# turnover" puts them first and any filter that fails to drop them is visible in
# the basket rather than only in a row count.

_ROWS = [
    # symbol,        underlying_type, tags,                 turnover, chg%
    ("BTCUSDT",      "COIN",          ["PoW"],              5.0e8,   -3.1),
    ("ETHUSDT",      "COIN",          ["Layer-1"],          4.0e8,   -4.2),
    ("UNIUSDT",      "COIN",          ["DeFi"],             1.0e8,   -5.7),
    ("CLANKERUSDT",  "COIN",          ["Alpha", "AI"],      3.0e7,  -12.5),
    ("FOLKSUSDT",    "COIN",          ["Alpha", "DeFi"],    2.0e7,   -9.9),
    ("CAPUSDT",      "COIN",          [],                   4.0e7,  -17.2),
    ("ALLUSDT",      "INDEX",         [],                   6.0e7,   -6.6),
    ("BTCDOMUSDT",   "INDEX",         [],                   5.0e7,   -2.5),
    ("SNDKUSDT",     "EQUITY",        ["TradFi"],           5.0e10, -11.7),
    ("SOXLUSDT",     "EQUITY",        ["TradFi", "ETF"],    4.0e10, -14.3),
    ("SKHYUSDT",     "KR_EQUITY",     ["TradFi"],           3.0e10,  -9.9),
    ("XAUTUSDT",     "COMMODITY",     ["TradFi"],           2.0e10,  -1.2),
]

#: ``underlying_type`` -> ``contractType``, the way the live venue assigns it:
#: TRADIFI_PERPETUAL tracks the underlying being off-chain, and COIN / INDEX
#: contracts are plain PERPETUALs. This is what makes the three "exclude TradFi"
#: encodings disagree about the two INDEX rows.
_CONTRACT_TYPE = {"COIN": "PERPETUAL", "INDEX": "PERPETUAL"}


def _universe() -> pd.DataFrame:
    """A bundle-shaped universe: keyed on ``instrument_id``, Binance camelCase."""
    return pd.DataFrame([
        {"instrument_id": symbol, "quoteVolume": turnover,
         "priceChangePercent": change, "available_time": 1_700_000_000_000}
        for symbol, _type, _tags, turnover, change in _ROWS
    ])


def _meta(*, wire_lists: bool = True) -> pd.DataFrame:
    """The registry frame, in the vendor vocabulary a bundle delivers.

    ``wire_lists`` keeps ``underlyingSubType`` as the JSON array the API actually
    sends. Tests that need the pre-flattened form ask for it explicitly, so no
    test can accidentally prove the easy case only.
    """
    return pd.DataFrame([
        {"instrument_id": symbol,
         "contractType": _CONTRACT_TYPE.get(underlying, "TRADIFI_PERPETUAL"),
         "underlyingType": underlying,
         "underlyingSubType": (list(tags) if wire_lists
                               else ",".join(tags)),
         "baseAsset": symbol[:-4], "quoteAsset": "USDT", "status": "TRADING",
         "available_time": 1_700_000_000_000}
        for symbol, underlying, tags, _turnover, _change in _ROWS
    ])


def _joined() -> pd.DataFrame:
    return ub.augment_with_contract_meta(_universe(), _meta())


def _symbols(frame: pd.DataFrame) -> list:
    return frame["instrument_id"].tolist()


# --------------------------------------------------------------------------- #
# augment: the join and its vocabulary                                        #
# --------------------------------------------------------------------------- #


def test_the_join_adds_six_columns_and_widens_nothing_else():
    joined = _joined()

    assert list(joined.columns)[: len(_universe().columns) + 1] == \
        list(_universe().columns) + ["symbol"]
    for column in META_COLUMNS:
        assert column in joined.columns, column
    assert len(joined) == len(_ROWS), "a left join must not change the row count"
    assert _symbols(joined) == _symbols(_universe()), "row order moved"


def test_the_registry_may_be_wider_than_the_universe_without_widening_it():
    """A registry covers the venue; the universe is whatever the ticker returned.

    So the join is a left join on the universe, and extra registry rows (a
    delisted or not-yet-trading contract) must not appear in the basket. Getting
    this backwards would put an unfillable instrument in a basket, and the
    metadata columns would all look perfectly populated.
    """
    meta = pd.concat([_meta(), pd.DataFrame([{
        "instrument_id": "DELISTEDUSDT", "contractType": "PERPETUAL",
        "underlyingType": "COIN", "underlyingSubType": ["Meme"],
        "baseAsset": "DELISTED", "quoteAsset": "USDT", "status": "SETTLING",
        "available_time": 1_700_000_000_000,
    }])], ignore_index=True)

    joined = ub.augment_with_contract_meta(_universe(), meta)

    assert "DELISTEDUSDT" not in _symbols(joined)
    assert len(joined) == len(_ROWS)


def test_the_join_reads_snake_case_as_well_as_the_vendor_camelCase():
    """Both vocabularies are real: the venue sends camelCase and a bundle keeps it,
    while a hand-built frame naturally writes snake_case. Accepting only one made
    the block untestable without a captured payload."""
    snake = _meta().rename(columns={
        "contractType": "contract_type", "underlyingType": "underlying_type",
        "underlyingSubType": "underlying_sub_type", "baseAsset": "base_asset",
        "quoteAsset": "quote_asset", "status": "contract_status",
    })

    pd.testing.assert_frame_equal(
        ub.augment_with_contract_meta(_universe(), snake), _joined())


def test_status_is_renamed_so_it_says_what_is_trading():
    """``status`` alone is ambiguous on a cross-section — the bundle envelope
    already uses that word for "could we read this source at all"."""
    joined = _joined()

    assert "status" not in joined.columns
    assert set(joined["contract_status"]) == {"TRADING"}


def test_a_second_augment_replaces_the_columns_rather_than_suffixing_them():
    """Two augment steps in one spec is a plausible authoring mistake, and pandas'
    default would answer it with ``underlying_type_x`` / ``_y`` — two columns, one
    of them silently unreachable by the name the spec uses."""
    twice = ub.augment_with_contract_meta(_joined(), _meta())

    assert [column for column in twice.columns if column.endswith(("_x", "_y"))] == []
    pd.testing.assert_frame_equal(twice, _joined())


# --------------------------------------------------------------------------- #
# the array mine                                                              #
# --------------------------------------------------------------------------- #


def test_a_multi_tag_coin_becomes_one_scalar_cell_at_the_join_boundary():
    """The specific shape that raises deeper down; see the module docstring."""
    joined = _joined().set_index("instrument_id")

    assert joined.loc["FOLKSUSDT", "underlying_sub_type"] == "Alpha,DeFi"
    assert joined.loc["SOXLUSDT", "underlying_sub_type"] == "TradFi,ETF"
    # No cell may still be a container: one is enough to raise.
    assert not any(isinstance(value, (list, tuple, set))
                   for value in joined["underlying_sub_type"])


def test_an_untagged_contract_is_an_empty_string_and_not_a_missing_value():
    """"The venue assigned no sector" is an answer; "we could not look it up" is
    not. 33 of 727 live contracts are untagged, so conflating the two would make
    every one of them undroppable-or-unkeepable for the wrong reason."""
    joined = _joined().set_index("instrument_id")

    assert joined.loc["CAPUSDT", "underlying_sub_type"] == ""
    assert joined["underlying_sub_type"].notna().all()


def test_a_tag_containing_the_separator_is_refused_rather_than_encoded():
    """Because the encoding would not survive the round trip, and the resulting
    tag set would be wrong in a way no reader could see."""
    meta = _meta()
    meta.loc[0, "underlyingSubType"] = ["Proof, of Work"]

    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_contract_meta(_universe(), meta)

    assert "separator" in str(excinfo.value)
    assert "_SUB_TYPE_SEPARATOR" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# augment: refusals                                                           #
# --------------------------------------------------------------------------- #


def test_an_empty_registry_is_refused_instead_of_nan_filled():
    """Unlike ``augment_with_news``: buzz is genuinely absent for most coins, but
    every listed contract has a contract type by construction, so an empty
    registry is a failed fetch. NaN-filling would turn it into an empty basket
    with nothing to point at."""
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_contract_meta(_universe(), _meta().iloc[0:0])

    assert "empty" in str(excinfo.value)
    assert "contract_meta" in str(excinfo.value), "the message must name the node"


def test_a_registry_missing_one_field_is_refused_at_the_join():
    """A partial registry hands the filters a NaN column, which reads as "this
    instrument has no sector" for EVERY symbol — a filter that then drops the
    whole universe, or keeps it, for a reason nothing records."""
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_contract_meta(
            _universe(), _meta().drop(columns=["underlyingType"]))

    assert "underlying_type" in str(excinfo.value)
    assert "underlyingType" in str(excinfo.value), "name the column it looked for"


def test_a_registry_that_does_not_cover_the_universe_is_refused():
    """The coverage guard, and the reason it is not a market fact.

    ``exchangeInfo`` is the listing registry of the same venue the ticker table
    came from, so total coverage is an identity (measured: 727 of 727). A hole
    means the wrong source — the classic one is a spot registry joined onto a
    futures universe — and every uncovered row becomes NaN metadata.
    """
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_contract_meta(_universe(), _meta().head(3))

    message = str(excinfo.value)
    assert "covers only 3 of 12" in message
    assert "market_type" in message, "the message must name the likeliest cause"


def test_coverage_just_above_the_floor_is_allowed_because_a_listing_race_is_real():
    """A symbol can enter the 24h ticker between the two fetches, and aborting a
    live basket over one row would be an alarm nobody can act on. It is reported
    instead — see the unknown-metadata tests below."""
    wide = pd.concat([_universe()] * 8, ignore_index=True)
    wide["instrument_id"] = ["SYM%03dUSDT" % index for index in range(len(wide))]
    meta = _meta()
    meta = pd.concat([meta] * 8, ignore_index=True)
    meta["instrument_id"] = wide["instrument_id"]

    joined = ub.augment_with_contract_meta(wide, meta.head(len(wide) - 2))

    assert joined["underlying_type"].isna().sum() == 2


def test_the_join_refuses_a_non_frame_source():
    with pytest.raises(ValueError) as excinfo:
        ub.augment_with_contract_meta(_universe(), {"symbol": "BTCUSDT"})

    assert "DataFrame" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# filter_underlying_type                                                      #
# --------------------------------------------------------------------------- #


def test_include_and_exclude_are_exact_and_case_insensitive():
    joined = _joined()

    assert set(_symbols(ub.filter_underlying_type(joined, include="coin"))) == {
        "BTCUSDT", "ETHUSDT", "UNIUSDT", "CLANKERUSDT", "FOLKSUSDT", "CAPUSDT"}
    assert "SKHYUSDT" in _symbols(
        ub.filter_underlying_type(joined, include=["KR_EQUITY"]))
    # Exact, never a substring: EQUITY must not sweep up KR_EQUITY, or "no US
    # stocks" would quietly also mean "no Korean stocks".
    assert set(_symbols(ub.filter_underlying_type(joined, include=["EQUITY"]))) == {
        "SNDKUSDT", "SOXLUSDT"}
    assert set(_symbols(ub.filter_underlying_type(joined, exclude=["EQUITY"]))) == \
        set(_symbols(joined)) - {"SNDKUSDT", "SOXLUSDT"}


def test_a_filter_with_no_criterion_is_refused():
    """Because it returns the frame unchanged, which is the same output a missing
    step gives — and the spec would read as if it were filtering."""
    with pytest.raises(ValueError) as excinfo:
        ub.filter_underlying_type(_joined())

    assert "include= or exclude=" in str(excinfo.value)
    assert "filter_crypto_only" in str(excinfo.value), "point at the named intent"


def test_include_and_exclude_together_are_refused_for_a_scalar_field():
    """One value per row means ``include`` already names the whole surviving set,
    so ``exclude`` can only contradict it — a way to write a contradiction and get
    an empty basket with no error. Contrast the sub-type test below."""
    with pytest.raises(ValueError) as excinfo:
        ub.filter_underlying_type(_joined(), include=["COIN"], exclude=["COIN"])

    assert "not both" in str(excinfo.value)


@pytest.mark.parametrize("argument", ["include", "exclude"])
def test_an_empty_category_list_is_refused(argument):
    """``include=[]`` empties the universe and ``exclude=[]`` drops nothing, and
    both look like a filter that is working."""
    with pytest.raises(ValueError) as excinfo:
        ub.filter_underlying_type(_joined(), **{argument: []})

    assert "empty" in str(excinfo.value)


def test_the_filter_names_the_augment_step_when_the_column_is_absent():
    """The message has to carry the fix: this is the mistake an LLM-generated spec
    makes most, and "missing column" alone reads as a bug in the data."""
    with pytest.raises(ValueError) as excinfo:
        ub.filter_crypto_only(_universe())

    message = str(excinfo.value)
    assert "underlying_type" in message
    assert "augment_with_contract_meta" in message
    assert "with: [contract_meta]" in message


# --------------------------------------------------------------------------- #
# filter_crypto_only: the three encodings                                     #
# --------------------------------------------------------------------------- #


def test_the_three_encodings_of_exclude_tradfi_disagree_and_the_whitelist_is_tightest():
    """The reason ``filter_crypto_only`` exists as its own name.

    All three answer the question they were asked; they are not equivalent, and a
    basket does not record which one was written. The difference is the INDEX
    rows — synthetic baskets of many coins, which are not TradFi and are also not
    a coin you can hold a per-asset thesis about.
    """
    joined = _joined()

    by_contract_type = joined[joined["contract_type"] != "TRADIFI_PERPETUAL"]
    by_tag = ub.filter_sub_type(joined, exclude=["TradFi"])
    by_whitelist = ub.filter_crypto_only(joined)

    assert set(_symbols(by_contract_type)) == set(_symbols(by_tag))
    assert set(_symbols(by_whitelist)) < set(_symbols(by_tag))
    assert set(_symbols(by_tag)) - set(_symbols(by_whitelist)) == {
        "ALLUSDT", "BTCDOMUSDT"}
    assert set(joined.loc[joined["instrument_id"].isin(["ALLUSDT", "BTCDOMUSDT"]),
                          "underlying_type"]) == {"INDEX"}


def test_crypto_only_is_exactly_the_coin_whitelist():
    pd.testing.assert_frame_equal(
        ub.filter_crypto_only(_joined()),
        ub.filter_underlying_type(_joined(), include=["COIN"]))


def test_crypto_only_changes_the_basket_a_turnover_ranking_would_return():
    """The end the user cares about, on rows built to reproduce the real
    distortion: tokenised equities out-trade mid-cap coins by 100x, so ranking by
    liquidity hands every slot to them."""
    # Four TradFi rows, so a top-4 is the sharpest statement available: every one
    # of them outranks every coin in the frame.
    ranked = _joined().nlargest(4, "quoteVolume")
    assert set(ranked["underlying_type"]) == {"EQUITY", "KR_EQUITY", "COMMODITY"}

    crypto = ub.filter_crypto_only(_joined()).nlargest(4, "quoteVolume")

    assert set(crypto["underlying_type"]) == {"COIN"}
    assert set(_symbols(crypto)).isdisjoint(_symbols(ranked))


# --------------------------------------------------------------------------- #
# filter_sub_type                                                             #
# --------------------------------------------------------------------------- #


def test_a_tag_filter_is_a_set_intersection_and_not_a_substring_match():
    """A tag is a whole name, not a fragment of the encoded cell.

    ``"Fi"`` is the discriminating case, and it is not contrived: it is a
    substring of both ``DeFi`` and ``TradFi``, and neither is it a tag. Under a
    set intersection it matches nothing — so ``exclude=['Fi']`` drops nothing and
    ``include=['Fi']`` keeps nothing, both reported. Under a substring match it
    would silently sweep up every DeFi *and* every TradFi row, i.e. put coins in a
    sector they were never tagged with while looking like it worked.
    """
    joined = _joined()

    assert set(_symbols(ub.filter_sub_type(joined, include=["AI"]))) == {
        "CLANKERUSDT"}
    assert set(_symbols(ub.filter_sub_type(joined, include=["Alpha"]))) == {
        "CLANKERUSDT", "FOLKSUSDT"}
    assert set(_symbols(ub.filter_sub_type(joined, include=["DeFi"]))) == {
        "UNIUSDT", "FOLKSUSDT"}

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert _symbols(ub.filter_sub_type(joined, include=["Fi"])) == []
        assert set(_symbols(ub.filter_sub_type(joined, exclude=["Fi"]))) == \
            set(_symbols(joined))
        # And the other direction of the same mistake: a whole tag must not be
        # matched by a prefix of it either.
        assert _symbols(ub.filter_sub_type(joined, include=["Trad"])) == []


def test_a_multi_tagged_row_is_matched_by_any_of_its_tags():
    joined = _joined()

    for tag in ("Alpha", "DeFi"):
        assert "FOLKSUSDT" in _symbols(ub.filter_sub_type(joined, include=[tag]))
        assert "FOLKSUSDT" not in _symbols(ub.filter_sub_type(joined, exclude=[tag]))


def test_include_and_exclude_together_are_meaningful_for_a_multi_valued_field():
    """The asymmetry with ``filter_underlying_type``, stated as the case that
    needs it: ``SOXLUSDT`` is tagged both ETF and TradFi, so "ETFs but not the
    TradFi ones" is a condition neither list can express alone."""
    joined = _joined()

    assert "SOXLUSDT" in _symbols(ub.filter_sub_type(joined, include=["ETF"]))
    assert _symbols(ub.filter_sub_type(joined, include=["ETF"],
                                       exclude=["TradFi"])) == []


def test_an_untagged_row_fails_every_include_and_passes_every_exclude():
    """The consequence of "no sector" being an answer rather than a hole."""
    joined = _joined()

    assert "CAPUSDT" not in _symbols(ub.filter_sub_type(joined, include=["DeFi"]))
    assert "CAPUSDT" in _symbols(ub.filter_sub_type(joined, exclude=["TradFi"]))


def test_a_tag_filter_reads_an_already_flattened_column():
    """A cross-section can reach the filter with the scalar already in place (a
    second spec step, a hand-built frame). Both encodings have to agree."""
    pre_flat = ub.augment_with_contract_meta(_universe(), _meta(wire_lists=False))

    pd.testing.assert_frame_equal(
        ub.filter_sub_type(pre_flat, exclude=["TradFi"]),
        ub.filter_sub_type(_joined(), exclude=["TradFi"]))


# --------------------------------------------------------------------------- #
# unknown metadata                                                            #
# --------------------------------------------------------------------------- #


def _unjoined() -> pd.DataFrame:
    """One row whose metadata never joined — the shape a listing race produces."""
    joined = _joined()
    joined.loc[joined["instrument_id"] == "SNDKUSDT", META_COLUMNS] = None
    return joined


@pytest.mark.parametrize("call, label", [
    (lambda frame: ub.filter_underlying_type(frame, exclude=["EQUITY"]),
     "filter_underlying_type"),
    (lambda frame: ub.filter_sub_type(frame, exclude=["TradFi"]), "filter_sub_type"),
])
def test_a_row_with_no_metadata_is_dropped_by_exclude_too(call, label):
    """The dangerous direction. If NaN read as "no tags", an un-joined EQUITY
    perpetual would sail through ``exclude=['TradFi']`` and land in a basket that
    the spec says excludes stocks — which is the precise outcome the whole
    feature exists to prevent."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kept = call(_unjoined())

    assert "SNDKUSDT" not in _symbols(kept)
    messages = [str(entry.message) for entry in caught
                if issubclass(entry.category, RuntimeWarning)]
    assert any(label in message and "missing" in message for message in messages), \
        messages


def test_a_row_with_no_metadata_is_dropped_by_include_too():
    """Symmetry, and the reason it is symmetric: an unknown can neither be shown
    to match nor be cleared, so the answer is the same in both directions."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        kept = ub.filter_underlying_type(_unjoined(), include=["EQUITY"])

    assert set(_symbols(kept)) == {"SOXLUSDT"}


def test_a_category_no_row_carries_is_reported_even_when_others_matched():
    """``_warn_matched_nothing`` only fires when a filter matches nothing at all,
    so ``include=['COIN', 'EQUTIY']`` was silent: the typo simply removed a
    category from the basket."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kept = ub.filter_underlying_type(_joined(), include=["COIN", "EQUTIY"])

    assert len(kept) == 6, "the good half of the criterion must still work"
    messages = [str(entry.message) for entry in caught]
    assert any("EQUTIY" in message and "COIN" in message for message in messages), \
        messages


def test_an_empty_frame_is_handled_without_warning_noise():
    """A previous step legitimately empties the universe; the metadata filter is
    not the place that complains about it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = ub.filter_crypto_only(_joined().iloc[0:0])

    assert out.empty
    assert [str(entry.message) for entry in caught] == []


# --------------------------------------------------------------------------- #
# the fetcher                                                                 #
# --------------------------------------------------------------------------- #


_RAW_EXCHANGE_INFO = {"symbols": [
    {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "underlyingType": "COIN",
     "underlyingSubType": ["PoW"], "baseAsset": "BTC", "quoteAsset": "USDT",
     "status": "TRADING", "pricePrecision": 2},
    {"symbol": "BTCUSDC", "contractType": "PERPETUAL", "underlyingType": "COIN",
     "underlyingSubType": ["PoW", "USDC"], "baseAsset": "BTC",
     "quoteAsset": "USDC", "status": "TRADING", "pricePrecision": 1},
    {"symbol": "OLDUSDT", "contractType": "PERPETUAL", "underlyingType": "COIN",
     "underlyingSubType": [], "baseAsset": "OLD", "quoteAsset": "USDT",
     "status": "SETTLING", "pricePrecision": 4},
]}


def test_the_fetcher_filters_nothing_and_hands_status_and_quote_over_as_columns():
    """``full_market_scan`` keeps only TRADING + USDT, and copying that here would
    have been the subtle bug: the USDC pairs would join to NaN metadata and
    ``filter_crypto_only`` would drop them for *being unknown* while appearing to
    drop them for *not being crypto* — the right basket from wrong reasoning, and
    it hides whether the by-name exclusion list is still needed."""
    captured = {}

    def fake(market_type="futures"):
        captured["market_type"] = market_type
        return _RAW_EXCHANGE_INFO

    original = blocks_data.fetch_exchange_info
    blocks_data.fetch_exchange_info = fake
    try:
        frame = blocks_data.fetch_contract_meta(market_type="futures")
    finally:
        blocks_data.fetch_exchange_info = original

    assert captured == {"market_type": "futures"}
    assert list(frame["symbol"]) == ["BTCUSDT", "BTCUSDC", "OLDUSDT"]
    assert set(frame["quoteAsset"]) == {"USDT", "USDC"}
    assert "SETTLING" in set(frame["status"])
    assert list(frame.columns) == list(blocks_data._CONTRACT_META_FIELDS), (
        "the column set is declared, not inherited from the response — an extra "
        "vendor field must not reach a spec by accident")
    # The list is passed through untouched; flattening belongs to the join.
    assert frame.loc[1, "underlyingSubType"] == ["PoW", "USDC"]


@pytest.mark.parametrize("payload, expected", [
    ({}, "no 'symbols' list"),
    ({"symbols": []}, "listed no symbols"),
    ({"symbols": [{"symbol": "BTCUSDT"}]}, "missing field"),
])
def test_the_fetcher_refuses_a_response_it_cannot_build_a_registry_from(
    payload, expected
):
    """Each of these would otherwise surface two steps later as "every symbol has
    unknown metadata", which points at the join instead of at the endpoint."""
    original = blocks_data.fetch_exchange_info
    blocks_data.fetch_exchange_info = lambda market_type="futures": payload
    try:
        with pytest.raises(RuntimeError) as excinfo:
            blocks_data.fetch_contract_meta()
    finally:
        blocks_data.fetch_exchange_info = original

    assert expected in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the three-stage bridge                                                      #
# --------------------------------------------------------------------------- #


def test_the_catalog_node_declares_a_forward_only_registry_with_its_hazard():
    node = get_node("contract_meta")

    assert node.emits is FrameKind.RANK
    assert node.availability is Availability.FORWARD_ONLY
    assert node.column_map == {"symbol": "instrument_id"}
    assert node.fetcher == "cyqnt_trd.blocks.data.fetch_contract_meta"
    # Non-BACKTESTABLE nodes must document how they lie if replayed. This one's
    # hazard is not "the value moved" but "the row exists at all".
    assert "delist" in node.pit_hazard.lower()


def test_the_live_plan_collects_contract_meta_only_when_a_spec_asks_for_it():
    """A whole-market registry is not part of a single-instrument plan, so it has
    to be appended — and only on demand, or every paper run pays for 851 rows it
    does not read."""
    assert SECTION_NODES["contract_meta"] == ("contract_meta",)

    without = requests_for_sections(["universe"])
    with_meta = requests_for_sections(["universe", "contract_meta"],
                                      market_type="futures")

    assert "contract_meta" not in [node for node, _params, _key in without]
    assert ("contract_meta", {"market_type": "futures"}, "contract_meta") in with_meta
    # The plan is otherwise unchanged: one appended request, nothing displaced.
    assert with_meta[:-1] == without


def test_a_spec_using_the_block_pulls_the_section_into_the_live_plan():
    """The half that made ``with: [funding]`` work for a live run, for the new
    source. Without it a spec validated, ran offline, and then met an absent frame
    the first time it was collected live."""
    spec = load_spec(str(SPEC_USER_CHAT))
    spec["selection"]["universe"].append(
        {"block": "universe.augment_with_contract_meta", "with": ["contract_meta"]})

    assert "contract_meta" in required_bundle_nodes(spec)
    assert "contract_meta" in live_sections_for_spec(spec)
    # A spec that does not use it must not collect it.
    assert "contract_meta" not in live_sections_for_spec(load_spec(str(SPEC_USER_CHAT)))


def test_the_block_may_not_fetch_its_own_source_from_a_spec():
    """``augment_with_contract_meta(tickers, None)`` calls exchangeInfo. Reachable
    from YAML, that turns ``validate`` on a frontend-supplied spec into outbound
    REST traffic and makes a backtest read today's listings."""
    assert FETCHES_WITHOUT_SOURCE["universe.augment_with_contract_meta"] == \
        "contract_meta"

    spec = load_spec(str(SPEC_USER_CHAT))
    spec["selection"]["universe"].append(
        {"block": "universe.augment_with_contract_meta"})

    errors, _warnings = validate_spec(spec)

    assert any("with: [contract_meta]" in error for error in errors), errors


def test_an_empty_contract_meta_frame_stops_the_run_instead_of_the_block():
    """"I read it and it was empty" is a source failure for this node, so the
    runner names the source rather than letting a ValueError surface from inside a
    universe step. ``ticker_rank`` is deliberately NOT in that set: Square having
    nothing to say is normal and ``augment_with_news`` NaN-fills it."""
    spec = load_spec(str(SPEC_USER_CHAT))
    spec["selection"]["universe"].append(
        {"block": "universe.augment_with_contract_meta", "with": ["contract_meta"]})
    bundle = _fixture()
    bundle["frames"]["contract_meta"]["rows"] = []

    with pytest.raises(BundleRunError) as excinfo:
        run_bundle(spec, bundle)

    assert "contract_meta" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# validate: the dry-run has to be able to fail                                #
# --------------------------------------------------------------------------- #


def _crypto_only_spec() -> dict:
    """``example_from_user_chat.yaml`` plus the two steps that close its gap #3.

    Built from the shipped spec rather than copied, so it cannot drift from it.
    """
    spec = load_spec(str(SPEC_USER_CHAT))
    steps = spec["selection"]["universe"]
    position = next(index for index, step in enumerate(steps)
                    if step["block"] == "universe.top_losers")
    steps[position:position] = [
        {"block": "universe.augment_with_contract_meta", "with": ["contract_meta"]},
        {"block": "universe.filter_crypto_only"},
    ]
    return spec


def test_the_metadata_steps_validate_green_on_the_synthetic_cross_section():
    errors, spec_warnings = validate_spec(_crypto_only_spec())

    assert errors == []
    assert [warning for warning in spec_warnings
            if "no candidates" in warning] == [], spec_warnings


def test_the_dry_run_universe_contains_both_sides_of_the_crypto_filter():
    """Otherwise validate proves nothing: on a crypto-only stand-in,
    ``filter_crypto_only`` drops no row — the same output a missing step gives, so
    a spec that forgot it would still dry-run green."""
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import (
        _synthetic_contract_meta,
        _synthetic_universe,
    )

    universe = _synthetic_universe()
    joined = ub.augment_with_contract_meta(
        universe, _synthetic_contract_meta(universe))

    kept = ub.filter_crypto_only(joined)
    assert 0 < len(kept) < len(joined), "one side of the filter is empty"
    assert set(joined["underlying_type"]) >= {"COIN", "EQUITY", "INDEX"}
    # The two shapes that break naive code have to be in the stand-in too.
    tags = joined["underlying_sub_type"]
    assert (tags == "").any(), "no untagged row"
    assert tags.str.contains(",").any(), "no multi-tag row"


def test_the_dry_run_catches_a_spec_that_filters_without_augmenting():
    """The most likely LLM mistake, and the message it gets."""
    spec = _crypto_only_spec()
    spec["selection"]["universe"] = [
        step for step in spec["selection"]["universe"]
        if step["block"] != "universe.augment_with_contract_meta"]

    errors, _warnings = validate_spec(spec)

    assert any("augment_with_contract_meta" in error for error in errors), errors


def test_a_misspelt_block_name_is_a_static_validation_error():
    """Why there are three named functions instead of one
    ``filter_meta(field=...)``: a wrong function name is caught by the static
    resolution pass, while a wrong ``field:`` string could only fail at run time —
    the difference between a validation error and a wrong basket."""
    spec = _crypto_only_spec()
    spec["selection"]["universe"][-2] = {"block": "universe.filter_crypto"}

    errors, _warnings = validate_spec(spec)

    assert any("filter_crypto" in error for error in errors), errors


# --------------------------------------------------------------------------- #
# the frozen cross-section: the numbers this feature exists for               #
# --------------------------------------------------------------------------- #


def _liquid_losers(joined: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """The shipped spec's own pipeline, up to the point it ranks."""
    frame = ub.filter_quote_volume(joined, 2_000_000)
    frame = ub.exclude_symbols(frame, ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    frame = ub.filter_quote_suffix(frame, "USDC", exclude=True)
    return ub.top_losers(frame, n)


def test_the_frozen_registry_covers_the_frozen_universe_completely():
    """The identity the coverage guard is built on, measured rather than assumed.

    Also what makes the assertions below meaningful: a partial join would give a
    non-COIN row NaN metadata and it would then be dropped as *unknown*, which is
    the right answer for the wrong reason.
    """
    universe, meta = _frozen("universe"), _frozen("contract_meta")

    joined = ub.augment_with_contract_meta(universe, meta)

    assert len(universe) == 727, "the fixture moved; re-derive the numbers below"
    assert joined["underlying_type"].notna().all()
    # The registry is wider than the universe (it lists SETTLING contracts too)
    # and the join must not import them.
    assert len(meta) > len(universe)
    assert len(joined) == len(universe)


def test_the_frozen_registry_still_carries_the_wire_arrays():
    """If a future recapture pre-flattened this column, every replay test would
    stop exercising ``_flatten_sub_type`` — the conversion that a live run needs —
    and would keep passing."""
    rows = _fixture()["frames"]["contract_meta"]["rows"]

    multi = [row["instrument_id"] for row in rows
             if isinstance(row["underlyingSubType"], list)
             and len(row["underlyingSubType"]) > 1]
    assert multi, "no multi-tag contract in the frozen registry"
    assert all(isinstance(row["underlyingSubType"], list) for row in rows)


def test_the_turnover_ranking_of_the_frozen_losers_is_entirely_tradfi():
    """The measured harm, on the cross-section the shipped example is pinned to.

    21 of the 30 biggest 24h losers above the spec's own $2m floor are not crypto,
    and because tokenised equities out-trade mid-cap coins by one to two orders of
    magnitude, all five slots of a turnover-ranked basket go to TradFi. This is
    what ``example_from_user_chat.yaml``'s golden basket is, and what its author
    asked to exclude.
    """
    joined = ub.augment_with_contract_meta(_frozen("universe"),
                                           _frozen("contract_meta"))
    losers = _liquid_losers(joined)

    assert len(losers) == 30
    assert int((losers["underlying_type"] != "COIN").sum()) == 21

    top5 = losers.nlargest(5, "quoteVolume")
    assert list(top5["instrument_id"]) == ["SNDKUSDT", "SOXLUSDT", "MUUSDT",
                                          "SKHYUSDT", "KORUUSDT"]
    assert set(top5["underlying_type"]) == {"EQUITY"}
    assert all("TradFi" in tags for tags in top5["underlying_sub_type"])
    # The magnitude, because it is what makes the ranking hopeless rather than
    # merely imperfect: the biggest COIN in this set is an order of magnitude
    # behind the biggest equity.
    biggest_coin = losers[losers["underlying_type"] == "COIN"]["quoteVolume"].max()
    assert top5["quoteVolume"].min() > biggest_coin * 2


def test_crypto_only_empties_the_tradfi_slots_on_the_frozen_cross_section():
    joined = ub.augment_with_contract_meta(_frozen("universe"),
                                           _frozen("contract_meta"))

    crypto = _liquid_losers(ub.filter_crypto_only(joined))

    assert len(crypto) == 30
    assert set(crypto["underlying_type"]) == {"COIN"}
    assert set(crypto["instrument_id"]).isdisjoint(
        ["SNDKUSDT", "SOXLUSDT", "MUUSDT", "SKHYUSDT", "KORUUSDT"])


# --------------------------------------------------------------------------- #
# end to end, through run_bundle                                              #
# --------------------------------------------------------------------------- #


#: What ``example_from_user_chat.yaml`` returns once the two metadata steps are
#: in it, on the frozen cross-section. Pinned here for the same reason the other
#: golden baskets are: this is a real market at a real instant, and a diff on this
#: list is the only way to see the basket change for a code reason.
CRYPTO_ONLY_BASKET = [
    (1, "HYPEUSDT", "short"), (2, "BANKUSDT", "short"), (3, "COTIUSDT", "short"),
    (4, "MMTUSDT", "short"), (5, "UNIUSDT", "short"),
]


def test_the_shipped_spec_plus_two_steps_returns_a_crypto_only_basket():
    """The whole feature, through the canonical decision path.

    ``filter_crypto_only`` sits BEFORE ``top_losers`` on purpose: narrowing the
    asset class first means the 30 losers are the 30 biggest CRYPTO losers, rather
    than whatever crypto survives a top-30 that stocks already filled.
    """
    output = run_bundle(_crypto_only_spec(), _fixture())

    assert output["signal_count"] == 1
    candidates = output["signals"][0]["candidates"]
    assert [(item["rank"], item["symbol"], item["direction"])
            for item in candidates] == CRYPTO_ONLY_BASKET
    for item in candidates:
        features = item["features"]
        assert features["underlying_type"] == "COIN"
        assert features["contract_status"] == "TRADING"
        assert isinstance(features["underlying_sub_type"], str)

    # And the shipped spec — unchanged — still returns the all-TradFi basket, so
    # this test measures the two steps and not a drifting fixture.
    shipped = run_bundle(str(SPEC_USER_CHAT), _fixture())
    assert [item["symbol"] for item in shipped["signals"][0]["candidates"]] == [
        "SNDKUSDT", "SOXLUSDT", "MUUSDT", "SKHYUSDT", "KORUUSDT"]


def test_a_multi_tag_coin_survives_run_bundle_into_the_candidate_features():
    """The array mine, end to end on real rows.

    ``interpreter.selection_fn`` builds ``features`` with ``if pd.notna(value)``,
    which raises on a 2-element array and not on a 1-element one. So this needs a
    real multi-tag contract *in the basket*, which is why the spec pins them by
    name instead of hoping the ranking picks one.
    """
    bundle = _fixture()
    multi = [row["instrument_id"] for row in bundle["frames"]["contract_meta"]["rows"]
             if isinstance(row["underlyingSubType"], list)
             and len(row["underlyingSubType"]) > 1]
    universe = {row["instrument_id"] for row in bundle["frames"]["universe"]["rows"]}
    listed = [symbol for symbol in multi if symbol in universe]
    assert listed, "the frozen universe lists no multi-tag contract"

    spec = {
        "spec_version": "1.0", "target": "standard_bot",
        "strategy": {"id": "multi_tag_probe"}, "run": {"mode": "backtest"},
        "data": {"symbol": "BTCUSDT", "market_type": "futures",
                 "primary": {"interval": "1h"}},
        "selection": {
            "universe": [
                {"block": "universe.only_symbols", "params": {"symbols": listed}},
                {"block": "universe.augment_with_contract_meta",
                 "with": ["contract_meta"]},
            ],
            "score": "quoteVolume", "top_k": 10, "dedupe_by": "none",
        },
    }

    candidates = run_bundle(spec, bundle)["signals"][0]["candidates"]

    assert {item["symbol"] for item in candidates} == set(listed)
    tags = {item["symbol"]: item["features"]["underlying_sub_type"]
            for item in candidates}
    assert all(isinstance(value, str) for value in tags.values()), tags
    assert any("," in value for value in tags.values()), tags


def test_replay_of_the_metadata_spec_is_byte_identical():
    """A metadata join must not smuggle a clock or a set-iteration order into the
    output; without this the golden basket above could not hold."""
    spec = _crypto_only_spec()

    first = run_bundle(spec, _fixture())
    second = run_bundle(spec, _fixture())

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_metadata_replay_touches_no_network(monkeypatch):
    """``with: [contract_meta]`` must never fall back to live exchangeInfo — the
    fallback exists for direct Python callers only."""
    calls = []

    def deny(*args, **kwargs):
        calls.append("blocks.data._request_json")
        raise AssertionError("replay must not fetch")

    monkeypatch.setattr(blocks_data, "_request_json", deny)

    output = run_bundle(_crypto_only_spec(), _fixture())

    assert [(item["rank"], item["symbol"], item["direction"])
            for item in output["signals"][0]["candidates"]] == CRYPTO_ONLY_BASKET
    assert calls == []


def test_the_selection_fn_refuses_to_run_the_step_without_the_frame():
    """``with:`` never falls back to the network, even in-process."""
    spec = _crypto_only_spec()

    with pytest.raises(SpecError) as excinfo:
        build_selection_fn(spec)(_frozen("universe"), None)

    assert "contract_meta" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# how the frame got into the fixture                                          #
# --------------------------------------------------------------------------- #


def _freeze_script():
    import importlib.util

    path = REPO / "scripts" / "freeze_selection_fixture.py"
    spec = importlib.util.spec_from_file_location("_freeze_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_collection(module, rows, monkeypatch):
    """Stand in for the live collector, so these tests touch no network."""
    def build(*, requests, decision_time, market_type):
        node, params, key = requests[0]
        assert params == {"market_type": market_type}
        return {
            "frames": {key: {"shape": "RankFrame@1.0", "status": "ok",
                             "rows": rows}},
            "source_status": {key: "ok"},
            "warnings": ["%s: available_time taken from the fetch time" % key],
        }

    import cyqnt_trd.standard_bot.data.live_bundle as live_bundle
    monkeypatch.setattr(live_bundle, "build_live_bundle", build)
    return module


def _registry_rows(bundle: dict) -> list:
    """A registry that covers the frozen universe, built from the frozen one."""
    return bundle["frames"]["contract_meta"]["rows"]


def test_add_frame_leaves_every_other_frame_and_the_decision_time_alone():
    """The property that let the golden baskets survive this feature landing.

    Recapturing the whole fixture would have replaced the universe, the ticker
    ranks and the funding snapshot with a different hour of market, so no reviewer
    could tell the new capability from the market having moved.
    """
    module = _freeze_script()
    frozen = _fixture()
    rows = _registry_rows(frozen)
    without = {key: value for key, value in frozen["frames"].items()
               if key != "contract_meta"}
    target = dict(frozen, frames=without, source_status={
        key: value for key, value in frozen["source_status"].items()
        if key != "contract_meta"})

    with pytest.MonkeyPatch.context() as monkeypatch:
        _fake_collection(module, rows, monkeypatch)
        rebuilt = module.add_frame(target, "contract_meta")

    assert rebuilt["decision_time"] == frozen["decision_time"]
    assert rebuilt["snapshot_id"] == frozen["snapshot_id"]
    for key, entry in without.items():
        assert json.dumps(rebuilt["frames"][key], sort_keys=True) == \
            json.dumps(entry, sort_keys=True), key
    assert rebuilt["source_status"]["contract_meta"] == "ok"


def test_add_frame_refuses_a_node_that_measures_a_market_instead_of_a_registry():
    """An allowlist and not an availability check: ``universe`` is FORWARD_ONLY
    too, and stamping today's prices with a two-week-old decision_time would be a
    fabrication, not a convenience.

    ``universe_bars`` joined the allowlist for a second and different reason — its
    endpoint is addressable BY TIME, so ``end_ms = decision_time`` returns exactly
    the bars a capture at that instant saw, and the freeze script CHECKS that no
    captured bar closed later. It is still an allowlist: ``universe`` is refused
    below precisely because no ``endTime`` can be sent to it.
    """
    module = _freeze_script()

    assert set(module.ADDABLE_NODES) == {"contract_meta", "universe_bars"}
    with pytest.raises(module.FreezeError) as excinfo:
        module.add_frame(_fixture(), "universe")

    assert "universe" in str(excinfo.value)
    assert "recapture" in str(excinfo.value)


def test_add_frame_refuses_to_overwrite_a_frame_that_is_already_there():
    """Two capture instants in one artifact is exactly what the fixture exists to
    prevent."""
    module = _freeze_script()

    with pytest.raises(module.FreezeError) as excinfo:
        module.add_frame(_fixture(), "contract_meta")

    assert "already carries" in str(excinfo.value)


def test_add_frame_refuses_a_registry_that_cannot_show_the_filter_working():
    """The anti-vacuous guard. A registry in which every row is a COIN would let
    ``filter_crypto_only`` pass its tests by dropping nothing — indistinguishable
    from the filter not running, which is worse than having no fixture."""
    module = _freeze_script()
    frozen = _fixture()
    all_coin = [dict(row, underlyingType="COIN", underlyingSubType=["Meme"])
                for row in _registry_rows(frozen)]
    target = dict(frozen, frames={key: value for key, value in frozen["frames"].items()
                                  if key != "contract_meta"})

    with pytest.MonkeyPatch.context() as monkeypatch:
        _fake_collection(module, all_coin, monkeypatch)
        with pytest.raises(module.FreezeError) as excinfo:
            module.add_frame(target, "contract_meta")

    assert "all one asset class" in str(excinfo.value)


def test_add_frame_refuses_a_registry_that_does_not_cover_the_frozen_universe():
    """Verified by calling the block that will read it, so a fixture that passes
    here cannot fail its coverage guard at test time — where the failure would
    point at the test instead of at the capture."""
    module = _freeze_script()
    frozen = _fixture()
    target = dict(frozen, frames={key: value for key, value in frozen["frames"].items()
                                  if key != "contract_meta"})

    with pytest.MonkeyPatch.context() as monkeypatch:
        _fake_collection(module, _registry_rows(frozen)[:10], monkeypatch)
        with pytest.raises(module.FreezeError) as excinfo:
            module.add_frame(target, "contract_meta")

    assert "does not fit this universe" in str(excinfo.value)
