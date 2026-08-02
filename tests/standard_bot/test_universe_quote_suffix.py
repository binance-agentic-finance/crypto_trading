"""Filtering a universe by quote currency, from YAML.

``UniverseFilter.filter_quote_suffix`` existed as a method only, and a YAML spec
can dispatch to nothing but *module-level* functions under
``cyqnt_trd.blocks.<module>``. So "exclude USDC-quoted pairs" — a condition that
shows up verbatim in real coin-selection requests — had to be written as a
hand-listed ``exclude_symbols`` roster (``BTCUSDC, ETHUSDC, SOLUSDC, …``) that
goes stale the next time the venue lists another USDC pair.

These tests pin the module-level function, the fact that the method is now a
delegation rather than a second copy of the logic, and that a spec using it both
resolves and dry-runs.

They also pin the two things a quote filter must not do quietly: name a quote
currency the universe does not have (which leaves exactly the basket that no
filter at all would) and hand back a frame in a different key vocabulary than it
received (which decided whether a LATER step raised, and made the same steps in a
different order behave differently while both orders validated green).
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from cyqnt_trd.blocks import universe
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (SpecError,
                                                              build_selection_fn,
                                                              resolve_block)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import validate_spec

DT = 1_776_322_799_999

# USDCUSDT is deliberately present: it is a real Binance pair whose BASE is USDC
# and whose QUOTE is USDT, so it is the row that tells a suffix match apart from
# a substring match.
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BTCUSDC", "ETHUSDC", "USDCUSDT", "ETHBTC"]


def _universe() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": SYMBOLS,
        "quoteVolume": [9e8, 7e8, 4e8, 3e8, 2e8, 1.2e8],
        "priceChangePercent": [-3.0, -4.0, -5.0, -6.0, -0.1, -7.0],
        "available_time": [DT] * len(SYMBOLS),
    })


def _symbols(frame: pd.DataFrame) -> list:
    return frame["symbol"].tolist()


# --------------------------------------------------------------------------- #
# keep / exclude                                                              #
# --------------------------------------------------------------------------- #


def test_it_keeps_only_the_named_quote_by_default():
    kept = universe.filter_quote_suffix(_universe(), "USDT")
    assert _symbols(kept) == ["BTCUSDT", "ETHUSDT", "USDCUSDT"]


def test_exclude_drops_the_named_quote_and_keeps_everything_else():
    """The half a YAML spec could not express at all before."""
    kept = universe.filter_quote_suffix(_universe(), "USDC", exclude=True)
    assert _symbols(kept) == ["BTCUSDT", "ETHUSDT", "USDCUSDT", "ETHBTC"]


def test_the_match_is_on_the_quote_not_a_substring():
    """USDCUSDT is quoted in USDT; excluding USDC must not take it out."""
    kept = universe.filter_quote_suffix(_universe(), "USDC", exclude=True)
    assert "USDCUSDT" in _symbols(kept)
    dropped = universe.filter_quote_suffix(_universe(), "USDC")
    assert "USDCUSDT" not in _symbols(dropped)


def test_several_quotes_can_be_named_at_once():
    """"Only USDT and USDC pairs" is one criterion, not two chained steps."""
    kept = universe.filter_quote_suffix(_universe(), ["USDT", "USDC"])
    assert _symbols(kept) == ["BTCUSDT", "ETHUSDT", "BTCUSDC", "ETHUSDC",
                              "USDCUSDT"]
    both_excluded = universe.filter_quote_suffix(
        _universe(), ["USDT", "USDC"], exclude=True)
    assert _symbols(both_excluded) == ["ETHBTC"]


def test_matching_is_case_insensitive_on_both_sides():
    frame = _universe()
    frame["symbol"] = [s.lower() for s in SYMBOLS]
    kept = universe.filter_quote_suffix(frame, "usdt")
    assert _symbols(kept) == ["btcusdt", "ethusdt", "usdcusdt"]
    assert _symbols(universe.filter_quote_suffix(frame, "USDT")) == _symbols(kept)


def test_it_does_not_mutate_the_caller_frame():
    frame = _universe()
    universe.filter_quote_suffix(frame, "USDT")
    assert _symbols(frame) == SYMBOLS


def test_a_quote_nothing_matches_yields_an_empty_frame_with_the_columns_intact():
    with pytest.warns(RuntimeWarning, match="kept nothing"):
        kept = universe.filter_quote_suffix(_universe(), "TUSD")
    assert kept.empty
    assert list(kept.columns) == list(_universe().columns)


@pytest.mark.parametrize("exclude", [False, True])
def test_an_already_empty_universe_passes_through(exclude):
    """A filter earlier in the chain may have emptied the frame; masking an empty
    column must not raise on the way out — nor claim the suffix was wrong, which
    is what an unguarded "matched nothing" check would say about every frame that
    an earlier step had already emptied."""
    empty = pd.DataFrame({"symbol": pd.Series([], dtype=object),
                          "quoteVolume": pd.Series([], dtype=float)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = universe.filter_quote_suffix(empty, "USDC", exclude=exclude)
    assert out.empty
    assert list(out.columns) == ["symbol", "quoteVolume"]


# --------------------------------------------------------------------------- #
# a quote the universe does not have is not a filter                          #
# --------------------------------------------------------------------------- #


def test_a_misspelt_quote_currency_says_so_instead_of_dropping_nothing():
    """The silent half of the blank-suffix bug ``_quote_suffixes`` already refuses.

    ``exclude="USDCC"`` returns the whole universe: not one row is dropped, the
    basket is identical to the unfiltered one, and the author declared that USDCC
    pairs are unacceptable. ``keep`` at least comes back empty and shows up
    downstream as an empty basket; ``exclude`` had no tell anywhere.
    """
    with pytest.warns(RuntimeWarning, match=r"USDCC.*dropped nothing"):
        kept = universe.filter_quote_suffix(_universe(), "USDCC", exclude=True)

    assert _symbols(kept) == SYMBOLS, "nothing was dropped, as the warning says"


def test_the_warning_names_a_symbol_of_the_universe_it_could_not_match():
    """"Nothing matched" is not actionable on its own; the shape that WOULD match
    is. ``exclude_symbols(['BTC'])`` — a base token where an instrument belongs —
    is the same mistake and gets the same treatment."""
    with pytest.warns(RuntimeWarning, match="BTCUSDT") as caught:
        universe.filter_quote_suffix(_universe(), "USDCC", exclude=True)
    assert "USDT / USDC" in str(caught[0].message)

    with pytest.warns(RuntimeWarning, match="not base tokens") as caught:
        universe.exclude_symbols(_universe(), ["BTC", "ETH"])
    assert "BTCUSDT" in str(caught[0].message)


def test_a_quote_the_universe_does_have_is_silent():
    """The counterpart that keeps the warning worth reading: the correctly spelt
    filter must not also warn, or nobody will look at either."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert len(universe.filter_quote_suffix(_universe(), "USDC", exclude=True)) == 4
        assert len(universe.exclude_symbols(_universe(), ["BTCUSDT"])) == 5


def test_an_empty_roster_is_refused_rather_than_warned_about():
    """Matching nothing is a property of the data — a narrowed frame may honestly
    contain no USDC pair — so it warns. An empty roster is wrong against every
    possible frame, so it raises, exactly as a blank suffix does."""
    with pytest.raises(ValueError, match="no symbols"):
        universe.exclude_symbols(_universe(), [])
    with pytest.raises(ValueError, match="empties the universe"):
        universe.only_symbols(_universe(), [])
    with pytest.raises(ValueError, match="one character at a time"):
        universe.only_symbols(_universe(), "BTCUSDT")


# --------------------------------------------------------------------------- #
# one key vocabulary in, the same one out                                     #
# --------------------------------------------------------------------------- #


def _canonical() -> pd.DataFrame:
    """The shape a ``cyqnt.input/v1`` bundle delivers: no ``symbol`` column."""
    frame = _universe().rename(columns={"symbol": "instrument_id"})
    return frame


def test_the_filters_do_not_widen_a_canonical_frame_with_a_derived_symbol():
    """``_with_symbol_column`` has to derive ``symbol`` to filter a RankFrame at
    all, but leaving it behind is what let a spec's step order change the answer —
    and what put a duplicate of the instrument into every candidate's
    ``features``."""
    frame = _canonical()
    for out in (universe.filter_quote_suffix(frame, "USDC", exclude=True),
                universe.exclude_symbols(frame, ["BTCUSDT"]),
                universe.only_symbols(frame, ["BTCUSDT", "ETHUSDT"])):
        assert list(out.columns) == list(frame.columns)


def test_excluding_a_quote_and_a_roster_commutes():
    """Before: ``exclude_symbols`` after ``filter_quote_suffix`` worked because the
    suffix step had injected ``symbol``; the same two steps the other way round
    raised ``missing 'symbol' column``. Both orders validated."""
    frame = _canonical()

    suffix_first = universe.exclude_symbols(
        universe.filter_quote_suffix(frame, "USDC", exclude=True), ["BTCUSDT"])
    roster_first = universe.filter_quote_suffix(
        universe.exclude_symbols(frame, ["BTCUSDT"]), "USDC", exclude=True)

    pd.testing.assert_frame_equal(suffix_first, roster_first)
    assert list(suffix_first["instrument_id"]) == ["ETHUSDT", "USDCUSDT", "ETHBTC"]


def test_the_fluent_builder_can_report_what_it_selected_from_a_canonical_frame():
    """``.symbols()`` required ``symbol`` while every filter before it accepted
    ``instrument_id``, so a chain fed from a bundle filtered correctly and then
    refused to say what it had chosen."""
    selected = (universe.UniverseFilter(_canonical())
                .filter_quote_suffix("USDC", exclude=True)
                .symbols())
    assert selected == ["BTCUSDT", "ETHUSDT", "USDCUSDT", "ETHBTC"]


# --------------------------------------------------------------------------- #
# it refuses what it cannot honestly do                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix", ["", "   ", ["USDT", ""]])
def test_a_blank_suffix_is_refused_not_treated_as_match_everything(suffix):
    """``endswith("")`` is True for every symbol, so a blank suffix would empty
    the universe under exclude=True and be a no-op otherwise — both look like a
    working filter."""
    with pytest.raises(ValueError, match="blank"):
        universe.filter_quote_suffix(_universe(), suffix)


@pytest.mark.parametrize("suffix", [None, 3, {"quote": "USDT"}])
def test_a_non_name_suffix_says_what_to_pass_instead(suffix):
    with pytest.raises(ValueError, match="USDT"):
        universe.filter_quote_suffix(_universe(), suffix)


def test_an_empty_suffix_list_is_refused():
    with pytest.raises(ValueError, match="at least one quote currency"):
        universe.filter_quote_suffix(_universe(), [])


def test_a_frame_with_no_instrument_column_is_refused():
    frame = pd.DataFrame({"quoteVolume": [1e9]})
    with pytest.raises(ValueError, match="instrument_id"):
        universe.filter_quote_suffix(frame, "USDT")


def test_a_base_token_frame_is_refused_with_the_join_it_needs():
    """A Square frame keys on 'ticker', which is BTC — not a tradable pair, and
    a base token has no quote to filter on."""
    frame = pd.DataFrame({"ticker": ["BTC", "ETH"], "mention_count": [5, 3]})
    with pytest.raises(ValueError, match="augment_with_news"):
        universe.filter_quote_suffix(frame, "USDT")


def test_the_canonical_instrument_id_shape_works_too():
    """A universe out of an input bundle is a RankFrame@1.0: no 'symbol' column,
    only 'instrument_id'. Requiring the raw ticker vocabulary meant the canonical
    shape could not be fed to its own block."""
    frame = pd.DataFrame({"instrument_id": SYMBOLS})
    kept = universe.filter_quote_suffix(frame, "USDC", exclude=True)
    # And it comes back keyed the way it went in — see
    # test_the_filters_do_not_widen_a_canonical_frame_with_a_derived_symbol.
    assert list(kept["instrument_id"]) == ["BTCUSDT", "ETHUSDT", "USDCUSDT",
                                          "ETHBTC"]


# --------------------------------------------------------------------------- #
# one implementation, two entry points                                        #
# --------------------------------------------------------------------------- #


def test_it_is_exported():
    assert "filter_quote_suffix" in universe.__all__


def test_the_fluent_method_delegates_to_the_function():
    """Not "gives the same answer today" — the method must be the function, so a
    fix in one cannot miss the other."""
    frame = _universe()
    fluent = (universe.UniverseFilter(frame)
              .filter_quote_suffix("USDC", exclude=True)
              .to_frame())
    assert _symbols(fluent) == _symbols(
        universe.filter_quote_suffix(frame, "USDC", exclude=True))

    import inspect
    body = inspect.getsource(universe.UniverseFilter.filter_quote_suffix)
    assert "filter_quote_suffix(self.df" in body, body
    assert "endswith" not in body, "the method still carries its own copy of the logic"


def test_the_method_still_accepts_its_original_positional_call():
    """``strategies/scalp_opportunity.py`` and BLOCKS_API.md both call
    ``.filter_quote_suffix("USDT")``."""
    kept = universe.UniverseFilter(_universe()).filter_quote_suffix("USDT").symbols()
    assert kept == ["BTCUSDT", "ETHUSDT", "USDCUSDT"]


# --------------------------------------------------------------------------- #
# reachable from a YAML spec — the whole point                                #
# --------------------------------------------------------------------------- #


def test_a_spec_can_resolve_it():
    assert resolve_block("universe.filter_quote_suffix") is universe.filter_quote_suffix


def test_it_needs_no_source_so_it_cannot_touch_the_network():
    """Unlike augment_with_news / augment_with_funding it takes only the running
    frame, so no ``with:`` declaration is required and validate stays offline."""
    from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import \
        FETCHES_WITHOUT_SOURCE
    assert "universe.filter_quote_suffix" not in FETCHES_WITHOUT_SOURCE


def _spec(**step_params):
    return {
        "strategy": {"id": "quote_suffix_sel"},
        "run": {"mode": "backtest"},
        "data": {"symbol": "BTCUSDT", "primary": {"interval": "1h"}},
        "selection": {
            "universe": [
                {"block": "universe.filter_quote_suffix", "params": step_params},
            ],
            "score": "quoteVolume",
            "top_k": 5,
            "dedupe_by": "none",
            "short_when": {"cond": "conditions.value_below",
                           "args": ["priceChangePercent", -2.0]},
        },
    }


def test_a_selection_spec_excludes_a_whole_quote_currency():
    """What the user asked for — "排除 USDC 計價對" — as one step instead of a
    hand-listed roster."""
    spec = _spec(suffix="USDC", exclude=True)
    candidates = build_selection_fn(spec)(_universe(), None)
    assert [c["symbol"] for c in candidates] == ["BTCUSDT", "ETHUSDT", "ETHBTC"]
    assert all(c["side"] == "short" for c in candidates)


def test_a_selection_spec_can_keep_two_quotes():
    spec = _spec(suffix=["USDT", "USDC"])
    symbols = [c["symbol"] for c in build_selection_fn(spec)(_universe(), None)]
    assert "ETHBTC" not in symbols
    assert "BTCUSDC" in symbols


def test_the_spec_validates_and_dry_runs_on_synthetic_data():
    """The dry-run universe lists USDC pairs because a real one does, so excluding
    them here narrows the frame instead of matching nothing."""
    errors, reported = validate_spec(_spec(suffix="USDC", exclude=True))
    assert errors == [], errors
    assert reported == [], reported


def test_validate_reports_the_misspelt_quote_the_dry_run_could_not_match():
    """Where the typo is meant to be caught: offline, before anything runs.

    The block warns at whatever frame it is handed; ``validate`` relays that out of
    the dry-run so a frontend calling ``validate_spec`` sees it at all. This spec
    used to come back ``errors=[] warnings=[]`` and then produce a basket
    identical to one with no exclusion in it.
    """
    errors, reported = validate_spec(_spec(suffix="USDCC", exclude=True))

    assert errors == [], errors
    assert any("USDCC" in item and "dry-run warned" in item for item in reported), \
        reported


def test_a_bad_param_is_reported_against_the_signature():
    spec = _spec(suffx="USDC")
    errors, _ = validate_spec(spec)
    assert any("filter_quote_suffix" in e for e in errors), errors
    with pytest.raises(SpecError, match="rejected the call"):
        build_selection_fn(spec)(_universe(), None)
