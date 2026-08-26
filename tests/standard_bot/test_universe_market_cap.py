"""Market cap: the supply fetcher, the join, and the two ways it lies.

The two failure modes these pin down, because neither shows up in a basket:

* a supply of ZERO is the venue's correct answer for a synthetic instrument
  (BTCDOMUSDT, ALLUSDT, XAUUSDT), and multiplied by a price it becomes a market
  cap of zero — a number a ``min_usd`` floor drops for being SMALL;
* Binance lists tokenised equities, so "top n by market cap" over an unnarrowed
  venue is a list of American corporations, and the rows look like any other.
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest

from cyqnt_trd.blocks import data as blocks_data
from cyqnt_trd.blocks import universe as ub

REPO = Path(__file__).resolve().parents[2]
SPEC_MARKET_CAP = (REPO / "docs" / "strategy_yaml_spec"
                   / "example_market_cap_screen.yaml")


def _supply_payload(supplies, start_ms: int = 1_787_000_000_000):
    """``openInterestHist`` rows carrying a supply series, oldest first."""
    return [
        {
            "symbol": "X",
            "sumOpenInterest": "10",
            "sumOpenInterestValue": "1000",
            "CMCCirculatingSupply": str(value),
            "timestamp": start_ms + index * 86_400_000,
        }
        for index, value in enumerate(supplies)
    ]


def _universe(symbols, price=2.0):
    return pd.DataFrame({"symbol": list(symbols),
                         "lastPrice": [price] * len(symbols),
                         "quoteVolume": [1e9] * len(symbols)})


# --------------------------------------------------------------------------- #
# the fetcher                                                                  #
# --------------------------------------------------------------------------- #

def test_only_the_latest_reading_is_emitted_and_the_flat_run_before_it_counted(
    monkeypatch
):
    """The series is read to measure staleness, not to be returned.

    A universe join wants one row per instrument; the earlier readings exist
    only to answer "has this number moved recently". Observed on BTCUSDT: six
    daily readings frozen at 20,071,518 before a single +3,216 catch-up, with
    nothing in the payload marking the frozen days.
    """
    monkeypatch.setattr(blocks_data, "_request_json",
                        lambda url, params: _supply_payload([100, 110, 120, 120, 120]))

    frame = blocks_data.fetch_circulating_supply_cross_section(["AAAUSDT"])

    assert frame["circulating_supply"].tolist() == [120.0]
    # Three readings carry 120; two of them precede the latest.
    assert frame["supply_unchanged_periods"].tolist() == [2]


def test_a_moving_supply_reports_zero_unchanged_periods(monkeypatch):
    """The flat-run counter must not fire on the healthy case."""
    monkeypatch.setattr(blocks_data, "_request_json",
                        lambda url, params: _supply_payload([100, 110, 120]))

    frame = blocks_data.fetch_circulating_supply_cross_section(["AAAUSDT"])

    assert frame["supply_unchanged_periods"].tolist() == [0]


def test_the_venues_literal_zero_becomes_unknown_rather_than_a_zero_cap(monkeypatch):
    """BTCDOMUSDT / ALLUSDT / XAUUSDT answer 0 because they have no token.

    Kept as 0 it prices them at a market cap of zero, which ``min_usd`` drops
    for being SMALL — the instrument then looks screened rather than absent.
    """
    monkeypatch.setattr(blocks_data, "_request_json",
                        lambda url, params: _supply_payload([0, 0, 0]))

    frame = blocks_data.fetch_circulating_supply_cross_section(["BTCDOMUSDT"])

    assert frame["circulating_supply"].isna().all()


def test_an_instrument_with_no_readings_is_left_out_without_failing(monkeypatch):
    """Same market state as the open-interest history: read it, it was empty."""
    def fake(url, params):
        return [] if params["symbol"] == "NEWUSDT" else _supply_payload([5, 6])

    monkeypatch.setattr(blocks_data, "_request_json", fake)

    frame = blocks_data.fetch_circulating_supply_cross_section(["AAAUSDT", "NEWUSDT"])

    assert frame["symbol"].tolist() == ["AAAUSDT"]


def test_a_dropped_supply_field_names_the_constant_to_update(monkeypatch):
    """A schema change must not degrade into a NaN column."""
    def fake(url, params):
        rows = _supply_payload([5])
        del rows[0]["CMCCirculatingSupply"]
        return rows

    monkeypatch.setattr(blocks_data, "_request_json", fake)

    with pytest.raises(RuntimeError, match="_SUPPLY_HIST_FIELDS"):
        blocks_data.fetch_circulating_supply_cross_section(["AAAUSDT"])


# --------------------------------------------------------------------------- #
# the join                                                                     #
# --------------------------------------------------------------------------- #

def test_the_cap_is_the_product_and_the_price_it_used_is_kept(monkeypatch):
    """A cap is two readings taken at different instants; a reader who sees
    only the product cannot tell a supply change from a price move."""
    supply = pd.DataFrame({"symbol": ["AAAUSDT"], "circulating_supply": [1_000.0]})

    joined = ub.augment_with_market_cap(_universe(["AAAUSDT"], price=2.5), supply)

    assert joined["market_cap_usd"].tolist() == [2_500.0]
    assert joined["market_cap_price"].tolist() == [2.5]


def test_a_frame_with_no_price_column_is_refused(monkeypatch):
    """Without a price the supply cannot become a cap, and a NaN cap column
    reads downstream as a small instrument."""
    supply = pd.DataFrame({"symbol": ["AAAUSDT"], "circulating_supply": [1_000.0]})

    with pytest.raises(ValueError, match="no price column"):
        ub.augment_with_market_cap(pd.DataFrame({"symbol": ["AAAUSDT"]}), supply)


def test_a_cross_section_with_no_usable_supply_at_all_is_a_failed_capture():
    """One synthetic instrument is a market state; a whole frame of them is not
    a screen anyone asked for, so it is refused rather than emptied."""
    supply = pd.DataFrame({"symbol": ["BTCDOMUSDT", "ALLUSDT"],
                           "circulating_supply": [0.0, 0.0]})

    with pytest.raises(ValueError, match="not one of the"):
        ub.augment_with_market_cap(_universe(["BTCDOMUSDT", "ALLUSDT"]), supply)


def test_a_partial_supply_frame_is_refused_before_it_becomes_a_short_basket():
    """Below the coverage floor the missing rows would carry NaN and every
    threshold below would return a basket short for an unrecorded reason."""
    universe = _universe([f"C{index}USDT" for index in range(10)])
    supply = pd.DataFrame({"symbol": ["C0USDT"], "circulating_supply": [1e6]})

    with pytest.raises(ValueError, match="covers only 1 of 10"):
        ub.augment_with_market_cap(universe, supply)


def test_the_supply_column_may_arrive_under_the_venues_own_spelling():
    """A caller holding a raw ``openInterestHist`` frame should not have to
    rename a column to use this block."""
    supply = pd.DataFrame({"symbol": ["AAAUSDT"], "CMCCirculatingSupply": [1_000.0]})

    joined = ub.augment_with_market_cap(_universe(["AAAUSDT"], price=3.0), supply)

    assert joined["market_cap_usd"].tolist() == [3_000.0]


# --------------------------------------------------------------------------- #
# the screens                                                                  #
# --------------------------------------------------------------------------- #

def test_a_screen_with_no_bound_is_refused():
    """With none it returns the frame unchanged, which is indistinguishable
    from the step not being there."""
    frame = pd.DataFrame({"symbol": ["AAAUSDT"], "market_cap_usd": [1.0]})

    with pytest.raises(ValueError, match="at least one bound"):
        ub.filter_market_cap(frame)


def test_unknown_caps_are_dropped_and_counted_rather_than_screened():
    frame = pd.DataFrame({"symbol": ["AAAUSDT", "BTCDOMUSDT"],
                          "market_cap_usd": [5e9, float("nan")]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kept = ub.filter_market_cap(frame, min_usd=1e9)

    assert kept["symbol"].tolist() == ["AAAUSDT"]
    assert any("market_cap_usd" in str(item.message) for item in caught)


def test_the_ceiling_is_what_makes_a_small_cap_screen_a_small_cap_screen():
    """Written with only a floor, "small caps" returns BTC first."""
    frame = pd.DataFrame({"symbol": ["BIG", "SMALL"],
                          "market_cap_usd": [1e12, 5e8]})

    assert ub.filter_market_cap(frame, max_usd=1e9)["symbol"].tolist() == ["SMALL"]


def test_a_ranking_excludes_unknowns_rather_than_padding_its_tail_with_them():
    """``nlargest`` alone puts the synthetic instruments at positions 3-4 as
    soon as n exceeds the number of real readings, where they read as small
    caps rather than as absent ones."""
    frame = pd.DataFrame({"symbol": ["AAA", "BBB", "BTCDOMUSDT", "ALLUSDT"],
                          "market_cap_usd": [2e9, 1e9, float("nan"), float("nan")]})

    assert ub.top_market_cap(frame, 4)["symbol"].tolist() == ["AAA", "BBB"]


def test_screening_the_crypto_first_is_what_changes_the_head_of_the_ranking():
    """Binance lists tokenised equities and they are enormous: on a 2026-08-26
    snapshot NVDAB, AAPLB, MSFT, AMZN, SPCX, META and AVGO all outrank BTC.
    Nothing in the emitted basket says the top of it is not crypto.
    """
    frame = pd.DataFrame({
        "symbol": ["NVDABUSDT", "AAPLBUSDT", "BTCUSDT", "ETHUSDT"],
        "market_cap_usd": [5.136e12, 4.545e12, 1.566e12, 2.97e11],
        "underlying_type": ["EQUITY", "EQUITY", "COIN", "COIN"],
    })

    assert ub.top_market_cap(frame, 2)["symbol"].tolist() == ["NVDABUSDT", "AAPLBUSDT"]
    crypto = ub.filter_crypto_only(frame)
    assert ub.top_market_cap(crypto, 2)["symbol"].tolist() == ["BTCUSDT", "ETHUSDT"]


def test_the_shipped_spec_validates_and_warns_about_nothing():
    """A warning in a shipped example trains its readers to skim the report.

    Also the standing probe behind the capability row: ``market_cap`` is
    ``expressible`` at cross_section scope because this spec validates clean,
    so if it stops doing so the row is stale and this is what says it first.
    """
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import load_spec, validate_spec

    errors, spec_warnings = validate_spec(load_spec(str(SPEC_MARKET_CAP)))

    assert errors == []
    assert spec_warnings == []


def test_the_shipped_spec_still_refuses_the_augment_placed_before_narrowing():
    """The negative half of the probe, kept next to the positive one.

    Hoisting the fan-out above ``filter_crypto_only`` is the mistake the
    ordering rule exists for, and it fails by COVERAGE rather than by anything
    naming the order — 87 % against a 90 % floor. A regression that relaxed the
    floor would make the shipped spec's ordering advice decorative.
    """
    import copy

    from cyqnt_trd.standard_bot.yaml_pipeline.spec import load_spec, validate_spec

    spec = copy.deepcopy(load_spec(str(SPEC_MARKET_CAP)))
    steps = spec["selection"]["universe"]
    hoisted = next(step for step in steps
                   if step.get("block") == "universe.augment_with_market_cap")
    steps.remove(hoisted)
    steps.insert(0, hoisted)

    errors, _spec_warnings = validate_spec(spec)

    assert any("covers only" in str(error) for error in errors)
