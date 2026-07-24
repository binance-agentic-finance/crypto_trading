"""Tests for the ready-to-use PUBLIC source specs.

No network: ``rest_source._request`` is monkeypatched with canned JSON matching
each public endpoint's real shape, so the tests pin the spec definitions (field
mapping, coercion, sort) and the registration/tier metadata — without hitting
Binance / alternative.me.
"""

from __future__ import annotations

import pytest

from cyqnt_trd.data_cli import rest_source
from cyqnt_trd.data_cli import (
    fetch_rest,
    register_public_sources,
    PUBLIC_SOURCES,
    SOURCE_TIERS,
)
from cyqnt_trd.data_cli._cache import cache_clear


@pytest.fixture(autouse=True)
def _setup():
    cache_clear()
    register_public_sources()
    yield
    cache_clear()


def _patch(monkeypatch, resp):
    monkeypatch.setattr(rest_source, "_request", lambda spec: resp)


def test_all_public_sources_registered():
    names = register_public_sources()
    assert set(names) == set(PUBLIC_SOURCES)
    # every registered source has a backtestability tier
    assert set(SOURCE_TIERS) == set(PUBLIC_SOURCES)
    assert set(SOURCE_TIERS.values()) <= {"BACKTESTABLE", "SEMI"}


def test_funding_rate_shape(monkeypatch):
    _patch(monkeypatch, [
        {"symbol": "BTCUSDT", "fundingTime": 1784851200000, "fundingRate": "0.00007598", "markPrice": "65069.66"},
    ])
    df = fetch_rest("funding_rate", refresh=True)
    assert list(df.columns) == ["symbol", "timestamp", "funding_rate", "mark_price"]
    assert df["funding_rate"].tolist() == [7.598e-05]
    assert df["timestamp"].tolist() == [1784851200000]


def test_open_interest_shape(monkeypatch):
    _patch(monkeypatch, [
        {"symbol": "BTCUSDT", "sumOpenInterest": "104877.15", "sumOpenInterestValue": "6826642993.1",
         "timestamp": 1784862000000},
    ])
    df = fetch_rest("open_interest", refresh=True)
    assert df["oi_base"].tolist() == [104877.15]
    assert df["oi_value"].tolist() == [6826642993.1]


def test_ls_account_shape(monkeypatch):
    _patch(monkeypatch, [
        {"symbol": "BTCUSDT", "longAccount": "0.6079", "shortAccount": "0.3921",
         "longShortRatio": "1.5504", "timestamp": 1784862000000},
    ])
    df = fetch_rest("ls_account", refresh=True)
    assert df["long_account"].tolist() == [0.6079]
    assert df["long_short_ratio"].tolist() == [1.5504]


def test_taker_ratio_shape(monkeypatch):
    _patch(monkeypatch, [
        {"buySellRatio": "1.0284", "buyVol": "1638.28", "sellVol": "1593.08", "timestamp": 1784858400000},
    ])
    df = fetch_rest("taker_ratio", refresh=True)
    assert df["buy_sell_ratio"].tolist() == [1.0284]
    assert list(df.columns) == ["timestamp", "buy_sell_ratio", "buy_vol", "sell_vol"]


def test_fear_greed_seconds_to_ms(monkeypatch):
    _patch(monkeypatch, {"data": [
        {"value": "28", "value_classification": "Fear", "timestamp": "1784851200"},
    ]})
    df = fetch_rest("fear_greed", refresh=True)
    # alternative.me returns epoch-seconds → coerced to ms
    assert df["timestamp"].tolist() == [1784851200000]
    assert df["value"].tolist() == [28]


def test_per_call_param_override(monkeypatch):
    captured = {}

    def fake(spec):
        captured["params"] = dict(spec.params)
        return [{"symbol": "ETHUSDT", "fundingTime": 1, "fundingRate": "0.0", "markPrice": "0"}]

    monkeypatch.setattr(rest_source, "_request", fake)
    fetch_rest("funding_rate", params={"symbol": "ETHUSDT", "limit": 5}, refresh=True)
    assert captured["params"]["symbol"] == "ETHUSDT"
    assert captured["params"]["limit"] == 5


def test_transport_error_degrades_to_empty(monkeypatch):
    _patch(monkeypatch, None)
    df = fetch_rest("funding_rate", refresh=True)
    assert df.empty and list(df.columns) == PUBLIC_SOURCES["funding_rate"].columns
