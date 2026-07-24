"""Tests for the generic configurable REST fetcher.

No network: the transport (``rest_source._request``) is monkeypatched to return
canned JSON, so the tests pin the *mechanism* — declarative field mapping,
coercions, envelope gating, sorting, per-call param overrides, external-config
loading, and the empty-typed-frame degradation contract.
"""

from __future__ import annotations

import json

import pytest

from cyqnt_trd.data_cli import rest_source
from cyqnt_trd.data_cli.rest_source import (
    FieldSpec,
    RestSourceSpec,
    fetch_rest,
    register_spec,
    load_specs,
    list_specs,
    EXAMPLE_SPECS,
)
from cyqnt_trd.data_cli._cache import cache_clear


@pytest.fixture(autouse=True)
def _clear():
    cache_clear()
    yield
    cache_clear()


def _spec(**over) -> RestSourceSpec:
    base = dict(
        name="demo",
        base_url="https://example.test",
        path="v1/series",
        data_path="data",
        fields={
            "timestamp": FieldSpec("ts", "ms"),
            "value": FieldSpec("val", "int"),
            "label": FieldSpec("cls", "str"),
        },
        sort_by="timestamp",
        ttl=60,
    )
    base.update(over)
    return RestSourceSpec(**base)


def _patch_resp(monkeypatch, resp):
    monkeypatch.setattr(rest_source, "_request", lambda spec: resp)


def test_maps_fields_coerces_and_sorts(monkeypatch):
    _patch_resp(monkeypatch, {"data": [
        {"ts": 1753086400, "val": "40", "cls": "Fear"},
        {"ts": 1753000000, "val": "55", "cls": "Greed"},
    ]})
    df = fetch_rest(_spec(), refresh=True)
    assert list(df.columns) == ["timestamp", "value", "label"]
    assert df["timestamp"].tolist() == [1753000000000, 1753086400000]  # sorted + s→ms
    assert df["value"].tolist() == [55, 40]


def test_dotted_key_and_top_level_list(monkeypatch):
    spec = _spec(data_path="", fields={"price": FieldSpec("quote.USD.price", "float")})
    _patch_resp(monkeypatch, [{"quote": {"USD": {"price": "123.5"}}}])
    df = fetch_rest(spec, refresh=True)
    assert df["price"].tolist() == [123.5]


def test_ok_check_failure_degrades_to_empty(monkeypatch):
    spec = _spec(ok_check={"key": "code", "equals": "0"})
    _patch_resp(monkeypatch, {"code": "500", "data": [{"ts": 1, "val": 2, "cls": "x"}]})
    df = fetch_rest(spec, refresh=True)
    assert df.empty and list(df.columns) == spec.columns


def test_transport_error_degrades_to_empty(monkeypatch):
    _patch_resp(monkeypatch, None)  # _request returns None on any transport failure
    df = fetch_rest(_spec(), refresh=True)
    assert df.empty and list(df.columns) == _spec().columns


def test_registry_and_param_override(monkeypatch):
    register_spec(_spec(name="reg_demo", params={"limit": 10}))
    assert "reg_demo" in list_specs()
    captured = {}

    def fake_request(spec):
        captured["params"] = dict(spec.params)
        return {"data": [{"ts": 1, "val": 1, "cls": "a"}]}

    monkeypatch.setattr(rest_source, "_request", fake_request)
    fetch_rest("reg_demo", params={"limit": 500, "symbol": "BTC"}, refresh=True)
    assert captured["params"] == {"limit": 500, "symbol": "BTC"}  # per-call override merged


def test_load_specs_from_external_config(tmp_path, monkeypatch):
    cfg = tmp_path / "sources.json"
    cfg.write_text(json.dumps([{
        "name": "from_file",
        "base_url": "https://example.test",
        "path": "x",
        "data_path": "rows",
        "fields": {"v": {"key": "v", "coerce": "int"}},
    }]))
    names = load_specs(str(cfg))
    assert names == ["from_file"]
    _patch_resp(monkeypatch, {"rows": [{"v": "7"}]})
    df = fetch_rest("from_file", refresh=True)
    assert df["v"].tolist() == [7]


def test_unknown_spec_raises():
    with pytest.raises(KeyError):
        fetch_rest("does_not_exist", refresh=True)


def test_example_spec_shape():
    # The shipped public example must be a valid, self-describing spec.
    fg = EXAMPLE_SPECS["fear_greed"]
    assert fg.base_url.startswith("https://")
    assert fg.columns == ["timestamp", "value", "value_classification"]
