# =============================================================================
# 兩個「一個物件兩層講不同的話」的缺陷
# =============================================================================
# 這兩個和先前那個空籃子 kind bug 是同一個形狀:同一個訊號,envelope 那一層與
# payload 那一層對同一件事給不同答案,而每一個欄位單獨看都很合理。
#
#   plugin.market_type = spot    →  payload.product      = usd_m_perpetual
#   envelope.time_horizon = swing →  payload.time_horizon = intraday
#
# product 那個最嚴重,因為 **product 是執行層路由用的欄位** —— 一份現貨選幣的籃子
# 會被交給永續場所,而輸出裡沒有任何東西反駁它:market_scope 是 cross_section
# (對的)、kind 是 selection(對的)、候選也都是真的現貨標的。
#
# 交易路徑一直都是對的(blocks/strategy.py 上方那個 sibling 就在 derive product),
# 只有選幣路徑漏了,於是它吃了 dataclass 的預設值。
#
# 順帶:fetch_contract_meta('spot') 的行為在這裡一起釘,因為它是同一個 spot 場景的
# 另一半 —— 這兩件事是同一次實驗撞到的。
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyqnt_trd.blocks.data import fetch_contract_meta
from cyqnt_trd.standard_bot.data import load_input_bundle
from cyqnt_trd.standard_bot.yaml_pipeline import bundle_runner
from cyqnt_trd.standard_bot.yaml_pipeline.spec import load_spec

REPO = Path(__file__).parents[2]
FIXTURE = REPO / "tests" / "standard_bot" / "fixtures" / "universe_derivatives.json"

SPEC = {
    "spec_version": "1.0", "target": "standard_bot",
    "strategy": {"id": "product_probe", "description": "market_type 對照"},
    "run": {"mode": "backtest"},
    "data": {"symbol": "BTCUSDT", "market_type": "futures",
             "primary": {"interval": "1h"}},
    "selection": {
        "universe": [{"block": "universe.filter_quote_volume",
                      "params": {"min_quote_volume": 1.0e7}}],
        "score": "quoteVolume", "top_k": 3, "dedupe_by": "base_asset",
        "long_when": {"cond": "conditions.value_above", "args": ["quoteVolume", 0]},
    },
}


def _envelope(market_type: str):
    spec = json.loads(json.dumps(SPEC))
    spec["data"]["market_type"] = market_type
    spec["strategy"]["id"] = "product_probe_" + market_type
    snapshot = load_input_bundle(str(FIXTURE))
    return bundle_runner._build_plugin(spec)._build_envelope(snapshot, {})


# ---------------------------------------------------------------------------
# product: the field an executor routes on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("market_type, expected", [
    ("spot", "spot"),
    ("futures", "usd_m_perpetual"),
])
def test_the_published_product_follows_the_declared_market_type(market_type, expected):
    assert _envelope(market_type).payload["product"] == expected


def test_a_spot_basket_is_not_labelled_perpetual():
    """Named separately because this is the direction that costs money.

    ``product`` is what an executor routes on, so a spot basket carrying
    "usd_m_perpetual" is handed to the wrong venue — and nothing else in the
    output disagrees: kind is selection, market_scope is cross_section, and the
    candidates really are spot instruments. Every field is right except the one
    that decides where the order goes.
    """
    payload = _envelope("spot").payload
    assert payload["product"] != "usd_m_perpetual"
    assert payload["market_scope"] == "cross_section"
    assert payload["kind"] == "selection"


# ---------------------------------------------------------------------------
# time_horizon: the two layers must agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("market_type", ["spot", "futures"])
def test_the_envelope_and_its_payload_agree_on_the_horizon(market_type):
    """A cross-sectional screen is one decision at one instant and carries no
    holding period, so neither value is a measurement — but a reader must not get
    two answers from one object."""
    envelope = _envelope(market_type)
    assert str(envelope.time_horizon) == envelope.payload["time_horizon"]


# ---------------------------------------------------------------------------
# fetch_contract_meta('spot'): refuse, and say what the venue has
# ---------------------------------------------------------------------------

def test_spot_contract_meta_is_refused_rather_than_invented():
    """Three ways to serve a spot caller; two of them are silent.

    * a frame missing the three asset-class columns → the join gives every row a
      NaN ``underlying_type``, and ``filter_crypto_only`` drops the WHOLE
      universe for being unknown while looking like "not crypto"
    * ``underlyingType="COIN"`` for every row → true today, a silent lie the day
      a tokenised equity is listed on spot
    * refuse and explain → this

    Asserted on the message, not just the raise, because the whole value of this
    behaviour is that the caller learns spot has no asset class and does not need
    one.
    """
    with pytest.raises(RuntimeError) as excinfo:
        fetch_contract_meta("spot")

    message = str(excinfo.value)
    assert "no asset class for spot" in message
    assert "USDⓈ-M only" in message, "the caller has to learn why it is not needed"
    assert "filter_quote_suffix" in message, "and what to use for the quote leg"


def test_futures_contract_meta_still_answers():
    """The mutation side: a guard that refused both would pass the test above."""
    frame = fetch_contract_meta("futures")
    assert len(frame) > 100
    for column in ("contractType", "underlyingType", "underlyingSubType"):
        assert column in frame.columns


def test_tokenised_equities_are_reachable_and_labelled():
    """The correction that started this: NVDAUSDT et al are REAL products.

    An earlier version of the experiment harness injected
    ``filter_crypto_only`` unconditionally as a "safety floor". That is the same
    silent failure as the original incident, in the other direction: a user who
    asks for NVDA gets it quietly removed. Both directions are silent — one hands
    over what was not asked for, the other takes away what was.
    """
    frame = fetch_contract_meta("futures")
    equities = frame[frame["underlyingType"] == "EQUITY"]
    assert len(equities) > 100, "USDⓈ-M carries a whole tokenised-equity book"

    listed = set(frame["symbol"])
    for symbol in ("NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "QQQUSDT"):
        assert symbol in listed, symbol
        row = frame[frame["symbol"] == symbol].iloc[0]
        assert row["underlyingType"] == "EQUITY"
        assert row["contractType"] == "TRADIFI_PERPETUAL"
