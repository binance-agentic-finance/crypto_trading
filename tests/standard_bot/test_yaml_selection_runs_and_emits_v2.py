"""A YAML selection spec must RUN, and must emit the one output contract.

The existing selection tests covered compiling, ranking, validating and
registering — and all passed while three things were untrue:

1. ``run`` dispatched purely on ``run.mode``, so a selection spec declaring
   ``mode: backtest`` (which the shipped example does) fell into the
   single-instrument bar backtest. It fetched 1000 BTCUSDT candles, never looked
   at ``selection:``, printed ``trades=0`` and exited 0. Nothing was wrong with
   the spec and nothing said so.
2. The emitted envelope was ``selection/v1`` with a three-key payload, while the
   StandardBot route emitted the 45-key ``cyqnt.signal/v2`` for the same job — so
   "one unified output format" was true of one of the two paths, and a basket from
   YAML carried no provenance, no data quality and no ``auto_trade_eligible``.
3. Buzz is measured per *base token*, so the same score joined onto every quote
   pair: a top-5 came back as ``BTCUSDC, BTCUSDT, SOLUSDT, SOLUSDC, BNBUSDT`` —
   three assets in five slots, double weight on two of them.

None of that was visible from the outside, which is the point of these tests.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

from cyqnt_trd.blocks import strategy as blocks_strategy
from cyqnt_trd.standard_bot.core import DataSnapshot, SnapshotMeta, UniverseBundle
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import build_selection_fn
from cyqnt_trd.standard_bot.yaml_pipeline.spec import SpecError, register_from_yaml

DT = 1_785_000_000_000

SPEC_YAML = """
spec_version: "1.0"
target: standard_bot
strategy:
  id: %(sid)s
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary:
    interval: "1h"
selection:
  universe:
    - block: universe.filter_quote_volume
      params: { min_quote_volume: 1000 }
    - block: universe.augment_with_news
      with: [ticker_rank]
  score: news_mention_count
  top_k: 3
  long_when:  { cond: conditions.value_above, args: [news_bull_ratio, 0.55] }
%(extra)s
"""


def _spec_file(tmp_path, sid: str, extra: str = "") -> str:
    path = tmp_path / ("%s.yaml" % sid)
    path.write_text(SPEC_YAML % {"sid": sid, "extra": extra}, encoding="utf-8")
    return str(path)


def _universe() -> pd.DataFrame:
    """Several quote pairs of the same base asset, as the real feed returns."""
    symbols = ["BTCUSDT", "BTCUSDC", "ETHUSDT", "ETHUSDC", "SOLUSDT", "XRPUSDT"]
    turnover = [9e9, 4e8, 5e9, 3e8, 1e9, 8e8]
    return pd.DataFrame({"symbol": symbols, "quoteVolume": turnover,
                         "lastPrice": [65000.0, 65001.0, 3000.0, 3001.0, 75.0, 0.5]})


def _rank() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["BTC", "ETH", "SOL", "XRP"],
        "mention_count": [500, 400, 300, 200],
        "bullish_count": [80, 80, 80, 10],
        "bearish_count": [20, 20, 20, 90],
        "neutral_count": [0, 0, 0, 0],
        "unique_authors": [50, 40, 30, 20],
        "rank": [1, 2, 3, 4],
    })


def _snapshot(universe=None, rank=None, status=None) -> DataSnapshot:
    return DataSnapshot(
        version="mvp/v1",
        universe=UniverseBundle(as_of=DT, universe=universe, ticker_rank=rank),
        meta=SnapshotMeta(snapshot_id="probe", assembled_at=DT, decision_as_of=DT,
                          source_status=dict(status or {}), partial_ok=True),
    )


# --------------------------------------------------------------------------- #
# 1. the basket is top_k distinct BETS, not top_k rows                        #
# --------------------------------------------------------------------------- #


def _candidates(**selection_overrides):
    spec = {
        "selection": {
            "universe": [
                {"block": "universe.filter_quote_volume",
                 "params": {"min_quote_volume": 1000}},
                {"block": "universe.augment_with_news", "with": ["ticker_rank"]},
            ],
            "score": "news_mention_count",
            "top_k": 3,
            "long_when": {"cond": "conditions.value_above",
                          "args": ["news_bull_ratio", 0.55]},
            **selection_overrides,
        }
    }
    return build_selection_fn(spec)(_universe(), _rank())


def test_quote_pairs_of_one_asset_do_not_take_several_slots():
    symbols = [c["symbol"] for c in _candidates()]
    assert symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"], symbols
    bases = [s.replace("USDT", "").replace("USDC", "") for s in symbols]
    assert len(set(bases)) == len(bases), (
        "a basket of %d must be %d distinct assets: %s" % (len(symbols), len(symbols), symbols))


def test_the_surviving_pair_is_the_more_liquid_one():
    """Every pair of a token scores the same, so the tie-break decides the fill."""
    symbols = [c["symbol"] for c in _candidates()]
    assert "BTCUSDT" in symbols and "BTCUSDC" not in symbols, (
        "BTCUSDT turns over 9e9 vs BTCUSDC 4e8 — picking by response order is an "
        "arbitrary choice about where the order gets filled")


def test_dedupe_can_be_turned_off_for_a_cross_quote_strategy():
    symbols = [c["symbol"] for c in _candidates(dedupe_by="none")]
    assert symbols == ["BTCUSDT", "BTCUSDC", "ETHUSDT"], symbols


def test_an_unknown_dedupe_mode_is_refused():
    with pytest.raises(SpecError) as excinfo:
        _candidates(dedupe_by="by_vibes")
    assert "dedupe_by" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 2. the output is cyqnt.signal/v2                                            #
# --------------------------------------------------------------------------- #


@pytest.fixture
def registered(tmp_path):
    sid = "yaml_sel_v2_probe"
    register_from_yaml(_spec_file(tmp_path, sid))
    plugin = blocks_strategy.get_selection_plugin(sid)
    assert plugin is not None
    yield plugin


def test_the_emitted_envelope_is_the_v2_contract(registered):
    envelope = registered.run(_snapshot(_universe(), _rank()), None).signals[0]
    assert envelope.version == "cyqnt.signal/v2"
    assert envelope.kind.value == "selection"

    payload = envelope.payload
    # the fields a consumer/executor needs and the 3-key payload never had
    for key in ("schema", "intent", "closes_side", "order_side", "reduce_only",
                "market_scope", "data_quality", "provenance", "universe_size",
                "auto_trade_eligible", "requires_confirmation", "dedup_key"):
        assert key in payload, "%s missing from a v2 selection payload" % key
    assert payload["schema"] == "cyqnt.signal/v2"
    assert payload["market_scope"] == "cross_section"
    assert payload["intent"] == "hold", "a ranking is not itself an order"
    assert payload["auto_trade_eligible"] is False
    assert payload["provenance"]["strategy_id"] == "yaml_sel_v2_probe"


def test_a_yaml_basket_and_a_standardbot_basket_have_the_same_key_set(registered):
    """The actual claim being tested: ONE output format, both routes."""
    from strategies.standard.blocks_reference_bots import BlocksNewsRankBot
    from cyqnt_trd.standard_bot.bot import BotContext

    yaml_payload = registered.run(_snapshot(_universe(), _rank()), None).signals[0].payload

    universe = pd.DataFrame({"instrument_id": ["BTCUSDT", "ETHUSDT"],
                             "available_time": [DT, DT], "quote_volume": [9e8, 5e8]})
    rank = pd.DataFrame({"instrument_id": ["BTCUSDT", "ETHUSDT"],
                         "available_time": [DT, DT], "rank": [1, 2],
                         "score": [90.0, 50.0], "mention_count": [300, 200],
                         "bullish_count": [80, 50], "bearish_count": [20, 50]})
    bot_payload = BlocksNewsRankBot().run(
        BotContext(decision_time=DT,
                   frames={"universe": universe, "ticker_rank": rank})).signals[0].payload

    # The StandardBot route additionally stamps the pre-v2 engine bridge keys
    # (bot._engine_compat_payload) so the old bar-based runners can read a v2
    # signal. Those are transport shims for a *trade* signal, not part of the
    # contract, and a cross-sectional basket has no target position to express —
    # so they are excluded rather than faked.
    from cyqnt_trd.standard_bot.bot import _ENGINE_COMPAT_KEYS

    shims = set(_ENGINE_COMPAT_KEYS) | {"bar_timestamp", "size_unresolved"}
    contract_yaml = set(yaml_payload) - {"as_of"}
    contract_bot = set(bot_payload) - shims

    assert contract_yaml == contract_bot, (
        "one output format means one key set. only in YAML: %s / only in "
        "StandardBot: %s" % (sorted(contract_yaml - contract_bot),
                             sorted(contract_bot - contract_yaml)))
    assert len(contract_yaml) >= 40, (
        "expected the full v2 contract, got %d keys" % len(contract_yaml))


def test_candidates_carry_the_v2_candidate_shape(registered):
    candidates = registered.run(_snapshot(_universe(), _rank()), None).signals[0] \
        .payload["candidates"]
    assert candidates
    for candidate in candidates:
        assert set(candidate) == {"symbol", "rank", "score", "direction", "reason",
                                  "features", "trade"}
        assert candidate["direction"] in ("long", "short", "neutral")


def test_an_empty_basket_is_emitted_with_a_reason_not_a_crash(registered):
    """"Nothing qualified" is an answer a rebalancing selector has to be able to
    give; going silent is indistinguishable from a failure."""
    envelope = registered.run(_snapshot(pd.DataFrame(), _rank()), None).signals[0]
    payload = envelope.payload
    assert payload["candidates"] == []
    assert payload["reason_codes"], "an empty basket must say why"
    assert "no name passed" in payload["summary"]


def test_a_failed_source_shows_up_as_degraded_quality(registered):
    envelope = registered.run(
        _snapshot(_universe(), _rank(), status={"ticker_rank": "error: upstream 500"}),
        None).signals[0]
    assert envelope.payload["data_quality"] == "degraded"
    assert envelope.payload["source_status"]["ticker_rank"].startswith("error")


# --------------------------------------------------------------------------- #
# 3. `run` must not silently do the wrong thing                               #
# --------------------------------------------------------------------------- #


def test_run_routes_a_selection_spec_to_the_selector(tmp_path, monkeypatch, capsys):
    """Not into the per-bar backtest, which has no universe to rank."""
    from cyqnt_trd.standard_bot.yaml_pipeline import cli

    called = {}

    def _boom(*_args, **_kwargs):
        called["bar_backtest"] = True
        raise AssertionError("a selection spec must never reach the bar backtest")

    monkeypatch.setattr(cli, "_run_backtest_vectorized", _boom)
    monkeypatch.setattr(cli, "_run_backtest_event", _boom)

    # the selector's data comes from the catalog; serve it locally
    from cyqnt_trd.standard_bot.runtime import data as data_runtime

    def _call(self, node, **_params):
        if node == "universe":
            return _universe()
        if node == "ticker_rank":
            return _rank()
        raise AssertionError("unexpected node %r" % node)

    monkeypatch.setattr(data_runtime.DataSession, "call", _call)

    out = tmp_path / "sel.json"
    args = argparse.Namespace(spec=_spec_file(tmp_path, "yaml_sel_route_probe"),
                              output_json=str(out), engine="vectorized",
                              input_json=None, start=False)
    assert cli.cmd_run(args) == 0
    assert "bar_backtest" not in called

    printed = capsys.readouterr().out
    assert "selection strategy=yaml_sel_route_probe" in printed
    assert "cyqnt.signal/v2" in printed
    assert "not a backtest" in printed, (
        "mode: backtest on a selector must not read as a backtest result — a "
        "0-trade line looked like 'the strategy did nothing wrong'")

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == "cyqnt.signal-batch/v1"
    assert written["signal_count"] == 1
    signal = written["signals"][0]
    assert signal["schema"] == "cyqnt.signal/v2"
    assert signal["kind"] == "selection"
    assert len(signal) == 42
    assert [c["symbol"] for c in signal["candidates"]] == [
        "BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_an_empty_basket_that_evaluated_nothing_is_still_refused():
    """The rule kept its teeth when the empty case was allowed.

    Allowing "nothing qualified" must not also allow "I looked at nothing and
    everything is fine": empty basket + empty universe + GOOD quality is a bot
    reporting all-clear without having evaluated anything, and that is exactly the
    shape the original hard refusal existed to stop.
    """
    from cyqnt_trd.standard_bot.core import (DataQuality, MarketScope, PositionIntent,
                                             Provenance, StandardSignal)

    def build(**kwargs):
        return StandardSignal(
            bot_id="probe", decision_time=DT, provenance=Provenance(strategy_id="probe"),
            market_scope=MarketScope.CROSS_SECTION, intent=PositionIntent.HOLD, **kwargs)

    with pytest.raises(ValueError, match="empty basket"):
        build(reason_codes=("empty_basket",))          # said why, ranked nothing
    with pytest.raises(ValueError, match="empty basket"):
        build(universe_size=726)                       # ranked, but says nothing

    # ranked a real universe and nothing qualified — a legitimate answer
    build(universe_size=726, reason_codes=("empty_basket",))
    # could not read the universe, and says so
    build(data_quality=DataQuality.INSUFFICIENT, reason_codes=("empty_basket",))
