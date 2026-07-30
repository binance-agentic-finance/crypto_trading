"""Coin selection, expressed in YAML.

The YAML dialect only ever described one shape of strategy: evaluate a frame of
*bars* for one instrument and emit long/short per bar. Selection is a different
shape — evaluate a frame of *symbols* once and emit a ranked basket — and the
dialect had no word for it, so the working selection engine underneath was
unreachable from a spec.

It turns out to need almost no new machinery: a column is a column, and
``conditions.value_above`` does not care whether the axis is time or instrument.
So ``selection:`` reuses the same block resolution, the same feature syntax and
the same comparators, with the universe frame in place of the bar frame.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (SpecError,
                                                              build_selection_fn)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import validate_spec

DT = 1_776_322_799_999
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
MENTIONS = [512, 388, 240, 175, 120, 96]
BULL = [0.81, 0.31, 0.62, 0.50, 0.28, 0.71]


def _spec(**selection_overrides):
    selection = {
        "universe": [
            {"block": "universe.filter_quote_volume",
             "params": {"min_quote_volume": 1e8}},
            {"block": "universe.augment_with_news", "with": ["ticker_rank"]},
        ],
        "score": "news_mention_count",
        "top_k": 5,
        "long_when": {"cond": "conditions.value_above",
                      "args": ["news_bull_ratio", 0.55]},
        "short_when": {"cond": "conditions.value_below",
                       "args": ["news_bull_ratio", 0.45]},
    }
    selection.update(selection_overrides)
    return {
        "strategy": {"id": "sel"},
        "run": {"mode": "backtest"},
        "data": {"symbol": "BTCUSDT", "primary": {"interval": "1h"}},
        "selection": selection,
    }


def _universe():
    return pd.DataFrame({"symbol": SYMBOLS,
                         "quoteVolume": [9e8, 7e8, 4e8, 3e8, 2e8, 1.2e8],
                         "available_time": [DT] * 6})


def _rank():
    return pd.DataFrame({"symbol": SYMBOLS, "rank": list(range(1, 7)),
                         "mention_count": MENTIONS, "bull_ratio": BULL,
                         "available_time": [DT] * 6})


def _run(spec):
    return build_selection_fn(spec)(_universe(), _rank())


# --------------------------------------------------------------------------- #
# it produces a basket                                                         #
# --------------------------------------------------------------------------- #


def test_a_selection_spec_ranks_the_universe():
    candidates = _run(_spec())
    assert [c["symbol"] for c in candidates] == ["BTCUSDT", "ETHUSDT", "SOLUSDT",
                                                 "XRPUSDT"]
    assert [c["score"] for c in candidates] == [512.0, 388.0, 240.0, 120.0]


def test_direction_comes_from_the_declared_rules():
    directions = {c["symbol"]: c["side"] for c in _run(_spec())}
    assert directions["BTCUSDT"] == "long"      # 0.81
    assert directions["ETHUSDT"] == "short"     # 0.31
    assert directions["SOLUSDT"] == "long"      # 0.62
    assert "BNBUSDT" not in directions, "0.50 satisfies neither rule"


def test_ranks_are_contiguous_and_the_total_matches_the_basket():
    """Numbering during the loop left holes where a symbol failed the direction
    filter, and reported "rank 5 of 5" beside a list of four."""
    candidates = _run(_spec())
    assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))
    for candidate in candidates:
        assert "of %d" % len(candidates) in candidate["reason"]


def test_top_k_and_min_score_narrow_the_basket():
    assert len(_run(_spec(top_k=2))) <= 2
    assert all(c["score"] >= 300 for c in _run(_spec(min_score=300)))


def test_features_are_computed_along_the_symbol_axis():
    """The same block, a different axis: ``rolling_zscore`` over the ranking."""
    spec = _spec(features={"buzz_z": {"block": "indicators.rolling_zscore",
                                      "input": "news_mention_count",
                                      "params": {"period": 3}}},
                 score="buzz_z")
    candidates = _run(spec)
    assert candidates
    assert any("buzz_z" in c["features"] for c in candidates)


def test_a_symbol_whose_score_is_unknown_is_dropped_not_defaulted():
    """A NaN ranked against real numbers sorts arbitrarily and then reads as a
    recommendation."""
    rank = _rank()
    rank.loc[rank["symbol"] == "BTCUSDT", "mention_count"] = float("nan")
    candidates = build_selection_fn(_spec())(_universe(), rank)
    assert "BTCUSDT" not in {c["symbol"] for c in candidates}


def test_an_empty_universe_yields_no_candidates():
    assert build_selection_fn(_spec())(pd.DataFrame(), _rank()) == []


# --------------------------------------------------------------------------- #
# validation                                                                   #
# --------------------------------------------------------------------------- #


def test_a_selection_spec_validates_and_dry_runs():
    errors, _ = validate_spec(_spec())
    assert errors == [], errors


def test_score_is_required():
    spec = _spec()
    del spec["selection"]["score"]
    errors, _ = validate_spec(spec)
    assert any("selection.score is required" in e for e in errors), errors


def test_a_spec_cannot_be_both_kinds_at_once():
    """They emit different signal kinds; a spec that is both is a spec that has
    not decided what it is."""
    spec = _spec()
    spec["signals"] = {"entry": {"long": {"cond": "conditions.rsi_overbought",
                                          "args": ["close"]}}}
    errors, _ = validate_spec(spec)
    assert any("not both" in e for e in errors), errors


def test_a_bad_block_in_the_universe_pipeline_is_caught_statically():
    spec = _spec(universe=[{"block": "universe.no_such_filter"}])
    errors, _ = validate_spec(spec)
    assert any("not a callable block" in e for e in errors), errors


def test_an_unknown_selection_key_is_refused():
    spec = _spec(topk=5)
    errors, _ = validate_spec(spec)
    assert any("unknown selection.topk" in e for e in errors), errors


def test_a_universe_step_returning_a_series_is_refused():
    """Each step must hand on the frame; a Series means the pipeline collapsed."""
    spec = _spec(universe=[{"block": "conditions.value_above",
                            "params": {"threshold": 1.0}}])
    errors, _ = validate_spec(spec)
    assert any("selection dry-run failed" in e or "must return" in e
               for e in errors), errors


# --------------------------------------------------------------------------- #
# it registers and publishes the one contract                                  #
# --------------------------------------------------------------------------- #


def test_it_registers_as_a_selection_strategy_not_a_trade_one(tmp_path):
    import yaml

    from cyqnt_trd.blocks import strategy as blocks_strategy
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import register_from_yaml

    spec = _spec()
    spec["strategy"]["id"] = "yaml_sel_probe"
    path = tmp_path / "sel.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    register_from_yaml(str(path))
    assert blocks_strategy.is_known_selection_strategy("yaml_sel_probe")
    assert not blocks_strategy.is_known_block_strategy("yaml_sel_probe"), (
        "registering a basket as a trade strategy is how a selection result "
        "ends up parsed as a per-bar signal")


def test_the_published_signal_is_the_same_schema_as_a_trade_signal(tmp_path):
    """The whole point: one output contract, whichever shape the spec is."""
    import yaml

    from cyqnt_trd.blocks import strategy as blocks_strategy
    from cyqnt_trd.standard_bot.adapter import envelope_to_signal
    from cyqnt_trd.standard_bot.core import (DataSnapshot, MarketScope, SnapshotMeta,
                                             UniverseBundle)
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import register_from_yaml

    spec = _spec()
    spec["strategy"]["id"] = "yaml_sel_publish"
    path = tmp_path / "sel.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    register_from_yaml(str(path))

    plugin = blocks_strategy.get_selection_plugin("yaml_sel_publish")
    snapshot = DataSnapshot(
        version="1",
        universe=UniverseBundle(universe=_universe(), ticker_rank=_rank(), as_of=DT),
        meta=SnapshotMeta(snapshot_id="s", decision_as_of=DT, assembled_at=DT))
    envelope = plugin.step(snapshot, {}, {"market_type": "futures"}).signals[0]

    payload = envelope_to_signal(envelope).to_dict()
    assert payload["schema"] == "cyqnt.signal/v2"
    assert payload["kind"] == "selection"
    assert payload["market_scope"] == MarketScope.CROSS_SECTION.value
    assert len(payload) == 42
    assert [c["symbol"] for c in payload["candidates"]][:2] == ["BTCUSDT", "ETHUSDT"]
