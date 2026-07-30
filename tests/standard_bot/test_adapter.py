"""The bridge between the block/YAML path and the v2 path.

The repo had two ways to write a strategy and each was missing exactly what the
other had. A YAML strategy could be backtested, papered and traded live, but it
published ``SignalEnvelope`` (``block/v1``, 10 keys) — not the contract handed
to consumers. A ``StandardBot`` published ``cyqnt.signal/v2`` and could read
every declared source and do coin selection, but no CLI could name it.

These tests pin the two translations and, more importantly, the one thing the
translation must NOT do: invent position knowledge an envelope never carried.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.adapter import (AdapterError, batch_to_signals,
                                            envelope_to_signal, register_standard_bot)
from cyqnt_trd.standard_bot.core import (MarketScope, PositionIntent, SignalEnvelope,
                                         SignalKind, SignalProvenance, SizeMode,
                                         TradeSide)

DT = 1_776_322_799_999


def _envelope(**overrides):
    payload = {"bar_timestamp": DT, "target_position": 1, "engine_size": 0.5,
               "exit_spec": {"type": "atr_stop_tp", "max_bars": 120, "side": "long",
                             "stop_mult": 2.0, "tp_mult": 4.0, "atr_at_entry": 41.5}}
    payload.update(overrides.pop("payload", {}))
    base = dict(
        version="block/v1", signal_id="sig-1", kind=SignalKind.TRADE, strength=1.0,
        provenance=SignalProvenance(plugin_id="yaml_bot", plugin_version="block/v1",
                                    config_hash="BTCUSDT|1h", input_fingerprint="snap-1"),
        instrument_id="BTCUSDT", side=TradeSide.BUY, time_horizon="swing",
        valid_until=DT + 3_600_000, payload=payload,
    )
    base.update(overrides)
    return SignalEnvelope(**base)


# --------------------------------------------------------------------------- #
# block / YAML -> v2                                                           #
# --------------------------------------------------------------------------- #


def test_a_block_envelope_becomes_a_full_v2_signal():
    signal = envelope_to_signal(_envelope())
    payload = signal.to_dict()
    assert payload["schema"] == "cyqnt.signal/v2"
    assert payload["kind"] == "trade"
    assert len(payload) == 42, sorted(payload)
    assert payload["symbol"] == "BTCUSDT"
    assert payload["bot_id"] == "yaml_bot"
    assert payload["provenance"]["snapshot_id"] == "snap-1"


def test_the_exit_spec_round_trips_into_an_exit_plan():
    """The engines' flat dict is structured back into the published shape."""
    plan = envelope_to_signal(_envelope()).exit_plan
    assert plan is not None
    assert plan.stop_loss.atr_mult == 2.0
    assert plan.stop_loss.atr_value == pytest.approx(41.5)
    assert not plan.stop_loss.trailing
    assert [leg.atr_mult for leg in plan.take_profit] == [4.0]
    assert plan.time_stop.max_bars == 120


def test_a_trailing_stop_is_recognised_as_trailing():
    signal = envelope_to_signal(_envelope(payload={
        "exit_spec": {"type": "atr_trailing_stop", "max_bars": 80,
                      "trail_mult": 3.0, "atr_at_entry": 12.0}}))
    assert signal.exit_plan.stop_loss.trailing
    assert signal.exit_plan.stop_loss.atr_mult == 3.0


def test_percent_and_absolute_exits_both_translate():
    pct = envelope_to_signal(_envelope(payload={
        "exit_spec": {"type": "pct_stop_tp", "stop_pct": 0.02, "tp_pct": 0.04}}))
    assert pct.exit_plan.stop_loss.pct == pytest.approx(0.02)
    assert pct.exit_plan.take_profit[0].pct == pytest.approx(0.04)

    absolute = envelope_to_signal(_envelope(payload={
        "exit_spec": {"type": "pct_stop_tp", "stop_loss_price": 74433.2,
                      "take_profit_price": 76000.0}}))
    assert absolute.exit_plan.stop_loss.price == pytest.approx(74433.2)
    assert absolute.exit_plan.take_profit[0].price == pytest.approx(76000.0)


def test_an_entry_with_no_exit_spec_still_states_a_way_out():
    """The contract refuses an entry with no exit. An envelope without an
    exit_spec means "exit on the opposite signal" — a real plan, so say it."""
    signal = envelope_to_signal(_envelope(payload={"exit_spec": None}))
    assert signal.exit_plan is not None
    assert signal.exit_plan.exit_on_opposite_signal


def test_size_becomes_an_equity_fraction():
    size = envelope_to_signal(_envelope()).size
    assert size.mode is SizeMode.EQUITY_PCT and size.value == pytest.approx(0.5)


@pytest.mark.parametrize("target,intent", [
    (1, PositionIntent.OPEN_LONG),
    (-1, PositionIntent.OPEN_SHORT),
    (0, PositionIntent.FLAT),
])
def test_target_position_maps_to_the_intent_it_can_justify(target, intent):
    signal = envelope_to_signal(_envelope(payload={"target_position": target,
                                                   "exit_spec": None}))
    assert signal.intent is intent


def test_the_adapter_never_invents_a_close_intent():
    """An envelope carries a TARGET, not the current position.

    ``CLOSE_LONG`` and ``OPEN_SHORT`` are both "target = short" to the block
    path; distinguishing them needs exposure the envelope never had. Emitting a
    close on a guess is how an executor flattens a position nobody asked it to
    touch, so the translation stays at the intent the data supports.
    """
    for target in (1, -1, 0):
        signal = envelope_to_signal(_envelope(payload={"target_position": target,
                                                       "exit_spec": None}))
        assert signal.intent not in (
            PositionIntent.CLOSE_LONG, PositionIntent.CLOSE_SHORT,
            PositionIntent.REDUCE_LONG, PositionIntent.REDUCE_SHORT,
            PositionIntent.FLIP_TO_LONG, PositionIntent.FLIP_TO_SHORT,
            PositionIntent.ADD_LONG, PositionIntent.ADD_SHORT,
        )


def test_an_undateable_envelope_is_refused():
    envelope = _envelope(valid_until=None, payload={"bar_timestamp": None})
    with pytest.raises(AdapterError, match="cannot date the signal"):
        envelope_to_signal(envelope)


def test_a_selection_envelope_becomes_a_selection_signal():
    envelope = _envelope(
        kind=SignalKind.SELECTION, instrument_id=None, side=TradeSide.FLAT,
        payload={"bar_timestamp": DT, "universe_size": 40, "candidates": [
            {"symbol": "BTCUSDT", "rank": 1, "score": 9.0, "side": "long"},
            {"symbol": "ETHUSDT", "rank": 2, "score": 4.0, "side": "short"},
        ]})
    payload = envelope_to_signal(envelope).to_dict()
    assert payload["kind"] == "selection"
    assert payload["market_scope"] == MarketScope.CROSS_SECTION.value
    assert payload["intent"] == "hold"
    assert [c["direction"] for c in payload["candidates"]] == ["long", "short"]
    assert payload["universe_size"] == 40


def test_an_empty_selection_envelope_is_refused():
    envelope = _envelope(kind=SignalKind.SELECTION, instrument_id=None,
                         payload={"bar_timestamp": DT, "candidates": []})
    with pytest.raises(AdapterError, match="no candidates"):
        envelope_to_signal(envelope)


def test_a_v2_payload_riding_an_envelope_is_not_re_derived():
    """A StandardBot already put the full signal in the payload; rebuilding it
    from the compat keys would drop everything the compat keys cannot express."""
    from cyqnt_trd.standard_bot.core import ExitPlan, Provenance, StandardSignal

    original = StandardSignal(
        bot_id="v2bot", decision_time=DT, provenance=Provenance(strategy_id="v2bot"),
        symbol="BTCUSDT", intent=PositionIntent.OPEN_LONG,
        exit_plan=ExitPlan(exit_on_opposite_signal=True),
        reason_codes=("a", "b"), summary="detail the envelope cannot hold",
        topic="mtf_trend",
    )
    envelope = _envelope(payload=original.to_dict())
    restored = envelope_to_signal(envelope)
    assert restored.reason_codes == ("a", "b")
    assert restored.summary == "detail the envelope cannot hold"
    assert restored.topic == "mtf_trend"


def test_batch_conversion():
    assert len(batch_to_signals([_envelope(), _envelope()])) == 2


# --------------------------------------------------------------------------- #
# v2 bot -> the registry every entrypoint reads                                #
# --------------------------------------------------------------------------- #


def test_a_standard_bot_answers_needed_timeframes():
    """The one protocol method a bot lacked, and the reason no CLI could
    schedule one: the registry asks this before assembling a snapshot."""
    from strategies.standard.multi_source_bot import MultiSourceBot

    assert MultiSourceBot().needed_timeframes() == ["1h"]


def test_needed_timeframes_is_read_off_the_declared_inputs():
    """Not configured separately — two places to state one timeframe is one
    place to get it wrong."""
    from strategies.standard.blocks_reference_bots import BlocksEmaCrossBot

    bot = BlocksEmaCrossBot()
    declared = {(r.params or {}).get("interval") for r in bot.required_data()}
    assert set(bot.needed_timeframes()) == {t for t in declared if t}


def test_registering_a_bot_makes_the_cli_able_to_name_it():
    from cyqnt_trd.blocks.strategy import is_known_block_strategy
    from strategies.standard.multi_source_bot import MultiSourceBot

    bot = MultiSourceBot()
    plugin_id = register_standard_bot(bot)
    assert is_known_block_strategy(plugin_id)


def test_the_registered_config_factory_hands_back_a_mapping():
    """Every shipped bot's ``normalize_config`` does ``merged.update(config)``.

    The block path's factory returns a ``SimpleNamespace``, which raises
    ``'X' object is not iterable`` on the first bar — that is why the
    pre-existing ``bind_to_signal_registry`` could not run three of the shipped
    bots. The bot factory must hand back something a dict can absorb.
    """
    from cyqnt_trd.blocks import strategy as blocks_strategy
    from strategies.standard.multi_source_bot import MultiSourceBot

    register_standard_bot(MultiSourceBot(), strategy_id="cfg_probe")
    factory = next(f for plugin, f in blocks_strategy._PENDING_REGISTRATIONS
                   if getattr(plugin, "plugin_id", None) == "multi_source_reference")
    config = factory({"instrument_id": "btcusdt", "timeframe": "1h"})
    assert isinstance(config, dict)
    assert config["instrument_id"] == "BTCUSDT"
    assert dict(MultiSourceBot().normalize_config(config))["timeframe"] == "1h"


# --------------------------------------------------------------------------- #
# the point of the whole thing                                                 #
# --------------------------------------------------------------------------- #


def test_yaml_trade_and_yaml_selection_publish_the_same_schema():
    """Both YAML shapes, through the adapter, produce one contract."""
    trade = envelope_to_signal(_envelope()).to_dict()
    selection = envelope_to_signal(_envelope(
        kind=SignalKind.SELECTION, instrument_id=None,
        payload={"bar_timestamp": DT, "candidates": [
            {"symbol": "BTCUSDT", "rank": 1, "score": 1.0, "side": "long"}]},
    )).to_dict()

    assert set(trade) == set(selection)
    assert trade["schema"] == selection["schema"] == "cyqnt.signal/v2"
    assert (trade["kind"], selection["kind"]) == ("trade", "selection")
