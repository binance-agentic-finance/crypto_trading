"""The complete standard-bot contract: position lifecycle output + StandardBot."""

from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.bot import (
    BotContext,
    BotKind,
    BotSpec,
    CapabilityError,
    DataRequest,
    StandardBot,
)
from cyqnt_trd.standard_bot.core import (
    AdvisoryAction,
    DataQuality,
    Direction,
    EntrySpec,
    EntryType,
    ExitPlan,
    MarketScope,
    PositionIntent,
    Provenance,
    SelectionCandidate,
    SignalKind,
    SizeMode,
    SizeSpec,
    StandardSignal,
    StopSpec,
    TakeProfitLeg,
    TimeStop,
)

NOW = 1_786_000_000_000


def _signal(**overrides):
    base = dict(
        bot_id="unit_bot",
        decision_time=NOW,
        provenance=Provenance(strategy_id="unit_bot"),
        symbol="BTCUSDT",
        intent=PositionIntent.OPEN_LONG,
        exit_plan=ExitPlan(stop_loss=StopSpec(pct=0.02)),
    )
    base.update(overrides)
    return StandardSignal(**base)


# --------------------------------------------------------------------------
# position lifecycle — the part that did not exist before
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent,target,closes,order_side,reduce_only,short_cap",
    [
        (PositionIntent.OPEN_LONG, "long", None, "buy", False, False),
        (PositionIntent.OPEN_SHORT, "short", None, "sell", False, True),
        (PositionIntent.ADD_LONG, "long", None, "buy", False, False),
        (PositionIntent.REDUCE_LONG, "long", "long", "sell", True, False),
        (PositionIntent.REDUCE_SHORT, "short", "short", "buy", True, False),
        (PositionIntent.CLOSE_LONG, "flat", "long", "sell", True, False),
        (PositionIntent.CLOSE_SHORT, "flat", "short", "buy", True, False),
        (PositionIntent.FLIP_TO_SHORT, "short", "long", "sell", False, True),
        (PositionIntent.FLIP_TO_LONG, "long", "short", "buy", False, False),
        (PositionIntent.FLAT, "flat", "any", None, True, False),
        (PositionIntent.HOLD, None, None, None, False, False),
    ],
)
def test_intent_answers_every_lifecycle_question(
    intent, target, closes, order_side, reduce_only, short_cap
):
    assert intent.target_side == target
    assert intent.closes_side == closes
    assert intent.order_side == order_side
    assert intent.reduce_only is reduce_only
    assert intent.requires_short_capability is short_cap


def test_closing_a_long_is_distinguishable_from_opening_a_short():
    close = PositionIntent.CLOSE_LONG
    short = PositionIntent.OPEN_SHORT
    # both sell at the exchange...
    assert close.order_side == short.order_side == "sell"
    # ...but only one creates negative exposure, and only one is reduce-only
    assert close.reduce_only and not short.reduce_only
    assert short.requires_short_capability and not close.requires_short_capability
    assert close.closes_side == "long" and short.closes_side is None


def test_signal_exposes_the_close_side_directly():
    signal = _signal(intent=PositionIntent.CLOSE_SHORT, exit_plan=None)
    assert signal.closes_side == "short"
    assert signal.order_side == "buy"


# --------------------------------------------------------------------------
# exit plan
# --------------------------------------------------------------------------


def test_exit_plan_resolves_absolute_prices_for_a_long():
    signal = _signal(
        exit_plan=ExitPlan(
            stop_loss=StopSpec(atr_mult=2.0, atr_value=500.0),
            take_profit=(TakeProfitLeg(close_pct=0.5, pct=0.03),
                         TakeProfitLeg(close_pct=0.5, price=62_000.0)),
            time_stop=TimeStop(max_bars=96),
        )
    )
    resolved = signal.resolved_exit_prices(60_000.0)
    assert resolved["stop_loss_price"] == pytest.approx(59_000.0)
    assert resolved["take_profit"][0]["price"] == pytest.approx(61_800.0)
    assert resolved["take_profit"][1]["price"] == pytest.approx(62_000.0)


def test_exit_plan_mirrors_for_a_short():
    signal = _signal(intent=PositionIntent.OPEN_SHORT,
                     exit_plan=ExitPlan(stop_loss=StopSpec(pct=0.02),
                                        take_profit=(TakeProfitLeg(close_pct=1.0, pct=0.05),)))
    resolved = signal.resolved_exit_prices(100.0)
    assert resolved["stop_loss_price"] == pytest.approx(102.0)   # stop is ABOVE
    assert resolved["take_profit"][0]["price"] == pytest.approx(95.0)


def test_take_profit_ladder_cannot_close_more_than_the_position():
    with pytest.raises(ValueError):
        ExitPlan(take_profit=(TakeProfitLeg(close_pct=0.7, pct=0.02),
                              TakeProfitLeg(close_pct=0.7, pct=0.04)))


def test_stop_spec_rejects_ambiguous_definitions():
    with pytest.raises(ValueError):
        StopSpec(price=100.0, pct=0.02)
    with pytest.raises(ValueError):
        StopSpec(atr_mult=2.0)          # no atr_value to resolve against


def test_entry_without_an_exit_plan_is_refused():
    with pytest.raises(ValueError) as excinfo:
        _signal(exit_plan=None)
    assert "no exit_plan" in str(excinfo.value)


def test_exit_only_signal_needs_no_exit_plan():
    assert _signal(intent=PositionIntent.CLOSE_LONG, exit_plan=None).intent.is_exit


def test_stop_states_whether_it_survives_the_daemon():
    plan = ExitPlan(stop_loss=StopSpec(pct=0.02))
    assert plan.stop_loss.exchange_managed is False
    assert plan.has_protection is True
    assert ExitPlan().has_protection is False


# --------------------------------------------------------------------------
# consistency rules
# --------------------------------------------------------------------------


def test_spot_cannot_open_a_short_but_can_close_a_long():
    with pytest.raises(ValueError) as excinfo:
        _signal(product="spot", intent=PositionIntent.OPEN_SHORT)
    assert "short capability" in str(excinfo.value)
    assert _signal(product="spot", intent=PositionIntent.CLOSE_LONG, exit_plan=None)


def test_reduce_only_follows_the_intent_not_the_author():
    signal = _signal(
        intent=PositionIntent.CLOSE_LONG, exit_plan=None,
        size=SizeSpec(mode=SizeMode.POSITION_PCT, value=1.0, reduce_only=False),
    )
    assert signal.size.reduce_only is True


def test_direction_is_kept_consistent_with_the_instruction():
    assert _signal(intent=PositionIntent.OPEN_SHORT).direction is Direction.SHORT


def test_actionable_signal_needs_an_instrument():
    with pytest.raises(ValueError) as excinfo:
        _signal(symbol=None)
    assert "needs a symbol" in str(excinfo.value)


def test_advisory_signal_cannot_be_auto_trade_eligible():
    with pytest.raises(ValueError):
        _signal(intent=PositionIntent.HOLD, exit_plan=None,
                advisory_action=AdvisoryAction.ALERT, auto_trade_eligible=True)


def test_cross_section_signal_must_carry_candidates():
    with pytest.raises(ValueError):
        _signal(intent=PositionIntent.HOLD, exit_plan=None, symbol=None,
                market_scope=MarketScope.CROSS_SECTION)
    ok = _signal(
        intent=PositionIntent.HOLD, exit_plan=None, symbol=None,
        market_scope=MarketScope.CROSS_SECTION,
        candidates=(SelectionCandidate(symbol="SOLUSDT", rank=1, score=0.9),),
    )
    assert ok.to_dict()["candidates"][0]["symbol"] == "SOLUSDT"


def test_signal_id_and_dedup_key_are_derived_when_absent():
    signal = _signal()
    assert signal.signal_id
    assert signal.dedup_key.startswith("unit_bot:binance:usd_m_perpetual:BTCUSDT:")
    # same inputs -> same id; different intent -> different id
    assert signal.signal_id == _signal().signal_id
    assert signal.signal_id != _signal(intent=PositionIntent.OPEN_SHORT).signal_id


# --------------------------------------------------------------------------
# execution request
# --------------------------------------------------------------------------


def test_execution_request_carries_everything_except_the_idempotency_key():
    signal = _signal(
        entry=EntrySpec(type=EntryType.LIMIT, price=58_200.0, time_in_force="IOC"),
        exit_plan=ExitPlan(stop_loss=StopSpec(pct=0.02),
                           take_profit=(TakeProfitLeg(close_pct=1.0, pct=0.05),)),
        size=SizeSpec(mode=SizeMode.RISK_PCT, value=0.01, leverage=5),
    )
    request = signal.to_execution_request()
    assert request["venue_class"] == "CEX_PERP"
    assert request["intent_type"] == "LIMIT"
    assert request["instrument"] == "BTCUSDT"
    assert request["side"] == "BUY"
    assert request["position_intent"] == "open_long"
    assert request["reduce_only"] is False
    assert request["params"]["time_in_force"] == "IOC"
    assert request["params"]["bracket"]["stop"] == pytest.approx(58_200.0 * 0.98)
    # the executor owns run identity
    assert "idempotency_key" not in request
    assert "strategy_instance_id" not in request


def test_spot_maps_to_the_spot_venue_class():
    signal = _signal(product="spot", intent=PositionIntent.CLOSE_LONG, exit_plan=None,
                     size=SizeSpec(mode=SizeMode.POSITION_PCT, value=1.0))
    request = signal.to_execution_request()
    assert request["venue_class"] == "CEX_SPOT"
    assert request["side"] == "SELL"
    assert request["reduce_only"] is True


def test_advisory_signal_refuses_to_produce_an_order():
    signal = _signal(intent=PositionIntent.HOLD, exit_plan=None,
                     advisory_action=AdvisoryAction.ALERT)
    with pytest.raises(ValueError) as excinfo:
        signal.to_execution_request()
    assert "not executable" in str(excinfo.value)


def test_hold_refuses_to_produce_an_order():
    with pytest.raises(ValueError):
        _signal(intent=PositionIntent.HOLD, exit_plan=None).to_execution_request()


# --------------------------------------------------------------------------
# StandardBot
# --------------------------------------------------------------------------


class _PerpBot(StandardBot):
    spec = BotSpec(bot_id="unit_perp", kind=BotKind.TRADE,
                   products=("usd_m_perpetual",), reads_positions=True)

    def required_data(self):
        return [
            DataRequest("klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 10}),
            DataRequest("funding", {"symbol": "BTCUSDT", "limit": 10}, required=False),
        ]

    def decide(self, ctx):
        held = ctx.side_of("BTCUSDT")
        intent = (
            PositionIntent.CLOSE_LONG if held == "long" else PositionIntent.OPEN_LONG
        )
        yield StandardSignal(
            bot_id=self.spec.bot_id, decision_time=ctx.decision_time,
            provenance=Provenance(strategy_id=self.spec.bot_id),
            symbol="BTCUSDT", intent=intent, score=60.0, confidence=0.6,
            exit_plan=ExitPlan(stop_loss=StopSpec(pct=0.02)) if intent.is_entry else None,
            size=SizeSpec(mode=SizeMode.RISK_PCT, value=0.01),
            data_quality=ctx.data_quality(required=["klines"]),
        )


def _ctx(**overrides):
    base = dict(
        decision_time=NOW,
        frames={"klines": pd.DataFrame({"close": [1.0, 2.0]})},
        source_status={"klines": "ok", "funding": "error"},
    )
    base.update(overrides)
    return BotContext(**base)


def test_bot_reads_current_exposure_and_emits_the_matching_intent():
    bot = _PerpBot()
    flat = bot.decide_checked(_ctx())
    assert flat[0].intent is PositionIntent.OPEN_LONG
    held = bot.decide_checked(_ctx(positions={"BTCUSDT": 1.5}))
    assert held[0].intent is PositionIntent.CLOSE_LONG
    assert held[0].closes_side == "long"


def test_framework_stamps_provenance_and_status_over_the_author():
    signals = _PerpBot().decide_checked(_ctx())
    signal = signals[0]
    assert signal.provenance.inputs == ("klines", "funding")
    assert signal.source_status == {"klines": "ok", "funding": "error"}
    assert signal.data_quality is DataQuality.DEGRADED


def test_missing_required_input_is_insufficient_not_degraded():
    ctx = _ctx(frames={}, source_status={"klines": "error", "funding": "error"})
    assert ctx.data_quality(required=["klines"]) is DataQuality.INSUFFICIENT


def test_context_refuses_an_undeclared_read():
    with pytest.raises(KeyError) as excinfo:
        _ctx().require("orderbook_depth")
    assert "Declare it in required_data" in str(excinfo.value)


def test_required_inputs_lists_the_declared_nodes():
    inputs = _PerpBot().required_inputs()
    assert inputs["klines"] is True
    assert inputs["funding"] is False
    assert inputs["market"] is False


def test_spot_bot_cannot_emit_a_short_intent():
    class _SpotBot(_PerpBot):
        spec = BotSpec(bot_id="unit_spot", kind=BotKind.TRADE,
                       products=("spot",), reads_positions=True)

        def decide(self, ctx):
            yield StandardSignal(
                bot_id="unit_spot", decision_time=ctx.decision_time,
                provenance=Provenance(strategy_id="unit_spot"),
                symbol="BTCUSDT", product="spot",
                intent=PositionIntent.ADD_SHORT,
                exit_plan=ExitPlan(stop_loss=StopSpec(pct=0.02)),
            )

    assert PositionIntent.ADD_SHORT not in _SpotBot.spec.resolved_intents()
    with pytest.raises(ValueError):
        _SpotBot().decide_checked(_ctx())


def test_advisory_bot_cannot_emit_an_actionable_intent():
    class _Advisory(_PerpBot):
        spec = BotSpec(bot_id="unit_advisory", kind=BotKind.ADVISORY)

        def decide(self, ctx):
            yield StandardSignal(
                bot_id="unit_advisory", decision_time=ctx.decision_time,
                provenance=Provenance(strategy_id="unit_advisory"),
                symbol="BTCUSDT", intent=PositionIntent.OPEN_LONG,
                exit_plan=ExitPlan(stop_loss=StopSpec(pct=0.02)),
            )

    with pytest.raises(CapabilityError) as excinfo:
        _Advisory().decide_checked(_ctx())
    assert "may not emit intent=open_long" in str(excinfo.value)


def test_advisory_bot_must_carry_an_advisory_action():
    class _Advisory(_PerpBot):
        spec = BotSpec(bot_id="unit_advisory2", kind=BotKind.ADVISORY)

        def decide(self, ctx):
            yield StandardSignal(
                bot_id="unit_advisory2", decision_time=ctx.decision_time,
                provenance=Provenance(strategy_id="unit_advisory2"),
                symbol="BTCUSDT", intent=PositionIntent.HOLD,
            )

    with pytest.raises(CapabilityError) as excinfo:
        _Advisory().decide_checked(_ctx())
    assert "advisory_action" in str(excinfo.value)


def test_run_returns_a_standard_batch_of_envelopes():
    batch = _PerpBot().run(_ctx())
    assert len(batch.trade_signals()) == 1
    envelope = batch.signals[0]
    assert envelope.kind is SignalKind.TRADE
    assert envelope.instrument_id == "BTCUSDT"
    assert envelope.payload["intent"] == "open_long"
    assert envelope.payload["closes_side"] is None


def test_step_suppresses_a_signal_already_emitted():
    bot = _PerpBot()
    state = bot.initialize_state()
    first = bot.step(_ctx(), state)
    assert len(first.signals) == 1
    second = bot.step(_ctx(), first.state)
    assert second.signals == []


def test_fetch_context_records_status_for_every_declared_node(monkeypatch):
    monkeypatch.setattr(
        "cyqnt_trd.data_cli.kline.fetch_klines",
        lambda **kw: pd.DataFrame({"close": [1.0]}), raising=False,
    )

    def boom(**kw):
        raise ConnectionError("funding down")

    monkeypatch.setattr(
        "cyqnt_trd.data_cli.funding.fetch_funding_history", boom, raising=False
    )
    ctx = _PerpBot().fetch_context(decision_time=NOW)
    assert ctx.decision_time == NOW
    assert set(ctx.frames) == {"klines"}
    assert ctx.source_status == {"klines": "ok", "funding": "error"}
    assert any("funding down" in warning for warning in ctx.warnings)
    assert ctx.data_quality(required=["klines"]) is DataQuality.DEGRADED


def test_bot_spec_serialises_its_capabilities():
    described = _PerpBot.spec.to_dict()
    assert described["kind"] == "trade"
    assert described["reads_positions"] is True
    assert "close_long" in described["allowed_intents"]
    spot = BotSpec(bot_id="s", kind=BotKind.TRADE, products=("spot",),
                   reads_positions=True).to_dict()
    assert "open_short" not in spot["allowed_intents"]
    assert "close_long" in spot["allowed_intents"]


def test_a_bot_that_does_not_read_positions_cannot_close_one():
    """The rule that matters most: we do not read user positions by default.

    An instruction to close asserts a position exists. A bot that never
    observed one is guessing, and guessing wrong sends an order in the wrong
    direction on a book it does not understand.
    """
    blind = BotSpec(bot_id="blind", kind=BotKind.TRADE)
    assert blind.reads_positions is False
    allowed = {intent.value for intent in blind.resolved_intents()}
    assert allowed == {"open_long", "open_short", "hold"}
    for absent in ("close_long", "close_short", "reduce_long", "flip_to_short",
                   "flat", "add_long"):
        assert absent not in allowed


def test_the_refusal_says_how_to_fix_it():
    class _Blind(_PerpBot):
        spec = BotSpec(bot_id="unit_blind", kind=BotKind.TRADE)

        def decide(self, ctx):
            yield StandardSignal(
                bot_id="unit_blind", decision_time=ctx.decision_time,
                provenance=Provenance(strategy_id="unit_blind"),
                symbol="BTCUSDT", intent=PositionIntent.CLOSE_LONG,
            )

    with pytest.raises(CapabilityError) as excinfo:
        _Blind().decide_checked(_ctx())
    message = str(excinfo.value)
    assert "reads_positions=False" in message
    assert "never observed" in message
    assert "contract_positions" in message


def test_a_declared_position_input_becomes_the_position_source(monkeypatch):
    """One answer to 'am I long or short' — the declared input, not a second dict."""
    class _WithPositions(_PerpBot):
        spec = BotSpec(bot_id="unit_positions", kind=BotKind.TRADE,
                       reads_positions=True)

        def required_data(self):
            return [DataRequest("contract_positions", {})]

    monkeypatch.setattr(
        "cyqnt_trd.data_cli.account.fetch_positions",
        lambda **kw: pd.DataFrame(
            [{"symbol": "BTCUSDT", "side": "short", "quantity": 2.0}]
        ),
        raising=False,
    )
    ctx = _WithPositions().fetch_context(decision_time=NOW)
    assert ctx.side_of("BTCUSDT") == "short"
    assert ctx.position("BTCUSDT") == -2.0
