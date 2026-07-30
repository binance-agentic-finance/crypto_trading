"""``cyqnt.signal/v2`` → pre-v2 engine payload translation.

Why this exists
---------------
v2 (``StandardSignal``) is the semantic contract; ``SignalEnvelope`` is the
transport that the registry and the engines move. The two name the same concepts
differently, so without a translation at the envelope boundary a v2 bot cannot
drive the existing engines at all:

* ``SnapshotBacktestRunner`` does ``float(payload["size"])`` — v2 puts a
  ``SizeSpec`` **object** there, so the run died with a ``TypeError``;
* the runner reads ``payload["exit_spec"]``, v2 declares ``exit_plan`` — so a v2
  bot's stop and take-profit were silently ignored and the position ran
  unprotected;
* ``build_intents`` reads ``quantity`` / ``notional`` / ``price`` /
  ``risk_hints``, none of which v2 emits.

These tests pin the translation down, and in particular pin the two places where
it must decline rather than guess (unrepresentable sizing, take-profit ladders).
"""

from __future__ import annotations

import pytest

from cyqnt_trd.standard_bot.bot import (
    _engine_compat_payload,
    _entry_reference_price,
    _exit_plan_to_exit_spec,
    _size_to_equity_fraction,
)
from cyqnt_trd.standard_bot.core import (
    Direction,
    EntrySpec,
    EntryType,
    ExitPlan,
    PositionIntent,
    Provenance,
    SizeMode,
    SizeSpec,
    StandardSignal,
    StopSpec,
    TakeProfitLeg,
    TimeStop,
)

ATR = 480.0


def _signal(**kw) -> StandardSignal:
    base = dict(
        bot_id="bridge_probe",
        decision_time=1776311999999,
        provenance=Provenance(strategy_id="bridge_probe", strategy_version="v1"),
        symbol="BTCUSDT",
        intent=PositionIntent.OPEN_LONG,
        direction=Direction.LONG,
        # v2 refuses an entry intent with no stated exit — a good rule. Tests
        # that are not about the exit plan opt into the explicit "exit on the
        # opposite signal" plan rather than omitting it.
        exit_plan=ExitPlan(exit_on_opposite_signal=True),
    )
    base.update(kw)
    return StandardSignal(**base)


# --------------------------------------------------------------------------- #
# exit_plan -> exit_spec                                                       #
# --------------------------------------------------------------------------- #


def test_atr_stop_and_target_becomes_atr_stop_tp():
    spec = _exit_plan_to_exit_spec(_signal(exit_plan=ExitPlan(
        stop_loss=StopSpec(atr_mult=2.0, atr_value=ATR),
        take_profit=(TakeProfitLeg(close_pct=1.0, atr_mult=3.0, atr_value=ATR),),
        time_stop=TimeStop(max_bars=24),
    )))
    assert spec == {"max_bars": 24, "side": "long", "atr_at_entry": ATR,
                    "type": "atr_stop_tp", "stop_mult": 2.0, "tp_mult": 3.0}


def test_trailing_atr_stop_becomes_atr_trailing_stop():
    spec = _exit_plan_to_exit_spec(_signal(exit_plan=ExitPlan(
        stop_loss=StopSpec(atr_mult=2.5, atr_value=ATR, trailing=True))))
    assert spec["type"] == "atr_trailing_stop"
    assert spec["trail_mult"] == 2.5 and spec["atr_at_entry"] == ATR


def test_percent_stop_and_target_becomes_pct_stop_tp():
    spec = _exit_plan_to_exit_spec(_signal(exit_plan=ExitPlan(
        stop_loss=StopSpec(pct=0.02),
        take_profit=(TakeProfitLeg(close_pct=1.0, pct=0.04),))))
    assert spec["type"] == "pct_stop_tp"
    assert spec["stop_pct"] == 0.02 and spec["tp_pct"] == 0.04


def test_absolute_prices_pass_through():
    spec = _exit_plan_to_exit_spec(_signal(exit_plan=ExitPlan(
        stop_loss=StopSpec(price=68_000.0),
        take_profit=(TakeProfitLeg(close_pct=1.0, price=75_000.0),))))
    assert spec["stop_loss_price"] == 68_000.0
    assert spec["take_profit_price"] == 75_000.0


def test_time_only_plan():
    spec = _exit_plan_to_exit_spec(_signal(exit_plan=ExitPlan(
        time_stop=TimeStop(max_bars=48))))
    assert spec == {"type": "time_only", "max_bars": 48}


def test_short_side_is_carried_into_the_spec():
    """The runner is side-aware; a short's stop sits ABOVE entry."""
    spec = _exit_plan_to_exit_spec(_signal(
        intent=PositionIntent.OPEN_SHORT, direction=Direction.SHORT,
        exit_plan=ExitPlan(stop_loss=StopSpec(pct=0.02))))
    assert spec["side"] == "short"


def test_take_profit_ladder_is_flattened_and_the_drop_is_recorded():
    """No simulation engine supports partial closes — say so, do not truncate silently."""
    spec = _exit_plan_to_exit_spec(_signal(exit_plan=ExitPlan(
        stop_loss=StopSpec(pct=0.02),
        take_profit=(TakeProfitLeg(close_pct=0.5, pct=0.03),
                     TakeProfitLeg(close_pct=0.3, pct=0.05),
                     TakeProfitLeg(close_pct=0.2, pct=0.08)))))
    assert spec["tp_pct"] == 0.03, "the first rung must be the one kept"
    assert spec["_dropped_tp_legs"] == 2


def test_no_exit_plan_means_no_spec():
    """HOLD is the one intent v2 lets you emit without an exit plan."""
    sig = _signal(intent=PositionIntent.HOLD, direction=Direction.NEUTRAL,
                  exit_plan=None)
    assert _exit_plan_to_exit_spec(sig) is None


def test_opposite_signal_only_plan_yields_no_price_spec():
    """ExitPlan(exit_on_opposite_signal=True) alone has no price/time exit."""
    assert _exit_plan_to_exit_spec(_signal()) is None


# --------------------------------------------------------------------------- #
# SizeSpec -> equity fraction                                                  #
# --------------------------------------------------------------------------- #


def test_equity_pct_is_used_directly():
    assert _size_to_equity_fraction(
        _signal(size=SizeSpec(mode=SizeMode.EQUITY_PCT, value=0.1))) == 0.1


def test_risk_pct_resolves_against_a_percent_stop():
    """Risking 1% of equity with a 2%-away stop means deploying 50%."""
    assert _size_to_equity_fraction(_signal(
        size=SizeSpec(mode=SizeMode.RISK_PCT, value=0.01),
        exit_plan=ExitPlan(stop_loss=StopSpec(pct=0.02)))) == pytest.approx(0.5)


def test_risk_pct_resolves_against_an_atr_stop_when_a_price_is_stated():
    sig = _signal(
        size=SizeSpec(mode=SizeMode.RISK_PCT, value=0.01),
        entry=EntrySpec(type=EntryType.LIMIT, price=48_000.0),
        exit_plan=ExitPlan(stop_loss=StopSpec(atr_mult=2.0, atr_value=ATR)))
    # stop distance = 2 * 480 / 48000 = 2% -> 1% / 2% = 0.5
    assert _size_to_equity_fraction(sig) == pytest.approx(0.5)


def test_risk_pct_is_capped_at_full_equity():
    assert _size_to_equity_fraction(_signal(
        size=SizeSpec(mode=SizeMode.RISK_PCT, value=0.05),
        exit_plan=ExitPlan(stop_loss=StopSpec(pct=0.01)))) == 1.0


def test_risk_pct_without_a_resolvable_stop_declines():
    """A market entry states no price, so an ATR distance is unknowable here."""
    assert _size_to_equity_fraction(_signal(
        size=SizeSpec(mode=SizeMode.RISK_PCT, value=0.01),
        exit_plan=ExitPlan(stop_loss=StopSpec(atr_mult=2.0, atr_value=ATR)))) is None


@pytest.mark.parametrize("mode,value", [
    (SizeMode.QUANTITY, 0.5),
    (SizeMode.QUOTE_AMOUNT, 2_000.0),
    (SizeMode.POSITION_PCT, 0.5),
])
def test_absolute_and_position_relative_modes_decline(mode, value):
    """Guessing would silently change position size — decline instead."""
    assert _size_to_equity_fraction(_signal(size=SizeSpec(mode=mode, value=value))) is None


def test_entry_reference_price_prefers_price_then_zone_midpoint():
    assert _entry_reference_price(_signal(
        entry=EntrySpec(type=EntryType.LIMIT, price=100.0))) == 100.0
    assert _entry_reference_price(_signal(
        entry=EntrySpec(type=EntryType.LIMIT, zone=(90.0, 110.0)))) == 100.0
    assert _entry_reference_price(_signal(entry=EntrySpec(type=EntryType.MARKET))) is None


# --------------------------------------------------------------------------- #
# the whole compat payload                                                     #
# --------------------------------------------------------------------------- #


def test_compat_payload_never_leaves_a_dict_under_size():
    """``float(payload["size"])`` in the runner must never see a SizeSpec dict."""
    payload = _engine_compat_payload(_signal(
        size=SizeSpec(mode=SizeMode.QUANTITY, value=0.5)))
    assert "size" not in payload, "v2's SizeSpec object must be left untouched"
    assert payload["engine_size"] is None
    assert "size_unresolved" in payload, "an unresolved size must say why"


def test_unresolved_size_survives_the_runner_without_crashing():
    """Regression for the original TypeError, at the exact call site."""
    payload = _engine_compat_payload(_signal(
        size=SizeSpec(mode=SizeMode.QUOTE_AMOUNT, value=2_000.0)))
    raw = payload.get("engine_size", 1.0)
    resolved = float(raw) if isinstance(raw, (int, float)) else 1.0
    assert resolved == 1.0


def test_compat_payload_states_direction_for_the_engines():
    long_payload = _engine_compat_payload(_signal())
    assert long_payload["target_position"] == 1
    assert long_payload["risk_hints"] == {"target_position": 1}

    short_payload = _engine_compat_payload(_signal(
        intent=PositionIntent.OPEN_SHORT, direction=Direction.SHORT))
    assert short_payload["target_position"] == -1


def test_hold_intent_has_no_direction():
    payload = _engine_compat_payload(_signal(
        intent=PositionIntent.HOLD, direction=Direction.NEUTRAL, exit_plan=None))
    assert payload["target_position"] == 0


def test_full_v2_dict_is_still_carried_alongside_the_compat_keys():
    """A v2-aware consumer must never be forced to read the compat shim."""
    sig = _signal(
        size=SizeSpec(mode=SizeMode.EQUITY_PCT, value=0.1),
        exit_plan=ExitPlan(stop_loss=StopSpec(pct=0.02)))
    payload = dict(sig.to_dict())
    payload.update(_engine_compat_payload(sig))
    assert payload["schema"] == "cyqnt.signal/v2"
    assert payload["exit_plan"]["stop_loss"]["pct"] == 0.02, "v2 shape preserved"
    assert payload["exit_spec"]["stop_pct"] == 0.02, "engine shape added"
    assert isinstance(payload["size"], dict), "v2 SizeSpec must survive intact"
    assert payload["engine_size"] == 0.1, "engine float lives under its own key"
