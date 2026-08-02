"""``kind`` — the one field that says which shape of answer you received.

Putting trade output and selection output in one envelope only pays off if a
consumer can tell them apart without guessing. The first cut of ``v2`` left that
to inference — ``market_scope == cross_section`` plus a non-empty ``candidates``
— and the inference was unsound: :class:`BotSpec` defaults ``market_scope`` to
``single``, so a selection bot that never overrode it published a ranked basket
labelled as a single-instrument signal. ``cyqnt.signal/v1`` had an explicit
``kind`` and losing it was a regression.

So ``kind`` is back, it reuses the enum ``SignalEnvelope`` already used rather
than inventing a second word for the same idea, and it is **derived from the
payload** rather than trusted: a producer cannot label a basket as a trade,
because the label is computed from the thing it labels.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.bot import BotContext
from cyqnt_trd.standard_bot.core import (AdvisoryAction, Direction, ExitPlan, MarketScope,
                                         PositionIntent, Provenance, SelectionCandidate,
                                         SignalKind, StandardSignal)

DT = 1_776_322_799_999


def _signal(**kw):
    params = dict(bot_id="t", decision_time=DT, provenance=Provenance(strategy_id="t"))
    params.update(kw)
    return StandardSignal(**params)


def _candidate(symbol="BTCUSDT", rank=1):
    return SelectionCandidate(symbol=symbol, rank=rank, score=1.0,
                              direction=Direction.LONG)


# --------------------------------------------------------------------------- #
# derivation                                                                   #
# --------------------------------------------------------------------------- #


def test_a_basket_is_a_selection_signal():
    signal = _signal(candidates=(_candidate(),), universe_size=10)
    assert signal.kind is SignalKind.SELECTION
    assert signal.to_dict()["kind"] == "selection"


def test_an_instruction_about_one_instrument_is_a_trade_signal():
    signal = _signal(symbol="BTCUSDT", intent=PositionIntent.OPEN_LONG,
                     exit_plan=ExitPlan(exit_on_opposite_signal=True))
    assert signal.kind is SignalKind.TRADE
    assert signal.to_dict()["kind"] == "trade"


def test_an_observation_is_an_alert_signal():
    signal = _signal(symbol="BTCUSDT", advisory_action=AdvisoryAction.AVOID)
    assert signal.kind is SignalKind.ALERT


def test_hold_with_nothing_else_stays_a_trade_signal():
    """``intent=hold`` already says "no change"; ``kind`` must not also say it.

    Two fields that can both mean "do nothing" are two fields that can disagree,
    which is why ``NOOP`` exists in the enum but v2 never emits it.
    """
    assert _signal(symbol="BTCUSDT").kind is SignalKind.TRADE


def test_an_empty_cross_section_is_still_a_selection_signal():
    """No qualified names is a selection result, not a trade instruction."""
    signal = _signal(
        market_scope=MarketScope.CROSS_SECTION,
        universe_size=10,
        reason_codes=("no_candidate_qualified",),
    )
    assert signal.kind is SignalKind.SELECTION
    assert signal.to_dict()["kind"] == "selection"


def test_an_empty_single_instrument_signal_stays_a_trade_signal():
    signal = _signal(
        symbol="BTCUSDT",
        universe_size=10,
        reason_codes=("no_entry",),
    )
    assert signal.kind is SignalKind.TRADE


def test_kind_uses_the_same_enum_as_the_engine_envelope():
    """One vocabulary. ``SignalEnvelope.kind`` and ``StandardSignal.kind`` are
    the same type, so a consumer switching on it does not need two mappings."""
    from cyqnt_trd.standard_bot.core.contracts import SignalKind as EnvelopeKind

    assert SignalKind is EnvelopeKind


# --------------------------------------------------------------------------- #
# it is derived, not trusted                                                   #
# --------------------------------------------------------------------------- #


def test_a_producer_cannot_mislabel_a_basket_as_a_trade():
    with pytest.raises(ValueError, match="contradicts the payload"):
        _signal(kind=SignalKind.TRADE, candidates=(_candidate(),))


def test_a_producer_cannot_mislabel_a_trade_as_a_basket():
    with pytest.raises(ValueError, match="contradicts the payload"):
        _signal(kind=SignalKind.SELECTION, symbol="BTCUSDT",
                intent=PositionIntent.OPEN_LONG,
                exit_plan=ExitPlan(exit_on_opposite_signal=True))


def test_declaring_the_kind_that_matches_is_accepted():
    signal = _signal(kind=SignalKind.SELECTION, candidates=(_candidate(),))
    assert signal.kind is SignalKind.SELECTION


def test_a_producer_cannot_mislabel_an_empty_cross_section_as_a_trade():
    with pytest.raises(ValueError, match="contradicts the payload"):
        _signal(
            kind=SignalKind.TRADE,
            market_scope=MarketScope.CROSS_SECTION,
            universe_size=10,
            reason_codes=("no_candidate_qualified",),
        )


def test_a_producer_cannot_mislabel_an_empty_single_signal_as_a_selection():
    with pytest.raises(ValueError, match="contradicts the payload"):
        _signal(
            kind=SignalKind.SELECTION,
            symbol="BTCUSDT",
            reason_codes=("no_entry",),
        )


# --------------------------------------------------------------------------- #
# scope stays consistent with kind                                             #
# --------------------------------------------------------------------------- #


def test_a_basket_is_promoted_to_cross_section_even_if_the_spec_forgot():
    """The exact defect: a SELECTION bot whose BotSpec kept the SINGLE default."""
    signal = _signal(candidates=(_candidate(),), market_scope=MarketScope.SINGLE)
    assert signal.market_scope is MarketScope.CROSS_SECTION


def test_cross_section_with_no_candidates_is_still_refused():
    with pytest.raises(ValueError, match="no candidates"):
        _signal(market_scope=MarketScope.CROSS_SECTION)


# --------------------------------------------------------------------------- #
# end to end through the reference bots                                        #
# --------------------------------------------------------------------------- #


def _universe():
    return pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                         "quoteVolume": [5e8, 3e8, 2e8],
                         "available_time": [DT] * 3})


def _ticker_rank():
    return pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                         "rank": [1, 2, 3], "mention_count": [120, 80, 45],
                         "bull_ratio": [0.9, 0.25, 0.6], "available_time": [DT] * 3})


def test_the_reference_selection_bot_self_describes():
    from strategies.standard.blocks_reference_bots import BlocksNewsRankBot

    bot = BlocksNewsRankBot()
    assert bot.spec.market_scope is MarketScope.CROSS_SECTION, (
        "a selection bot must declare its scope; relying on the contract to "
        "correct it hides the mistake from anyone reading the spec")
    ctx = BotContext(decision_time=DT,
                     frames={"universe": _universe(), "ticker_rank": _ticker_rank()})
    signal = bot.decide_checked(ctx)[0]
    assert signal.kind is SignalKind.SELECTION
    assert signal.market_scope is MarketScope.CROSS_SECTION
    assert signal.candidates


def test_trade_and_selection_still_share_one_key_set():
    """The premise of a single output format: same keys, different fields filled."""
    trade = _signal(symbol="BTCUSDT", intent=PositionIntent.OPEN_LONG,
                    exit_plan=ExitPlan(exit_on_opposite_signal=True)).to_dict()
    selection = _signal(candidates=(_candidate(),), universe_size=3).to_dict()
    assert set(trade) == set(selection)
    assert "kind" in trade
    assert trade["kind"] != selection["kind"]


# --------------------------------------------------------------------------- #
# the published schema must describe exactly what to_dict() emits              #
# --------------------------------------------------------------------------- #


def test_schema_properties_match_the_emitted_keys_exactly():
    """A published schema that omits an emitted key is worse than no schema.

    The generator walks dataclass fields, so keys ``to_dict()`` derives from the
    intent had to be added by hand — and ``reduce_only`` was missed. Every sample
    then failed the very schema it was meant to conform to, and anyone writing
    the executor against the schema never learnt the flag exists. This test is
    the reason that cannot recur: the two sets must be equal, both directions.
    """
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    schema = json.loads((repo / "strategies/_standard/signal.schema.v2.json")
                        .read_text(encoding="utf-8"))
    emitted = set(_signal(symbol="BTCUSDT", intent=PositionIntent.OPEN_LONG,
                          exit_plan=ExitPlan(exit_on_opposite_signal=True))
                  .to_dict())
    documented = set(schema["properties"])
    assert emitted - documented == set(), (
        "emitted but undocumented: %s — add it in docs/gen_signal_schema_v2.py"
        % sorted(emitted - documented))
    assert documented - emitted == set(), (
        "documented but never emitted: %s" % sorted(documented - emitted))
    assert set(schema["required"]) <= emitted
