"""Sample bots ported from the local funding books, plus custom data sources."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.standard_bot.bot import BotContext, CapabilityError
from cyqnt_trd.standard_bot.core import (
    AdvisoryAction,
    DataQuality,
    Direction,
    MarketScope,
    PositionIntent,
)
from cyqnt_trd.standard_bot.data import custom_sources as cs
from cyqnt_trd.standard_bot.data.catalog import get_node, is_custom_node
from cyqnt_trd.standard_bot.runtime import data as data_runtime
from strategies.standard.funding_carry_gated import FundingCarryGated
from strategies.standard.funding_crowding_neutral import FundingCrowdingNeutral
from strategies.standard.funding_oi_crowding_monitor import FundingOiCrowdingMonitor

NOW = 1_786_000_000_000
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]


def _funding(rate: float, n: int = 200, jitter: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    stamps = NOW - np.arange(n)[::-1] * 28_800_000
    rates = rate + (rng.normal(0.0, jitter, n) if jitter else np.zeros(n))
    return pd.DataFrame(
        {"symbol": "X", "rate": rates, "timestamp": stamps, "mark_price": 100.0}
    )


def _by_symbol(signals):
    return {s.symbol: s for s in signals if s.symbol}


# --------------------------------------------------------------------------
# N003 — funding crowding, dollar neutral
# --------------------------------------------------------------------------

_RATES = [0.0009, 0.0007, 0.0004, 0.0001, 0.0,
          -0.0001, -0.0003, -0.0005, -0.0007, -0.0011]


def _crowding_ctx(**overrides):
    frames = {
        "funding_%s" % symbol: _funding(rate, seed=index)
        for index, (symbol, rate) in enumerate(zip(UNIVERSE, _RATES))
    }
    base = dict(
        decision_time=NOW, frames=frames,
        source_status={key: "ok" for key in frames},
    )
    base.update(overrides)
    return BotContext(**base)


def test_crowding_shorts_the_payers_and_longs_the_paid():
    signals = _by_symbol(FundingCrowdingNeutral().decide_checked(_crowding_ctx()))
    # highest positive funding -> crowded long -> short it
    assert signals["BTCUSDT"].intent is PositionIntent.OPEN_SHORT
    assert signals["ETHUSDT"].intent is PositionIntent.OPEN_SHORT
    # most negative funding -> crowded short -> long it
    assert signals["DOTUSDT"].intent is PositionIntent.OPEN_LONG
    assert signals["LINKUSDT"].intent is PositionIntent.OPEN_LONG
    # the middle of the cross-section is left alone
    assert "XRPUSDT" not in signals


def test_legs_are_balanced_and_gross_is_respected():
    signals = [s for s in FundingCrowdingNeutral().decide_checked(_crowding_ctx())
               if s.intent.is_entry]
    longs = [s for s in signals if s.intent.target_side == "long"]
    shorts = [s for s in signals if s.intent.target_side == "short"]
    assert len(longs) == len(shorts)
    total = sum(s.size.value for s in signals)
    assert total == pytest.approx(FundingCrowdingNeutral().config["gross"], abs=1e-6)


def test_a_name_leaving_the_basket_is_closed_on_the_side_it_was_held():
    ctx = _crowding_ctx(positions={"AVAXUSDT": 3.0, "SOLUSDT": -2.0})
    signals = _by_symbol(FundingCrowdingNeutral().decide_checked(ctx))
    # held long, no longer wanted -> close the LONG (sell, reduce-only)
    assert signals["AVAXUSDT"].intent is PositionIntent.CLOSE_LONG
    assert signals["AVAXUSDT"].closes_side == "long"
    assert signals["AVAXUSDT"].order_side == "sell"
    assert signals["AVAXUSDT"].intent.reduce_only
    # held short, no longer wanted -> close the SHORT (buy back)
    assert signals["SOLUSDT"].intent is PositionIntent.CLOSE_SHORT
    assert signals["SOLUSDT"].closes_side == "short"
    assert signals["SOLUSDT"].order_side == "buy"


def test_a_side_flip_is_one_instruction_not_two():
    # BTC has the most positive funding -> the basket wants it SHORT.
    ctx = _crowding_ctx(positions={"BTCUSDT": 4.0})   # currently held LONG
    signals = _by_symbol(FundingCrowdingNeutral().decide_checked(ctx))
    assert signals["BTCUSDT"].intent is PositionIntent.FLIP_TO_SHORT
    assert signals["BTCUSDT"].closes_side == "long"
    assert signals["BTCUSDT"].intent.target_side == "short"


def test_a_thin_cross_section_flattens_instead_of_betting_directionally():
    frames = {"funding_BTCUSDT": _funding(0.0009), "funding_ETHUSDT": _funding(-0.0009)}
    ctx = BotContext(
        decision_time=NOW, frames=frames,
        source_status={k: "ok" for k in frames},
        positions={"SOLUSDT": 2.0},
    )
    signals = FundingCrowdingNeutral().decide_checked(ctx)
    assert [s.intent for s in signals] == [PositionIntent.CLOSE_LONG]
    assert "INSUFFICIENT_CROSS_SECTION" in signals[0].reason_codes


def test_basket_snapshot_ranks_the_whole_universe():
    signals = FundingCrowdingNeutral().decide_checked(_crowding_ctx())
    basket = [s for s in signals if s.market_scope is MarketScope.CROSS_SECTION]
    assert len(basket) == 1
    candidates = basket[0].candidates
    assert len(candidates) == len(UNIVERSE)
    assert [c.rank for c in candidates] == list(range(1, len(UNIVERSE) + 1))
    # rank 1 is the most negative funding -> the long leg
    assert candidates[0].direction is Direction.LONG
    assert candidates[-1].direction is Direction.SHORT


def test_entries_declare_how_the_position_ends():
    entries = [s for s in FundingCrowdingNeutral().decide_checked(_crowding_ctx())
               if s.intent.is_entry]
    assert entries
    for signal in entries:
        assert signal.exit_plan is not None
        assert signal.exit_plan.exit_on_opposite_signal is True


def test_crowding_declares_only_backtestable_inputs():
    nodes = [request.node for request in FundingCrowdingNeutral().required_data()]
    assert set(nodes) == {"funding"}
    assert data_runtime.validate_nodes(set(nodes), for_backtest=True) == []


# --------------------------------------------------------------------------
# A004 — gated funding carry
# --------------------------------------------------------------------------


def _carry_ctx(**overrides):
    frames = {
        "funding_BTCUSDT": _funding(0.00012, jitter=0.00002, seed=1),  # stable +
        "funding_ETHUSDT": _funding(0.00010, jitter=0.00040, seed=2),  # sign flips
        "funding_SOLUSDT": _funding(0.00001, jitter=0.00001, seed=3),  # below floor
        "funding_BNBUSDT": _funding(0.00009, jitter=0.00002, seed=4),  # stable +
    }
    base = dict(decision_time=NOW, frames=frames,
                source_status={key: "ok" for key in frames})
    base.update(overrides)
    return BotContext(**base)


def test_carry_shorts_the_perp_to_receive_funding():
    signals = _by_symbol(FundingCarryGated().decide_checked(_carry_ctx()))
    assert signals["BTCUSDT"].intent is PositionIntent.OPEN_SHORT
    assert "FUNDING_REGIME_STABLE_POSITIVE" in signals["BTCUSDT"].reason_codes


def test_unstable_funding_regime_is_not_entered():
    signals = _by_symbol(FundingCarryGated().decide_checked(_carry_ctx()))
    assert "ETHUSDT" not in signals   # flips too often, never opened


def test_carry_below_the_floor_is_not_entered():
    signals = _by_symbol(FundingCarryGated().decide_checked(_carry_ctx()))
    assert "SOLUSDT" not in signals


def test_a_broken_regime_closes_the_existing_carry_leg():
    ctx = _carry_ctx(positions={"ETHUSDT": -5.0})   # short carry leg already on
    signals = _by_symbol(FundingCarryGated().decide_checked(ctx))
    eth = signals["ETHUSDT"]
    assert eth.intent is PositionIntent.CLOSE_SHORT
    assert eth.closes_side == "short"
    assert eth.order_side == "buy"
    assert eth.intent.reduce_only is True
    assert eth.intent.requires_short_capability is False
    assert "FUNDING_REGIME_UNSTABLE" in eth.reason_codes


def test_a_still_valid_leg_is_left_alone():
    ctx = _carry_ctx(positions={"BTCUSDT": -5.0})
    signals = _by_symbol(FundingCarryGated().decide_checked(ctx))
    assert "BTCUSDT" not in signals   # no churn, no redundant HOLD


def test_missing_funding_never_reads_as_zero_carry():
    ctx = _carry_ctx(
        frames={"funding_BTCUSDT": pd.DataFrame()},
        source_status={"funding_BTCUSDT": "degraded"},
    )
    signals = FundingCarryGated().decide_checked(ctx)
    assert all(s.intent is not PositionIntent.OPEN_SHORT for s in signals)


def test_carry_leg_states_its_unhedged_risk():
    signal = _by_symbol(FundingCarryGated().decide_checked(_carry_ctx()))["BTCUSDT"]
    assert signal.exit_plan.stop_loss is not None
    assert "delta-hedged" in signal.exit_plan.note
    assert "对冲" in signal.recommended_behavior


# --------------------------------------------------------------------------
# D001 — advisory monitor
# --------------------------------------------------------------------------


def _oi(growth: float):
    return pd.DataFrame(
        {"timestamp": [NOW - 86_400_000, NOW], "oi_base": [1.0, 1.0],
         "oi_value": [100.0, 100.0 * (1.0 + growth)], "oi_change_bps": [0.0, 0.0]}
    )


def _monitor_ctx(**overrides):
    frames = {
        "funding_BTCUSDT": _funding(0.0020), "oi_BTCUSDT": _oi(0.35),
        "funding_ETHUSDT": _funding(0.0008), "oi_ETHUSDT": _oi(0.02),
        "funding_SOLUSDT": _funding(0.0010),          # no OI at all
    }
    base = dict(
        decision_time=NOW, frames=frames,
        source_status={**{k: "ok" for k in frames}, "oi_SOLUSDT": "error"},
    )
    base.update(overrides)
    return BotContext(**base)


def test_monitor_escalates_to_avoid_when_funding_and_oi_agree():
    signals = _by_symbol(FundingOiCrowdingMonitor().decide_checked(_monitor_ctx()))
    btc = signals["BTCUSDT"]
    assert btc.advisory_action is AdvisoryAction.AVOID
    assert "SQUEEZE_RISK" in btc.reason_codes
    assert btc.direction is Direction.SHORT       # risk is on the crowded long side


def test_monitor_never_emits_an_instruction():
    for signal in FundingOiCrowdingMonitor().decide_checked(_monitor_ctx()):
        assert signal.intent is PositionIntent.HOLD
        assert signal.auto_trade_eligible is False
        with pytest.raises(ValueError):
            signal.to_execution_request()


def test_missing_oi_is_reported_not_treated_as_zero():
    signals = _by_symbol(FundingOiCrowdingMonitor().decide_checked(_monitor_ctx()))
    sol = signals["SOLUSDT"]
    assert "OI_UNAVAILABLE" in sol.reason_codes
    assert sol.advisory_action is AdvisoryAction.WATCH   # cannot reach avoid unconfirmed
    assert sol.data_quality is DataQuality.DEGRADED


def test_quiet_funding_produces_no_alert():
    ctx = _monitor_ctx(
        frames={"funding_BTCUSDT": _funding(0.00001), "oi_BTCUSDT": _oi(0.0)},
        source_status={"funding_BTCUSDT": "ok", "oi_BTCUSDT": "ok"},
    )
    assert FundingOiCrowdingMonitor().decide_checked(ctx) == []


def test_monitor_carries_evidence_for_both_inputs():
    btc = _by_symbol(FundingOiCrowdingMonitor().decide_checked(_monitor_ctx()))["BTCUSDT"]
    sources = {item.source for item in btc.evidence}
    assert sources == {"data.funding", "data.open_interest"}


def test_advisory_capability_blocks_an_instruction_written_by_mistake():
    class _Rogue(FundingOiCrowdingMonitor):
        def decide(self, ctx):
            from cyqnt_trd.standard_bot.core import (
                ExitPlan, Provenance, StandardSignal, StopSpec,
            )

            yield StandardSignal(
                bot_id=self.spec.bot_id, decision_time=ctx.decision_time,
                provenance=Provenance(strategy_id=self.spec.bot_id),
                symbol="BTCUSDT", intent=PositionIntent.OPEN_SHORT,
                exit_plan=ExitPlan(stop_loss=StopSpec(pct=0.02)),
            )

    with pytest.raises(CapabilityError):
        _Rogue().decide_checked(_monitor_ctx())


# --------------------------------------------------------------------------
# user-registered data sources
# --------------------------------------------------------------------------


@pytest.fixture
def _cleanup_custom():
    created = []
    yield created
    for name in created:
        cs.unregister_node(name)


def test_a_callable_becomes_a_first_class_node(_cleanup_custom):
    def team_feed(*, symbol="BTCUSDT"):
        return pd.DataFrame([{"symbol": symbol, "score": 0.8}])

    spec = cs.register_callable_node(
        name="team_score", fn=team_feed, availability="FORWARD_ONLY",
        pit_hazard="live poll, no history endpoint",
        param_schema=[{"key": "symbol", "type": "str", "required": True}],
        returns_columns=("symbol", "score"),
    )
    _cleanup_custom.append(spec.name)

    assert spec.name == "custom_team_score"
    assert is_custom_node(spec.name)
    assert get_node(spec.name).availability.value == "FORWARD_ONLY"
    # callable through the same surface as a built-in
    frame = data_runtime.custom_team_score(symbol="ETHUSDT")
    assert frame.loc[0, "symbol"] == "ETHUSDT"
    # and subject to the same replay discipline
    assert data_runtime.validate_nodes([spec.name]) == []
    problems = data_runtime.validate_nodes([spec.name], for_backtest=True)
    assert problems and "no history endpoint" in problems[0]


def test_a_custom_node_cannot_shadow_a_builtin():
    with pytest.raises(ValueError) as excinfo:
        cs.register_callable_node(
            name="klines", fn=lambda **kw: pd.DataFrame(), availability="BACKTESTABLE"
        )
    assert "already a catalog node" in str(excinfo.value)
    assert "NOT override" in str(excinfo.value)


def test_availability_must_be_stated():
    with pytest.raises(ValueError) as excinfo:
        cs.register_callable_node(
            name="no_tier", fn=lambda **kw: pd.DataFrame(), availability="whenever"
        )
    assert "availability must be one of" in str(excinfo.value)


def test_non_backtestable_registration_must_declare_its_hazard():
    with pytest.raises(ValueError) as excinfo:
        cs.register_callable_node(
            name="no_hazard", fn=lambda **kw: pd.DataFrame(), availability="SEMI"
        )
    assert "requires pit_hazard" in str(excinfo.value)


def test_a_file_node_is_gated_to_the_decision_time(tmp_path, _cleanup_custom):
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        {"ts": [1_700_000_000_000, 1_800_000_000_000], "value": [1.0, 2.0]}
    ).to_csv(path, index=False)

    spec = cs.register_file_node(
        name="oi_panel", path=str(path), availability="SEMI",
        time_column="ts", pit_hazard="research capture; starts at the first row",
    )
    _cleanup_custom.append(spec.name)

    assert len(data_runtime.custom_oi_panel()) == 2
    gated = data_runtime.custom_oi_panel(as_of_ms=1_750_000_000_000)
    assert len(gated) == 1   # epoch-ms parsed as ms, not nanoseconds
    assert gated.loc[0, "value"] == 1.0


def test_a_file_node_without_a_time_column_says_it_cannot_be_gated(tmp_path, _cleanup_custom):
    path = tmp_path / "flat.csv"
    pd.DataFrame({"value": [1.0]}).to_csv(path, index=False)
    spec = cs.register_file_node(
        name="flat_panel", path=str(path), availability="SEMI"
    )
    _cleanup_custom.append(spec.name)
    assert "cannot be gated" in get_node(spec.name).pit_hazard


def test_a_custom_node_plugs_into_a_bot(_cleanup_custom):
    """The monitor accepts a user OI panel without editing the bot."""
    panel = pd.DataFrame({"BTCUSDT": [100.0, 150.0]})

    spec = cs.register_callable_node(
        name="deep_oi", fn=lambda **kw: panel, availability="SEMI",
        pit_hazard="research capture, starts 2024-01",
    )
    _cleanup_custom.append(spec.name)

    bot = FundingOiCrowdingMonitor(oi_panel_node=spec.name, universe=["BTCUSDT"])
    assert spec.name in [request.node for request in bot.required_data()]

    ctx = BotContext(
        decision_time=NOW,
        frames={"funding_BTCUSDT": _funding(0.0020), "oi_panel": panel},
        source_status={"funding_BTCUSDT": "ok", "oi_panel": "ok"},
    )
    signal = bot.decide_checked(ctx)[0]
    # +50% from the user panel confirms the crowd, without any built-in OI node
    assert "OI_BUILDUP_CONFIRMS" in signal.reason_codes
    assert signal.advisory_action is AdvisoryAction.AVOID
