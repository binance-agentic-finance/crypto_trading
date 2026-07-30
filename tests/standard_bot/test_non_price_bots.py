"""Sample bots whose primary signal is not price."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.standard_bot.bot import BotContext
from cyqnt_trd.standard_bot.core import (
    AdvisoryAction,
    Direction,
    FrameKind,
    PositionIntent,
    TypedFrame,
)
from strategies.standard.news_catalyst_trade import NewsCatalystTrade
from strategies.standard.social_flow_divergence import SocialFlowDivergence

NOW = 1_786_000_000_000
T = pd.Timestamp(NOW, unit="ms", tz="UTC")


def _typed(node, kind, frame):
    return TypedFrame(node=node, kind=kind, frame=frame, as_of=NOW)


def _news_row(event_id, title, ticker, *, minutes_ago=5, reliability=0.85, tendency=1):
    return {
        "event_id": event_id, "title": title, "summary": "", "body": "",
        "event_time": T - pd.Timedelta(minutes=minutes_ago + 1),
        "available_time": T - pd.Timedelta(minutes=minutes_ago),
        "source_id": "square.getNews", "topic": "unclassified",
        "tickers": [ticker], "tendency": tendency,
        "source_reliability": reliability, "url": "https://example.invalid/%s" % event_id,
    }


def _bars(symbol, *, drift=0.0005, n=40):
    close = 100 * (1 + drift) ** np.arange(n)
    stamps = [T - pd.Timedelta(minutes=15 * (n - i)) for i in range(n)]
    return pd.DataFrame({
        "instrument_id": symbol, "timeframe": "15m",
        "open_time": stamps, "close_time": stamps, "available_time": T,
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
        "volume": 100.0, "quote_volume": 100.0 * close, "trades": 10,
    })


def _funding_metric(bps):
    return pd.DataFrame({
        "event_time": [T], "available_time": [T], "instrument_id": ["X"],
        "metric": ["rate"], "value": [bps / 10_000.0],
    })


# --------------------------------------------------------------------------
# news_catalyst_trade — trigger is the event, price only vetoes
# --------------------------------------------------------------------------


def _news_ctx(rows, *, drift=0.0005, funding_bps=2.0, universe=("ABCUSDT",)):
    typed = {"news": _typed("news", FrameKind.EVENT, pd.DataFrame(rows))}
    status = {"news": "ok"}
    for symbol in universe:
        typed["klines_%s" % symbol] = _typed(
            "klines", FrameKind.BAR, _bars(symbol, drift=drift))
        typed["funding_%s" % symbol] = _typed(
            "funding", FrameKind.METRIC, _funding_metric(funding_bps))
        status["klines_%s" % symbol] = "ok"
        status["funding_%s" % symbol] = "ok"
    return BotContext(decision_time=NOW, typed=typed, source_status=status)


def _run_news(rows, **kwargs):
    universe = kwargs.pop("universe", ("ABCUSDT",))
    bot = NewsCatalystTrade(universe=list(universe))
    return bot.decide_checked(_news_ctx(rows, universe=universe, **kwargs))


def test_a_verified_listing_opens_a_position():
    signals = _run_news([_news_row("n1", "Binance Will List ABC (ABC)", "ABC")])
    assert len(signals) == 1
    signal = signals[0]
    assert signal.symbol == "ABCUSDT"
    assert signal.intent is PositionIntent.OPEN_LONG
    assert "LISTING" in signal.reason_codes


def test_holding_period_comes_from_the_event_type():
    """A listing pulse and a mainnet upgrade must not share a horizon."""
    listing = _run_news([_news_row("n1", "Binance Will List ABC (ABC)", "ABC")])[0]
    upgrade = _run_news([_news_row("n2", "ABC mainnet upgrade goes live", "ABC")])[0]
    assert listing.exit_plan.time_stop.max_seconds < upgrade.exit_plan.time_stop.max_seconds
    assert listing.horizon_seconds == listing.exit_plan.time_stop.max_seconds


def test_a_catalyst_position_does_not_wait_for_a_reverse_signal():
    signal = _run_news([_news_row("n1", "Binance Will List ABC (ABC)", "ABC")])[0]
    assert signal.exit_plan.exit_on_opposite_signal is False
    assert signal.exit_plan.time_stop is not None


def test_an_unverified_source_does_not_trade():
    assert _run_news(
        [_news_row("n1", "Binance Will List ABC (ABC)", "ABC", reliability=0.3)]
    ) == []


def test_stale_news_is_not_a_catalyst():
    """Old enough and it is a news recap, not a trigger."""
    assert _run_news(
        [_news_row("n1", "Binance Will List ABC (ABC)", "ABC", minutes_ago=60)]
    ) == []


def test_age_and_repost_lateness_are_different_kinds_of_late():
    """A fresh repost of an old story, and an original we polled late, are
    both stale — but only one of them shows up in lead_time."""
    original = _news_row("n1", "Binance Will List ABC (ABC)", "ABC", minutes_ago=90)
    repost = _news_row("n2", "Binance Will List ABC (ABC)", "ABC", minutes_ago=1)
    # the repost is seconds old (low age) but 89 minutes down the chain (high lead)
    assert _run_news([original, repost]) == []


def test_a_move_that_already_happened_is_not_chased():
    assert _run_news(
        [_news_row("n1", "Binance Will List ABC (ABC)", "ABC")], drift=0.02
    ) == []


def test_extreme_funding_vetoes_the_catalyst():
    assert _run_news(
        [_news_row("n1", "Binance Will List ABC (ABC)", "ABC")], funding_bps=25.0
    ) == []


def test_a_bearish_event_is_not_traded_long():
    assert _run_news([_news_row("n1", "Binance will delist ABC", "ABC")]) == []


def test_reposts_do_not_fire_twice():
    rows = [
        _news_row("n1", "Binance Will List ABC (ABC)", "ABC", minutes_ago=6),
        _news_row("n2", "Binance Will List ABC (ABC)", "ABC", minutes_ago=4),
    ]
    assert len(_run_news(rows)) == 1


def test_an_untargeted_event_is_skipped():
    row = _news_row("n1", "Fed holds interest rate steady", "ABC")
    row["tickers"] = []
    assert _run_news([row]) == []


def test_the_news_bot_cannot_be_backtested_and_says_so():
    from cyqnt_trd.standard_bot.runtime import data as data_runtime

    nodes = {request.node for request in NewsCatalystTrade().required_data()}
    problems = data_runtime.validate_nodes(nodes, for_backtest=True)
    assert any("news" in problem for problem in problems)


# --------------------------------------------------------------------------
# social_flow_divergence — no price input at all
# --------------------------------------------------------------------------


def _rank_row(token, rank, mentions, bullish=100, bearish=20):
    return {
        "available_time": T, "event_time": T, "instrument_id": token,
        "rank": rank, "mention_count": mentions,
        "bullish_count": bullish, "bearish_count": bearish,
    }


def _flow_metric(deposit, withdraw):
    return pd.DataFrame({
        "event_time": [T, T], "available_time": [T, T],
        "instrument_id": ["X", "X"],
        "metric": ["large_deposit_amt", "large_withdraw_amt"],
        "value": [deposit, withdraw],
    })


def _flow_ctx(ranks, flows, *, funding_bps=3.0):
    typed = {"ticker_rank": _typed("ticker_rank", FrameKind.RANK, pd.DataFrame(ranks))}
    status = {"ticker_rank": "ok"}
    for token, (deposit, withdraw) in flows.items():
        typed["flow_%s" % token] = _typed(
            "large_flow", FrameKind.METRIC, _flow_metric(deposit, withdraw))
        status["flow_%s" % token] = "ok"
        typed["funding_%sUSDT" % token] = _typed(
            "funding", FrameKind.METRIC, _funding_metric(funding_bps))
        status["funding_%sUSDT" % token] = "ok"
    return BotContext(decision_time=NOW, typed=typed, source_status=status)


def test_hot_plus_inflow_reads_as_distribution():
    ctx = _flow_ctx([_rank_row("BTC", 1, 900)], {"BTC": (8e6, 1e6)})
    signal = SocialFlowDivergence().decide_checked(ctx)[0]
    assert signal.direction is Direction.SHORT
    assert signal.advisory_action is AdvisoryAction.ALERT
    assert "DISTRIBUTION_SHAPE" in signal.reason_codes


def test_hot_plus_outflow_reads_as_accumulation():
    ctx = _flow_ctx([_rank_row("ETH", 2, 600)], {"ETH": (0.5e6, 6e6)})
    signal = SocialFlowDivergence().decide_checked(ctx)[0]
    assert signal.direction is Direction.LONG
    assert "ACCUMULATION_SHAPE" in signal.reason_codes


def test_heat_alone_is_not_a_signal():
    """A source that says nothing about what money did contributes nothing."""
    ctx = _flow_ctx([_rank_row("BTC", 1, 900)], {})
    assert SocialFlowDivergence().decide_checked(ctx) == []


def test_flow_alone_is_not_a_signal():
    ctx = _flow_ctx([_rank_row("SOL", 40, 5)], {"SOL": (9e6, 0.1e6)})
    assert SocialFlowDivergence().decide_checked(ctx) == []


def test_flat_flow_is_not_a_divergence():
    ctx = _flow_ctx([_rank_row("BTC", 1, 900)], {"BTC": (1.0e6, 0.9e6)})
    assert SocialFlowDivergence().decide_checked(ctx) == []


def test_already_crowded_funding_downgrades_the_alert():
    ctx = _flow_ctx([_rank_row("BTC", 1, 900)], {"BTC": (8e6, 1e6)}, funding_bps=20.0)
    signal = SocialFlowDivergence().decide_checked(ctx)[0]
    assert signal.advisory_action is AdvisoryAction.WATCH
    assert "FUNDING_ALREADY_CROWDED" in signal.reason_codes


def test_it_never_emits_an_instruction():
    ctx = _flow_ctx([_rank_row("BTC", 1, 900)], {"BTC": (8e6, 1e6)})
    for signal in SocialFlowDivergence().decide_checked(ctx):
        assert signal.intent is PositionIntent.HOLD
        assert signal.auto_trade_eligible is False
        with pytest.raises(ValueError):
            signal.to_execution_request()


def test_it_uses_no_price_input_at_all():
    nodes = {request.node for request in SocialFlowDivergence().required_data()}
    assert "klines" not in nodes
    assert nodes == {"ticker_rank", "large_flow", "funding"}


def test_both_evidence_sources_are_cited():
    ctx = _flow_ctx([_rank_row("BTC", 1, 900)], {"BTC": (8e6, 1e6)})
    signal = SocialFlowDivergence().decide_checked(ctx)[0]
    assert {item.source for item in signal.evidence} == {
        "data.ticker_rank", "data.large_flow"
    }
