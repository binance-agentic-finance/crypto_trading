from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.advisory import (
    BnDataCatalogAdapter,
    FrameValidationError,
    SquarePublicAdapter,
    canvas_definition,
    create_advisory_bot,
    register_advisory_bots,
    signals_to_frame,
)
from cyqnt_trd.standard_bot.core import DataSnapshot, SignalKind, SnapshotMeta, SourceStatus
from cyqnt_trd.standard_bot.signal import SignalPluginRegistry, SignalState


NOW = 1_786_000_000_000
NOW_DT = pd.to_datetime(NOW, unit="ms", utc=True)


def _snapshot(**frames):
    return DataSnapshot(
        version="1.0",
        meta=SnapshotMeta(snapshot_id="snapshot-1", assembled_at=NOW, decision_as_of=NOW),
        frames=frames,
    )


def _market(rows):
    defaults = {
        "event_time": NOW_DT,
        "available_time": NOW_DT,
        "venue": "binance",
        "product": "usd_m_perpetual",
        "symbol": "BTCUSDT",
        "unit": "",
        "window": "24h",
        "source_id": "fixture",
        "quality": "good",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _news(**overrides):
    row = {
        "event_id": "evt-1",
        "published_at": NOW_DT,
        "available_time": NOW_DT,
        "source_id": "official",
        "topic": "exchange_announcement",
        "signal_family": "news_sentiment",
        "urgency": "breaking",
        "event_type": "listing",
        "symbol": "ABCUSDT",
        "venue": "binance",
        "product": "spot",
        "title": "Binance will list ABC",
        "summary": "",
        "url": "https://example.invalid/evt-1",
        "bias": "positive",
        "source_reliability": 0.9,
        "corroboration_count": 2,
        "quality_flags": "",
        "importance": 0.8,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_price_volume_output_contract():
    frame = _market(
        [
            {"metric": "price_change_24h_pct", "value": 6.2, "unit": "pct"},
            {"metric": "volume_ratio_24h", "value": 2.4, "unit": "ratio"},
            {"metric": "open_interest_change_24h_pct", "value": 24.0, "unit": "pct"},
        ]
    )
    batch = create_advisory_bot("price_volume_monitor").run(_snapshot(market_metrics=frame))
    output = signals_to_frame(batch)
    assert len(output) == 1
    assert output.iloc[0]["direction"] == "long"
    assert output.iloc[0]["product"] == "usd_m_perpetual"
    assert not bool(output.iloc[0]["auto_trade_eligible"])
    assert batch.signals[0].kind == SignalKind.ALERT


def test_social_monitor_degrades_without_sentiment():
    frame = _market(
        [
            {"metric": "rank", "value": 2},
            {"metric": "mention_count", "value": 500},
        ]
    )
    batch = create_advisory_bot("social_sentiment_monitor").run(
        _snapshot(market_metrics=frame)
    )
    assert batch.signals[0].payload["direction"] == "neutral"
    assert batch.signals[0].payload["data_quality"] == "degraded"


def test_price_monitor_does_not_mix_spot_and_futures_metrics():
    futures = _market(
        [
            {"metric": "price_change_24h_pct", "value": 6.0},
            {"metric": "volume_ratio_24h", "value": 2.2},
        ]
    )
    spot = _market(
        [
            {"metric": "price_change_24h_pct", "value": 1.0, "product": "spot"},
            {"metric": "volume_ratio_24h", "value": 1.1, "product": "spot"},
        ]
    )
    batch = create_advisory_bot("price_volume_monitor").run(
        _snapshot(market_metrics=pd.concat([spot, futures], ignore_index=True))
    )
    assert len(batch.signals) == 1
    assert batch.signals[0].payload["product"] == "usd_m_perpetual"


@pytest.mark.parametrize(
    "bot_id,event,expected_direction,expected_action",
    [
        ("exchange_event_monitor", {}, "long", "alert"),
        (
            "macro_institutional_monitor",
            {"topic": "institutional_flow", "event_type": "etf_flow", "flow_usd": -50_000_000},
            "short",
            "watch",
        ),
        (
            "catalyst_risk_monitor",
            {"topic": "security_incident", "event_type": "security", "title": "ABC exploit"},
            "short",
            "avoid",
        ),
    ],
)
def test_news_bots(bot_id, event, expected_direction, expected_action):
    batch = create_advisory_bot(bot_id).run(_snapshot(news_events=_news(**event)))
    assert len(batch.signals) == 1
    payload = batch.signals[0].payload
    assert payload["direction"] == expected_direction
    assert payload["action"] == expected_action
    assert payload["evidence"]
    assert payload["auto_trade_eligible"] is False


def test_future_available_time_is_rejected():
    frame = _market([{"metric": "rank", "value": 1}])
    frame["available_time"] = NOW_DT + pd.Timedelta(seconds=1)
    with pytest.raises(FrameValidationError, match="future data"):
        create_advisory_bot("social_sentiment_monitor").run(
            _snapshot(market_metrics=frame)
        )


def test_event_time_after_availability_is_rejected():
    frame = _news(published_at=NOW_DT + pd.Timedelta(seconds=1))
    with pytest.raises(FrameValidationError, match="later than available_time"):
        create_advisory_bot("exchange_event_monitor").run(
            _snapshot(news_events=frame)
        )


def test_empty_required_frame_emits_data_quality_alert():
    empty = _market([{"metric": "rank", "value": 1}]).iloc[0:0]
    snapshot = DataSnapshot(
        version="1.0",
        meta=SnapshotMeta(
            snapshot_id="empty",
            assembled_at=NOW,
            decision_as_of=NOW,
            source_status={"market_metrics": SourceStatus.ERROR},
            warnings=["upstream timeout"],
        ),
        frames={"market_metrics": empty},
    )
    batch = create_advisory_bot("social_sentiment_monitor").run(snapshot)
    assert batch.signals[0].payload["reason_codes"][0] == "DATA_UNAVAILABLE"
    assert batch.signals[0].payload["data_quality"] == "insufficient"


def test_incremental_step_suppresses_repeated_alert():
    frame = _news()
    bot = create_advisory_bot("exchange_event_monitor")
    first = bot.step(_snapshot(news_events=frame), SignalState(plugin_id=bot.plugin_id))
    later = DataSnapshot(
        version="1.0",
        meta=SnapshotMeta(
            snapshot_id="snapshot-2",
            assembled_at=NOW + 60_000,
            decision_as_of=NOW + 60_000,
        ),
        frames={"news_events": frame},
    )
    second = bot.step(later, first.state)
    assert len(first.signals) == 1
    assert second.signals == []


def test_invalid_threshold_config_fails_fast():
    frame = _market([{"metric": "price_change_24h_pct", "value": 6.0}])
    with pytest.raises(ValueError, match="positive"):
        create_advisory_bot("price_volume_monitor").run(
            _snapshot(market_metrics=frame),
            {"price_move_24h_pct": 0},
        )


def test_unverified_exchange_post_has_no_directional_bias():
    frame = _news(source_reliability=0.35, corroboration_count=0, url="")
    batch = create_advisory_bot("exchange_event_monitor").run(
        _snapshot(news_events=frame)
    )
    assert batch.signals[0].payload["direction"] == "neutral"
    assert batch.signals[0].payload["action"] == "investigate"


def test_canvas_definition_keeps_existing_shell_and_typed_ports():
    dsl = canvas_definition("exchange_event_monitor")
    assert dsl["version"] == "1.1"
    assert dsl["nodes"] and dsl["edges"]
    assert dsl["interface"]["inputs"]["news_events"]["schema"] == "NewsEventFrame@1.0"
    assert dsl["interface"]["outputs"]["alerts"]["schema"] == "BotSignalFrame@1.0"
    assert dsl["bot"]["exposure"] is None
    assert dsl["bot"]["allowedStages"] == ["research", "advisory", "shadow"]


def test_bn_data_adapter_delegates_without_reimplementing_fetch():
    class Result:
        data = [{"value": 1.2}]
        resolved_via = "public:rest"
        validation_status = "BACKTESTABLE"
        pit = True

    class Resolver:
        def resolve(self, field_key, symbol, **kwargs):
            assert field_key == "funding_rate"
            assert symbol == "BTCUSDT"
            return Result()

    result = BnDataCatalogAdapter(Resolver()).fetch("funding_rate", "BTCUSDT")
    assert result.status == "ok"
    assert result.frame.iloc[0]["source_field"] == "funding_rate"
    assert bool(result.frame.iloc[0]["pit_safe"]) is True


def test_bn_data_forward_placeholder_is_not_ok():
    class Result:
        data = {"_forward_only": True, "field": "social_rank"}
        resolved_via = "forward_collect"
        validation_status = "RESEARCH_ONLY"
        pit = False

    class Resolver:
        def resolve(self, field_key, symbol, **kwargs):
            return Result()

    result = BnDataCatalogAdapter(Resolver()).fetch("social_rank", "BTCUSDT")
    assert result.status == "degraded"
    assert result.frame.empty


def test_square_news_adapter_deduplicates_tickers(monkeypatch):
    raw = pd.DataFrame(
        [
            {
                "id": "evt-dup",
                "date": NOW,
                "title": "ABC listing",
                "summary": "",
                "body": "",
                "web_link": "https://example.invalid/dup",
                "tickers": ["ABC"],
                "user_input_tickers": ["ABC"],
                "author_name": "Binance",
                "author_role": "official",
                "is_created_by_ai": False,
                "tendency": 1,
            }
        ]
    )
    monkeypatch.setattr(
        "cyqnt_trd.standard_bot.advisory.data.square_news.fetch_news",
        lambda **kwargs: raw,
    )
    result = SquarePublicAdapter().fetch_news_events(captured_at=NOW_DT)
    assert len(result.frame) == 1
    assert result.frame.iloc[0]["symbol"] == "ABCUSDT"
    assert result.frame.iloc[0]["bias"] == "neutral"


def test_advisory_bot_runs_through_canonical_plugin_registry():
    frame = _market(
        [
            {"metric": "price_change_24h_pct", "value": -7.0},
            {"metric": "volume_ratio_24h", "value": 2.5},
        ]
    )
    registry = register_advisory_bots(SignalPluginRegistry())
    result = registry.run_pipeline_step(
        _snapshot(market_metrics=frame),
        [{"plugin_id": "price_volume_monitor", "config": {}}],
    )
    assert result.batch.signals[0].payload["direction"] == "short"
    assert len(result.states) == 1


def test_required_inputs_match_the_standard_plugin_shape():
    # same Dict[str, bool] surface as BlockStrategyPlugin / SelectionStrategyPlugin
    inputs = create_advisory_bot("exchange_event_monitor").required_inputs()
    assert inputs["news_events"] is True
    assert inputs["market"] is False


def test_required_data_query_exposes_typed_frame_queries():
    query = create_advisory_bot("exchange_event_monitor").required_data_query()
    assert query.frames["news_events"].schema == "NewsEventFrame@1.0"
    assert "news_feed" in query.frames["news_events"].field_keys
