"""Advisory route: snapshot assembler, live market-metric adapter, and the
derivatives positioning bot."""

from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.advisory import (
    BinanceMarketMetricAdapter,
    FetchFrame,
    create_advisory_bot,
    list_advisory_bots,
    signals_to_frame,
)
from cyqnt_trd.standard_bot.data import build_advisory_snapshot, run_advisory
from cyqnt_trd.standard_bot.advisory import data as advisory_data
from cyqnt_trd.standard_bot.core import SignalKind, SourceStatus

NOW = 1_786_000_000_000
NOW_DT = pd.to_datetime(NOW, unit="ms", utc=True)


def _metric_rows(rows, *, symbol="BTCUSDT", product="usd_m_perpetual", available=None):
    defaults = {
        "event_time": available or NOW_DT,
        "available_time": available or NOW_DT,
        "venue": "binance",
        "product": product,
        "symbol": symbol,
        "unit": "",
        "window": "24h",
        "source_id": "fixture",
        "quality": "ok",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _fetched(frame, status="ok"):
    return FetchFrame(frame=frame, status=status, source="fixture")


def _run(rows, config=None, **kwargs):
    snapshot = build_advisory_snapshot(
        frames={"market_metrics": _fetched(_metric_rows(rows, **kwargs))},
        decision_as_of=NOW_DT,
    )
    bot = create_advisory_bot("derivatives_positioning_monitor")
    return bot.run(snapshot, config or {})


def _payload(batch):
    assert len(batch.signals) == 1
    return batch.signals[0].payload


# --------------------------------------------------------------------------
# assembler
# --------------------------------------------------------------------------


def test_assembler_records_source_status_and_decision_time():
    snapshot = build_advisory_snapshot(
        frames={
            "market_metrics": _fetched(
                _metric_rows([{"metric": "price_change_24h_pct", "value": 1.0}])
            ),
            "news_events": FetchFrame(
                frame=pd.DataFrame(), status="error", source="square", warning="boom"
            ),
        },
        decision_as_of=NOW_DT,
    )
    assert snapshot.meta.decision_as_of == NOW
    assert snapshot.meta.source_status["market_metrics"] is SourceStatus.OK
    assert snapshot.meta.source_status["news_events"] is SourceStatus.ERROR
    assert any("boom" in warning for warning in snapshot.meta.warnings)


def test_assembler_gates_rows_published_after_the_decision_cutoff():
    future = NOW_DT + pd.Timedelta(minutes=5)
    frame = pd.concat(
        [
            _metric_rows([{"metric": "price_change_24h_pct", "value": 1.0}]),
            _metric_rows(
                [{"metric": "volume_ratio_24h", "value": 9.0}], available=future
            ),
        ],
        ignore_index=True,
    )
    snapshot = build_advisory_snapshot(
        frames={"market_metrics": _fetched(frame)}, decision_as_of=NOW_DT
    )
    kept = snapshot.frames["market_metrics"]
    assert list(kept["metric"]) == ["price_change_24h_pct"]
    assert any("dropped 1 row" in warning for warning in snapshot.meta.warnings)


def test_assembler_defaults_decision_time_to_latest_availability():
    latest = NOW_DT + pd.Timedelta(minutes=7)
    frame = pd.concat(
        [
            _metric_rows([{"metric": "price_change_24h_pct", "value": 1.0}]),
            _metric_rows([{"metric": "volume_ratio_24h", "value": 2.0}], available=latest),
        ],
        ignore_index=True,
    )
    snapshot = build_advisory_snapshot(frames={"market_metrics": _fetched(frame)})
    assert snapshot.meta.decision_as_of == int(latest.timestamp() * 1000)
    assert len(snapshot.frames["market_metrics"]) == 2


def test_assembler_downgrades_ok_frame_that_was_fully_gated_out():
    future = NOW_DT + pd.Timedelta(minutes=5)
    snapshot = build_advisory_snapshot(
        frames={
            "market_metrics": _fetched(
                _metric_rows(
                    [{"metric": "price_change_24h_pct", "value": 1.0}], available=future
                )
            )
        },
        decision_as_of=NOW_DT,
    )
    # An "ok" fetch whose every row is in the future is NOT evidence of "no events".
    assert snapshot.meta.source_status["market_metrics"] is SourceStatus.DEGRADED


def test_assembler_requires_at_least_one_frame():
    with pytest.raises(ValueError):
        build_advisory_snapshot(frames={})


# --------------------------------------------------------------------------
# derivatives positioning bot
# --------------------------------------------------------------------------


def test_crowded_long_funding_reads_short_and_stays_advisory():
    batch = _run(
        [
            {"metric": "funding_rate_annualized_pct", "value": 55.0},
            {"metric": "price_change_24h_pct", "value": 1.0},
        ]
    )
    payload = _payload(batch)
    assert batch.signals[0].kind is SignalKind.ALERT
    assert payload["direction"] == "short"
    assert payload["action"] == "watch"
    assert "FUNDING_CROWDED_LONG" in payload["reason_codes"]
    assert payload["auto_trade_eligible"] is False
    assert payload["evidence"]


def test_negative_funding_reads_long():
    payload = _payload(_run([{"metric": "funding_rate_annualized_pct", "value": -60.0}]))
    assert payload["direction"] == "long"
    assert "FUNDING_CROWDED_SHORT" in payload["reason_codes"]


def test_funding_is_derived_from_the_8h_rate_when_apr_is_absent():
    payload = _payload(_run([{"metric": "funding_rate_8h", "value": 0.0005}]))
    # 0.0005 * 3 * 365 * 100 = 54.75% annualized -> crowded long
    assert payload["direction"] == "short"
    assert "54.75" in payload["summary"]


def test_two_agreeing_votes_escalate_watch_to_alert():
    payload = _payload(
        _run(
            [
                {"metric": "funding_rate_annualized_pct", "value": 40.0},
                {"metric": "open_interest_change_24h_pct", "value": 25.0},
                {"metric": "price_change_24h_pct", "value": -3.0},
            ]
        )
    )
    assert payload["direction"] == "short"
    assert payload["action"] == "alert"
    assert "OI_UP_PRICE_DOWN_NEW_SHORTS" in payload["reason_codes"]


def test_full_crowding_stack_escalates_to_avoid():
    payload = _payload(
        _run(
            [
                {"metric": "funding_rate_annualized_pct", "value": 120.0},
                {"metric": "open_interest_change_24h_pct", "value": 35.0},
                {"metric": "long_short_account_ratio", "value": 3.4},
                {"metric": "price_change_24h_pct", "value": 4.0},
            ]
        )
    )
    assert payload["direction"] == "short"
    assert payload["action"] == "avoid"
    assert "SQUEEZE_RISK" in payload["reason_codes"]
    assert payload["urgency"] == "breaking"


def test_oi_unwind_is_reported_without_claiming_a_direction():
    payload = _payload(
        _run(
            [
                {"metric": "open_interest_change_24h_pct", "value": -22.0},
                {"metric": "price_change_24h_pct", "value": 5.0},
            ]
        )
    )
    assert payload["direction"] == "neutral"
    assert "OI_DOWN_PRICE_UP_SHORT_COVERING" in payload["reason_codes"]


def test_opposing_votes_of_similar_weight_refuse_to_pick_a_side():
    payload = _payload(
        _run(
            [
                # funding: crowded long -> short (severity 1.5)
                {"metric": "funding_rate_annualized_pct", "value": 45.0},
                # OI + price: new longs -> long (severity 1.5)
                {"metric": "open_interest_change_24h_pct", "value": 30.0},
                {"metric": "price_change_24h_pct", "value": 4.0},
            ]
        )
    )
    assert payload["direction"] == "neutral"
    assert "CONFLICTING_POSITIONING" in payload["reason_codes"]
    assert payload["action"] == "watch"


def test_quiet_positioning_produces_no_alert():
    batch = _run(
        [
            {"metric": "funding_rate_annualized_pct", "value": 5.0},
            {"metric": "open_interest_change_24h_pct", "value": 2.0},
            {"metric": "long_short_account_ratio", "value": 1.1},
            {"metric": "price_change_24h_pct", "value": 0.4},
        ]
    )
    assert batch.signals == []


def test_missing_funding_is_not_read_as_flat_funding():
    payload = _payload(
        _run(
            [
                {"metric": "open_interest_change_24h_pct", "value": 30.0},
                {"metric": "price_change_24h_pct", "value": 4.0},
            ]
        )
    )
    assert "FUNDING_CROWDED_LONG" not in payload["reason_codes"]
    assert "FUNDING_CROWDED_SHORT" not in payload["reason_codes"]
    assert payload["data_quality"] == "degraded"
    assert "资金费率年化=NA" in payload["summary"]


def test_spot_rows_never_receive_perp_positioning_alerts():
    batch = _run(
        [
            {"metric": "funding_rate_annualized_pct", "value": 90.0},
            {"metric": "long_short_account_ratio", "value": 4.0},
        ],
        product="spot",
    )
    assert batch.signals == []


def test_stale_metrics_are_skipped():
    stale = NOW_DT - pd.Timedelta(hours=6)
    batch = _run(
        [{"metric": "funding_rate_annualized_pct", "value": 90.0}], available=stale
    )
    assert batch.signals == []


def test_invalid_derivatives_config_fails_fast():
    bot = create_advisory_bot("derivatives_positioning_monitor")
    with pytest.raises(ValueError):
        bot.normalize_config({"funding_annualized_extreme_pct": 10.0})
    with pytest.raises(ValueError):
        bot.normalize_config({"long_short_ratio_high": 0.4})
    with pytest.raises(ValueError):
        bot.normalize_config({"unknown_knob": 1})


def test_derivatives_bot_is_registered_and_declares_its_data():
    ids = [item["bot_id"] for item in list_advisory_bots()]
    assert "derivatives_positioning_monitor" in ids
    meta = create_advisory_bot("derivatives_positioning_monitor").meta
    keys = {item.field_key for item in meta.data_requirements}
    assert {"funding_rate", "open_interest", "long_short_ratio"} <= keys
    assert meta.advisory_only is True


def test_output_frame_carries_the_standard_columns():
    batch = _run([{"metric": "funding_rate_annualized_pct", "value": 55.0}])
    frame = signals_to_frame(batch)
    assert frame.loc[0, "bot_id"] == "derivatives_positioning_monitor"
    assert frame.loc[0, "product"] == "usd_m_perpetual"
    assert bool(frame.loc[0, "auto_trade_eligible"]) is False


# --------------------------------------------------------------------------
# live market-metric adapter (fetchers stubbed; no network)
# --------------------------------------------------------------------------


class _StubModules:
    """Minimal stand-ins for the data_cli modules the adapter imports lazily."""

    def __init__(self, **frames):
        self.frames = frames
        self.calls = []

    def install(self, monkeypatch):
        import sys
        import types

        def module(name, **attrs):
            mod = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(mod, key, value)
            monkeypatch.setitem(sys.modules, "cyqnt_trd.data_cli.%s" % name, mod)
            monkeypatch.setattr(
                "cyqnt_trd.data_cli.%s" % name, mod, raising=False
            )
            return mod

        module("ticker", fetch_24h_ticker=self._reader("ticker"))
        module("kline", fetch_klines=self._reader("kline"))
        module("oi", fetch_oi_history=self._reader("oi"))
        module("funding", fetch_funding_history=self._reader("funding"))
        module(
            "ratios",
            fetch_long_short_ratio=self._reader("long_short"),
            fetch_basis=self._reader("basis"),
        )

    def _reader(self, key):
        def read(*args, **kwargs):
            self.calls.append(key)
            value = self.frames.get(key)
            if isinstance(value, Exception):
                raise value
            return pd.DataFrame() if value is None else value

        return read


def _adapter_fixtures():
    return {
        "ticker": pd.DataFrame([{"change_pct": 3.5}]),
        "kline": pd.DataFrame(
            {"quote_volume": [100.0, 100.0, 300.0], "close_time": [1, 2, NOW]}
        ),
        "oi": pd.DataFrame(
            {"timestamp": [NOW - 86_400_000, NOW], "oi_value": [100.0, 130.0]}
        ),
        "funding": pd.DataFrame([{"rate": 0.0004, "timestamp": NOW}]),
        "long_short": pd.DataFrame([{"long_short_ratio": 2.4, "timestamp": NOW}]),
        "basis": pd.DataFrame([{"basis_rate": 0.004, "timestamp": NOW}]),
    }


def test_market_metric_adapter_emits_one_row_per_metric(monkeypatch):
    _StubModules(**_adapter_fixtures()).install(monkeypatch)
    fetched = BinanceMarketMetricAdapter().fetch_market_metrics(
        ["BTCUSDT"], captured_at=NOW_DT
    )
    assert fetched.status == "ok"
    metrics = dict(zip(fetched.frame["metric"], fetched.frame["value"]))
    assert metrics["price_change_24h_pct"] == pytest.approx(3.5)
    assert metrics["volume_ratio_24h"] == pytest.approx(3.0)
    assert metrics["open_interest_change_24h_pct"] == pytest.approx(30.0)
    assert metrics["funding_rate_8h"] == pytest.approx(0.0004)
    assert metrics["funding_rate_annualized_pct"] == pytest.approx(0.0004 * 3 * 365 * 100)
    assert metrics["long_short_account_ratio"] == pytest.approx(2.4)
    assert metrics["basis_rate_pct"] == pytest.approx(0.4)
    assert (fetched.frame["available_time"] == NOW_DT).all()
    assert (fetched.frame["event_time"] <= fetched.frame["available_time"]).all()


def test_market_metric_adapter_degrades_on_a_failing_source(monkeypatch):
    fixtures = _adapter_fixtures()
    fixtures["funding"] = RuntimeError("cli unavailable")
    _StubModules(**fixtures).install(monkeypatch)
    fetched = BinanceMarketMetricAdapter().fetch_market_metrics(
        ["BTCUSDT"], captured_at=NOW_DT
    )
    assert fetched.status == "degraded"
    assert "cli unavailable" in fetched.warning
    # missing funding must be absent, never zero-filled
    assert "funding_rate_8h" not in set(fetched.frame["metric"])


def test_market_metric_adapter_skips_perp_only_metrics_on_spot(monkeypatch):
    _StubModules(**_adapter_fixtures()).install(monkeypatch)
    fetched = BinanceMarketMetricAdapter(
        product="spot", market="spot"
    ).fetch_market_metrics(["BTCUSDT"], captured_at=NOW_DT)
    emitted = set(fetched.frame["metric"])
    assert emitted == {"price_change_24h_pct", "volume_ratio_24h"}
    assert "perpetual-only" in fetched.warning


def test_market_metric_adapter_errors_when_nothing_could_be_read(monkeypatch):
    _StubModules().install(monkeypatch)
    fetched = BinanceMarketMetricAdapter().fetch_market_metrics(
        ["BTCUSDT"], captured_at=NOW_DT
    )
    assert fetched.status == "error"
    assert fetched.frame.empty


def test_market_metric_adapter_requires_symbols():
    with pytest.raises(ValueError):
        BinanceMarketMetricAdapter().fetch_market_metrics([])


def test_run_advisory_bot_end_to_end_offline(monkeypatch):
    _StubModules(**_adapter_fixtures()).install(monkeypatch)
    # No decision_as_of: the assembler dates the snapshot off the freshest
    # available_time the adapter stamped, which is what a live monitor does.
    result = run_advisory("derivatives_positioning_monitor", symbols=["BTCUSDT"])
    frame = signals_to_frame(result.batch)
    # ran through SignalPluginRegistry.run_pipeline_step, so state came back too
    assert len(result.states) == 1
    assert len(frame) == 1
    assert frame.loc[0, "symbol"] == "BTCUSDT"
    # +43.8% APR funding says "crowded long", OI +30% with price +3.5% says
    # "new longs" — a near-tie, so no side is claimed.
    assert frame.loc[0, "direction"] == "neutral"
    assert "CONFLICTING_POSITIONING" in frame.loc[0, "reason_codes"]


def test_run_advisory_bot_rejects_market_metric_bot_without_symbols():
    with pytest.raises(ValueError):
        run_advisory("derivatives_positioning_monitor")


# --------------------------------------------------------------------------
# standard-bot wiring: same registry, same run_pipeline_step, no executor path
# --------------------------------------------------------------------------


def test_advisory_bots_are_seeded_into_the_standard_registry():
    from cyqnt_trd.standard_bot.entrypoints.common import make_registry

    registry = make_registry()
    for item in list_advisory_bots():
        assert registry.get(item["bot_id"]).plugin_id == item["bot_id"]
    # and they coexist with the builtin trade plugins in one registry
    assert registry.get("moving_average_cross") is not None


def test_advisory_bot_satisfies_the_signal_plugin_surface():
    bot = create_advisory_bot("derivatives_positioning_monitor")
    assert bot.plugin_id == "derivatives_positioning_monitor"
    assert bot.plugin_version
    assert isinstance(bot.required_inputs(), dict)
    assert all(isinstance(value, bool) for value in bot.required_inputs().values())
    assert bot.initialize_state().plugin_id == bot.plugin_id


def test_build_advisory_pipeline_validates_config_before_any_fetch():
    from cyqnt_trd.standard_bot.entrypoints.common import build_advisory_pipeline

    spec = build_advisory_pipeline(
        bot="derivatives_positioning_monitor", config={"oi_growth_24h_pct": 10.0}
    )
    assert spec.plugin_chain == [
        {
            "plugin_id": "derivatives_positioning_monitor",
            "config": {"oi_growth_24h_pct": 10.0},
        }
    ]
    with pytest.raises(ValueError):
        build_advisory_pipeline(
            bot="derivatives_positioning_monitor", config={"oi_growth_24h_pct": -1}
        )


def test_registry_step_suppresses_an_alert_already_delivered():
    from cyqnt_trd.standard_bot.entrypoints.common import make_registry

    registry = make_registry()
    rows = [{"metric": "funding_rate_annualized_pct", "value": 55.0}]
    chain = [{"plugin_id": "derivatives_positioning_monitor", "config": {}}]
    snapshot = build_advisory_snapshot(
        frames={"market_metrics": _fetched(_metric_rows(rows))}, decision_as_of=NOW_DT
    )
    first = registry.run_pipeline_step(snapshot, chain)
    assert len(first.batch.signals) == 1
    second = registry.run_pipeline_step(
        snapshot, chain, previous_states=first.states
    )
    assert second.batch.signals == []


def test_alert_signals_never_become_execution_intents():
    from cyqnt_trd.standard_bot.execution.interfaces import ExecutionPlanner

    batch = _run([{"metric": "funding_rate_annualized_pct", "value": 120.0}])
    assert batch.signals  # the bot did emit
    assert ExecutionPlanner().build_intents(batch) == []


# --------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------


def _run_cli(monkeypatch, argv):
    import sys

    from cyqnt_trd.standard_bot.entrypoints import mvp_advisory

    monkeypatch.setattr(sys, "argv", ["mvp_advisory"] + argv)
    return mvp_advisory.main()


def test_cli_lists_bots(monkeypatch, capsys):
    import json

    assert _run_cli(monkeypatch, ["--list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert "derivatives_positioning_monitor" in {item["bot_id"] for item in listed}


def test_cli_replays_frames_from_csv_without_network(monkeypatch, tmp_path, capsys):
    import json

    csv_path = tmp_path / "market_metrics.csv"
    _metric_rows(
        [
            {"metric": "funding_rate_annualized_pct", "value": 120.0},
            {"metric": "open_interest_change_24h_pct", "value": 35.0},
            {"metric": "long_short_account_ratio", "value": 3.4},
            {"metric": "price_change_24h_pct", "value": 4.0},
        ],
        available=pd.Timestamp.now(tz="UTC").floor("ms"),
    ).to_csv(csv_path, index=False)

    code = _run_cli(
        monkeypatch,
        [
            "--bot", "derivatives_positioning_monitor",
            "--frame", "market_metrics=%s" % csv_path,
            "--format", "json",
        ],
    )
    assert code == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["action"] == "avoid"
    assert rows[0]["auto_trade_eligible"] is False


def test_cli_requires_a_bot(monkeypatch):
    assert _run_cli(monkeypatch, []) == 2


def test_cli_rejects_an_unknown_bot_before_fetching(monkeypatch):
    with pytest.raises(KeyError):
        _run_cli(monkeypatch, ["--bot", "no_such_monitor"])


def test_empty_required_frame_with_error_status_reports_data_unavailable():
    snapshot = build_advisory_snapshot(
        frames={
            "market_metrics": FetchFrame(
                frame=pd.DataFrame(
                    columns=advisory_data._MARKET_METRIC_COLUMNS
                ),
                status="error",
                source="fixture",
                warning="cli unavailable",
            )
        },
        decision_as_of=NOW_DT,
    )
    batch = create_advisory_bot("derivatives_positioning_monitor").run(snapshot, {})
    payload = _payload(batch)
    assert "DATA_UNAVAILABLE" in payload["reason_codes"]
    assert payload["data_quality"] == "insufficient"
    assert payload["action"] == "investigate"
