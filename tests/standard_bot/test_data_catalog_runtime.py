"""Data-node catalog, the runtime data/action surface, and the DAG strategy
that compiles onto them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.standard_bot.core import SignalKind
from cyqnt_trd.standard_bot.data.catalog import (
    Availability,
    DataUnavailable,
    SourcePath,
    get_node,
    list_node_names,
    list_nodes,
)
from cyqnt_trd.standard_bot.runtime import action, data


# --------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------


def test_every_node_is_self_describing():
    for spec in list_nodes():
        assert spec.name and spec.description
        assert isinstance(spec.availability, Availability)
        assert isinstance(spec.source_path, SourcePath)
        assert spec.returns.type
        # anything not freely replayable must say how it lies
        if spec.availability in (Availability.SEMI, Availability.FORWARD_ONLY):
            assert spec.pit_hazard or spec.notes, (
                "%s is %s but documents no PIT hazard" % (spec.name, spec.availability.value)
            )


#: Fetcher modules that are intentionally absent from this (public) package.
#: ``cyqnt_trd.data_cli.internal*`` hardcodes corporate-network hostnames
#: (corporate-network hosts, deliberately not named here), so it is kept out
#: of the public repo and injected via a private config instead — see the note in
#: ``cyqnt_trd/data_cli/__init__.py``. The catalog still *describes* those nodes,
#: because knowing a source exists (and how it lies under replay) is useful even
#: where the client is not shipped.
OPTIONAL_FETCHER_MODULES = ("cyqnt_trd.data_cli.internal",)


def _is_optional_fetcher(spec) -> bool:
    return any(spec.fetcher.startswith(m) for m in OPTIONAL_FETCHER_MODULES)


def test_every_wired_fetcher_resolves():
    unresolved = []
    skipped = []
    for spec in list_nodes():
        if not spec.fetcher:
            continue
        try:
            assert callable(spec.resolve_fetcher())
        except DataUnavailable as exc:
            (skipped if _is_optional_fetcher(spec) else unresolved).append(str(exc))
    assert unresolved == [], (
        "these fetchers are wired but cannot be imported, and they are NOT "
        "declared optional: %s" % unresolved
    )


def test_optional_internal_fetchers_fail_loudly_when_absent():
    """Nodes backed by the unshipped internal client must say so, not return empty.

    Every one of them still has to be fully described in the catalog (schema,
    availability, PIT hazard) so a strategy author can see what exists; only the
    HTTP client is missing.
    """
    optional = [s for s in list_nodes() if s.fetcher and _is_optional_fetcher(s)]
    assert optional, "expected the catalog to describe internal-backed nodes"
    for spec in optional:
        assert spec.returns.type, "%s must still declare its return schema" % spec.name
        assert isinstance(spec.availability, Availability)
        try:
            spec.resolve_fetcher()
        except DataUnavailable as exc:
            assert "internal" in str(exc), (
                "%s should name the missing internal module, got: %s" % (spec.name, exc)
            )
        else:
            # Present (private config injected it) — then it must be callable.
            assert callable(spec.resolve_fetcher())


def test_unwired_node_reports_why_instead_of_returning_empty():
    spec = get_node("btc_dominance")
    assert spec.availability is Availability.EXTERNAL_PENDING
    with pytest.raises(DataUnavailable) as excinfo:
        spec.fetch()
    assert "no fetcher wired" in str(excinfo.value)


def test_backtestable_set_is_the_honest_short_list():
    from cyqnt_trd.standard_bot.data.catalog import backtestable_node_names

    names = backtestable_node_names()
    # deep public history
    assert {"klines", "funding", "fear_greed", "ahr999"} <= set(names)
    # snapshot-only sources must NOT be in it
    assert "ticker_rank" not in names
    assert "bdp_screen" not in names
    assert "orderbook_depth" not in names


def test_unknown_node_lists_the_known_ones():
    with pytest.raises(KeyError) as excinfo:
        get_node("no_such_feed")
    assert "klines" in str(excinfo.value)


# --------------------------------------------------------------------------
# runtime.data
# --------------------------------------------------------------------------


def _stub(monkeypatch, node: str, value):
    spec = get_node(node)
    monkeypatch.setattr(
        spec.__class__, "fetch",
        lambda self, **kw: value if self.name == node else spec.fetch(**kw),
        raising=False,
    )


def test_validate_nodes_blocks_a_backtest_on_snapshot_data():
    problems = data.validate_nodes(["klines", "ticker_rank"], for_backtest=True)
    assert len(problems) == 1
    assert "ticker_rank" in problems[0]
    assert "FORWARD_ONLY" in problems[0]
    # the reason must be quoted, not just the verdict
    assert "no history" in problems[0] or "forward-collect" in problems[0]


def test_validate_nodes_allows_snapshot_data_for_live():
    assert data.validate_nodes(["klines", "ticker_rank"]) == []


def test_validate_nodes_flags_a_feed_that_is_not_onboarded():
    problems = data.validate_nodes(["btc_dominance"])
    assert problems and "EXTERNAL_PENDING" in problems[0]


def test_session_stamps_one_decision_time_for_every_node(monkeypatch):
    frame = pd.DataFrame({"close": [1.0, 2.0]})
    monkeypatch.setattr(
        "cyqnt_trd.data_cli.kline.fetch_klines", lambda **kw: frame, raising=False
    )
    with data.session(as_of_ms=1_700_000_000_000) as sess:
        sess.call("klines", symbol="BTCUSDT", interval="1h", limit=2)
        assert sess.as_of_ms == 1_700_000_000_000
        assert sess.results["klines"].as_of == 1_700_000_000_000
        assert sess.status_map() == {"klines": "ok"}


def test_backtest_session_refuses_a_forward_only_node():
    with data.session(require_backtestable=True) as sess:
        with pytest.raises(DataUnavailable) as excinfo:
            sess.call("ticker_rank", window="24h")
    assert "requires BACKTESTABLE" in str(excinfo.value)


def test_empty_frame_is_degraded_not_ok(monkeypatch):
    monkeypatch.setattr(
        "cyqnt_trd.data_cli.news.fetch_news", lambda **kw: pd.DataFrame(), raising=False
    )
    with data.session() as sess:
        sess.call("news", lang="en", page_size=5)
    assert sess.status_map()["news"] == "degraded"
    assert "do not read this as 'no data'" in sess.results["news"].warning


def test_collect_degrades_a_failing_node_without_sinking_the_rest(monkeypatch):
    frame = pd.DataFrame({"close": [1.0]})
    monkeypatch.setattr(
        "cyqnt_trd.data_cli.kline.fetch_klines", lambda **kw: frame, raising=False
    )

    def boom(**kw):
        raise ConnectionError("cli unreachable")

    monkeypatch.setattr(
        "cyqnt_trd.data_cli.funding.fetch_funding_history", boom, raising=False
    )
    frames, sess = data.collect(
        [
            ("klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 1}),
            ("funding", {"symbol": "BTCUSDT", "limit": 10}),
        ]
    )
    assert set(frames) == {"klines"}
    assert sess.status_map() == {"klines": "ok", "funding": "error"}
    assert any("cli unreachable" in w for w in sess.warnings())


def test_generated_node_function_enforces_required_params():
    with pytest.raises(TypeError) as excinfo:
        data.klines(interval="1h")
    assert "symbol" in str(excinfo.value)


def test_generated_node_function_applies_catalog_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "cyqnt_trd.data_cli.kline.fetch_klines",
        lambda **kw: seen.update(kw) or pd.DataFrame({"close": [1.0]}),
        raising=False,
    )
    data.klines(symbol="BTCUSDT")
    assert seen["interval"] == "1h"
    assert seen["limit"] == 500
    # The catalog declares this param as ``market_type`` (that is the public name
    # in DATA_API.md and in a spec) while the fetcher takes ``market``. The node
    # must hand over the fetcher's spelling: injecting the declared one is what
    # made every data.klines() call raise TypeError.
    assert seen["market"] == "futures"
    assert "market_type" not in seen


def test_describe_matches_the_generated_doc_surface():
    described = data.describe("funding")
    assert described["availability"] == "BACKTESTABLE"
    assert described["module"] == "data"
    assert {item["key"] for item in described["params"]} == {"symbol", "limit"}


# --------------------------------------------------------------------------
# runtime.action
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_sink():
    action.reset()
    yield
    action.reset()


def test_order_emits_a_trade_envelope_and_places_nothing():
    envelope = action.order(
        {"symbol": "BTCUSDT", "side": "long", "size": 0.1},
        exit_config={"stop_loss": {"type": "atr_trailing", "multiplier": 2.0}},
    )
    assert envelope.kind is SignalKind.TRADE
    assert envelope.instrument_id == "BTCUSDT"
    assert envelope.side.value == "buy"
    assert envelope.payload["exit_spec"]["stop_loss"]["multiplier"] == 2.0
    # default sink records, it does not execute
    assert isinstance(action.get_sink(), action.RecordingSink)
    assert len(action.current_batch().trade_signals()) == 1


def test_order_none_is_the_gate_closed_branch_not_an_error():
    assert action.order(None) is None
    assert action.current_batch().signals == []


def test_order_rejects_a_symbol_without_a_usable_side():
    with pytest.raises(ValueError) as excinfo:
        action.order({"symbol": "BTCUSDT", "side": "maybe"})
    assert "side must be one of" in str(excinfo.value)


def test_alert_is_never_routable_to_an_order():
    from cyqnt_trd.standard_bot.execution.interfaces import ExecutionPlanner

    action.alert({"symbol": "ETHUSDT", "direction": "short", "score": 80,
                  "auto_trade_eligible": True})
    batch = action.current_batch()
    assert batch.signals[0].kind is SignalKind.ALERT
    # the payload claim is overridden, and ALERT is not a trade signal anyway
    assert batch.signals[0].payload["auto_trade_eligible"] is False
    assert ExecutionPlanner().build_intents(batch) == []


def test_a_custom_sink_receives_every_envelope():
    class Capturing(action.ActionSink):
        def __init__(self):
            self.seen = []

        def emit(self, envelope):
            self.seen.append(envelope)

    sink = Capturing()
    previous = action.set_sink(sink)
    try:
        action.order({"symbol": "BTCUSDT", "side": "long"})
        action.notify("hello", symbol="BTCUSDT")
    finally:
        action.set_sink(previous)
    assert [e.kind for e in sink.seen] == [SignalKind.TRADE, SignalKind.NOOP]


def test_close_position_targets_flat():
    envelope = action.close_position("BTCUSDT", reason="stop")
    assert envelope.side.value == "flat"
    assert envelope.payload["target_position"] == 0


# --------------------------------------------------------------------------
# the compiled DAG strategy
# --------------------------------------------------------------------------


def _synthetic_klines(n: int = 300, *, trend: float = 1.0) -> pd.DataFrame:
    idx = np.arange(n)
    close = 100.0 + trend * idx * 0.15 + np.sin(idx / 9.0) * 1.2
    high = close + 0.6
    low = close - 0.6
    open_ = np.concatenate([[close[0]], close[:-1]])
    close_time = 1_700_000_000_000 + idx * 3_600_000
    return pd.DataFrame(
        {
            "open_time": close_time - 3_600_000,
            "open": open_, "high": high, "low": low, "close": close,
            "volume": np.full(n, 100.0), "quote_volume": np.full(n, 100.0) * close,
            "close_time": close_time, "trades": np.full(n, 10),
        }
    )


def _install_dag_stubs(monkeypatch, *, funding=None, oi=None, ls=None):
    klines = _synthetic_klines()
    monkeypatch.setattr(
        "cyqnt_trd.data_cli.kline.fetch_klines", lambda **kw: klines, raising=False
    )

    def _or_fail(value, message):
        def reader(**kw):
            if value is None:
                raise ConnectionError(message)
            return value

        return reader

    monkeypatch.setattr(
        "cyqnt_trd.data_cli.funding.fetch_funding_history",
        _or_fail(funding, "funding unreachable"), raising=False,
    )
    monkeypatch.setattr(
        "cyqnt_trd.data_cli.oi.fetch_oi_history",
        _or_fail(oi, "oi unreachable"), raising=False,
    )
    monkeypatch.setattr(
        "cyqnt_trd.data_cli.ratios.fetch_long_short_ratio",
        _or_fail(ls, "ls unreachable"), raising=False,
    )
    return klines


def test_dag_runs_and_degrades_when_positioning_data_is_missing(monkeypatch):
    _install_dag_stubs(monkeypatch)
    from strategies.dag import mtf_trend_breakout_v2 as dag

    result = dag.run_dag()
    assert result["data_quality"] == "degraded"
    assert result["source_status"]["klines"] == "ok"
    assert result["source_status"]["funding"] == "error"
    # something was emitted either way — silence is never the answer
    assert action.current_batch().signals


def test_dag_emits_a_reason_when_the_gate_is_closed(monkeypatch):
    # flat series -> no MACD golden cross, no ADX trend
    flat = _synthetic_klines(trend=0.0)
    monkeypatch.setattr(
        "cyqnt_trd.data_cli.kline.fetch_klines", lambda **kw: flat, raising=False
    )
    for path in (
        "cyqnt_trd.data_cli.funding.fetch_funding_history",
        "cyqnt_trd.data_cli.oi.fetch_oi_history",
        "cyqnt_trd.data_cli.ratios.fetch_long_short_ratio",
    ):
        monkeypatch.setattr(path, lambda **kw: pd.DataFrame(), raising=False)
    from strategies.dag import mtf_trend_breakout_v2 as dag

    result = dag.run_dag()
    assert result["entry"] is False
    assert result["blocked_by"]
    alerts = [s for s in action.current_batch().signals if s.kind is SignalKind.ALERT]
    assert len(alerts) == 1
    assert alerts[0].payload["reason_codes"] == result["blocked_by"]


def test_dag_refuses_to_run_if_a_declared_node_is_not_runnable(monkeypatch):
    from strategies.dag import mtf_trend_breakout_v2 as dag

    monkeypatch.setattr(dag, "DATA_NODES", ["klines", "btc_dominance"])
    with pytest.raises(ValueError) as excinfo:
        dag.run_dag()
    assert "EXTERNAL_PENDING" in str(excinfo.value)


def test_both_forms_share_one_registered_strategy():
    from cyqnt_trd.blocks import strategy as strat
    from strategies.dag import mtf_trend_breakout_v2 as dag

    plugin = strat.get_block_plugin(dag.BOT_ID)
    assert plugin is not None
    assert plugin.required_inputs()["derivatives"] is True

    long, short = dag.make_signals(_synthetic_klines())
    assert len(long) == len(short) == 300
    assert long.dtype == bool
    assert not short.any()  # long-only template


def test_backtest_form_actually_fires_and_runs_through_the_engine():
    """A conjunction of three *cross* events never fires — guard against it.

    The v1 spec ANDs three `*_cross` conditions; crosses are single-bar events,
    so the strategy validates, compiles, and then produces zero signals forever.
    This locks in the trigger + state-filter shape that does fire.
    """
    from cyqnt_trd.standard_bot.simulation.vectorized_backtest import (
        run_vectorized_backtest,
    )
    from strategies.dag import mtf_trend_breakout_v2 as dag

    rng = np.random.default_rng(7)
    n = 1200
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    idx = np.arange(n)
    df = pd.DataFrame(
        {
            "open_time": 1_700_000_000_000 + idx * 3_600_000 - 3_600_000,
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close + 1.0, "low": close - 1.0, "close": close,
            "volume": np.full(n, 100.0), "quote_volume": np.full(n, 100.0) * close,
            "close_time": 1_700_000_000_000 + idx * 3_600_000,
            "trades": np.full(n, 10),
        }
    )
    long, _short = dag.make_signals(df)
    assert long.sum() >= 1, "entry conjunction never fires — check cross-vs-state"

    result = run_vectorized_backtest(
        df=df, signal_fn=dag.make_signals,
        exit_cfg={"type": "atr_stop_tp", "atr_period": 14, "stop_mult": 2.0, "tp_mult": 3.0},
        size=0.1, initial_capital=10_000.0, long_only=True,
    )
    assert result.trade_count >= 1


def test_backtest_form_prefers_the_real_htf_column_when_the_engine_joined_it():
    from strategies.dag import mtf_trend_breakout_v2 as dag

    df = _synthetic_klines(400)
    column = "_htf_%s_sma_%d" % (dag.CONFIG["htf_interval"], dag.CONFIG["htf_ma_slow"])
    # HTF filter forced off => no entries regardless of the 1h trigger
    df[column] = df["close"] + 1_000.0
    long, _ = dag.make_signals(df)
    assert not long.any()


def test_positioning_veto_needs_two_agreeing_votes():
    from strategies.dag import mtf_trend_breakout_v2 as dag

    close = pd.Series(np.linspace(100, 130, 60))
    hot_funding = pd.Series(np.full(60, 0.0020))          # 20 bps -> crowded long
    rising_oi = pd.Series(100.0 * 1.3 ** np.arange(60))   # +30% per bar -> surge
    calm_ls = pd.Series(np.full(60, 1.0))                 # not crowded

    veto_two, reasons_two = dag._positioning_veto(
        close=close, funding=hot_funding, open_interest=rising_oi,
        long_short=calm_ls, cfg=dag.CONFIG,
    )
    assert bool(veto_two.iloc[-1]) is True
    assert "FUNDING_CROWDED_LONG" in reasons_two

    veto_one, _ = dag._positioning_veto(
        close=close, funding=hot_funding,
        open_interest=pd.Series(np.full(60, 100.0)),      # flat OI, no vote
        long_short=calm_ls, cfg=dag.CONFIG,
    )
    assert bool(veto_one.iloc[-1]) is False


def test_positioning_veto_cannot_fire_on_fewer_than_two_readable_inputs():
    from strategies.dag import mtf_trend_breakout_v2 as dag

    close = pd.Series(np.linspace(100, 130, 40))
    veto, reasons = dag._positioning_veto(
        close=close, funding=pd.Series(np.full(40, 0.0020)),
        open_interest=None, long_short=None, cfg=dag.CONFIG,
    )
    assert not veto.any()
    assert "OI_MISSING" in reasons and "LS_MISSING" in reasons


# --------------------------------------------------------------------------
# generated doc stays in sync with the catalog
# --------------------------------------------------------------------------


def test_data_api_doc_is_not_stale():
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo / "docs" / "gen_data_api.py"), "--check"],
        capture_output=True, text=True, cwd=str(repo),
    )
    assert result.returncode == 0, result.stderr


def test_data_api_doc_lists_every_catalog_node():
    from pathlib import Path

    doc = (
        Path(__file__).resolve().parents[2]
        / "cyqnt_trd" / "standard_bot" / "data" / "DATA_API.md"
    ).read_text(encoding="utf-8")
    for name in list_node_names():
        assert "`data.%s`" % name in doc
