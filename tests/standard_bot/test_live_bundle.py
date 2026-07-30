"""``build_live_bundle`` — one JSON from live nodes, and the invariants it owes.

The tests that matter here are not "is the file small". They are:

* every requested node has a status, including the ones that failed;
* times on the wire are integer epoch ms, one representation only;
* rows the caller could not have known at ``decision_time`` are gone;
* a frame whose series does not reach back to the first bar is *reported*.

That last one is the regression this module exists for. The earlier bundle bound
metric frames by row count, so 240 rows of 5-minute open interest covered 20
hours next to 300 hourly bars: 93% of bars had no open-interest value and nothing
said so. A small file looked like a fixed file.

No network: every test drives the fetch through a stub catalog node.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.data import live_bundle as LB
from cyqnt_trd.standard_bot.data.catalog import (
    Availability, DataNodeSpec, FrameKind, ParamSpec, ReturnSpec, SourcePath,
    register_node, unregister_node,
)

HOUR = 3_600_000
T0 = 1_780_000_000_000            # arbitrary fixed epoch ms
DECISION = T0 + 300 * HOUR


def _bars(n: int, *, end: int = DECISION, step: int = HOUR) -> pd.DataFrame:
    stamps = [end - (n - 1 - i) * step for i in range(n)]
    return pd.DataFrame({
        "open_time": [s - step for s in stamps], "close_time": stamps,
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": [100.0 + i for i in range(n)], "volume": 5.0,
    })


def _metric_series(n: int, *, end: int = DECISION, step: int) -> pd.DataFrame:
    stamps = [end - (n - 1 - i) * step for i in range(n)]
    return pd.DataFrame({"timestamp": stamps, "oi_value": [1000.0 + i for i in range(n)]})


@pytest.fixture
def stub_nodes(request):
    """Register throwaway catalog nodes backed by in-memory frames."""
    made = []

    def make(name, frame_or_exc, *, emits, availability=Availability.BACKTESTABLE,
             column_map=None, value_columns=(), constants=None):
        holder = {"payload": frame_or_exc}
        fq = "%s._stub_%s" % (__name__, name)

        def fetcher(**_):
            payload = holder["payload"]
            if isinstance(payload, Exception):
                raise payload
            return payload

        globals()["_stub_%s" % name] = fetcher
        spec = DataNodeSpec(
            name=name, description="stub", strategy_types=("T",),
            source_path=SourcePath.PUBLIC_BINANCE, availability=availability,
            returns=ReturnSpec("pd.DataFrame", "stub"),
            params=(ParamSpec("symbol", "str"),),
            pit_hazard="" if availability is Availability.BACKTESTABLE else "stub",
            fetcher=fq, emits=emits, column_map=dict(column_map or {}),
            constants=dict(constants or {}), value_columns=tuple(value_columns),
        )
        register_node(spec, replace=True)
        made.append(name)
        return spec

    yield make
    for name in made:
        unregister_node(name)


def _build(plan, **kwargs):
    return LB.build_live_bundle(requests=plan, decision_time=DECISION,
                                symbol="BTCUSDT", interval="1h", **kwargs)


def test_every_requested_node_gets_a_status_including_failures(stub_nodes):
    stub_nodes("t_bars", _bars(10), emits=FrameKind.BAR,
               constants={"instrument_id": "BTCUSDT", "timeframe": "1h"})
    stub_nodes("t_broken", RuntimeError("upstream said no"), emits=FrameKind.METRIC)

    bundle = _build([("t_bars", {}, "t_bars"), ("t_broken", {}, "t_broken")])

    assert set(bundle["source_status"]) == {"t_bars", "t_broken"}
    assert bundle["source_status"]["t_bars"] == "ok"
    assert bundle["source_status"]["t_broken"].startswith("error:")
    # the failed node keeps its slot, so "not read" != "read and empty"
    assert bundle["frames"]["t_broken"]["rows"] == []
    assert "upstream said no" in bundle["frames"]["t_broken"]["reason"]


def test_times_on_the_wire_are_integer_epoch_ms(stub_nodes):
    stub_nodes("t_bars", _bars(5), emits=FrameKind.BAR,
               constants={"instrument_id": "BTCUSDT", "timeframe": "1h"})
    bundle = _build([("t_bars", {}, "t_bars")])
    rows = bundle["frames"]["t_bars"]["rows"]
    for column in ("open_time", "close_time", "available_time"):
        assert all(isinstance(row[column], int) for row in rows), column
    json.dumps(bundle)          # must round-trip with no custom encoder


def test_rows_we_could_not_have_known_are_dropped(stub_nodes):
    """The gate the runtime declared but never applied."""
    future = _bars(4, end=DECISION + 3 * HOUR)
    stub_nodes("t_bars", future, emits=FrameKind.BAR,
               constants={"instrument_id": "BTCUSDT", "timeframe": "1h"})
    bundle = _build([("t_bars", {}, "t_bars")])
    entry = bundle["frames"]["t_bars"]
    assert entry["rows"], "the in-window bar must survive"
    assert all(row["close_time"] <= DECISION for row in entry["rows"])
    assert entry["rows_after_decision_time_dropped"] == 3


def test_nothing_fetched_is_thrown_away_by_default(stub_nodes):
    """What was requested is what the strategy gets.

    The bundle used to shorten every non-bar frame to the bar span. On a real
    snapshot that discarded 1,482 rows — most of funding, fear_greed and ahr999,
    every one of them asked for by an explicit ``limit`` — to save 303 KB. The
    1.7 MB bundle that motivated a window came from a granularity mismatch, and
    that is fixed where the data is requested, not by deleting what came back.
    """
    stub_nodes("t_bars", _bars(10), emits=FrameKind.BAR,
               constants={"instrument_id": "BTCUSDT", "timeframe": "1h"})
    # a daily series reaching far further back than the 10 bars
    stub_nodes("t_long", _metric_series(400, step=24 * HOUR),
               emits=FrameKind.METRIC, column_map={"timestamp": "event_time"},
               constants={"instrument_id": "BTCUSDT"}, value_columns=("oi_value",))

    bundle = _build([("t_bars", {}, "t_bars"), ("t_long", {}, "t_long")])
    assert len(bundle["frames"]["t_long"]["rows"]) == 400, (
        "every fetched row must survive: %d kept"
        % len(bundle["frames"]["t_long"]["rows"]))
    # the window is still stated, so a reader knows what the bars cover
    assert "window_start" in bundle["frames"]["t_long"]

    trimmed = _build([("t_bars", {}, "t_bars"), ("t_long", {}, "t_long")],
                     trim_to_bars=True)
    assert len(trimmed["frames"]["t_long"]["rows"]) < 400, "opt-in trim still works"


def test_metric_window_follows_the_bars_not_a_row_count(stub_nodes):
    """A 5-minute series must keep its cadence, not be capped by row count."""
    stub_nodes("t_bars", _bars(300), emits=FrameKind.BAR,
               constants={"instrument_id": "BTCUSDT", "timeframe": "1h"})
    stub_nodes("t_oi", _metric_series(8640, step=5 * 60_000),  # 30 days of 5m
               emits=FrameKind.METRIC, column_map={"timestamp": "event_time"},
               constants={"instrument_id": "BTCUSDT"}, value_columns=("oi_value",))

    bundle = _build([("t_bars", {}, "t_bars"), ("t_oi", {}, "t_oi")])
    oi = bundle["frames"]["t_oi"]
    times = [row["event_time"] for row in oi["rows"]]

    bars = [row["close_time"] for row in bundle["frames"]["t_bars"]["rows"]]
    assert min(times) <= min(bars) + HOUR, (
        "the series must reach back to the first bar; trimming by row count is "
        "what left 93%% of bars uncovered")
    gaps = {b - a for a, b in zip(sorted(times), sorted(times)[1:])}
    assert gaps == {5 * 60_000}, "the 5m cadence must be preserved, not resampled"
    assert len(times) == 8640, "every fetched row is kept by default"


def test_a_series_that_starts_after_the_bars_is_reported(stub_nodes):
    """The exact original defect: OI covering 20h against 300h of bars."""
    stub_nodes("t_bars", _bars(300), emits=FrameKind.BAR,
               constants={"instrument_id": "BTCUSDT", "timeframe": "1h"})
    stub_nodes("t_oi", _metric_series(240, step=5 * 60_000),   # only ~20h
               emits=FrameKind.METRIC, availability=Availability.SEMI,
               column_map={"timestamp": "event_time"},
               constants={"instrument_id": "BTCUSDT"}, value_columns=("oi_value",))

    bundle = _build([("t_bars", {}, "t_bars"), ("t_oi", {}, "t_oi")])
    holes = [w for w in bundle["warnings"] if "have\n" not in w and "no value for it" in w]
    assert any("t_oi" in w for w in holes), (
        "a metric series covering 20h of a 300h window must be reported, not "
        "silently shipped: %s" % bundle["warnings"])


def test_snapshot_sources_do_not_raise_a_coverage_hole(stub_nodes):
    """A FORWARD_ONLY snapshot has one timestamp by nature — warning is noise."""
    stub_nodes("t_bars", _bars(300), emits=FrameKind.BAR,
               constants={"instrument_id": "BTCUSDT", "timeframe": "1h"})
    stub_nodes("t_book", pd.DataFrame({"price": [1.0, 2.0], "qty": [3.0, 4.0],
                                       "side": ["bid", "ask"]}),
               emits=FrameKind.BOOK, availability=Availability.FORWARD_ONLY,
               column_map={"qty": "quantity"},
               constants={"instrument_id": "BTCUSDT"})

    bundle = _build([("t_bars", {}, "t_bars"), ("t_book", {}, "t_book")])
    assert not [w for w in bundle["warnings"] if "t_book" in w and "no value for it" in w]
    # …and the book still normalises: level is derived from row order
    assert [row["level"] for row in bundle["frames"]["t_book"]["rows"]] == [1, 1]


def test_an_empty_response_is_empty_not_an_error(stub_nodes):
    """'Read it, there was nothing' must not present as a fetch failure."""
    stub_nodes("t_empty", pd.DataFrame(columns=["timestamp", "oi_value"]),
               emits=FrameKind.METRIC, column_map={"timestamp": "event_time"},
               constants={"instrument_id": "BTCUSDT"}, value_columns=("oi_value",))
    bundle = _build([("t_empty", {}, "t_empty")])
    assert bundle["source_status"]["t_empty"] == "empty"
    assert bundle["frames"]["t_empty"]["rows"] == []
    assert "reason" not in bundle["frames"]["t_empty"]


def test_bundle_loads_back_into_a_snapshot(stub_nodes):
    """The live artifact must be readable by the same loader as the offline one."""
    from cyqnt_trd.standard_bot.core import MarketBundle
    from cyqnt_trd.standard_bot.data.input_bundle import load_input_bundle

    stub_nodes("klines_stub", _bars(12), emits=FrameKind.BAR,
               constants={"instrument_id": "BTCUSDT", "timeframe": "1h"})
    bundle = _build([("klines_stub", {}, "klines")])
    snapshot = load_input_bundle(bundle)
    bars = snapshot.require_market().bars[MarketBundle.key("BTCUSDT", "1h")]
    assert len(bars) == 12
    assert snapshot.meta.decision_as_of == DECISION


def test_a_slow_collection_does_not_discard_the_freshest_sources(stub_nodes, monkeypatch):
    """The decision time is when collection FINISHES, not when it started.

    Collecting seventeen nodes takes tens of seconds, and a snapshot source
    stamps itself with the moment it was generated — after the session began. The
    gate used to run mid-loop against the session's start time, so those rows
    were dropped for being "from the future" and the frame arrived empty, which
    reads as "the source had nothing" rather than "we threw it away".
    ``ticker_rank`` lost all 20 of its rows to this while a direct call to the
    same node returned them every time.
    """
    stub_nodes("t_bars", _bars(10), emits=FrameKind.BAR,
               constants={"instrument_id": "BTCUSDT", "timeframe": "1h"})

    # Collection is slow, and the snapshot stamps itself when it is generated —
    # i.e. after the session started but before collection ends. That is the
    # real shape: ticker_rank came back 22.4s after as_of because the sixteen
    # nodes ahead of it took that long.
    def _slow_then_fresh(**_):
        import time as _t

        _t.sleep(0.30)                       # the other nodes taking their time
        now = int(_t.time() * 1000)
        return pd.DataFrame({"instrument_id": ["BTCUSDT"], "rank": [1],
                             "score": [90.0], "available_time": [now],
                             "event_time": [now]})

    stub_nodes("t_snapshot", pd.DataFrame(), emits=FrameKind.RANK,
               availability=Availability.FORWARD_ONLY)
    globals()["_stub_t_snapshot"] = _slow_then_fresh

    bundle = LB.build_live_bundle(
        requests=[("t_bars", {}, "t_bars"), ("t_snapshot", {}, "t_snapshot")],
        symbol="BTCUSDT", interval="1h")          # live: no decision_time

    assert bundle["decision_time_basis"] == "collection_complete"
    assert bundle["frames"]["t_snapshot"]["rows"], (
        "a source stamped after the session started must not be gated away in "
        "live mode: %s" % bundle["source_status"])
    assert bundle["decision_time"] >= bundle["frames"]["t_snapshot"]["rows"][0]["available_time"]


def test_replay_still_refuses_rows_published_after_the_decision(stub_nodes):
    """The relaxation is live-only. An explicit decision_time stays strict."""
    stub_nodes("t_bars", _bars(10), emits=FrameKind.BAR,
               constants={"instrument_id": "BTCUSDT", "timeframe": "1h"})
    late = DECISION + 10 * HOUR
    stub_nodes("t_late", pd.DataFrame({"instrument_id": ["BTCUSDT"], "rank": [1],
                                       "score": [1.0], "available_time": [late],
                                       "event_time": [late]}),
               emits=FrameKind.RANK, availability=Availability.FORWARD_ONLY)

    bundle = LB.build_live_bundle(
        requests=[("t_bars", {}, "t_bars"), ("t_late", {}, "t_late")],
        symbol="BTCUSDT", interval="1h", decision_time=DECISION)

    assert bundle["decision_time_basis"] == "replay"
    assert bundle["decision_time"] == DECISION
    assert bundle["frames"]["t_late"]["rows"] == [], "a replay must not read the future"
    assert bundle["frames"]["t_late"]["rows_after_decision_time_dropped"] == 1
