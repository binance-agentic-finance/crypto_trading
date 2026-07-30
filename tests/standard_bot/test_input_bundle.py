"""Input bundle: one JSON, every source, one decision time.

The two things worth pinning down are the ones that were wrong first time:

* **the PIT gate** — a row is only in the bundle if we could already have known
  it at ``decision_time``. Gating once at build time is the whole point; a
  reader that forgets cannot reintroduce lookahead.
* **the lookback window** — a bundle is the input at ONE moment, so each series
  carries a window, not its whole history. Bars were bounded by ``max_bars``
  while metric frames were not, so a single 1h decision pulled in 30 days of
  5-minute open interest: 12,144 rows and 94% of a 1.7 MB file. Bounding the
  metric frames the same way took the same bundle to 163 KB with identical
  strategy output.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.data.input_bundle import (
    FRAME_SHAPES, build_input_bundle, load_input_bundle, write_input_bundle)
from cyqnt_trd.standard_bot.data.internal_slots import INTERNAL_SLOTS
from cyqnt_trd.standard_bot.core import Bar, MarketBundle

HOUR = 3_600_000
DT = 100 * HOUR


def _bars(n=400, tf="1h", symbol="BTCUSDT"):
    out = []
    for i in range(n):
        close_time = (i + 1) * HOUR - 1
        out.append(Bar(open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.5 + i,
                       volume=10.0, timestamp=close_time, instrument_id=symbol,
                       timeframe=tf, confirmed=True, quote_volume=1000.0,
                       extras={"open_time": close_time - HOUR + 1,
                               "close_time": close_time}))
    return out


def _metric_df(n, metric="open_interest", step=HOUR // 12, symbol="BTCUSDT"):
    return pd.DataFrame({
        "timestamp": [i * step for i in range(n)],
        "instrument_id": [symbol] * n,
        metric: [1000.0 + i for i in range(n)],
    })


def _build(**kw):
    params = dict(symbol="BTCUSDT", interval="1h", decision_time=DT,
                  bars=_bars(), max_bars=300)
    params.update(kw)
    return build_input_bundle(**params)


# --------------------------------------------------------------------------- #
# structure                                                                    #
# --------------------------------------------------------------------------- #


def test_bundle_is_one_json_with_every_declared_source():
    bundle = _build(
        news_frame=pd.DataFrame({"event_id": ["e1"], "event_time": [DT - HOUR],
                                 "available_time": [DT - HOUR], "source_id": ["sq"],
                                 "topic": ["etf"]}),
        ticker_rank_frame=pd.DataFrame({"instrument_id": ["BTCUSDT"],
                                        "available_time": [DT], "rank": [1]}),
        universe_frame=pd.DataFrame({"instrument_id": ["BTCUSDT"],
                                     "available_time": [DT], "quote_volume": [5e8]}))
    assert bundle["schema"] == "cyqnt.input/v1"
    assert bundle["decision_time"] == DT
    assert {"klines", "news", "ticker_rank", "universe"} <= set(bundle["frames"])
    for key, spec in bundle["frames"].items():
        assert spec["shape"] == FRAME_SHAPES.get(key, "RawFrame@1.0")
    assert json.dumps(bundle)          # must be JSON-serialisable end to end


def test_round_trip_rebuilds_a_usable_snapshot(tmp_path):
    bundle = _build(universe_frame=pd.DataFrame(
        {"instrument_id": ["BTCUSDT"], "available_time": [DT], "quote_volume": [5e8]}))
    path = write_input_bundle(bundle, str(tmp_path / "b.json"))
    snap = load_input_bundle(path)
    assert snap.meta.decision_as_of == DT
    # 400 bars were offered but only 100 close at or before DT, so the PIT gate
    # keeps 100 — max_bars=300 never binds. Asserting 300 here would have been
    # asserting that the gate leaked.
    assert len(snap.require_market().bars[MarketBundle.key("BTCUSDT", "1h")]) == 100
    assert snap.universe is not None


# --------------------------------------------------------------------------- #
# PIT gate                                                                     #
# --------------------------------------------------------------------------- #


def test_rows_after_the_decision_time_are_dropped():
    future = pd.DataFrame({"event_id": ["past", "future"],
                           "event_time": [DT - HOUR, DT + HOUR],
                           "available_time": [DT - HOUR, DT + HOUR],
                           "source_id": ["sq", "sq"], "topic": ["a", "b"]})
    bundle = _build(news_frame=future)
    ids = [row["event_id"] for row in bundle["frames"]["news"]["rows"]]
    assert ids == ["past"], "a row we could not yet have known must not be bundled"


def test_decision_time_defaults_to_the_last_confirmed_bar():
    bundle = build_input_bundle(symbol="BTCUSDT", interval="1h", bars=_bars(10))
    assert bundle["decision_time"] == 10 * HOUR - 1


# --------------------------------------------------------------------------- #
# lookback window — the 1.7 MB bug                                             #
# --------------------------------------------------------------------------- #


def test_metric_frames_are_windowed_per_series(tmp_path):
    """Each (instrument, metric) keeps its newest N — not its whole history."""
    wide = _metric_df(5_000)
    wide["open_interest_value"] = wide["open_interest"] * 2
    root = tmp_path / "futures" / "BTCUSDT"
    root.mkdir(parents=True)
    wide.to_parquet(root / "open_interest_1h.parquet")

    bundle = _build(derivatives_dir=str(tmp_path), metric_lookback=50)
    rows = bundle["frames"]["open_interest"]["rows"]
    per_metric = {}
    for row in rows:
        per_metric.setdefault(row["metric"], []).append(row)
    assert set(per_metric) == {"open_interest", "open_interest_value"}
    for metric, series in per_metric.items():
        assert len(series) == 50, "%s kept %d rows, expected the newest 50" % (
            metric, len(series))
        # newest kept, oldest dropped
        assert series[-1]["value"] > series[0]["value"]


def test_unbounded_lookback_is_opt_in():
    wide = _metric_df(500)
    bundle = _build(bars=_bars(10), metric_lookback=None, max_event_rows=None,
                    extra_frames={"internal_metrics": pd.DataFrame({
                        "event_time": [i for i in range(500)],
                        "available_time": [i for i in range(500)],
                        "instrument_id": ["BTCUSDT"] * 500,
                        "metric": ["x"] * 500,
                        "value": [float(i) for i in range(500)]})})
    assert len(bundle["frames"]["internal_metrics"]["rows"]) == 500


def test_windowing_keeps_the_bundle_small(tmp_path):
    """Regression on size: the whole point is that one decision is not 1.7 MB."""
    wide = _metric_df(8_640)                      # 30 days of 5-minute OI
    wide["open_interest_value"] = wide["open_interest"] * 2
    root = tmp_path / "futures" / "BTCUSDT"
    root.mkdir(parents=True)
    wide.to_parquet(root / "open_interest_1h.parquet")

    bounded = _build(derivatives_dir=str(tmp_path))          # default lookback
    unbounded = _build(derivatives_dir=str(tmp_path), metric_lookback=None)
    n_bounded = len(bounded["frames"]["open_interest"]["rows"])
    n_unbounded = len(unbounded["frames"]["open_interest"]["rows"])
    assert n_bounded == 240 * 2, "two metrics x the default 240-row window"
    # The PIT gate already dropped everything after DT, so "unbounded" means
    # every 5-minute observation up to the decision — 1,201 per metric here.
    assert n_unbounded == 1_201 * 2
    assert n_unbounded > n_bounded * 2
    assert (len(json.dumps(bounded).encode())
            < len(json.dumps(unbounded).encode()) / 2), "and far smaller on disk"


# --------------------------------------------------------------------------- #
# internal slots — fields public, client private                               #
# --------------------------------------------------------------------------- #


def test_declared_internal_slots_report_status_even_with_no_client():
    bundle = _build(declare_internal=list(INTERNAL_SLOTS))
    for key in INTERNAL_SLOTS:
        assert key in bundle["source_status"], (
            "%s must appear in source_status so 'no client' is distinguishable "
            "from 'returned nothing'" % key)
        assert key not in bundle["frames"]


def test_supplying_an_internal_frame_normalises_it_to_the_declared_shape():
    etf = pd.DataFrame({"token": ["BTC"], "date": [DT], "flow": [3.1e8],
                        "net_assets": [9.2e10], "close_price": [74985.9]})
    bundle = _build(internal_frames={"internal_etf_flow": etf})
    rows = bundle["frames"]["internal_etf_flow"]["rows"]
    assert {row["metric"] for row in rows} == {"flow", "net_assets", "close_price"}
    for row in rows:
        assert set(row) == {"event_time", "available_time", "instrument_id",
                            "metric", "value"}, "must match MetricFrame@1.0"


def test_public_and_internal_bundles_have_the_same_shape():
    """A bundle built outside the network differs only in rows, never in structure."""
    public = _build(declare_internal=list(INTERNAL_SLOTS))
    internal = _build(declare_internal=list(INTERNAL_SLOTS), internal_frames={
        "internal_etf_flow": pd.DataFrame({"token": ["BTC"], "date": [DT],
                                           "flow": [1.0], "net_assets": [2.0],
                                           "close_price": [3.0]})})
    assert set(public["source_status"]) == set(internal["source_status"])
    assert set(internal["frames"]) - set(public["frames"]) == {"internal_etf_flow"}


def test_non_pit_safe_internal_data_warns():
    radar = pd.DataFrame({"symbol": ["BTCUSDT"], "metric": ["oi_change_24h"],
                          "value": [-0.05], "event_time": [DT]})
    bundle = _build(internal_frames={"internal_futures_radar": radar})
    assert any("point-in-time" in w for w in bundle["warnings"]), (
        "a snapshot source with no history must say so, or someone will replay it")


def test_unknown_internal_slot_is_reported_not_silently_dropped():
    bundle = _build(declare_internal=["internal_does_not_exist"])
    assert any("unknown internal slot" in w for w in bundle["warnings"])
