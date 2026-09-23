"""Raw snapshot -> bundle -> load_bundle, on a synthetic snapshot directory.

What is pinned: a rebuilt daily bundle loads exactly like the shipped one; a
non-daily grid and open-interest extras survive the round trip; open interest
is as-of aligned (never read before it is available, never carried across a
feed gap); and the builder refuses to overwrite by default or to run on an
incomplete snapshot.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.eval import load_bundle
from cyqnt_trd.eval.bundle import Panel, to_long
from cyqnt_trd.eval.snapshot import acquire, build_bundle, interval_ms, pipeline

START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2025-04-11", tz="UTC")          # 100 daily bars
SYMBOLS = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]


def _klines(interval: str, seed: int) -> pd.DataFrame:
    step = pd.Timedelta(milliseconds=interval_ms(interval))
    ts = pd.date_range(START, END - step, freq=step)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(ts))))
    open_ = close * np.exp(rng.normal(0, 0.002, len(ts)))
    return pd.DataFrame({"ts": ts, "open": open_, "high": np.maximum(open_, close) * 1.01,
                         "low": np.minimum(open_, close) * 0.99, "close": close,
                         "volume": rng.uniform(1e3, 2e3, len(ts)),
                         "quote_volume": rng.uniform(1e5, 2e5, len(ts))})


def _funding(directory, symbol):
    events = pd.date_range(START, END - pd.Timedelta(hours=8), freq="8h")
    frame = pd.DataFrame({"fundingTime": events.asi8 // 1_000_000 + 4, "fundingRate": 0.0001,
                          "markPrice": 100.0, "fundingIntervalHours": 8,
                          "funding_input_valid": True})
    path = directory / "funding" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({
        "schema_version": 3, "status": "ok", "symbol": symbol, "pagination_complete": True,
        "queried_from": str(START), "queried_through_exclusive": str(END),
        "event_gap_diagnostics": {"gap_affected_utc_days": []}}))


def _open_interest(directory, symbol, *, first: pd.Timestamp, gap=None):
    stamps = pd.date_range(first, END - pd.Timedelta(hours=1), freq="1h")
    if gap is not None:
        stamps = stamps[(stamps < gap[0]) | (stamps >= gap[1])]
    hours = ((stamps - START) / pd.Timedelta(hours=1)).astype(float)
    frame = pd.DataFrame({"timestamp": stamps.asi8 // 1_000_000,
                          "sumOpenInterest": hours, "sumOpenInterestValue": hours * 10})
    path = directory / "open_interest_1h" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@pytest.fixture
def snapshot(tmp_path):
    d = tmp_path / "snap"
    d.mkdir()
    (d / "acquisition_config.json").write_text(json.dumps(
        {"start": str(START), "end_exclusive": str(END), "symbols": SYMBOLS}))
    far = pd.Timestamp("2100-01-01", tz="UTC").value // 1_000_000
    (d / "exchange_info.json").write_text(json.dumps({"symbols": [
        {"symbol": s, "onboardDate": START.value // 1_000_000, "deliveryDate": far} for s in SYMBOLS]}))
    (d / "universes.json").write_text(json.dumps(
        {"current_top10": SYMBOLS, "historical_union": SYMBOLS, "needed_symbols": SYMBOLS}))
    for k, s in enumerate(SYMBOLS):
        for interval, sub in (("1d", "daily"), ("4h", "klines_4h")):
            (d / sub).mkdir(exist_ok=True)
            _klines(interval, k).to_parquet(d / sub / f"{s}.parquet", index=False)
        _funding(d, s)
    return d


def _build(snapshot, out, *extra):
    return build_bundle.main(["--data-dir", str(snapshot), "--out", str(out), *extra])


def test_daily_rebuild_loads_like_the_shipped_bundle(snapshot, tmp_path):
    out = tmp_path / "b.parquet"
    assert _build(snapshot, out) == 0
    panel = load_bundle(out)
    assert panel.cell_scheme == "daily_utc" and panel.extras == {}
    assert panel.symbols == SYMBOLS and len(panel.index) == 100
    raw = _klines("1d", 0).set_index("ts")
    np.testing.assert_allclose(panel.close["AAAUSDT"].to_numpy(), raw.close.to_numpy())
    # three settlements a day at rate 1e-4 x mark 100
    assert panel.funding.stack().to_numpy() == pytest.approx(0.03)
    # >= 60 observed bars before a name is eligible
    assert not panel.mask.iloc[58].any() and panel.mask.iloc[59].all()
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["cell_scheme"] == "daily_utc" and meta["pool"] == "current"


def test_intraday_grid_and_open_interest_survive_the_round_trip(snapshot, tmp_path):
    for s in SYMBOLS:
        _open_interest(snapshot, s, first=START + pd.Timedelta(days=70))
    out = tmp_path / "b4h.parquet"
    assert _build(snapshot, out, "--interval", "4h", "--open-interest", "1h") == 0
    panel = load_bundle(out)
    assert panel.cell_scheme == "regular"
    assert len(panel.index) == 600 and (np.diff(panel.index.asi8) == 4 * 3_600 * 10**9).all()
    assert sorted(panel.extras) == ["open_interest", "open_interest_value"]
    # a settlement just after 00/08/16h lands in that 4h cell, the others pay nothing
    f = panel.funding["AAAUSDT"]
    np.testing.assert_allclose(f[f.index.hour % 8 == 0].to_numpy(), 0.01)
    assert (f[f.index.hour % 8 != 0] == 0).all()


def test_open_interest_is_as_of_and_never_leaks_or_bridges_gaps(snapshot):
    first = START + pd.Timedelta(days=70)
    gap = (START + pd.Timedelta(days=80), START + pd.Timedelta(days=81))
    _open_interest(snapshot, "AAAUSDT", first=first, gap=gap)
    index = pd.date_range(START, END - pd.Timedelta(hours=4), freq="4h")
    frames, detail = pipeline.open_interest_panel(index, ["AAAUSDT", "BBBUSDT"], snapshot, period="1h")
    oi = frames["open_interest"]["AAAUSDT"]
    hours = (index - START) / pd.Timedelta(hours=1)
    # cell t decides at t+4h; a 1h stamp is available one hour later -> stamp t+3h
    live = (index >= first) & ((index + pd.Timedelta(hours=4) <= gap[0]) | (index >= gap[1]))
    np.testing.assert_array_equal(oi[live].to_numpy(), (hours + 3)[live])
    assert oi[index + pd.Timedelta(hours=3) < first].isna().all()        # nothing before the feed starts
    inside = (index >= gap[0] + pd.Timedelta(hours=1)) & (index + pd.Timedelta(hours=4) < gap[1])
    assert inside.any() and oi[inside].isna().all()                      # no bridging a gap
    assert frames["open_interest"]["BBBUSDT"].isna().all()               # no file -> missing, not zero
    assert detail[1]["status"] == "missing_file"


def test_lifecycle_uses_the_bar_width():
    index = pd.date_range("2025-01-01", periods=6, freq="4h", tz="UTC")
    registry = {"symbols": [{"symbol": "A", "onboardDate": index[0].value // 1_000_000,
                             "deliveryDate": pd.Timestamp("2025-01-01 10:00", tz="UTC").value // 1_000_000}]}
    live_bar, live_open = pipeline.lifecycle_masks(index, ["A"], registry, pd.Timedelta(hours=4))
    assert live_bar.A.tolist() == [True, True, False, False, False, False]    # 08-12h crosses delivery
    assert live_open.A.tolist() == [True, True, True, False, False, False]


def test_builder_refuses_to_overwrite_by_default_and_reports_missing_inputs(snapshot, tmp_path, capsys):
    with pytest.raises(SystemExit, match="--out is required"):
        build_bundle.main(["--data-dir", str(snapshot)])
    assert build_bundle.main(["--data-dir", str(snapshot), "--dry-run"]) == 0
    assert "ready to build" in capsys.readouterr().out
    empty = tmp_path / "empty"
    empty.mkdir()
    assert build_bundle.main(["--data-dir", str(empty), "--dry-run"]) == 1
    with pytest.raises(SystemExit, match="snapshot incomplete"):
        build_bundle.main(["--data-dir", str(empty), "--out", str(tmp_path / "x.parquet")])
    assert build_bundle.main(["--data-dir", str(snapshot), "--dry-run", "--open-interest", "1h"]) == 1


@pytest.mark.parametrize("bad", ["1w", "0h", "7h", "abc"])
def test_intervals_that_do_not_tile_the_day_are_refused(snapshot, tmp_path, bad):
    if bad == "7h":
        with pytest.raises(ValueError, match="does not tile"):
            pipeline.load_inputs("current", snapshot, interval=bad)
    else:
        with pytest.raises(SystemExit):
            _build(snapshot, tmp_path / "x.parquet", "--interval", bad)


def test_extras_cannot_shadow_core_bundle_fields(snapshot, tmp_path):
    _build(snapshot, tmp_path / "b.parquet")
    p = load_bundle(tmp_path / "b.parquet")
    clash = Panel(**{f: getattr(p, f) for f in ("open", "high", "low", "close", "volume", "quote_volume")},
                  funding=p.funding, mask=p.mask, extras={"eligible": p.close * 0})
    with pytest.raises(ValueError, match="collide"):
        to_long(clash)


def test_open_interest_acquisition_merges_with_what_is_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "DATA", tmp_path)
    end = pd.Timestamp("2025-02-01", tz="UTC")
    step = 3_600_000
    stop = end.value // 1_000_000

    def fake(first_ms, n):
        return [{"symbol": "AAAUSDT", "timestamp": first_ms + i * step,
                 "sumOpenInterest": str(i), "sumOpenInterestValue": str(10 * i)} for i in range(n)]

    calls = []

    def api(method, url, params=None, **kw):
        calls.append(params)
        start = params["startTime"]
        n = min(500, (params["endTime"] + 1 - start) // step)
        return fake(start, n)

    monkeypatch.setattr(acquire, "api", api)
    # the window is counted from the server's now; pretend "now" is the cut-off
    monkeypatch.setattr(acquire, "now_ms", lambda: stop)
    first = acquire.oi_one("AAAUSDT", end, "1h")
    assert first["status"] == "ok"
    # starts inside the ~30-day window (an older startTime is a 400), not at START
    assert calls[0]["startTime"] == stop - acquire.OI_HISTORY_DAYS * 86_400_000 + 3_600_000
    assert first["rows"] == acquire.OI_HISTORY_DAYS * 24 - 1
    calls.clear()
    later = pd.Timestamp("2025-02-02", tz="UTC")
    second = acquire.oi_one("AAAUSDT", later, "1h")
    assert calls[0]["startTime"] == stop                  # resumes after the last stored stamp
    assert second["rows"] == first["rows"] + 24 and second["new_rows"] == 24
    stored = pd.read_parquet(tmp_path / "open_interest_1h" / "AAAUSDT.parquet")
    assert stored.timestamp.is_unique and stored.timestamp.is_monotonic_increasing


def test_open_interest_for_a_cut_off_older_than_the_window_is_no_data_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "DATA", tmp_path)
    end = pd.Timestamp("2025-02-01", tz="UTC")
    monkeypatch.setattr(acquire, "now_ms", lambda: end.value // 1_000_000 + 40 * 86_400_000)

    def api(*a, **k):
        raise AssertionError("no request may be sent for an unreachable window")

    monkeypatch.setattr(acquire, "api", api)
    result = acquire.oi_one("AAAUSDT", end, "1h")
    assert result["status"] == "no_data" and "window" in result["reason"]


def test_klines_stage_needs_the_daily_stage_first(tmp_path):
    with pytest.raises(SystemExit, match="run --stage daily first"):
        acquire.main(["--data-dir", str(tmp_path), "--stage", "klines", "--interval", "4h"])
