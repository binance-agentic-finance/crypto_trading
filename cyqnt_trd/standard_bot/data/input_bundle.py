"""One JSON in, everything in it — ``cyqnt.input/v1`` bundles.

The problem this solves
-----------------------
Until now "the input" meant a different file format per source: OHLCV had
``--input-json``, funding and open interest were parquet under
``--derivatives-dir``, Square news had **no file format at all** (live API only),
and internal BigData nodes had their own client. A bot that wants price *and*
funding *and* news therefore could not be fed from one artifact, which means a
run could not be reproduced, versioned, diffed or handed to someone else.

An input bundle is that one artifact: **every declared source, normalised to the
canonical frame shapes, gated to a single ``decision_time``, in one JSON file.**

    {
      "schema": "cyqnt.input/v1",
      "decision_time": 1776311999999,
      "frames": {
        "klines":        {"shape": "BarFrame@1.0",    "rows": [...]},
        "funding":       {"shape": "MetricFrame@1.0", "rows": [...]},
        "open_interest": {"shape": "MetricFrame@1.0", "rows": [...]},
        "news":          {"shape": "EventFrame@1.0",  "rows": [...]},
        "ticker_rank":   {"shape": "RankFrame@1.0",   "rows": [...]}
      },
      "source_status": {"klines": "ok", "news": "error: no offline source"},
      "warnings": [...]
    }

Two invariants make it worth having:

**One clock.** Every row is filtered to ``available_time <= decision_time``
before it is written. ``available_time`` is when we could first have *known* the
row, which is not when the thing *happened* — conflating the two is how a
walk-forward backtest silently reads the future. Gating once, at bundle build
time, means no downstream reader can get it wrong.

**One vocabulary.** Funding, open interest, liquidations and any internal metric
all land as ``MetricFrame`` rows (``instrument_id`` / ``metric`` / ``value`` /
``event_time`` / ``available_time``). A bot reading three of them uses one set of
column names instead of three, and a *new* source needs no new plumbing — it is
just more MetricFrame rows.

A source that could not be read is recorded in ``source_status`` rather than
omitted, so "I did not read it" stays distinguishable from "I read it and it was
empty".
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..core import (
    Bar, DataSnapshot, MarketBundle, SnapshotMeta, UniverseBundle,
)
from ..core.input_contract import (INPUT_SCHEMA_VERSION, FrameKind, TypedFrame,
                                   schema_for)
from .internal_slots import (INTERNAL_SLOTS, internal_client_available,
                             normalize_internal_frame, slot_frame_shapes)

__all__ = [
    "build_input_bundle",
    "load_input_bundle",
    "write_input_bundle",
    "read_input_bundle",
    "BUNDLE_NAMESPACE",
]

BUNDLE_NAMESPACE = uuid.UUID("c1a5f0b2-7e34-5d19-9a6c-3f82b41d0e75")

#: node key -> canonical shape. Adding a source means adding a row here, not a
#: new container: the reader does not need to know what "funding" means, only
#: that it is a MetricFrame.
FRAME_SHAPES: Dict[str, str] = {
    "klines": "BarFrame@1.0",
    # Bars for MANY instruments at MANY timeframes, for a cross-sectional screen
    # that runs a technical indicator on each candidate. ``BarFrame@1.0`` already
    # requires ``instrument_id`` AND ``timeframe``, so that grain is legal in the
    # shape as it stands and nothing here needed widening.
    #
    # It is a SEPARATE key from ``klines`` on purpose: :func:`load_input_bundle`
    # special-cases ``klines`` into a ``MarketBundle`` keyed on the FIRST row's
    # instrument, so a multi-symbol frame landed there would file every
    # instrument's bars under one name and the strategy would read the wrong
    # series with no error anywhere.
    "universe_bars": "BarFrame@1.0",
    "funding": "MetricFrame@1.0",
    "open_interest": "MetricFrame@1.0",
    "liquidations": "MetricFrame@1.0",
    "long_short_ratio": "MetricFrame@1.0",
    "taker_ratio": "MetricFrame@1.0",
    "internal_metrics": "MetricFrame@1.0",
    "news": "EventFrame@1.0",
    "ticker_rank": "RankFrame@1.0",
    "universe": "RankFrame@1.0",
    "positions": "PositionFrame@1.0",
    "orderbook": "BookFrame@1.0",
}
# Internal-domain slots are declared in the public repo (fields, shape, PIT
# safety) while their client is not — see internal_slots.py. A bundle built
# without the client is structurally identical, only source_status differs.
FRAME_SHAPES.update(slot_frame_shapes())


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _rows(frame) -> List[Dict[str, Any]]:
    """DataFrame -> list of JSON-safe dicts (NaN becomes null)."""
    if frame is None:
        return []
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        return list(frame)
    if frame.empty:
        return []
    clean = frame.where(pd.notna(frame), None)
    out = []
    for record in clean.to_dict(orient="records"):
        out.append({k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()})
    return out


def _pit(rows: Sequence[Dict[str, Any]], decision_time: int) -> List[Dict[str, Any]]:
    """Keep only rows we could already have known at ``decision_time``.

    Rows with no ``available_time`` are kept: the caller has asserted the frame
    is already gated (that is what ``source_status`` is for). Dropping them
    silently would be worse than trusting an explicit contract.
    """
    kept = []
    for row in rows:
        at = row.get("available_time")
        if at is None or int(at) <= decision_time:
            kept.append(row)
    return kept


#: Columns that identify one SERIES inside a long frame, per canonical shape.
#:
#: ``metric`` for a MetricFrame and ``timeframe`` for a BarFrame, and the
#: difference is load-bearing rather than tidy: a bar frame has no ``metric``
#: column at all, so keying on it collapses every timeframe of one instrument into
#: a single bucket. A three-timeframe capture then loses two of them to the
#: 240-row tail — and the symptom is not an error, it is "the 15m indicator is
#: always NaN", which the joining block reports as a warm-up failure pointing at
#: the capture's ``limit`` instead of at this line.
_BAR_SHAPE = "BarFrame@1.0"
_SERIES_GRAIN: Dict[str, tuple] = {
    _BAR_SHAPE: ("instrument_id", "timeframe"),
    "MetricFrame@1.0": ("instrument_id", "metric"),
}
_DEFAULT_SERIES_GRAIN = ("instrument_id", "metric")


def _tail_per_series(rows, limit: Optional[int], *,
                     grain: Sequence[str] = _DEFAULT_SERIES_GRAIN):
    """Keep only the newest *limit* rows of each series, as *grain* defines one.

    A bundle is the input at ONE decision time, so each series needs a lookback
    window, not its whole history. Without this the bars were bounded by
    ``max_bars`` while metric frames were not, and a single 1h decision dragged
    in 30 days of 5-minute open interest — 12,144 rows and 94% of a 1.7 MB file
    for data no strategy was going to read.

    See :data:`_SERIES_GRAIN` for why the key is not always
    ``(instrument_id, metric)``.
    """
    if not limit or limit <= 0:
        return list(rows)
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row.get(name) for name in grain), []).append(row)
    kept: List[Dict[str, Any]] = []
    for series in buckets.values():
        series.sort(key=lambda r: (r.get("available_time") or 0, r.get("event_time") or 0))
        kept.extend(series[-int(limit):])
    kept.sort(key=lambda r: (r.get("available_time") or 0, r.get("event_time") or 0))
    return kept


def _metric_rows(frame, *, instrument_id: str, metrics: Sequence[str],
                 time_col: str = "timestamp") -> List[Dict[str, Any]]:
    """Wide parquet (one column per metric) -> long MetricFrame rows."""
    rows: List[Dict[str, Any]] = []
    for record in _rows(frame):
        ts = record.get(time_col)
        if ts is None:
            continue
        ts = int(ts)
        for metric in metrics:
            if metric not in record or record[metric] is None:
                continue
            rows.append({
                "event_time": ts,
                # A snapshot metric is knowable at the instant it is stamped.
                "available_time": ts,
                "instrument_id": str(record.get("instrument_id") or instrument_id),
                "metric": metric,
                "value": float(record[metric]),
            })
    return rows


def _read_parquet(path: str):
    import pandas as pd

    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# build                                                                        #
# --------------------------------------------------------------------------- #


def build_input_bundle(
    *,
    symbol: str,
    interval: str,
    decision_time: Optional[int] = None,
    market_type: str = "futures",
    historical_dir: Optional[str] = None,
    storage_timeframe: str = "1m",
    derivatives_dir: Optional[str] = None,
    liquidations_dir: Optional[str] = None,
    bars: Optional[Sequence[Bar]] = None,
    news_frame: Any = None,
    ticker_rank_frame: Any = None,
    universe_frame: Any = None,
    extra_frames: Optional[Dict[str, Any]] = None,
    positions: Optional[Dict[str, float]] = None,
    equity: Optional[float] = None,
    max_bars: Optional[int] = None,
    metric_lookback: Optional[int] = 240,
    max_event_rows: Optional[int] = 200,
    internal_frames: Optional[Dict[str, Any]] = None,
    declare_internal: Sequence[str] = (),
) -> Dict[str, Any]:
    """Collect every available source at one ``decision_time`` into one dict.

    Anything that cannot be read is reported in ``source_status`` instead of
    being dropped. ``extra_frames`` is the extension point: pass
    ``{"internal_metrics": df}`` (or any key in :data:`FRAME_SHAPES`) and it is
    normalised and gated exactly like a built-in source — which is how internal
    BigData nodes join the bundle without this module importing their client.
    """
    symbol = symbol.upper()
    status: Dict[str, str] = {}
    warnings: List[str] = []
    frames: Dict[str, Dict[str, Any]] = {}

    # ---- 1. bars ---------------------------------------------------------
    if bars is None and historical_dir:
        from .historical import HistoricalParquetMarketDataAdapter
        from ..core import MarketQuery, TimeRange

        try:
            bundle = HistoricalParquetMarketDataAdapter(
                data_root=historical_dir, market_type=market_type,
                resample_source_timeframe=storage_timeframe,
            ).fetch_market(MarketQuery(instruments=[symbol], timeframes=[interval],
                                       time_range=TimeRange()))
            bars = bundle.bars.get(MarketBundle.key(symbol, interval), [])
        except Exception as exc:
            bars = []
            status["klines"] = "error: %s" % type(exc).__name__
            warnings.append("klines unavailable: %s" % exc)

    bars = list(bars or [])
    if decision_time is None:
        confirmed = [b.timestamp for b in bars if b.confirmed]
        decision_time = max(confirmed) if confirmed else int(time.time() * 1000)
    decision_time = int(decision_time)

    bar_rows = [{
        "instrument_id": b.instrument_id, "timeframe": b.timeframe,
        "open_time": int(b.extras.get("open_time") or b.timestamp),
        "close_time": int(b.extras.get("close_time") or b.timestamp),
        "open": float(b.open), "high": float(b.high), "low": float(b.low),
        "close": float(b.close), "volume": float(b.volume),
        "quote_volume": (None if b.quote_volume is None else float(b.quote_volume)),
        "confirmed": bool(b.confirmed),
        # a confirmed bar is knowable at its close
        "available_time": int(b.extras.get("close_time") or b.timestamp),
    } for b in bars if b.confirmed]
    bar_rows = _pit(bar_rows, decision_time)
    if max_bars:
        bar_rows = bar_rows[-int(max_bars):]
    if bar_rows:
        frames["klines"] = {"shape": FRAME_SHAPES["klines"], "rows": bar_rows}
        status.setdefault("klines", "ok")
    else:
        status.setdefault("klines", "empty")

    # ---- 2. derivatives ---------------------------------------------------
    if derivatives_dir:
        base = os.path.join(derivatives_dir, market_type, symbol)
        fund = _read_parquet(os.path.join(base, "funding_rate.parquet"))
        rows = _tail_per_series(
            _pit(_metric_rows(fund, instrument_id=symbol,
                              metrics=("funding_rate", "mark_price")), decision_time),
            metric_lookback)
        if rows:
            frames["funding"] = {"shape": FRAME_SHAPES["funding"], "rows": rows}
            status["funding"] = "ok"
        else:
            status["funding"] = "empty"

        oi = _read_parquet(os.path.join(base, "open_interest_%s.parquet" % interval))
        oi_tf = interval
        if oi is None:
            # The OI filename is timeframe-bound; fall back to any available one
            # rather than silently reporting "no open interest" when a 5m file is
            # sitting right there.
            import glob as _glob
            for path in sorted(_glob.glob(os.path.join(base, "open_interest_*.parquet"))):
                oi = _read_parquet(path)
                if oi is not None:
                    oi_tf = os.path.basename(path)[len("open_interest_"):-len(".parquet")]
                    warnings.append(
                        "open_interest_%s.parquet not found; used %s instead"
                        % (interval, os.path.basename(path)))
                    break
        rows = _tail_per_series(
            _pit(_metric_rows(oi, instrument_id=symbol,
                              metrics=("open_interest", "open_interest_value")),
                 decision_time),
            metric_lookback)
        if rows:
            frames["open_interest"] = {"shape": FRAME_SHAPES["open_interest"],
                                       "rows": rows, "source_timeframe": oi_tf}
            status["open_interest"] = "ok"
        else:
            status["open_interest"] = "empty"

    if liquidations_dir:
        path = os.path.join(liquidations_dir, market_type, symbol,
                            "liquidation_%s.parquet" % interval)
        liq = _read_parquet(path)
        if liq is None:
            import glob as _glob
            for cand in sorted(_glob.glob(os.path.join(
                    liquidations_dir, market_type, symbol, "liquidation_*.parquet"))):
                liq = _read_parquet(cand)
                if liq is not None:
                    break
        rows = _tail_per_series(_pit(_metric_rows(liq, instrument_id=symbol, metrics=(
            "long_liq_notional_usd", "short_liq_notional_usd",
            "total_liq_notional_usd", "net_liq_notional_usd", "liq_imbalance_ratio",
        )), decision_time), metric_lookback)
        if rows:
            frames["liquidations"] = {"shape": FRAME_SHAPES["liquidations"], "rows": rows}
            status["liquidations"] = "ok"
        else:
            status["liquidations"] = "empty"

    # ---- 3. news / universe ----------------------------------------------
    for key, frame in (("news", news_frame), ("ticker_rank", ticker_rank_frame),
                       ("universe", universe_frame)):
        if frame is None:
            continue
        rows = _pit(_rows(frame), decision_time)
        if max_event_rows and len(rows) > max_event_rows:
            rows = rows[-int(max_event_rows):]
        frames[key] = {"shape": FRAME_SHAPES[key], "rows": rows}
        status[key] = "ok" if rows else "empty"

    # ---- 4. anything else (internal BigData, custom REST, …) -------------
    for key, frame in (extra_frames or {}).items():
        shape = FRAME_SHAPES.get(key, "RawFrame@1.0")
        rows = _tail_per_series(
            _pit(_rows(frame), decision_time), metric_lookback,
            grain=_SERIES_GRAIN.get(shape, _DEFAULT_SERIES_GRAIN))
        # ``max_event_rows`` caps EVENTS — a news feed, where the newest 200 items
        # are the ones a decision reads. A bar frame is not events: its size is
        # already bounded per series by ``metric_lookback`` above, and a flat cap
        # across a multi-symbol × multi-timeframe capture would keep whole series
        # for the instruments that sort last and none for the rest. The joining
        # block would then refuse for want of warm-up, naming the wrong cause.
        if shape != _BAR_SHAPE and max_event_rows and len(rows) > max_event_rows:
            rows = rows[-int(max_event_rows):]
        frames[key] = {"shape": shape, "rows": rows}
        status[key] = "ok" if rows else "empty"

    # ---- 5. internal-domain slots -----------------------------------------
    # Declared slots always appear in source_status so a consumer can tell
    # "this deployment has no internal client" from "the node returned nothing".
    for key in sorted(set(declare_internal) | set(internal_frames or {})):
        slot = INTERNAL_SLOTS.get(key)
        if slot is None:
            warnings.append("unknown internal slot %r (see internal_slots.py)" % key)
            continue
        raw = (internal_frames or {}).get(key)
        if raw is None:
            status[key] = ("declared: no data supplied" if internal_client_available()
                           else "unavailable: internal client not installed")
            continue
        rows = _tail_per_series(
            _pit(normalize_internal_frame(key, raw, decision_time=decision_time),
                 decision_time),
            metric_lookback if slot.shape == "MetricFrame@1.0" else max_event_rows)
        frames[key] = {"shape": slot.shape, "rows": rows, "pit_safe": slot.pit_safe}
        status[key] = "ok" if rows else "empty"
        if rows and not slot.pit_safe:
            warnings.append(
                "%s is a snapshot with no point-in-time history; collect it "
                "forward, do not replay it in a walk-forward backtest" % key)

    snapshot_id = str(uuid.uuid5(
        BUNDLE_NAMESPACE, "%s|%s|%s|%d" % (symbol, interval, market_type, decision_time)))

    return {
        "schema": INPUT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "decision_time": decision_time,
        "market_type": market_type,
        "instruments": [symbol],
        "primary_timeframe": interval,
        "frames": frames,
        "source_status": status,
        "warnings": warnings,
        "positions": dict(positions or {}),
        "equity": equity,
    }


# --------------------------------------------------------------------------- #
# load                                                                         #
# --------------------------------------------------------------------------- #


def load_input_bundle(bundle: Any) -> DataSnapshot:
    """Rebuild a :class:`DataSnapshot` from a bundle dict or a path to one.

    ``klines`` becomes ``DataSnapshot.market``; ``universe`` / ``ticker_rank``
    become ``DataSnapshot.universe``; **every other frame lands in
    ``DataSnapshot.frames`` under its own key** — which is why a new data source
    needs no change here.
    """
    import pandas as pd

    if isinstance(bundle, (str, bytes, os.PathLike)):
        bundle = json.loads(open(bundle, encoding="utf-8").read())
    if bundle.get("schema") != INPUT_SCHEMA_VERSION:
        raise ValueError("not a %s bundle: schema=%r"
                         % (INPUT_SCHEMA_VERSION, bundle.get("schema")))

    decision_time = int(bundle["decision_time"])
    frames_in = bundle.get("frames") or {}
    interval = bundle.get("primary_timeframe") or ""

    market = None
    if "klines" in frames_in:
        bars: List[Bar] = []
        for row in frames_in["klines"]["rows"]:
            bars.append(Bar(
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row["volume"]),
                timestamp=int(row["close_time"]),
                instrument_id=str(row["instrument_id"]),
                timeframe=str(row.get("timeframe") or interval),
                confirmed=bool(row.get("confirmed", True)),
                quote_volume=(None if row.get("quote_volume") is None
                              else float(row["quote_volume"])),
                extras={"open_time": int(row.get("open_time") or row["close_time"]),
                        "close_time": int(row["close_time"]),
                        "available_time": int(row.get("available_time")
                                              or row["close_time"])},
            ))
        if bars:
            # One key for the whole list, taken from the FIRST bar — so a frame
            # holding more than one (instrument, timeframe) files every bar under
            # whichever pair happened to sort first, and the strategy reads a
            # series that is several series concatenated.
            #
            # The comment on FRAME_SHAPES already spells out this hazard for
            # multiple INSTRUMENTS, which is why ``universe_bars`` exists as a
            # separate key. Multiple TIMEFRAMES is the same hazard through the
            # same line and had no guard: 99 1h bars plus 99 4h bars loaded as
            # one 198-bar "1h" series whose time axis steps forward 98 times,
            # jumps back 16.5 days, then steps forward again — and every
            # indicator on the primary timeframe was computed on that, with no
            # error and no warning. Refused here rather than joined, because
            # neither the caller nor the reader can see it downstream.
            grains = {(bar.instrument_id, bar.timeframe) for bar in bars}
            if len(grains) > 1:
                instruments = sorted({g[0] for g in grains})
                timeframes = sorted({g[1] for g in grains})
                raise ValueError(
                    "frames.klines holds %d (instrument, timeframe) pairs "
                    "(instruments=%s, timeframes=%s) but it is loaded into a "
                    "MarketBundle under a single key, so every bar would end up "
                    "filed as %s — one series made of several, with a time axis "
                    "that jumps backwards where they join. Nothing downstream can "
                    "see that. Put multi-instrument or multi-timeframe bars in the "
                    "'universe_bars' frame instead: it is the same BarFrame@1.0 "
                    "shape, it keeps the (instrument_id, timeframe) grain, and it "
                    "is what universe.augment_with_indicator reads."
                    % (len(grains), instruments[:5], timeframes,
                       MarketBundle.key(*sorted(grains)[0])))
            key = MarketBundle.key(bars[0].instrument_id, bars[0].timeframe)
            market = MarketBundle(bars={key: bars})

    universe = None
    uni_rows = (frames_in.get("universe") or {}).get("rows")
    rank_rows = (frames_in.get("ticker_rank") or {}).get("rows")
    if uni_rows or rank_rows:
        universe = UniverseBundle(
            as_of=decision_time,
            universe=pd.DataFrame(uni_rows) if uni_rows else None,
            ticker_rank=pd.DataFrame(rank_rows) if rank_rows else None,
        )

    frame_tables = {
        key: pd.DataFrame(spec.get("rows") or [])
        for key, spec in frames_in.items()
        if isinstance(spec, dict)
    }
    other = {key: table for key, table in frame_tables.items()
             if key not in ("klines", "universe", "ticker_rank") and not table.empty}

    # The shape name is authoritative.  A colleague-provided custom node can
    # therefore join the input merely by choosing one of the canonical shapes;
    # no node-specific loader branch is needed here.
    shape_to_kind = {
        schema.name: kind for kind in FrameKind
        if (schema := schema_for(kind)) is not None
    }
    typed = {}
    statuses = dict(bundle.get("source_status") or {})
    for key, spec in frames_in.items():
        if not isinstance(spec, dict):
            continue
        kind = shape_to_kind.get(str(spec.get("shape") or ""))
        if kind is None:
            continue
        typed[key] = TypedFrame(
            node=key,
            kind=kind,
            frame=frame_tables.get(key),
            status=str(statuses.get(key, "ok")),
            as_of=decision_time,
            warnings=tuple(spec.get("warnings") or ()),
        )

    return DataSnapshot(
        version="mvp/v1",
        market=market,
        universe=universe,
        frames=other,
        typed=typed,
        positions={str(key).upper(): float(value)
                   for key, value in (bundle.get("positions") or {}).items()},
        equity=(None if bundle.get("equity") is None else float(bundle["equity"])),
        config=dict(bundle.get("config") or {}),
        run_id=str(bundle.get("run_id") or ""),
        trace_id=str(bundle.get("trace_id") or ""),
        meta=SnapshotMeta(
            snapshot_id=str(bundle.get("snapshot_id") or ""),
            assembled_at=decision_time,
            decision_as_of=decision_time,
            primary_timeframe=interval or None,
            source_status=statuses,
            warnings=list(bundle.get("warnings") or []),
            partial_ok=True,
            trace_id=str(bundle.get("trace_id") or "") or None,
        ),
    )


def write_input_bundle(bundle: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, separators=(",", ":"))
    return path


def read_input_bundle(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
