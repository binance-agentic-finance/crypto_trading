"""Live API -> one ``cyqnt.input/v1`` JSON, ready for a strategy to decide on.

``build_input_bundle`` assembles a bundle from files on disk. This is its live
sibling: it **calls the catalog nodes over the network** and lands every response
in the same envelope, so paper/live signal generation reads exactly the artifact
shape a replay does.

    bundle = build_live_bundle(symbol="BTCUSDT", interval="1h")
    #   {"schema": "cyqnt.input/v1",
    #    "decision_time": 1785415010565,
    #    "frames": {"klines":  {"shape": "BarFrame@1.0",    "status": "ok", "rows": [...]},
    #               "funding": {"shape": "MetricFrame@1.0", "status": "ok", "rows": [...]},
    #               "news":    {"shape": "EventFrame@1.0",  "status": "ok", "rows": [...]}},
    #    "source_status": {..., "open_interest": "error: CLI returned non-JSON"},
    #    "warnings": [...]}

Four things it does that a loop of ``requests.get`` calls does not:

**Normalises through the catalog.** Every response goes through the node's own
``normalize()``, so the metric vocabulary comes from one place. Hand-rolling the
long-form conversion is how the same field ended up named ``rate`` in one
artifact and ``funding_rate`` in another, with consumers silently reading
``None`` from whichever they did not expect.

**One clock, and it actually gates.** Every node is fetched inside one
``DataSession`` and every row is then dropped unless
``available_time <= decision_time``. Recording an ``as_of`` without filtering on
it — which is what the runtime did — means a replay reads rows it could not have
known.

**One time window, expressed in time.** Sources arrive at wildly different
cadences (1h bars, 8h funding, 5m open interest). Windowing them by *row count*
gives each a different span: 240 rows is 20 hours of 5-minute data and 80 days of
8-hourly data. So the window here is the bar span, in milliseconds, applied to
every frame.

**Reports what it could not read.** A node that fails appears in
``source_status`` with the reason and an empty ``rows``, never as an absent key —
"I did not read it" and "I read it and it was empty" stay different facts.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.input_contract import AVAILABLE_TIME, EVENT_TIME, FrameKind
from .catalog import DataUnavailable, get_node
from .input_bundle import BUNDLE_NAMESPACE, _rows

__all__ = [
    "build_live_bundle",
    "default_live_requests",
    "LiveRequest",
]

#: one request: node name, call params, and the key it is stored under. The alias
#: is what makes several instruments of the same node coexist in one bundle
#: (``klines:BTCUSDT`` / ``klines:ETHUSDT``).
LiveRequest = Tuple[str, Dict[str, Any], str]


def default_live_requests(
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    market_type: str = "futures",
    news_page_size: int = 50,
    include_account: bool = False,
) -> List[LiveRequest]:
    """Every node a single-instrument decision can currently use.

    Nodes that are wired but unreachable from this environment are *included on
    purpose*: the point of the bundle is that the strategy sees their status
    rather than silently missing them. ``include_account`` is opt-in because it
    needs credentials.
    """
    per_symbol = [
        ("klines", {"symbol": symbol, "interval": interval, "limit": limit,
                    "market_type": market_type}),
        ("ticker_24h", {"symbol": symbol, "market_type": market_type}),
        ("funding", {"symbol": symbol, "limit": 200}),
        ("open_interest", {"symbol": symbol, "period": interval, "limit": limit}),
        ("long_short_ratio", {"symbol": symbol, "period": interval, "limit": limit}),
        ("top_trader_ratio", {"symbol": symbol, "period": interval, "limit": limit}),
        ("taker_volume", {"symbol": symbol, "period": interval, "limit": limit}),
        ("basis", {"pair": symbol, "period": interval, "limit": limit}),
        ("orderbook_depth", {"symbol": symbol, "limit": 100,
                             "market_type": market_type}),
    ]
    market_wide = [
        ("news", {"page_size": news_page_size}),
        ("hot_post", {"limit": 20}),
        ("topic_trending", {"limit": 10}),
        ("ticker_rank", {"window": "24h", "limit": 20}),
        # Square keys social nodes off the BASE asset, not the pair.
        ("sentiment", {"token": _base_token(symbol)}),
        ("fear_greed", {"limit": 400}),
        ("ahr999", {"limit": 400}),
        ("universe", {"market_type": market_type}),
    ]
    account = [("contract_positions", {}), ("account_balance", {})]
    out: List[LiveRequest] = [(node, params, node)
                              for node, params in per_symbol + market_wide]
    if include_account:
        out.extend((node, params, node) for node, params in account)
    return out


def _to_epoch_ms(frame, schema):
    """Canonical time columns -> integer epoch ms.

    Normalisation leaves them as pandas timestamps, which are not
    JSON-serialisable, and ISO strings would not match the declared contract
    (``input.schema.v1.json`` types every time column as ``integer``). Fixing it
    here means the wire format has exactly one representation of a time.
    """
    import pandas as pd

    out = frame.copy()
    for column in (getattr(schema, "time_columns", ()) or ()):
        if column not in out.columns:
            continue
        parsed = pd.to_datetime(out[column], utc=True, errors="coerce")
        millis = parsed.astype("int64") // 10 ** 6
        out[column] = millis.where(parsed.notna(), other=pd.NA).astype("Int64")
    return out


def _known_at(row: Dict[str, Any]) -> Optional[int]:
    """When the row became KNOWABLE — the value the PIT gate compares."""
    for key in (AVAILABLE_TIME, EVENT_TIME, "close_time", "open_time"):
        value = row.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _happened_at(row: Dict[str, Any]) -> Optional[int]:
    """When the row's event OCCURRED — the value a time window compares.

    These must not be the same lookup. Most sources supply no publication lag,
    so normalisation fills ``available_time`` with the fetch time — one identical
    constant on every row. Windowing on that keeps everything (every row looks
    like it arrived just now), which is exactly the bug where a 66-day funding
    series sat next to a 12-day bar series and nothing trimmed it.
    """
    for key in (EVENT_TIME, "close_time", "open_time", AVAILABLE_TIME):
        value = row.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _base_token(symbol: str) -> str:
    value = str(symbol).upper()
    for quote in ("USDT", "USDC", "FDUSD", "TUSD", "BUSD", "USD"):
        if value.endswith(quote) and len(value) > len(quote):
            return value[: -len(quote)]
    return value


def build_live_bundle(
    *,
    requests: Optional[Sequence[LiveRequest]] = None,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    market_type: str = "futures",
    decision_time: Optional[int] = None,
    positions: Optional[Dict[str, float]] = None,
    equity: Optional[float] = None,
    include_account: bool = False,
    trim_to_bars: bool = False,
) -> Dict[str, Any]:
    """Fetch every requested node live and return one ``cyqnt.input/v1`` dict.

    ``decision_time`` defaults to the session clock.

    **Every row fetched is kept.** What was asked for is what the strategy gets;
    a source is not silently shortened to the bar window. Coverage is *reported*
    instead — each frame carries ``window_start`` / ``window_end`` and a warning
    when a series does not reach back to the first bar.

    ``trim_to_bars=True`` opts into discarding rows outside the bar span. It is
    off by default because the trade is bad: on a real snapshot it dropped 1,482
    rows — most of ``funding``, ``fear_greed`` and ``ahr999``, all of them
    explicitly requested via ``limit`` — to save 303 KB. The 1.7 MB bundle that
    motivated a window was caused by a *granularity* mismatch (5-minute open
    interest against hourly bars), and that is fixed at the request, not by
    throwing away what came back.
    """
    from . import catalog as _catalog  # noqa: F401  (ensure custom nodes registered)
    from ..runtime import data as data_runtime

    plan = list(requests if requests is not None else default_live_requests(
        symbol=symbol, interval=interval, limit=limit, market_type=market_type,
        include_account=include_account))

    frames: Dict[str, Dict[str, Any]] = {}
    status: Dict[str, str] = {}
    warnings: List[str] = []

    # THE decision time, and which mode we are in.
    #
    # Replay (an explicit ``decision_time``) must be gated strictly: anything
    # published after that instant is the future and has to be dropped.
    #
    # Live is different. The session stamps ``as_of`` when collection STARTS, but
    # collecting 17 nodes takes tens of seconds, and a snapshot source stamps
    # itself with the moment it was generated — i.e. *after* the start. Gating
    # live data on the start time therefore discards the freshest sources for
    # being "from the future": ``ticker_rank`` came back 22.4s after ``as_of``
    # and was silently dropped to an empty frame, which then reads as "Square
    # had nothing" rather than "we threw it away". A live decision is made when
    # collection FINISHES, so that is the cutoff.
    replaying = decision_time is not None

    with data_runtime.session(as_of_ms=decision_time) as session:
        as_of = session.as_of_ms
        cutoff = int(decision_time) if replaying else as_of
        for node, params, key in plan:
            try:
                spec = get_node(node)
            except KeyError:
                status[key] = "error: unknown node"
                warnings.append("%s: no such catalog node" % key)
                continue
            shape = spec.input_schema.name if spec.input_schema else "RawFrame@1.0"
            try:
                raw = session.call(node, **params)
            except DataUnavailable as exc:
                # Declared but unreadable: keep the slot, state the reason. A
                # strategy that needs it can abstain; one that does not is
                # unaffected. Dropping the key would make the two look alike.
                frames[key] = {"shape": shape, "status": "error", "rows": [],
                               "reason": exc.reason}
                status[key] = "error: %s" % exc.reason
                continue
            try:
                normalized, notes, inferred = spec.normalize(
                    raw, available_time=as_of, params=params)
            except Exception as exc:                     # shape mismatch, bad melt…
                frames[key] = {"shape": shape, "status": "unnormalised", "rows": [],
                               "reason": "%s: %s" % (type(exc).__name__, exc)}
                status[key] = "error: normalise failed (%s)" % type(exc).__name__
                warnings.append("%s: %s" % (key, exc))
                continue

            fallback = (getattr(raw, "attrs", {}) or {}).get("source_fallback", "")
            rows = _rows(_to_epoch_ms(normalized, spec.input_schema))
            # The point-in-time gate runs ONCE, after the loop, against the
            # final decision time — not here. Gating mid-loop against the
            # session's start time discarded every source stamped with the
            # instant it was generated, because collecting the other sixteen
            # nodes had already taken longer than that. Rows dropped in the loop
            # cannot be recovered afterwards, so nothing is dropped in the loop.
            entry: Dict[str, Any] = {
                "shape": shape,
                "status": "ok" if rows else "empty",
                "availability": spec.availability.value,
                "rows": rows,
            }
            if spec.pit_hazard:
                entry["pit_hazard"] = spec.pit_hazard
            if fallback:
                # Which source actually answered. Two runs served by different
                # transports are not comparable, so this must be visible rather
                # than inferred from row counts.
                entry["source_fallback"] = fallback
                warnings.append("%s: %s" % (key, fallback))
            if inferred:
                entry["available_time_inferred"] = True
            frames[key] = entry
            status[key] = entry["status"]
            warnings.extend("%s: %s" % (key, note) for note in notes)

    if not replaying:
        # Collection is done; this is the instant the decision is made.
        cutoff = max(cutoff, int(time.time() * 1000))
    _regate(frames, cutoff)

    warnings.extend(_window_to_bar_span(frames, cutoff=cutoff, trim=trim_to_bars))
    # Gating/trimming can empty a frame, so re-sync — but never over a failure:
    # ``source_status`` carries the reason a node could not be read, and
    # replacing it with a bare "error" throws away the only diagnosis.
    for key, entry in frames.items():
        if not status.get(key, "").startswith("error"):
            status[key] = entry["status"]

    return {
        "schema": "cyqnt.input/v1",
        "snapshot_id": str(uuid.uuid5(
            BUNDLE_NAMESPACE, "live|%s|%s|%s|%d" % (symbol, interval, market_type, cutoff))),
        "decision_time": cutoff,
        # replay: gated to the caller's instant. live: the moment collection
        # finished, which is when the decision is actually being made.
        "decision_time_basis": "replay" if replaying else "collection_complete",
        "market_type": market_type,
        "instruments": [symbol],
        "primary_timeframe": interval,
        "frames": frames,
        "source_status": status,
        "warnings": warnings,
        "positions": dict(positions or {}),
        "equity": equity,
    }


def _regate(frames: Dict[str, Dict[str, Any]], cutoff: int) -> None:
    """Re-apply the point-in-time gate with the final decision time.

    Rows are gated as they arrive so a slow source cannot smuggle in something
    published later than the decision; but in live mode the decision time is only
    known once every source has answered. Sources fetched early keep their result,
    and a source stamped after the session started is no longer mistaken for the
    future.
    """
    for entry in frames.values():
        dropped = entry.pop("rows_after_decision_time_dropped", 0)
        rows = [row for row in (entry.get("rows") or ())
                if _known_at(row) is None or _known_at(row) <= cutoff]
        if len(rows) != len(entry.get("rows") or ()):
            dropped += len(entry["rows"]) - len(rows)
        entry["rows"] = rows
        if dropped:
            entry["rows_after_decision_time_dropped"] = dropped
        if entry.get("status") in ("ok", "empty"):
            entry["status"] = "ok" if rows else "empty"


def _window_to_bar_span(frames: Dict[str, Dict[str, Any]], *, cutoff: int,
                        trim: bool = False) -> List[str]:
    """Annotate each frame with the bar window, and report coverage holes.

    Only discards rows when ``trim`` is set. Returns warnings.

    A row count is the wrong unit for a lookback: the same 240 rows is 20 hours
    of 5-minute open interest and 80 days of 8-hourly funding, so bounding by
    count leaves the frames in one bundle covering wildly different periods —
    and a metric frame shorter than the bars means most bars simply have no
    value for it, which is worse than a large file.
    """
    bar_times: List[int] = []
    for entry in frames.values():
        if entry.get("shape") != "BarFrame@1.0":
            continue
        for row in entry.get("rows") or ():
            stamp = row.get("close_time") or row.get("open_time")
            if stamp is not None:
                bar_times.append(int(stamp))
    if not bar_times:
        return []

    start = min(bar_times)
    notes: List[str] = []
    for key, entry in frames.items():
        if entry.get("shape") == "BarFrame@1.0" or not entry.get("rows"):
            continue
        rows = entry["rows"]
        if trim:
            # Trim on when the event HAPPENED: available_time is a constant for
            # most of these sources, so trimming on it would keep everything.
            kept = [row for row in rows if (_happened_at(row) or start) >= start]
            if len(kept) != len(rows):
                entry["rows"] = kept
                entry["status"] = "ok" if kept else "empty"
                entry["rows_before_window_dropped"] = len(rows) - len(kept)
        entry["window_start"] = start
        entry["window_end"] = cutoff
        # A frame that starts LATER than the bars leaves early bars with no value.
        # That is a coverage hole and has to be said out loud — silently shipping
        # one is what made 93% of bars read no open interest.
        #
        # Only for sources that are *supposed* to be a series: a FORWARD_ONLY
        # snapshot (order book, 24h ticker, social rank) has exactly one
        # timestamp by nature, so it always "starts late" and warning about it
        # would drown the case that matters. And a series is allowed to be one
        # cadence short of the first bar — 8-hourly funding legitimately has no
        # print inside the first few 1h bars — so require the hole to be large.
        if entry.get("availability") not in ("BACKTESTABLE", "SEMI"):
            continue
        covered = [t for t in (_happened_at(row) for row in entry["rows"]) if t is not None]
        if not covered or min(covered) <= start:
            continue
        uncovered = sum(1 for t in bar_times if t < min(covered))
        if uncovered > max(1, len(bar_times) // 20):
            notes.append(
                "%s starts %.1fh after the first bar — %d of %d bars (%.0f%%) have "
                "no value for it; either widen the source or narrow the bars"
                % (key, (min(covered) - start) / 3.6e6, uncovered, len(bar_times),
                   100.0 * uncovered / len(bar_times)))
    return notes
