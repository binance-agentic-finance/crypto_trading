"""``to_panel`` — a bundle, flattened onto the bar clock, ready for ``blocks``.

Why this exists
---------------
Every function in ``cyqnt_trd.blocks`` takes a ``pd.Series`` (or an OHLCV frame),
and the divergence-style ones take **two Series on the same index**::

    derivatives.oi_price_divergence(price: pd.Series, oi: pd.Series, lookback=20)
    derivatives.funding_rate_state(funding: pd.Series, ...)

A ``BarFrame`` satisfies that directly — ``bars["close"]`` *is* a Series. Nothing
else in the bundle does. ``MetricFrame`` is long form (one row per
instrument × metric × time), so handing it over goes wrong in two ways:

* ``oi_change_pct(frame)`` raises ``TypeError: cannot convert DataFrame to Series``;
* ``oi_change_pct(frame["value"])`` does **not** raise, and is silently wrong —
  that column interleaves every metric of the node, so the result is a rate of
  change computed across ``oi_base`` and ``oi_value`` alternately.

The second is the dangerous one, and no amount of care at the call site prevents
it. So the conversion belongs here, once.

What it produces
----------------
One wide frame, indexed by bar close time, one column per field::

    panel = to_panel(bundle)

    panel["close"]                                      # OHLCV, as blocks expect
    derivatives.oi_price_divergence(panel["close"], panel["oi_value"])
    derivatives.funding_rate_state(panel["rate"])
    panel["news_count_24h"]                             # events aggregated per bar

Alignment is **as-of on ``available_time``**, never on ``event_time``: a bar may
only see values that were knowable by the time it closed. A funding print stamped
inside a bar but published after it closes must not appear on that bar, and
choosing the wrong column is exactly how a backtest reads the future.

Multi-instrument
----------------
``to_panel(bundle, symbols=[...])`` returns a ``(symbol, field)`` column
MultiIndex over the same time axis — the shape a cross-sectional / news-selection
strategy ranks each bar. ``panel["BTCUSDT"]`` slices back to the single-symbol
frame blocks consume.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = ["to_panel", "PanelError"]

#: shapes that carry a per-bar numeric series
_METRIC_SHAPE = "MetricFrame@1.0"
_BAR_SHAPE = "BarFrame@1.0"
_EVENT_SHAPE = "EventFrame@1.0"
_RANK_SHAPE = "RankFrame@1.0"

#: OHLCV columns taken from the bar frame, in this order
_BAR_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume")

#: default windows for counting events into each bar
_EVENT_WINDOWS = {"1h": 3_600_000, "24h": 86_400_000}


class PanelError(ValueError):
    """The bundle cannot be flattened onto a bar clock."""


def _frames(bundle: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(bundle, dict) and "frames" in bundle:
        return bundle["frames"] or {}
    raise PanelError("expected a cyqnt.input/v1 bundle dict with a 'frames' key")


def _rows_of(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(entry.get("rows") or ())


def _bar_rows(frames: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in frames.values():
        if entry.get("shape") == _BAR_SHAPE:
            out.extend(_rows_of(entry))
    return out


def _instruments(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({str(row.get("instrument_id") or "").upper()
                   for row in rows if row.get("instrument_id")})


def to_panel(
    bundle: Any,
    *,
    symbol: Optional[str] = None,
    symbols: Optional[Iterable[str]] = None,
    event_windows: Optional[Dict[str, int]] = None,
    include_rank: bool = True,
):
    """Flatten a bundle into a bar-indexed wide frame (see the module docstring).

    ``symbol`` selects one instrument (the default when the bundle holds exactly
    one). ``symbols`` returns a ``(symbol, field)`` column MultiIndex.
    """
    import pandas as pd

    frames = _frames(bundle)
    bars = _bar_rows(frames)
    if not bars:
        raise PanelError(
            "the bundle has no BarFrame, so there is no bar clock to align to. "
            "Request a klines node, or align against your own time axis.")

    available = _instruments(bars)
    if symbols is not None:
        wanted = [str(s).upper() for s in symbols]
        missing = [s for s in wanted if s not in available]
        if missing:
            raise PanelError("no bars for %s (bundle has %s)" % (missing, available))
        parts = {s: to_panel(bundle, symbol=s, event_windows=event_windows,
                             include_rank=include_rank) for s in wanted}
        return pd.concat(parts, axis=1)

    if symbol is None:
        if len(available) > 1:
            raise PanelError(
                "the bundle holds bars for %s — pass symbol= to pick one, or "
                "symbols= for a cross-sectional panel" % available)
        symbol = available[0] if available else ""
    symbol = str(symbol).upper()

    # ---- time axis: confirmed bar closes for this instrument -----------------
    frame = pd.DataFrame([row for row in bars
                          if str(row.get("instrument_id") or "").upper() == symbol])
    if frame.empty:
        raise PanelError("no bars for %s (bundle has %s)" % (symbol, available))
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.sort_values("close_time").drop_duplicates("close_time", keep="last")
    axis = pd.DatetimeIndex(frame["close_time"], name="close_time")

    cutoffs = _cutoffs(axis, bundle.get("decision_time"))
    panel = pd.DataFrame(index=axis)
    for field in _BAR_FIELDS:
        if field in frame.columns:
            panel[field] = pd.to_numeric(frame[field].values, errors="coerce")

    # ---- metric frames: pivot to one column per metric, as-of onto the axis --
    for key, entry in frames.items():
        if entry.get("shape") != _METRIC_SHAPE:
            continue
        rows = _rows_of(entry)
        if not rows:
            continue
        long = pd.DataFrame(rows)
        if "metric" not in long.columns or "value" not in long.columns:
            continue
        if "instrument_id" in long.columns:
            owner = long["instrument_id"].astype(str).str.upper()
            # market-wide readings (fear & greed, macro) carry no instrument or a
            # sentinel; keep them for every symbol rather than dropping them.
            long = long[_owns(long["instrument_id"], symbol)
                        | owner.isin({"", "MARKET", "NONE"}) | owner.isna()]
        if long.empty:
            continue
        units = _declared_units(long)
        for name, series in _metric_columns(long, node=key).items():
            column = _unique(panel, name, node=key)
            panel[column] = _asof(series, axis, cutoffs)
            if name in units:
                # The unit is declared by the catalog and rides every MetricFrame
                # row, then dies here: pivoting metric -> column keeps the value
                # and drops everything beside it. A bare ``rate`` column says
                # nothing about whether it is a ratio or bps, and that ambiguity
                # is what makes a double conversion invisible. So the units come
                # with the panel.
                panel.attrs.setdefault("units", {})[column] = units[name]

    # ---- event frames: count per window, ending at each bar ------------------
    windows = dict(event_windows if event_windows is not None else _EVENT_WINDOWS)
    for key, entry in frames.items():
        if entry.get("shape") != _EVENT_SHAPE:
            continue
        rows = _rows_of(entry)
        if not rows:
            continue
        events = pd.DataFrame(rows)
        stamps = _known_at_series(events)
        if stamps is None or stamps.empty:
            continue
        if "instrument_id" in events.columns:
            owner = events["instrument_id"].astype(str).str.upper()
            # An event with no instrument is market-wide news, which is exactly
            # what a catalyst strategy reads — it must not be filtered away.
            keep = (_owns(events["instrument_id"], symbol)
                    | owner.isin({"", "NONE"}) | owner.isna())
            stamps = stamps[keep.values]
        for label, span in windows.items():
            panel["%s_count_%s" % (key, label)] = _rolling_count(stamps, axis, span, cutoffs)

    # ---- rank frames: this instrument's own score, as-of --------------------
    if include_rank:
        for key, entry in frames.items():
            if entry.get("shape") != _RANK_SHAPE:
                continue
            rows = _rows_of(entry)
            if not rows:
                continue
            rank = pd.DataFrame(rows)
            if "instrument_id" not in rank.columns:
                continue
            mine = rank[_owns(rank["instrument_id"], symbol)]
            if mine.empty:
                continue
            stamps = _known_at_series(mine)
            for field in ("rank", "score"):
                if field not in mine.columns:
                    continue
                series = pd.Series(
                    pd.to_numeric(mine[field], errors="coerce").values,
                    index=pd.DatetimeIndex(stamps)).dropna()
                if series.empty:
                    continue
                panel[_unique(panel, "%s_%s" % (key, field), node=key)] = _asof(series, axis, cutoffs)

    panel.attrs["symbol"] = symbol
    panel.attrs["decision_time"] = bundle.get("decision_time")
    panel.attrs["source_status"] = dict(bundle.get("source_status") or {})
    return panel


def _base_token(symbol: str) -> str:
    value = str(symbol).upper()
    for quote in ("USDT", "USDC", "FDUSD", "TUSD", "BUSD", "USD"):
        if value.endswith(quote) and len(value) > len(quote):
            return value[: -len(quote)]
    return value


def _owns(series, symbol: str):
    """Rows belonging to this instrument, matching the base asset too.

    Square keys its social frames on the BASE token (``BTC``), while the bar
    clock is a pair (``BTCUSDT``). Comparing the two directly drops every social
    reading for the symbol it was fetched for — the panel simply had no
    ticker_rank column and nothing said why.
    """
    upper = series.astype(str).str.upper()
    wanted = {symbol, _base_token(symbol)}
    return upper.isin(wanted)


def _declared_units(long) -> Dict[str, str]:
    """{metric name: unit} for the metrics in a long-form frame, when declared."""
    if "unit" not in long.columns or "metric" not in long.columns:
        return {}
    out: Dict[str, str] = {}
    for name, group in long.groupby(long["metric"].astype(str), sort=False):
        declared = group["unit"].dropna().astype(str)
        if not declared.empty:
            out[str(name)] = declared.iloc[0]
    return out


def _metric_columns(long, *, node: str) -> Dict[str, Any]:
    """Long-form metric rows -> {column name: Series indexed by available_time}."""
    import pandas as pd

    stamps = _known_at_series(long)
    if stamps is None:
        return {}
    work = long.copy()
    # Assign the Series, not ``.values``: a numpy datetime64 array drops the
    # timezone, and a tz-naive key cannot be merged against the tz-aware bar axis.
    work["__at__"] = stamps
    out: Dict[str, Any] = {}
    for name, group in work.groupby(work["metric"].astype(str), sort=False):
        series = pd.Series(
            pd.to_numeric(group["value"], errors="coerce").values,
            index=pd.DatetimeIndex(group["__at__"]),
        ).dropna()
        if series.empty:
            continue
        out[str(name)] = series.sort_index()
    return out


def _known_at_series(frame):
    """``available_time`` as UTC timestamps — the only lookahead-safe key.

    ``event_time`` says when the thing happened, which may be *before* it was
    publishable. Aligning on it puts a value on a bar that could not have seen
    it, so it is used only as a fallback for sources that supply nothing else.
    """
    import pandas as pd

    for column in ("available_time", "event_time", "close_time"):
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], unit="ms", utc=True, errors="coerce")
            if parsed.notna().any():
                return parsed
    return None


def _utc(values):
    import pandas as pd

    stamps = pd.DatetimeIndex(values)
    return stamps.tz_localize("UTC") if stamps.tz is None else stamps.tz_convert("UTC")


def _cutoffs(axis, decision_time):
    """Per-bar as-of deadlines.

    Every bar but the last is cut at its own close, which is what keeps a replay
    honest. The **final** bar is cut at ``decision_time`` instead, because that
    row is the decision the bot is making now: a live bot deciding after the bar
    closed genuinely knows the current 24h ticker, order book and social ranking,
    all of which are stamped after the close. Cutting the last row at the close
    too would hide every snapshot source behind a NaN — correct-looking, and
    useless for the one row that matters.
    """
    import pandas as pd

    stamps = _utc(axis)
    if decision_time is None or len(stamps) == 0:
        return stamps
    deadline = pd.Timestamp(int(decision_time), unit="ms", tz="UTC")
    if deadline <= stamps[-1]:
        return stamps
    return stamps[:-1].append(pd.DatetimeIndex([deadline]))


def _asof(series, axis, cutoffs=None):
    """Latest value at or before each bar's deadline.

    Bars before the first reading stay NaN rather than being back-filled — a
    strategy must be able to see that it had no reading yet.
    """
    import pandas as pd

    left = pd.DataFrame({"__t__": _utc(axis if cutoffs is None else cutoffs)})
    right = pd.DataFrame({"__t__": _utc(series.index),
                          "__v__": series.values}).sort_values("__t__")
    merged = pd.merge_asof(left, right, on="__t__", direction="backward")
    return pd.Series(merged["__v__"].values, index=axis)


def _rolling_count(stamps, axis, span_ms: int, cutoffs=None):
    """How many events fell in (deadline - span, deadline] for each bar."""
    import numpy as np
    import pandas as pd

    ordered = np.sort(_utc(pd.DatetimeIndex(stamps)).asi8)
    ends = _utc(axis if cutoffs is None else cutoffs).asi8
    starts = ends - int(span_ms) * 1_000_000          # ms -> ns
    upper = np.searchsorted(ordered, ends, side="right")
    lower = np.searchsorted(ordered, starts, side="right")
    return pd.Series(upper - lower, index=axis)


def _unique(panel, name: str, *, node: str = "") -> str:
    """Keep column names distinct, and say WHICH source the duplicate came from.

    ``long_short_ratio`` and ``top_trader_ratio`` both emit a metric called
    ``long_short_ratio`` — they measure different populations (all accounts vs
    the top traders). Numbering the second one ``long_short_ratio_2`` keeps them
    apart but tells the reader nothing, and picking one silently would be worse.
    """
    if name not in panel.columns:
        return name
    if node:
        qualified = "%s.%s" % (node, name)
        if qualified not in panel.columns:
            return qualified
    index = 2
    while "%s_%d" % (name, index) in panel.columns:
        index += 1
    return "%s_%d" % (name, index)

# --------------------------------------------------------------------------- #
# the same alignment, for a DataSnapshot instead of a bundle                   #
# --------------------------------------------------------------------------- #


def attach_frames_to_bars(bars, snapshot):
    """Merge ``DataSnapshot.frames`` onto an existing bar DataFrame, as-of.

    ``to_panel`` builds a panel from a bundle; this does the same alignment for
    the object the block plugins actually receive. Without it a bundle could
    carry funding, open interest, taker flow and news while ``make_signals(df)``
    saw only OHLCV — the multi-source input existed and never reached the code
    that was written to read it.

    Alignment is on ``available_time``, not ``event_time``: a bar sees a reading
    only if it was knowable by the time the bar closed. Existing columns are
    never overwritten.
    """
    import pandas as pd

    frames = getattr(snapshot, "frames", None) or {}
    if bars is None or not len(bars) or not frames:
        return bars

    time_column = next((c for c in ("close_time", "timestamp", "open_time")
                        if c in bars.columns), None)
    if time_column is None:
        return bars
    axis = pd.to_datetime(bars[time_column], unit="ms", utc=True, errors="coerce")
    if axis.isna().all():
        axis = pd.to_datetime(bars[time_column], utc=True, errors="coerce")
    if axis.isna().all():
        return bars

    instrument = ""
    if "instrument_id" in bars.columns and len(bars):
        instrument = str(bars["instrument_id"].iloc[0]).upper()

    out = bars.copy()
    index = pd.DatetimeIndex(axis)
    for key, frame in frames.items():
        rows = frame
        if rows is None or not len(rows):
            continue
        long = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        if "metric" not in long.columns or "value" not in long.columns:
            continue                       # only long-form metrics fold into bars
        if "instrument_id" in long.columns and instrument:
            owner = long["instrument_id"].astype(str).str.upper()
            long = long[_owns(long["instrument_id"], instrument)
                        | owner.isin({"", "MARKET", "NONE"}) | owner.isna()]
        if long.empty:
            continue
        units = _declared_units(long)
        for name, series in _metric_columns(long, node=key).items():
            column = name if name not in out.columns else "%s.%s" % (key, name)
            if column in out.columns:
                continue                   # never overwrite what the bars carry
            out[column] = _asof(series, index).values
            if name in units:
                out.attrs.setdefault("units", {})[column] = units[name]
    return out
