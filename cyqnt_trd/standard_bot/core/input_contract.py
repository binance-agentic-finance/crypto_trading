"""``cyqnt.input/v1`` — the typed input contract.

The output side got a full contract: grouped fields, enums, validation at
construction. The input side did not. ``BotContext.frames`` is
``Dict[str, Any]`` — raw, vendor-shaped DataFrames. ``funding`` gives you
``rate``/``timestamp``, ``open_interest`` gives you ``oi_value``/``timestamp``,
``klines`` gives you ``close``/``close_time``, ``ticker_rank`` gives you
``ticker``/``mention_count``. A bot reading three sources has to know three
column vocabularies, and nothing checks that what arrived is what was promised.

This module gives the input the same treatment.

Seven canonical shapes
----------------------
Every data node emits one of these, and only these:

``BarFrame@1.0``       OHLCV, one row per instrument × timeframe × bar
``MetricFrame@1.0``    long form, one row per instrument × metric × time
``PanelFrame@1.0``     wide numeric panel, time × instrument (cross-section input)
``EventFrame@1.0``     discrete events — news, announcements, calendar, unlocks
``RankFrame@1.0``      cross-sectional snapshot ranking
``PositionFrame@1.0``  account state
``BookFrame@1.0``      order-book ladder

Two columns every shape carries
-------------------------------
``event_time``      when the thing happened
``available_time``  when *we* could first have known it

Keeping both is what makes a PIT gate possible at all. A frame with only one
timestamp cannot distinguish "printed at 09:00" from "published at 09:00 and
readable at 09:07", and every backtest built on it is optimistic by the
publication lag. Normalisation therefore requires ``available_time`` and
derives it from the fetch time when the source does not supply one — recording
that it did so, rather than pretending the source was instantaneous.

Normalisation is declarative
----------------------------
A node declares ``emits`` plus a ``column_map`` from its vendor names to the
canonical ones; :func:`normalize_frame` does the rename, adds declared
constants, melts wide→long for metric frames, and validates the result. Nothing
is guessed: a node that has not declared a shape stays raw and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "INPUT_SCHEMA_VERSION",
    "FrameKind",
    "FrameSchema",
    "FrameValidationError",
    "BAR_FRAME",
    "METRIC_FRAME",
    "PANEL_FRAME",
    "EVENT_FRAME",
    "RANK_FRAME",
    "POSITION_FRAME",
    "BOOK_FRAME",
    "SCHEMAS",
    "schema_for",
    "normalize_frame",
    "TypedFrame",
]

INPUT_SCHEMA_VERSION = "cyqnt.input/v1"

#: canonical time columns, present on every shape that has rows over time
EVENT_TIME = "event_time"
AVAILABLE_TIME = "available_time"

#: canonical instrument column. One name everywhere: ``symbol`` / ``ticker`` /
#: ``coin`` / ``cardCoin`` all normalise to this.
INSTRUMENT = "instrument_id"


class FrameValidationError(ValueError):
    """A frame does not match the shape its node declared."""


class FrameKind(str, Enum):
    BAR = "bar"
    METRIC = "metric"
    PANEL = "panel"
    EVENT = "event"
    RANK = "rank"
    POSITION = "position"
    BOOK = "book"
    #: the node has not declared a canonical shape yet — reachable, but the
    #: caller gets the raw frame and no guarantees
    RAW = "raw"


@dataclass(frozen=True)
class FrameSchema:
    """One canonical input shape."""

    kind: FrameKind
    name: str
    #: columns that must be present after normalisation
    required: Tuple[str, ...]
    #: columns that are meaningful when present
    optional: Tuple[str, ...] = ()
    #: columns parsed to UTC timestamps (epoch-ms ints are read as ms)
    time_columns: Tuple[str, ...] = ()
    #: columns coerced to float
    numeric_columns: Tuple[str, ...] = ()
    #: column that bounds when a row became knowable, used to fill
    #: ``available_time`` per row when the source supplies none.
    #:
    #: Filling it with the *fetch* time instead — one identical constant on every
    #: row — is not a small inaccuracy: it claims a 300-bar history all became
    #: knowable at once, which defeats the PIT gate (every row passes) and any
    #: time window (every row looks current). For bars the honest bound is exact:
    #: a candle is knowable when it closes.
    available_from: str = ""
    #: what one row means — stated so a reader never has to infer it
    row_grain: str = ""
    description: str = ""

    @property
    def id(self) -> str:
        return self.name

    def validate(self, frame: Any, *, node: str = "") -> None:
        """Check columns, time ordering and PIT sanity. Empty frames pass."""
        import pandas as pd

        label = node or self.name
        if not isinstance(frame, pd.DataFrame):
            raise FrameValidationError(
                "%s must be a pandas DataFrame, got %s" % (label, type(frame).__name__)
            )
        missing = [column for column in self.required if column not in frame.columns]
        if missing:
            raise FrameValidationError(
                "%s (%s) missing required column(s): %s"
                % (label, self.name, ", ".join(missing))
            )
        if frame.empty:
            return
        if EVENT_TIME in frame.columns and AVAILABLE_TIME in frame.columns:
            event = pd.to_datetime(frame[EVENT_TIME], utc=True, errors="coerce")
            available = pd.to_datetime(frame[AVAILABLE_TIME], utc=True, errors="coerce")
            both = event.notna() & available.notna()
            if bool((event[both] > available[both]).any()):
                raise FrameValidationError(
                    "%s has rows whose event_time is AFTER available_time — the "
                    "data claims to have been readable before it happened" % label
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "schema": self.name,
            "row_grain": self.row_grain,
            "description": self.description,
            "required": list(self.required),
            "optional": list(self.optional),
            "time_columns": list(self.time_columns),
            "numeric_columns": list(self.numeric_columns),
        }


# ---------------------------------------------------------------------------
# the seven shapes
# ---------------------------------------------------------------------------

BAR_FRAME = FrameSchema(
    kind=FrameKind.BAR,
    name="BarFrame@1.0",
    row_grain="one instrument × timeframe × bar",
    description="OHLCV candles. The base series every technical block reads.",
    required=(INSTRUMENT, "timeframe", "open_time", "close_time",
              "open", "high", "low", "close", "volume", AVAILABLE_TIME),
    optional=("quote_volume", "trades", "confirmed", EVENT_TIME),
    available_from="close_time",
    time_columns=("open_time", "close_time", AVAILABLE_TIME, EVENT_TIME),
    numeric_columns=("open", "high", "low", "close", "volume", "quote_volume"),
)

METRIC_FRAME = FrameSchema(
    kind=FrameKind.METRIC,
    name="MetricFrame@1.0",
    row_grain="one instrument × metric × time",
    description=(
        "Long-form numeric observations. One row per measurement, so a source "
        "that fails contributes zero rows instead of a zero value."
    ),
    required=(EVENT_TIME, AVAILABLE_TIME, INSTRUMENT, "metric", "value"),
    optional=("venue", "product", "unit", "window", "source_id", "quality"),
    time_columns=(EVENT_TIME, AVAILABLE_TIME),
    numeric_columns=("value",),
)

PANEL_FRAME = FrameSchema(
    kind=FrameKind.PANEL,
    name="PanelFrame@1.0",
    row_grain="one timestamp; one column per instrument",
    description=(
        "Wide time × instrument numeric panel — the natural input for a "
        "cross-sectional strategy that ranks a universe each bar."
    ),
    required=(EVENT_TIME, AVAILABLE_TIME),
    optional=("metric",),
    time_columns=(EVENT_TIME, AVAILABLE_TIME),
)

EVENT_FRAME = FrameSchema(
    kind=FrameKind.EVENT,
    name="EventFrame@1.0",
    row_grain="one event × instrument (instrument may be null for market-wide)",
    description=(
        "Discrete dated events: news, announcements, calendar entries, unlocks, "
        "macro releases. Carries provenance so a directional read can be "
        "refused when the source is unverified."
    ),
    required=("event_id", EVENT_TIME, AVAILABLE_TIME, "source_id", "topic"),
    optional=(INSTRUMENT, "urgency", "title", "summary", "url", "bias",
              "event_type", "source_reliability", "corroboration_count",
              "quality_flags"),
    time_columns=(EVENT_TIME, AVAILABLE_TIME),
    numeric_columns=("source_reliability", "corroboration_count"),
)

RANK_FRAME = FrameSchema(
    kind=FrameKind.RANK,
    name="RankFrame@1.0",
    row_grain="one instrument at one as-of",
    description=(
        "Cross-sectional snapshot: a scored/ranked instrument list. Feature "
        "columns beyond the required ones are passed through untouched."
    ),
    required=(AVAILABLE_TIME, INSTRUMENT),
    optional=(EVENT_TIME, "rank", "score", "venue", "product", "source_id"),
    time_columns=(EVENT_TIME, AVAILABLE_TIME),
    numeric_columns=("rank", "score"),
)

POSITION_FRAME = FrameSchema(
    kind=FrameKind.POSITION,
    name="PositionFrame@1.0",
    row_grain="one open position",
    description=(
        "Account state. This is what the position lifecycle reads to know "
        "whether an exit means CLOSE_LONG or CLOSE_SHORT."
    ),
    required=(AVAILABLE_TIME, INSTRUMENT, "side", "quantity"),
    optional=("venue", "product", "entry_price", "leverage", "margin_ratio",
              "unrealized_pnl", "notional"),
    time_columns=(AVAILABLE_TIME, EVENT_TIME),
    numeric_columns=("quantity", "entry_price", "leverage", "margin_ratio",
                     "unrealized_pnl", "notional"),
)

BOOK_FRAME = FrameSchema(
    kind=FrameKind.BOOK,
    name="BookFrame@1.0",
    row_grain="one price level on one side",
    description="Order-book ladder snapshot.",
    required=(AVAILABLE_TIME, INSTRUMENT, "side", "level", "price", "quantity"),
    optional=(EVENT_TIME, "venue", "product"),
    time_columns=(AVAILABLE_TIME, EVENT_TIME),
    numeric_columns=("level", "price", "quantity"),
)

SCHEMAS: Dict[FrameKind, FrameSchema] = {
    schema.kind: schema
    for schema in (BAR_FRAME, METRIC_FRAME, PANEL_FRAME, EVENT_FRAME,
                   RANK_FRAME, POSITION_FRAME, BOOK_FRAME)
}


def schema_for(kind: FrameKind) -> Optional[FrameSchema]:
    return SCHEMAS.get(kind)


# ---------------------------------------------------------------------------
# typed handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedFrame:
    """A normalised frame plus the metadata a reader needs to trust it."""

    node: str
    kind: FrameKind
    frame: Any
    #: "ok" | "degraded" | "error", carried from the fetch
    status: str = "ok"
    #: decision time this frame was gated to
    as_of: int = 0
    availability: str = ""
    #: how this field lies if replayed naively; empty when it replays honestly
    pit_hazard: str = ""
    warnings: Tuple[str, ...] = ()
    #: True when available_time was inferred from fetch time, not supplied
    available_time_inferred: bool = False

    @property
    def schema(self) -> Optional[FrameSchema]:
        return schema_for(self.kind)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def empty(self) -> bool:
        return self.frame is None or bool(getattr(self.frame, "empty", True))

    def __len__(self) -> int:
        return 0 if self.frame is None else int(len(self.frame))

    # ---- convenience readers ----

    def latest(self, column: str, *, instrument: Optional[str] = None) -> Optional[float]:
        """Most recent value of ``column``, optionally for one instrument."""
        import pandas as pd

        if self.empty or column not in self.frame.columns:
            return None
        frame = self.frame
        if instrument is not None and INSTRUMENT in frame.columns:
            frame = frame[frame[INSTRUMENT].astype(str).str.upper() == instrument.upper()]
        if frame.empty:
            return None
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(series.iloc[-1]) if not series.empty else None

    def metric(self, name: str, *, instrument: Optional[str] = None) -> Optional[float]:
        """Latest value of one metric — MetricFrame only."""
        if self.kind is not FrameKind.METRIC or self.empty:
            return None
        frame = self.frame[self.frame["metric"].astype(str) == name]
        if instrument is not None and INSTRUMENT in frame.columns:
            frame = frame[frame[INSTRUMENT].astype(str).str.upper() == instrument.upper()]
        if frame.empty:
            return None
        import pandas as pd

        series = pd.to_numeric(frame["value"], errors="coerce").dropna()
        return float(series.iloc[-1]) if not series.empty else None

    def series(self, column: str, *, instrument: Optional[str] = None):
        """Numeric column as a time-indexed Series."""
        import pandas as pd

        if self.empty or column not in self.frame.columns:
            return pd.Series(dtype=float)
        frame = self.frame
        if instrument is not None and INSTRUMENT in frame.columns:
            frame = frame[frame[INSTRUMENT].astype(str).str.upper() == instrument.upper()]
        values = pd.to_numeric(frame[column], errors="coerce")
        time_column = EVENT_TIME if EVENT_TIME in frame.columns else None
        if time_column:
            return pd.Series(values.values, index=frame[time_column].values).dropna()
        return values.dropna().reset_index(drop=True)

    def instruments(self) -> List[str]:
        if self.empty or INSTRUMENT not in self.frame.columns:
            return []
        return sorted(
            {str(value).upper() for value in self.frame[INSTRUMENT].dropna().unique()}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "kind": self.kind.value,
            "schema": self.schema.name if self.schema else None,
            "rows": len(self),
            "status": self.status,
            "as_of": self.as_of,
            "availability": self.availability,
            "pit_hazard": self.pit_hazard,
            "available_time_inferred": self.available_time_inferred,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def _parse_time(values):
    """Parse a time column, reading bare integers as epoch MILLISECONDS.

    ``pd.to_datetime`` treats a bare int64 as nanoseconds, which turns every
    epoch-ms stamp into 1970 and silently disables any PIT comparison.
    """
    import pandas as pd

    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    # Already a datetime? Then there is nothing to infer — only a timezone to
    # normalise. Falling through to the numeric branch would be silent data
    # loss: ``pd.to_numeric`` renders datetime64 as NANOSECONDS (~1.8e18), the
    # magnitude test below reads that as milliseconds, and every value lands in
    # year 58000 — i.e. NaT under errors="coerce". Nothing downstream notices,
    # because the PIT check only compares timestamps that are both non-null.
    if pd.api.types.is_datetime64_any_dtype(values):
        if getattr(values.dtype, "tz", None) is not None:
            return values.dt.tz_convert("UTC")
        return values.dt.tz_localize("UTC")
    try:
        numeric = pd.to_numeric(values, errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(values, utc=True, errors="coerce")
    if numeric.notna().all() and len(numeric):
        unit = "s" if float(numeric.abs().max()) < 1e11 else "ms"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def normalize_frame(
    frame: Any,
    *,
    kind: FrameKind,
    node: str = "",
    column_map: Optional[Mapping[str, str]] = None,
    constants: Optional[Mapping[str, Any]] = None,
    value_columns: Sequence[str] = (),
    available_time: Optional[int] = None,
    validate: bool = True,
):
    """Rename, enrich and validate a vendor frame into a canonical shape.

    Returns ``(frame, warnings, available_time_inferred)``.

    Steps, in order:

    1. rename vendor columns via ``column_map``;
    2. add ``constants`` (``timeframe``, ``venue``, ``product``, a fixed
       ``metric`` name for a single-metric source, …);
    3. for ``METRIC``, melt the declared ``value_columns`` into
       ``metric`` / ``value`` rows;
    4. parse every declared time column, epoch-ms aware;
    5. fill ``available_time`` from the fetch time when absent — and say so,
       because "we assumed no publication lag" is a claim, not a default;
    6. coerce numerics and validate against the schema.
    """
    import pandas as pd

    warnings: List[str] = []
    inferred = False
    schema = schema_for(kind)

    if frame is None:
        return None, ["%s: no frame" % (node or kind.value)], False
    if not isinstance(frame, pd.DataFrame):
        raise FrameValidationError(
            "%s: expected a DataFrame, got %s" % (node or kind.value, type(frame).__name__)
        )
    if kind is FrameKind.RAW or schema is None:
        return frame, warnings, False

    out = frame.copy()

    # 1. rename
    if column_map:
        present = {src: dst for src, dst in column_map.items() if src in out.columns}
        # A rename onto a column that already exists would produce two columns
        # with the same name, and every later ``out[col]`` returns a DataFrame
        # instead of a Series. The canonical column already there wins; the
        # vendor alias is dropped.
        collisions = [
            src for src, dst in present.items() if dst in out.columns and src != dst
        ]
        if collisions:
            out = out.drop(columns=collisions)
            present = {src: dst for src, dst in present.items() if src not in collisions}
            warnings.append(
                "%s: ignored alias(es) %s — the canonical column was already present"
                % (node or kind.value, ", ".join(sorted(collisions)))
            )
        out = out.rename(columns=present)
        absent = [src for src in column_map if src not in frame.columns]
        if absent:
            warnings.append(
                "%s: declared source column(s) absent from the response: %s"
                % (node or kind.value, ", ".join(sorted(absent)))
            )

    # 2. constants
    for key, value in (constants or {}).items():
        if key not in out.columns:
            out[key] = value

    # 3. wide -> long for metric frames
    #
    # Skipped when ``metric`` is already present: a single-metric source (a node
    # that pins ``constants={"metric": ...}``, or a response that is already long
    # form) is not wide, and melting it would both lose the pinned name and
    # collide with the existing ``value`` column.
    if kind is FrameKind.METRIC and "metric" not in out.columns:
        melt_cols = [column for column in value_columns if column in out.columns]
        # A response that already carries a bare ``value`` column must be melted
        # along with the rest — left as an id_var it collides with the melt's own
        # output name and pandas refuses rather than silently picking one.
        if "value" in out.columns and "value" not in melt_cols:
            melt_cols.append("value")
        if not melt_cols:
            # No declared value columns: melt every remaining numeric column.
            # Guessing which one matters would be worse than exposing all of
            # them under their own names.
            reserved = set(schema.required) | set(schema.optional)
            melt_cols = [
                column for column in out.columns
                if column not in reserved
                and pd.to_numeric(out[column], errors="coerce").notna().any()
            ]
        if melt_cols:
            id_cols = [column for column in out.columns if column not in melt_cols]
            out = out.melt(
                id_vars=id_cols, value_vars=melt_cols,
                var_name="metric", value_name="__value__",
            ).rename(columns={"__value__": "value"})
            out = out[out["value"].notna()].reset_index(drop=True)

    # 4. times
    for column in schema.time_columns:
        if column in out.columns:
            out[column] = _parse_time(out[column])

    # 5. available_time — per row wherever the shape names a bound for it
    if AVAILABLE_TIME in schema.required and AVAILABLE_TIME not in out.columns:
        derived_from = ""
        for candidate in (schema.available_from, EVENT_TIME):
            if candidate and candidate in out.columns and not out.empty:
                derived_from = candidate
                break
        if derived_from:
            out[AVAILABLE_TIME] = out[derived_from]
            if derived_from != schema.available_from:
                # event_time as the bound asserts the reading was publishable the
                # instant it happened. That is an assumption, so it is recorded;
                # it is still far better than the fetch time, which would claim
                # the whole history arrived at once.
                inferred = True
                warnings.append(
                    "%s: available_time derived per row from %s — the source "
                    "states no publication lag, so a replay may be optimistic by "
                    "that lag" % (node or kind.value, derived_from)
                )
        elif available_time is None:
            raise FrameValidationError(
                "%s: %s requires available_time and the source supplies none; "
                "pass the fetch time so the gap is recorded rather than assumed"
                % (node or kind.value, schema.name)
            )
        else:
            # A true snapshot (order book, ranking) has no per-row event time.
            # The fetch instant IS when it became knowable.
            out[AVAILABLE_TIME] = pd.to_datetime(int(available_time), unit="ms", utc=True)
            inferred = True
            warnings.append(
                "%s: available_time taken from the fetch time — this source is a "
                "snapshot with no per-row event time, so it cannot be replayed "
                "bar by bar" % (node or kind.value)
            )
    if (
        EVENT_TIME in (schema.required + schema.optional)
        and EVENT_TIME not in out.columns
        and AVAILABLE_TIME in out.columns
    ):
        out[EVENT_TIME] = out[AVAILABLE_TIME]

    # clamp clock skew rather than fail a live monitor on it
    if EVENT_TIME in out.columns and AVAILABLE_TIME in out.columns and not out.empty:
        skewed = out[EVENT_TIME] > out[AVAILABLE_TIME]
        if bool(skewed.any()):
            count = int(skewed.sum())
            out.loc[skewed, EVENT_TIME] = out.loc[skewed, AVAILABLE_TIME]
            warnings.append(
                "%s: clamped %d row(s) whose event_time was after available_time"
                % (node or kind.value, count)
            )

    # 6. derive the identity columns a shape requires but the source omits
    if kind is FrameKind.EVENT and "event_id" not in out.columns and not out.empty:
        # Events need a stable identity for de-duplication. Derive one from the
        # fields that make an event unique rather than leave it blank, and make
        # it deterministic so the same event hashes the same on the next poll.
        import hashlib

        def _event_id(row) -> str:
            seed = "|".join(
                str(row.get(column, ""))
                for column in ("source_id", "topic", EVENT_TIME, INSTRUMENT, "title")
            )
            return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

        out["event_id"] = out.apply(_event_id, axis=1)
        warnings.append(
            "%s: event_id derived from source_id+topic+time+instrument+title "
            "(the source supplies none)" % (node or kind.value)
        )
    if kind is FrameKind.POSITION and "side" not in out.columns and "quantity" in out.columns:
        quantities = pd.to_numeric(out["quantity"], errors="coerce")
        out["side"] = quantities.map(
            lambda value: "flat" if pd.isna(value) or value == 0
            else ("long" if value > 0 else "short")
        )
    if (
        kind is FrameKind.BOOK
        and "level" not in out.columns
        and "side" in out.columns
        and not out.empty
    ):
        # Depth endpoints return the ladder already ordered best-first per side
        # and leave the rung number implicit. It is required by the shape (a
        # consumer asking "what is at level 3" cannot count rows itself once the
        # frame has been filtered), so derive it from that order and say so.
        out["level"] = out.groupby("side").cumcount() + 1
        warnings.append(
            "%s: level derived from row order within each side (the source "
            "returns the ladder best-first and states no rung number)"
            % (node or kind.value)
        )

    # 7. numerics + instrument casing
    for column in schema.numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if INSTRUMENT in out.columns:
        out[INSTRUMENT] = out[INSTRUMENT].astype(str).str.upper()

    # An empty response is a fact, not a failure. Without this an empty METRIC
    # body raised FrameValidationError for "missing metric/value" — there was
    # nothing to melt, so the columns were never created — and the caller saw
    # "error" where the truth was "read it, there was nothing in it".
    if out.empty:
        for column in schema.required:
            if column not in out.columns:
                out[column] = pd.Series(dtype="object")

    if validate:
        schema.validate(out, node=node)
    return out, warnings, inferred
