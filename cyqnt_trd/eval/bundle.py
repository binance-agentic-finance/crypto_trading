"""The panel a factor is scored on.

The tracked bundle is the **ten USDⓈ-M perpetuals with the largest 30-day quote
volume** (ranked over 2026-08-11…2026-09-09), daily UTC, 2021-01-01…2026-09-09.
It ships with the repo (~0.9 MB) so `cyqnt_trd.eval` runs on a clean checkout with
no download and no network.

Three things the bundle carries that a bare OHLCV table does not:

* ``funding`` — the *actual* per-settlement funding paid per unit of base asset,
  already accumulated to the daily grid. Not an assumed 8-hour rate.
* ``mask`` — who was eligible to be ranked/held on each date: listed, at least 60
  observed bars, and inside the contract's own lifecycle. Ranking outside the
  mask is the most common way a crypto cross-section leaks.
* ``meta`` — the ranking window, the selection caveat, and the source snapshot,
  so a scorecard can state what it was computed on.

Selection caveat, repeated because it changes how results read: the ten names are
chosen by volume **as of the snapshot date** and then looked at backwards. That is
retrospective selection. `cyqnt_trd.eval.snapshot` can also build the point-in-time
monthly top-ten (`--pool historical`) for comparison; the framework defaults to the
fixed ten because a capability users can re-run must be reproducible, not because
it is less biased.

A bundle written by `cyqnt_trd.eval.snapshot.build_bundle` may carry more than the
shipped one: ``extras`` fields (e.g. ``open_interest``) are stored as further
``field`` values, and a non-daily grid records its ``cell_scheme`` in the meta
file. The shipped bundle has neither, so loading it is unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = HERE / "data" / "top10_daily.parquet"
PRICE_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume")
#: Field names a bundle file reserves; anything else in its ``field`` column is an extra.
CORE_FIELDS = (*PRICE_FIELDS, "funding", "eligible")


@dataclass(frozen=True)
class Panel:
    """Aligned daily panels; every frame shares one index and one column set."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    quote_volume: pd.DataFrame
    funding: pd.DataFrame
    mask: pd.DataFrame
    meta: dict = field(default_factory=dict)
    #: Optional aligned frames beyond OHLCV+funding: taker_buy/taker_sell,
    #: open_interest, long_short_ratio, liq_buy/liq_sell, as-of aligned event
    #: pulses. Same cells x symbols grid, same "missing is NaN" rule. Deliberately
    #: outside the panel fingerprint (see provenance.panel_fingerprint) so adding
    #: one does not invalidate a calibration computed on price and funding alone.
    extras: dict = field(default_factory=dict)
    #: Which cell-sampling scheme the index came from, and therefore what
    #: ``validate`` demands of it. ``daily_utc`` (default) is the shipped bundle's
    #: contract and what ``engine.evaluate_factor`` requires. ``regular`` is any
    #: other fixed cell width (1h bars, 4h bars). ``irregular`` is volume/dollar/
    #: vol bars, where uneven spacing is the whole point. Defaulting to the strict
    #: option keeps every existing panel and the shipped calibration unaffected.
    cell_scheme: str = "daily_utc"

    @property
    def symbols(self):
        return list(self.close.columns)

    @property
    def index(self):
        return self.close.index

    def field(self, name: str) -> pd.DataFrame:
        """Read a panel frame by name, core field or extra.

        Lets an adapter ask for ``taker_buy`` the same way it asks for ``close``
        without knowing which of the two carries it.
        """
        if name in (*PRICE_FIELDS, "funding", "mask"):
            return getattr(self, name)
        if name in self.extras:
            return self.extras[name]
        raise KeyError(f"panel has no field {name!r}; core fields are "
                       f"{[*PRICE_FIELDS, 'funding', 'mask']}, "
                       f"extras are {sorted(self.extras)}")

    def with_extras(self, **frames: pd.DataFrame) -> "Panel":
        """Return a copy carrying additional aligned frames."""
        merged = {**self.extras, **frames}
        return Panel(**{f: getattr(self, f) for f in PRICE_FIELDS},
                     funding=self.funding, mask=self.mask, meta=dict(self.meta),
                     extras=merged, cell_scheme=self.cell_scheme)

    def describe(self) -> str:
        first, last = self.index[0].date(), self.index[-1].date()
        per_day = float(self.mask.sum(axis=1).mean())
        if self.cell_scheme == "daily_utc":
            return (f"{len(self.symbols)} symbols, {len(self.index)} daily bars "
                    f"{first} → {last}, {per_day:.1f} eligible names/day")
        return (f"{len(self.symbols)} symbols, {len(self.index)} {self.cell_scheme} bars "
                f"{first} → {last}, {per_day:.1f} eligible names/bar")

    def validate(self) -> None:
        ref = self.close
        if not isinstance(ref, pd.DataFrame):
            raise TypeError("panel close must be a DataFrame")
        if not isinstance(ref.index, pd.DatetimeIndex) or ref.index.tz is None:
            raise ValueError("panel index must be a tz-aware DatetimeIndex (UTC)")
        if not len(ref.index) or not ref.index.is_unique or not ref.index.is_monotonic_increasing:
            raise ValueError("panel index must be nonempty, unique, and increasing")
        utc = ref.index.tz_convert("UTC")
        if self.cell_scheme == "daily_utc":
            if (not ref.index.tz_localize(None).equals(utc.tz_localize(None))
                    or not utc.equals(utc.normalize())
                    or (len(utc) > 1 and not (utc[1:] - utc[:-1] == pd.Timedelta(days=1)).all())):
                raise ValueError("panel requires a complete daily UTC midnight grid")
        elif self.cell_scheme == "regular":
            # Any fixed cell width. Still has to be gapless and evenly spaced: a
            # cell index with holes makes "h cells forward" mean different amounts
            # of time in different places, and every horizon claim stops meaning
            # one thing.
            steps = np.diff(utc.asi8)
            if len(steps) and not (steps == steps[0]).all():
                raise ValueError("a 'regular' panel requires evenly spaced cells with no gaps")
        elif self.cell_scheme == "irregular":
            pass        # volume/dollar/vol bars: spacing is the point, nothing to check
        else:
            raise ValueError(f"unknown cell_scheme {self.cell_scheme!r}; "
                             f"use 'daily_utc', 'regular' or 'irregular'")
        if (not len(ref.columns) or not ref.columns.is_unique
                or any(not isinstance(s, str) or not s.strip() for s in ref.columns)):
            raise ValueError("panel requires unique, nonempty string symbol columns")
        for name in ("open", "high", "low", "volume", "quote_volume", "funding", "mask"):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"panel frame '{name}' must be a DataFrame")
            if not frame.index.equals(ref.index) or list(frame.columns) != list(ref.columns):
                raise ValueError(f"panel frame '{name}' is not aligned with 'close'")
        if not all(is_bool_dtype(dtype) for dtype in self.mask.dtypes) or self.mask.isna().any().any():
            raise ValueError("panel mask must be boolean with no missing values")
        for name, frame in self.extras.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"panel extra '{name}' must be a DataFrame")
            if not frame.index.equals(ref.index) or list(frame.columns) != list(ref.columns):
                raise ValueError(f"panel extra '{name}' is not aligned with 'close'")
            if any(not is_numeric_dtype(d) or is_complex_dtype(d) for d in frame.dtypes):
                raise ValueError(f"panel extra '{name}' must contain real numeric values")
            if np.isinf(frame.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"panel extra '{name}' contains infinite values")

        # NaN is an unknown observation, including when a position is held.
        # Preserve it for coverage/PnL diagnostics; zero or infinity is never
        # a valid replacement for an absent price or funding observation.
        values = {}
        for name in (*PRICE_FIELDS, "funding"):
            frame = getattr(self, name)
            if any(not is_numeric_dtype(dtype) or is_complex_dtype(dtype) or is_bool_dtype(dtype)
                   for dtype in frame.dtypes):
                raise ValueError(f"panel frame '{name}' must contain real numeric values")
            values[name] = frame.to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values[name]).any():
                raise ValueError(f"panel frame '{name}' contains infinite values")
            if name in ("open", "high", "low", "close") and (values[name] <= 0).any():
                raise ValueError(f"panel prices in '{name}' must be positive or missing")
            if name in ("volume", "quote_volume") and (values[name] < 0).any():
                raise ValueError(f"panel values in '{name}' must be nonnegative or missing")
        if ((values["high"] < values["low"]).any()
                or any((values[name] > values["high"]).any()
                       or (values[name] < values["low"]).any() for name in ("open", "close"))):
            raise ValueError("panel OHLC prices must satisfy low <= open/close <= high")


def load_bundle(path: str | Path | None = None) -> Panel:
    """Load the tracked top-10 bundle (or a bundle written in the same layout)."""
    path = Path(path) if path is not None else DEFAULT_BUNDLE
    if not path.exists():
        raise FileNotFoundError(
            f"bundle not found: {path}\n"
            "The tracked bundle lives at cyqnt_trd/eval/data/top10_daily.parquet. "
            "To build a different universe, see cyqnt_trd/eval/README.md "
            "(python -m cyqnt_trd.eval.snapshot.build_bundle --help).")
    long = pd.read_parquet(path)
    if not {"ts", "symbol", "field", "value"}.issubset(long.columns):
        raise ValueError("bundle requires ts, symbol, field, and value columns")
    required = {*PRICE_FIELDS, "funding", "eligible"}
    if not required.issubset(long.field.unique()):
        raise ValueError(f"bundle is missing fields: {sorted(required - set(long.field.unique()))}")
    wide = {f: long[long.field == f].pivot(index="ts", columns="symbol", values="value")
            for f in long.field.unique()}
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    order = meta.get("symbols") or sorted(wide["close"].columns)
    if (not isinstance(order, list) or any(not isinstance(symbol, str) for symbol in order)
            or len(order) != len(set(order))
            or set(order) != set(wide["close"].columns)):
        raise ValueError("bundle metadata symbols must match the price symbols exactly")
    frames = {f: wide[f].reindex(columns=order).sort_index() for f in wide}
    eligible = frames["eligible"]
    if eligible.isna().any().any() or not eligible.isin([0, 1]).all().all():
        raise ValueError("bundle eligible values must be explicit 0 or 1, with no missing values")
    panel = Panel(
        **{f: frames[f] for f in PRICE_FIELDS},
        funding=frames["funding"],
        mask=eligible.astype(bool),
        meta=meta,
        extras={f: frames[f] for f in frames if f not in CORE_FIELDS},
        cell_scheme=meta.get("cell_scheme", "daily_utc"),
    )
    panel.validate()
    return panel


def to_long(panel: Panel) -> pd.DataFrame:
    """Inverse of the loader's reshape; used when writing a bundle."""
    panel.validate()
    out = []
    clash = sorted(set(panel.extras) & set(CORE_FIELDS))
    if clash:
        raise ValueError(f"panel extras {clash} collide with reserved bundle fields")
    fields = {**{f: getattr(panel, f) for f in PRICE_FIELDS},
              "funding": panel.funding, "eligible": panel.mask.astype(float),
              **panel.extras}
    for name, frame in fields.items():
        x = frame.rename_axis(index="ts", columns="symbol").reset_index().melt(
            id_vars="ts", var_name="symbol", value_name="value")
        x["field"] = name
        out.append(x)
    return pd.concat(out, ignore_index=True)
