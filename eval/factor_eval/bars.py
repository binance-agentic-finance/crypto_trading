"""Stage 0 and 1: raw fine-grained bars -> cells -> an aligned Panel.

A cell is one decision-and-hold unit. Time bars are the default because a
cross-section needs every symbol to be comparable at the same `t`, and equal time
gives that for free. The information-driven schemes (volume, dollar, vol) sample
more finely when more is happening, which is the point of them, but they produce a
**different cell boundary per symbol** -- so a cross-sectional strategy has to
re-align them onto one common grid, and every cell where a symbol had no bar is
`mask=False` rather than a forward-filled guess.

Sampling rule, applied by every scheme: a cell closes on the bar that **crosses**
the threshold, and that bar belongs to the closing cell. The final partial cell is
dropped -- it has not met its own definition yet, and keeping it would make the
last cell of a backtest systematically different from all the others.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bundle import PRICE_FIELDS, Panel

__all__ = ["SCHEMES", "sample_bars", "align_bars", "build_panel", "bar_meta"]

SCHEMES = ("time", "volume", "dollar", "vol")

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "quote_volume": "sum"}


def _check_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("raw bars must be a DataFrame")
    missing = sorted(set(PRICE_FIELDS) - set(raw.columns))
    if missing:
        raise ValueError(f"raw bars are missing columns: {missing}")
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.tz is None:
        raise ValueError("raw bars need a tz-aware DatetimeIndex (UTC)")
    if not raw.index.is_monotonic_increasing or not raw.index.is_unique:
        raise ValueError("raw bar timestamps must be unique and sorted")
    return raw[list(PRICE_FIELDS)].astype(float)


def _group_by_threshold(load: np.ndarray, threshold: float) -> np.ndarray:
    """Cell id per raw bar: a cell closes on the bar that crosses `threshold`."""
    ids = np.empty(len(load), dtype=np.int64)
    cell, running = 0, 0.0
    for i, value in enumerate(load):
        ids[i] = cell
        running += 0.0 if not np.isfinite(value) else float(value)
        if running >= threshold:
            cell += 1
            running = 0.0
    return ids


def sample_bars(raw: pd.DataFrame, scheme: str = "time", **params) -> pd.DataFrame:
    """Aggregate fine-grained bars into cells.

    `time`     `interval="1h"`      fixed duration
    `volume`   `threshold=...`      cumulative base volume
    `dollar`   `threshold=...`      cumulative quote volume
    `vol`      `target_vol=...`     cumulative absolute log return

    Returns one row per cell indexed by the cell's **open time**, carrying OHLCV
    plus `n_raw` (how many raw bars went in) and `close_time`.
    """
    raw = _check_raw(raw)
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")

    stamps = raw.index.to_series()
    if scheme == "time":
        interval = params.pop("interval", "1D")
        if params:
            raise TypeError(f"unexpected params for time bars: {sorted(params)}")
        grouped = raw.resample(interval, label="left", closed="left")
        bars = grouped.agg(_AGG)
        bars["n_raw"] = grouped.size()
        bars["close_time"] = stamps.resample(interval, label="left", closed="left").last()
        # A period with no raw bar is a real gap, not a zero-volume cell.
        return bars[bars["n_raw"] > 0]

    if scheme == "volume":
        load, threshold = raw["volume"].to_numpy(float), params.pop("threshold")
    elif scheme == "dollar":
        load, threshold = raw["quote_volume"].to_numpy(float), params.pop("threshold")
    else:
        returns = np.abs(np.log(raw["close"] / raw["close"].shift(1)).to_numpy(float))
        load, threshold = np.nan_to_num(returns, nan=0.0), params.pop("target_vol")
    if params:
        raise TypeError(f"unexpected params for {scheme} bars: {sorted(params)}")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a positive finite number")

    ids = _group_by_threshold(load, float(threshold))
    grouped = raw.groupby(ids)
    bars = grouped.agg(_AGG)
    bars["n_raw"] = grouped.size()
    bars["close_time"] = stamps.groupby(ids).last().to_numpy()
    bars.index = pd.DatetimeIndex(stamps.groupby(ids).first().to_numpy(),
                                  name=raw.index.name).tz_convert(raw.index.tz)
    # The last cell never crossed its threshold, so it is not the same kind of
    # object as the others: drop it rather than let a half-filled cell end the
    # sample.
    return bars.iloc[:-1] if len(bars) > 1 else bars


def align_bars(bars: dict[str, pd.DataFrame], grid: pd.DatetimeIndex
               ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Put per-symbol cells on one common grid; return frames plus a coverage mask.

    Cells where a symbol produced no bar are left missing and marked ineligible.
    Forward filling instead would invent a price the market never printed, and a
    cross-section built on invented prices ranks fiction.
    """
    if not isinstance(grid, pd.DatetimeIndex) or grid.tz is None:
        raise ValueError("grid must be a tz-aware DatetimeIndex")
    symbols = list(bars)
    frames = {f: pd.DataFrame(index=grid, columns=symbols, dtype=float)
              for f in PRICE_FIELDS}
    covered = pd.DataFrame(False, index=grid, columns=symbols)
    for symbol, frame in bars.items():
        aligned = frame.reindex(grid)
        for f in PRICE_FIELDS:
            frames[f][symbol] = aligned[f]
        covered[symbol] = aligned["close"].notna().to_numpy()
    return frames, covered


def build_panel(bars: dict[str, pd.DataFrame], *, grid: pd.DatetimeIndex | None = None,
                funding: pd.DataFrame | None = None,
                extras: dict[str, pd.DataFrame] | None = None,
                min_history: int = 60, cell_scheme: str = "regular",
                meta: dict | None = None) -> Panel:
    """Stage 1: per-symbol cells -> one aligned `Panel`.

    Eligibility (`mask`) is coverage AND at least `min_history` observed cells
    behind this one. That second condition is what stops a freshly listed name from
    entering the cross-section on its first print, where every rolling factor is
    still warming up and its rank is noise.

    `funding` defaults to zero, which is correct for spot and wrong for perpetuals
    -- pass the real per-settlement series for a perp panel rather than letting a
    zero stand in for an unpaid cost.
    """
    if not bars:
        raise ValueError("build_panel needs at least one symbol")
    if grid is None:
        grid = pd.DatetimeIndex(sorted(set().union(*(f.index for f in bars.values()))))
        grid = grid.tz_convert("UTC") if grid.tz is not None else grid.tz_localize("UTC")
    frames, covered = align_bars(bars, grid)

    history = covered.cumsum().shift(1).fillna(0.0)
    mask = covered & (history >= min_history)

    if funding is None:
        funding = pd.DataFrame(0.0, index=grid, columns=list(bars))
    else:
        funding = funding.reindex(index=grid, columns=list(bars))
    panel = Panel(**frames, funding=funding, mask=mask.astype(bool),
                  meta={"grid": str(grid.freq) if grid.freq is not None else "irregular",
                        "n_cells": len(grid), "min_history": min_history,
                        **(meta or {})},
                  extras={k: v.reindex(index=grid, columns=list(bars))
                          for k, v in (extras or {}).items()},
                  cell_scheme=cell_scheme)
    panel.validate()
    return panel


def bar_meta(scheme: str, **params) -> dict:
    """The record of how stage 0 sampled, to be carried in `BacktestSpec.grid`."""
    return {"scheme": scheme, **params,
            "rule": "a cell closes on the bar that crosses the threshold; "
                    "the final partial cell is dropped"}
