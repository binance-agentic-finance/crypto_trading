"""The panel a factor is scored on.

The tracked bundle is the **ten USDⓈ-M perpetuals with the largest 30-day quote
volume** (ranked over 2026-08-11…2026-09-09), daily UTC, 2021-01-01…2026-09-09.
It ships with the repo (~0.9 MB) so `factor_eval` runs on a clean checkout with
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
retrospective selection. `eval/alpha101_crypto/` also evaluates a point-in-time
monthly top-ten for comparison; the framework defaults to the fixed ten because a
capability users can re-run must be reproducible, not because it is less biased.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = HERE / "data" / "top10_daily.parquet"
PRICE_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume")


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

    @property
    def symbols(self):
        return list(self.close.columns)

    @property
    def index(self):
        return self.close.index

    def describe(self) -> str:
        first, last = self.index[0].date(), self.index[-1].date()
        per_day = float(self.mask.sum(axis=1).mean())
        return (f"{len(self.symbols)} symbols, {len(self.index)} daily bars "
                f"{first} → {last}, {per_day:.1f} eligible names/day")

    def validate(self) -> None:
        ref = self.close
        if not isinstance(ref.index, pd.DatetimeIndex) or ref.index.tz is None:
            raise ValueError("panel index must be a tz-aware DatetimeIndex (UTC)")
        for name in ("open", "high", "low", "volume", "quote_volume", "funding", "mask"):
            frame = getattr(self, name)
            if not frame.index.equals(ref.index) or list(frame.columns) != list(ref.columns):
                raise ValueError(f"panel frame '{name}' is not aligned with 'close'")
        if self.mask.dtypes.map(lambda d: d != bool).any():
            raise ValueError("panel mask must be boolean")


def load_bundle(path: str | Path | None = None) -> Panel:
    """Load the tracked top-10 bundle (or a bundle written in the same layout)."""
    path = Path(path) if path is not None else DEFAULT_BUNDLE
    if not path.exists():
        raise FileNotFoundError(
            f"bundle not found: {path}\n"
            "The tracked bundle lives at factor_eval/data/top10_daily.parquet. "
            "To build a different universe, see eval/README.md.")
    long = pd.read_parquet(path)
    wide = {f: long[long.field == f].pivot(index="ts", columns="symbol", values="value")
            for f in long.field.unique()}
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    order = meta.get("symbols") or sorted(wide["close"].columns)
    frames = {f: wide[f].reindex(columns=order).sort_index() for f in wide}
    panel = Panel(
        **{f: frames[f] for f in PRICE_FIELDS},
        funding=frames["funding"],
        mask=frames["eligible"].fillna(0.0).astype(bool),
        meta=meta,
    )
    panel.validate()
    return panel


def to_long(panel: Panel) -> pd.DataFrame:
    """Inverse of the loader's reshape; used when writing a bundle."""
    out = []
    fields = {**{f: getattr(panel, f) for f in PRICE_FIELDS},
              "funding": panel.funding, "eligible": panel.mask.astype(float)}
    for name, frame in fields.items():
        x = frame.rename_axis(index="ts", columns="symbol").reset_index().melt(
            id_vars="ts", var_name="symbol", value_name="value")
        x["field"] = name
        out.append(x)
    return pd.concat(out, ignore_index=True)
