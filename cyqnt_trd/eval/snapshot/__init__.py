"""Raw snapshot → panel bundle: what it takes to rebuild or extend the tracked bundle.

The bundle that ships with ``cyqnt_trd.eval`` (``data/top10_daily.parquet``) was
built from a raw snapshot of public Binance USDⓈ-M data. The snapshot itself is
~168 MB and is not tracked; these three modules regenerate and consume it:

``acquire``
    Download: daily klines, per-settlement funding (with mark-price proxies and
    gap diagnostics), and — additive — klines at other bar widths and
    open-interest history.
``pipeline``
    Load: contract-lifecycle masks, the current / point-in-time top-ten pools,
    funding accumulated to the cell grid, open interest as-of aligned.
``build_bundle``
    Write a bundle in the layout :func:`cyqnt_trd.eval.load_bundle` reads.

Snapshot layout under the data directory::

    acquisition_config.json  exchange_info.json  universes.json
    daily_manifest.json  funding_manifest.json  historical_top10.csv  current_top10.csv
    daily/<SYM>.parquet               1d klines
    klines_<interval>/<SYM>.parquet   other bar widths          (--stage klines)
    funding/<SYM>.parquet + .json     settlements + audit stamp
    funding_raw/  funding_mark_klines/
    open_interest_<period>/<SYM>.parquet                         (--stage oi)

The data directory defaults to ``$CYQNT_EVAL_SNAPSHOT``, else ``./eval_snapshot``
(git-ignored), so nothing is written inside the installed package.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = ["default_data_dir", "interval_ms", "klines_dir", "oi_dir"]

_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}


def default_data_dir() -> Path:
    env = os.environ.get("CYQNT_EVAL_SNAPSHOT")
    return Path(env).expanduser() if env else Path.cwd() / "eval_snapshot"


def interval_ms(interval: str) -> int:
    """Width of a Binance interval string (``15m``, ``4h``, ``1d``) in milliseconds.

    Weekly/monthly bars are refused: they do not tile the UTC day grid the
    panel's lifecycle and funding logic assume.
    """
    m = re.fullmatch(r"(\d+)([mhd])", str(interval))
    if not m or int(m.group(1)) <= 0:
        raise ValueError(f"unsupported interval {interval!r}; use <n>m, <n>h or <n>d")
    return int(m.group(1)) * _UNIT_MS[m.group(2)]


def klines_dir(data_dir: Path, interval: str) -> Path:
    """``daily/`` for 1d (the shipped snapshot's layout), ``klines_<interval>/`` otherwise."""
    interval_ms(interval)
    return Path(data_dir) / ("daily" if interval == "1d" else f"klines_{interval}")


def oi_dir(data_dir: Path, period: str) -> Path:
    interval_ms(period)
    return Path(data_dir) / f"open_interest_{period}"
