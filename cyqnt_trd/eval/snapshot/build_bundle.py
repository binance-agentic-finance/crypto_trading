"""Build a panel bundle from a raw snapshot.

The bundle that ships with the repo (``cyqnt_trd/eval/data/top10_daily.parquet``)
was produced by this builder from the reference snapshot. Run it when you want
a *different* universe, a newer cut-off, another bar width, or open interest in
``Panel.extras``; the shipped bundle is the fixed thing every scorecard and the
shipped calibration are measured on, so ``--out`` is required and a rebuilt
bundle needs its own calibration (``python -m cyqnt_trd.eval --bundle ... calibrate``).

    # snapshot first (network, ~168 MB):  python -m cyqnt_trd.eval.snapshot.acquire --data-dir eval_snapshot
    python -m cyqnt_trd.eval.snapshot.build_bundle --data-dir eval_snapshot --pool current --out my_bundle.parquet
    python -m cyqnt_trd.eval.snapshot.build_bundle --data-dir eval_snapshot --pool historical --out hist.parquet
    python -m cyqnt_trd.eval.snapshot.build_bundle --data-dir eval_snapshot --interval 4h --open-interest 1h --out top10_4h.parquet
    python -m cyqnt_trd.eval.snapshot.build_bundle --data-dir eval_snapshot --dry-run     # check inputs, write nothing

`--pool current` = the ten largest by 30-day quote volume as of the snapshot
(retrospective selection). `--pool historical` = the point-in-time monthly top ten
(80 names over the reference period, ~7–10 eligible on any given day).

A ``1d`` bundle has ``cell_scheme='daily_utc'`` — what ``evaluate()`` and the
calibration require. Any other width is ``'regular'``: usable with
``framework.backtest_weights`` / ``portfolio.simulate``, not with the scorecard.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import default_data_dir, interval_ms, klines_dir, oi_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m cyqnt_trd.eval.snapshot.build_bundle",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None,
                    help="snapshot directory (default: $CYQNT_EVAL_SNAPSHOT or ./eval_snapshot)")
    ap.add_argument("--pool", choices=["current", "historical"], default="current")
    ap.add_argument("--interval", default="1d", help="bar width of the grid (1d, 4h, 1h, ...)")
    ap.add_argument("--min-history", type=int, default=60,
                    help="observed bars before a name is eligible (default 60, the shipped rule)")
    ap.add_argument("--open-interest", metavar="PERIOD", default=None,
                    help="add open_interest / open_interest_value extras from open_interest_<PERIOD>/")
    ap.add_argument("--out", default=None, help="bundle path to write (.parquet; meta goes next to it)")
    ap.add_argument("--dry-run", action="store_true",
                    help="check that the snapshot has what this build needs, write nothing")
    return ap


def plan(args) -> dict:
    """What the build will read, and what is missing. No data is loaded."""
    data = Path(args.data_dir or default_data_dir()).expanduser()
    interval_ms(args.interval)
    need = [data / "acquisition_config.json", data / "exchange_info.json", data / "universes.json"]
    if args.pool == "historical":
        need.append(data / "daily_manifest.json")
    missing = [str(p) for p in need if not p.exists()]
    symbols = []
    universes = data / "universes.json"
    if universes.exists():
        info = json.loads(universes.read_text())
        symbols = info["current_top10"] if args.pool == "current" else info["historical_union"]
    dirs = {"klines": klines_dir(data, args.interval), "funding": data / "funding"}
    if args.pool == "historical":
        dirs["daily (pool ranking)"] = data / "daily"
    if args.open_interest:
        dirs["open_interest"] = oi_dir(data, args.open_interest)
    per_dir = {}
    for label, d in dirs.items():
        absent = [s for s in symbols if not (d / f"{s}.parquet").exists()]
        per_dir[label] = {"path": str(d), "symbols": len(symbols), "missing": absent}
    return {"data_dir": str(data), "pool": args.pool, "interval": args.interval,
            "missing_files": missing, "inputs": per_dir}


def build(args) -> Path:
    from ..bundle import Panel, to_long
    from .pipeline import funding_panel, load_inputs, open_interest_panel

    data = Path(args.data_dir or default_data_dir()).expanduser()
    raw, mask, design = load_inputs(args.pool, data, interval=args.interval,
                                    min_history=args.min_history)
    funding, coverage = funding_panel(raw["open"].index, raw["open"].columns, data)
    extras, oi_detail = {}, None
    if args.open_interest:
        extras, oi_detail = open_interest_panel(raw["open"].index, raw["open"].columns, data,
                                                period=args.open_interest)
    daily = args.interval == "1d"
    panel = Panel(**{f: raw[f] for f in ("open", "high", "low", "close", "volume", "quote_volume")},
                  funding=funding, mask=mask.astype(bool), extras=extras,
                  cell_scheme="daily_utc" if daily else "regular")
    panel.validate()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    to_long(panel).to_parquet(out, index=False, compression="zstd")

    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pool": args.pool,
        "interval": args.interval,
        "cell_scheme": panel.cell_scheme,
        "symbols": list(panel.close.columns),
        "start": str(panel.index[0]), "end": str(panel.index[-1]),
        "bars": int(len(panel.index)),
        "eligible_cells": int(panel.mask.sum().sum()),
        "mean_eligible_per_bar": round(float(panel.mask.sum(axis=1).mean()), 3),
        "funding_observations": int(panel.funding.notna().sum().sum()),
        "funding_price_coverage": coverage,
        "ranking": design.get("ranking_scope"),
        "selection_caveat": design.get("selection_caveat"),
        "source": "cyqnt_trd.eval.snapshot (public Binance USDⓈ-M endpoints)",
        "fields": {
            "open/high/low/close": f"{args.interval} UTC klines, lifecycle-masked",
            "volume": "base asset volume", "quote_volume": "USDT quote volume",
            "funding": "actual per-settlement funding accumulated to the cell grid, "
                       "quote per unit of base; positive is an expense for a long",
            "eligible": f"listed, >={args.min_history} observed bars, inside contract lifecycle, in pool",
        },
    }
    if daily:
        meta["mean_eligible_per_day"] = meta["mean_eligible_per_bar"]
    if extras:
        meta["extras"] = sorted(extras)
        meta["fields"]["open_interest"] = (f"openInterestHist {args.open_interest}, as-of the bar close, "
                                           f"available one period after its stamp; exchange keeps ~30 days, "
                                           f"older cells are missing")
        meta["open_interest_coverage"] = oi_detail
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) and {out.with_suffix('.meta.json').name}")
    print(panel.describe())
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        interval_ms(args.interval)
        if args.open_interest:
            interval_ms(args.open_interest)
    except ValueError as e:
        raise SystemExit(str(e))
    report = plan(args)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        ready = not report["missing_files"] and not any(v["missing"] for v in report["inputs"].values())
        print("ready to build" if ready else "NOT ready: fetch the missing inputs with "
              "python -m cyqnt_trd.eval.snapshot.acquire")
        return 0 if ready else 1
    if not args.out:
        raise SystemExit("--out is required: the shipped bundle is bound to the shipped calibration, "
                         "so a rebuild never overwrites it by default")
    if report["missing_files"]:
        raise SystemExit("snapshot incomplete, missing: " + ", ".join(report["missing_files"])
                         + "\nrun python -m cyqnt_trd.eval.snapshot.acquire first (~168 MB)")
    build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
