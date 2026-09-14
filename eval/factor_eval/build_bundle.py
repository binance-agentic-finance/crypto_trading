"""Build the tracked panel bundle from a raw snapshot.

The bundle that ships with the repo was produced by this script from the
reference study's snapshot (`eval/alpha101_crypto/data/`, not tracked — see that
directory's README). Run it only when you want a *different* universe or a newer
cut-off; the shipped bundle is the fixed thing every scorecard is measured on.

    python eval/factor_eval/build_bundle.py --pool current
    python eval/factor_eval/build_bundle.py --pool historical --out my_bundle.parquet

`--pool current` = the ten largest by 30-day quote volume as of the snapshot
(retrospective selection). `--pool historical` = the point-in-time monthly top ten
(80 names over the period, ~7–10 eligible on any given day).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "alpha101_crypto"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", choices=["current", "historical"], default="current")
    ap.add_argument("--out", default=str(HERE / "data" / "top10_daily.parquet"))
    args = ap.parse_args()

    if not (STUDY / "data" / "daily").exists():
        sys.exit(f"raw snapshot missing under {STUDY / 'data'}; "
                 f"run {STUDY.name}/acquire_data.py first (~168 MB)")
    sys.path.insert(0, str(STUDY))
    import run_pipeline as study                      # noqa: E402

    raw, mask, design = study.load_inputs(args.pool)
    funding, coverage = study.funding_panel(raw["open"].index, raw["open"].columns)

    sys.path.insert(0, str(HERE.parent))
    from factor_eval.bundle import Panel, to_long     # noqa: E402

    panel = Panel(**{f: raw[f] for f in ("open", "high", "low", "close",
                                         "volume", "quote_volume")},
                  funding=funding, mask=mask.astype(bool))
    panel.validate()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    to_long(panel).to_parquet(out, index=False, compression="zstd")

    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pool": args.pool,
        "symbols": list(panel.close.columns),
        "start": str(panel.index[0]), "end": str(panel.index[-1]),
        "bars": int(len(panel.index)),
        "eligible_cells": int(panel.mask.sum().sum()),
        "mean_eligible_per_day": round(float(panel.mask.sum(axis=1).mean()), 3),
        "funding_observations": int(panel.funding.notna().sum().sum()),
        "funding_price_coverage": coverage,
        "ranking": design.get("ranking_scope"),
        "selection_caveat": design.get("selection_caveat"),
        "source": "eval/alpha101_crypto/data (public Binance USDⓈ-M endpoints)",
        "fields": {
            "open/high/low/close": "daily UTC klines, lifecycle-masked",
            "volume": "base asset volume", "quote_volume": "USDT quote volume",
            "funding": "actual per-settlement funding accumulated to the daily grid, "
                       "quote per unit of base; positive is an expense for a long",
            "eligible": "listed, >=60 observed bars, inside contract lifecycle, in pool",
        },
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) and {out.with_suffix('.meta.json').name}")
    print(Panel(**{f: raw[f] for f in ('open', 'high', 'low', 'close', 'volume', 'quote_volume')},
                funding=funding, mask=mask.astype(bool)).describe())


if __name__ == "__main__":
    main()
