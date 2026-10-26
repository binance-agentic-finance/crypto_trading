"""CLI:  python -m cyqnt_trd.eval <score|calibrate|demo|bundle>

    python -m cyqnt_trd.eval demo
    python -m cyqnt_trd.eval score --factor cyqnt_trd/eval/examples/example_factors.py:reversal_5d
    python -m cyqnt_trd.eval score --factor cyqnt_trd.eval.examples.example_factors:reversal_5d
    python -m cyqnt_trd.eval score --factor my_ideas.py:my_factor --markdown card.md --json card.json
    python -m cyqnt_trd.eval calibrate --trials 100 --h 5 --cost 12
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Repository root, so `python cyqnt_trd/eval/__main__.py` also resolves the
# `cyqnt_trd` package. (Inserting `cyqnt_trd/` itself would expose its
# subpackages — `utils`, `strategies`, ... — as top-level names.)
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cyqnt_trd.eval import evaluate, load_bundle          # noqa: E402
from cyqnt_trd.eval.engine import DEFAULT_SPLITS, evaluate_factor   # noqa: E402
from cyqnt_trd.eval.provenance import calibration_contract          # noqa: E402


def _splits(args):
    return json.loads(Path(args.splits).read_text()) if args.splits else DEFAULT_SPLITS


def _load_callable(spec: str):
    """``path/to/file.py:name`` or ``package.module:name``."""
    if ":" not in spec:
        raise SystemExit("--factor must be 'file.py:function' or 'module:function'")
    where, name = spec.rsplit(":", 1)
    if where.endswith(".py"):
        path = Path(where).resolve()
        if not path.exists():
            raise SystemExit(f"no such file: {path}")
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(where)
    if not hasattr(module, name):
        raise SystemExit(f"{where} has no attribute {name!r}")
    return getattr(module, name)


def cmd_score(args):
    if args.quick and args.report_dir:
        raise ValueError("--report-dir requires diagnostics; omit --quick")
    factor = _load_callable(args.factor)
    panel = load_bundle(args.bundle)
    card = evaluate(factor, panel, name=args.name or args.factor.rsplit(":", 1)[-1],
                    primary_h=args.h, cost_bps=args.cost,
                    entry_lag=args.entry_lag, splits=_splits(args),
                    calibration=args.calibration, trials_seen=args.trials_seen,
                    with_diagnostics=not args.quick, n_bootstrap=args.bootstrap,
                    benchmark_symbol=args.benchmark,
                    with_incremental=not args.skip_incremental)
    print(card)
    if args.markdown:
        Path(args.markdown).write_text(card.to_markdown())
        print(f"\nwrote {args.markdown}")
    if args.json:
        Path(args.json).write_text(json.dumps(card.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
        print(f"wrote {args.json}")
    if args.report_dir:
        from cyqnt_trd.eval.plots import render_diagnostics
        paths = render_diagnostics(card, args.report_dir)
        print(f"wrote {paths['html']}")
    return 0 if card.verdict.startswith("PASS") else 1


def cmd_calibrate(args):
    """Re-measure the noise floor for a setting the shipped calibration does not cover."""
    panel = load_bundle(args.bundle)
    if args.trials < 30:
        raise ValueError("calibrate requires at least 30 trials; use 100 or more for a p95 screen")
    if not np.isfinite(args.phi) or abs(args.phi) >= 1:
        raise ValueError("AR(1) phi must be finite and strictly between -1 and 1")
    splits = _splits(args)
    contract = calibration_contract(panel, primary_h=args.h, cost_bps=args.cost,
                                    splits=splits, entry_lag=args.entry_lag)
    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(args.trials):
        noise = rng.standard_normal(panel.mask.shape)
        x = np.zeros_like(noise)
        for t in range(1, len(x)):
            x[t] = args.phi * x[t - 1] + noise[t]
        sig = pd.DataFrame(x, index=panel.index, columns=panel.symbols).where(panel.mask)
        out = evaluate_factor(sig, panel.open, panel.close, panel.mask, panel.funding,
                              horizons=(args.h,), cost_bps=args.cost,
                              splits=splits, entry_lag=args.entry_lag)
        rows.append(out["metrics"].assign(trial=k))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{args.trials}", flush=True)
    null = pd.concat(rows, ignore_index=True)
    cal = {"source": "recomputed by `python -m cyqnt_trd.eval calibrate`",
           "generator": f"AR(1) phi={args.phi} gaussian panels, dev-frozen sign, identical engine path",
           "trials": int(args.trials), "primary_h": int(args.h), "cost_bps": float(args.cost),
           "pool": panel.meta.get("pool", "custom bundle"), "entry_lag": args.entry_lag,
           "seed": int(args.seed), "contract": contract}
    for field in ("ic_mean", "icir", "ic_win", "ic_t_hac", "net_bp", "net_available_bp",
                  "sharpe_net", "turnover", "breakeven_cost_bps"):
        cal[field] = {}
        for split in ("dev", "val", "oot"):
            v = null.loc[null.split == split, field].astype(float).dropna()
            cal[field][split] = {q: (round(float(v.quantile(p)), 6) if len(v) else None)
                                 for q, p in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99))}
            cal[field][split]["n"] = int(len(v))
    Path(args.out).write_text(json.dumps(cal, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    print("ic_mean p95:", {s: cal["ic_mean"][s]["p95"] for s in ("dev", "val", "oot")})
    return 0


def cmd_demo(args):
    from cyqnt_trd.eval.examples import example_factors as ex
    panel = load_bundle(args.bundle)
    print(f"panel: {panel.describe()}\n")
    for name in ex.DEMO:
        card = evaluate(getattr(ex, name), panel, name=name, with_incremental=not args.fast)
        first = f" (blocking/missing: {', '.join(card.blocking)})" if card.blocking else ""
        print(f"{name:22s} {card.verdict:17s}{first}")
    return 0


def cmd_bundle(args):
    panel = load_bundle(args.bundle)
    print(panel.describe())
    print(json.dumps(panel.meta, ensure_ascii=False, indent=2))
    return 0


def cmd_capabilities(args):
    """List the capability manifests and whether each one runs on this panel."""
    from cyqnt_trd.eval.capability_registry import capability_factors, load_index
    index = load_index(args.index)
    print(index.describe())
    panel = load_bundle(args.bundle)
    rejected = {}
    factors = capability_factors(index, panel=panel, rejected=rejected)
    print(f"\nrunnable candidates: {len(factors)}")
    for name in factors:
        print(f"  {name}")
    if args.rejected:
        print(f"\nrejected: {len(rejected)}")
        for name, reason in rejected.items():
            print(f"  {name:44s} {reason}")
    return 0


def cmd_stages(args):
    """Print the pipeline stages with the input and output format of each."""
    from cyqnt_trd.eval.framework import describe_stages
    print(describe_stages())
    return 0


def cmd_compare(args):
    """Candidates -> matrix -> construction -> comparison against the baselines."""
    from cyqnt_trd.eval.experiment import run_comparison, write_report
    panel = load_bundle(args.bundle)
    result = run_comparison(
        panel, source=args.source, k=args.k, seed=args.seed, primary_h=args.h,
        cost_bps=args.cost, entry_lag=args.entry_lag, rebalance=args.rebalance,
        cap=args.cap, benchmark_symbol=args.benchmark, dedupe=not args.no_dedupe,
        max_corr=args.max_corr, confirm=not args.no_confirm,
        admit_on_noise_floor=not args.no_ic_floor, splits=_splits(args),
        calibration=args.calibration)
    print(result.summary())
    print(f"admitted: {', '.join(result.admitted)}\n")
    print(result.table.round(4).to_string())
    if args.out:
        paths = write_report(result, args.out)
        print(f"\nwrote {paths['markdown']}\nwrote {paths['json']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cyqnt_trd.eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=None, help="path to a bundle parquet")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score one factor")
    s.add_argument("--factor", required=True, help="file.py:function or module:function")
    s.add_argument("--name", default=None)
    s.add_argument("--h", type=int, default=3, help="primary horizon in days")
    s.add_argument("--cost", type=float, default=6.5, help="one-way cost in bp")
    s.add_argument("--calibration", help="matching calibration JSON from calibrate")
    s.add_argument("--entry-lag", type=int, default=2)
    s.add_argument("--splits", help="JSON mapping of dev/val/oot to [start, end]")
    s.add_argument("--trials-seen", type=int, default=1, help="all variants examined before selecting this factor")
    s.add_argument("--report-dir", help="write the complete matrix and offline diagnostic figures as HTML")
    s.add_argument("--quick", action="store_true", help="six gates only; omit the full diagnostic matrix")
    s.add_argument("--bootstrap", type=int, default=100, help="moving-block bootstrap draws per diagnostic")
    s.add_argument("--benchmark", default="BTCUSDT", help="explicit benchmark symbol for residual targets")
    s.add_argument("--skip-incremental", action="store_true",
                   help="skip G5 (faster; the per-date projection is the slow part)")
    s.add_argument("--markdown", default=None)
    s.add_argument("--json", default=None)
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("calibrate", help="re-measure the noise floor for a setting")
    c.add_argument("--trials", type=int, default=100)
    c.add_argument("--h", type=int, default=3)
    c.add_argument("--cost", type=float, default=6.5)
    c.add_argument("--entry-lag", type=int, default=2)
    c.add_argument("--splits", help="JSON mapping of dev/val/oot to [start, end]")
    c.add_argument("--phi", type=float, default=0.9)
    c.add_argument("--seed", type=int, default=20260910)
    c.add_argument("--out", default=str(HERE / "calibration" / "null_custom.json"))
    c.set_defaults(func=cmd_calibrate)

    d = sub.add_parser("demo", help="score the bundled example factors")
    d.add_argument("--fast", action="store_true", help="skip G5")
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("bundle", help="describe the panel")
    b.set_defaults(func=cmd_bundle)

    p = sub.add_parser("capabilities", help="list capability nodes runnable on this panel")
    p.add_argument("--index", default=None, help="path to a capability index snapshot")
    p.add_argument("--rejected", action="store_true", help="also show what could not run and why")
    p.set_defaults(func=cmd_capabilities)

    g = sub.add_parser("stages", help="print the pipeline stages and their I/O formats")
    g.set_defaults(func=cmd_stages)

    x = sub.add_parser("compare", help="run the full chain and compare against baselines")
    x.add_argument("--source", default="capability", choices=("capability", "alpha101"))
    x.add_argument("--k", type=int, default=None, help="sample k candidates; default uses all")
    x.add_argument("--seed", type=int, default=20260917)
    x.add_argument("--h", type=int, default=3, help="primary horizon in days")
    x.add_argument("--cost", type=float, default=6.5, help="one-way cost in bp")
    x.add_argument("--entry-lag", type=int, default=2)
    x.add_argument("--rebalance", type=int, default=3)
    x.add_argument("--cap", type=float, default=0.35)
    x.add_argument("--max-corr", type=float, default=0.8)
    x.add_argument("--benchmark", default="BTCUSDT")
    x.add_argument("--calibration", help="matching calibration JSON from calibrate")
    x.add_argument("--splits", help="JSON mapping of dev/val/oot to [start, end]")
    x.add_argument("--no-dedupe", action="store_true", help="keep near-duplicate factors")
    x.add_argument("--no-confirm", action="store_true", help="skip the full-matrix second pass")
    x.add_argument("--no-ic-floor", action="store_true", help="admit regardless of dev IC")
    x.add_argument("--out", default=None, help="directory for comparison.md and comparison.json")
    x.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
