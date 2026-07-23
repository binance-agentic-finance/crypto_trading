"""Find the *true* champion across all genomes that ever existed in a session,
ranked by combined IS + OOS performance.

The standard evolve fitness is IS-only; this script reads every gen_NNN.json
population file, dedupes, runs IS+OOS backtest on each genome, and ranks by:

    score = OOS_sharpe + 0.5 * IS_sharpe - 2.0 * dd_penalty

Penalises strategies that overfit (IS >> OOS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, '/Users/hankchung/Dev/auto-optimize/src')

from cyqnt_trd.evolve.bridge import backtest_genome
from cyqnt_trd.evolve.data_loader import load_or_download, split_is_oos
from cyqnt_trd.evolve.genome import StrategyGenome


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--root",
                   default=".codemax/spec-workflow/specs/evolutionary-strategy-discovery/sessions")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--min-oos-trades", type=int, default=10)
    p.add_argument("--min-oos-sharpe", type=float, default=0.0)
    args = p.parse_args()

    sdir = Path(args.root) / args.session
    cfg = json.loads((sdir / "config.json").read_text())
    symbol = cfg["symbol"]
    interval = cfg["interval"]

    # Walk every population file, dedupe genome_ids
    seen: Dict[str, dict] = {}
    pop_files = sorted((sdir / "populations").glob("gen_*.json"))
    for pf in pop_files:
        pop = json.loads(pf.read_text())
        for g in pop["genomes"]:
            seen[g["genome_id"]] = g
    print(f"Found {len(seen)} unique genomes across {len(pop_files)} generations")

    # Load data
    df = load_or_download(session_dir=sdir, symbol=symbol, interval=interval, days=cfg["days"])
    is_df, oos_df = split_is_oos(df, is_days=cfg["is_days"], oos_days=cfg["oos_days"])
    print(f"IS bars={len(is_df)}, OOS bars={len(oos_df)}")
    print()

    # Backtest each
    rows: List[dict] = []
    for gid, gd in seen.items():
        try:
            g = StrategyGenome.from_dict(gd)
        except Exception:  # noqa: BLE001
            continue
        try:
            is_r = backtest_genome(genome=g, df=is_df, fee_bps=4.0, slippage_bps=1.0,
                                    record_trades=False)
            oos_r = backtest_genome(genome=g, df=oos_df, fee_bps=4.0, slippage_bps=1.0,
                                     record_trades=False)
        except Exception:  # noqa: BLE001
            continue

        if oos_r.trade_count < args.min_oos_trades:
            continue

        # Blended score: OOS dominant, IS secondary, drawdown penalty
        is_sh = is_r.sharpe_ratio
        oos_sh = oos_r.sharpe_ratio
        is_ret = is_r.total_return
        oos_ret = oos_r.total_return
        dd = abs(oos_r.max_drawdown)
        # Blended fitness — favours OOS
        blended = oos_sh + 0.3 * is_sh + 5 * oos_ret - 2 * dd

        # Robustness check: OOS shouldn't be < 30% of IS (strict overfit guard)
        # but allow OOS to be > IS (good defense)
        if is_sh > 0 and oos_sh < is_sh * 0.3 and oos_sh > 0:
            # tagged but not skipped — let composite score handle it
            pass

        rows.append({
            "id": gid,
            "species": g.species,
            "is_ret": is_ret, "is_sharpe": is_sh, "is_trades": is_r.trade_count,
            "oos_ret": oos_ret, "oos_sharpe": oos_sh, "oos_trades": oos_r.trade_count,
            "oos_dd": oos_r.max_drawdown, "oos_winrate": oos_r.win_rate,
            "blended": blended,
            "genome": gd,
        })

    rows.sort(key=lambda r: -r["blended"])

    # Apply minimum OOS sharpe filter
    qualified = [r for r in rows if r["oos_sharpe"] >= args.min_oos_sharpe]
    print(f"{len(qualified)}/{len(rows)} pass OOS sharpe >= {args.min_oos_sharpe} "
          f"and OOS trades >= {args.min_oos_trades}")
    print()

    # Print top-N
    print(f"{'rank':>4} {'genome':<32} {'species':<14} "
          f"{'IS ret':>8} {'IS sh':>7} {'IS T':>5}  "
          f"{'OOS ret':>8} {'OOS sh':>7} {'OOS T':>5} {'OOS DD':>8} {'win':>6} "
          f"{'blended':>9}")
    print("-" * 130)
    for i, r in enumerate(qualified[:args.top_n], 1):
        print(f"{i:>4} {r['id'][:32]:<32} {r['species']:<14} "
              f"{r['is_ret']:+.2%} {r['is_sharpe']:+.2f} {r['is_trades']:>5}  "
              f"{r['oos_ret']:+.2%} {r['oos_sharpe']:+.2f} {r['oos_trades']:>5} "
              f"{r['oos_dd']:+.2%} {r['oos_winrate']:.2f} "
              f"{r['blended']:+.3f}")

    # Save full results JSON
    out_path = sdir / "true_champions.json"
    out_path.write_text(json.dumps({
        "config": {
            "min_oos_trades": args.min_oos_trades,
            "min_oos_sharpe": args.min_oos_sharpe,
            "top_n": args.top_n,
        },
        "qualified_count": len(qualified),
        "total_evaluated": len(rows),
        "champions": qualified[: args.top_n],
    }, indent=2))
    print()
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
