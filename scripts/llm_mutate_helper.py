"""LLM-mutate helper.

This is a *deterministic* fallback mutation generator that the LLM uses to
speed up the evolution loop. The LLM still designs the high-level logic
(targeted parameter perturbations of top performers, occasional new species
injections), but this script encodes the most common patterns and emits
mutations.json automatically.

Usage::

    python scripts/llm_mutate_helper.py \
        --session xrp_15m_run1 \
        --gen 3
    # → writes sessions/<id>/mutations_003.json

The strategy:
    - 5 mutations targeting the TOP 5 (each gets one of: add filter,
      change exit, change size, perturb param, replace factor)
    - 5 mutations injecting NEW genomes by combining best entry factors
      with best exit/filter combinations seen so far
    - 5 mutations perturbing mid-pack survivors to explore parameter space

Avoids targeting culled genomes (it reads the latest *_results.json).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

DEFAULT_ROOT = Path(".codemax/spec-workflow/specs/evolutionary-strategy-discovery/sessions")


def load_latest_results(sdir: Path):
    files = sorted(sdir.glob("gen_*_results.json"))
    if not files:
        raise SystemExit("no gen_*_results.json found yet")
    last = files[-1]
    return json.loads(last.read_text())


def load_population(sdir: Path, gen: int):
    p = sdir / "populations" / f"gen_{gen:03d}.json"
    return json.loads(p.read_text())


def gen_mutations(sdir: Path, target_gen: int, seed: int):
    rng = random.Random(seed + target_gen)
    results = load_latest_results(sdir)
    pop = load_population(sdir, target_gen)
    pop_ids = {g["genome_id"] for g in pop["genomes"]}

    top5 = results["top_5"]
    bottom5 = results["bottom_5"]

    mutations = []

    # 1) Boost top 1: add an additional vol filter or bump size
    if top5:
        leader = top5[0]
        if leader["genome_id"] in pop_ids:
            existing_filters = {f["filter_type"] for f in leader["genome"]["filters"]}
            if "atr_above_pct" not in existing_filters and len(leader["genome"]["filters"]) < 3:
                mutations.append({
                    "target_genome_id": leader["genome_id"],
                    "action": "add_filter",
                    "detail": {"filter_type": "atr_above_pct",
                               "params": {"period": 14, "threshold": rng.choice([0.003, 0.0035, 0.004])}},
                    "reasoning": f"top fitness {leader['fitness']:.2f} — add ATR vol gate",
                })
            elif leader["genome"]["size"] < 0.65:
                mutations.append({
                    "target_genome_id": leader["genome_id"],
                    "action": "change_size",
                    "detail": {"size": min(0.7, leader["genome"]["size"] + 0.1)},
                    "reasoning": "top fitness — bump exposure",
                })

    # 2-3) Replicate top 2-3 with parameter variations
    for i in range(1, min(4, len(top5))):
        cand = top5[i]
        if cand["genome_id"] not in pop_ids:
            continue
        # Pick a single numeric param in entry_factors and ±15-20%
        g = cand["genome"]
        if not g["entry_factors"]:
            continue
        # Try perturbing exit params via change_exit (preserves type)
        old_exit = g["exit_params"]
        new_exit = dict(old_exit)
        # Perturb 1-2 numeric values
        keys = [k for k in new_exit if isinstance(new_exit[k], (int, float))
                and k not in ("max_bars",)]
        if keys:
            k = rng.choice(keys)
            factor = rng.choice([0.85, 0.92, 1.08, 1.15])
            v = new_exit[k]
            new_exit[k] = max(0.0001, float(v) * factor) if isinstance(v, float) else max(1, int(round(v * factor)))
            mutations.append({
                "target_genome_id": cand["genome_id"],
                "action": "change_exit",
                "detail": {"exit_type": g["exit_type"], "exit_params": new_exit},
                "reasoning": f"perturb top-{i+1} exit param {k} ×{factor}",
            })

    # 4-5) Add a slope or regime filter to a top performer that lacks one
    for i, cand in enumerate(top5):
        if len(mutations) >= 5:
            break
        if cand["genome_id"] not in pop_ids:
            continue
        existing = {f["filter_type"] for f in cand["genome"]["filters"]}
        if "price_above_ma" not in existing and len(cand["genome"]["filters"]) < 3:
            mutations.append({
                "target_genome_id": cand["genome_id"],
                "action": "add_filter",
                "detail": {"filter_type": "price_above_ma",
                           "params": {"period": rng.choice([100, 200]), "ma_type": "ema"}},
                "reasoning": f"add long-term regime gate to top-{i+1}",
            })

    # 6-10) Inject 5 new genomes by combining top entry factors with diverse exits
    top_factors = []
    for cand in top5[:3]:
        for f in cand["genome"]["entry_factors"]:
            top_factors.append(copy.deepcopy(f))

    if len(top_factors) >= 2:
        # Combo 1: top1 entry + top2 exit
        combo = {
            "genome_id": f"hyb_g{target_gen}_combo1",
            "species": top5[0]["genome"]["species"],
            "entry_factors": [copy.deepcopy(top5[0]["genome"]["entry_factors"][0])],
            "entry_logic": "all_of",
            "exit_type": top5[1]["genome"]["exit_type"],
            "exit_params": copy.deepcopy(top5[1]["genome"]["exit_params"]),
            "filters": [
                {"filter_type": "atr_above_pct", "params": {"period": 14, "threshold": 0.003}},
                {"filter_type": "price_above_ma", "params": {"period": 100, "ma_type": "ema"}},
            ],
            "size": 0.5, "preferred_interval": "15m",
        }
        mutations.append({"action": "new_genome", "detail": combo,
                         "reasoning": "hybrid top1 entry + top2 exit + dual filter"})

    # Combo 2: dual factor (top1 + top3)
    if len(top5) >= 3:
        combo2 = {
            "genome_id": f"hyb_g{target_gen}_combo2",
            "species": "momentum",
            "entry_factors": [
                copy.deepcopy(top5[0]["genome"]["entry_factors"][0]),
                copy.deepcopy(top5[2]["genome"]["entry_factors"][0]),
            ],
            "entry_logic": "all_of",
            "exit_type": "ma_cross_exit",
            "exit_params": {"period": 13, "ma_type": "ema", "max_bars": 20},
            "filters": [
                {"filter_type": "atr_above_pct", "params": {"period": 14, "threshold": 0.003}},
            ],
            "size": 0.5, "preferred_interval": "15m",
        }
        # may need to limit to 4 factors if duplicates
        if len(combo2["entry_factors"]) <= 4:
            mutations.append({"action": "new_genome", "detail": combo2,
                             "reasoning": "dual factor: top1 + top3 entries"})

    # Always inject one fresh exploration genome (random within a known-good template)
    fresh_templates = [
        {
            "genome_id": f"explore_g{target_gen}_a",
            "species": "trend",
            "entry_factors": [{"indicator": "ema", "condition": "cross_above",
                              "params": {"fast": rng.choice([7, 9, 11]), "slow": rng.choice([21, 26, 30])}}],
            "entry_logic": "all_of",
            "exit_type": "ma_cross_exit",
            "exit_params": {"period": rng.choice([13, 21]), "ma_type": "ema", "max_bars": rng.choice([18, 22, 24])},
            "filters": [
                {"filter_type": "atr_above_pct", "params": {"period": 14, "threshold": rng.choice([0.0025, 0.003, 0.0035])}},
                {"filter_type": "price_above_ma", "params": {"period": rng.choice([100, 200]), "ma_type": "ema"}},
            ],
            "size": 0.5, "preferred_interval": "15m",
        },
        {
            "genome_id": f"explore_g{target_gen}_b",
            "species": "momentum",
            "entry_factors": [
                {"indicator": "macd", "condition": "macd_golden_cross",
                 "params": {"fast": rng.choice([8, 12]), "slow": rng.choice([21, 26]), "signal": 9}},
                {"indicator": "rsi", "condition": "above_threshold",
                 "params": {"period": 14, "threshold": rng.choice([50, 55])}},
            ],
            "entry_logic": "all_of",
            "exit_type": "atr_stop_tp",
            "exit_params": {"stop_mult": rng.choice([1.5, 2.0]), "tp_mult": rng.choice([2.5, 3.0, 3.5]),
                           "max_bars": 20, "atr_period": 14},
            "filters": [
                {"filter_type": "atr_above_pct", "params": {"period": 14, "threshold": 0.003}},
            ],
            "size": 0.5, "preferred_interval": "15m",
        },
    ]
    for fresh in fresh_templates:
        mutations.append({"action": "new_genome", "detail": fresh,
                         "reasoning": "exploration: randomized known-good template"})

    # 11-15) Perturb mid-pack survivors with random mutations on top species
    survivors = results["survivors"]
    mid_pack = [g for g in pop["genomes"] if g["genome_id"] in survivors][3:13]
    rng.shuffle(mid_pack)
    needed = max(0, 15 - len(mutations))
    for cand in mid_pack[:needed]:
        if len(cand.get("filters", [])) < 3:
            mutations.append({
                "target_genome_id": cand["genome_id"],
                "action": "add_filter",
                "detail": {"filter_type": rng.choice(["adx_above", "atr_above_pct", "ma_slope_positive"]),
                           "params": ({"period": 14, "threshold": 20} if True else {})},
                "reasoning": "explore: add filter to mid-pack survivor",
            })
        else:
            # change size
            mutations.append({
                "target_genome_id": cand["genome_id"],
                "action": "change_size",
                "detail": {"size": rng.choice([0.4, 0.5, 0.6])},
                "reasoning": "explore: vary mid-pack size",
            })

    # Cap to 15
    return mutations[:15]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--gen", type=int, required=True, help="target generation (e.g. 3 to write mutations_003.json)")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sdir = Path(args.root) / args.session
    mutations = gen_mutations(sdir, args.gen, args.seed)
    out = sdir / f"mutations_{args.gen:03d}.json"
    out.write_text(json.dumps({
        "_meta": {"generation_target": args.gen, "auto_generated": True,
                  "count": len(mutations)},
        "mutations": mutations,
    }, indent=2))
    print(f"wrote {out} with {len(mutations)} mutations")


if __name__ == "__main__":
    main()
