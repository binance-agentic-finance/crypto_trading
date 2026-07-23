"""Command-line interface for the evolution loop.

Subcommands::

    init      — create session + download data + IS/OOS split
    inject    — load initial population from JSON
    step      — backtest current pop → fitness → selection → crossover →
                 emit gen_NNN_results.json
    mutate    — apply LLM-emitted mutations.json on the latest population
    validate  — run OOS on top-N → emit oos_results.json
    status    — print summary of the session
    export    — emit champion .py files + REPORT.md

Usage example::

    python -m cyqnt_trd.evolve.cli init --session xrp_15m_run1 \\
        --symbol XRPUSDT --interval 15m --days 90 --population-size 50

    python -m cyqnt_trd.evolve.cli inject --session xrp_15m_run1 \\
        --population initial_pop.json

    python -m cyqnt_trd.evolve.cli step --session xrp_15m_run1
    # → writes gen_001_results.json

    python -m cyqnt_trd.evolve.cli mutate --session xrp_15m_run1 \\
        --mutations mutations_001.json
    # → applies LLM mutations into populations/gen_001.json

    # ... repeat step/mutate 20 times ...

    python -m cyqnt_trd.evolve.cli validate --session xrp_15m_run1
    python -m cyqnt_trd.evolve.cli export --session xrp_15m_run1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from .data_loader import load_or_download, split_is_oos
from .engine import (
    EvolutionConfig,
    apply_llm_mutations,
    build_generation_result,
    build_next_generation,
    evaluate_population,
)
from .oos import pick_top_n_by_fitness, run_oos_validation
from .population import Population
from .session import (
    SessionConfig,
    init_session,
    load_config,
    load_mutations,
    load_population,
    save_generation_result,
    save_oos_results,
    save_population,
    session_dir,
    write_config,
)

logger = logging.getLogger(__name__)


# ── Default sessions root ─────────────────────────────────────────────────

DEFAULT_SESSIONS_ROOT = Path(
    ".codemax/spec-workflow/specs/evolutionary-strategy-discovery/sessions"
).resolve()


def _resolve_root(arg: Optional[str]) -> Path:
    return Path(arg).resolve() if arg else DEFAULT_SESSIONS_ROOT


# ── init ──────────────────────────────────────────────────────────────────

def cmd_init(args) -> int:
    root = _resolve_root(args.sessions_root)
    sdir = session_dir(root, args.session)
    if sdir.exists() and not args.force:
        print(f"[error] session already exists: {sdir}", file=sys.stderr)
        return 2
    cfg = init_session(
        root=root,
        session_id=args.session,
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
        is_days=args.is_days,
        oos_days=args.oos_days,
        population_size=args.population_size,
        seed=args.seed,
    )
    print(f"[init] created session at {sdir}")

    # Download data
    print(f"[init] downloading {args.days}d of {args.symbol} {args.interval} ...")
    df = load_or_download(
        session_dir=sdir,
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
        refresh=args.refresh,
    )
    df_is, df_oos = split_is_oos(df, is_days=args.is_days, oos_days=args.oos_days)
    print(
        f"[init] total bars={len(df)}; IS={len(df_is)} ({df_is.index[0]} → {df_is.index[-1]}), "
        f"OOS={len(df_oos)} ({df_oos.index[0]} → {df_oos.index[-1]})"
    )
    print(f"[init] config: {sdir / 'config.json'}")
    return 0


# ── inject ────────────────────────────────────────────────────────────────

def cmd_inject(args) -> int:
    root = _resolve_root(args.sessions_root)
    sdir = session_dir(root, args.session)
    cfg = load_config(sdir)

    raw = json.loads(Path(args.population).read_text())
    # Accept either {"population": [...]} or {"genomes": [...]}
    items = raw.get("population", raw.get("genomes", []))
    if not items:
        print("[error] population JSON must contain 'population' or 'genomes' list",
              file=sys.stderr)
        return 2

    pop = Population(generation=0)
    for i, gd in enumerate(items):
        from .genome import StrategyGenome, cap_max_bars
        # Auto-fill genome_id if missing
        if "genome_id" not in gd:
            gd["genome_id"] = f"gen0_{i:03d}"
        gd.setdefault("generation", 0)
        g = StrategyGenome.from_dict(gd)
        cap_max_bars(g)
        pop.add(g)

    # Validate
    errs = pop.validate()
    if errs:
        print(f"[warn] {len(errs)} invalid genomes:")
        for gid, es in errs.items():
            print(f"    {gid}: {es}")
        if not args.allow_invalid:
            print("[error] use --allow-invalid to skip them; aborting", file=sys.stderr)
            return 2
        # drop invalid
        pop.genomes = [g for g in pop.genomes if g.genome_id not in errs]

    save_population(sdir, pop)
    cfg.current_generation = 0
    write_config(sdir, cfg)
    print(f"[inject] saved gen_000.json with {len(pop)} genomes")
    print(f"[inject] species: {pop.species_distribution()}")
    return 0


# ── step ──────────────────────────────────────────────────────────────────

def cmd_step(args) -> int:
    root = _resolve_root(args.sessions_root)
    sdir = session_dir(root, args.session)
    cfg = load_config(sdir)

    # Load latest population
    cur_gen = cfg.current_generation
    pop = load_population(sdir, cur_gen)
    if not pop.genomes:
        print(f"[error] generation {cur_gen} is empty", file=sys.stderr)
        return 2

    # Load IS data
    df = load_or_download(
        session_dir=sdir, symbol=cfg.symbol, interval=cfg.interval, days=cfg.days,
    )
    df_is, _ = split_is_oos(df, is_days=cfg.is_days, oos_days=cfg.oos_days)

    print(f"[step] gen {cur_gen}: evaluating {len(pop)} genomes on IS ({len(df_is)} bars) ...")
    t0 = time.perf_counter()
    scored = evaluate_population(pop, df_is, cfg.evolution)
    elapsed_eval = time.perf_counter() - t0
    print(f"[step]   evaluation done in {elapsed_eval:.2f}s")

    # Build next gen pool (selection + crossover + random mutation)
    next_gen_id = cur_gen + 1
    next_pop, culled, xover, rmut = build_next_generation(
        scored=scored, config=cfg.evolution, generation=next_gen_id,
    )
    elapsed_total = time.perf_counter() - t0

    # Save the next-gen pool (overwriteable later by mutate command)
    save_population(sdir, next_pop)

    # Build & save result JSON for the LLM
    result = build_generation_result(
        generation=next_gen_id,  # the LLM is mutating into generation N+1
        symbol=cfg.symbol, interval=cfg.interval,
        scored=scored, next_pop=next_pop, culled=culled,
        crossover_ids=xover, mutation_ids=rmut,
        elapsed=elapsed_total, config=cfg.evolution,
    )
    # The result describes the EVALUATION of cur_gen; tag accordingly
    result.generation = cur_gen
    res_dict = result.to_dict()
    save_generation_result(sdir, cur_gen, res_dict)

    # Update config
    cfg.current_generation = next_gen_id
    if cur_gen not in cfg.completed_generations:
        cfg.completed_generations.append(cur_gen)
    write_config(sdir, cfg)

    print(f"[step]   best fitness = {result.best_fitness:.3f}")
    print(f"[step]   avg  fitness = {result.avg_fitness:.3f}")
    print(f"[step]   species: {result.species_distribution}")
    print(f"[step]   next gen pool size = {len(next_pop)} "
          f"(survivors={len(result.survivors)}, "
          f"crossover={len(xover)}, random_mut={len(rmut)})")
    print(f"[step]   need {result.need_mutations} LLM mutations for gen {next_gen_id}")
    print(f"[step] wrote gen_{cur_gen:03d}_results.json")
    return 0


# ── mutate ────────────────────────────────────────────────────────────────

def cmd_mutate(args) -> int:
    root = _resolve_root(args.sessions_root)
    sdir = session_dir(root, args.session)
    cfg = load_config(sdir)

    # The CURRENT generation (after step) is where mutations apply
    cur_gen = cfg.current_generation
    pop = load_population(sdir, cur_gen)

    raw = json.loads(Path(args.mutations).read_text())
    mutations = raw.get("mutations", raw if isinstance(raw, list) else [])

    added, errors = apply_llm_mutations(
        pop, mutations, generation=cur_gen, config=cfg.evolution,
    )
    save_population(sdir, pop)
    print(f"[mutate] applied {len(added)} mutations into gen {cur_gen} "
          f"(pool size now {len(pop)})")
    if errors:
        print(f"[mutate] {len(errors)} mutation errors:")
        for e in errors:
            print(f"    {e}")
    return 0


# ── validate ──────────────────────────────────────────────────────────────

def cmd_validate(args) -> int:
    root = _resolve_root(args.sessions_root)
    sdir = session_dir(root, args.session)
    cfg = load_config(sdir)

    cur_gen = cfg.current_generation
    pop = load_population(sdir, cur_gen)

    # Need fitness history to pick top-N. Use last entry of fitness_history.
    scored_map = {g.genome_id: (g.fitness_history[-1] if g.fitness_history else float("-inf"))
                  for g in pop.genomes}
    top = pick_top_n_by_fitness(pop, scored_map, n=args.top_n)
    if not top:
        print("[error] no candidates found in population", file=sys.stderr)
        return 2
    print(f"[validate] top-{args.top_n} candidates by IS fitness:")
    for g in top:
        f = scored_map.get(g.genome_id, float("nan"))
        print(f"    {g.genome_id} ({g.species}) fitness={f:.3f}")

    # Load full data, split
    df = load_or_download(
        session_dir=sdir, symbol=cfg.symbol, interval=cfg.interval, days=cfg.days,
    )
    df_is, df_oos = split_is_oos(df, is_days=cfg.is_days, oos_days=cfg.oos_days)
    print(f"[validate] IS bars={len(df_is)}, OOS bars={len(df_oos)}")

    result = run_oos_validation(
        candidates=top,
        df_is=df_is,
        df_oos=df_oos,
        config=cfg.evolution,
        top_n=args.top_n,
    )
    save_oos_results(sdir, result)

    print(f"[validate] champions ({len(result['champions'])}/{args.top_n}):")
    for c in result["candidates"]:
        flag = "✓" if c["passes_degradation_check"] else "✗"
        print(f"    {flag} {c['genome_id']:<28} "
              f"IS sharpe={c['is_sharpe']:.2f}, OOS sharpe={c['oos_sharpe']:.2f}, "
              f"OOS ret={c['oos_metrics'].get('total_return', 0):.2%}, "
              f"OOS trades={c['oos_metrics'].get('trade_count', 0)}")
        if c.get("degradation_reason"):
            print(f"        reason: {c['degradation_reason']}")
    return 0


# ── status ────────────────────────────────────────────────────────────────

def cmd_status(args) -> int:
    root = _resolve_root(args.sessions_root)
    sdir = session_dir(root, args.session)
    cfg = load_config(sdir)
    print(f"session: {cfg.session_id}")
    print(f"  symbol/interval: {cfg.symbol} {cfg.interval}")
    print(f"  days={cfg.days} (IS={cfg.is_days}, OOS={cfg.oos_days})")
    print(f"  current generation: {cfg.current_generation}")
    print(f"  completed: {cfg.completed_generations}")
    print(f"  population_size: {cfg.evolution.population_size}")
    return 0


# ── export ────────────────────────────────────────────────────────────────

def cmd_export(args) -> int:
    from .export import export_champions
    from .report import write_report

    root = _resolve_root(args.sessions_root)
    sdir = session_dir(root, args.session)
    n_files = export_champions(sdir)
    rpt = write_report(sdir)
    print(f"[export] wrote {n_files} champion .py files in {sdir / 'champions'}")
    print(f"[export] report: {rpt}")
    return 0


# ── diagnose ──────────────────────────────────────────────────────────────

def cmd_diagnose(args) -> int:
    """Run Phase A diagnostic on one or more genomes from the latest gen."""
    import json

    from .diagnostic import diagnose_genome, format_diagnosis

    root = _resolve_root(args.sessions_root)
    sdir = session_dir(root, args.session)
    cfg = load_config(sdir)

    # Pick the population to diagnose against
    target_gen = args.gen if args.gen is not None else cfg.current_generation
    pop = load_population(sdir, target_gen)

    # Filter to requested ids (or use top-N from latest results)
    if args.genome_id:
        targets = [g for g in pop.genomes if g.genome_id in set(args.genome_id)]
        missing = set(args.genome_id) - {g.genome_id for g in targets}
        for m in missing:
            print(f"[warn] genome {m} not found in gen {target_gen}", file=sys.stderr)
    else:
        # Use last results JSON to pick top-N
        results_files = sorted(sdir.glob("gen_*_results.json"))
        if not results_files:
            print("[error] no gen_*_results.json — run `step` first", file=sys.stderr)
            return 2
        last_results = json.loads(results_files[-1].read_text())
        top_ids = [t["genome_id"] for t in last_results.get("top_5", [])][: args.top_n]
        targets = [g for g in pop.genomes if g.genome_id in set(top_ids)]
        if not targets:
            # Fallback: maybe top-N got carried over to next gen — search across all populations
            for f in sorted(sdir.glob("populations/gen_*.json"), reverse=True):
                p = Population.from_dict(json.loads(f.read_text()))
                found = [g for g in p.genomes if g.genome_id in set(top_ids)]
                if found:
                    targets = found
                    break

    if not targets:
        print("[error] no genomes selected to diagnose", file=sys.stderr)
        return 2

    # Load IS data
    df = load_or_download(
        session_dir=sdir, symbol=cfg.symbol, interval=cfg.interval, days=cfg.days,
    )
    if args.use_oos:
        _, df_eval = split_is_oos(df, is_days=cfg.is_days, oos_days=cfg.oos_days)
        slice_label = "OOS"
    else:
        df_eval, _ = split_is_oos(df, is_days=cfg.is_days, oos_days=cfg.oos_days)
        slice_label = "IS"

    print(f"[diagnose] running Phase A on {len(targets)} genome(s) over {slice_label} "
          f"({len(df_eval)} bars)")

    # Output dir for JSON dumps
    out_dir = sdir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for g in targets:
        try:
            diag = diagnose_genome(
                g, df_eval,
                auto_opt_path=args.auto_opt_path,
                fee_bps=cfg.evolution.fee_bps,
                slippage_bps=cfg.evolution.slippage_bps,
                initial_capital=cfg.evolution.initial_capital,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] diagnose {g.genome_id} threw: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        # Save JSON
        out_file = out_dir / f"{g.genome_id}_{slice_label}.json"
        out_file.write_text(json.dumps(diag, indent=2))
        results.append((g.genome_id, out_file))

        # Print pretty
        print()
        print(format_diagnosis(diag))
        print()
        print(f"  → wrote {out_file.relative_to(sdir)}")

    if not results:
        return 1

    print()
    print(f"[diagnose] done. {len(results)} report(s) in {out_dir}")
    return 0


# ── argparse plumbing ─────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cyqnt_trd.evolve.cli")
    p.add_argument("--sessions-root", default=None,
                   help=f"Override sessions root (default: {DEFAULT_SESSIONS_ROOT})")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init")
    sp.add_argument("--session", required=True)
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--interval", required=True, choices=["5m", "15m", "1h"])
    sp.add_argument("--days", type=int, default=90)
    sp.add_argument("--is-days", type=int, default=60)
    sp.add_argument("--oos-days", type=int, default=30)
    sp.add_argument("--population-size", type=int, default=50)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--refresh", action="store_true")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("inject")
    sp.add_argument("--session", required=True)
    sp.add_argument("--population", required=True, help="path to initial population JSON")
    sp.add_argument("--allow-invalid", action="store_true")
    sp.set_defaults(func=cmd_inject)

    sp = sub.add_parser("step")
    sp.add_argument("--session", required=True)
    sp.set_defaults(func=cmd_step)

    sp = sub.add_parser("mutate")
    sp.add_argument("--session", required=True)
    sp.add_argument("--mutations", required=True)
    sp.set_defaults(func=cmd_mutate)

    sp = sub.add_parser("validate")
    sp.add_argument("--session", required=True)
    sp.add_argument("--top-n", type=int, default=5)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("status")
    sp.add_argument("--session", required=True)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("export")
    sp.add_argument("--session", required=True)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("diagnose",
                        help="Phase A: regime/vol/time-segmented diagnosis "
                             "of a genome via auto-optimize")
    sp.add_argument("--session", required=True)
    sp.add_argument("--genome-id", action="append",
                    help="genome_id to diagnose (repeatable). "
                         "If omitted, diagnoses the top-N from latest gen results.")
    sp.add_argument("--top-n", type=int, default=3,
                    help="how many top genomes to diagnose when --genome-id is omitted")
    sp.add_argument("--gen", type=int, default=None,
                    help="which generation pool to look up (default: current_generation)")
    sp.add_argument("--use-oos", action="store_true",
                    help="diagnose against OOS slice instead of IS")
    sp.add_argument("--auto-opt-path", default=None,
                    help="path to the auto-optimize 'src' directory "
                         "(default: the AUTO_OPT_SRC environment variable, if set)")
    sp.set_defaults(func=cmd_diagnose)

    return p


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
