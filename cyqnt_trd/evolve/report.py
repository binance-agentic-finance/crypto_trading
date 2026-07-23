"""Report writer — emits REPORT.md with full evolution + OOS summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def write_report(session_dir: Path) -> Path:
    """Generate REPORT.md inside the session directory."""
    cfg = json.loads((session_dir / "config.json").read_text())
    symbol = cfg["symbol"]
    interval = cfg["interval"]

    # Collect generation results
    gen_files = sorted(session_dir.glob("gen_*_results.json"))
    oos_path = session_dir / "oos_results.json"
    oos = json.loads(oos_path.read_text()) if oos_path.exists() else None

    lines: List[str] = []
    lines.append(f"# Evolution Report — {symbol} {interval}")
    lines.append("")
    lines.append(f"- session: `{cfg['session_id']}`")
    lines.append(f"- created: {cfg.get('created_at', '?')}")
    lines.append(f"- days: {cfg['days']} (IS={cfg['is_days']}, OOS={cfg['oos_days']})")
    lines.append(f"- population size: {cfg['evolution']['population_size']}")
    lines.append(f"- generations evaluated: {len(gen_files)}")
    lines.append("")

    # ── Fitness curve ──
    if gen_files:
        lines.append("## Fitness Trajectory (best / avg per generation)")
        lines.append("")
        lines.append("| Gen | Best Fitness | Avg Fitness | Pop Size | Top species |")
        lines.append("|-----|--------------|-------------|----------|-------------|")
        for f in gen_files:
            r = json.loads(f.read_text())
            top_species = ""
            if r.get("top_5"):
                top_species = r["top_5"][0]["species"]
            lines.append(
                f"| {r['generation']} | {r['best_fitness']:.3f} | "
                f"{r['avg_fitness']:.3f} | {r['population_size']} | {top_species} |"
            )
        lines.append("")

    # ── Final species distribution ──
    if gen_files:
        last = json.loads(gen_files[-1].read_text())
        lines.append("## Final Species Distribution")
        lines.append("")
        for s, n in last.get("species_distribution", {}).items():
            lines.append(f"- **{s}**: {n}")
        lines.append("")

    # ── OOS validation ──
    if oos is not None:
        lines.append("## OOS Validation Results")
        lines.append("")
        lines.append(
            f"Degradation threshold: OOS Sharpe ≥ "
            f"{oos.get('degradation_threshold', 0.4):.2f} × IS Sharpe"
        )
        lines.append("")
        lines.append("### Top Candidates")
        lines.append("")
        lines.append("| # | genome_id | species | IS Sharpe | OOS Sharpe | OOS Return | OOS Trades | Pass |")
        lines.append("|---|-----------|---------|-----------|-------------|------------|------------|------|")
        for i, c in enumerate(oos.get("candidates", []), 1):
            sp = c["genome"].get("species", "?")
            mark = "✓" if c["passes_degradation_check"] else "✗"
            lines.append(
                f"| {i} | `{c['genome_id']}` | {sp} | "
                f"{c['is_sharpe']:.2f} | {c['oos_sharpe']:.2f} | "
                f"{c['oos_metrics'].get('total_return', 0):.2%} | "
                f"{c['oos_metrics'].get('trade_count', 0)} | {mark} |"
            )
        lines.append("")
        lines.append(f"### Champions ({len(oos.get('champions', []))})")
        lines.append("")
        for cid in oos.get("champions", []):
            lines.append(f"- `{cid}`")
        lines.append("")

        # ── Per-champion detail ──
        for c in oos.get("candidates", []):
            if not c["passes_degradation_check"]:
                continue
            lines.append("---")
            lines.append("")
            lines.append(f"### Champion `{c['genome_id']}`")
            lines.append("")
            g = c["genome"]
            lines.append(f"- species: **{g.get('species')}**")
            lines.append(f"- preferred_interval: {g.get('preferred_interval')}")
            lines.append(f"- entry_logic: {g.get('entry_logic')}")
            lines.append(f"- size: {g.get('size')}")
            lines.append("")
            lines.append("**Entry factors:**")
            for f in g.get("entry_factors", []):
                lines.append(f"- {f['indicator']} / {f['condition']} / {f.get('params', {})}")
            lines.append("")
            if g.get("filters"):
                lines.append("**Filters:**")
                for fl in g["filters"]:
                    lines.append(f"- {fl['filter_type']} / {fl.get('params', {})}")
                lines.append("")
            lines.append(f"**Exit:** `{g.get('exit_type')}` / {g.get('exit_params')}")
            lines.append("")
            lines.append("**IS metrics:**")
            for k, v in c["is_metrics"].items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.4f}")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")
            lines.append("**OOS metrics:**")
            for k, v in c["oos_metrics"].items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.4f}")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")
            if g.get("parent_ids"):
                lines.append(f"**Lineage:** parents = {g['parent_ids']}")
                lines.append("")
            if g.get("mutation_log"):
                lines.append("**Mutation log:**")
                for m in g["mutation_log"]:
                    lines.append(f"- {m}")
                lines.append("")

    out = session_dir / "REPORT.md"
    out.write_text("\n".join(lines))
    return out
