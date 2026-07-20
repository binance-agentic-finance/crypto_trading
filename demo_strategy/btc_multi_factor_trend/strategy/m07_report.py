"""Module ⑦ REPORT — human-readable text report from a pipeline_result.

The JSON is the machine-readable output. This module produces the
banner + leaderboard + tier breakdown printed to stdout, so an operator
can see at a glance what happened.
"""
from __future__ import annotations

from datetime import datetime, timezone

from demo_strategy._shared.fmt import fmt_pct, fmt_price, fmt_scalar, banner


def render_report(pipeline_result: dict) -> str:
    lines = []
    mode = pipeline_result.get("mode", "balanced")
    ts   = pipeline_result.get("generated_at", "")
    lines.append(banner(f"BTC MULTI-FACTOR TREND · mode={mode} · {ts}"))
    lines.append("")

    results = sorted(pipeline_result["results"],
                     key=lambda r: -r["score"]["total"])

    # ── Header row ──
    lines.append(f"  {'SYMBOL':<10s} {'PRICE':>12s}  {'SCORE':>5s}  "
                 f"{'VERDICT':<18s} {'DIR':<6s} {'SIZE(USDT)':>10s} {'STOP':>8s}")
    lines.append("  " + "-" * 82)
    for r in results:
        d = r["decision"]
        lines.append(
            f"  {r['symbol']:<10s} {fmt_price(r['price']):>12s}  "
            f"{r['score']['total']:>5d}  {r['score']['verdict']:<18s} "
            f"{d.get('direction','WATCH'):<6s} {fmt_scalar(d.get('size_usdt'),1):>10s} "
            f"{fmt_price(d.get('stop_price') or 0):>8s}"
        )
    lines.append("")

    # ── Tier breakdown per row ──
    for r in results:
        lines.append(f"  {r['symbol']} · tier breakdown:")
        for t in r["score"]["tiers"]:
            lines.append(f"    {t['name']:<15s} {t['score']:>+3d}   {t['reason']}")
        # decision detail
        d = r["decision"]
        lines.append(f"    → direction={d.get('direction')}, "
                     f"lev={d.get('leverage')}, "
                     f"stop%={d.get('stop_pct')}, "
                     f"risk={d.get('risk_usdt')} USDT")
        if "execution" in r:
            lines.append(f"    ↳ execution: {r['execution']}")
        lines.append("")

    # ── Execution summary ──
    exe = pipeline_result.get("execution") or {}
    if not exe.get("skipped"):
        lines.append(banner(
            f"EXECUTION · live={exe.get('live')} · filled={exe.get('count_filled')}",
            char="-"))

    return "\n".join(lines)
