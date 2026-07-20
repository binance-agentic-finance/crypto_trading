"""Module ⑦ REPORT — social-hotspot-brief format.

Shows attention data alongside the technical score so a reader can see
"why this token surfaced" vs "does it also look technically sound".
"""
from __future__ import annotations

from demo_strategy._shared.fmt import fmt_pct, fmt_price, fmt_scalar, banner


def render_report(pipeline_result: dict) -> str:
    lines = []
    ts = pipeline_result.get("generated_at", "")
    n  = pipeline_result.get("candidates", 0)
    lines.append(banner(f"BINANCE SQUARE BUZZ · {n} candidates · {ts}"))
    lines.append("")

    results = sorted(pipeline_result["results"],
                     key=lambda r: -r["score"]["total"])

    lines.append(f"  {'SYMBOL':<10s} {'PRICE':>10s}  {'SCORE':>5s}  "
                 f"{'VERDICT':<18s} {'DIR':<6s} {'LOC':>4s} {'SEC':>4s} {'DEEP':>5s}")
    lines.append("  " + "-" * 76)
    for r in results:
        att = r["attention"]
        lines.append(
            f"  {r['symbol']:<10s} {fmt_price(r['price']):>10s}  "
            f"{r['score']['total']:>5d}  {r['score']['verdict']:<18s} "
            f"{r['decision'].get('direction','WATCH'):<6s} "
            f"{len(att.get('locales', [])):>4d} "
            f"{len(att.get('sections', [])):>4d} "
            f"{att.get('deep_mentions', 0):>5d}"
        )
    lines.append("")

    for r in results:
        lines.append(f"  {r['symbol']} · tier breakdown:")
        for t in r["score"]["tiers"]:
            lines.append(f"    {t['name']:<22s} {t['score']:>+3d}   {t['reason']}")
        d = r["decision"]
        if d.get("direction") == "WATCH":
            lines.append(f"    → WATCH ({d.get('reason','')})")
        else:
            lines.append(f"    → direction={d['direction']}, "
                         f"size={fmt_scalar(d.get('size_usdt'), 1)} USDT, "
                         f"stop%={d.get('stop_pct')}")
        if "execution" in r:
            lines.append(f"    ↳ execution: {r['execution']}")
        lines.append("")

    exe = pipeline_result.get("execution") or {}
    if not exe.get("skipped"):
        lines.append(banner(
            f"EXECUTION · live={exe.get('live')} · filled={exe.get('count_filled')}",
            char="-"))
    return "\n".join(lines)
