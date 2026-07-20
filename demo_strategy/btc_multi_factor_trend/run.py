#!/usr/bin/env python3
"""BTC Multi-Factor Trend — CLI entry point.

Wires up config load + CLI overlay + calls `strategy.run(cfg)`, then
writes JSON + prints human-readable report.

Usage:
    python3 run.py                             # balanced, dry-run
    python3 run.py --mode aggressive           # tighter stops, higher lev
    python3 run.py --symbols BTCUSDT ETHUSDT   # add ETH
    python3 run.py --execute --live            # real orders
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC Multi-Factor Trend pipeline")
    p.add_argument("--mode", default="balanced",
                   choices=["defensive", "balanced", "aggressive"])
    p.add_argument("--symbols", nargs="+", default=None,
                   help="override config.symbols")
    p.add_argument("--delay", type=float, default=0.3,
                   help="seconds between REST calls")
    p.add_argument("--profile", default=None, help="binance-cli profile")
    p.add_argument("--execute", action="store_true",
                   help="enable order dispatch (dry-run unless --live)")
    p.add_argument("--live", action="store_true",
                   help="send REAL orders (requires --execute)")
    p.add_argument("--config",
                   default=str(Path(__file__).parent / "config" / "config.json"))
    p.add_argument("--out", default=None,
                   help="override output JSON path")
    p.add_argument("--min-verdict", default=None,
                   choices=["STRONG_CANDIDATE", "CANDIDATE"],
                   help="override execution.min_verdict")
    return p.parse_args()


def main() -> int:
    args = _cli()

    # ── bootstrap sys.path so atomic_strategy_lib + _shared resolve ──
    THIS = Path(__file__).resolve().parent
    # add parent so `from demo_strategy._shared…` works
    sys.path.insert(0, str(THIS.parent.parent))
    from demo_strategy._shared import (
        bootstrap_sys_path, workspace_dir,
        load_cfg, apply_cli_overrides, activate_mode,
    )
    bootstrap_sys_path()

    # add THIS/strategy for the pipeline module
    sys.path.insert(0, str(THIS))
    from strategy import run, render_report  # noqa: E402

    # ── config load + overlay + activate mode ──
    cfg = load_cfg(args.config)
    cfg = apply_cli_overrides(cfg, {
        "symbols":                args.symbols,
        "_delay_sec":             args.delay,
        "_profile":               args.profile,
        "execution.enabled":      True if args.execute else None,
        "execution.live":         True if args.live    else None,
        "execution.min_verdict":  args.min_verdict,
    })
    activate_mode(cfg, args.mode)

    # ── run pipeline ──
    result = run(cfg)

    # ── write JSON + print report ──
    out_path = Path(args.out) if args.out else \
        workspace_dir("btc_multi_factor_trend") / "pipeline_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                   default=str),
                        encoding="utf-8")

    print(render_report(result))
    print(f"\n  ↳ pipeline_result.json → {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
