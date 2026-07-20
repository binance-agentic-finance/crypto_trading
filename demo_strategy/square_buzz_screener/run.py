#!/usr/bin/env python3
"""Binance Square Buzz Screener — CLI entry point.

Usage:
    python3 run.py                              # scan EN + CN, deep fetch
    python3 run.py --locales en                 # EN only
    python3 run.py --no-deep                    # skip hashtag deep fetch
    python3 run.py --execute --live             # real orders
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Square Buzz Screener pipeline")
    p.add_argument("--locales", nargs="+", default=None,
                   help="override universe.locales; e.g. --locales en zh-CN")
    p.add_argument("--no-deep", action="store_true",
                   help="skip hashtag deep-fetch (faster, less signal)")
    p.add_argument("--max-candidates", type=int, default=None)
    p.add_argument("--delay", type=float, default=0.3)
    p.add_argument("--profile", default=None)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--live", action="store_true")
    p.add_argument("--min-verdict", default=None,
                   choices=["STRONG_CANDIDATE", "CANDIDATE"])
    p.add_argument("--config",
                   default=str(Path(__file__).parent / "config" / "config.json"))
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = _cli()

    THIS = Path(__file__).resolve().parent
    sys.path.insert(0, str(THIS.parent.parent))
    from demo_strategy._shared import (
        bootstrap_sys_path, workspace_dir,
        load_cfg, apply_cli_overrides, activate_mode,
    )
    bootstrap_sys_path()

    sys.path.insert(0, str(THIS))
    from strategy import run, render_report  # noqa: E402

    cfg = load_cfg(args.config)
    cfg = apply_cli_overrides(cfg, {
        "universe.locales":              args.locales,
        "universe.hashtag_deep_fetch":   False if args.no_deep else None,
        "universe.max_candidates":       args.max_candidates,
        "_delay_sec":                    args.delay,
        "_profile":                      args.profile,
        "execution.enabled":             True if args.execute else None,
        "execution.live":                True if args.live    else None,
        "execution.min_verdict":         args.min_verdict,
    })
    activate_mode(cfg, "default")

    result = run(cfg)

    out_path = Path(args.out) if args.out else \
        workspace_dir("square_buzz_screener") / "pipeline_result.json"
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
