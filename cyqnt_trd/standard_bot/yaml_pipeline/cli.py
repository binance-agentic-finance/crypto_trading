"""CLI for the YAML strategy pipeline.

    python -m cyqnt_trd.standard_bot.yaml_pipeline validate strategy.yaml
    python -m cyqnt_trd.standard_bot.yaml_pipeline run      strategy.yaml [options]

``run`` dispatches on ``run.mode``:

* ``backtest`` — registers the strategy and runs ``mvp_backtest --engine python``
  in-process (works fully offline with ``--input-json``).
* ``paper``    — builds the documented ``mvp_paper_daemon --engine python``
  command; prints it, and only spawns it when ``--start`` is given.
* ``live``     — enforces the trading-modes safety rules (paper-stage first,
  time-bounded session, ``max_notional`` cap) and prints the two-process
  daemon + ``mvp_live_executor`` sequence (dry-run first). It NEVER places a
  real order itself; a human runs the printed executor command after CONFIRM.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

DAEMON_LOADER = "cyqnt_trd.standard_bot.yaml_pipeline._daemon_loader"

from .interpreter import SpecError, build_make_signals
from .spec import load_spec, register_from_yaml, validate_spec


def _data_source_argv(spec: Dict[str, Any], overrides: argparse.Namespace) -> List[str]:
    """Map ``data.source`` (or CLI overrides) to backtest data flags."""
    if getattr(overrides, "input_json", None):
        return ["--input-json", overrides.input_json]
    source = (spec.get("data") or {}).get("source") or {}
    stype = source.get("type", "binance_rest")
    if stype == "input_json":
        return ["--input-json", source["path"]]
    if stype == "historical_parquet":
        argv = ["--historical-dir", source.get("dir", "data/historical")]
        if source.get("storage_timeframe"):
            argv += ["--storage-timeframe", source["storage_timeframe"]]
        if source.get("download_missing"):
            argv.append("--download-missing")
        if source.get("start_ts"):
            argv += ["--start-ts", str(source["start_ts"])]
        if source.get("end_ts"):
            argv += ["--end-ts", str(source["end_ts"])]
        return argv
    if stype == "binance_rest":
        return ["--allow-remote-api"]
    return []


def _common_run_fields(spec: Dict[str, Any]):
    data = spec.get("data") or {}
    symbol = data["symbol"]
    interval = (data.get("primary") or {})["interval"]
    market_type = data.get("market_type", "futures")
    fees = (spec.get("risk") or {}).get("fees") or {}
    commission_bps = fees.get("commission_bps", 4.0)
    slippage_bps = fees.get("slippage_bps", 2.0)
    return symbol, interval, market_type, commission_bps, slippage_bps


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        spec = load_spec(args.spec)
    except SpecError as exc:
        print(f"LOAD ERROR: {exc}")
        return 2
    errors, warnings = validate_spec(spec)
    for w in warnings:
        print(f"  warning: {w}")
    if errors:
        print(f"INVALID ({len(errors)} error(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1
    sid = (spec.get("strategy") or {}).get("id", "?")
    print(f"OK: spec '{sid}' is valid and dry-ran successfully on synthetic data.")
    if warnings:
        print(f"    ({len(warnings)} warning(s) above)")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _run_backtest_vectorized(spec: Dict[str, Any], args: argparse.Namespace) -> int:
    """Backtest via the vectorized engine (long+short, matches paper/live)."""
    from pathlib import Path

    from ._data import load_ohlcv
    from ..simulation.vectorized_backtest import run_vectorized_backtest

    df, src = load_ohlcv(spec, input_json=getattr(args, "input_json", None))
    entry = (spec.get("signals") or {}).get("entry") or {}
    exit_cfg = (spec.get("risk") or {}).get("exit")
    fees = (spec.get("risk") or {}).get("fees") or {}
    size = float((spec.get("sizing") or {}).get("size", 0.95))
    interval = str((spec.get("data") or {}).get("primary", {}).get("interval", "1h"))
    long_only = ((spec.get("data") or {}).get("market_type") == "spot") or (not entry.get("short"))
    initial = float((spec.get("backtest") or {}).get("initial_capital", 10000.0))

    make_signals = build_make_signals(spec)
    res = run_vectorized_backtest(
        df=df, signal_fn=make_signals, exit_cfg=exit_cfg, timeframe=interval,
        size=size, fee_bps=float(fees.get("commission_bps", 4.0)),
        slippage_bps=float(fees.get("slippage_bps", 2.0)),
        initial_capital=initial, long_only=long_only)

    print(f"[yaml_pipeline] backtest engine=vectorized "
          f"({'long-only' if long_only else 'long+short'}) data={src} bars={len(df)}")
    print(f"  total_return={res.total_return*100:.2f}%  pnl=${res.total_pnl:.2f}  "
          f"final_equity=${res.final_equity:.2f}")
    print(f"  sharpe={res.sharpe_ratio:.3f}  max_dd={res.max_drawdown*100:.2f}%  "
          f"win_rate={res.win_rate*100:.1f}%  trades={res.trade_count}")
    if getattr(args, "output_json", None):
        import json
        Path(args.output_json).write_text(json.dumps(
            {"engine": "vectorized", "source": src, "bars": len(df),
             "long_only": long_only, **res.to_dict()}, ensure_ascii=False, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


def _run_backtest_event(spec: Dict[str, Any], args: argparse.Namespace) -> int:
    """Backtest via the event-driven SnapshotBacktestRunner (long-only reference)."""
    from ..entrypoints import mvp_backtest

    symbol, interval, market_type, commission_bps, slippage_bps = _common_run_fields(spec)
    sid = spec["strategy"]["id"]
    bt = spec.get("backtest") or {}
    entry = (spec.get("signals") or {}).get("entry") or {}

    argv = [
        "mvp_backtest",
        "--engine", "python",
        "--strategy", sid,
        "--symbol", symbol,
        "--interval", interval,
        "--market-type", market_type,
        "--initial-capital", str(bt.get("initial_capital", 10000.0)),
        "--commission-bps", str(commission_bps),
        "--slippage-bps", str(slippage_bps),
        "--execution-model", bt.get("execution_model", "next_bar_open"),
    ]
    if not entry.get("short"):
        argv.append("--long-only")
    argv += _data_source_argv(spec, args)
    if getattr(args, "output_json", None):
        argv += ["--output-json", args.output_json]

    print(f"[yaml_pipeline] backtest → {' '.join(argv[1:])}")
    old = sys.argv
    try:
        sys.argv = argv
        return mvp_backtest.main()
    finally:
        sys.argv = old


def _paper_command(spec: Dict[str, Any]) -> List[str]:
    symbol, interval, market_type, _cb, _sb = _common_run_fields(spec)
    sid = spec["strategy"]["id"]
    run = spec.get("run") or {}
    schedule = (spec.get("data") or {}).get("primary") or {}
    fees = (spec.get("risk") or {}).get("fees") or {}
    cmd = [
        "python", "-m", "cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon",
        "--engine", "python",
        "--strategy", sid,
        "--strategy-module", DAEMON_LOADER,
        "--symbol", symbol,
        "--interval", interval,
        "--market-type", market_type,
        "--state-dir", f"./watcher/{sid}_{symbol}_{interval}",
        "--poll-interval", str(schedule.get("poll_interval", 3570)),
        "--warm-up-bars", str(schedule.get("warm_up_bars", 120)),
        "--initial-capital", str((spec.get("backtest") or {}).get("initial_capital", 10000.0)),
        "--fee-bps", str(fees.get("commission_bps", 4.0)),
        "--slippage-bps", str(fees.get("slippage_bps", 2.0)),
    ]
    if run.get("duration_end_at"):
        cmd += ["--session-end-at", run["duration_end_at"]]
    return cmd


def _run_paper(spec: Dict[str, Any], args: argparse.Namespace) -> int:
    cmd = _paper_command(spec)
    abspath = os.path.abspath(args.spec)
    print("[yaml_pipeline] paper trade daemon command:")
    print(f"  CYQNT_YAML_SPEC={abspath} \\")
    print("  " + " ".join(cmd))
    print("  這是模擬交易，尚未動用真實資金。")
    if not args.start:
        print("  (dry: pass --start to actually spawn the daemon; needs a live/historical data feed)")
        return 0
    import subprocess

    env = {**os.environ, "CYQNT_YAML_SPEC": abspath}
    return subprocess.call(cmd, env=env)


def _run_live(spec: Dict[str, Any], args: argparse.Namespace) -> int:
    symbol, interval, _mt, _cb, _sb = _common_run_fields(spec)
    sid = spec["strategy"]["id"]
    guards = (spec.get("risk") or {}).get("live_guards") or {}
    max_notional = guards.get("max_notional")
    state_dir = f"./watcher/{sid}_{symbol}_{interval}"

    print("[yaml_pipeline] LIVE mode — safety sequence (this CLI never places orders itself):")
    print("  規則(trading-modes):先跑 paper stage → 查餘額 → 明確 CONFIRM → 有時長上限。")
    print()
    print("  1) 訊號來源(paper daemon,與 paper 模式完全相同):")
    print(f"     CYQNT_YAML_SPEC={os.path.abspath(args.spec)} \\")
    print("     " + " ".join(_paper_command(spec)))
    print()
    print("  2) 先 dry-run 驗證 live executor(只印指令、不下單):")
    print(
        f"     python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor "
        f"--state-dir {state_dir} --symbol {symbol} --max-notional {max_notional} "
        f"--notional-fraction {guards.get('notional_fraction', 0.95)} --dry-run"
    )
    print()
    print("  3) 人工確認 dry-run 正確且已 CONFIRM 後,移除 --dry-run 才會真的下單。")
    print(f"  緊急停止:touch {state_dir}/EMERGENCY_STOP")
    print()
    print("  注意:futures live executor 僅在『已驗證 futures CLI 的環境』適用;"
          "否則先以 `binance-cli spot get-account` 為準。")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        spec = register_from_yaml(args.spec)
    except SpecError as exc:
        print(f"VALIDATION FAILED (not running):\n{exc}")
        return 1

    mode = (spec.get("run") or {}).get("mode")
    if mode == "backtest":
        if getattr(args, "engine", "vectorized") == "event":
            return _run_backtest_event(spec, args)
        return _run_backtest_vectorized(spec, args)
    if mode == "paper":
        return _run_paper(spec, args)
    if mode == "live":
        return _run_live(spec, args)
    print(f"unknown run.mode {mode!r}")
    return 1


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cyqnt_trd.standard_bot.yaml_pipeline",
        description="Validate and run declarative YAML strategy specs on standard_bot",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="static + dry-run validation of a spec")
    p_val.add_argument("spec", help="path to the strategy YAML")
    p_val.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="register + run a spec (backtest/paper/live)")
    p_run.add_argument("spec", help="path to the strategy YAML")
    p_run.add_argument("--input-json", default=None,
                       help="override data source with a local kline JSON (offline backtest)")
    p_run.add_argument("--output-json", default=None, help="write backtest result JSON here")
    p_run.add_argument("--engine", choices=["vectorized", "event"], default="vectorized",
                       help="backtest engine: 'vectorized' (long+short, matches paper/live; default) "
                            "or 'event' (SnapshotBacktestRunner, long-only reference)")
    p_run.add_argument("--start", action="store_true",
                       help="paper mode: actually spawn the daemon (needs a data feed)")
    p_run.set_defaults(func=cmd_run)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
