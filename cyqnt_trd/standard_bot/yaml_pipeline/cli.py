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


def _extra_data_argv(spec: Dict[str, Any]) -> List[str]:
    """Flags for the ``data:`` sections beyond OHLCV.

    The event engine reads these files through its own adapter, so the spec's
    declared sources have to be handed on or ``--engine event`` would quietly run
    the OHLCV-only version of the same strategy.
    """
    from .vocabulary import DATA_SECTIONS, declared_sections

    data = spec.get("data") or {}
    argv: List[str] = []
    for key in declared_sections(spec):
        argv += [DATA_SECTIONS[key].cli_flag, (data.get(key) or {})["dir"]]
    return argv


def _data_source_argv(spec: Dict[str, Any], overrides: argparse.Namespace) -> List[str]:
    """Map ``data.source`` (or CLI overrides) to backtest data flags."""
    if getattr(overrides, "input_json", None):
        return ["--input-json", overrides.input_json] + _extra_data_argv(spec)
    source = (spec.get("data") or {}).get("source") or {}
    stype = source.get("type", "binance_rest")
    if stype == "input_json":
        return ["--input-json", source["path"]] + _extra_data_argv(spec)
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
        return argv + _extra_data_argv(spec)
    if stype == "binance_rest":
        return ["--allow-remote-api"] + _extra_data_argv(spec)
    return _extra_data_argv(spec)


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


def _run_selection(spec: Dict[str, Any], args: argparse.Namespace) -> int:
    """Evaluate a ``selection:`` spec once and emit the ranked basket.

    A selection spec ranks a universe at ONE point in time; it has no per-bar
    equity curve, and this repo has no cross-sectional backtest engine. Before
    this, ``run`` dispatched purely on ``run.mode``, so a selection spec declaring
    ``mode: backtest`` fell into the single-instrument bar backtest: it fetched
    1000 BTCUSDT candles, ignored the ``selection:`` section entirely, and printed
    ``trades=0`` with a clean exit code. Nothing was wrong with the spec and
    nothing said so — the most expensive kind of silence.

    So: run the selector for real, print the basket, and refuse to call it a
    backtest.
    """
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from cyqnt_trd.blocks import strategy as _strategy

    from ..core import DataSnapshot, SnapshotMeta, UniverseBundle
    from ..runtime import data as data_runtime

    strategy_id = spec["strategy"]["id"]
    plugin = _strategy.get_selection_plugin(strategy_id)
    if plugin is None:                                  # register_from_yaml ran first
        print(f"no selection plugin registered as {strategy_id!r}")
        return 1

    market_type = (spec.get("data") or {}).get("market_type", "futures")
    status: Dict[str, str] = {}
    frames: Dict[str, Any] = {}
    with data_runtime.session() as session:
        as_of = session.as_of_ms
        for node, params in (("universe", {"market_type": market_type}),
                             ("ticker_rank", {"window": "24h", "limit": 30})):
            try:
                frames[node] = session.call(node, **params)
                status[node] = "ok" if len(frames[node]) else "empty"
            except data_runtime.DataUnavailable as exc:
                frames[node] = None
                status[node] = "error: %s" % exc.reason

    snapshot = DataSnapshot(
        version="mvp/v1",
        universe=UniverseBundle(as_of=as_of, universe=frames.get("universe"),
                                ticker_rank=frames.get("ticker_rank")),
        meta=SnapshotMeta(snapshot_id="yaml-selection-%d" % as_of, assembled_at=as_of,
                          decision_as_of=as_of, source_status=status, partial_ok=True),
    )
    envelope = plugin.run(snapshot, SimpleNamespace(market_type=market_type)).signals[0]
    payload = envelope.payload
    candidates = payload.get("candidates") or []

    print(f"[yaml_pipeline] selection strategy={strategy_id} market={market_type} "
          f"universe={payload.get('universe_size')} as_of={as_of}")
    print(f"  output={envelope.version} kind={envelope.kind.value} "
          f"data_quality={payload.get('data_quality')}")
    for key, value in status.items():
        if value != "ok":
            print(f"  {key}: {value}")
    if not candidates:
        print("  no candidate passed the filters — nothing to rank")
    for candidate in candidates:
        print("  #%-2d %-12s score=%-12.6g %-8s %s"
              % (candidate["rank"], candidate["symbol"], candidate["score"],
                 candidate["direction"], candidate.get("reason", "")))
    if (spec.get("run") or {}).get("mode") == "backtest":
        print("  NOTE: run.mode=backtest, but a cross-sectional selector has no "
              "per-bar equity curve and this repo has no cross-sectional backtest "
              "engine. The above is ONE decision point, not a backtest.")
    if getattr(args, "output_json", None):
        Path(args.output_json).write_text(json.dumps(
            {"signal_id": envelope.signal_id, "version": envelope.version,
             "kind": envelope.kind.value, "payload": payload},
            ensure_ascii=False, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


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


def _warn_paper_drops_declared_data(spec: Dict[str, Any]) -> None:
    """Say so when paper/live cannot supply data the backtest had.

    ``_extra_data_argv`` feeds ``--derivatives-dir`` / ``--liquidations-dir``
    into the backtest, but ``mvp_paper_daemon`` defines neither flag, so those
    columns simply do not exist in paper or live. A spec reading
    ``funding_rate`` / ``open_interest`` then backtests against real values and
    runs against nothing — same strategy id, two different strategies, and the
    only symptom is that it stops trading.
    """
    declared = [key for key in declared_sections(spec) if key != "primary"]
    if not declared:
        return
    print(
        "  WARNING: this spec declares %s, and mvp_paper_daemon has no flag for "
        "it — those columns will be ABSENT in paper/live while the backtest had "
        "them. Conditions reading them evaluate False, so the strategy silently "
        "becomes a different one. Verify the paper run produces the trades the "
        "backtest did before trusting it." % ", ".join("data.%s" % k for k in declared))


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
    _warn_paper_drops_declared_data(spec)
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
    # Shape AND mode, not shape alone. Dispatching on mode alone sent every
    # selection spec into the per-bar backtest, which has no universe to rank and
    # reported a clean trades=0. But dispatching on shape alone is worse: a
    # selection spec declaring ``mode: live`` then printed a basket and exited 0
    # — no daemon, no executor, no warning — and the operator believes live is
    # running. A single-shot evaluation may only stand in for a backtest, which
    # a cross-sectional selector genuinely has no engine for; it may never stand
    # in for execution.
    if isinstance(spec.get("selection"), dict):
        if mode in ("paper", "live"):
            print(
                "run.mode=%s is not supported for a selection spec: there is no "
                "resolver turning ranked candidates into per-symbol orders "
                "(build_intents only accepts kind=trade), so nothing would be "
                "executed and this command would exit 0 while doing nothing.\n"
                "  Use run.mode=backtest to evaluate one decision point and emit "
                "the cyqnt.signal/v2 basket, then hand that basket to whatever "
                "places the orders." % mode)
            return 1
        return _run_selection(spec, args)
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
                       help="backtest engine: 'vectorized' (in-process run_vectorized_backtest; "
                            "default) or 'event' (mvp_backtest --engine python / "
                            "SnapshotBacktestRunner). BOTH support long+short on futures "
                            "(spot is long-only by design). 'event' additionally honours "
                            "backtest.execution_model and data.htf[], which the vectorized "
                            "path ignores.")
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
