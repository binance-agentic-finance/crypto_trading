"""Produce the input bundle, and run a strategy against it.

    live nodes -> build_live_snapshot -> DataSnapshot(market + universe + frames)
               -> plugin.run() -> cyqnt.signal/v2

Why this exists
---------------
``cyqnt.input/v1`` was defined, tested, and documented, and **no command
produced one**. The only non-test callers of ``build_live_bundle`` were the docs
generator and the test suite, so the contract existed on paper while every actual
run went through ``assemble_snapshot`` — which has no ``frames`` parameter and
therefore hands a strategy bars and nothing else.

That combination is worse than a missing feature, because it is quiet. A strategy
registered with ``needs={"derivatives": True}`` still ran; it just ran without
derivatives. Measured on ``funding_squeeze_panel``: through this path
``make_signals`` receives 35 columns, through the bars-only path 13.

Examples::

    # write the bundle and show what each source returned
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_input_bundle \\
        --symbol BTCUSDT --interval 1h --out tmp/bundle.json

    # only the sources a spec declares, then decide with them
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_input_bundle \\
        --sections derivatives --strategy funding_squeeze_panel

    # replay yesterday's bundle offline — no network, same decision
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_input_bundle \\
        --replay tmp/bundle.json --strategy funding_squeeze_panel

``--replay`` is the reason the bundle is written at all: a live decision that
cannot be reproduced cannot be reviewed after it loses money.
"""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a cyqnt.input/v1 bundle from live nodes and "
                    "optionally run a strategy on it")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=500,
                        help="bars requested (also sizes the per-node windows)")
    parser.add_argument("--market-type", default="futures",
                        choices=("spot", "futures"))
    parser.add_argument(
        "--sections", default=None,
        help="comma-separated data sections to fetch (derivatives,news,"
             "orderbook,universe). Omit for every node a single-instrument "
             "decision can use.")
    parser.add_argument("--out", default=None,
                        help="write the bundle JSON here (enables replay)")
    parser.add_argument("--strategy", default=None,
                        help="registered strategy id to run on the bundle")
    parser.add_argument("--replay", default=None,
                        help="load this bundle instead of fetching (no network)")
    parser.add_argument("--include-account", action="store_true",
                        help="also fetch positions/balance (needs credentials)")
    parser.add_argument("--format", default="text", choices=("text", "json"))
    return parser


def _load_strategy_modules() -> None:
    """Import the strategy package so registration side effects happen.

    A strategy registers itself at import time, so ``--strategy X`` on a fresh
    process finds an empty registry unless something imports the modules first.
    """
    import importlib
    import pkgutil

    try:
        package = importlib.import_module("cyqnt_trd.strategies")
    except Exception:
        return
    for info in pkgutil.iter_modules(getattr(package, "__path__", [])):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module("cyqnt_trd.strategies.%s" % info.name)
        except Exception as exc:          # a broken sibling must not hide the rest
            print("  (skipped cyqnt_trd.strategies.%s: %s)" % (info.name, exc),
                  file=sys.stderr)


def _run_strategy(snapshot, strategy_id: str, *, symbol: str, interval: str,
                  market_type: str) -> Dict[str, Any]:
    """Run whichever plugin kind is registered under ``strategy_id``."""
    from cyqnt_trd.blocks import strategy as blocks_strategy

    _load_strategy_modules()

    selection = blocks_strategy.get_selection_plugin(strategy_id)
    trade = blocks_strategy._KNOWN_BLOCK_PLUGINS.get(strategy_id)
    if selection is None and trade is None:
        known = sorted(set(blocks_strategy._KNOWN_BLOCK_PLUGINS)
                       | set(blocks_strategy._KNOWN_SELECTION_PLUGINS))
        raise SystemExit("no strategy registered as %r. known: %s"
                         % (strategy_id, ", ".join(known) or "(none)"))

    if selection is not None:
        batch = selection.run(snapshot, SimpleNamespace(market_type=market_type))
    else:
        batch = trade.run(snapshot, SimpleNamespace(
            instrument_id=symbol, symbol=symbol, timeframe=interval,
            interval=interval, market_type=market_type))
    envelopes = list(getattr(batch, "signals", []) or [])
    return {
        "strategy": strategy_id,
        "kind": "selection" if selection is not None else "trade",
        "signals": [
            {"version": env.version,
             "signal_id": env.signal_id,
             "kind": getattr(env.kind, "value", str(env.kind)),
             "instrument_id": env.instrument_id,
             "payload": env.payload}
            for env in envelopes
        ],
    }


def _column_count(snapshot, *, symbol: str, interval: str) -> Optional[int]:
    """How many columns a ``make_signals`` would actually receive.

    Printed because it is the one number that separates "the frames are attached"
    from "the strategy is quietly running on price alone".
    """
    from cyqnt_trd.blocks import strategy as blocks_strategy

    seen: Dict[str, Any] = {}

    def spy(df):
        import pandas as pd

        seen["columns"] = list(df.columns)
        return (pd.Series(False, index=df.index), pd.Series(False, index=df.index))

    try:
        probe = blocks_strategy.build_plugin("__column_probe__", spy)
        probe.run(snapshot, SimpleNamespace(
            instrument_id=symbol, symbol=symbol, timeframe=interval,
            interval=interval))
    except Exception:
        return None
    return len(seen.get("columns") or []) or None


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    from ..data import load_input_bundle
    from ..data.live_snapshot import build_live_snapshot

    if args.replay:
        with open(args.replay, encoding="utf-8") as handle:
            bundle = json.load(handle)
        snapshot = load_input_bundle(bundle)
        origin = "replay %s" % args.replay
    else:
        sections = ([s.strip() for s in args.sections.split(",") if s.strip()]
                    if args.sections else None)
        snapshot, bundle = build_live_snapshot(
            sections=sections, symbol=args.symbol, interval=args.interval,
            limit=args.limit, market_type=args.market_type,
            include_account=args.include_account, write_bundle=args.out)
        origin = "live"

    status: Dict[str, str] = dict(bundle.get("source_status") or {})
    frames = bundle.get("frames") or {}
    result: Dict[str, Any] = {
        "schema": bundle.get("schema"),
        "decision_time": bundle.get("decision_time"),
        "origin": origin,
        "nodes_ok": sum(1 for v in status.values() if v == "ok"),
        "nodes_total": len(status),
        "source_status": status,
        "warnings": list(bundle.get("warnings") or []),
        "strategy_columns": _column_count(
            snapshot, symbol=args.symbol, interval=args.interval),
    }
    if args.out and not args.replay:
        result["bundle_path"] = args.out

    if args.strategy:
        result.update(_run_strategy(
            snapshot, args.strategy, symbol=args.symbol,
            interval=args.interval, market_type=args.market_type))

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    print("[input_bundle] %s schema=%s decision_time=%s"
          % (origin, result["schema"], result["decision_time"]))
    print("  nodes ok %d/%d   frames %d   strategy sees %s columns"
          % (result["nodes_ok"], result["nodes_total"], len(frames),
             result["strategy_columns"] or "?"))
    for key, value in sorted(status.items()):
        if value != "ok":
            print("  %-20s %s" % (key, value))
    for warning in result["warnings"][:8]:
        print("  ! %s" % warning)
    if result.get("bundle_path"):
        print("  bundle -> %s" % result["bundle_path"])

    for signal in result.get("signals", []):
        payload = signal["payload"]
        print("  %s %s %s" % (signal["version"], signal["kind"],
                              signal.get("instrument_id") or ""))
        if signal["kind"] == "selection":
            candidates = payload.get("candidates") or []
            print("    universe=%s candidates=%d data_quality=%s"
                  % (payload.get("universe_size"), len(candidates),
                     payload.get("data_quality")))
            for candidate in candidates[:10]:
                print("      %-12s %-8s %s" % (candidate.get("symbol"),
                                               candidate.get("direction"),
                                               candidate.get("reason")))
        else:
            print("    %s %s size=%s reason=%s"
                  % (payload.get("intent"), payload.get("direction"),
                     payload.get("size"), payload.get("reason_codes")))
    if args.strategy and not result.get("signals"):
        print("  %s produced no signal on this snapshot" % args.strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
