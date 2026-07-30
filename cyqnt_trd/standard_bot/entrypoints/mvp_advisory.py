"""
Run one advisory/monitor cycle on the standard bot route:

    adapter -> build_advisory_snapshot -> SignalPluginRegistry.run_pipeline_step
            -> SignalBatch(ALERT) -> BotSignalFrame

Same registry and same ``run_pipeline_step`` call the trade and selection
entrypoints use; the only difference is the snapshot carries named frames and
the emitted signals are ``kind=ALERT``. There is deliberately no executor
branch — advisory signals carry ``auto_trade_eligible=false`` and
``build_intents`` only consumes ``kind=TRADE``.

Examples::

    # what bots exist and what each one needs
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_advisory --list

    # live derivatives crowding check on two perps
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_advisory \\
        --bot derivatives_positioning_monitor --symbols BTCUSDT,ETHUSDT

    # replay from frames captured earlier (no network)
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_advisory \\
        --bot price_volume_monitor \\
        --frame market_metrics=tmp/market_metrics.csv --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict

from ..advisory import (
    FetchFrame,
    canvas_definition,
    list_advisory_bots,
    signals_to_frame,
)
from ..data import run_advisory
from .common import build_advisory_pipeline, make_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one standard-bot advisory (monitor) cycle"
    )
    parser.add_argument("--bot", default=None, help="advisory bot id (plugin_id)")
    parser.add_argument(
        "--list", action="store_true", help="list registered advisory bots and exit"
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="comma-separated symbols for market-metric bots, e.g. BTCUSDT,ETHUSDT",
    )
    parser.add_argument("--venue", default="binance")
    parser.add_argument(
        "--product",
        choices=["spot", "usd_m_perpetual", "mixed"],
        default="usd_m_perpetual",
    )
    parser.add_argument(
        "--market", choices=["spot", "futures"], default="futures",
        help="which Binance leg the market metrics are read from",
    )
    parser.add_argument(
        "--config", default=None, help="JSON dict of bot config overrides"
    )
    parser.add_argument(
        "--frame",
        action="append",
        default=[],
        metavar="NAME=PATH.csv",
        help="offline replay: load a frame from CSV instead of fetching it "
             "(repeatable). Disables all network access.",
    )
    parser.add_argument(
        "--decision-as-of",
        default=None,
        help="ISO timestamp cut-off; defaults to the latest available_time in the data",
    )
    parser.add_argument("--news-page-size", type=int, default=50)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument(
        "--canvas", action="store_true", help="print the Canvas DSL for --bot and exit"
    )
    return parser


def _load_offline_frames(specs) -> Dict[str, FetchFrame]:
    import pandas as pd

    frames: Dict[str, FetchFrame] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit("--frame expects NAME=PATH.csv, got %r" % spec)
        name, path = spec.split("=", 1)
        frame = pd.read_csv(path)
        frames[name.strip()] = FetchFrame(
            frame=frame, status="ok", source="file:%s" % path
        )
    return frames


def main() -> int:
    args = build_parser().parse_args()

    if args.list:
        print(json.dumps(list_advisory_bots(), ensure_ascii=False, indent=2, default=str))
        return 0
    if not args.bot:
        build_parser().print_usage(sys.stderr)
        print("error: --bot is required (or use --list)", file=sys.stderr)
        return 2

    config = json.loads(args.config) if args.config else {}

    if args.canvas:
        print(json.dumps(canvas_definition(args.bot, config), ensure_ascii=False, indent=2))
        return 0

    # Resolve the plugin + validate its config before anything is fetched.
    registry = make_registry()
    pipeline = build_advisory_pipeline(bot=args.bot, config=config)
    registry.get(args.bot)

    frames = _load_offline_frames(args.frame) if args.frame else None
    symbols = (
        [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        if args.symbols
        else None
    )

    result = run_advisory(
        args.bot,
        frames=frames,
        symbols=symbols,
        config=pipeline.plugin_chain[0]["config"],
        venue=args.venue,
        product=args.product,
        market=args.market,
        decision_as_of=args.decision_as_of,
        news_page_size=args.news_page_size,
        registry=registry,
    )
    output = signals_to_frame(result.batch)

    if args.format == "json":
        print(output.to_json(orient="records", force_ascii=False, date_format="iso"))
    else:
        if output.empty:
            print("no alerts (bot ran, 0 decisions)")
        else:
            columns = [
                "symbol", "product", "direction", "action", "score",
                "confidence", "topic", "reason_codes", "data_quality", "summary",
            ]
            print(output[columns].to_string(index=False))
    print(
        "alerts=%d bot=%s auto_trade_eligible=false" % (len(output), args.bot),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
