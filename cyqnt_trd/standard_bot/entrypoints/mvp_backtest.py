"""
Run the standard bot MVP as a market-only historical backtest.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from ..core import BacktestRequest, MarketQuery, SignalPipelineSpec, TimeRange
from ..data import AlignmentPolicy, timeframe_to_ms
from ..data.derivatives import HistoricalBinanceDerivativesDownloader
from ..data.liquidations import HistoricalBinanceLiquidationRecorder
from ..data.snapshot import HistoricalSnapshotAssembler
from .common import (
    add_historical_data_arguments,
    build_market_data_adapter,
    build_strategy_pipeline,
    build_time_range,
    infer_contract_multiplier,
    make_registry,
)
from ..simulation import NumbaBacktestRunner, SnapshotBacktestRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standard bot MVP backtest")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--market-type", choices=["spot", "futures", "cme"], default="spot")
    parser.add_argument("--engine", choices=["python", "numba"], default="numba")
    parser.add_argument("--strategy", default="moving_average_cross")
    parser.add_argument("--strategy-module", default=None, help="Python module to import for external strategy registration")
    parser.add_argument("--extra-params", default=None, help="JSON dict of extra strategy config params")
    parser.add_argument("--secondary-interval", default="1h")
    parser.add_argument("--fast-window", type=int, default=5)
    parser.add_argument("--slow-window", type=int, default=20)
    parser.add_argument("--ma-period", type=int, default=5)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--oversold", type=float, default=30.0)
    parser.add_argument("--overbought", type=float, default=70.0)
    parser.add_argument("--donchian-window", type=int, default=20)
    parser.add_argument("--breakout-buffer-bps", type=float, default=0.0)
    parser.add_argument("--adx-period", type=int, default=14)
    parser.add_argument("--adx-threshold", type=float, default=25.0)
    parser.add_argument("--atr-ma-period", type=int, default=20)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--atr-multiplier", type=float, default=2.0)
    parser.add_argument("--bollinger-period", type=int, default=20)
    parser.add_argument("--bollinger-stddev-multiplier", type=float, default=2.0)
    parser.add_argument("--macd-fast-period", type=int, default=12)
    parser.add_argument("--macd-slow-period", type=int, default=26)
    parser.add_argument("--macd-signal-period", type=int, default=9)
    parser.add_argument("--macd-histogram-threshold", type=float, default=0.0)
    parser.add_argument("--oi-threshold-bps", type=float, default=0.0)
    parser.add_argument("--max-funding-rate-bps", type=float, default=100.0)
    parser.add_argument("--long-liquidation-threshold-usd", type=float, default=100_000.0)
    parser.add_argument("--short-liquidation-threshold-usd", type=float, default=100_000.0)
    parser.add_argument("--liquidation-imbalance-ratio", type=float, default=0.60)
    parser.add_argument("--primary-ma-period", type=int, default=20)
    parser.add_argument("--reference-ma-period", type=int, default=20)
    parser.add_argument("--spread-threshold-bps", type=float, default=0.0)
    parser.add_argument("--entry-threshold", type=float, default=0.0)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--commission-bps", type=float, default=10.0)
    parser.add_argument("--taker-fee-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--impact-slippage-bps", type=float, default=0.0)
    parser.add_argument("--funding-bps-per-bar", type=float, default=0.0)
    parser.add_argument("--max-bar-volume-fraction", type=float, default=0.10)
    parser.add_argument("--contract-multiplier", type=float, default=None)
    parser.add_argument("--quantity-step", type=float, default=0.0)
    parser.add_argument("--min-quantity", type=float, default=0.0)
    parser.add_argument("--fixed-fee-per-contract", type=float, default=0.0)
    parser.add_argument("--execution-model", choices=["close_fill", "next_bar_open"], default="close_fill",
                        help="Fill model: 'close_fill' (legacy, fill at signal bar close) or "
                             "'next_bar_open' (realistic, fill at next bar open)")
    parser.add_argument("--tail-bars", type=int, default=120)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--download-derivatives-missing", action="store_true")
    parser.add_argument("--record-liquidations-seconds", type=int, default=0)
    parser.add_argument("--long-only", default=None, action="store_true",
                        help="Only open long positions (no shorts). Auto-detected from "
                             "--market-type if not specified (spot=long_only, futures=both).")
    return add_historical_data_arguments(parser)


def build_market_query(args: argparse.Namespace) -> MarketQuery:
    timeframes = [args.interval]
    if args.strategy == "multi_timeframe_ma_spread" and args.secondary_interval not in timeframes:
        timeframes.append(args.secondary_interval)

    # If this is a block strategy with HTF requirements, include those TFs
    # in the market query so HTF bars are available in every snapshot.
    try:
        from ...blocks.strategy import get_block_plugin  # type: ignore

        plugin = get_block_plugin(args.strategy)
        if plugin is not None:
            for htf in plugin.needed_timeframes():
                if htf and htf not in timeframes:
                    timeframes.append(htf)
    except ImportError:
        pass

    return MarketQuery(
        instruments=[args.symbol.upper()],
        timeframes=timeframes,
        time_range=build_time_range(limit=args.limit, start_ts=args.start_ts, end_ts=args.end_ts),
    )


def load_market_bundle(args: argparse.Namespace, market_query: MarketQuery):
    adapter = build_market_data_adapter(
        args=args,
        symbol=args.symbol.upper(),
        timeframe=args.interval,
    )
    try:
        return adapter.fetch_market(market_query)
    except FileNotFoundError as error:
        if args.download_missing and (market_query.time_range.start_ts is None or market_query.time_range.end_ts is None):
            raise ValueError("--download-missing requires both --start-ts and --end-ts") from error
        raise RuntimeError(
            "missing local historical parquet data; rerun with --download-missing or --allow-remote-api"
        ) from error


def main() -> int:
    args = build_parser().parse_args()
    if args.strategy_module:
        import importlib
        importlib.import_module(args.strategy_module)
    extra_params = json.loads(args.extra_params) if args.extra_params else None
    if args.download_derivatives_missing:
        if args.start_ts is None or args.end_ts is None:
            raise ValueError("--download-derivatives-missing requires both --start-ts and --end-ts")
        if args.strategy == "oi_funding_breakout":
            HistoricalBinanceDerivativesDownloader(
                data_root=args.derivatives_dir,
                market_type=args.market_type,
            ).download_bundle(
                instrument_id=args.symbol.upper(),
                timeframe=args.interval,
                start_ts=int(args.start_ts),
                end_ts=int(args.end_ts),
            )
    if args.record_liquidations_seconds > 0:
        if args.strategy != "liquidation_reversal":
            raise ValueError("--record-liquidations-seconds is currently supported for liquidation_reversal only")
        HistoricalBinanceLiquidationRecorder(
            data_root=args.liquidations_dir,
            market_type=args.market_type,
        ).record_and_store(
            duration_sec=int(args.record_liquidations_seconds),
            timeframe=args.interval,
            timeframe_ms=timeframe_to_ms(args.interval),
            instrument_ids=[args.symbol.upper()],
        )

    # Resolve long_only: explicit flag > auto-detect from market_type
    if args.long_only is None:
        long_only = (args.market_type == "spot")
    else:
        long_only = bool(args.long_only)

    policy = AlignmentPolicy(policy_id="bar_close_v1", primary_timeframe=args.interval)
    market_query = build_market_query(args)
    market_bundle = load_market_bundle(args, market_query)
    key = market_bundle.key(args.symbol.upper(), args.interval)
    bars = market_bundle.bars.get(key, [])
    if not bars:
        raise ValueError("no bars available for %s" % key)

    registry = make_registry()
    pipeline = build_strategy_pipeline(
        strategy=args.strategy,
        symbol=args.symbol.upper(),
        interval=args.interval,
        secondary_interval=args.secondary_interval,
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        entry_threshold=args.entry_threshold,
        ma_period=args.ma_period,
        rsi_period=args.rsi_period,
        oversold=args.oversold,
        overbought=args.overbought,
        donchian_window=args.donchian_window,
        breakout_buffer_bps=args.breakout_buffer_bps,
        adx_period=args.adx_period,
        adx_threshold=args.adx_threshold,
        atr_ma_period=args.atr_ma_period,
        atr_period=args.atr_period,
        atr_multiplier=args.atr_multiplier,
        bollinger_period=args.bollinger_period,
        bollinger_stddev_multiplier=args.bollinger_stddev_multiplier,
        macd_fast_period=args.macd_fast_period,
        macd_slow_period=args.macd_slow_period,
        macd_signal_period=args.macd_signal_period,
        macd_histogram_threshold=args.macd_histogram_threshold,
        oi_threshold_bps=args.oi_threshold_bps,
        max_funding_rate_bps=args.max_funding_rate_bps,
        long_liquidation_threshold_usd=args.long_liquidation_threshold_usd,
        short_liquidation_threshold_usd=args.short_liquidation_threshold_usd,
        liquidation_imbalance_ratio=args.liquidation_imbalance_ratio,
        primary_ma_period=args.primary_ma_period,
        reference_ma_period=args.reference_ma_period,
        spread_threshold_bps=args.spread_threshold_bps,
        extra_params=extra_params,
    )

    request = BacktestRequest(
        request_id=str(uuid.uuid4()),
        instruments=[args.symbol.upper()],
        primary_timeframe=args.interval,
        start_ts=bars[0].timestamp,
        end_ts=bars[-1].timestamp,
        signal_pipeline=pipeline,
        initial_capital=args.initial_capital,
        fee_model={
            "commission_bps": args.commission_bps,
            "taker_fee_bps": args.taker_fee_bps if args.taker_fee_bps is not None else args.commission_bps,
            "funding_bps_per_bar": args.funding_bps_per_bar,
            "fixed_fee_per_contract": args.fixed_fee_per_contract,
        },
        slippage_model={
            "slippage_bps": args.slippage_bps,
            "impact_slippage_bps": args.impact_slippage_bps,
            "max_bar_volume_fraction": args.max_bar_volume_fraction,
        },
        extras={
            "engine": args.engine,
            "execution_assumption": "signal_at_bar_close_fill_next_bar_open",
            "execution_model": args.execution_model,
            "market_type": args.market_type,
            "long_only": long_only,
            "contract_multiplier": (
                args.contract_multiplier
                if args.contract_multiplier is not None
                else infer_contract_multiplier(market_type=args.market_type, symbol=args.symbol.upper())
            ),
            "quantity_step": args.quantity_step,
            "min_quantity": args.min_quantity,
        },
    )

    if args.engine == "numba":
        result = NumbaBacktestRunner().run(
            request=request,
            market_bundle=market_bundle,
        )
    else:
        snapshots = HistoricalSnapshotAssembler(policy=policy, tail_bars=args.tail_bars).build(market_bundle)
        request.start_ts = snapshots[0].meta.decision_as_of or snapshots[0].meta.assembled_at
        request.end_ts = snapshots[-1].meta.decision_as_of or snapshots[-1].meta.assembled_at
        result = SnapshotBacktestRunner(signal_registry=registry).run(
            request=request,
            snapshots=snapshots,
        )

    trades = result.extras.get("trades", [])
    print(
        "engine=%s run_id=%s snapshots=%s trades=%s total_return=%.6f final_equity=%.2f"
        % (
            args.engine,
            result.extras.get("run_id"),
            int(result.metrics.get("snapshot_count", 0)),
            int(result.metrics.get("trade_count", 0)),
            float(result.total_return),
            float(result.metrics.get("final_equity", args.initial_capital)),
        )
    )
    if args.engine == "numba":
        print(
            "execution_model=signal_at_bar_close_fill_next_bar_open "
            "taker_fee_bps=%.4f funding_bps_per_bar=%.4f max_bar_volume_fraction=%.4f "
            "contract_multiplier=%.4f quantity_step=%.4f fixed_fee_per_contract=%.4f"
            % (
                float(request.fee_model.get("taker_fee_bps", 0.0)),
                float(request.fee_model.get("funding_bps_per_bar", 0.0)),
                float(request.slippage_model.get("max_bar_volume_fraction", 0.0)),
                float(request.extras.get("contract_multiplier", 1.0)),
                float(request.extras.get("quantity_step", 0.0)),
                float(request.fee_model.get("fixed_fee_per_contract", 0.0)),
            )
        )
    if trades:
        print("last_trade=%s" % json.dumps(trades[-1], ensure_ascii=False))

    if args.output_json:
        payload = {
            "request_id": result.request_id,
            "total_return": result.total_return,
            "metrics": result.metrics,
            "equity_curve": [point.__dict__ for point in result.equity_curve],
            "signal_batches": [
                {
                    "batch_id": batch.batch_id,
                    "created_at": batch.created_at,
                    "signals": [
                        {
                            "signal_id": signal.signal_id,
                            "instrument_id": signal.instrument_id,
                            "side": None if signal.side is None else signal.side.value,
                            "strength": signal.strength,
                            "payload": signal.payload,
                        }
                        for signal in batch.signals
                    ],
                }
                for batch in result.signal_batches
            ],
            "trades": trades,
            "extras": result.extras,
        }
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote_output=%s" % args.output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
