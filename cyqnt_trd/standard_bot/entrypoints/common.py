"""
Shared helpers for standard bot CLI entrypoints.
"""

from __future__ import annotations

import argparse
import json

from ..core import TimeRange
from ..core import SignalPipelineSpec
from ..data.adapters import BinanceRestMarketDataAdapter, HistoricalJsonMarketDataAdapter
from ..data.downloader import HistoricalBinanceDownloader
from ..data.historical import HistoricalParquetMarketDataAdapter, LocalFirstMarketDataAdapter
from ..signal import SignalPluginRegistry, register_builtin_plugins
from ..simulation import NumbaBacktestRunner


def make_registry() -> SignalPluginRegistry:
    registry = SignalPluginRegistry()
    register_builtin_plugins(registry)
    # Advisory/monitor bots are ordinary SignalPlugins (kind=ALERT), so they
    # live in the same registry and are addressable by plugin_id from any
    # entrypoint. They never reach an executor: build_intents only takes
    # kind=TRADE, and every advisory payload carries auto_trade_eligible=false.
    from ..advisory import register_advisory_bots

    register_advisory_bots(registry)
    # Drain any block-strategy registrations queued via
    # ``cyqnt_trd.blocks.strategy.register(...)`` (typically by an
    # external --strategy-module imported earlier in main()).
    try:
        from ...blocks.strategy import flush_pending_into  # type: ignore

        flush_pending_into(registry)
    except ImportError:
        # blocks package not installed — fine, just skip
        pass
    return registry


def build_advisory_pipeline(*, bot: str, config: dict | None = None) -> SignalPipelineSpec:
    """Advisory sibling of :func:`build_strategy_pipeline`.

    Validates the bot id and its config against the bot's own config dataclass
    before anything is fetched, then returns the standard one-plugin chain.
    """
    from ..advisory import create_advisory_bot

    create_advisory_bot(bot).normalize_config(dict(config or {}))
    return SignalPipelineSpec(plugin_chain=[{"plugin_id": bot, "config": dict(config or {})}])


def add_historical_data_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--historical-dir", default="data/historical")
    parser.add_argument("--derivatives-dir", default="data/derivatives")
    parser.add_argument("--liquidations-dir", default="data/liquidations")
    parser.add_argument("--storage-timeframe", default="1m")
    parser.add_argument("--start-ts", type=int, default=None)
    parser.add_argument("--end-ts", type=int, default=None)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--allow-remote-api", action="store_true")
    return parser


def build_time_range(*, limit: int, start_ts: int | None, end_ts: int | None) -> TimeRange:
    if start_ts is not None or end_ts is not None:
        return TimeRange(start_ts=start_ts, end_ts=end_ts)
    return TimeRange(tail_bars=limit)


def build_market_data_adapter(
    *,
    args: argparse.Namespace,
    symbol: str,
    timeframe: str,
):
    if getattr(args, "input_json", None):
        return HistoricalJsonMarketDataAdapter(
            data_path=args.input_json,
            instrument_id=symbol.upper(),
            timeframe=timeframe,
        )

    local_adapter = HistoricalParquetMarketDataAdapter(
        data_root=args.historical_dir,
        market_type=args.market_type,
        resample_source_timeframe=args.storage_timeframe,
        derivatives_data_root=getattr(args, "derivatives_dir", None),
        liquidations_data_root=getattr(args, "liquidations_dir", None),
    )
    remote_adapter = None
    if getattr(args, "allow_remote_api", False):
        remote_adapter = BinanceRestMarketDataAdapter(market_type=args.market_type)
    downloader = None
    if getattr(args, "download_missing", False):
        downloader = HistoricalBinanceDownloader(
            data_root=args.historical_dir,
            market_type=args.market_type,
        )
    return LocalFirstMarketDataAdapter(
        local_adapter=local_adapter,
        storage_timeframe=args.storage_timeframe,
        remote_adapter=remote_adapter,
        downloader=downloader,
    )


def infer_contract_multiplier(*, market_type: str, symbol: str) -> float:
    if market_type == "cme":
        from ..data.cme import cme_contract_multiplier

        return float(cme_contract_multiplier(symbol))
    return 1.0


def build_strategy_pipeline(
    *,
    strategy: str,
    symbol: str,
    interval: str,
    secondary_interval: str = "1h",
    fast_window: int,
    slow_window: int,
    entry_threshold: float,
    ma_period: int,
    rsi_period: int,
    oversold: float,
    overbought: float,
    donchian_window: int = 20,
    breakout_buffer_bps: float = 0.0,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    atr_ma_period: int = 20,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
    bollinger_period: int = 20,
    bollinger_stddev_multiplier: float = 2.0,
    macd_fast_period: int = 12,
    macd_slow_period: int = 26,
    macd_signal_period: int = 9,
    macd_histogram_threshold: float = 0.0,
    oi_threshold_bps: float = 0.0,
    max_funding_rate_bps: float = 100.0,
    long_liquidation_threshold_usd: float = 100_000.0,
    short_liquidation_threshold_usd: float = 100_000.0,
    liquidation_imbalance_ratio: float = 0.60,
    primary_ma_period: int = 20,
    reference_ma_period: int = 20,
    spread_threshold_bps: float = 0.0,
    extra_params: dict | None = None,
) -> SignalPipelineSpec:
    if strategy == "moving_average_cross":
        plugin_spec = {
            "plugin_id": "moving_average_cross",
            "config": {
                "instrument_id": symbol,
                "timeframe": interval,
                "fast_window": fast_window,
                "slow_window": slow_window,
                "entry_threshold": entry_threshold,
            },
        }
    elif strategy == "price_moving_average":
        plugin_spec = {
            "plugin_id": "price_moving_average",
            "config": {
                "instrument_id": symbol,
                "timeframe": interval,
                "period": ma_period,
                "entry_threshold": entry_threshold,
            },
        }
    elif strategy == "rsi_reversion":
        plugin_spec = {
            "plugin_id": "rsi_reversion",
            "config": {
                "instrument_id": symbol,
                "timeframe": interval,
                "period": rsi_period,
                "oversold": oversold,
                "overbought": overbought,
            },
        }
    elif strategy == "donchian_breakout":
        plugin_spec = {
            "plugin_id": "donchian_breakout",
            "config": {
                "instrument_id": symbol,
                "timeframe": interval,
                "lookback_window": donchian_window,
                "breakout_buffer_bps": breakout_buffer_bps,
            },
        }
    elif strategy == "adx_trend_strength":
        plugin_spec = {
            "plugin_id": "adx_trend_strength",
            "config": {
                "instrument_id": symbol,
                "timeframe": interval,
                "period": adx_period,
                "adx_threshold": adx_threshold,
            },
        }
    elif strategy == "atr_breakout":
        plugin_spec = {
            "plugin_id": "atr_breakout",
            "config": {
                "instrument_id": symbol,
                "timeframe": interval,
                "ma_period": atr_ma_period,
                "atr_period": atr_period,
                "atr_multiplier": atr_multiplier,
            },
        }
    elif strategy == "bollinger_mean_reversion":
        plugin_spec = {
            "plugin_id": "bollinger_mean_reversion",
            "config": {
                "instrument_id": symbol,
                "timeframe": interval,
                "period": bollinger_period,
                "stddev_multiplier": bollinger_stddev_multiplier,
            },
        }
    elif strategy == "macd_trend_follow":
        plugin_spec = {
            "plugin_id": "macd_trend_follow",
            "config": {
                "instrument_id": symbol,
                "timeframe": interval,
                "fast_period": macd_fast_period,
                "slow_period": macd_slow_period,
                "signal_period": macd_signal_period,
                "histogram_threshold": macd_histogram_threshold,
            },
        }
    elif strategy == "oi_funding_breakout":
        plugin_spec = {
            "plugin_id": "oi_funding_breakout",
            "config": {
                "instrument_id": symbol,
                "timeframe": interval,
                "lookback_window": donchian_window,
                "breakout_buffer_bps": breakout_buffer_bps,
                "oi_threshold_bps": oi_threshold_bps,
                "max_funding_rate_bps": max_funding_rate_bps,
            },
        }
    elif strategy == "liquidation_reversal":
        plugin_spec = {
            "plugin_id": "liquidation_reversal",
            "config": {
                "instrument_id": symbol,
                "timeframe": interval,
                "long_liquidation_threshold_usd": long_liquidation_threshold_usd,
                "short_liquidation_threshold_usd": short_liquidation_threshold_usd,
                "liquidation_imbalance_ratio": liquidation_imbalance_ratio,
            },
        }
    elif strategy == "multi_timeframe_ma_spread":
        plugin_spec = {
            "plugin_id": "multi_timeframe_ma_spread",
            "config": {
                "instrument_id": symbol,
                "primary_timeframe": interval,
                "reference_timeframe": secondary_interval,
                "primary_ma_period": primary_ma_period,
                "reference_ma_period": reference_ma_period,
                "spread_threshold_bps": spread_threshold_bps,
            },
        }
    else:
        if strategy in NumbaBacktestRunner.registered_kernels():
            config = {
                "instrument_id": symbol,
                "timeframe": interval,
            }
            if extra_params:
                config.update(extra_params)
            plugin_spec = {
                "plugin_id": strategy,
                "config": config,
            }
        else:
            # Check whether this is a block strategy registered via
            # cyqnt_trd.blocks.strategy.register(...) in an external
            # --strategy-module file.
            is_block_strategy = False
            try:
                from ...blocks.strategy import is_known_block_strategy  # type: ignore

                is_block_strategy = is_known_block_strategy(strategy)
            except ImportError:
                pass
            if is_block_strategy:
                config = {
                    "instrument_id": symbol,
                    "timeframe": interval,
                }
                if extra_params:
                    config.update(extra_params)
                plugin_spec = {
                    "plugin_id": strategy,
                    "config": config,
                }
            else:
                raise ValueError(
                    "unsupported strategy '%s'; register external strategies with "
                    "NumbaBacktestRunner.register_kernel(...) or "
                    "cyqnt_trd.blocks.strategy.register(...) before use" % strategy
                )

    return SignalPipelineSpec(plugin_chain=[plugin_spec])
