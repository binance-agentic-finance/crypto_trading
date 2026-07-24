"""
binance-cli / binance-pro-cli subprocess wrappers — pandas DataFrame outputs.

Sourced from atomic_strategy_lib.data, ported to cyqnt_trd as part of the
atomic→cyqnt_trd migration. Default data path uses subprocess to local
binance-cli + binance-pro-cli installations (set BINANCE_CLI / BINANCE_PRO_CLI
env vars to override).

Usage::

    from cyqnt_trd.data_cli import fetch_klines, fetch_funding_rate

    df = fetch_klines("BTCUSDT", "1h", limit=500)
    funding_df = fetch_funding_rate("BTCUSDT")
"""

from ._subprocess import run_binance_cli, run_binance_pro_cli
from ._cache import cache_get, cache_set, cache_clear

# Market data (binance-cli)
from .kline import fetch_klines, fetch_klines_multi_tf
from .ticker import fetch_24h_ticker, fetch_ticker_price, fetch_price
from .funding import fetch_funding_rate, fetch_funding_history
from .oi import fetch_open_interest, fetch_oi_history
from .orderbook import fetch_orderbook_depth, orderbook_imbalance
from .ratios import fetch_long_short_ratio
from .account import fetch_account_balance, fetch_positions
from .scanner import full_market_scan, scan_with_filter

# PUBLIC Binance Square news / social (vendored stdlib client)
from .news import (
    fetch_news,
    fetch_sentiment,
    fetch_ticker_rank,
    fetch_topic_trending,
    fetch_hot_post,
)

# Generic, configurable REST → typed DataFrame fetcher (no hardcoded endpoints;
# point it at any JSON API/website via a RestSourceSpec or an external config).
from .rest_source import (
    fetch_rest,
    register_spec,
    get_spec,
    load_specs,
    list_specs,
    RestSourceSpec,
    FieldSpec,
    EXAMPLE_SPECS,
)

# AI/workflow data (binance-pro-cli)
from .pro import pro_indicators_fetch, pro_trade_signal_query, pro_trade_signal_rank
from .workflow import workflow_leaderboard, workflow_token, workflow_analysis


__all__ = [
    # subprocess primitives
    "run_binance_cli",
    "run_binance_pro_cli",
    # cache
    "cache_get",
    "cache_set",
    "cache_clear",
    # klines
    "fetch_klines",
    "fetch_klines_multi_tf",
    # ticker
    "fetch_24h_ticker",
    "fetch_ticker_price",
    "fetch_price",
    # funding
    "fetch_funding_rate",
    "fetch_funding_history",
    # open interest
    "fetch_open_interest",
    "fetch_oi_history",
    # orderbook
    "fetch_orderbook_depth",
    "orderbook_imbalance",
    # ratios
    "fetch_long_short_ratio",
    # account
    "fetch_account_balance",
    "fetch_positions",
    # scanner
    "full_market_scan",
    "scan_with_filter",
    # news / social (PUBLIC Square)
    "fetch_news",
    "fetch_sentiment",
    "fetch_ticker_rank",
    "fetch_topic_trending",
    "fetch_hot_post",
    # generic configurable REST fetcher
    "fetch_rest",
    "register_spec",
    "get_spec",
    "load_specs",
    "list_specs",
    "RestSourceSpec",
    "FieldSpec",
    "EXAMPLE_SPECS",
    # AI signals (binance-pro-cli)
    "pro_indicators_fetch",
    "pro_trade_signal_query",
    "pro_trade_signal_rank",
    # workflow
    "workflow_leaderboard",
    "workflow_token",
    "workflow_analysis",
]
