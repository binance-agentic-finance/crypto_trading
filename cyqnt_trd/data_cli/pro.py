"""
binance-pro-cli exclusive data: search, workflow, web3 commands.

Atomic source: atomic_strategy_lib/data/pro.py
CLI commands replicated (all via binance-pro-cli):
  search indicators:     binance-pro-cli search indicators --symbol S --product P ...
  search tradesignal:    binance-pro-cli search tradesignal-query --symbol S
  search signal rank:    binance-pro-cli search tradesignal-rank
  search trending:       binance-pro-cli search trending
  workflow arb-scan:     binance-pro-cli workflow arb-scan --symbols BTC,ETH,SOL
  workflow analysis:     binance-pro-cli workflow analysis --symbol S --product P ...

All functions return raw dict/list — no DataFrame conversion needed since
these are complex nested structures whose schema varies by command.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ._subprocess import run_binance_pro_cli
from ._cache import cache_get, cache_set, TTL_PRO


def pro_indicators_fetch(
    symbol: str,
    product: str = "um-perp",
    interval: str = "1h",
    limit: int = 5,
    indicator_ids: Optional[Sequence[str]] = None,
    indicators_json: Optional[str] = None,
    allow_partial: bool = False,
    *,
    ttl: int = TTL_PRO,
    refresh: bool = False,
) -> dict:
    """
    Fetch computed indicators from binance-pro-cli.

    CLI::

        binance-pro-cli search indicators \\
            --symbol BTCUSDT --product um-perp --interval 1h --limit 5

    Parameters
    ----------
    product : str
        ``"spot"``, ``"um-perp"``, ``"web3-alpha"``
    indicator_ids : list[str], optional
        Specific indicator IDs to fetch.
    indicators_json : str, optional
        JSON string of indicator config.
    allow_partial : bool
        Pass ``--allow-partial`` flag.

    Returns
    -------
    dict
        Raw indicator payload.
    """
    key = ("pro_indicators", symbol, product, interval, limit)
    if not refresh:
        cached = cache_get(key)
        if cached is not None and not cached.empty:
            return cached.iloc[0].to_dict()

    args = ["search", "indicators",
            "--symbol", symbol, "--product", product,
            "--interval", interval, "--limit", str(limit)]
    if indicator_ids:
        args.extend(["--indicator-ids", *indicator_ids])
    if indicators_json:
        args.extend(["--indicators-json", indicators_json])
    if allow_partial:
        args.append("--allow-partial")

    return run_binance_pro_cli(args)


def pro_trade_signal_query(
    symbol: str,
) -> dict:
    """
    Fetch AI trade signal for a specific token.

    CLI::

        binance-pro-cli search tradesignal-query --symbol BTCUSDT

    Returns
    -------
    dict
    """
    args = ["search", "tradesignal-query", "--symbol", symbol]
    return run_binance_pro_cli(args)


def pro_trade_signal_rank() -> list:
    """
    Fetch ranked list of tokens with active AI trade signals.

    CLI::

        binance-pro-cli search tradesignal-rank

    Returns
    -------
    list[dict]
    """
    args = ["search", "tradesignal-rank"]
    raw = run_binance_pro_cli(args)
    return raw if isinstance(raw, list) else []


def pro_trending() -> dict:
    """
    Fetch trending topics / narrative heat.

    CLI::

        binance-pro-cli search trending

    Returns
    -------
    dict
    """
    args = ["search", "trending"]
    return run_binance_pro_cli(args)


def pro_arb_scan(
    symbols: list[str],
) -> dict:
    """
    Aggregated arbitrage scan: basis, funding, L/S ratio, OI.

    CLI::

        binance-pro-cli workflow arb-scan --symbols BTC,ETH,SOL

    Parameters
    ----------
    symbols : list[str]
        Base symbols (e.g. ``["BTC", "ETH", "SOL"]``).

    Returns
    -------
    dict
    """
    args = ["workflow", "arb-scan", "--symbols", ",".join(symbols)]
    return run_binance_pro_cli(args)


def pro_analysis(
    symbol: str,
    product: str = "um-perp",
    interval: str = "1h",
    limit: int = 5,
    *,
    ttl: int = TTL_PRO,
    refresh: bool = False,
) -> dict:
    """
    Full workflow analysis with context-safe defaults.

    CLI::

        binance-pro-cli workflow analysis \\
            --symbol BTCUSDT --product um-perp --interval 1h --limit 5

    Returns
    -------
    dict
    """
    args = ["workflow", "analysis",
            "--symbol", symbol, "--product", product,
            "--interval", interval, "--limit", str(limit)]
    return run_binance_pro_cli(args)
