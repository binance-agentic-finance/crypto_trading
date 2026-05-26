"""
binance-pro-cli workflow commands: leaderboard, token, analysis, briefing, etc.

Atomic source: atomic_strategy_lib/data/workflow.py
CLI commands replicated (all via binance-pro-cli):
  workflow leaderboard:   binance-pro-cli workflow leaderboard --market M --lang en
  workflow token:         binance-pro-cli workflow token --symbol S --market M --product P
  workflow analysis:      binance-pro-cli workflow analysis --symbol S --product P ...
  workflow tradingbot:    binance-pro-cli workflow tradingbot --symbol S --product P ...
  workflow top-strategies:binance-pro-cli workflow top-strategies --direction D
  workflow smart-money:   binance-pro-cli workflow smart-money --chain C [--symbol S]
  workflow briefing:      binance-pro-cli workflow briefing --symbols S --chain C
  workflow arb-scan:      binance-pro-cli workflow arb-scan --symbols BTC,ETH

All functions return raw dict — no DataFrame needed for nested workflow results.
"""

from __future__ import annotations

from typing import Optional

from ._subprocess import run_binance_pro_cli


def workflow_leaderboard(
    market: str = "all",
    symbol: Optional[str] = None,
    lang: str = "en",
) -> dict:
    """
    Run the discovery workflow — market leaderboard.

    CLI::

        binance-pro-cli workflow leaderboard --market all --lang en

    Parameters
    ----------
    market : str
        ``"all"``, ``"alpha"``, ``"spot"``, ``"futures"``

    Returns
    -------
    dict
    """
    args = ["workflow", "leaderboard", "--market", market, "--lang", lang]
    if symbol:
        args.extend(["--symbol", symbol])
    return run_binance_pro_cli(args)


def workflow_token(
    symbol: str,
    market: str = "all",
    product: str = "spot",
) -> dict:
    """
    Run the token research workflow (meta + dynamic in one call).

    CLI::

        binance-pro-cli workflow token --symbol BTC --market all --product spot

    Returns
    -------
    dict
    """
    args = ["workflow", "token",
            "--symbol", symbol, "--market", market, "--product", product]
    return run_binance_pro_cli(args)


def workflow_analysis(
    symbol: str,
    product: str = "um-perp",
    interval: str = "1h",
    limit: int = 5,
) -> dict:
    """
    Decision analysis workflow with context-safe defaults.

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


def workflow_tradingbot(
    symbol: str,
    product: str = "um-perp",
    direction: Optional[str] = None,
    interval: str = "1h",
    limit: int = 5,
) -> dict:
    """
    Bot-ready trading spec generator (does NOT place trades).

    CLI::

        binance-pro-cli workflow tradingbot \\
            --symbol BTCUSDT --product um-perp --interval 1h --limit 5

    Returns
    -------
    dict
    """
    args = ["workflow", "tradingbot",
            "--symbol", symbol, "--product", product,
            "--interval", interval, "--limit", str(limit)]
    if direction:
        args.extend(["--direction", direction])
    return run_binance_pro_cli(args)


def workflow_top_strategies(
    direction: str = "neutral",
) -> dict:
    """
    Top trading bot strategies across spot grid, futures grid, COIN-M, DCA.

    CLI::

        binance-pro-cli workflow top-strategies --direction neutral

    Parameters
    ----------
    direction : str
        ``"neutral"``, ``"long"``, ``"short"``

    Returns
    -------
    dict
    """
    args = ["workflow", "top-strategies", "--direction", direction]
    return run_binance_pro_cli(args)


def workflow_smart_money(
    symbol: Optional[str] = None,
    chain: str = "CT_501",
) -> dict:
    """
    Aggregated smart money view: on-chain signals + inflow ranking.

    CLI::

        binance-pro-cli workflow smart-money --chain CT_501 [--symbol BTC]

    Returns
    -------
    dict
    """
    args = ["workflow", "smart-money", "--chain", chain]
    if symbol:
        args.extend(["--symbol", symbol])
    return run_binance_pro_cli(args)


def workflow_briefing(
    symbols: str,
    chain: str = "CT_501",
    wallet: Optional[str] = None,
) -> dict:
    """
    Morning briefing: smart money + futures features + optional wallet position.

    CLI::

        binance-pro-cli workflow briefing --symbols BTC,ETH,SOL --chain CT_501

    Parameters
    ----------
    symbols : str
        Comma-separated base symbols, e.g. ``"BTC,ETH,SOL"``.

    Returns
    -------
    dict
    """
    args = ["workflow", "briefing", "--symbols", symbols, "--chain", chain]
    if wallet:
        args.extend(["--wallet", wallet])
    return run_binance_pro_cli(args)


def workflow_arb_scan(
    symbols: list[str],
) -> dict:
    """
    Futures-spot premium + funding rate + long-short scan.

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


# ---------------------------------------------------------------------------
# Smart money inflow alias (used by atomic scanner)
# ---------------------------------------------------------------------------

def smart_money_inflow(
    symbol: Optional[str] = None,
    chain: str = "CT_501",
) -> dict:
    """Alias for :func:`workflow_smart_money`."""
    return workflow_smart_money(symbol=symbol, chain=chain)
