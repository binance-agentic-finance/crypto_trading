"""shim — atomic.data.web3

Forward to ``binance-pro-cli web3 ...`` subcommands. The previous
implementation was a global stub returning empty lists/dicts, which made
ondo-daily / earnings-tradfi-scanner / macro-event-driven-scanner all
look like "no Ondo tokens exist" silently.

We now actually call the CLI for the most-used endpoints:

  - ``web3_tokenized_list``    -> ``binance-pro-cli web3 tokenized list``
  - ``web3_tokenized_market``  -> ``binance-pro-cli web3 tokenized market``
  - ``web3_tokenized_dynamic`` -> ``binance-pro-cli web3 tokenized dynamic``
  - ``web3_rank``              -> ``binance-pro-cli web3 rank ...``
  - ``smart_money_inflow_rank``-> ``binance-pro-cli web3 signal ...``

Less-common names fall through to ``__getattr__`` and return the safe
empty default rather than crashing.
"""
from __future__ import annotations

from typing import Optional

from cyqnt_trd.data_cli._subprocess import run_binance_pro_cli, CLIError


def _unwrap(raw):
    """binance-pro-cli wraps payloads in {'capability', 'success', 'data', ...}.
    Return raw['data'] when present, else raw."""
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    return raw


def web3_tokenized_list(token_type: str | int | None = None,
                        profile: Optional[str] = None,
                        binary: str = "binance-pro-cli",
                        **_legacy) -> list[dict]:
    """List all tokenized stock tokens (Ondo)."""
    args = ["web3", "tokenized", "list", "--no-compact"]
    if token_type is not None:
        args += ["--type", str(token_type)]
    try:
        raw = run_binance_pro_cli(args)
    except CLIError:
        return []
    data = _unwrap(raw)
    return data if isinstance(data, list) else []


def web3_tokenized_market(profile: Optional[str] = None,
                          binary: str = "binance-pro-cli",
                          **_legacy) -> dict:
    """Overall Ondo market open/close status."""
    args = ["web3", "tokenized", "market", "--no-compact"]
    try:
        raw = run_binance_pro_cli(args)
    except CLIError:
        return {}
    return raw if isinstance(raw, dict) else {}


def web3_tokenized_dynamic(chain: str | None = None,
                           address: str | None = None,
                           profile: Optional[str] = None,
                           binary: str = "binance-pro-cli",
                           **_legacy) -> dict:
    """Real-time on-chain + US stock fundamentals data for a token."""
    if not chain or not address:
        return {}
    args = ["web3", "tokenized", "dynamic",
            "--chain", str(chain), "--address", str(address),
            "--no-compact"]
    try:
        raw = run_binance_pro_cli(args)
    except CLIError:
        return {}
    return raw if isinstance(raw, dict) else {}


def web3_tokenized_meta(chain: str | None = None,
                        address: str | None = None,
                        profile: Optional[str] = None,
                        binary: str = "binance-pro-cli",
                        **_legacy) -> dict:
    """Tokenized stock metadata and company info."""
    if not chain or not address:
        return {}
    args = ["web3", "tokenized", "meta",
            "--chain", str(chain), "--address", str(address),
            "--no-compact"]
    try:
        raw = run_binance_pro_cli(args)
    except CLIError:
        return {}
    return raw if isinstance(raw, dict) else {}


def web3_tokenized_asset(chain: str | None = None,
                         address: str | None = None,
                         profile: Optional[str] = None,
                         binary: str = "binance-pro-cli",
                         **_legacy) -> dict:
    """Per-asset trading status with corporate action codes."""
    if not chain or not address:
        return {}
    args = ["web3", "tokenized", "asset",
            "--chain", str(chain), "--address", str(address),
            "--no-compact"]
    try:
        raw = run_binance_pro_cli(args)
    except CLIError:
        return {}
    return raw if isinstance(raw, dict) else {}


def web3_rank(rank_type: str | None = None,
              profile: Optional[str] = None,
              binary: str = "binance-pro-cli",
              **_legacy) -> list[dict]:
    """Crypto market rankings (web3 rank)."""
    args = ["web3", "rank", "--no-compact"]
    if rank_type:
        args += ["--type", str(rank_type)]
    try:
        raw = run_binance_pro_cli(args)
    except CLIError:
        return []
    data = _unwrap(raw)
    return data if isinstance(data, list) else []


def smart_money_inflow_rank(*args, **kwargs) -> list[dict]:
    """Smart money on-chain trading signals (web3 signal)."""
    cli_args = ["web3", "signal", "--no-compact"]
    try:
        raw = run_binance_pro_cli(cli_args)
    except CLIError:
        return []
    data = _unwrap(raw)
    return data if isinstance(data, list) else []


# Aliases / less-common names that case scripts sometimes import — keep
# them as no-ops returning empty lists so a missing endpoint doesn't
# crash the pipeline.
def _empty_list(*args, **kwargs):
    return []


social_hype_rank = _empty_list
meme_rank = _empty_list


def __getattr__(name):
    """Catch-all: any other web3 function returns a safe default."""
    if name.startswith("_"):
        raise AttributeError(name)
    return _empty_list
