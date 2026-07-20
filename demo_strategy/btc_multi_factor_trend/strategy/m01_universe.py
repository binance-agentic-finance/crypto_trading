"""Module ① UNIVERSE — pick which symbols to evaluate.

For BTC multi-factor trend this is usually a single symbol (BTCUSDT) but the
config allows a small basket (BTC/ETH) for demonstration purposes. This
module keeps the "what universe" concern isolated so that swapping in a
scanner-based universe later is trivial.
"""
from __future__ import annotations


def pick_universe(cfg: dict) -> list[str]:
    """Return the symbol list to evaluate.

    Reads `cfg.symbols` (list) — CLI can override via `--symbols` before
    we get here. Guarantees at least one symbol.
    """
    syms = cfg.get("symbols") or ["BTCUSDT"]
    if not isinstance(syms, (list, tuple)):
        raise ValueError(f"cfg.symbols must be a list, got {type(syms)}")
    return [str(s).upper() for s in syms if s]
