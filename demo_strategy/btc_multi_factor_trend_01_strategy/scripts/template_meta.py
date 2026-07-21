"""TEMPLATE_META — declared metadata for the control-plane / UI.

UI card: "strategy identity" — strategy_id, display_name, version,
description, config_schema (renders the config form).
Kept in its own file so editing metadata doesn't touch template logic.
"""
from __future__ import annotations

from demo_strategy._shared.blocks import TemplateMeta


TEMPLATE_META = TemplateMeta(
    strategy_id  = "demo_btc_multi_factor_trend",
    display_name = "BTC Multi-Factor Trend (Demo, block-composed)",
    version      = "2.1.0",
    author       = "crypto_trading.demo_strategy",
    description  = (
        "Composes 5 signal blocks (ema/rsi/macd/atr/resonance) with 2 "
        "scoring blocks (hierarchical/verdict_gate) and 1 decision block "
        "(direction_vote). Per-tier scorers and per-source direction voters "
        "each live in their own file — tune via YAML, no code edit."
    ),
    config_schema = {
        "type": "object",
        "properties": {
            "signals":  {"type": "object"},
            "scoring":  {"type": "object"},
            "decision": {"type": "object"},
            "tiers":    {"type": "object"},
        },
    },
)
