"""TEMPLATE_META — declared metadata for Square Buzz Screener.

UI card: "strategy identity".
"""
from __future__ import annotations

from demo_strategy._shared.blocks import TemplateMeta


TEMPLATE_META = TemplateMeta(
    strategy_id  = "demo_square_buzz_screener",
    display_name = "Binance Square Buzz Screener (Demo, block-composed)",
    version      = "2.1.0",
    author       = "crypto_trading.demo_strategy",
    description  = (
        "Cross-sectional attention + technical composite. 2 attention "
        "blocks + 5 technical blocks. Each tier scorer and the AVOID-verdict "
        "gate live in separate files — tune via YAML, no code edit."
    ),
    config_schema = {
        "type": "object",
        "properties": {
            "signals": {"type": "object"},
            "scoring": {"type": "object"},
            "tiers":   {"type": "object"},
        },
    },
)
