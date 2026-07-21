"""Paper daemon entry — cross-sectional Square Buzz Screener.

Uses `SelectionDaemon` directly (attention metadata is stashed on the
cfg by the universe_provider and picked up by the daemon).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOT_DIR.parent))

from demo_strategy._shared.bot import (
    SelectionDaemon, load_bot_config,
    build_data_adapter, build_order_router,
)
from scripts import template, universe_source


def _universe_provider(cfg):
    """Called by SelectionDaemon each tick: return symbols + stash
    attention metadata on cfg for the template."""
    syms, meta = universe_source.get_universe(cfg)
    cfg["_universe_meta_cache"] = meta
    return syms


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_bot_config(BOT_DIR)
    cfg["runtime"] = "paper"

    d = SelectionDaemon(
        cfg=cfg,
        template_module=template,
        data_adapter=build_data_adapter(),
        order_router=build_order_router(mode="paper"),
        universe_provider=_universe_provider,
    )
    return d.run()


if __name__ == "__main__":
    sys.exit(main())
