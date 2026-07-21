"""Live daemon entry — same wiring as paper_trade with a live OrderRouter."""
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
from scripts import template
from scripts.paper_trade import _universe_provider


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_bot_config(BOT_DIR)
    cfg["runtime"] = "live"

    d = SelectionDaemon(
        cfg=cfg,
        template_module=template,
        data_adapter=build_data_adapter(),
        order_router=build_order_router(mode="live"),
        universe_provider=_universe_provider,
    )
    return d.run()


if __name__ == "__main__":
    sys.exit(main())
