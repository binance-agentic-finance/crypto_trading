"""Live trading daemon entry point — identical to paper_trade.py except
build_order_router(mode="live") picks the Kafka / atomic_lib router."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOT_DIR.parent))

from demo_strategy._shared.bot import (
    Daemon, load_bot_config, build_data_adapter, build_order_router,
)
from scripts import template


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_bot_config(BOT_DIR)
    cfg["runtime"] = "live"

    d = Daemon(
        cfg=cfg,
        template_module=template,
        data_adapter=build_data_adapter(),
        order_router=build_order_router(mode="live"),
        kind="signal",
    )
    return d.run()


if __name__ == "__main__":
    sys.exit(main())
