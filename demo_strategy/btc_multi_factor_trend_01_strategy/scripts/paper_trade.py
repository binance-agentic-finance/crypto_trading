"""Paper trading daemon entry point for btc_multi_factor_trend.

`bash run.sh paper-fg` → `python3 -m scripts.paper_trade`
The daemon reads config, builds adapters, and hands the template to
Daemon.run() — see demo_strategy._shared.bot.daemon.

Same file is called for both paper and live; the ONLY difference is the
`runtime` in config.yaml (or the runtime the supervisor requests, which
overrides the config value).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# scripts/ dir → parent bot_dir → parent demo_strategy/ → PYTHONPATH
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
    # paper daemon forces paper runtime + DryRunRouter regardless of yaml
    cfg["runtime"] = "paper"

    d = Daemon(
        cfg=cfg,
        template_module=template,
        data_adapter=build_data_adapter(),
        order_router=build_order_router(mode="paper"),
        kind="signal",
    )
    return d.run()


if __name__ == "__main__":
    sys.exit(main())
