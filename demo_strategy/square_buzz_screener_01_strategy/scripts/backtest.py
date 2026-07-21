"""Backtest — replay historical selection ticks over a fixed universe.

Because attention data isn't historicalizable (past Square scrapes aren't
persisted), backtest mode uses a hand-picked static basket from config
+ zeroed attention meta. The technical scoring still runs realistically.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOT_DIR.parent))

from demo_strategy._shared.bot import load_bot_config, build_data_adapter
from demo_strategy._shared.blocks.contracts import SelectionContext
from scripts import template


_BACKTEST_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_bot_config(BOT_DIR)
    data = build_data_adapter()

    bars_by = {}
    for s in _BACKTEST_UNIVERSE:
        df = data.fetch_bars(s, cfg["interval"], limit=cfg.get("bars_limit", 200))
        if df is not None and not df.empty:
            bars_by[s] = df

    if not bars_by:
        print(json.dumps({"error": "no bars"}))
        return 1

    latest_close = max(int(df.index[-1]) for df in bars_by.values())
    ctx = SelectionContext(
        universe=list(bars_by.keys()),
        bars_by_symbol=bars_by,
        timeframe=cfg["interval"],
        config=cfg.get("params", {}),
        close_time=latest_close,
        metadata={
            # backtest without live attention scrape → empty meta
            "attention":       {s: {} for s in bars_by},
            "max_basket_size": cfg.get("execution", {}).get("max_basket_size", 5),
        },
    )
    d = template.calculate_selection(ctx)
    summary = {
        "strategy":   cfg.get("template_id"),
        "universe":   list(bars_by.keys()),
        "close_time": latest_close,
        "basket":     d.weights,
        "reason":     d.reason,
        "metadata":   d.metadata,
    }
    out = BOT_DIR / "state" / f"backtest_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
