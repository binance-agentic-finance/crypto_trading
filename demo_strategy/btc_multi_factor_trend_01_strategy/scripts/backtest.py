"""Backtest — one-shot walk over historical bars.

`bash run.sh backtest` → python3 -m scripts.backtest
Reads N days of bars, replays `calculate_signal` on every bar close,
emits a summary JSON + prints a mini leaderboard.
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
from demo_strategy._shared.blocks.contracts import (
    MarketData, AccountData, StrategyContext,
)
from scripts import template


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_bot_config(BOT_DIR)
    symbol, interval = cfg["symbol"], cfg["interval"]

    data = build_data_adapter()
    bars = data.fetch_bars(symbol, interval, limit=cfg.get("bars_limit", 500))
    if bars is None or bars.empty:
        print(json.dumps({"error": "no bars"}))
        return 1

    # replay: for each bar index, provide bars[:i+1] to the template
    trades = []
    open_side = "flat"
    open_entry = 0.0
    equity = 0.0
    n_signals = 0

    for i in range(60, len(bars)):
        window = bars.iloc[:i + 1]
        ctx = StrategyContext(
            market=MarketData(bars=window, symbol=symbol, timeframe=interval),
            account=AccountData(),
            config=cfg.get("params", {}),
            bar_index=len(window) - 1,
            close_time=int(window.index[-1]),
        )
        d = template.calculate_signal(ctx)
        n_signals += 1
        if d.side == open_side:
            continue
        # side change → close prev (if any) + open new
        price_now = float(window["close"].iloc[-1])
        if open_side in ("long", "short"):
            pnl = (price_now - open_entry) / open_entry
            if open_side == "short":
                pnl = -pnl
            equity += pnl
            trades.append({
                "close_ts": int(window.index[-1]), "side": open_side,
                "entry": open_entry, "exit": price_now, "pnl": pnl,
            })
        open_side  = d.side
        open_entry = price_now if d.side in ("long", "short") else 0.0

    summary = {
        "strategy":   cfg.get("template_id"),
        "symbol":     symbol,
        "interval":   interval,
        "n_bars":     len(bars),
        "n_signals":  n_signals,
        "n_trades":   len(trades),
        "equity_ret": round(equity, 4),
        "trades_head": trades[:5],
    }
    out = BOT_DIR / "state" / f"backtest_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
