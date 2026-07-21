"""DryRunRouter — paper mode / dev; logs intent, no side effects.

UI card: "execution · dry-run" — always-on fallback used for paper &
backtest modes.
"""
from __future__ import annotations

import time


class DryRunRouter:
    def submit_single(self, *, symbol, side, qty, entry, stop, mode, metadata):
        return {"status": "dry_run", "ts": time.time(),
                "symbol": symbol, "side": side, "qty": qty,
                "entry": entry, "stop": stop, "mode": mode}

    def submit_portfolio(self, *, weights, notional_usdt, mode, metadata):
        return {"status": "dry_run", "ts": time.time(),
                "weights": weights, "notional_usdt": notional_usdt,
                "mode": mode}
