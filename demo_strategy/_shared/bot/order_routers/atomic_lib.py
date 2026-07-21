"""AtomicLibRouter — direct binance-cli execution via atomic_strategy_lib.

UI card: "execution · atomic_lib" — local live-trade path when outside
bdp-bot cluster but binance-cli profile is configured.
"""
from __future__ import annotations


class AtomicLibRouter:
    """Uses `atomic_strategy_lib.execution.orders.{market_order, stop_order}`."""

    def __init__(self):
        from atomic_strategy_lib.execution.orders import (   # noqa: PLC0415
            market_order, stop_order,
        )
        self._market_order = market_order
        self._stop_order   = stop_order

    def submit_single(self, *, symbol, side, qty, entry, stop, mode, metadata):
        entry_res = self._market_order(
            symbol, side=side.upper(), qty=qty, market=mode,
            profile=metadata.get("profile", "default"),
        )
        stop_res = self._stop_order(
            symbol,
            side="SELL" if side.lower() == "long" else "BUY",
            qty=qty, stop_price=stop, market=mode,
            profile=metadata.get("profile", "default"),
        )
        return {"status": "filled", "entry": entry_res, "stop": stop_res}

    def submit_portfolio(self, *, weights, notional_usdt, mode, metadata):
        outcomes = []
        for sym, w in weights.items():
            if w <= 0:
                continue
            qty_usdt = notional_usdt * w
            r = self._market_order(
                sym, side="BUY", qty=None, quote_qty=qty_usdt,
                market=mode, profile=metadata.get("profile", "default"),
            )
            outcomes.append({"symbol": sym, "weight": w, "result": r})
        return {"status": "filled", "outcomes": outcomes}
