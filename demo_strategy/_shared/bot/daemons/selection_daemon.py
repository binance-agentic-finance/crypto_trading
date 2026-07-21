"""SelectionDaemon — cross-sectional daemon for templates that expose
`calculate_selection(ctx: SelectionContext) → SelectionDecision`.

Differs from SignalDaemon in three ways:
    - calls a universe_provider() each tick
    - fetches bars for N symbols (not one)
    - routes as PortfolioTargetCommand (not per-symbol OrderCommand)

UI card: "daemon · selection" — pairs with "universe source" card.
"""
from __future__ import annotations

import time
from typing import Callable

from demo_strategy._shared.blocks.contracts import (
    SelectionContext, SelectionDecision,
)
from .base import DaemonBase


class SelectionDaemon(DaemonBase):
    def __init__(self, *,
                 cfg: dict,
                 template_module,
                 data_adapter,
                 order_router,
                 universe_provider: Callable[[dict], list[str]]):
        """`universe_provider(cfg)` MUST return a symbol list. It may also
        mutate `cfg['_universe_meta_cache']` to stash attention metadata
        that the template reads back via ctx.metadata."""
        if universe_provider is None:
            raise ValueError("SelectionDaemon needs universe_provider")
        super().__init__(cfg=cfg, template_module=template_module,
                         data_adapter=data_adapter, order_router=order_router,
                         universe_provider=universe_provider)

    def _tick(self) -> None:
        data_cfg = self.cfg.get("data", {})
        interval = data_cfg.get("interval", self.cfg.get("interval"))
        limit    = data_cfg.get("bars_limit", self.cfg.get("bars_limit", 200))

        universe = self.universe_provider(self.cfg)
        if not universe:
            self.book.append_event("empty_universe")
            return

        # per-symbol bars fetch
        bars_by = {}
        latest_close = 0
        for sym in universe:
            df = self.data.fetch_bars(sym, interval, limit=limit)
            if df is None or df.empty:
                continue
            bars_by[sym] = df
            latest_close = max(latest_close, int(df.index[-1]))

        if not bars_by:
            self.book.append_event("data_gap", universe_size=len(universe))
            return
        if latest_close <= self.book.state().get("last_bar_ts", 0):
            return

        # metadata passthrough (attention, etc.)
        sizing = self.cfg.get("sizing", {}).get("basket", {})
        meta = {
            "attention":       self.cfg.get("_universe_meta_cache", {}),
            "max_basket_size": sizing.get("max_size",
                                          self.cfg.get("execution", {}).get("max_basket_size", 5)),
        }
        ctx = SelectionContext(
            universe=list(bars_by.keys()),
            bars_by_symbol=bars_by,
            timeframe=interval,
            config=self.cfg,       # hand whole cfg to template (6-section access)
            close_time=latest_close,
            metadata=meta,
        )
        decision: SelectionDecision = self._safe_call(
            "calculate_selection", ctx,
            fallback=SelectionDecision(weights={}, reason="template failure"),
        )
        if not isinstance(decision, SelectionDecision):
            decision = SelectionDecision(weights={},
                                          reason="non-SelectionDecision returned")

        self.book.set_state(last_bar_ts=latest_close,
                            last_signal_ts=int(time.time() * 1000))
        self.book.append_event(
            "signal", kind="selection",
            n_positions=len(decision.weights),
            reason=decision.reason, close_time=latest_close,
        )
        if decision.weights:
            self._route_portfolio(decision)
        self.book.flush_state()

    def _route_portfolio(self, decision: SelectionDecision) -> None:
        exe_cfg = self.cfg.get("execution", {})
        res = self.orders.submit_portfolio(
            weights=decision.weights,
            notional_usdt=exe_cfg.get("notional_usdt", 1000),
            mode=exe_cfg.get("mode", "futures"),
            metadata={
                "strategy_id": self.cfg.get("template_id"),
                "bot_name":    self.cfg.get("name"),
                "reason":      decision.reason,
                "profile":     exe_cfg.get("profile", "default"),
                **(decision.metadata or {}),
            },
        )
        self.book.append_event(
            "order_placed", kind="selection",
            weights=decision.weights, result=res,
        )
