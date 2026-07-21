"""SignalDaemon — single-symbol daemon for templates that expose
`calculate_signal(ctx: StrategyContext) → StrategyDecision`.

UI card: "daemon · single-symbol" — connects to the "template output"
StrategyDecision card by drawing a line at order dispatch.
"""
from __future__ import annotations

import time

from demo_strategy._shared.blocks.contracts import (
    MarketData, AccountData, StrategyContext, StrategyDecision, flat_decision,
)
from .base import DaemonBase


class SignalDaemon(DaemonBase):

    def _tick(self) -> None:
        # ── read 6-section business config ──────────────────────────
        data_cfg = self.cfg.get("data", {})
        symbol   = data_cfg.get("symbol", self.cfg.get("symbol"))
        interval = data_cfg.get("interval", self.cfg.get("interval"))
        limit    = data_cfg.get("bars_limit", self.cfg.get("bars_limit", 250))

        # ── data fetch (framework side, not template) ────────────────
        bars = self.data.fetch_bars(symbol, interval, limit=limit)
        if bars is None or bars.empty:
            self.book.append_event("data_gap", symbol=symbol, interval=interval)
            return

        close_time_ms = int(bars.index[-1])
        if close_time_ms <= self.book.state().get("last_bar_ts", 0):
            return   # already processed this bar

        # ── build context and invoke template ────────────────────────
        # We hand the WHOLE strategy config to the template so it can
        # navigate the 6 business sections (indicators/sizing/risk/etc.).
        ctx = StrategyContext(
            market=MarketData(bars=bars, symbol=symbol, timeframe=interval),
            account=AccountData(),
            config=self.cfg,
            bar_index=len(bars) - 1,
            close_time=close_time_ms,
        )
        decision: StrategyDecision = self._safe_call(
            "calculate_signal", ctx,
            fallback=flat_decision("template failure"),
        )
        if not isinstance(decision, StrategyDecision):
            decision = flat_decision("non-StrategyDecision returned")

        # ── persist signal + optionally dispatch order ───────────────
        self.book.set_state(last_bar_ts=close_time_ms,
                            last_signal_ts=int(time.time() * 1000))
        self.book.append_event(
            "signal", symbol=symbol,
            side=decision.side, strength=decision.strength,
            reason=decision.reason, close_time=close_time_ms,
        )
        if decision.side in ("long", "short"):
            self._route_single(symbol, decision,
                               entry=float(bars["close"].iloc[-1]))
        self.book.flush_state()

    # ── order dispatch (SignalDaemon-specific) ────────────────────────
    def _route_single(self, symbol: str,
                      decision: StrategyDecision, entry: float) -> None:
        exe_cfg   = self.cfg.get("execution", {})
        risk_cfg  = self.cfg.get("risk", {})
        # stop_pct now lives under risk.stop_loss (with algo=fixed_pct); we
        # fall back to legacy execution.stop_pct for older configs.
        stop_pct  = (risk_cfg.get("stop_loss", {}).get("stop_pct")
                     or exe_cfg.get("stop_pct", 3.0)) / 100.0
        qty_usdt  = exe_cfg.get("qty_usdt", 100)
        qty       = qty_usdt / entry if entry > 0 else 0
        stop      = entry * (1 - stop_pct) if decision.side == "long" \
                    else entry * (1 + stop_pct)

        res = self.orders.submit_single(
            symbol=symbol, side=decision.side, qty=qty,
            entry=entry, stop=stop,
            mode=exe_cfg.get("mode", "futures"),
            metadata={
                "strategy_id": self.cfg.get("template_id"),
                "bot_name":    self.cfg.get("name"),
                "reason":      decision.reason,
                "profile":     exe_cfg.get("profile", "default"),
            },
        )
        self.book.append_event(
            "order_placed", symbol=symbol,
            side=decision.side, qty=qty, entry=entry, stop=stop, result=res,
        )
        if res.get("status") == "filled":
            self.book.append_trade(
                symbol=symbol, side=decision.side, qty=qty, price=entry,
            )
