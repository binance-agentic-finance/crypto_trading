"""DaemonBase — shared lifecycle scaffolding for every daemon flavor.

Responsibilities kept here (framework, NOT template):
    - SIGTERM handling (clean shutdown → state.json::status = "stopped")
    - state.json flush cadence
    - events.jsonl / trades.jsonl append helpers (via StateBook)
    - safe-template invocation with fault isolation

Subclasses implement `_tick()` — that's the only method that differs.

UI card: "daemon · shared" — read-only, shows the lifecycle wiring.
"""
from __future__ import annotations

import logging
import signal
import time
from typing import Any, Callable

from demo_strategy._shared.bot.state_writer import StateBook

log = logging.getLogger(__name__)

_INTERVAL_SEC = {
    "1m":   60,
    "5m":   300,
    "15m":  900,
    "1h":   3600,
    "4h":   14_400,
    "1d":   86_400,
}


class DaemonBase:
    """Shared daemon lifecycle. Subclass implements `_tick()`."""

    def __init__(self, *, cfg: dict[str, Any],
                 template_module,
                 data_adapter,
                 order_router,
                 universe_provider: Callable[[dict], list[str]] | None = None):
        self.cfg               = cfg
        self.template          = template_module
        self.data              = data_adapter
        self.orders            = order_router
        self.universe_provider = universe_provider

        self.book = StateBook(cfg["_bot_dir"])
        self._running = True

        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT,  self._on_signal)

    # ── lifecycle ────────────────────────────────────────────────────
    def _on_signal(self, signum, frame):
        log.info("Daemon received signal %s — stopping", signum)
        self._running = False

    def run(self) -> int:
        self.book.set_state(status="running")
        self.book.flush_state()
        self.book.append_event("started",
            runtime=self.cfg.get("runtime"),
            template=self.cfg.get("template_id"),
            symbol=self.cfg.get("symbol"),
            interval=self.cfg.get("interval"),
        )

        interval = self.cfg.get("interval", "1h")
        sleep_sec = min(60, _INTERVAL_SEC.get(interval, 3600) // 4)

        try:
            while self._running:
                self._safe_tick()
                for _ in range(sleep_sec):
                    if not self._running:
                        break
                    time.sleep(1)
        except Exception as e:   # noqa: BLE001
            log.exception("Daemon loop crashed: %s", e)
            self.book.set_state(status="error")
            self.book.flush_state()
            self.book.append_event("error", message=str(e), etype=type(e).__name__)
            return 1

        self.book.set_state(status="stopped")
        self.book.flush_state()
        self.book.append_event("stopped")
        return 0

    # ── per-tick ─────────────────────────────────────────────────────
    def _safe_tick(self) -> None:
        try:
            self._tick()
        except Exception as e:   # noqa: BLE001
            log.warning("tick failed: %s", e)
            self.book.append_event(
                "error", where="tick",
                message=str(e), etype=type(e).__name__,
            )

    def _tick(self) -> None:
        raise NotImplementedError("subclass must implement _tick")

    # ── template call wrapper with fault isolation ───────────────────
    def _safe_call(self, method_name: str, ctx, fallback):
        """Invoke `self.template.<method_name>(ctx)`; on any error emit
        a template_error event and return the caller-provided fallback."""
        fn = getattr(self.template, method_name, None)
        if fn is None:
            self.book.append_event(
                "template_error", where=method_name,
                etype="AttributeError",
                message=f"template has no {method_name}",
            )
            return fallback
        try:
            return fn(ctx)
        except Exception as e:  # noqa: BLE001
            log.warning("template.%s raised: %s", method_name, e)
            self.book.append_event(
                "template_error", where=method_name,
                etype=type(e).__name__, message=str(e),
            )
            return fallback
