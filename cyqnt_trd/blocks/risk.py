"""Portfolio-level risk control state machine.

This module turns the user-described risk rules (max loss per trade,
consecutive-loss pause, max drawdown halt, daily loss cap, max
positions, leverage limits, funding-rate windows, etc.) into a
**stateful** object that strategy code can query before opening a new
position and update after every closed trade.

Designed for both live and backtest use:

* :class:`RiskConfig` — declarative, immutable configuration.
* :class:`RiskState` — mutable runtime state (counters, pause flags, …).
* :class:`RiskGuard` — composes a config + state and exposes
  :meth:`can_open_new` / :meth:`on_trade_closed` / :meth:`on_equity_update`.

Examples
--------
>>> from cyqnt_trd.blocks import risk
>>> config = risk.RiskConfig(
...     max_loss_per_trade_pct=0.10,
...     consecutive_loss_pause=(3, 2 * 3600 * 1000),  # 3 losses → 2h pause
...     max_drawdown_halt_pct=0.50,
...     daily_max_loss_pct=0.08,
...     max_positions=3,
...     leverage=8,
...     margin_per_trade_pct=0.15,
...     min_available_margin_pct=0.50,
...     funding_buffer_min=15,
...     funding_settle_hours_utc=(0, 8, 16),
... )
>>> guard = risk.RiskGuard(config)
>>> ok, reason = guard.can_open_new(now_ms=now, equity=10_000, used_margin=2_000)
>>> if not ok:
...     print(f"Blocked: {reason}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

__all__ = [
    "RiskConfig",
    "RiskState",
    "RiskGuard",
    "is_funding_window",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_funding_window(
    now_ms: int,
    settle_hours_utc: Iterable[int] = (0, 8, 16),
    buffer_min: int = 15,
) -> bool:
    """Return True if *now_ms* falls within ``buffer_min`` minutes of any settlement hour.

    Used to block new entries around Binance USDT-M perpetual funding-rate
    settlements (00:00 / 08:00 / 16:00 UTC).
    """
    if buffer_min < 0:
        raise ValueError(f"buffer_min must be >= 0, got {buffer_min}")
    minute_of_day = (now_ms // 60_000) % (24 * 60)
    settle_minutes = [h * 60 for h in settle_hours_utc]
    return any(
        min(abs(minute_of_day - s), 24 * 60 - abs(minute_of_day - s)) <= buffer_min
        for s in settle_minutes
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskConfig:
    """Declarative risk-rule configuration.

    All fields are optional — set to ``None`` (or use the defaults) to
    disable that particular rule.
    """

    # -- per-trade --
    max_loss_per_trade_pct: Optional[float] = None
    """Max portion of equity that any single trade may lose. e.g. 0.10 = 10%."""

    # -- portfolio drawdown --
    max_drawdown_halt_pct: Optional[float] = None
    """Permanent stop if total equity drops by this fraction from the peak."""

    daily_max_loss_pct: Optional[float] = None
    """Pause new trades when day-loss exceeds this fraction of starting equity."""

    monthly_max_loss_pct: Optional[float] = None
    """Pause new trades when month-loss exceeds this fraction."""

    # -- streak / cooldown --
    consecutive_loss_pause: Optional[Tuple[int, int]] = None
    """``(loss_count, pause_ms)`` — pause new trades for ``pause_ms`` after streak."""

    per_symbol_cooldown: Optional[Tuple[int, int]] = None
    """``(loss_count, pause_ms)`` — same as above but per-symbol."""

    # -- capacity --
    max_positions: Optional[int] = None
    """Max number of simultaneously open positions."""

    leverage: int = 1
    """Maximum leverage multiplier (informational; pass to execution layer)."""

    margin_per_trade_pct: Optional[float] = None
    """Initial margin as fraction of equity per trade."""

    min_available_margin_pct: Optional[float] = None
    """Reject new trades if available margin would drop below this fraction."""

    total_exposure_cap_pct: Optional[float] = None
    """Total notional exposure cap as fraction of equity (incl. leverage)."""

    # -- funding-rate window --
    funding_buffer_min: int = 0
    funding_settle_hours_utc: Tuple[int, ...] = (0, 8, 16)

    # -- blacklist / blocked symbols --
    blacklist: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Mutable runtime state
# ---------------------------------------------------------------------------


@dataclass
class RiskState:
    """Mutable counters and timestamps tracked across the trading session.

    Most fields are updated by :class:`RiskGuard` automatically; user
    code rarely needs to touch them directly.
    """

    consecutive_losses: int = 0
    pause_until_ms: Optional[int] = None
    halted: bool = False

    open_positions: int = 0

    starting_equity: float = 0.0
    peak_equity: float = 0.0
    last_equity: float = 0.0

    daily_anchor_ts_ms: Optional[int] = None
    daily_start_equity: float = 0.0

    monthly_anchor_ts_ms: Optional[int] = None
    monthly_start_equity: float = 0.0

    per_symbol_losses: dict = field(default_factory=dict)
    per_symbol_pause_until: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


class RiskGuard:
    """Compose a :class:`RiskConfig` + :class:`RiskState` and check rules.

    Typical lifecycle::

        guard = RiskGuard(config)
        guard.on_equity_update(now_ms=t0, equity=10_000)   # initialise
        ok, reason = guard.can_open_new(
            now_ms=t1, symbol="BTCUSDT",
            equity=current_equity, used_margin=used,
        )
        if ok:
            # ... open the trade ...
            guard.on_position_opened()
        else:
            print("Blocked:", reason)

        # ... eventually:
        guard.on_trade_closed(symbol="BTCUSDT", pnl=-50.0, now_ms=t2)
    """

    def __init__(
        self, config: RiskConfig, state: Optional[RiskState] = None
    ) -> None:
        self.config = config
        self.state = state or RiskState()

    # ---- Equity tracking ----

    def on_equity_update(self, now_ms: int, equity: float) -> None:
        """Update equity snapshots — call on every bar / heartbeat."""
        s, c = self.state, self.config
        if s.starting_equity == 0.0:
            s.starting_equity = equity
            s.peak_equity = equity
            s.daily_anchor_ts_ms = now_ms
            s.daily_start_equity = equity
            s.monthly_anchor_ts_ms = now_ms
            s.monthly_start_equity = equity
        s.last_equity = equity
        s.peak_equity = max(s.peak_equity, equity)

        # daily / monthly anchor rollover
        if s.daily_anchor_ts_ms is not None and now_ms - s.daily_anchor_ts_ms >= 24 * 3600 * 1000:
            s.daily_anchor_ts_ms = now_ms
            s.daily_start_equity = equity
        if s.monthly_anchor_ts_ms is not None and now_ms - s.monthly_anchor_ts_ms >= 30 * 24 * 3600 * 1000:
            s.monthly_anchor_ts_ms = now_ms
            s.monthly_start_equity = equity

        # check max drawdown halt
        if c.max_drawdown_halt_pct is not None and s.peak_equity > 0:
            dd = 1.0 - equity / s.peak_equity
            if dd >= c.max_drawdown_halt_pct:
                s.halted = True

    # ---- Trade close hook ----

    def on_trade_closed(
        self,
        symbol: str,
        pnl: float,
        now_ms: int,
    ) -> None:
        """Update streak / cooldown counters after a trade closes."""
        s, c = self.state, self.config
        was_loss = pnl < 0

        if was_loss:
            s.consecutive_losses += 1
            s.per_symbol_losses[symbol] = s.per_symbol_losses.get(symbol, 0) + 1
        else:
            s.consecutive_losses = 0
            s.per_symbol_losses[symbol] = 0

        if c.consecutive_loss_pause and s.consecutive_losses >= c.consecutive_loss_pause[0]:
            s.pause_until_ms = now_ms + c.consecutive_loss_pause[1]
            s.consecutive_losses = 0

        if c.per_symbol_cooldown and s.per_symbol_losses.get(symbol, 0) >= c.per_symbol_cooldown[0]:
            s.per_symbol_pause_until[symbol] = now_ms + c.per_symbol_cooldown[1]
            s.per_symbol_losses[symbol] = 0

        s.open_positions = max(0, s.open_positions - 1)

    # ---- Position open / close hooks ----

    def on_position_opened(self) -> None:
        self.state.open_positions += 1

    # ---- Pre-open check ----

    def can_open_new(
        self,
        now_ms: int,
        equity: float,
        used_margin: float = 0.0,
        symbol: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check if a new entry is permitted right now.

        Returns ``(True, None)`` if all checks pass, otherwise
        ``(False, reason_string)``.
        """
        s, c = self.state, self.config

        if s.halted:
            return False, "halted_max_drawdown"

        if symbol and symbol in c.blacklist:
            return False, f"blacklisted:{symbol}"

        if c.max_positions is not None and s.open_positions >= c.max_positions:
            return False, f"max_positions:{c.max_positions}"

        if s.pause_until_ms is not None and now_ms < s.pause_until_ms:
            return False, f"paused_until_{s.pause_until_ms}"

        if symbol and symbol in s.per_symbol_pause_until:
            until = s.per_symbol_pause_until[symbol]
            if now_ms < until:
                return False, f"symbol_paused_until_{until}"

        if c.daily_max_loss_pct is not None and s.daily_start_equity > 0:
            day_loss = 1.0 - equity / s.daily_start_equity
            if day_loss >= c.daily_max_loss_pct:
                return False, f"daily_loss_breach:{day_loss:.4f}"

        if c.monthly_max_loss_pct is not None and s.monthly_start_equity > 0:
            month_loss = 1.0 - equity / s.monthly_start_equity
            if month_loss >= c.monthly_max_loss_pct:
                return False, f"monthly_loss_breach:{month_loss:.4f}"

        if c.min_available_margin_pct is not None and equity > 0:
            available = equity - used_margin
            if available / equity < c.min_available_margin_pct:
                return False, f"insufficient_available_margin:{available/equity:.3f}"

        if c.funding_buffer_min > 0 and is_funding_window(
            now_ms, c.funding_settle_hours_utc, c.funding_buffer_min
        ):
            return False, "funding_window"

        return True, None
