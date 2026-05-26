"""Portfolio risk guard state machine.

Source attribution: rewritten from `cyqnt_trd.blocks.risk`; atomic
`limits` functions remain in the migration matrix until parity is verified.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskConfig:
    max_positions: int | None = None
    max_drawdown_halt_pct: float | None = None
    daily_max_loss_pct: float | None = None
    leverage_cap: float = 1.0
    blacklist: tuple[str, ...] = ()


@dataclass
class RiskState:
    open_positions: int = 0
    starting_equity: float = 0.0
    peak_equity: float = 0.0
    daily_start_equity: float = 0.0
    halted: bool = False
    blocked_reasons: list[str] = field(default_factory=list)


class RiskGuard:
    def __init__(self, config: RiskConfig, state: RiskState | None = None) -> None:
        self.config = config
        self.state = state or RiskState()

    def on_equity_update(self, equity: float) -> None:
        if equity < 0:
            raise ValueError("equity must be >= 0")
        state = self.state
        if state.starting_equity == 0:
            state.starting_equity = equity
            state.daily_start_equity = equity
            state.peak_equity = equity
        state.peak_equity = max(state.peak_equity, equity)
        if self.config.max_drawdown_halt_pct is not None and state.peak_equity:
            drawdown = (state.peak_equity - equity) / state.peak_equity
            if drawdown >= self.config.max_drawdown_halt_pct:
                state.halted = True

    def can_open_new(
        self,
        symbol: str,
        equity: float,
        requested_leverage: float = 1.0,
    ) -> tuple[bool, str]:
        reasons: list[str] = []
        state = self.state
        config = self.config
        if state.halted:
            reasons.append("risk halted")
        if symbol in config.blacklist:
            reasons.append("symbol blacklisted")
        if config.max_positions is not None and state.open_positions >= config.max_positions:
            reasons.append("max positions reached")
        if requested_leverage > config.leverage_cap:
            reasons.append("leverage cap exceeded")
        if config.daily_max_loss_pct is not None and state.daily_start_equity:
            loss = max(state.daily_start_equity - equity, 0.0) / state.daily_start_equity
            if loss >= config.daily_max_loss_pct:
                reasons.append("daily max loss reached")
        return not reasons, "; ".join(reasons)

    def record_block(self, reasons: list[str]) -> None:
        """Explicit recording path; previously folded into can_open_new."""
        self.state.blocked_reasons = list(reasons)

    def on_position_opened(self) -> None:
        self.state.open_positions += 1

    def on_position_closed(self) -> None:
        self.state.open_positions = max(0, self.state.open_positions - 1)

