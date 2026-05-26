"""SnapshotBacktestRunner — symbol-name compat for `standard_bot` callers.

`cyqnt_trd.standard_bot.simulation.runner.SnapshotBacktestRunner`
operates over a list of `DataSnapshot` objects, runs a pluggable signal
pipeline, and emits `BacktestResult`. The new repo's runtime is built
around pandas DataFrames + callable strategy factories, so the input
shape changed (snapshots → DataFrame slice) but the inner cash /
position / fee / slippage bookkeeping is identical.

Numerical parity vs standard_bot at 1e-9 is already validated by
`tests/integration/test_parity_step9_long_only_backtest.py`. This class
exists so users coming from the standard_bot API have a familiar entry
point that wraps the canonical `run_long_only_backtest` in a class
shape — without forcing them to rewire the snapshot pipeline.

Per `docs/migration/overlap-policy.md` this is class C (intentional
divergence): same math, different I/O shape; numerical equivalence
verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from ai_pro_trading_library.library.core.protocols import (
    Bar,
    MarketBundle,
    StrategySpec,
)
from ai_pro_trading_library.library.data.conversions import frame_to_bars
from ai_pro_trading_library.library.strategy.registry import (
    StrategyRegistry,
    default_registry,
)
from ai_pro_trading_library.runtime.backtest.long_only_engine import (
    LongOnlyResult,
    run_long_only_backtest,
)


@dataclass
class SnapshotBacktestRunner:
    """Long-only single-position runner — DataFrame-shape sibling of
    `standard_bot.simulation.runner.SnapshotBacktestRunner`.

    Two entry points:

    - `run(bars, entry_signal, exit_signal)` — pre-computed signals, the
      thin wrapper Step 17 added.
    - `run_snapshot(spec, bundle, registry=None)` — the real snapshot
      pipeline: resolves the strategy factory from `default_registry`,
      pulls the primary symbol/timeframe DataFrame out of `bundle`,
      computes the entry signal, converts DataFrame rows to `Bar`s, and
      runs the long-only engine.

    Parameters:
    - `commission_bps` / `slippage_bps`: bps execution costs (used when
      no explicit fee_model / slippage_model is provided).
    - `fee_model` / `slippage_model`: optional `runtime.execution.models`
      Protocol implementations that override the bps params.
    - `initial_capital`: starting cash. Default 10000 to match fixtures.
    - `force_exit_at_end`: close any open position on the final bar.
    """

    initial_capital: float = 10_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    force_exit_at_end: bool = True
    fee_model: object | None = None
    slippage_model: object | None = None

    def run(
        self,
        bars: Sequence[Bar],
        entry_signal: Sequence[bool],
        exit_signal: Sequence[bool] | None = None,
    ) -> LongOnlyResult:
        return run_long_only_backtest(
            bars=bars,
            entry_signal=entry_signal,
            exit_signal=exit_signal,
            initial_capital=self.initial_capital,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            fee_model=self.fee_model,
            slippage_model=self.slippage_model,
            force_exit_at_end=self.force_exit_at_end,
        )

    def run_snapshot(
        self,
        spec: StrategySpec,
        bundle: MarketBundle,
        *,
        registry: StrategyRegistry | None = None,
        exit_when_false: bool = True,
    ) -> LongOnlyResult:
        """Run a `StrategySpec` against a `MarketBundle` and return result.

        Steps:
        1. Validate `spec.symbols` and pull the primary `(symbol, timeframe)`
           DataFrame from `bundle.frames`.
        2. Resolve the strategy factory from `registry` (default
           `default_registry`); fall back to `engine.build_signal_from_spec`
           if the strategy_id isn't registered.
        3. Run the factory to get the entry signal pd.Series.
        4. Convert DataFrame rows to `Bar` dataclasses.
        5. If `exit_when_false`, exit_signal = NOT entry_signal; otherwise
           rely on `force_exit_at_end` to close the position.
        6. Delegate to `run_long_only_backtest` with this runner's costs.
        """
        spec.require_symbols()
        symbol = spec.symbols[0]
        tf = spec.timeframe
        if symbol not in bundle.frames:
            raise KeyError(f"bundle has no symbol {symbol!r}; available: {list(bundle.frames)}")
        if tf not in bundle.frames[symbol]:
            raise KeyError(
                f"bundle has no timeframe {tf!r} for {symbol}; "
                f"available: {list(bundle.frames[symbol])}"
            )
        frame: pd.DataFrame = bundle.frames[symbol][tf]
        if frame.empty:
            return LongOnlyResult(
                total_return=0.0,
                final_equity=float(self.initial_capital),
                initial_capital=float(self.initial_capital),
                trade_count=0,
                snapshot_count=0,
            )

        reg = registry if registry is not None else default_registry
        try:
            signal_series = reg.build_signal(spec, frame)
        except KeyError:
            from ai_pro_trading_library.runtime.backtest.engine import (
                build_signal_from_spec,
            )
            signal_series = build_signal_from_spec(spec, frame)

        entry = [bool(v) for v in signal_series.fillna(False).tolist()]
        if exit_when_false:
            exit_signal: list[bool] | None = [not v for v in entry]
        else:
            exit_signal = None

        bars = frame_to_bars(frame, instrument_id=symbol, timeframe=tf)
        return self.run(bars, entry, exit_signal)


__all__ = ["SnapshotBacktestRunner"]
