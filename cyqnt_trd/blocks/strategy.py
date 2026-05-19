"""Strategy registration helper — bridges blocks to ``standard_bot``.

This module lets users wrap a function-based strategy in a single call
and have it registered as a SignalPlugin so that
``mvp_backtest --engine python --strategy <name> --strategy-module <module>``
finds it.

Two-step usage in a user-authored strategy script::

    from cyqnt_trd.blocks import indicators as ind, conditions as cond, strategy

    def make_signals(df):
        ma20 = ind.sma(df["close"], 20)
        ma60 = ind.sma(df["close"], 60)
        return cond.ma_cross_above(ma20, ma60), cond.ma_cross_below(ma20, ma60)

    strategy.register("my_ma_cross", make_signals)

When the script is loaded via ``--strategy-module my_strategy_script``,
the call to :func:`register` fires at import time and inserts the
strategy into the global ``SignalPluginRegistry`` used by the Python
engine.

The signal function must accept a single argument:

* a ``pandas.DataFrame`` with the standard OHLCV columns indexed by
  bar order. Optionally extra columns produced by the framework
  (``timestamp`` etc.) are also present.

The signal function must return either:

* a tuple ``(long_signal, short_signal)`` — both are boolean
  ``pandas.Series`` aligned to the input DataFrame index. Use ``None``
  for short to declare a long-only strategy::

      return long_signal, None

* a single boolean ``pandas.Series`` — interpreted as a long-only
  strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd

from .data import bars_to_df

__all__ = [
    "register",
    "build_plugin",
    "BlockStrategyPlugin",
]


SignalFnOutput = Union[pd.Series, Tuple[pd.Series, Optional[pd.Series]]]
SignalFn = Callable[[pd.DataFrame], SignalFnOutput]


@dataclass
class _RegisteredStrategy:
    plugin_id: str
    version: str
    signal_fn: SignalFn


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def register(
    strategy_id: str,
    signal_fn: SignalFn,
    *,
    version: str = "block/v1",
) -> None:
    """Register a block-based strategy with the standard_bot registry.

    Side-effect contract: this function imports
    :mod:`cyqnt_trd.standard_bot.signal` and calls
    ``SignalPluginRegistry.register(plugin, factory)`` on the global
    registry used by the ``--engine python`` backtest path.

    Numba support is intentionally not provided here — the blocks are
    pure-Python. After validating a strategy, users can promote it to a
    Numba kernel via ``NumbaBacktestRunner.register_kernel()`` directly.
    """
    plugin = build_plugin(strategy_id, signal_fn, version=version)
    factory = _make_config_factory(plugin.plugin_id)
    _register_with_global(plugin, factory)


def build_plugin(
    strategy_id: str,
    signal_fn: SignalFn,
    *,
    version: str = "block/v1",
) -> "BlockStrategyPlugin":
    """Construct (without registering) a SignalPlugin from a signal function."""
    if not strategy_id or not isinstance(strategy_id, str):
        raise ValueError("strategy_id must be a non-empty string")
    if not callable(signal_fn):
        raise TypeError("signal_fn must be callable")
    return BlockStrategyPlugin(
        plugin_id=strategy_id,
        plugin_version=version,
        signal_fn=signal_fn,
    )


def _register_with_global(plugin: "BlockStrategyPlugin", factory: Callable[[dict], object]) -> None:
    # Lazy import to avoid heavy deps at module-import time.
    from ..standard_bot.signal.registry import SignalPluginRegistry  # type: ignore  # noqa: F401

    # The mvp_backtest entrypoint uses make_registry() which creates a fresh
    # registry per run. We attach the registration to a per-process global
    # registry list and rely on entrypoints.common to discover it.
    _PENDING_REGISTRATIONS.append((plugin, factory))
    _KNOWN_BLOCK_STRATEGY_IDS.add(plugin.plugin_id)


# ---------------------------------------------------------------------------
# SignalPlugin implementation
# ---------------------------------------------------------------------------


@dataclass
class BlockStrategyPlugin:
    """A SignalPlugin adapter that wraps a user-supplied signal function.

    Implements the minimum SignalPlugin protocol surface so the
    Python-engine backtest path (``SnapshotBacktestRunner``) accepts it.
    """

    plugin_id: str
    plugin_version: str
    signal_fn: SignalFn

    # ---- SignalPlugin protocol ----

    def required_inputs(self) -> Dict[str, bool]:
        return {"market": True, "social": False, "onchain": False}

    def initialize_state(self):
        from ..standard_bot.signal.interfaces import SignalState  # type: ignore

        return SignalState(plugin_id=self.plugin_id, values={"cursor": None})

    def run(self, snapshot, config, context=None):  # type: ignore[no-untyped-def]
        df, instrument_id, timeframe = self._extract_df(snapshot, config)
        signals = []
        if not df.empty:
            long_s, short_s = self._call_signal_fn(df)
            signals = self._envelope_from_signals(
                df=df,
                long_s=long_s,
                short_s=short_s,
                snapshot=snapshot,
                instrument_id=instrument_id,
                timeframe=timeframe,
            )
        return self._make_batch(snapshot, signals)

    def step(self, snapshot, state, config, context=None):  # type: ignore[no-untyped-def]
        from ..standard_bot.signal.interfaces import StepSignalResult  # type: ignore

        df, instrument_id, timeframe = self._extract_df(snapshot, config)
        if df.empty:
            return StepSignalResult(state=state, signals=[])

        # IMPORTANT: pass the FULL DataFrame to the signal function so that
        # rolling/EMA/RSI indicators have enough history. We then post-filter
        # the emitted envelopes to only those bars at-or-after the cursor.
        cursor = state.values.get("cursor") if state else None
        long_s, short_s = self._call_signal_fn(df)

        if cursor is not None:
            mask = df["close_time"] > cursor
            emit_df = df[mask]
            long_s = long_s[mask] if hasattr(long_s, "__getitem__") else long_s
            short_s = short_s[mask] if (short_s is not None and hasattr(short_s, "__getitem__")) else short_s
        else:
            emit_df = df

        envelopes = self._envelope_from_signals(
            df=emit_df,
            long_s=long_s,
            short_s=short_s,
            snapshot=snapshot,
            instrument_id=instrument_id,
            timeframe=timeframe,
        )

        new_cursor = int(df["close_time"].iloc[-1]) if "close_time" in df.columns else cursor
        from ..standard_bot.signal.interfaces import SignalState  # type: ignore

        return StepSignalResult(
            state=SignalState(plugin_id=self.plugin_id, values={"cursor": new_cursor}),
            signals=envelopes,
        )

    # ---- helpers ----

    def _extract_df(self, snapshot, config) -> Tuple[pd.DataFrame, str, str]:
        # `config` is what the factory built — for block strategies it has
        # `instrument_id` and `timeframe` (set by entrypoints.common).
        instrument_id = getattr(config, "instrument_id", None) or config.get("instrument_id")
        timeframe = getattr(config, "timeframe", None) or config.get("timeframe")
        if not instrument_id or not timeframe:
            raise ValueError("config missing instrument_id / timeframe")

        from ..standard_bot.core import MarketBundle  # type: ignore

        market = snapshot.require_market()
        bars = market.bars.get(MarketBundle.key(instrument_id, timeframe), [])
        df = bars_to_df(bars)
        # Provide both `timestamp` and `close_time` aliases so user code can
        # use whichever feels natural.
        if not df.empty and "close_time" in df.columns and "timestamp" not in df.columns:
            df = df.copy()
            df["timestamp"] = df["close_time"]
        return df, str(instrument_id).upper(), str(timeframe)

    def _call_signal_fn(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, Optional[pd.Series]]:
        out = self.signal_fn(df)
        if isinstance(out, tuple):
            if len(out) != 2:
                raise ValueError(
                    f"signal_fn must return (long_signal, short_signal); "
                    f"got tuple of length {len(out)}"
                )
            long_s, short_s = out
        else:
            long_s, short_s = out, None
        if long_s is None:
            long_s = pd.Series(False, index=df.index)
        if not isinstance(long_s, pd.Series):
            raise TypeError(f"long_signal must be Series, got {type(long_s).__name__}")
        if short_s is not None and not isinstance(short_s, pd.Series):
            raise TypeError(f"short_signal must be Series or None, got {type(short_s).__name__}")
        long_s = long_s.reindex(df.index, fill_value=False).fillna(False).astype(bool)
        if short_s is not None:
            short_s = short_s.reindex(df.index, fill_value=False).fillna(False).astype(bool)
        return long_s, short_s

    def _envelope_from_signals(
        self,
        df: pd.DataFrame,
        long_s: pd.Series,
        short_s: Optional[pd.Series],
        snapshot,
        instrument_id: str,
        timeframe: str,
    ) -> list:
        """Convert per-bar boolean signals into SignalEnvelope objects."""
        import time as _time
        import uuid as _uuid

        from ..standard_bot.core import (  # type: ignore
            SignalEnvelope, SignalKind, SignalProvenance, TradeSide,
        )

        ns = _uuid.UUID("6d011acb-2b89-5dc5-bd38-fa9f903e6495")
        envelopes = []
        snapshot_id = snapshot.meta.snapshot_id
        for i, ts in enumerate(df["close_time"].tolist() if "close_time" in df.columns else df.index):
            bar_ts = int(ts)
            triggered_long = bool(long_s.iloc[i]) if i < len(long_s) else False
            triggered_short = bool(short_s.iloc[i]) if (short_s is not None and i < len(short_s)) else False
            if not triggered_long and not triggered_short:
                continue
            if triggered_long and triggered_short:
                # both — give priority to the *new* signal: pick stronger by index,
                # but to keep determinism we tie-break to long.
                triggered_short = False
            side = TradeSide.BUY if triggered_long else TradeSide.SELL
            target = 1 if triggered_long else -1
            payload = {
                "bar_timestamp": bar_ts,
                "target_position": target,
                "risk_hints": {"target_position": target},
            }
            sig_id = str(
                _uuid.uuid5(
                    ns, f"{snapshot_id}|{instrument_id}|{timeframe}|{bar_ts}|{side.value}"
                )
            )
            envelopes.append(
                SignalEnvelope(
                    version=self.plugin_version,
                    signal_id=sig_id,
                    kind=SignalKind.TRADE,
                    instrument_id=instrument_id,
                    side=side,
                    strength=1.0,
                    time_horizon="swing",
                    valid_until=snapshot.meta.decision_as_of,
                    payload=payload,
                    provenance=SignalProvenance(
                        plugin_id=self.plugin_id,
                        plugin_version=self.plugin_version,
                        config_hash=f"{instrument_id}|{timeframe}",
                        input_fingerprint=snapshot_id,
                    ),
                )
            )
        return envelopes

    def _make_batch(self, snapshot, signals):
        import time as _time
        import uuid as _uuid

        from ..standard_bot.core import SignalBatch  # type: ignore

        ns = _uuid.UUID("6d011acb-2b89-5dc5-bd38-fa9f903e6495")
        batch_id = str(
            _uuid.uuid5(ns, f"{self.plugin_id}|{snapshot.meta.snapshot_id}|{len(signals)}")
        )
        return SignalBatch(
            signals=list(signals),
            batch_id=batch_id,
            created_at=int(_time.time() * 1000),
        )


# ---------------------------------------------------------------------------
# Hookable global registration list
# ---------------------------------------------------------------------------

# Strategy modules are imported by ``mvp_backtest`` *after* it builds a
# fresh SignalPluginRegistry via ``entrypoints/common.make_registry()``.
# Since we don't own that registry, we expose a *pending registrations*
# list that ``standard_bot`` can drain from. ``entrypoints.common`` will
# be patched to call :func:`flush_pending_into` after seeding builtins.

_PENDING_REGISTRATIONS: list = []

#: A permanent set of strategy IDs that have ever been registered via
#: :func:`register` in the current process. Used by
#: ``entrypoints.common.build_strategy_pipeline`` to recognise block
#: strategies in its fallback branch even after :func:`flush_pending_into`
#: has drained the pending list.
_KNOWN_BLOCK_STRATEGY_IDS: set = set()


def _make_config_factory(plugin_id: str) -> Callable[[dict], object]:
    """Return a config factory that wraps an arbitrary raw dict.

    Block strategies don't have a fixed config schema — entrypoints/common
    passes in ``{"instrument_id": ..., "timeframe": ..., **extra_params}``
    so we just return a SimpleNamespace.
    """
    from types import SimpleNamespace

    def factory(raw: dict) -> object:
        normalized = {**raw}
        if "instrument_id" in normalized and isinstance(normalized["instrument_id"], str):
            normalized["instrument_id"] = normalized["instrument_id"].upper()
        return SimpleNamespace(**normalized)

    factory.__name__ = f"_block_config_factory_{plugin_id}"
    return factory


def flush_pending_into(registry) -> int:  # type: ignore[no-untyped-def]
    """Drain any block strategies that were registered via :func:`register`.

    Should be called by the entrypoint after ``register_builtin_plugins``.
    Returns the number of strategies installed.
    """
    count = 0
    while _PENDING_REGISTRATIONS:
        plugin, factory = _PENDING_REGISTRATIONS.pop(0)
        try:
            registry.register(plugin, factory)
            count += 1
        except ValueError:
            # Already registered (re-import) — silently skip.
            pass
    return count


def is_known_block_strategy(strategy_id: str) -> bool:
    """Return True if a block strategy with this ID has been registered."""
    return strategy_id in _KNOWN_BLOCK_STRATEGY_IDS


def registered_block_strategies() -> list:
    """Return the list of currently pending block strategies (for tests)."""
    return [(plugin, factory) for plugin, factory in _PENDING_REGISTRATIONS]
