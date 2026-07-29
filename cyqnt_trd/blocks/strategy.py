"""Strategy registration helper — bridges blocks to ``standard_bot``.

This module lets users wrap a function-based strategy in a single call
and have it registered as a SignalPlugin so that
``mvp_backtest --engine python --strategy <name>``
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

Multi-timeframe (HTF) support
-----------------------------

A strategy can declare that it needs higher-timeframe context by
passing ``htf_specs`` to :func:`register`. Each entry is a tuple
``(htf_tf, sma_period)`` describing one HTF SMA the strategy needs.
At runtime the plugin attaches a column named
``_htf_<tf>_sma_<period>`` to the DataFrame passed into the signal
function, with the most-recent CONFIRMED HTF SMA value forward-filled
onto every base-TF bar. Lookahead-safe by construction.

Example::

    strategy.register(
        "my_strategy",
        make_signals,
        htf_specs=[("4h", 200)],   # need 4h SMA(200) projected onto base TF
    )

    def make_signals(df):
        # df has the standard OHLCV columns plus _htf_4h_sma_200
        ma20 = ind.sma(df["close"], 20)
        long = (df["close"] > df["_htf_4h_sma_200"]) & cond.ma_cross_above(ma20, ind.sma(df["close"], 60))
        return long, None

The signal function must accept a single argument:

* a ``pandas.DataFrame`` with the standard OHLCV columns indexed by
  bar order. Optionally extra columns produced by the framework
  (``timestamp``, HTF columns) are also present.

The signal function must return either:

* a tuple ``(long_signal, short_signal)`` — both are boolean
  ``pandas.Series`` aligned to the input DataFrame index. Use ``None``
  for short to declare a long-only strategy::

      return long_signal, None

* a single boolean ``pandas.Series`` — interpreted as a long-only
  strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from .data import bars_to_df

__all__ = [
    "register",
    "build_plugin",
    "BlockStrategyPlugin",
    "register_selection",
    "build_selection_plugin",
    "SelectionStrategyPlugin",
    "is_known_selection_strategy",
    "get_selection_plugin",
]


SignalFnOutput = Union[pd.Series, Tuple[pd.Series, Optional[pd.Series]]]
SignalFn = Callable[[pd.DataFrame], SignalFnOutput]
HtfSpec = Tuple[str, int]  # (htf_timeframe, sma_period)
# A selection strategy ranks a universe of symbols at one point in time.
# Contract: selection_fn(universe_df, ticker_rank_df, *, ticker_rank_prev,
#           klines, as_of_ms, market_type) -> list[candidate dict]
SelectionFn = Callable[..., List[Dict]]


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
    htf_specs: Optional[List[HtfSpec]] = None,
    exit_cfg: Optional[Dict] = None,
    size: float = 1.0,
    needs: Optional[Dict[str, bool]] = None,
) -> None:
    """Register a block-based strategy with the standard_bot registry.

    :param htf_specs: optional list of ``(htf_tf, sma_period)`` tuples for
        multi-timeframe context (see :func:`_attach_htf_columns`).
    :param exit_cfg: optional dict describing how the runner should exit
        positions opened by this strategy. Supported types:

        - ``{"type": "time_only", "max_bars": N}`` — exit after N bars
        - ``{"type": "pct_stop_tp", "stop_pct": 0.02, "tp_pct": 0.04, "max_bars": 80}``
        - ``{"type": "atr_stop_tp", "atr_period": 14, "stop_mult": 2.0,
          "tp_mult": 4.0, "max_bars": 80}`` — plugin precomputes absolute
          stop/TP prices from ATR at entry bar
        - ``{"type": "ma_cross_exit", "period": 35, "ma_type": "ema",
          "max_bars": 160}`` — exit when close crosses opposite to entry
          direction. Strategy must also emit opposite signal (SELL for
          long-side cross-down) OR provide ``max_bars`` to enforce time cap.

        When ``exit_cfg=None`` (default), exits are driven solely by
        opposite-side signals from the strategy (existing behavior).
    :param size: fraction of equity to deploy per trade, in ``(0, 1]``.
        Default ``1.0`` (full equity, matching pre-Phase-2 behavior).
    :param needs: optional map of extra input requirements, e.g.
        ``{"derivatives": True}`` for a strategy whose ``make_signals`` reads
        ``funding_rate`` / ``open_interest`` columns. Merged into
        :meth:`BlockStrategyPlugin.required_inputs`. Additive; the default
        ``{"market": True, "social": False, "onchain": False,
        "derivatives": False}`` is unchanged when ``needs`` is None.
    """
    plugin = build_plugin(
        strategy_id, signal_fn,
        version=version,
        htf_specs=htf_specs,
        exit_cfg=exit_cfg,
        size=size,
        needs=needs,
    )
    factory = _make_config_factory(plugin.plugin_id)
    _register_with_global(plugin, factory)


def build_plugin(
    strategy_id: str,
    signal_fn: SignalFn,
    *,
    version: str = "block/v1",
    htf_specs: Optional[List[HtfSpec]] = None,
    exit_cfg: Optional[Dict] = None,
    size: float = 1.0,
    needs: Optional[Dict[str, bool]] = None,
) -> "BlockStrategyPlugin":
    """Construct (without registering) a SignalPlugin from a signal function."""
    if not strategy_id or not isinstance(strategy_id, str):
        raise ValueError("strategy_id must be a non-empty string")
    if not callable(signal_fn):
        raise TypeError("signal_fn must be callable")
    if not (0.0 < float(size) <= 1.0):
        raise ValueError(f"size must be in (0, 1], got {size}")
    return BlockStrategyPlugin(
        plugin_id=strategy_id,
        plugin_version=version,
        signal_fn=signal_fn,
        htf_specs=list(htf_specs) if htf_specs else [],
        exit_cfg=dict(exit_cfg) if exit_cfg else None,
        size=float(size),
        needs=dict(needs) if needs else None,
    )


def register_selection(
    strategy_id: str,
    selection_fn: SelectionFn,
    *,
    version: str = "selection/v1",
    market_type: str = "futures",
) -> None:
    """Register a cross-sectional SELECTION strategy.

    This is the sibling of :func:`register` for strategies that rank a
    *universe* of symbols rather than emit per-bar long/short on one
    instrument. It routes through the SAME registration mechanism
    (``_PENDING_REGISTRATIONS`` → ``flush_pending_into`` →
    ``SignalPluginRegistry.register``) and the SAME
    ``run_pipeline_step`` execution path — the only difference is the plugin
    reads ``snapshot.universe`` and emits a single ``kind=SELECTION`` envelope.

    ``selection_fn`` contract::

        selection_fn(universe_df, ticker_rank_df, *, ticker_rank_prev,
                     klines, as_of_ms, market_type) -> list[candidate dict]

    Extra keyword args it does not use should be absorbed with ``**_``.
    """
    plugin = build_selection_plugin(
        strategy_id, selection_fn, version=version, market_type=market_type
    )
    factory = _make_config_factory(plugin.plugin_id)
    _register_selection_with_global(plugin, factory)


def build_selection_plugin(
    strategy_id: str,
    selection_fn: SelectionFn,
    *,
    version: str = "selection/v1",
    market_type: str = "futures",
) -> "SelectionStrategyPlugin":
    """Construct (without registering) a SELECTION SignalPlugin."""
    if not strategy_id or not isinstance(strategy_id, str):
        raise ValueError("strategy_id must be a non-empty string")
    if not callable(selection_fn):
        raise TypeError("selection_fn must be callable")
    return SelectionStrategyPlugin(
        plugin_id=strategy_id,
        plugin_version=version,
        selection_fn=selection_fn,
        market_type=market_type,
    )


def _register_with_global(plugin: "BlockStrategyPlugin", factory: Callable[[dict], object]) -> None:
    # Lazy import to avoid heavy deps at module-import time.
    from ..standard_bot.signal.registry import SignalPluginRegistry  # type: ignore  # noqa: F401

    # The mvp_backtest entrypoint uses make_registry() which creates a fresh
    # registry per run. We attach the registration to a per-process global
    # registry list and rely on entrypoints.common to discover it.
    _PENDING_REGISTRATIONS.append((plugin, factory))
    _KNOWN_BLOCK_STRATEGY_IDS.add(plugin.plugin_id)
    _KNOWN_BLOCK_PLUGINS[plugin.plugin_id] = plugin


def _register_selection_with_global(plugin: "SelectionStrategyPlugin", factory: Callable[[dict], object]) -> None:
    # Lazy import to avoid heavy deps at module-import time.
    from ..standard_bot.signal.registry import SignalPluginRegistry  # type: ignore  # noqa: F401

    _PENDING_REGISTRATIONS.append((plugin, factory))
    _KNOWN_SELECTION_STRATEGY_IDS.add(plugin.plugin_id)
    _KNOWN_SELECTION_PLUGINS[plugin.plugin_id] = plugin


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
    htf_specs: List[HtfSpec] = field(default_factory=list)
    exit_cfg: Optional[Dict] = None
    size: float = 1.0
    needs: Optional[Dict[str, bool]] = None

    # ---- SignalPlugin protocol ----

    def required_inputs(self) -> Dict[str, bool]:
        base = {"market": True, "social": False, "onchain": False, "derivatives": False}
        if self.needs:
            base.update({k: bool(v) for k, v in self.needs.items()})
        return base

    def needed_timeframes(self) -> List[str]:
        """Return distinct HTF timeframes this strategy needs (besides primary).

        Used by entrypoints (e.g. ``mvp_backtest``) to expand the MarketQuery
        so the corresponding HTF bars are available in every snapshot.
        """
        return sorted({tf for (tf, _) in self.htf_specs})

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

        # Pre-compute ATR series on the FULL df (before cursor filter) so that
        # the rolling RMA inside ATR has enough history. Without this, the
        # filtered emit_df (typically just 1 row at the cursor head) would
        # yield NaN ATR, and the entry-bar absolute stop/TP prices baked into
        # exit_spec would be NaN — silently disabling the ATR stop. See bug
        # report: Issue #1.
        atr_series_full = None
        if self.exit_cfg and self.exit_cfg.get("type") in ("atr_stop_tp", "atr_trailing_stop"):
            try:
                from . import indicators as _ind  # type: ignore
                atr_period = int(self.exit_cfg.get("atr_period", 14))
                atr_series_full = _ind.atr(df, period=atr_period)
            except Exception:
                atr_series_full = None

        if cursor is not None:
            mask = df["close_time"] > cursor
            emit_df = df[mask]
            long_s = long_s[mask] if hasattr(long_s, "__getitem__") else long_s
            short_s = short_s[mask] if (short_s is not None and hasattr(short_s, "__getitem__")) else short_s
            atr_series_emit = atr_series_full[mask] if atr_series_full is not None else None
        else:
            emit_df = df
            atr_series_emit = atr_series_full

        envelopes = self._envelope_from_signals(
            df=emit_df,
            long_s=long_s,
            short_s=short_s,
            snapshot=snapshot,
            instrument_id=instrument_id,
            timeframe=timeframe,
            atr_series=atr_series_emit,
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

        # ---- HTF column attachment (additive, safe-by-default) ----
        # If this plugin declares htf_specs, look up HTF bars from the
        # MarketBundle, compute SMA on each HTF, and ffill-align onto the
        # base-TF DataFrame via close_time. The aligned column is named
        # `_htf_<tf>_sma_<period>`. Lookahead-safe: a base bar at open_time T
        # only sees the most-recent HTF bar whose close_time <= T.
        if self.htf_specs and not df.empty:
            df = self._attach_htf_columns(df, market, instrument_id)

        return df, str(instrument_id).upper(), str(timeframe)

    def _attach_htf_columns(self, df: pd.DataFrame, market, instrument_id: str) -> pd.DataFrame:
        from ..standard_bot.core import MarketBundle  # type: ignore
        import numpy as np

        df = df.copy()
        # Pre-extract base bar open_time as int64 array for searchsorted
        if "open_time" not in df.columns:
            return df  # nothing to align against
        base_open_times = df["open_time"].astype("int64").values

        for htf, period in self.htf_specs:
            col = f"_htf_{htf}_sma_{int(period)}"
            if col in df.columns:
                continue  # already attached upstream
            htf_bars = market.bars.get(MarketBundle.key(instrument_id, htf), [])
            if not htf_bars:
                df[col] = np.nan
                continue
            htf_df = bars_to_df(htf_bars)
            if htf_df.empty or "close_time" not in htf_df.columns:
                df[col] = np.nan
                continue
            # Compute SMA on HTF closes
            sma_vals = htf_df["close"].rolling(window=int(period), min_periods=int(period)).mean().values
            htf_close_times = htf_df["close_time"].astype("int64").values
            # For each base bar's open_time T, find the latest HTF bar with
            # close_time <= T. searchsorted with side="right" gives insertion
            # point AFTER existing equal entries, so we subtract 1.
            idx = np.searchsorted(htf_close_times, base_open_times, side="right") - 1
            idx = np.clip(idx, 0, len(htf_close_times) - 1)
            aligned = sma_vals[idx]
            # Mask the early region where there's no closed HTF bar at all
            no_htf_yet = base_open_times < htf_close_times[0]
            aligned = aligned.astype(float, copy=True)
            aligned[no_htf_yet] = np.nan
            df[col] = aligned

        return df

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
        atr_series: Optional[pd.Series] = None,
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

        # Pre-compute ATR series ONCE if exit_cfg is atr_stop_tp — used to
        # determine entry-bar absolute stop/TP prices. This avoids the runner
        # needing to know about ATR.
        # If a caller (e.g. step()) already supplied an ATR series computed on
        # the full pre-cursor-filter df, use it as-is. This prevents recomputing
        # ATR on a 1-row emit_df which would yield NaN. See Issue #1.
        if atr_series is None and self.exit_cfg and \
                self.exit_cfg.get("type") in ("atr_stop_tp", "atr_trailing_stop"):
            try:
                from . import indicators as _ind  # type: ignore
                atr_period = int(self.exit_cfg.get("atr_period", 14))
                atr_series = _ind.atr(df, period=atr_period)
            except Exception:
                atr_series = None

        close_arr = df["close"].astype(float).values if "close" in df.columns else None

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

            # Embed sizing fraction so runner can scale qty
            if self.size != 1.0:
                payload["size"] = float(self.size)

            # Embed exit_spec on entry signals (BUY/SELL openings).
            # For atr_stop_tp, compute absolute stop/TP prices using ATR at
            # the entry bar; runner just compares to bar high/low.
            if triggered_long:  # long entry
                exit_spec = self._compute_exit_spec(
                    side="long",
                    entry_close=float(close_arr[i]) if close_arr is not None else None,
                    atr_value=float(atr_series.iloc[i]) if (atr_series is not None and i < len(atr_series)) else None,
                )
                if exit_spec is not None:
                    payload["exit_spec"] = exit_spec
            elif triggered_short:  # short entry
                exit_spec = self._compute_exit_spec(
                    side="short",
                    entry_close=float(close_arr[i]) if close_arr is not None else None,
                    atr_value=float(atr_series.iloc[i]) if (atr_series is not None and i < len(atr_series)) else None,
                )
                if exit_spec is not None:
                    payload["exit_spec"] = exit_spec

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

    def _compute_exit_spec(self, *, side: str, entry_close, atr_value) -> Optional[Dict]:
        """Build an exit_spec dict from this plugin's exit_cfg, baking in
        absolute prices for atr-based stops. Runner reads this and applies."""
        if not self.exit_cfg:
            # register(exit_cfg=None) is documented as "exits are driven solely
            # by opposite-side signals". Make that explicit instead of emitting
            # no exit_spec at all.
            #
            # Returning None left SnapshotBacktestRunner's Step 2 disabled, so
            # reversals fell through to its Step 3 flip path, which needs TWO
            # bars (queue close -> fill at bar i+1's open -> queue entry -> fill
            # at bar i+2's open). run_vectorized_backtest defaults exit_cfg to
            # {"type": "opposite_signal"} and reverses in ONE bar. Every one of
            # the 8 pre-existing strategies registers without exit_cfg, so they
            # all disagreed between the two engines (e.g. ma_cross_v1 on BTC 1h:
            # 19 round trips / -3.74% event vs 37 / +7.51% vectorized).
            return {"type": "opposite_signal", "max_bars": 9999}
        cfg = dict(self.exit_cfg)
        etype = cfg.get("type")
        out: Dict = {"type": etype}

        if etype == "time_only":
            out["max_bars"] = int(cfg.get("max_bars", 9999))
            return out

        if etype == "pct_stop_tp":
            stop_pct = float(cfg.get("stop_pct", 0.02))
            tp_pct = float(cfg.get("tp_pct", 0.04))
            out["max_bars"] = int(cfg.get("max_bars", 9999))
            if entry_close is None:
                # Defer absolute prices to runner using entry fill price
                out["stop_pct"] = stop_pct
                out["tp_pct"] = tp_pct
                return out
            if side == "long":
                out["stop_loss_price"] = entry_close * (1 - stop_pct)
                out["take_profit_price"] = entry_close * (1 + tp_pct)
            else:  # short
                out["stop_loss_price"] = entry_close * (1 + stop_pct)
                out["take_profit_price"] = entry_close * (1 - tp_pct)
            # Also keep relative for runner that uses fill price
            out["stop_pct"] = stop_pct
            out["tp_pct"] = tp_pct
            return out

        if etype == "atr_stop_tp":
            stop_mult = float(cfg.get("stop_mult", 2.0))
            tp_mult = float(cfg.get("tp_mult", 3.0))
            out["max_bars"] = int(cfg.get("max_bars", 9999))
            out["stop_mult"] = stop_mult
            out["tp_mult"] = tp_mult
            if atr_value is None or entry_close is None:
                # Cannot compute absolute prices; runner should fall back to
                # max_bars only.
                return out
            if side == "long":
                out["stop_loss_price"] = entry_close - stop_mult * atr_value
                out["take_profit_price"] = entry_close + tp_mult * atr_value
            else:
                out["stop_loss_price"] = entry_close + stop_mult * atr_value
                out["take_profit_price"] = entry_close - tp_mult * atr_value
            out["atr_at_entry"] = atr_value
            return out

        if etype == "atr_trailing_stop":
            # Native ATR trailing stop. The runner threads ``running_peak``
            # through the spec on every bar (mutated in place):
            #   long:  running_peak = max(peak, bar_high)
            #          stop_price   = running_peak - trail_mult * atr_at_entry
            #          trigger      = bar_low <= stop_price
            #   short: running_peak = min(peak, bar_low)
            #          stop_price   = running_peak + trail_mult * atr_at_entry
            #          trigger      = bar_high >= stop_price
            # ``atr_at_entry`` is fixed at entry; using a static ATR keeps the
            # implementation simple and matches the H003 hypothesis. See
            # CYQNT_TRD_BUG_REPORT.md Issue #3.
            trail_mult = float(cfg.get("trail_mult", 3.0))
            out["max_bars"] = int(cfg.get("max_bars", 9999))
            out["trail_mult"] = trail_mult
            out["atr_period"] = int(cfg.get("atr_period", 14))
            if atr_value is None or entry_close is None:
                # Cannot compute trail width; runner should fall back to
                # max_bars only.
                return out
            out["atr_at_entry"] = atr_value
            # Initialize running_peak to entry_close; runner will replace
            # this with fill_price in _finalize_exit_prices, then update
            # each bar.
            out["running_peak"] = entry_close
            return out

        if etype == "ma_cross_exit":
            # Runner will rely on the strategy emitting opposite signal AND
            # check max_bars cap.
            out["max_bars"] = int(cfg.get("max_bars", 9999))
            out["period"] = int(cfg.get("period", 50))
            out["ma_type"] = cfg.get("ma_type", "ema")
            return out

        if etype == "opposite_signal":
            out["max_bars"] = int(cfg.get("max_bars", 9999))
            return out

        return None

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
# SELECTION plugin implementation
# ---------------------------------------------------------------------------


@dataclass
class SelectionStrategyPlugin:
    """A SignalPlugin adapter for cross-sectional SELECTION strategies.

    Unlike :class:`BlockStrategyPlugin` (single-instrument, per-bar boolean
    signals), a selection strategy ranks a *universe* of symbols at one point
    in time. It reads the :class:`UniverseBundle` off ``snapshot.universe`` and
    emits ONE ``SELECTION``-kind :class:`SignalEnvelope` carrying the ranked
    candidates in ``payload["candidates"]``. It rides the exact same registry /
    ``run_pipeline_step`` mechanism as trade plugins — only the input source
    (universe vs per-bar bars) and the emitted ``kind`` differ.
    """

    plugin_id: str
    plugin_version: str
    selection_fn: SelectionFn
    market_type: str = "futures"

    # ---- SignalPlugin protocol ----

    def required_inputs(self) -> Dict[str, bool]:
        return {
            "market": False, "social": True, "onchain": False,
            "derivatives": False, "universe": True,
        }

    def needed_timeframes(self) -> List[str]:
        return []

    def initialize_state(self):
        from ..standard_bot.signal.interfaces import SignalState  # type: ignore

        return SignalState(plugin_id=self.plugin_id, values={})

    def run(self, snapshot, config, context=None):  # type: ignore[no-untyped-def]
        return self._make_batch(snapshot, [self._build_envelope(snapshot, config)])

    def step(self, snapshot, state, config, context=None):  # type: ignore[no-untyped-def]
        from ..standard_bot.signal.interfaces import StepSignalResult  # type: ignore

        return StepSignalResult(state=state, signals=[self._build_envelope(snapshot, config)])

    # ---- helpers ----

    def _resolve_market_type(self, config) -> str:
        mt = getattr(config, "market_type", None)
        if mt is None and hasattr(config, "get"):
            mt = config.get("market_type")
        return str(mt or self.market_type)

    def _build_envelope(self, snapshot, config):
        import uuid as _uuid

        from ..standard_bot.core import (  # type: ignore
            SignalEnvelope, SignalKind, SignalProvenance,
        )

        ub = getattr(snapshot, "universe", None)
        decision_as_of = snapshot.meta.decision_as_of
        candidates: List[Dict] = []
        universe_size = 0
        if ub is not None:
            as_of = int(getattr(ub, "as_of", 0) or 0) or int(decision_as_of or 0)
            candidates = list(
                self.selection_fn(
                    ub.universe,
                    ub.ticker_rank,
                    ticker_rank_prev=ub.ticker_rank_prev,
                    klines=ub.klines,
                    as_of_ms=as_of,
                    market_type=self._resolve_market_type(config),
                )
                or []
            )
            try:
                universe_size = int(len(ub.universe)) if ub.universe is not None else 0
            except TypeError:
                universe_size = 0
        else:
            as_of = int(decision_as_of or 0)

        ns = _uuid.UUID("6d011acb-2b89-5dc5-bd38-fa9f903e6495")
        sig_id = str(
            _uuid.uuid5(ns, f"{snapshot.meta.snapshot_id}|{self.plugin_id}|selection|{as_of}")
        )
        return SignalEnvelope(
            version=self.plugin_version,
            signal_id=sig_id,
            kind=SignalKind.SELECTION,
            instrument_id=None,
            side=None,
            strength=1.0,
            time_horizon="swing",
            valid_until=decision_as_of,
            payload={
                "candidates": candidates,
                "as_of": as_of,
                "universe_size": universe_size,
            },
            provenance=SignalProvenance(
                plugin_id=self.plugin_id,
                plugin_version=self.plugin_version,
                config_hash=self._resolve_market_type(config),
                input_fingerprint=snapshot.meta.snapshot_id,
            ),
        )

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

#: A permanent map of strategy ID → BlockStrategyPlugin. Survives flushing
#: of ``_PENDING_REGISTRATIONS`` so callers (e.g. ``mvp_backtest``) can
#: introspect a registered strategy's metadata such as ``htf_specs`` /
#: ``needed_timeframes()`` AFTER ``make_registry()`` has run.
_KNOWN_BLOCK_PLUGINS: Dict[str, "BlockStrategyPlugin"] = {}

#: SELECTION-strategy analogues of the block registries above. Selection
#: plugins share ``_PENDING_REGISTRATIONS`` (so ``flush_pending_into`` drains
#: them into the same registry) but are tracked separately so entrypoints can
#: branch on strategy kind.
_KNOWN_SELECTION_STRATEGY_IDS: set = set()
_KNOWN_SELECTION_PLUGINS: Dict[str, "SelectionStrategyPlugin"] = {}


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


def is_known_selection_strategy(strategy_id: str) -> bool:
    """Return True if a SELECTION strategy with this ID has been registered."""
    return strategy_id in _KNOWN_SELECTION_STRATEGY_IDS


def get_selection_plugin(strategy_id: str) -> Optional["SelectionStrategyPlugin"]:
    """Return the registered :class:`SelectionStrategyPlugin` for *strategy_id*."""
    return _KNOWN_SELECTION_PLUGINS.get(strategy_id)


def get_block_plugin(strategy_id: str) -> Optional["BlockStrategyPlugin"]:
    """Return the registered :class:`BlockStrategyPlugin` for *strategy_id*.

    Returns None if no such block strategy has been registered. Useful for
    entrypoints that need to inspect the plugin's ``htf_specs`` /
    ``needed_timeframes()`` to assemble the correct ``MarketQuery``.
    """
    return _KNOWN_BLOCK_PLUGINS.get(strategy_id)


def registered_block_strategies() -> list:
    """Return the list of currently pending block strategies (for tests)."""
    return [(plugin, factory) for plugin, factory in _PENDING_REGISTRATIONS]
