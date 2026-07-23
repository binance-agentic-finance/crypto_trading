"""Regression tests for cyqnt-trd engine bugs discovered during run_001.

See ``auto_opt_experiments/CYQNT_TRD_BUG_REPORT.md`` for full context.

Bug #1 — ATR-NaN in step-mode ``BlockStrategyPlugin._envelope_from_signals``
    Step mode filters ``df`` down to a 1-row ``emit_df`` then computes
    ATR on it, yielding NaN and producing exit_specs with
    ``atr_at_entry=NaN`` so the runner's stop-loss never triggers.

Bug #2 — Snapshot ``tail_bars=120`` over-clips HTF bars
    ``HistoricalSnapshotAssembler`` applied a single ``tail_bars`` cap to
    every timeframe, so a 4h HTF series was clipped to 30 bars in a
    1h-primary backtest, breaking SMA(N) for any N > 30.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks import indicators as ind, strategy
from cyqnt_trd.blocks.data import df_to_bars
from cyqnt_trd.standard_bot.core import (
    Bar,
    BundleMeta,
    DataSnapshot,
    MarketBundle,
    SnapshotMeta,
)
from cyqnt_trd.standard_bot.data.alignment import AlignmentPolicy
from cyqnt_trd.standard_bot.data.snapshot import HistoricalSnapshotAssembler


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0.0, 1.0, size=n).cumsum()
    close = np.clip(close, 50.0, None)
    high = close + np.abs(rng.normal(0.0, 0.5, size=n))
    low = close - np.abs(rng.normal(0.0, 0.5, size=n))
    open_ = close + rng.normal(0.0, 0.2, size=n)
    volume = np.abs(rng.normal(1000.0, 200.0, size=n))
    open_time = np.arange(n, dtype=np.int64) * 60_000
    close_time = open_time + 60_000 - 1
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": close_time,
            "quote_volume": volume * close,
            "trades": (volume / 10).astype("int64"),
        }
    )


def _make_bars_for_tf(n: int, *, tf: str, instrument: str = "BTCUSDT", interval_ms: int = 60_000):
    """Generate synthetic Bars of given count and timeframe."""
    df = _make_ohlcv(n=n, seed=1)
    open_time = np.arange(n, dtype=np.int64) * interval_ms
    close_time = open_time + interval_ms - 1
    bars = []
    for i in range(n):
        bars.append(
            Bar(
                open=float(df["open"].iloc[i]),
                high=float(df["high"].iloc[i]),
                low=float(df["low"].iloc[i]),
                close=float(df["close"].iloc[i]),
                volume=float(df["volume"].iloc[i]),
                timestamp=int(close_time[i]),
                instrument_id=instrument,
                timeframe=tf,
                confirmed=True,
                quote_volume=float(df["quote_volume"].iloc[i]),
                extras={"open_time": int(open_time[i]), "close_time": int(close_time[i])},
            )
        )
    return bars


# ---------------------------------------------------------------------------
# Bug #1 — ATR is computed on full df, not 1-row emit_df
# ---------------------------------------------------------------------------


def _build_snapshot(bars: list[Bar], snapshot_id: str = "snap_test") -> DataSnapshot:
    bundle = MarketBundle(
        bars={MarketBundle.key("BTCUSDT", "1m"): bars},
        meta=BundleMeta(data_source="test"),
    )
    return DataSnapshot(
        version="v1",
        market=bundle,
        social=None,
        onchain=None,
        meta=SnapshotMeta(snapshot_id=snapshot_id, assembled_at=0),
    )


def _register_atr_strategy(strategy_id: str = "__test_atr_step__"):
    strategy._PENDING_REGISTRATIONS.clear()
    strategy._KNOWN_BLOCK_STRATEGY_IDS.discard(strategy_id)
    strategy._KNOWN_BLOCK_PLUGINS.pop(strategy_id, None)

    def make_signals(df_: pd.DataFrame):
        # Simple deterministic signal: long whenever close > shifted close,
        # short whenever close < shifted close. Forces every bar to either
        # long or short so step mode actually emits an entry.
        long_s = df_["close"] > df_["close"].shift(1)
        short_s = df_["close"] < df_["close"].shift(1)
        return long_s.fillna(False).astype(bool), short_s.fillna(False).astype(bool)

    strategy.register(
        strategy_id,
        make_signals,
        exit_cfg={
            "type": "atr_stop_tp",
            "atr_period": 14,
            "stop_mult": 2.0,
            "tp_mult": 4.0,
            "max_bars": 9999,
        },
    )
    return strategy.get_block_plugin(strategy_id)


def test_bug1_step_mode_emits_finite_atr_in_exit_spec():
    """In step mode, per-step emits past ATR warmup must have finite ATR.

    Bug #1: ``BlockStrategyPlugin.step`` filters df down to a 1-row emit_df
    based on cursor, then computed ATR on that 1-row df → NaN. The fix
    pre-computes ATR on the FULL df before the cursor filter.

    To exercise this: feed snapshots one bar at a time with state retained
    so each step (after the first) emits exactly the new bar. ATR must be
    finite once we're past the warm-up period.
    """
    plugin = _register_atr_strategy("__test_bug1_step__")
    df = _make_ohlcv(n=200)
    bars = df_to_bars(df, instrument_id="BTCUSDT", timeframe="1m")

    state = plugin.initialize_state()
    from types import SimpleNamespace

    cfg = SimpleNamespace(instrument_id="BTCUSDT", timeframe="1m")

    # Build cursor state pointing past the ATR warm-up so that subsequent
    # snapshots emit exactly the new bars under cursor filtering. This is
    # the regime where Bug #1 used to produce NaN ATR.
    warmup = 30  # > atr_period(14)
    primer_snap = _build_snapshot(bars[:warmup], snapshot_id="primer")
    primer_result = plugin.step(primer_snap, state, cfg)
    state = primer_result.state

    finite_atr_count = 0
    nan_atr_count = 0
    for i in range(warmup, len(bars)):
        snap = _build_snapshot(bars[: i + 1], snapshot_id=f"snap_{i}")
        result = plugin.step(snap, state, cfg)
        state = result.state
        for s in result.signals:
            es = s.payload.get("exit_spec")
            if es and es.get("type") == "atr_stop_tp":
                atr_at_entry = es.get("atr_at_entry")
                if atr_at_entry is None or (
                    isinstance(atr_at_entry, float) and math.isnan(atr_at_entry)
                ):
                    nan_atr_count += 1
                else:
                    assert math.isfinite(atr_at_entry), (
                        f"unexpected non-finite atr_at_entry={atr_at_entry!r} "
                        f"in {es!r}"
                    )
                    sp = es.get("stop_loss_price")
                    assert sp is not None and math.isfinite(sp), (
                        f"stop_loss_price not finite: {es!r}"
                    )
                    finite_atr_count += 1

    assert finite_atr_count > 0, (
        "expected step-mode emits past ATR warm-up to have finite atr_at_entry; "
        f"got finite={finite_atr_count}, nan={nan_atr_count}"
    )
    # Bug #1 regression: pre-fix, ALL post-warm-up step emits had NaN ATR.
    # After fix, NaN should be exceedingly rare past warm-up.
    assert nan_atr_count == 0, (
        f"Bug #1 regression: {nan_atr_count} step-mode emits past warm-up "
        f"still have NaN atr_at_entry"
    )


def test_bug1_run_mode_still_works():
    """Regression guard: run() path (no cursor filter) keeps producing finite ATR.

    Early warm-up bars (< atr_period) legitimately have NaN ATR. We just
    require that the majority of post-warmup signals carry finite ATR.
    """
    plugin = _register_atr_strategy("__test_bug1_run__")
    df = _make_ohlcv(n=200)
    bars = df_to_bars(df, instrument_id="BTCUSDT", timeframe="1m")
    snap = _build_snapshot(bars, snapshot_id="snap_full")
    from types import SimpleNamespace

    cfg = SimpleNamespace(instrument_id="BTCUSDT", timeframe="1m")
    batch = plugin.run(snap, cfg)
    finite_atr = 0
    nan_atr = 0
    for s in batch.signals:
        es = s.payload.get("exit_spec")
        if es and es.get("type") == "atr_stop_tp":
            v = es.get("atr_at_entry")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                nan_atr += 1
            else:
                assert math.isfinite(v), es
                finite_atr += 1
    assert finite_atr > 0, "run() path should produce finite atr_at_entry post-warmup"
    # NaN signals are only the genuine pre-warmup bars; should be a small
    # fraction (<= ATR period * 2 to be safe).
    assert nan_atr <= 30, f"too many NaN ATR signals in run(): {nan_atr}"


# ---------------------------------------------------------------------------
# Bug #2 — HistoricalSnapshotAssembler should not over-clip HTF bars
# ---------------------------------------------------------------------------


def _make_mtf_bundle():
    """Build a MarketBundle with 1h primary + 4h HTF bars (well over 200 each).

    Primary spans 1200 hours so at the final snapshot anchor at least
    1200 / 4 = 300 confirmed 4h bars are visible — enough for HTF SMA(200)
    after the assembler's tail clipping.
    """
    primary = _make_bars_for_tf(n=1200, tf="1h", interval_ms=3_600_000)
    htf = _make_bars_for_tf(n=400, tf="4h", interval_ms=14_400_000)
    return MarketBundle(
        bars={
            MarketBundle.key("BTCUSDT", "1h"): primary,
            MarketBundle.key("BTCUSDT", "4h"): htf,
        },
        meta=BundleMeta(data_source="test"),
    )


def test_bug2_htf_bars_not_overclipped_by_default():
    """tail_bars=120 must not clip 4h HTF bars when primary is 1h."""
    bundle = _make_mtf_bundle()
    policy = AlignmentPolicy(policy_id="bar_close_v1", primary_timeframe="1h")
    asm = HistoricalSnapshotAssembler(policy=policy, tail_bars=120)
    snaps = asm.build(bundle)
    last = snaps[-1]
    primary_bars = last.market.bars[MarketBundle.key("BTCUSDT", "1h")]
    htf_bars = last.market.bars[MarketBundle.key("BTCUSDT", "4h")]
    assert len(primary_bars) == 120, "primary tf should still be capped at tail_bars"
    # HTF should retain enough bars for SMA(200) — pre-fix this would be 30.
    assert len(htf_bars) >= 200, (
        f"HTF over-clipped: got {len(htf_bars)} bars, expected >= 200 (Bug #2)"
    )


def test_bug2_dict_tail_bars_per_tf():
    """When tail_bars is a dict, each timeframe gets its own cap."""
    bundle = _make_mtf_bundle()
    policy = AlignmentPolicy(policy_id="bar_close_v1", primary_timeframe="1h")
    asm = HistoricalSnapshotAssembler(
        policy=policy,
        tail_bars={"1h": 100, "4h": 250, "default": 50},
    )
    snaps = asm.build(bundle)
    last = snaps[-1]
    assert len(last.market.bars[MarketBundle.key("BTCUSDT", "1h")]) == 100
    assert len(last.market.bars[MarketBundle.key("BTCUSDT", "4h")]) == 250


def test_bug2_legacy_behavior_when_htf_tail_bars_disabled():
    """Setting htf_tail_bars=0 restores the pre-fix uniform tail_bars cap."""
    bundle = _make_mtf_bundle()
    policy = AlignmentPolicy(policy_id="bar_close_v1", primary_timeframe="1h")
    asm = HistoricalSnapshotAssembler(policy=policy, tail_bars=120, htf_tail_bars=0)
    snaps = asm.build(bundle)
    last = snaps[-1]
    primary = last.market.bars[MarketBundle.key("BTCUSDT", "1h")]
    htf = last.market.bars[MarketBundle.key("BTCUSDT", "4h")]
    assert len(primary) == 120
    assert len(htf) == 120  # legacy: clipped to same 120


def test_bug2_primary_only_unaffected():
    """Single-TF backtests should be unchanged by the HTF override."""
    primary = _make_bars_for_tf(n=600, tf="1m", interval_ms=60_000)
    bundle = MarketBundle(
        bars={MarketBundle.key("BTCUSDT", "1m"): primary},
        meta=BundleMeta(data_source="test"),
    )
    policy = AlignmentPolicy(policy_id="bar_close_v1", primary_timeframe="1m")
    asm = HistoricalSnapshotAssembler(policy=policy, tail_bars=200)
    snaps = asm.build(bundle)
    last = snaps[-1]
    bars = last.market.bars[MarketBundle.key("BTCUSDT", "1m")]
    assert len(bars) == 200


# ---------------------------------------------------------------------------
# Combined integration: HTF SMA(200) is no longer all-NaN after fix
# ---------------------------------------------------------------------------


def test_bug2_integration_htf_sma_not_nan():
    """End-to-end: a strategy needing 4h SMA(200) gets a non-NaN aligned column."""
    strategy_id = "__test_bug2_integ__"
    strategy._PENDING_REGISTRATIONS.clear()
    strategy._KNOWN_BLOCK_STRATEGY_IDS.discard(strategy_id)
    strategy._KNOWN_BLOCK_PLUGINS.pop(strategy_id, None)

    def make_signals(d):
        long_s = d["close"] > d.get("_htf_4h_sma_200", pd.Series(np.nan, index=d.index))
        return long_s.fillna(False).astype(bool), pd.Series(False, index=d.index)

    strategy.register(strategy_id, make_signals, htf_specs=[("4h", 200)])
    plugin = strategy.get_block_plugin(strategy_id)

    bundle = _make_mtf_bundle()
    policy = AlignmentPolicy(policy_id="bar_close_v1", primary_timeframe="1h")
    asm = HistoricalSnapshotAssembler(policy=policy, tail_bars=120)
    snaps = asm.build(bundle)

    from types import SimpleNamespace

    cfg = SimpleNamespace(instrument_id="BTCUSDT", timeframe="1h")
    last = snaps[-1]
    df, _, _ = plugin._extract_df(last, cfg)
    htf_col = "_htf_4h_sma_200"
    assert htf_col in df.columns
    nan_count = int(df[htf_col].isna().sum())
    assert nan_count < len(df), (
        f"HTF SMA(200) column entirely NaN ({nan_count}/{len(df)}) after fix"
    )
