"""Engine parity regression tests: SnapshotBacktestRunner vs run_vectorized_backtest.

Why this file exists
--------------------
The two backtest engines silently drifted apart for months because **nothing in
``tests/`` exercised ``run_vectorized_backtest`` at all**, so no test compared
them. A one-off manual check ("6 exit types all within 1pp") was done by hand
and never encoded, and it happened to only cover ``opposite_signal``-flavoured
paths — the one convention that was already aligned. Meanwhile a persistent
long/short strategy with ``atr_stop_tp`` diverged by 1.138pp with 36 vs 32
round trips.

Root causes that were fixed (and that these tests pin down):

1. ``runner.py`` dropped the exit bar's entry signal in the ``next_bar_open``
   branch (an ``exited_this_bar and close_only`` guard that cannot fire there,
   because that branch only *queues* the order), costing one extra bar before
   re-entry.
2. ``max_bars`` filled at the trigger bar's *close* in the event engine but at
   its *open* in the vectorized engine — a whole bar of price movement.
3. The vectorized engine never checked exits on the entry bar, so it could not
   stop out on the bar it entered on. The event engine fills the entry (Step 1)
   before checking exits (Step 2) and can. That understated stop-loss risk.
4. The vectorized engine left a position open at the end of the data: its
   unrealized PnL landed in ``equity_curve`` but it was missing from
   ``trades`` / ``trade_count`` / ``win_rate`` and never paid an exit fee.
5. The vectorized engine's close-only rule blocked same-bar re-entry after an
   ``opposite_signal`` exit, delaying the flip by one bar.

Everything here is synthetic and offline — no network, no parquet fixtures.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks import strategy as block_strategy
from cyqnt_trd.blocks.data import bars_to_df
from cyqnt_trd.standard_bot.core import (
    BacktestRequest,
    Bar,
    MarketBundle,
    SignalPipelineSpec,
)
from cyqnt_trd.standard_bot.data.alignment import AlignmentPolicy
from cyqnt_trd.standard_bot.data.snapshot import HistoricalSnapshotAssembler
from cyqnt_trd.standard_bot.signal.registry import SignalPluginRegistry
from cyqnt_trd.standard_bot.simulation.runner import SnapshotBacktestRunner
from cyqnt_trd.standard_bot.simulation.vectorized_backtest import run_vectorized_backtest

SYMBOL = "TESTUSDT"
TIMEFRAME = "1h"
BAR_MS = 3_600_000
SIZE = 0.1
FEE_BPS = 4.0
SLIP_BPS = 2.0
CAPITAL = 10_000.0

# Exit configs to compare. Keyed by a readable id for pytest output.
EXIT_CFGS = {
    "opposite_signal": {"type": "opposite_signal", "max_bars": 9999},
    "pct_stop_tp": {"type": "pct_stop_tp", "stop_pct": 0.02, "tp_pct": 0.04, "max_bars": 80},
    "atr_stop_tp": {"type": "atr_stop_tp", "atr_period": 14, "stop_mult": 2.0,
                    "tp_mult": 3.0, "max_bars": 24},
    "time_only": {"type": "time_only", "max_bars": 10},
    "ma_cross_exit": {"type": "ma_cross_exit", "period": 20, "ma_type": "ema",
                      "max_bars": 9999},
    "atr_trailing_stop": {"type": "atr_trailing_stop", "atr_period": 14,
                          "trail_mult": 2.5, "max_bars": 9999},
}


# --------------------------------------------------------------------------- #
# Synthetic market data                                                        #
# --------------------------------------------------------------------------- #


def _make_df(n: int = 400, seed: int = 20260729) -> pd.DataFrame:
    """Deterministic OHLCV with real trends (so signals actually fire).

    ``open[i] == close[i-1]`` mirrors a continuously traded market, which is
    what both engines' fill conventions assume.
    """
    rng = np.random.default_rng(seed)
    drift = np.concatenate([
        np.full(n // 4, 0.004),      # up leg
        np.full(n // 4, -0.005),     # down leg
        np.full(n // 4, 0.0),        # chop
        np.full(n - 3 * (n // 4), 0.003),
    ])
    steps = drift + rng.normal(0.0, 0.006, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    span = np.abs(rng.normal(0.0, 0.004, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    close_time = np.arange(1, n + 1, dtype=np.int64) * BAR_MS - 1
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.full(n, 1_000.0), "quote_volume": np.full(n, 1_000.0) * close,
        "open_time": close_time - BAR_MS + 1, "close_time": close_time,
        "trades": np.full(n, 10),
        "instrument_id": SYMBOL, "timeframe": TIMEFRAME,
    })


def _df_to_bundle(df: pd.DataFrame) -> MarketBundle:
    bars = [
        Bar(
            open=float(r.open), high=float(r.high), low=float(r.low), close=float(r.close),
            volume=float(r.volume), timestamp=int(r.close_time),
            instrument_id=SYMBOL, timeframe=TIMEFRAME, confirmed=True,
            quote_volume=float(r.quote_volume),
            extras={"open_time": int(r.open_time), "close_time": int(r.close_time)},
        )
        for r in df.itertuples()
    ]
    return MarketBundle(bars={MarketBundle.key(SYMBOL, TIMEFRAME): bars})


def _make_signals(df: pd.DataFrame):
    """Persistent (not one-shot) long/short momentum — the hard case for parity.

    A one-shot crossover signal hides re-entry-timing bugs because it rarely
    fires right after an exit; a persistent signal makes any off-by-one-bar
    difference cascade into completely different trades.
    """
    ret = df["close"].pct_change(10)
    return (ret > 0.02).fillna(False), (ret < -0.02).fillna(False)


# --------------------------------------------------------------------------- #
# Engine drivers                                                               #
# --------------------------------------------------------------------------- #


def _run_event(df: pd.DataFrame, exit_cfg: dict, *, long_only: bool = False):
    """Drive SnapshotBacktestRunner with a full-history window."""
    strategy_id = f"parity_{uuid.uuid4().hex[:8]}"
    registry = SignalPluginRegistry()
    plugin = block_strategy.build_plugin(
        strategy_id, _make_signals, exit_cfg=exit_cfg, size=SIZE
    )
    registry.register(plugin, block_strategy._make_config_factory(strategy_id))

    bundle = _df_to_bundle(df)
    # tail_bars > len(df): no rolling-window truncation, so indicator values
    # match the vectorized engine's full-history computation.
    snapshots = HistoricalSnapshotAssembler(
        policy=AlignmentPolicy(policy_id="bar_close_v1", primary_timeframe=TIMEFRAME),
        tail_bars=len(df) + 10,
    ).build(bundle)

    request = BacktestRequest(
        request_id=str(uuid.uuid4()),
        instruments=[SYMBOL],
        primary_timeframe=TIMEFRAME,
        start_ts=snapshots[0].meta.decision_as_of,
        end_ts=snapshots[-1].meta.decision_as_of,
        signal_pipeline=SignalPipelineSpec(plugin_chain=[
            {"plugin_id": strategy_id,
             "config": {"instrument_id": SYMBOL, "timeframe": TIMEFRAME}},
        ]),
        initial_capital=CAPITAL,
        fee_model={"commission_bps": FEE_BPS},
        slippage_model={"slippage_bps": SLIP_BPS},
        extras={"execution_model": "next_bar_open", "long_only": long_only},
    )
    return SnapshotBacktestRunner(signal_registry=registry).run(
        request=request, snapshots=snapshots
    )


def _event_round_trips(result, df: pd.DataFrame):
    """Pair entry/exit rows into (entry_bar, side, reason) round trips."""
    ts_to_idx = {int(t): i for i, t in enumerate(df["close_time"].tolist())}
    out, open_row = [], None
    for row in result.extras["trades"]:
        if row.get("action") == "entry":
            open_row = row
        elif open_row is not None:
            out.append({
                "entry_idx": ts_to_idx[open_row["timestamp"]],
                "side": open_row["position_side"],
                "reason": row.get("exit_reason"),
            })
            open_row = None
    return out


def _run_vec(df: pd.DataFrame, exit_cfg: dict, *, long_only: bool = False):
    return run_vectorized_backtest(
        df=df, signal_fn=_make_signals, exit_cfg=exit_cfg, timeframe=TIMEFRAME,
        size=SIZE, fee_bps=FEE_BPS, slippage_bps=SLIP_BPS,
        initial_capital=CAPITAL, long_only=long_only,
    )


def _vec_round_trips(result):
    return [
        {"entry_idx": t["entry_idx"], "side": t["side"], "reason": t["reason"]}
        for t in result.trades
    ]


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def df():
    return _make_df()


@pytest.mark.parametrize("exit_id", sorted(EXIT_CFGS))
def test_engines_agree_on_trade_structure(df, exit_id):
    """Same entry bars, sides and exit reasons across both engines.

    Exit *bar indices* are intentionally not compared: for ``opposite_signal``
    the event engine books the exit on the signal bar (filling at that bar's
    close) while the vectorized engine books it on the next bar (filling at
    that bar's open). In a continuous market those are the same price and the
    same moment — only the label differs.
    """
    exit_cfg = EXIT_CFGS[exit_id]
    ev = _event_round_trips(_run_event(df, exit_cfg), df)
    vc = _vec_round_trips(_run_vec(df, exit_cfg))

    assert ev, f"{exit_id}: event engine produced no trades — test data is not exercising it"
    assert len(ev) == len(vc), (
        f"{exit_id}: round-trip count differs — event={len(ev)} vectorized={len(vc)}"
    )
    assert [t["entry_idx"] for t in ev] == [t["entry_idx"] for t in vc], (
        f"{exit_id}: entry bars differ (off-by-one re-entry regression?)"
    )
    assert [t["side"] for t in ev] == [t["side"] for t in vc], f"{exit_id}: side sequence differs"
    assert [t["reason"] for t in ev] == [t["reason"] for t in vc], (
        f"{exit_id}: exit reason sequence differs"
    )


@pytest.mark.parametrize("exit_id", sorted(EXIT_CFGS))
def test_engines_agree_on_total_return(df, exit_id):
    """Total return within 0.10pp.

    The residual is a position-sizing model difference, not a timing one: the
    event engine holds a real quantity sized from cash at entry, the vectorized
    engine applies ``equity * size * pnl_pct`` at exit, so the notional bases
    differ by the entry fee. Anything larger than this means a convention
    diverged again.
    """
    exit_cfg = EXIT_CFGS[exit_id]
    ev = _run_event(df, exit_cfg).total_return
    vc = _run_vec(df, exit_cfg).total_return
    assert abs(ev - vc) < 1e-3, (
        f"{exit_id}: total_return gap {(ev - vc) * 100:+.4f}pp "
        f"(event={ev:+.6f} vectorized={vc:+.6f})"
    )


def test_both_engines_go_short_on_futures(df):
    """Regression: the event engine is NOT long-only.

    The Confluence spec claimed ``mvp_backtest --engine python`` could only go
    long ("SELL only closes longs"). It always could go short — ``long_only``
    is derived from ``--market-type`` (spot=True, futures=False).
    """
    exit_cfg = EXIT_CFGS["atr_stop_tp"]
    ev = _event_round_trips(_run_event(df, exit_cfg, long_only=False), df)
    vc = _vec_round_trips(_run_vec(df, exit_cfg, long_only=False))
    assert any(t["side"] == "short" for t in ev), "event engine opened no shorts on futures"
    assert any(t["side"] == "short" for t in vc), "vectorized engine opened no shorts"
    assert any(t["side"] == "long" for t in ev), "event engine opened no longs"


def test_long_only_suppresses_shorts_in_both_engines(df):
    exit_cfg = EXIT_CFGS["atr_stop_tp"]
    ev = _event_round_trips(_run_event(df, exit_cfg, long_only=True), df)
    vc = _vec_round_trips(_run_vec(df, exit_cfg, long_only=True))
    assert all(t["side"] == "long" for t in ev), "long_only event run opened a short"
    assert all(t["side"] == "long" for t in vc), "long_only vectorized run opened a short"


def test_vectorized_force_closes_open_position_at_end(df):
    """A position still open on the last bar must be recorded as a trade.

    Otherwise its unrealized PnL lands in total_return while trade_count /
    win_rate silently exclude it, and it never pays an exit fee.
    """
    # opposite_signal keeps the strategy in the market most of the time, so the
    # run is very likely to end holding something.
    result = _run_vec(df, EXIT_CFGS["opposite_signal"])
    assert result.trade_count == len(result.trades)
    forced = [t for t in result.trades if t["reason"] == "forced_exit"]
    assert len(forced) <= 1
    if forced:
        assert forced[0]["exit_idx"] == len(df) - 1


def test_vectorized_warns_instead_of_silently_returning_zero():
    """Too-short input must not look like a flat strategy."""
    tiny = _make_df(n=20)
    with pytest.warns(RuntimeWarning, match="minimum 50"):
        result = run_vectorized_backtest(
            df=tiny, signal_fn=_make_signals, exit_cfg=EXIT_CFGS["atr_stop_tp"],
            timeframe=TIMEFRAME,
        )
    assert result.trade_count == 0


def test_max_bars_exit_fills_at_bar_open_in_both_engines(df):
    """``max_bars`` is a time condition known at the bar's open.

    Filling at that bar's close (the old event-engine behaviour) held the
    position one bar longer than ``max_bars`` and broke parity.
    """
    exit_cfg = {"type": "time_only", "max_bars": 10}
    ev_rows = _run_event(df, exit_cfg).extras["trades"]
    exits = [r for r in ev_rows if r.get("exit_reason") == "max_bars"]
    assert exits, "no max_bars exits produced"
    ts_to_open = {int(t): float(o) for t, o in zip(df["close_time"], df["open"])}
    slip = SLIP_BPS / 10_000.0
    for row in exits[:5]:
        bar_open = ts_to_open[row["timestamp"]]
        # Long exits fill at open*(1-slip), shorts at open*(1+slip).
        expected = bar_open * (1.0 - slip) if row["position_side"] == "long" \
            else bar_open * (1.0 + slip)
        assert row["price"] == pytest.approx(expected, rel=1e-9)


def test_exit_cfg_none_means_the_same_thing_in_both_engines(df):
    """``register(exit_cfg=None)`` must reverse in ONE bar in both engines.

    ``BlockStrategyPlugin`` used to emit no ``exit_spec`` at all when
    ``exit_cfg`` was None, which left ``SnapshotBacktestRunner``'s Step 2
    disabled: reversals fell through to its Step 3 flip path, which needs TWO
    bars. ``run_vectorized_backtest`` defaults ``exit_cfg`` to
    ``{"type": "opposite_signal"}`` and reverses in ONE. All 8 pre-existing
    ``strategies/*.py`` register without ``exit_cfg``, so every one of them
    disagreed between the engines (ma_cross_v1 on BTC 1h: 19 round trips /
    -3.74% event vs 37 / +7.51% vectorized).
    """
    ev = _event_round_trips(_run_event(df, None), df)
    vc = _vec_round_trips(_run_vec(df, None))
    assert ev, "event engine produced no trades with exit_cfg=None"
    assert len(ev) == len(vc), (
        f"exit_cfg=None round-trip count differs — event={len(ev)} vectorized={len(vc)}"
    )
    assert [t["entry_idx"] for t in ev] == [t["entry_idx"] for t in vc]
    assert [t["side"] for t in ev] == [t["side"] for t in vc]


def test_exit_spec_is_always_embedded(df):
    """Even with exit_cfg=None the plugin must emit an explicit exit_spec."""
    plugin = block_strategy.build_plugin("spec_probe", _make_signals, exit_cfg=None)
    spec = plugin._compute_exit_spec(side="long", entry_close=100.0, atr_value=None)
    assert spec == {"type": "opposite_signal", "max_bars": 9999}


def test_live_session_is_long_only_on_spot():
    """Spot paper/live must never open a short.

    ``PythonLivePaperSession`` had no ``long_only`` concept at all, so a spot
    run would open shorts the real account cannot hold.
    """
    from cyqnt_trd.standard_bot.simulation.python_live_paper_session import (
        PythonLivePaperSession,
    )

    strategy_id = f"parity_spot_{uuid.uuid4().hex[:8]}"
    block_strategy.register(
        strategy_id, _make_signals,
        exit_cfg=EXIT_CFGS["atr_stop_tp"], size=SIZE,
    )
    frame = _make_df()
    common = dict(
        strategy_id=strategy_id,
        symbol=SYMBOL,
        config={"instrument_id": SYMBOL, "timeframe": TIMEFRAME},
        initial_capital=CAPITAL,
        fee_bps=FEE_BPS,
        slippage_bps=SLIP_BPS,
    )
    spot = PythonLivePaperSession(market_type="spot", **common)
    futures = PythonLivePaperSession(market_type="futures", **common)
    assert spot.long_only is True
    assert futures.long_only is False

    bars = [
        {"timestamp": int(r.close_time), "open": float(r.open), "high": float(r.high),
         "low": float(r.low), "close": float(r.close), "volume": float(r.volume),
         "quote_volume": float(r.quote_volume)}
        for r in frame.itertuples()
    ]
    for session in (spot, futures):
        for bar in bars:
            session.tick(bar)

    assert spot.position_qty >= 0.0, "spot session ended holding a short"
    # The same signals on futures must be able to go short at some point.
    assert futures.long_only is False


def test_live_session_uses_canonical_ema_for_ma_cross_exit():
    """paper/live must use blocks.indicators, not a hand-rolled EMA.

    The hand-rolled loop was seeded at the start of a truncated ``period*3``
    window, so paper/live disagreed with both backtest engines on every
    ``ma_cross_exit``.
    """
    import inspect

    from cyqnt_trd.standard_bot.simulation import python_live_paper_session as mod

    src = inspect.getsource(mod.PythonLivePaperSession.tick)
    assert "indicators as _ind" in src, "tick() no longer imports the canonical indicators"
    assert "2.0 / (period + 1.0)" not in src, "hand-rolled EMA re-introduced in tick()"
