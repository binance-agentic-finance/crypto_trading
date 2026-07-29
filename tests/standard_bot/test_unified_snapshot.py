"""Tests for unified (trade + selection) DataSnapshot assembly.

``DataSnapshot`` always had both ``market`` (per-bar, trade) and ``universe``
(cross-sectional, selection) slots, but no assembler populated both:
``assemble_snapshot()`` did not even accept a ``universe`` argument,
``HistoricalSnapshotAssembler`` filled only ``market``, and
``build_selection_snapshot`` filled only ``universe``. These tests pin the
unified path down, including the point-in-time invariant *between* the two
halves — which nothing could check while they never coexisted.

All synthetic and offline: no network, no parquet.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.blocks import strategy as block_strategy
from cyqnt_trd.standard_bot.core import Bar, MarketBundle
from cyqnt_trd.standard_bot.data import (
    build_unified_snapshot,
    build_universe_bundle,
    universe_klines_to_market_bundle,
)

BAR_MS = 3_600_000
TF = "1h"


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _kline_df(base: float, n: int = 60) -> pd.DataFrame:
    close_time = [(i + 1) * BAR_MS - 1 for i in range(n)]
    return pd.DataFrame({
        "open_time": [c - (BAR_MS - 1) for c in close_time],
        "close_time": close_time,
        "open": [base + i for i in range(n)],
        "high": [base + i + 1 for i in range(n)],
        "low": [base + i - 1 for i in range(n)],
        "close": [base + i + 0.5 for i in range(n)],
        "volume": [10.0] * n,
        "quote_volume": [1_000.0] * n,
    })


def _market_bundle(symbol: str = "BTCUSDT", n: int = 60) -> MarketBundle:
    df = _kline_df(100.0, n)
    bars = [
        Bar(open=float(r.open), high=float(r.high), low=float(r.low),
            close=float(r.close), volume=float(r.volume),
            timestamp=int(r.close_time), instrument_id=symbol, timeframe=TF,
            confirmed=True, quote_volume=float(r.quote_volume),
            extras={"open_time": int(r.open_time), "close_time": int(r.close_time)})
        for r in df.itertuples()
    ]
    return MarketBundle(bars={MarketBundle.key(symbol, TF): bars})


def _universe_bundle(as_of: int, *, klines=None):
    return build_universe_bundle(
        as_of_ms=as_of,
        universe_df=pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                                  "quoteVolume": [5e8, 3e8, 2e8]}),
        ticker_rank_df=pd.DataFrame({
            "ticker": ["BTC", "ETH", "SOL"], "mention_count": [120, 80, 50],
            "bullish_count": [90, 20, 30], "bearish_count": [10, 50, 10],
            "neutral_count": [5, 5, 5], "unique_authors": [40, 30, 20],
            "rank": [1, 2, 3]}),
        klines=klines,
    )


LAST_BAR_TS = 60 * BAR_MS - 1


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


def test_one_snapshot_carries_both_market_and_universe():
    mb = _market_bundle()
    ub = _universe_bundle(LAST_BAR_TS)
    snap = build_unified_snapshot(market_bundle=mb, universe_bundle=ub,
                                  primary_timeframe=TF)
    assert snap.market is not None, "market half missing"
    assert snap.universe is not None, "universe half missing"
    assert snap.meta.decision_as_of == LAST_BAR_TS
    assert len(snap.require_market().bars[MarketBundle.key("BTCUSDT", TF)]) == 60


def test_decision_as_of_defaults_to_last_confirmed_bar():
    snap = build_unified_snapshot(market_bundle=_market_bundle(), primary_timeframe=TF)
    assert snap.meta.decision_as_of == LAST_BAR_TS
    assert snap.universe is None      # trade-only stays trade-only


def test_unconfirmed_bars_do_not_set_decision_as_of():
    """An in-flight bar must not become the PIT cutoff."""
    mb = _market_bundle()
    key = MarketBundle.key("BTCUSDT", TF)
    tail = mb.bars[key][-1]
    mb.bars[key].append(Bar(
        open=tail.close, high=tail.close + 1, low=tail.close - 1, close=tail.close,
        volume=1.0, timestamp=tail.timestamp + BAR_MS, instrument_id="BTCUSDT",
        timeframe=TF, confirmed=False, quote_volume=1.0,
        extras={"open_time": tail.timestamp + 1, "close_time": tail.timestamp + BAR_MS}))
    snap = build_unified_snapshot(market_bundle=mb, primary_timeframe=TF)
    assert snap.meta.decision_as_of == LAST_BAR_TS


def test_universe_from_the_future_is_rejected():
    """A ranking built after the last confirmed bar is lookahead."""
    mb = _market_bundle()
    ub = _universe_bundle(LAST_BAR_TS + 10 ** 7)
    with pytest.raises(ValueError, match="AFTER decision_as_of"):
        build_unified_snapshot(market_bundle=mb, universe_bundle=ub,
                               primary_timeframe=TF)


def test_strict_pit_false_downgrades_to_warning():
    mb = _market_bundle()
    ub = _universe_bundle(LAST_BAR_TS + 10 ** 7)
    with pytest.warns(RuntimeWarning, match="AFTER decision_as_of"):
        snap = build_unified_snapshot(market_bundle=mb, universe_bundle=ub,
                                      primary_timeframe=TF, strict_pit=False)
    assert snap.universe is not None


def test_universe_at_or_before_cutoff_is_accepted():
    mb = _market_bundle()
    for as_of in (LAST_BAR_TS, LAST_BAR_TS - BAR_MS):
        snap = build_unified_snapshot(market_bundle=mb,
                                      universe_bundle=_universe_bundle(as_of),
                                      primary_timeframe=TF)
        assert snap.universe.as_of == as_of


def test_fold_universe_klines_exposes_candidates_through_market():
    """Per-candidate klines become real Bars on the standard market path.

    ``UniverseBundle.klines`` is a third place bars can live — raw DataFrames
    keyed by bare symbol, versus ``MarketBundle``'s ``Bar`` objects keyed
    ``"SYMBOL|TIMEFRAME"``. Folding reconciles them.
    """
    ub = _universe_bundle(LAST_BAR_TS,
                          klines={"SOLUSDT": _kline_df(150.0),
                                  "AVAXUSDT": _kline_df(30.0)})
    plain = build_unified_snapshot(universe_bundle=ub, primary_timeframe=TF)
    assert plain.market is None

    folded = build_unified_snapshot(universe_bundle=ub, primary_timeframe=TF,
                                    fold_universe_klines=True)
    assert folded.market is not None
    assert set(folded.market.bars) == {MarketBundle.key("SOLUSDT", TF),
                                       MarketBundle.key("AVAXUSDT", TF)}
    sol = folded.market.bars[MarketBundle.key("SOLUSDT", TF)]
    assert len(sol) == 60
    assert isinstance(sol[0], Bar)
    assert sol[-1].close == pytest.approx(209.5)


def test_folding_never_overwrites_an_existing_market_series():
    """The PIT-clipped market series wins over convenience frames."""
    mb = _market_bundle("SOLUSDT")
    original_close = mb.bars[MarketBundle.key("SOLUSDT", TF)][-1].close
    ub = _universe_bundle(LAST_BAR_TS, klines={"SOLUSDT": _kline_df(999.0)})
    folded = universe_klines_to_market_bundle(ub, timeframe=TF, base=mb)
    assert folded.bars[MarketBundle.key("SOLUSDT", TF)][-1].close == original_close


def test_unified_snapshot_drives_both_plugin_kinds():
    """The whole point: ONE snapshot, both strategy types."""
    def make_signals(df):
        fast = df["close"].rolling(5).mean()
        slow = df["close"].rolling(20).mean()
        return (fast > slow).fillna(False), (fast < slow).fillna(False)

    def selection_fn(universe_df, ticker_rank_df=None, **_):
        return [{"symbol": str(universe_df.iloc[0]["symbol"]).upper(),
                 "rank": 1, "score": 1.0, "side": "long"}]

    from types import SimpleNamespace

    trade_plugin = block_strategy.build_plugin("uni_trade", make_signals, size=0.1)
    sel_plugin = block_strategy.build_selection_plugin("uni_sel", selection_fn)

    snap = build_unified_snapshot(market_bundle=_market_bundle(),
                                  universe_bundle=_universe_bundle(LAST_BAR_TS),
                                  primary_timeframe=TF)

    tb = trade_plugin.run(snap, SimpleNamespace(instrument_id="BTCUSDT", timeframe=TF))
    assert tb.trade_signals(), "trade plugin produced nothing from the unified snapshot"

    sb = sel_plugin.run(snap, SimpleNamespace(market_type="futures"))
    sels = sb.selection_signals()
    assert sels and sels[0].payload["candidates"], "selection plugin produced nothing"


def test_assemble_snapshot_universe_arg_is_additive():
    """Existing callers that omit ``universe`` are unaffected."""
    from cyqnt_trd.standard_bot.data import AlignmentPolicy, assemble_snapshot

    snap = assemble_snapshot(
        version="v", snapshot_id="s", assembled_at=LAST_BAR_TS,
        policy=AlignmentPolicy(policy_id="p", primary_timeframe=TF),
        market=_market_bundle(), decision_as_of=LAST_BAR_TS,
    )
    assert snap.universe is None
    assert snap.market is not None
