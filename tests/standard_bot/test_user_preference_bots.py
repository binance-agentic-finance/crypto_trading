"""Smoke tests for the 5 user-preference bots (T1/T2/T3 trade, N1/N2 selection).

These bots must ride the SAME route as the 8 pre-existing strategies:
  - trade bots:      make_signals(df) -> (long, short) + strategy.register()
  - selection bots:  selection_fn(...) -> candidates + strategy.register_selection()

``tests/test_strategies_compile.py`` only py_compiles the strategy files; it never
imports/executes them, so a broken register()/register_selection() call would slip
through. These tests close that gap: they import (→ register) the 5 modules and
exercise make_signals / the SelectionStrategyPlugin end-to-end on a DataSnapshot.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks import strategy as S
from cyqnt_trd.standard_bot.core import (
    DataSnapshot,
    MarketBundle,
    SignalKind,
    SnapshotMeta,
    UniverseBundle,
)
from cyqnt_trd.blocks.data import df_to_bars
from types import SimpleNamespace

TRADE_MODULES = {
    "structured_trade_plan": "strategies.technical.structured_trade_plan",
    "mtf_trend_follow": "strategies.technical.mtf_trend_follow",
    "deriv_positioning": "strategies.technical.deriv_positioning",
}
SELECTION_MODULES = {
    "news_catalyst_selector": "strategies.news.news_catalyst_selector",
    "social_heat_breakout": "strategies.news.social_heat_breakout",
}


@pytest.fixture
def bots():
    """Import + reload the 5 modules so their module-level register() /
    register_selection() re-fires into the current process globals — robust to
    another test's autouse fixture having cleared the registries first."""
    mods = {}
    for name, path in {**TRADE_MODULES, **SELECTION_MODULES}.items():
        mods[name] = importlib.reload(importlib.import_module(path))
    return mods


def _ohlcv(n: int = 400, *, seed: int = 0) -> pd.DataFrame:
    """Deterministic trending OHLCV with enough history for EMA(200)."""
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(np.linspace(0.02, 0.12, n)) + np.sin(np.arange(n) / 8))
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000 + rng.random(n) * 10,
            "open_time": np.arange(n) * 3_600_000,
            "close_time": (np.arange(n) + 1) * 3_600_000,
        }
    )


# --------------------------------------------------------------------------- #
# Registration                                                                #
# --------------------------------------------------------------------------- #


def test_trade_bots_registered(bots):
    for bot_id in TRADE_MODULES:
        assert S.is_known_block_strategy(bot_id), f"{bot_id} not registered as block strategy"
        assert S.get_block_plugin(bot_id) is not None


def test_deriv_bot_declares_derivatives(bots):
    plugin = S.get_block_plugin("deriv_positioning")
    reqs = plugin.required_inputs()
    assert reqs["derivatives"] is True
    # backward-compatible base keys unchanged
    assert reqs["market"] is True and reqs["social"] is False and reqs["onchain"] is False


def test_trend_bots_do_not_need_derivatives(bots):
    for bot_id in ("structured_trade_plan", "mtf_trend_follow"):
        assert S.get_block_plugin(bot_id).required_inputs()["derivatives"] is False


def test_selection_bots_registered(bots):
    for bot_id in SELECTION_MODULES:
        assert S.is_known_selection_strategy(bot_id), f"{bot_id} not registered as selection"
        assert S.get_selection_plugin(bot_id) is not None
        # selection bots must NOT masquerade as block strategies
        assert not S.is_known_block_strategy(bot_id)


# --------------------------------------------------------------------------- #
# Trade bots: make_signals                                                    #
# --------------------------------------------------------------------------- #


def test_trade_make_signals_shapes(bots):
    df = _ohlcv()
    for bot_id, mod in ((k, bots[k]) for k in TRADE_MODULES):
        long, short = mod.make_signals(df)
        assert long.dtype == bool and short.dtype == bool, bot_id
        assert len(long) == len(df) and len(short) == len(df), bot_id
        # long/short must be mutually exclusive per bar
        assert not bool((long & short).any()), f"{bot_id} emits long & short on same bar"


def test_t1_fires_long_in_uptrend(bots):
    long, short = bots["structured_trade_plan"].make_signals(_ohlcv())
    assert int(long.sum()) > 0  # a persistent uptrend should vote long somewhere


def test_t3_degrades_without_derivatives(bots):
    """T3 must NOT raise when open_interest/funding are absent — just no signals."""
    long, short = bots["deriv_positioning"].make_signals(_ohlcv())
    assert int(long.sum()) == 0 and int(short.sum()) == 0


def test_t3_fires_with_derivatives(bots):
    df = _ohlcv()
    n = len(df)
    df["open_interest"] = 1e6 + np.cumsum(np.linspace(-60, 60, n))
    df["funding_rate"] = np.linspace(-0.0008, 0.0008, n)
    long, short = bots["deriv_positioning"].make_signals(df)
    assert int(long.sum()) + int(short.sum()) > 0


def test_t3_reads_funding_oi_through_bars_to_df(bots):
    """The bars_to_df extras spillover must surface funding_rate/open_interest
    as df columns on the standard snapshot→bars_to_df path (no manual glue)."""
    df = _ohlcv(120)
    n = len(df)
    df["open_interest"] = 1e6 + np.arange(n) * 10.0
    df["funding_rate"] = 0.0001
    bars = df_to_bars(df, "BTCUSDT", "1h")
    df2 = __import__("cyqnt_trd.blocks.data", fromlist=["bars_to_df"]).bars_to_df(bars)
    assert "open_interest" in df2.columns and "funding_rate" in df2.columns
    assert df2["open_interest"].notna().all()


# --------------------------------------------------------------------------- #
# Trade bots: end-to-end on a DataSnapshot (same engine path as 8 built-ins)  #
# --------------------------------------------------------------------------- #


def test_trade_bot_runs_on_snapshot(bots):
    df = _ohlcv()
    bars = df_to_bars(df, "BTCUSDT", "1h")
    mb = MarketBundle(bars={MarketBundle.key("BTCUSDT", "1h"): bars})
    snap = DataSnapshot(
        version="v",
        market=mb,
        meta=SnapshotMeta(snapshot_id="s1", assembled_at=0,
                          decision_as_of=int(df["close_time"].iloc[-1])),
    )
    plugin = S.get_block_plugin("structured_trade_plan")
    batch = plugin.run(snap, SimpleNamespace(instrument_id="BTCUSDT", timeframe="1h"))
    # all emitted envelopes are TRADE-kind; selection filter is empty
    assert all(sig.kind == SignalKind.TRADE for sig in batch.signals)
    assert batch.selection_signals() == []


# --------------------------------------------------------------------------- #
# Selection bots: SelectionStrategyPlugin over a UniverseBundle               #
# --------------------------------------------------------------------------- #


def _run_selection(bot_id: str, ub: UniverseBundle):
    snap = DataSnapshot(
        version="v",
        meta=SnapshotMeta(snapshot_id="s1", assembled_at=0, decision_as_of=ub.as_of),
        universe=ub,
    )
    plugin = S.get_selection_plugin(bot_id)
    batch = plugin.run(snap, SimpleNamespace(market_type="futures"))
    sels = batch.selection_signals()
    assert len(sels) == 1
    env = sels[0]
    assert env.kind == SignalKind.SELECTION
    assert env.side is None and env.instrument_id is None  # cross-sectional, not per-instrument
    assert batch.trade_signals() == []  # inert to the trade/execution path
    return env


def test_n1_selection(bots):
    universe_df = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                                "quoteVolume": [5e8, 3e8, 2e8]})
    rank_df = pd.DataFrame({"ticker": ["BTC", "ETH", "SOL"], "mention_count": [120, 80, 50],
                            "bullish_count": [90, 20, 30], "bearish_count": [10, 50, 10],
                            "neutral_count": [5, 5, 5], "unique_authors": [40, 30, 20],
                            "rank": [1, 2, 3]})
    env = _run_selection("news_catalyst_selector", UniverseBundle(
        as_of=111, universe=universe_df, ticker_rank=rank_df))
    cands = env.payload["candidates"]
    assert len(cands) == 3
    by_sym = {c["symbol"]: c for c in cands}
    assert by_sym["BTCUSDT"]["side"] == "long"    # bull ratio 0.9
    assert by_sym["ETHUSDT"]["side"] == "short"   # bull ratio ~0.29
    # ranked by score descending
    assert [c["rank"] for c in cands] == [1, 2, 3]


def test_n2_heat_breakout_selection(bots):
    n = 40
    close = pd.Series(np.r_[np.linspace(100, 101, n - 1), [108.0]])  # breakout on last bar
    df = pd.DataFrame({"open": close.shift(1).fillna(100.0), "high": close + 0.5,
                       "low": close - 0.5, "close": close,
                       "volume": np.r_[np.full(n - 1, 1000.0), [9000.0]],
                       "close_time": (np.arange(n) + 1) * 3_600_000})
    rank_now = pd.DataFrame({"ticker": ["BTC", "ETH"], "mention_count": [200, 60]})
    rank_prev = pd.DataFrame({"ticker": ["BTC", "ETH"], "mention_count": [100, 55]})
    env = _run_selection("social_heat_breakout", UniverseBundle(
        as_of=222, ticker_rank=rank_now, ticker_rank_prev=rank_prev, klines={"BTCUSDT": df}))
    cands = env.payload["candidates"]
    # BTC delta=100 >= min_delta(50) selected; ETH delta=5 filtered out
    assert [c["symbol"] for c in cands] == ["BTCUSDT"]
    assert cands[0]["features"]["breakout"] is True
    assert "trade" in cands[0]  # confirmed → embedded trade plan


def test_selection_empty_without_universe(bots):
    """No universe bundle on the snapshot → empty candidates, not a crash."""
    snap = DataSnapshot(version="v",
                        meta=SnapshotMeta(snapshot_id="s", assembled_at=0, decision_as_of=1))
    for bot_id in SELECTION_MODULES:
        batch = S.get_selection_plugin(bot_id).run(snap, SimpleNamespace(market_type="futures"))
        env = batch.selection_signals()[0]
        assert env.payload["candidates"] == []


# --------------------------------------------------------------------------- #
# Export dicts (JS-consumable cyqnt.signal/v1) still valid                     #
# --------------------------------------------------------------------------- #


def test_trade_export_dict(bots):
    plan = bots["mtf_trend_follow"].generate(_ohlcv(), "BTCUSDT", "1h")
    for sig in plan:
        assert sig["schema"] == "cyqnt.signal/v1" and sig["kind"] == "trade"
        assert sig["symbol"] == "BTCUSDT" and sig["side"] in ("long", "short")


def test_selection_export_dict(bots):
    universe_df = pd.DataFrame({"symbol": ["BTCUSDT"], "quoteVolume": [5e8]})
    rank_df = pd.DataFrame({"ticker": ["BTC"], "mention_count": [120],
                            "bullish_count": [90], "bearish_count": [10],
                            "neutral_count": [5], "unique_authors": [40], "rank": [1]})
    out = bots["news_catalyst_selector"].generate_selection(universe_df, rank_df, as_of_ms=111)
    assert out[0]["schema"] == "cyqnt.signal/v1" and out[0]["kind"] == "selection"
