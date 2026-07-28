"""Tests for the live SELECTION assembler (Square API -> DataSnapshot.universe).

All network-hitting fetches are bypassed by passing frames in (or monkeypatching
fetch_klines), so these run fully offline. They prove the wiring:
Square/universe data -> UniverseBundle -> DataSnapshot.universe ->
register_selection() plugin -> ranked candidates, on the standard route.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.standard_bot.data import (
    build_selection_snapshot,
    build_universe_bundle,
    run_selection,
)
from cyqnt_trd.standard_bot.core import DataSnapshot, UniverseBundle


@pytest.fixture(autouse=True)
def _register_selection_bots():
    """Import + reload N1/N2 so register_selection() re-fires into current globals."""
    importlib.reload(importlib.import_module("strategies.news.news_catalyst_selector"))
    importlib.reload(importlib.import_module("strategies.news.social_heat_breakout"))


def _universe():
    return pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                         "quoteVolume": [5e8, 3e8, 2e8]})


def _rank():
    return pd.DataFrame({"ticker": ["BTC", "ETH", "SOL"], "mention_count": [120, 80, 50],
                         "bullish_count": [90, 20, 30], "bearish_count": [10, 50, 10],
                         "neutral_count": [5, 5, 5], "unique_authors": [40, 30, 20],
                         "rank": [1, 2, 3]})


def _breakout_klines(symbol="BTCUSDT"):
    n = 40
    close = pd.Series(np.r_[np.linspace(100, 101, n - 1), [108.0]])
    return {symbol: pd.DataFrame({
        "open": close.shift(1).fillna(100.0), "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.r_[np.full(n - 1, 1000.0), [9000.0]],
        "close_time": (np.arange(n) + 1) * 3_600_000})}


# --------------------------------------------------------------------------- #
# Assembler primitives                                                        #
# --------------------------------------------------------------------------- #


def test_build_universe_bundle_offline():
    ub = build_universe_bundle(as_of_ms=111, universe_df=_universe(), ticker_rank_df=_rank())
    assert isinstance(ub, UniverseBundle)
    assert ub.as_of == 111
    assert list(ub.universe["symbol"]) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert ub.ticker_rank is not None and ub.klines == {}
    assert ub.meta.data_source == "binance_square+24h_ticker"


def test_build_selection_snapshot_populates_universe():
    snap = build_selection_snapshot(as_of_ms=222, universe_df=_universe(), ticker_rank_df=_rank())
    assert isinstance(snap, DataSnapshot)
    assert snap.universe is not None
    assert snap.universe.as_of == 222
    assert snap.meta.decision_as_of == 222
    # market slot untouched — this is a selection-only snapshot
    assert snap.market is None


def test_kline_top_fetches_candidate_klines(monkeypatch):
    calls = []

    def fake_fetch_klines(symbol, interval, *, limit=200, market_type="futures"):
        calls.append((symbol, interval))
        return _breakout_klines(symbol)[symbol]

    import cyqnt_trd.blocks.data as _data
    monkeypatch.setattr(_data, "fetch_klines", fake_fetch_klines)

    ub = build_universe_bundle(as_of_ms=1, universe_df=_universe(), ticker_rank_df=_rank(),
                               kline_top=2, kline_interval="1h")
    assert set(ub.klines) == {"BTCUSDT", "ETHUSDT"}   # top-2 tickers only
    assert calls == [("BTCUSDT", "1h"), ("ETHUSDT", "1h")]


def test_kline_fetch_failure_is_tolerated(monkeypatch):
    def boom(symbol, interval, *, limit=200, market_type="futures"):
        raise RuntimeError("network down")

    import cyqnt_trd.blocks.data as _data
    monkeypatch.setattr(_data, "fetch_klines", boom)

    ub = build_universe_bundle(as_of_ms=1, universe_df=_universe(), ticker_rank_df=_rank(),
                               kline_top=3)
    assert ub.klines == {}   # failures skipped, bundle still built


# --------------------------------------------------------------------------- #
# run_selection: standard route end-to-end                                    #
# --------------------------------------------------------------------------- #


def test_run_selection_n1():
    cands = run_selection("news_catalyst_selector", as_of_ms=111,
                          universe_df=_universe(), ticker_rank_df=_rank())
    assert len(cands) == 3
    by_sym = {c["symbol"]: c for c in cands}
    assert by_sym["BTCUSDT"]["side"] == "long"
    assert by_sym["ETHUSDT"]["side"] == "short"


def test_run_selection_n2_with_klines():
    rank_now = pd.DataFrame({"ticker": ["BTC", "ETH"], "mention_count": [200, 60]})
    rank_prev = pd.DataFrame({"ticker": ["BTC", "ETH"], "mention_count": [100, 55]})
    cands = run_selection("social_heat_breakout", as_of_ms=222,
                          ticker_rank_df=rank_now, ticker_rank_prev_df=rank_prev,
                          klines=_breakout_klines("BTCUSDT"))
    assert [c["symbol"] for c in cands] == ["BTCUSDT"]     # ETH delta below threshold
    assert cands[0]["features"]["breakout"] is True
    assert "trade" in cands[0]                              # breakout confirmed -> trade plan


def test_run_selection_unregistered_raises():
    with pytest.raises(ValueError, match="no SELECTION strategy registered"):
        run_selection("does_not_exist", universe_df=_universe(), ticker_rank_df=_rank())
