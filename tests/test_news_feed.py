"""Tests for the PUBLIC news data layer + lookahead-safe feature builder.

The lookahead-safety guarantee is the reason this module exists, so the tests
lean hard on it:

* ``test_future_perturbation_does_not_change_past_bars`` — the headline
  invariant: perturbing/adding a capture with ``avail_ts > T`` never changes any
  feature attached to a bar at time ``T``.
* ``test_gating_uses_availability_not_content_time`` — proves gating keys on the
  capture's *availability* time, not the article's *content* time.
* ``test_first_seen_dedup_and_warmup`` — a news id spanning N captures is counted
  once (at first-seen); warmup bars are NaN (ratio) / 0.0 (count, flag).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks import news_feed, universe
from cyqnt_trd.blocks.news_feed import (
    attach_news_features,
    build_pit_feature_frame,
    load_pit_index,
    ticker_rank_universe,
)
from cyqnt_trd.data_cli import (
    fetch_news,
    fetch_sentiment,
    fetch_ticker_rank,
)
from cyqnt_trd.data_cli import news as news_mod
from cyqnt_trd.data_cli._cache import cache_clear


# ---------------------------------------------------------------------------
# PIT fixture builders
# ---------------------------------------------------------------------------
def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _envelope(data) -> dict:
    return {"code": "000000", "message": None, "data": data, "success": True}


def write_capture(
    root,
    captured_at_ms: int,
    *,
    token: str = "BTC",
    news_items=None,
    search_items=None,
    sentiment=None,      # (bullish, bearish)
    ticker_rank=None,    # list of dicts {ticker, mentionCount, bullishCount, bearishCount, neutralCount}
    completed_ms=None,
) -> str:
    """Write one capture dir under *root* and return its path.

    ``completed_ms`` defaults to ``captured_at_ms`` so ``avail_ts`` equals
    ``captured_at_ms`` regardless of ``avail_field``.
    """
    root = str(root)
    completed_ms = captured_at_ms if completed_ms is None else completed_ms
    stamp = datetime.fromtimestamp(captured_at_ms / 1000, tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    day = stamp[:8]
    cap_dir = os.path.join(root, day, f"capture_{stamp}Z")
    os.makedirs(cap_dir, exist_ok=True)

    manifest = {
        "captured_at_ms": int(captured_at_ms),
        "captured_at_utc": _ms_to_iso(captured_at_ms),
        "capture_completed_at_utc": _ms_to_iso(completed_ms),
        "env": "prod",
        "lang": "en",
        "endpoint_received_at_utc": {"news_page_01": _ms_to_iso(completed_ms)},
    }
    _write_json(os.path.join(cap_dir, "capture_manifest.json"), manifest)

    _write_json(
        os.path.join(cap_dir, "news_page_01.json"),
        _envelope({"items": list(news_items or []), "generatedAt": int(captured_at_ms)}),
    )
    if search_items is not None:
        _write_json(
            os.path.join(cap_dir, f"square_search_{token}.json"),
            _envelope({"items": list(search_items), "generatedAt": int(captured_at_ms)}),
        )
    if sentiment is not None:
        bull, bear = sentiment
        _write_json(
            os.path.join(cap_dir, f"square_sentiment_{token}.json"),
            _envelope({
                "bullishValue": bull, "bearishValue": bear,
                "totalValue": bull + bear, "pollStatus": None,
                "generatedAt": int(captured_at_ms),
            }),
        )
    if ticker_rank is not None:
        _write_json(
            os.path.join(cap_dir, "square_ticker_rank.json"),
            _envelope({"items": list(ticker_rank), "generatedAt": int(captured_at_ms)}),
        )
    return cap_dir


def _news_item(_id: str, *, tickers=("BTCUSDT",), date=None, generated=None) -> dict:
    return {
        "id": _id,
        "title": f"news {_id}",
        "summary": "",
        "date": date if date is not None else 0,
        "latestReleaseTime": date if date is not None else 0,
        "contentType": 2,
        "tendency": 0,
        "detectedLang": "en",
        "isCreatedByAI": False,
        "authorName": "a",
        "authorRole": "",
        "likeCount": 1, "commentCount": 0, "shareCount": 0, "viewCount": 10,
        "quoteCount": 0, "bookmarkCount": 0, "rank": 0, "score": 0.0,
        "hashtagList": [], "tickers": list(tickers), "userInputTickers": [],
    }


def _rank_item(ticker, mention, bull, bear, neutral=0) -> dict:
    return {
        "ticker": ticker, "mentionCount": mention, "uniqueAuthors": mention // 2,
        "totalEngagement": mention * 3, "bullishCount": bull, "bearishCount": bear,
        "neutralCount": neutral, "topPosts": [],
    }


def _bars(close_times) -> pd.DataFrame:
    close_times = list(close_times)
    return pd.DataFrame({
        "open_time": [t - 500 for t in close_times],
        "close_time": close_times,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })


# ===========================================================================
# 1. Data layer: typed cache-miss frames, no exceptions
# ===========================================================================
class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    def get_news(self, **kw):
        return self._resp

    def get_sentiment(self, **kw):
        return self._resp

    def get_ticker_rank(self, **kw):
        return self._resp


def _patch_client(monkeypatch, resp):
    monkeypatch.setattr(news_mod, "get_public_client", lambda **kw: _FakeClient(resp))


def test_fetch_cache_miss_returns_empty_typed_df(monkeypatch):
    cache_clear()
    # code == 000000 with data == None → cache miss → empty typed frame, no raise
    _patch_client(monkeypatch, _envelope(None))
    df = fetch_news(lang="en", page_size=5)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == news_mod.NEWS_COLUMNS

    s = fetch_sentiment("BTC")
    assert s.empty and list(s.columns) == news_mod.SENTIMENT_COLUMNS

    r = fetch_ticker_rank()
    assert r.empty and list(r.columns) == news_mod.TICKER_RANK_COLUMNS


def test_fetch_transport_error_returns_empty_typed_df(monkeypatch):
    cache_clear()
    # a transport-level error envelope must also not raise
    _patch_client(monkeypatch, {"_error": "boom"})
    df = fetch_news()
    assert df.empty and list(df.columns) == news_mod.NEWS_COLUMNS


def test_fetch_news_parses_typed_rows(monkeypatch):
    cache_clear()
    items = [_news_item("N1", tickers=["BTCUSDT"]), _news_item("N2", tickers=["ETHUSDT"])]
    _patch_client(monkeypatch, _envelope({"items": items, "generatedAt": 1784764804580}))
    df = fetch_news()
    assert len(df) == 2
    assert df["id"].tolist() == ["N1", "N2"]
    assert df["generated_at"].iloc[0] == 1784764804580
    assert df["tickers"].iloc[0] == ["BTCUSDT"]


def test_fetch_sentiment_bull_ratio(monkeypatch):
    cache_clear()
    _patch_client(monkeypatch, _envelope({
        "bullishValue": 60, "bearishValue": 40, "totalValue": 100,
        "pollStatus": None, "generatedAt": 123,
    }))
    df = fetch_sentiment("btc")
    assert len(df) == 1
    assert df["token"].iloc[0] == "BTC"
    assert df["bull_ratio"].iloc[0] == pytest.approx(0.6)


# ===========================================================================
# 2. THE lookahead-safety test: future perturbation cannot move past bars
# ===========================================================================
def _standard_pit(root):
    # c1 available at t=10_000, c2 available at t=20_000
    write_capture(root, 10_000, token="BTC",
                  news_items=[_news_item("N1")],
                  sentiment=(60, 40),
                  ticker_rank=[_rank_item("BTC", 100, 6, 4)])
    write_capture(root, 20_000, token="BTC",
                  news_items=[_news_item("N1"), _news_item("N2")],
                  sentiment=(30, 70),
                  ticker_rank=[_rank_item("BTC", 200, 3, 7)])


def test_future_perturbation_does_not_change_past_bars(tmp_path):
    root = tmp_path / "pit"
    _standard_pit(root)
    cfg = {"pit_root": str(root)}
    # Bars: b0 warmup (<c1), b1 sees only c1, b2 sees c2
    df = _bars([5_000, 15_000, 25_000])

    before = attach_news_features(df, cfg, "BTCUSDT")

    # Perturb a FUTURE capture (c2, avail 20_000 > b1's 15_000): rewrite it with
    # wildly different values and an extra first-seen news id.
    write_capture(root, 20_000, token="BTC",
                  news_items=[_news_item("N1"), _news_item("N2"), _news_item("N3")],
                  sentiment=(99, 1),
                  ticker_rank=[_rank_item("BTC", 9999, 99, 1)])

    after = attach_news_features(df, cfg, "BTCUSDT")

    past = [0, 1]  # bars with close_time < 20_000 (the perturbed capture's avail)
    for col in news_feed.NEWS_FEATURE_COLUMNS:
        b = before[col].to_numpy()
        a = after[col].to_numpy()
        for i in past:
            assert (np.isnan(b[i]) and np.isnan(a[i])) or b[i] == a[i], (
                f"past bar {i} column {col} changed: {b[i]} -> {a[i]}"
            )

    # And the perturbation MUST have moved the bar that legitimately sees c2,
    # otherwise the test proves nothing.
    assert after["_news_sentiment_bull_ratio"].iloc[2] == pytest.approx(0.99)
    assert before["_news_sentiment_bull_ratio"].iloc[2] == pytest.approx(0.3)
    assert after["_news_ticker_mention_count"].iloc[2] == 9999
    assert after["_news_count"].iloc[2] != before["_news_count"].iloc[2]


def test_attach_matches_latest_available_capture(tmp_path):
    root = tmp_path / "pit"
    _standard_pit(root)
    cfg = {"pit_root": str(root)}
    df = _bars([5_000, 15_000, 25_000])
    out = attach_news_features(df, cfg, "BTCUSDT")

    # bar 0: no capture available yet (warmup)
    assert np.isnan(out["_news_sentiment_bull_ratio"].iloc[0])
    assert out["_news_ticker_mention_count"].iloc[0] == 0.0
    # bar 1: latest capture with avail <= 15_000 is c1
    assert out["_news_sentiment_bull_ratio"].iloc[1] == pytest.approx(0.6)
    assert out["_news_ticker_mention_count"].iloc[1] == 100
    assert out["_news_ticker_mention_rank"].iloc[1] == 1.0
    # bar 2: latest capture with avail <= 25_000 is c2
    assert out["_news_sentiment_bull_ratio"].iloc[2] == pytest.approx(0.3)
    assert out["_news_ticker_mention_count"].iloc[2] == 200


# ===========================================================================
# 3. Availability time, NOT content time
# ===========================================================================
def test_gating_uses_availability_not_content_time(tmp_path):
    root = tmp_path / "pit"
    # Capture available at t=2000, but its news CONTENT date is in the far
    # future (5000) and sentiment generatedAt is also future. A correct
    # implementation gates on availability (2000), so a bar at 3000 sees it.
    write_capture(root, 2000, token="BTC",
                  news_items=[_news_item("NX", date=5000, generated=5000)],
                  sentiment=(70, 30),
                  ticker_rank=[_rank_item("BTC", 50, 7, 3)])
    cfg = {"pit_root": str(root)}
    df = _bars([1000, 3000])  # b0 < avail, b1 >= avail but < content date
    out = attach_news_features(df, cfg, "BTCUSDT")

    # b0 (1000 < avail 2000): warmup
    assert out["_news_count"].iloc[0] == 0.0
    assert np.isnan(out["_news_sentiment_bull_ratio"].iloc[0])
    # b1 (3000 >= avail 2000): sees the news even though content date is 5000.
    # If gating erroneously used content time (5000 > 3000) this would be 0.
    assert out["_news_count"].iloc[1] == 1.0
    assert out["_news_event_flag"].iloc[1] == 1.0
    assert out["_news_sentiment_bull_ratio"].iloc[1] == pytest.approx(0.7)


# ===========================================================================
# 4. First-seen dedup + warmup fills
# ===========================================================================
def test_first_seen_dedup_and_warmup(tmp_path):
    root = tmp_path / "pit"
    _standard_pit(root)  # N1 in c1&c2, N2 only in c2
    cfg = {"pit_root": str(root)}
    df = _bars([5_000, 15_000, 25_000])
    out = attach_news_features(df, cfg, "BTCUSDT")

    # N1 counted once at first-seen bar (b1), N2 once at b2 — never double-counted
    assert out["_news_count"].tolist() == [0.0, 1.0, 1.0]
    assert out["_news_count"].sum() == 2.0  # two unique ids total
    assert out["_news_event_flag"].tolist() == [0.0, 1.0, 1.0]

    # Warmup bar b0: ratio-like → NaN, count/flag-like → 0.0
    assert np.isnan(out["_news_sentiment_bull_ratio"].iloc[0])
    assert np.isnan(out["_news_ticker_mention_rank"].iloc[0])
    assert np.isnan(out["_news_ticker_bull_ratio"].iloc[0])
    assert out["_news_ticker_mention_count"].iloc[0] == 0.0
    assert out["_news_count"].iloc[0] == 0.0
    assert out["_news_event_flag"].iloc[0] == 0.0


def test_first_seen_dedup_across_search_source(tmp_path):
    root = tmp_path / "pit"
    # Same id appears via global news in c1 and via per-token search in c2 —
    # still one unique id, counted once.
    write_capture(root, 10_000, token="BTC",
                  news_items=[_news_item("DUP")])
    write_capture(root, 20_000, token="BTC",
                  news_items=[], search_items=[_news_item("DUP"), _news_item("FRESH")])
    cfg = {"pit_root": str(root)}
    out = attach_news_features(_bars([5_000, 15_000, 25_000]), cfg, "BTCUSDT")
    assert out["_news_count"].tolist() == [0.0, 1.0, 1.0]  # DUP@b1, FRESH@b2


def test_non_matching_ticker_not_counted(tmp_path):
    root = tmp_path / "pit"
    write_capture(root, 10_000, token="BTC",
                  news_items=[_news_item("E1", tickers=["ETHUSDT"])])
    cfg = {"pit_root": str(root)}
    out = attach_news_features(_bars([5_000, 15_000]), cfg, "BTCUSDT")
    assert out["_news_count"].tolist() == [0.0, 0.0]


# ===========================================================================
# 5. Config / robustness
# ===========================================================================
def test_missing_pit_root_strict_raises():
    with pytest.raises(ValueError):
        attach_news_features(_bars([1, 2]), {"pit_root": "/no/such/dir"}, "BTCUSDT")


def test_missing_pit_root_non_strict_attaches_warmup():
    out = attach_news_features(
        _bars([1, 2]), {"pit_root": "/no/such/dir", "strict": False}, "BTCUSDT"
    )
    assert out["_news_count"].tolist() == [0.0, 0.0]
    assert np.isnan(out["_news_sentiment_bull_ratio"]).all()


def test_empty_df_returns_typed_columns(tmp_path):
    root = tmp_path / "pit"
    _standard_pit(root)
    out = attach_news_features(_bars([]), {"pit_root": str(root)}, "BTCUSDT")
    assert out.empty
    for col in news_feed.NEWS_FEATURE_COLUMNS:
        assert col in out.columns


def test_as_of_open_is_stricter(tmp_path):
    root = tmp_path / "pit"
    write_capture(root, 10_000, token="BTC", sentiment=(60, 40),
                  ticker_rank=[_rank_item("BTC", 100, 6, 4)])
    cfg = {"pit_root": str(root)}
    # A single bar whose open_time (9_000) is before the capture but close_time
    # (11_000) is after. as_of='open' must NOT see it; as_of='close' must.
    df = pd.DataFrame({"open_time": [9_000], "close_time": [11_000],
                       "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    open_out = attach_news_features(df, cfg, "BTCUSDT", as_of="open")
    close_out = attach_news_features(df, cfg, "BTCUSDT", as_of="close")
    assert np.isnan(open_out["_news_sentiment_bull_ratio"].iloc[0])
    assert close_out["_news_sentiment_bull_ratio"].iloc[0] == pytest.approx(0.6)


def test_as_of_open_without_open_time_raises(tmp_path):
    """Regression: as_of='open' must RAISE when open_time is missing, not
    silently fall back to close_time (which would leak within-bar future data).
    """
    root = tmp_path / "pit"
    write_capture(root, 10_000, token="BTC", sentiment=(60, 40))
    cfg = {"pit_root": str(root)}
    # Frame has ONLY close_time (no open_time).
    df = pd.DataFrame({"close_time": [11_000, 12_000],
                       "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    with pytest.raises(ValueError, match="open_time"):
        attach_news_features(df, cfg, "BTCUSDT", as_of="open")
    # as_of='close' on the same frame must still work (close_time present).
    out = attach_news_features(df, cfg, "BTCUSDT", as_of="close")
    assert out["_news_sentiment_bull_ratio"].iloc[0] == pytest.approx(0.6)


# ===========================================================================
# 6. ticker_rank_universe (selection) — lookahead-safe
# ===========================================================================
def test_ticker_rank_universe_lookahead(tmp_path):
    root = tmp_path / "pit"
    write_capture(root, 10_000, ticker_rank=[_rank_item("BTC", 100, 6, 4), _rank_item("ETH", 50, 3, 2)])
    write_capture(root, 20_000, ticker_rank=[_rank_item("SOL", 300, 9, 1), _rank_item("BTC", 120, 6, 4)])
    idx = load_pit_index(str(root))

    # Before any capture → empty
    assert ticker_rank_universe(idx, 5_000).empty
    # Between captures → only c1's ranking (BTC top by mentions)
    u1 = ticker_rank_universe(idx, 15_000)
    assert u1["ticker"].tolist() == ["BTC", "ETH"]
    assert u1["rank"].tolist() == [1, 2]
    # After c2 → c2's ranking (SOL top)
    u2 = ticker_rank_universe(idx, 25_000, top_n=1)
    assert u2["ticker"].tolist() == ["SOL"]


# ===========================================================================
# 7. universe.py news helpers
# ===========================================================================
def _rank_df():
    return pd.DataFrame([
        {"rank": 1, "ticker": "BNB", "mention_count": 36013, "unique_authors": 9651,
         "total_engagement": 0, "bullish_count": 219, "bearish_count": 115, "neutral_count": 35679, "generated_at": 0},
        {"rank": 2, "ticker": "BTC", "mention_count": 27722, "unique_authors": 10202,
         "total_engagement": 0, "bullish_count": 945, "bearish_count": 439, "neutral_count": 26338, "generated_at": 0},
        {"rank": 3, "ticker": "ETH", "mention_count": 8619, "unique_authors": 4957,
         "total_engagement": 0, "bullish_count": 519, "bearish_count": 180, "neutral_count": 7920, "generated_at": 0},
    ])


def test_universe_news_helpers():
    tickers = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]})
    aug = universe.augment_with_news(tickers, _rank_df())

    # base <-> <BASE>USDT join
    btc = aug[aug["symbol"] == "BTCUSDT"].iloc[0]
    assert btc["news_mention_count"] == 27722
    assert btc["news_bull_ratio"] == pytest.approx(945 / (945 + 439))
    # XRP has no news row → NaN
    assert np.isnan(aug[aug["symbol"] == "XRPUSDT"].iloc[0]["news_mention_count"])

    # top_mentioned → BNB is highest
    assert universe.top_mentioned(aug, n=1)["symbol"].tolist() == ["BNBUSDT"]

    # top_bullish → ETH has the highest bull ratio among the three
    assert universe.top_bullish(aug, n=1)["symbol"].tolist() == ["ETHUSDT"]

    # filter_sentiment keeps only >= 0.7 bull ratio (BTC .68, ETH .74, BNB .656)
    kept = set(universe.filter_sentiment(aug, min_bull_ratio=0.7)["symbol"])
    assert kept == {"ETHUSDT"}


def test_universe_augment_empty_rank_gives_nan():
    tickers = pd.DataFrame({"symbol": ["BTCUSDT"]})
    aug = universe.augment_with_news(tickers, pd.DataFrame())
    assert np.isnan(aug["news_mention_count"].iloc[0])


def test_universe_fluent_builder():
    tickers = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "BNBUSDT"]})
    out = universe.UniverseFilter(tickers).with_news(_rank_df()).top_mentioned(2).to_frame()
    assert set(out["symbol"]) == {"BNBUSDT", "BTCUSDT"}


# ===========================================================================
# 8. load_pit_index basics
# ===========================================================================
def test_load_pit_index_sorted_and_empty(tmp_path):
    assert load_pit_index(str(tmp_path / "missing")).empty
    root = tmp_path / "pit"
    write_capture(root, 20_000, sentiment=(1, 1))
    write_capture(root, 10_000, sentiment=(1, 1))
    idx = load_pit_index(str(root))
    assert idx["avail_ts"].tolist() == [10_000, 20_000]  # sorted ascending
