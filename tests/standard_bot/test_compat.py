"""Tests for cyqnt_trd.compat — backward-compatible atomic dataclass layer.

Round-trip coverage:
  - Candle ↔ DataFrame (df_to_candles / candles_to_df)
  - Signal series ↔ List[Signal]
  - TradePlan dict serialization round-trip
  - Verdict string constants and RANK dict
  - All atomic enum / type names are importable from cyqnt_trd.compat
  - to_dict / from_dict helpers on Candle, Signal, Score, TradePlan, OrderResult
"""

from __future__ import annotations

import math
import pytest
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Smoke: all public names must be importable
# ---------------------------------------------------------------------------

def test_compat_imports():
    """All atomic types are importable from cyqnt_trd.compat."""
    from cyqnt_trd.compat import (
        Candle, Ticker, FundingRate, OpenInterest, OIHistoryPoint,
        OrderBookLevel, OrderBook, Balance, Position, ExchangeFilter,
        Signal, Score, Verdict,
        TradePlan, OrderResult, VelocityMetrics,
        extract_closes,
        df_to_candles, candles_to_df,
        signal_series_to_signals, signals_to_series,
        trade_plan_to_dict, dict_to_trade_plan,
    )
    assert Candle is not None
    assert Verdict is not None


def test_top_level_cyqnt_trd_imports():
    """Atomic types must also be importable from cyqnt_trd top-level."""
    from cyqnt_trd import compat
    assert hasattr(compat, "Candle")
    assert hasattr(compat, "Verdict")
    assert hasattr(compat, "TradePlan")


# ---------------------------------------------------------------------------
# Candle dataclass
# ---------------------------------------------------------------------------

def _make_candle(**kwargs):
    from cyqnt_trd.compat import Candle
    defaults = dict(timestamp=1_700_000_000_000,
                    open=100.0, high=110.0, low=95.0,
                    close=105.0, volume=1000.0)
    defaults.update(kwargs)
    return Candle(**defaults)


def test_candle_defaults():
    c = _make_candle()
    assert c.quote_volume == 0.0
    assert c.trades == 0


def test_candle_to_dict_round_trip():
    from cyqnt_trd.compat import Candle
    c = _make_candle(quote_volume=5_000.0, trades=42)
    d = c.to_dict()
    c2 = Candle.from_dict(d)
    assert c2.timestamp == c.timestamp
    assert c2.close == c.close
    assert c2.quote_volume == 5_000.0
    assert c2.trades == 42


# ---------------------------------------------------------------------------
# df_to_candles / candles_to_df round-trip
# ---------------------------------------------------------------------------

def _make_df(n: int = 5, start_ts: int = 1_700_000_000_000) -> pd.DataFrame:
    ts = [start_ts + i * 60_000 for i in range(n)]
    opens  = [100.0 + i for i in range(n)]
    highs  = [o + 5.0 for o in opens]
    lows   = [o - 3.0 for o in opens]
    closes = [o + 2.0 for o in opens]
    vols   = [1000.0 + i * 100 for i in range(n)]
    return pd.DataFrame({
        "close_time": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
        "quote_volume": [v * 100 for v in vols],
        "trades": [i * 10 for i in range(n)],
    })


def test_df_to_candles_length():
    from cyqnt_trd.compat import df_to_candles
    df = _make_df(7)
    candles = df_to_candles(df)
    assert len(candles) == 7


def test_df_to_candles_field_values():
    from cyqnt_trd.compat import df_to_candles
    df = _make_df(3)
    candles = df_to_candles(df)
    for i, c in enumerate(candles):
        assert c.close == pytest.approx(102.0 + i)
        assert c.volume == pytest.approx(1000.0 + i * 100)
        assert c.trades == i * 10


def test_candles_to_df_round_trip():
    """df → candles → df must reproduce the same numeric values."""
    from cyqnt_trd.compat import df_to_candles, candles_to_df
    original = _make_df(10)
    candles = df_to_candles(original)
    recovered = candles_to_df(candles)

    for col in ["open", "high", "low", "close", "volume"]:
        assert list(original[col]) == pytest.approx(list(recovered[col]))
    assert list(original["close_time"]) == list(recovered["timestamp"])


def test_candles_to_df_columns():
    from cyqnt_trd.compat import df_to_candles, candles_to_df
    candles = df_to_candles(_make_df(4))
    df = candles_to_df(candles)
    expected = {"timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"}
    assert expected.issubset(set(df.columns))


def test_candles_to_df_empty():
    from cyqnt_trd.compat import candles_to_df
    df = candles_to_df([])
    assert len(df) == 0
    assert "close" in df.columns


def test_df_to_candles_no_timestamp_col():
    """If no timestamp/close_time column, row index is used."""
    from cyqnt_trd.compat import df_to_candles
    df = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5],
                        "close": [1.5], "volume": [100.0]})
    candles = df_to_candles(df)
    assert len(candles) == 1
    assert candles[0].timestamp == 0   # row index


def test_df_to_candles_uses_timestamp_col():
    """'timestamp' column takes precedence over 'close_time'."""
    from cyqnt_trd.compat import df_to_candles
    df = pd.DataFrame({
        "timestamp": [999],
        "open": [1.0], "high": [2.0], "low": [0.5],
        "close": [1.5], "volume": [100.0],
    })
    candles = df_to_candles(df)
    assert candles[0].timestamp == 999


def test_df_to_candles_missing_col_raises():
    from cyqnt_trd.compat import df_to_candles
    df = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5],
                        "close": [1.5]})   # missing 'volume'
    with pytest.raises(ValueError, match="volume"):
        df_to_candles(df)


# ---------------------------------------------------------------------------
# signal_series_to_signals / signals_to_series round-trip
# ---------------------------------------------------------------------------

def test_signal_series_to_signals_count():
    from cyqnt_trd.compat import signal_series_to_signals
    s = pd.Series([False, True, False, True, True, False])
    signals = signal_series_to_signals(s, "test_signal", direction="BULLISH")
    assert len(signals) == 3


def test_signal_series_to_signals_fields():
    from cyqnt_trd.compat import signal_series_to_signals
    s = pd.Series([True])
    sigs = signal_series_to_signals(s, "rsi_oversold", direction="BEARISH", strength=0.8)
    assert sigs[0].name == "rsi_oversold"
    assert sigs[0].direction == "BEARISH"
    assert sigs[0].strength == pytest.approx(0.8)


def test_signals_to_series_round_trip():
    """signal_series_to_signals → signals_to_series must reproduce original bool mask."""
    from cyqnt_trd.compat import signal_series_to_signals, signals_to_series
    original = pd.Series([False, True, False, False, True], name="entry")
    sigs = signal_series_to_signals(original, "entry_signal")
    recovered = signals_to_series(sigs, original.index)
    assert list(original) == list(recovered)


def test_signals_to_series_empty():
    from cyqnt_trd.compat import signals_to_series
    idx = pd.RangeIndex(5)
    series = signals_to_series([], idx)
    assert not series.any()
    assert len(series) == 5


def test_signals_to_series_all_false_for_out_of_range():
    """Signals with positions outside df_index should be silently ignored."""
    from cyqnt_trd.compat import Signal, signals_to_series
    sig = Signal("x", 100.0, "NEUTRAL", metadata={"bar_index": 100})
    idx = pd.RangeIndex(5)
    series = signals_to_series([sig], idx)
    assert not series.any()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def test_verdict_string_constants():
    from cyqnt_trd.compat import Verdict
    assert Verdict.STRONG_CANDIDATE == "STRONG_CANDIDATE"
    assert Verdict.CANDIDATE == "CANDIDATE"
    assert Verdict.WATCHLIST == "WATCHLIST"
    assert Verdict.SKIP == "SKIP"
    assert Verdict.AVOID == "AVOID"


def test_verdict_rank_ordering():
    from cyqnt_trd.compat import Verdict
    # Stronger verdicts must have lower rank numbers
    assert Verdict.RANK["STRONG_CANDIDATE"] < Verdict.RANK["CANDIDATE"]
    assert Verdict.RANK["CANDIDATE"] < Verdict.RANK["WATCHLIST"]
    assert Verdict.RANK["WATCHLIST"] < Verdict.RANK["SKIP"]
    assert Verdict.RANK["SKIP"] < Verdict.RANK["AVOID"]


def test_verdict_all_present():
    from cyqnt_trd.compat import Verdict
    expected = {"STRONG_CANDIDATE", "CANDIDATE", "WATCHLIST", "SKIP", "AVOID"}
    assert expected == set(Verdict.ALL)
    assert expected == set(Verdict.RANK.keys())


def test_verdict_is_valid():
    from cyqnt_trd.compat import Verdict
    assert Verdict.is_valid("CANDIDATE")
    assert not Verdict.is_valid("INVALID_VERDICT")


def test_verdict_usable_as_string():
    """Verdict works in plain string comparisons (atomic-style usage)."""
    from cyqnt_trd.compat import Verdict
    verdict = Verdict.CANDIDATE
    assert verdict == "CANDIDATE"
    assert "CANDIDATE" in Verdict.RANK


# ---------------------------------------------------------------------------
# TradePlan dict serialization
# ---------------------------------------------------------------------------

def _make_trade_plan():
    from cyqnt_trd.compat import TradePlan, Verdict
    return TradePlan(
        symbol="BTCUSDT",
        direction="LONG",
        verdict=Verdict.CANDIDATE,
        score=0.72,
        entry_price=40_000.0,
        stop_price=38_500.0,
        stop_pct=0.0375,
        leverage=5.0,
        notional=500.0,
        max_loss=18.75,
        reasons=["rsi_ok", "macd_ok"],
    )


def test_trade_plan_to_dict():
    from cyqnt_trd.compat import trade_plan_to_dict
    plan = _make_trade_plan()
    d = trade_plan_to_dict(plan)
    assert d["symbol"] == "BTCUSDT"
    assert d["verdict"] == "CANDIDATE"
    assert d["score"] == pytest.approx(0.72)
    assert d["reasons"] == ["rsi_ok", "macd_ok"]


def test_trade_plan_dict_round_trip():
    from cyqnt_trd.compat import trade_plan_to_dict, dict_to_trade_plan
    plan = _make_trade_plan()
    plan2 = dict_to_trade_plan(trade_plan_to_dict(plan))
    assert plan2.symbol == plan.symbol
    assert plan2.direction == plan.direction
    assert plan2.verdict == plan.verdict
    assert plan2.score == pytest.approx(plan.score)
    assert plan2.entry_price == pytest.approx(plan.entry_price)
    assert plan2.stop_price == pytest.approx(plan.stop_price)
    assert plan2.leverage == pytest.approx(plan.leverage)
    assert plan2.reasons == plan.reasons


def test_trade_plan_dict_is_json_serializable():
    import json
    from cyqnt_trd.compat import trade_plan_to_dict
    d = trade_plan_to_dict(_make_trade_plan())
    json_str = json.dumps(d)   # must not raise
    assert "BTCUSDT" in json_str


# ---------------------------------------------------------------------------
# Signal to_dict / from_dict
# ---------------------------------------------------------------------------

def test_signal_round_trip():
    from cyqnt_trd.compat import Signal
    s = Signal(name="rsi_over", value=28.5, direction="BULLISH", strength=0.9,
               metadata={"period": 14})
    d = s.to_dict()
    s2 = Signal.from_dict(d)
    assert s2.name == "rsi_over"
    assert s2.strength == pytest.approx(0.9)
    assert s2.metadata["period"] == 14


# ---------------------------------------------------------------------------
# Score to_dict / from_dict
# ---------------------------------------------------------------------------

def test_score_round_trip():
    from cyqnt_trd.compat import Score
    sc = Score(value=0.65, reasons=["a", "b"], factors={"rsi": 0.3, "macd": 0.35})
    d = sc.to_dict()
    sc2 = Score.from_dict(d)
    assert sc2.value == pytest.approx(0.65)
    assert sc2.reasons == ["a", "b"]
    assert sc2.factors["macd"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# OrderResult to_dict / from_dict
# ---------------------------------------------------------------------------

def test_order_result_round_trip():
    from cyqnt_trd.compat import OrderResult
    o = OrderResult(success=True, order_id="123", symbol="BTCUSDT",
                    side="BUY", order_type="MARKET", quantity=0.01,
                    price=40_000.0, status="FILLED", dry_run=False,
                    raw={"cliOrdId": "abc"})
    d = o.to_dict()
    o2 = OrderResult.from_dict(d)
    assert o2.success is True
    assert o2.order_id == "123"
    assert o2.quantity == pytest.approx(0.01)
    assert o2.raw["cliOrdId"] == "abc"


# ---------------------------------------------------------------------------
# extract_closes helper
# ---------------------------------------------------------------------------

def test_extract_closes_from_candles():
    from cyqnt_trd.compat import Candle, extract_closes
    candles = [
        Candle(1, 10.0, 11.0, 9.0, 10.5, 100.0),
        Candle(2, 10.5, 12.0, 10.0, 11.5, 200.0),
    ]
    closes = extract_closes(candles)
    assert closes == pytest.approx([10.5, 11.5])


def test_extract_closes_from_floats():
    from cyqnt_trd.compat import extract_closes
    floats = [1.0, 2.0, 3.0]
    assert extract_closes(floats) is floats


# ---------------------------------------------------------------------------
# Position / Balance field integrity
# ---------------------------------------------------------------------------

def test_position_fields():
    from cyqnt_trd.compat import Position
    p = Position(
        symbol="BTCUSDT", direction="LONG", entry_price=40_000.0,
        quantity=0.1, unrealized_pnl=50.0, leverage=10,
        margin_type="ISOLATED",
    )
    assert p.notional == 0.0      # default
    assert p.mark_price == 0.0   # default
    assert p.direction == "LONG"


def test_balance_fields():
    from cyqnt_trd.compat import Balance
    b = Balance(asset="USDT", free=1000.0, locked=0.0, total=1000.0)
    assert b.total == pytest.approx(1000.0)
