"""``to_panel`` — the bundle shape ``cyqnt_trd.blocks`` can actually consume.

The invariants worth testing are the ones a caller cannot check for themselves:

* long-form metrics become one column per metric, so nobody ever passes the
  interleaved ``value`` column to a block (that does not raise — it silently
  computes across two different metrics);
* a reading is on a bar **only if it was knowable when that bar closed**;
* the final bar is cut at ``decision_time``, because that row is the live
  decision and a snapshot source is stamped after the last close;
* two nodes emitting the same metric name stay distinguishable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.data.panel import PanelError, to_panel

HOUR = 3_600_000
T0 = 1_780_000_000_000


def _bar(close_ms: int, close: float, symbol: str = "BTCUSDT") -> dict:
    return {"instrument_id": symbol, "timeframe": "1h",
            "open_time": close_ms - HOUR, "close_time": close_ms,
            "available_time": close_ms, "event_time": close_ms,
            "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 10.0, "quote_volume": 1000.0}


def _metric(at_ms: int, name: str, value: float, symbol: str = "BTCUSDT") -> dict:
    return {"instrument_id": symbol, "metric": name, "value": value,
            "event_time": at_ms, "available_time": at_ms}


def _bundle(frames: dict, *, decision_time: int) -> dict:
    return {"schema": "cyqnt.input/v1", "decision_time": decision_time,
            "frames": frames, "source_status": {}, "warnings": []}


def _bars_frame(rows):
    return {"shape": "BarFrame@1.0", "status": "ok", "rows": rows}


def _metrics_frame(rows, availability="BACKTESTABLE"):
    return {"shape": "MetricFrame@1.0", "status": "ok",
            "availability": availability, "rows": rows}


# --------------------------------------------------------------------------- #


def test_ohlcv_columns_are_what_blocks_expect():
    from cyqnt_trd.blocks import indicators as I

    bars = [_bar(T0 + i * HOUR, 100.0 + i) for i in range(40)]
    panel = to_panel(_bundle({"klines": _bars_frame(bars)}, decision_time=T0 + 39 * HOUR))

    assert list(panel.columns[:6]) == ["open", "high", "low", "close",
                                       "volume", "quote_volume"]
    assert isinstance(panel.index, pd.DatetimeIndex)
    # the actual contract: a block takes it unchanged
    assert I.ema(panel["close"], 10).notna().any()
    assert I.atr(panel, 14).notna().any()


def test_long_form_metrics_become_one_column_each():
    """The whole point: never hand a block the interleaved ``value`` column."""
    bars = [_bar(T0 + i * HOUR, 100.0 + i) for i in range(3)]
    rows = []
    for i in range(3):
        at = T0 + i * HOUR
        rows.append(_metric(at, "oi_base", 1000.0 + i))
        rows.append(_metric(at, "oi_value", 5_000_000.0 + i))
    panel = to_panel(_bundle({"klines": _bars_frame(bars),
                              "open_interest": _metrics_frame(rows)},
                             decision_time=T0 + 2 * HOUR))

    assert "oi_base" in panel and "oi_value" in panel
    assert list(panel["oi_base"]) == [1000.0, 1001.0, 1002.0]
    assert list(panel["oi_value"]) == [5_000_000.0, 5_000_001.0, 5_000_002.0]
    assert "value" not in panel.columns, "the interleaved column must never survive"


def test_a_reading_stamped_after_a_bar_close_is_not_on_that_bar():
    """Lookahead is the failure this function exists to make impossible."""
    bars = [_bar(T0 + i * HOUR, 100.0 + i) for i in range(3)]
    # knowable one millisecond AFTER bar 1 closed -> may only appear from bar 2
    late = T0 + 1 * HOUR + 1
    panel = to_panel(_bundle({"klines": _bars_frame(bars),
                              "funding": _metrics_frame([_metric(late, "rate", 0.001)])},
                             decision_time=T0 + 2 * HOUR))

    assert pd.isna(panel["rate"].iloc[0]), "bar 0 could not know it"
    assert pd.isna(panel["rate"].iloc[1]), "bar 1 closed before it was knowable"
    assert panel["rate"].iloc[2] == 0.001


def test_earlier_bars_are_not_back_filled_from_the_future():
    bars = [_bar(T0 + i * HOUR, 100.0 + i) for i in range(4)]
    panel = to_panel(_bundle(
        {"klines": _bars_frame(bars),
         "funding": _metrics_frame([_metric(T0 + 2 * HOUR, "rate", 0.002)])},
        decision_time=T0 + 3 * HOUR))
    assert list(panel["rate"].isna()) == [True, True, False, False]
    assert panel["rate"].iloc[3] == 0.002, "carried forward, which is legitimate"


def test_the_final_bar_is_cut_at_the_decision_time():
    """A snapshot is stamped after the last close; the live row must still see it."""
    bars = [_bar(T0 + i * HOUR, 100.0 + i) for i in range(3)]
    last_close = T0 + 2 * HOUR
    snapshot_at = last_close + 90_000            # 90s after the bar closed
    panel = to_panel(_bundle(
        {"klines": _bars_frame(bars),
         "ticker_24h": _metrics_frame([_metric(snapshot_at, "change_pct", 1.25)],
                                      availability="FORWARD_ONLY")},
        decision_time=snapshot_at + 1_000))

    assert panel["change_pct"].notna().sum() == 1, "only the decision row may see it"
    assert panel["change_pct"].iloc[-1] == 1.25
    assert panel["change_pct"].iloc[:-1].isna().all()


def test_a_snapshot_after_the_decision_time_is_not_used_at_all():
    bars = [_bar(T0 + i * HOUR, 100.0 + i) for i in range(3)]
    last_close = T0 + 2 * HOUR
    panel = to_panel(_bundle(
        {"klines": _bars_frame(bars),
         "ticker_24h": _metrics_frame([_metric(last_close + 10 * HOUR, "change_pct", 9.0)])},
        decision_time=last_close + 60_000))
    assert panel["change_pct"].isna().all()


def test_two_nodes_with_the_same_metric_name_stay_distinguishable():
    """``long_short_ratio`` means different populations in two different nodes."""
    bars = [_bar(T0 + i * HOUR, 100.0 + i) for i in range(2)]
    all_accounts = [_metric(T0 + i * HOUR, "long_short_ratio", 1.3) for i in range(2)]
    top_traders = [_metric(T0 + i * HOUR, "long_short_ratio", 1.9) for i in range(2)]
    panel = to_panel(_bundle(
        {"klines": _bars_frame(bars),
         "long_short_ratio": _metrics_frame(all_accounts),
         "top_trader_ratio": _metrics_frame(top_traders)},
        decision_time=T0 + HOUR))

    assert panel["long_short_ratio"].iloc[-1] == 1.3
    assert panel["top_trader_ratio.long_short_ratio"].iloc[-1] == 1.9


def test_market_wide_metrics_reach_every_instrument():
    """Fear & greed has no instrument — dropping it would lose the whole source."""
    bars = [_bar(T0 + i * HOUR, 100.0 + i) for i in range(2)]
    rows = [{"instrument_id": "MARKET", "metric": "fear_greed", "value": 28.0,
             "event_time": T0, "available_time": T0}]
    panel = to_panel(_bundle({"klines": _bars_frame(bars),
                              "fear_greed": _metrics_frame(rows)},
                             decision_time=T0 + HOUR))
    assert panel["fear_greed"].iloc[-1] == 28.0


def test_events_are_counted_per_window_and_not_carried_forever():
    bars = [_bar(T0 + i * HOUR, 100.0 + i) for i in range(5)]
    events = [{"event_id": "e%d" % i, "source_id": "square", "topic": "news",
               "instrument_id": "BTCUSDT",
               "event_time": T0 + HOUR // 2, "available_time": T0 + HOUR // 2}
              for i in range(3)]
    panel = to_panel(_bundle({"klines": _bars_frame(bars),
                              "news": {"shape": "EventFrame@1.0", "status": "ok",
                                       "rows": events}},
                             decision_time=T0 + 4 * HOUR),
                     event_windows={"1h": HOUR, "24h": 24 * HOUR})

    assert panel["news_count_1h"].iloc[0] == 0, "bar 0 closed before they arrived"
    assert panel["news_count_1h"].iloc[1] == 3
    assert panel["news_count_1h"].iloc[3] == 0, "the 1h window has moved past them"
    assert panel["news_count_24h"].iloc[3] == 3, "still inside 24h"


def test_multi_instrument_panel_is_a_symbol_field_multiindex():
    bars = ([_bar(T0 + i * HOUR, 100.0 + i, "BTCUSDT") for i in range(3)]
            + [_bar(T0 + i * HOUR, 50.0 + i, "ETHUSDT") for i in range(3)])
    rows = [_metric(T0 + i * HOUR, "rate", 0.001 * (i + 1), "BTCUSDT") for i in range(3)]
    bundle = _bundle({"klines": _bars_frame(bars), "funding": _metrics_frame(rows)},
                     decision_time=T0 + 2 * HOUR)

    panel = to_panel(bundle, symbols=["BTCUSDT", "ETHUSDT"])
    assert isinstance(panel.columns, pd.MultiIndex)
    assert panel[("BTCUSDT", "close")].iloc[-1] == 102.0
    assert panel[("ETHUSDT", "close")].iloc[-1] == 52.0
    # a per-instrument metric must not leak across symbols
    assert panel[("BTCUSDT", "rate")].iloc[-1] == 0.003
    assert ("ETHUSDT", "rate") not in panel.columns or panel[("ETHUSDT", "rate")].isna().all()
    # slicing back gives the single-symbol frame blocks consume
    from cyqnt_trd.blocks import indicators as I
    assert I.ema(panel["BTCUSDT"]["close"], 2).notna().any()


def test_an_ambiguous_bundle_refuses_to_guess():
    bars = ([_bar(T0, 100.0, "BTCUSDT")] + [_bar(T0, 50.0, "ETHUSDT")])
    with pytest.raises(PanelError) as excinfo:
        to_panel(_bundle({"klines": _bars_frame(bars)}, decision_time=T0))
    assert "symbol=" in str(excinfo.value)


def test_a_bundle_with_no_bars_says_so():
    with pytest.raises(PanelError) as excinfo:
        to_panel(_bundle({"funding": _metrics_frame([_metric(T0, "rate", 0.1)])},
                         decision_time=T0))
    assert "no BarFrame" in str(excinfo.value)
