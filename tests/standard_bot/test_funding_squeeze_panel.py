"""``funding_squeeze_panel`` — 一支要四個資料源對在同一條時間軸上的 block 策略。

它同時是那條輸入鏈路的活體證明:``build_live_bundle()`` → ``to_panel()`` →
``make_signals(df)``。repo 裡沒有別的策略走這條,所以壞掉不會有人發現。

介面就是既有的那個:``make_signals(df) -> (long, short)``,兩條對齊 ``df``
的布林 Series,檔尾 ``strategy.register(...)``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cyqnt_trd.blocks import strategy as blocks_strategy
from cyqnt_trd.strategies.funding_squeeze_panel import BOT_ID, CONFIG, make_signals

HOUR = 3_600_000
T0 = 1_785_000_000_000
N = 60


def _df(*, drift: float, funding_ratio=None, oi_slope=None,
        taker_buy=None, taker_sell=None, rows: int = N) -> pd.DataFrame:
    """OHLCV,選配掛上衍生品欄位。

    ``drift`` / ``oi_slope`` 是整段的總變化;策略看的是 12 根的窗口,所以
    要夠陡才會過門檻。``funding_ratio`` 是**原始比率**(0.0001 == 1 bp),
    因為 panel 帶的就是原始比率。
    """
    close = pd.Series(np.linspace(60_000.0, 60_000.0 * (1.0 + drift), rows))
    frame = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": 1_000.0,
        "close_time": [T0 + i * HOUR for i in range(rows)],
    })
    if funding_ratio is not None:
        frame["rate"] = funding_ratio
    if oi_slope is not None:
        frame["oi_value"] = np.linspace(1e9, 1e9 * (1.0 + oi_slope), rows)
    if taker_buy is not None:
        frame["buy_vol"], frame["sell_vol"] = taker_buy, taker_sell
    return frame


def _crowded_short(**overrides):
    """價格在跌、空方付錢、OI 增加、主動賣壓。"""
    base = dict(drift=-0.30, funding_ratio=-0.0006, oi_slope=0.40,
                taker_buy=400.0, taker_sell=1_000.0)
    base.update(overrides)
    return _df(**base)


# --------------------------------------------------------------------------- #
# 介面本身                                                                      #
# --------------------------------------------------------------------------- #


def test_it_is_a_block_strategy_registered_under_the_standard_interface():
    assert blocks_strategy.is_known_block_strategy(BOT_ID)
    plugin = blocks_strategy._KNOWN_BLOCK_PLUGINS[BOT_ID]
    assert plugin.exit_cfg["type"] == "atr_stop_tp"
    assert plugin.size == 0.1


def test_make_signals_returns_two_aligned_boolean_series():
    df = _crowded_short()
    long, short = make_signals(df)
    for name, series in (("long", long), ("short", short)):
        assert isinstance(series, pd.Series), name
        assert series.dtype == bool, name
        assert series.index.equals(df.index), name
    assert not (long & short).any(), "多空不能同一根同時成立"


# --------------------------------------------------------------------------- #
# 論點                                                                          #
# --------------------------------------------------------------------------- #


def test_a_crowded_short_goes_long():
    long, short = make_signals(_crowded_short())
    assert long.any(), "四個條件同時成立時必須進場"
    assert not short.any()


def test_a_crowded_long_goes_short():
    long, short = make_signals(_df(drift=0.30, funding_ratio=0.0006, oi_slope=0.40,
                                   taker_buy=1_000.0, taker_sell=400.0))
    assert short.any()
    assert not long.any()


def test_funding_is_read_as_a_ratio_not_as_bps():
    """會讓這支策略在幾乎任何費率下都進場的單位陷阱。

    ``derivatives.funding_rate_state`` 自己會 ×10,000:它吃的是**原始比率**,
    只有門檻是 bps。傳已經換算成 bps 的值會乘兩次 —— 0.0001 變成 10,000 bps,
    於是任何微小的正費率都被判成極端逼倉,資金費率這一腿就完全失去鑑別力。
    """
    ordinary = 0.0001                      # 1 bp,離 3 bp 門檻還很遠
    long, short = make_signals(_df(drift=0.30, funding_ratio=ordinary, oi_slope=0.40,
                                   taker_buy=1_000.0, taker_sell=400.0))
    assert not short.any(), "1 bp 不是逼倉;會進場代表費率被換算了兩次"


def test_funding_pointing_the_other_way_vetoes_the_trade():
    """價格說該做多逼空,費率卻說多方才是擁擠的那邊。"""
    long, _ = make_signals(_crowded_short(funding_ratio=0.0006))
    assert not long.any()


def test_unwinding_open_interest_vetoes_the_trade():
    """已經在離場的人群不是可以反做的人群。"""
    long, _ = make_signals(_crowded_short(oi_slope=-0.40))
    assert not long.any()


def test_passive_taker_flow_vetoes_the_trade():
    long, _ = make_signals(_crowded_short(taker_buy=1_000.0, taker_sell=1_000.0))
    assert not long.any(), "買賣量相當不算主動賣壓"


def test_a_flat_market_produces_nothing():
    long, short = make_signals(_crowded_short(drift=0.0))
    assert not long.any() and not short.any()


# --------------------------------------------------------------------------- #
# 降級                                                                          #
# --------------------------------------------------------------------------- #


def test_it_still_runs_on_a_plain_ohlcv_frame():
    """沒有衍生品欄位時只用價格動能,不會崩、也不會把缺值當成 0。"""
    long, short = make_signals(_df(drift=-0.30))
    assert isinstance(long, pd.Series) and long.dtype == bool
    assert long.any(), "只剩價格條件時應該仍會進場"


def test_a_missing_source_does_not_silently_become_zero():
    """缺一個讀數 ≠ 讀到 0。缺 taker 欄位時該條件不參與否決,
    而不是被當成「沒有主動賣壓」把訊號殺光。"""
    full = _crowded_short()
    without_taker = full.drop(columns=["buy_vol", "sell_vol"])
    long_full, _ = make_signals(full)
    long_partial, _ = make_signals(without_taker)
    assert long_full.any() and long_partial.any()


def test_the_registered_plugin_declares_it_needs_derivatives():
    plugin = blocks_strategy._KNOWN_BLOCK_PLUGINS[BOT_ID]
    assert plugin.required_inputs()["derivatives"] is True


# --------------------------------------------------------------------------- #
# 與 to_panel 的接線                                                            #
# --------------------------------------------------------------------------- #


def test_a_panel_from_the_input_bundle_feeds_make_signals_unchanged():
    """to_panel() 產出的欄位名就是這支策略讀的那些 —— 中間不需要轉接。"""
    import json
    import os

    from cyqnt_trd.standard_bot.data import to_panel

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo, "docs/standard_bot_io/samples/live_bundle_example.json")
    if not os.path.exists(path):                     # 樣本可選
        return
    with open(path, encoding="utf-8") as handle:
        panel = to_panel(json.load(handle))

    long, short = make_signals(panel)
    assert long.index.equals(panel.index) and short.index.equals(panel.index)
    assert long.dtype == bool and short.dtype == bool
    # 策略讀的四個欄位,panel 都給得出來
    assert {"close", "rate", "oi_value", "buy_vol", "sell_vol"} <= set(panel.columns)


# --------------------------------------------------------------------------- #
# 單位守門(在 blocks 層,對所有呼叫端生效)                                       #
# --------------------------------------------------------------------------- #


def test_funding_rate_state_refuses_an_already_converted_input():
    """把換算過的 bps 再傳進去必須當場失敗,而不是安靜地給錯分類。

    這個錯誤沒有型別可以擋 —— 簽名是 Series 進、Series 出。它唯一的症狀是
    分類器失去鑑別力:0.0001 被乘成 10,000 bps,任何微小正費率都變極端逼倉。
    所以守門用數值範圍:交易所把資金費率封在 ±0.75%,超過 1.0 的值在任何行情
    下都不可能是比率。
    """
    import pytest as _pytest

    from cyqnt_trd.blocks import derivatives as deriv

    ratio = pd.Series([0.0001, 0.0006, -0.0006])
    assert list(deriv.funding_rate_state(ratio, 3.0, -3.0)) == [
        "neutral", "bullish_squeeze", "bearish_squeeze"]

    with _pytest.raises(ValueError, match="RAW RATIO"):
        deriv.funding_rate_state(ratio * 1e4, 3.0, -3.0)


def test_the_guard_does_not_fire_on_a_real_extreme_funding_rate():
    """守門不能誤傷真實行情:±0.75% 是交易所上下限,必須照常分類。"""
    from cyqnt_trd.blocks import derivatives as deriv

    assert list(deriv.funding_rate_state(pd.Series([0.0075, -0.0075]), 3.0, -3.0)) == [
        "bullish_squeeze", "bearish_squeeze"]


def test_oi_change_can_be_asked_for_in_percent():
    """名字寫 _pct 但預設回小數;門檻用百分比寫時必須能明確要求同一單位。"""
    from cyqnt_trd.blocks import derivatives as deriv

    oi = pd.Series([1e9, 1.05e9])
    assert round(float(deriv.oi_change_pct(oi).iloc[-1]), 6) == 0.05
    assert round(float(deriv.oi_change_pct(oi, as_percent=True).iloc[-1]), 6) == 5.0


def test_the_panel_carries_the_declared_unit_of_every_metric():
    """單位在 catalog 就宣告了,不該在 pivot 時被丟掉 —— 沒有單位的裸欄位
    正是雙重換算看不出來的原因。"""
    import json
    import os

    from cyqnt_trd.standard_bot.data import to_panel

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo, "docs/standard_bot_io/samples/live_bundle_example.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        panel = to_panel(json.load(handle))

    units = panel.attrs.get("units") or {}
    assert units.get("rate") == "ratio", (
        "funding 的單位是 catalog 宣告的,策略要能據此斷言而不是靠猜:%s" % units)
    assert units.get("oi_value") == "notional"
