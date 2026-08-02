# =============================================================================
# min_bars_multiple 只對「純滾動窗口」指標成立 —— 而那份名單是量出來的
# =============================================================================
# augment_with_indicator 預設要求 period × 3 根 K 線,理由是 RMA / EMA 這一族
# 窗口一滿就給數字,但那個數字還帶著遞迴的種子 —— 不會是 NaN,所以只有長度檢查
# 抓得到。純滾動窗口(highest / lowest / donchian / range_gain_pct)沒有種子,
# period 根就是精確答案,對它們要求 3 倍會把「上市不到九個月」的幣全部擋掉,
# 而那正是三個月漲幅螢幕要找的東西。所以 min_bars_multiple 是可寫的。
#
# 問題是「哪些指標屬於那一族」原本只寫在註解裡,沒有任何程式檢查。實測後果:
# 11 根 K 線的 capture + 三個 supertrend 步驟 + min_bars_multiple: 1 →
# 順利跑完並回報「五檔幣在 4h / 1h / 15m 全部偏空」,零警告;同一份 spec 在完整
# 歷史上的正確答案是 0 檔。而不寫這個旋鈕時觸發的那條錯誤訊息,**自己就在建議
# 設這個旋鈕**。
#
# 這支測試把那份名單從註解變成可驗證的量測:每個指標各跑 400 根與 15 根
# (= min_bars_multiple: 1 允許的長度),末值相同才算純滾動。名單漂掉就紅。
# =============================================================================
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from cyqnt_trd.blocks import indicators as I
from cyqnt_trd.blocks.universe import (
    _INDICATOR_WARMUP_MULTIPLE,
    _WARMUP_FREE_INDICATORS,
)

PERIOD = 14
LONG = 400
#: What ``min_bars_multiple: 1`` permits: ``1 * period + window_bars - 1``.
SHORT = PERIOD + 1
SEEDS = (7, 11, 23, 101, 202)


def _frames(seed: int):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, LONG)))
    df = pd.DataFrame({
        "open": close.shift(1).bfill(),
        "close": close,
        "high": close + abs(rng.normal(0, 0.5, LONG)),
        "low": close - abs(rng.normal(0, 0.5, LONG)),
        "volume": abs(rng.normal(1e6, 1e5, LONG)),
    })
    return close, df


def _indicators():
    for name in sorted(n for n in dir(I) if not n.startswith("_")):
        fn = getattr(I, name)
        if callable(fn) and getattr(fn, "__module__", "") == I.__name__:
            yield name, fn


def _last(value):
    value = value[0] if isinstance(value, tuple) else value
    return float(pd.Series(value).iloc[-1])


def _measure(seed: int) -> tuple[set, set]:
    """(warm-up free, undecidable) for one random series."""
    close, df = _frames(seed)
    free, undecidable = set(), set()
    for name, fn in _indicators():
        signature = inspect.signature(fn)
        params = list(signature.parameters)
        source = df if params[0] in ("df", "bars", "ohlc") else close
        kwargs = {p: PERIOD for p in params[1:]
                  if "period" in p or p in ("length", "window", "n")}
        try:
            full, cut = _last(fn(source, **kwargs)), _last(fn(source.iloc[-SHORT:], **kwargs))
        except Exception:                       # noqa: BLE001 - undecidable, not a failure
            undecidable.add(name)
            continue
        if np.isnan(full) or np.isnan(cut):
            undecidable.add(name)
        elif abs(full - cut) <= 1e-9 * max(1.0, abs(full)):
            free.add(name)
    return free, undecidable


def test_the_warmup_free_set_is_what_measurement_says_it_is():
    """The set in universe.py must equal the measured one, exactly.

    Not a subset check in either direction. A name that drops out silently
    starts refusing a screen that used to work; a name that creeps in silently
    starts accepting a wrong answer.
    """
    measured = [_measure(seed) for seed in SEEDS]
    free = set.intersection(*[m[0] for m in measured])

    assert _WARMUP_FREE_INDICATORS == free, (
        "only in the code: %s / only in the measurement: %s"
        % (sorted(_WARMUP_FREE_INDICATORS - free), sorted(free - _WARMUP_FREE_INDICATORS)))


def test_the_indicator_that_caused_this_is_not_in_the_set():
    """Named, not just covered by the set comparison above.

    ``supertrend`` is what the demo used, and its Wilder-smoothed ATR is the
    exact mechanism: at 15 bars the seed is still most of the value.
    """
    assert "supertrend" not in _WARMUP_FREE_INDICATORS
    assert "range_gain_pct" in _WARMUP_FREE_INDICATORS, (
        "the docstring names this one as the reason the knob exists at all")


def test_an_undecidable_indicator_is_left_out_rather_than_guessed():
    """Being wrongly INSIDE the set is a silent wrong answer; being wrongly
    outside it is a refusal that says what to do. So the tie goes to refusing.
    """
    undecidable = set.union(*[m[1] for m in (_measure(seed) for seed in SEEDS)])
    assert not (undecidable & _WARMUP_FREE_INDICATORS), sorted(
        undecidable & _WARMUP_FREE_INDICATORS)


def test_a_seeded_indicator_refuses_the_knob_and_says_why():
    from cyqnt_trd.blocks import universe as ub
    import pytest

    universe = pd.DataFrame({"symbol": ["BTCUSDT"], "quoteVolume": [1e9]})
    with pytest.raises(ValueError, match="only sound for a pure rolling-window"):
        ub.augment_with_indicator(universe, pd.DataFrame(), indicator="supertrend",
                                  timeframe="4h", as_="x", min_bars_multiple=1)


def test_a_rolling_indicator_still_accepts_it():
    """The mutation side. A check that refused both would pass the test above
    while making the three-month screen impossible again."""
    from cyqnt_trd.blocks import universe as ub

    universe = pd.DataFrame({"symbol": ["BTCUSDT"], "quoteVolume": [1e9]})
    try:
        ub.augment_with_indicator(universe, pd.DataFrame(), indicator="range_gain_pct",
                                  timeframe="1d", as_="x", min_bars_multiple=1)
    except ValueError as exc:
        assert "only sound for a pure rolling-window" not in str(exc), exc
    assert _INDICATOR_WARMUP_MULTIPLE == 3
