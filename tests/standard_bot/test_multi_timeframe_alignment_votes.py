# =============================================================================
# multi_timeframe_alignment 只吃是/否,不吃方向欄
# =============================================================================
# 這個 block 的名字正好就是「多時框共振」,所以它是使用者(和轉換器)最會拿來
# 表達那個需求的東西。而它原本的實作是 `ensure_series(s).fillna(False).astype(bool)`
# —— 套在**呼叫端給的資料**上。
#
# Supertrend 的方向欄是 -1.0 / +1.0,而 bool(-1.0) 是 True。所以把三個方向欄
# 餵進去,「三個時框同時偏空」就被翻成「無條件通過」:實測凍結截面上五檔全過,
# 包含一檔三個時框都**偏多**的幣,而且五檔全被標成 short,validate 完全綠燈。
#
# 這個模組裡其他每一處 .astype(bool) 都是套在比較的結果上(本來就是布林),
# 只有這裡套在外來資料上 —— 那個不對稱就是 bug 的位置。
#
# 為什麼是拒絕而不是「猜對的那個」:對 -1 而言,呼叫端可能是指「偏空,算它一票」
# 也可能是「不成立,丟掉」,兩個答案互斥,而數值本身沒有任何資訊分辨。
# 能做的只有拒絕,並說出該寫什麼。
# =============================================================================
from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.blocks import conditions as C


def test_boolean_votes_are_what_it_is_for():
    out = C.multi_timeframe_alignment(
        pd.Series([True, True, False]),
        pd.Series([True, False, False]))
    assert list(out) == [True, False, False]


def test_zero_one_still_counts_as_a_vote():
    """0/1 is unambiguous, so it is accepted — the refusal is aimed at values
    that carry a meaning the cast would throw away, not at integer booleans."""
    out = C.multi_timeframe_alignment(pd.Series([1, 1, 0]), pd.Series([1, 0, 0]))
    assert list(out) == [True, False, False]


def test_a_direction_column_is_refused_rather_than_cast():
    """The whole point. -1 is truthy, so casting turns AND into "always"."""
    direction = pd.Series([-1.0, 1.0, -1.0])

    with pytest.raises(ValueError) as excinfo:
        C.multi_timeframe_alignment(direction, direction, direction)

    message = str(excinfo.value)
    assert "-1" in message
    assert "conditions.value_below" in message, (
        "the error has to name the conversion to write, or the next person "
        "writes the same spec with a different column")


def test_the_refusal_names_which_signal_was_wrong():
    """Three steps in, "one of them is not a vote" is not actionable."""
    with pytest.raises(ValueError, match="signal 2"):
        C.multi_timeframe_alignment(
            pd.Series([True, False]), pd.Series([True, True]), pd.Series([-1.0, 1.0]))


def test_nan_is_a_no_vote_not_a_refusal():
    """A NaN is a missing reading, not an ambiguous one: it means the condition
    did not hold, which is what fillna(False) already meant. Only values that
    carry a DIFFERENT meaning are refused."""
    out = C.multi_timeframe_alignment(
        pd.Series([1.0, float("nan"), 1.0]), pd.Series([1.0, 1.0, 0.0]))
    assert list(out) == [True, False, False]


def test_the_error_travels_through_the_yaml_dry_run(tmp_path):
    """It has to fire at validate, not only at run.

    A spec that validates green and then returns the whole universe labelled
    short is the failure this exists to stop; catching it only at run time
    means the converter's retry loop never sees it either.
    """
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import validate_spec

    spec = {
        "spec_version": "1.0", "target": "standard_bot",
        "strategy": {"id": "mtf_align_probe"},
        "run": {"mode": "backtest"},
        "data": {"symbol": "BTCUSDT", "market_type": "futures",
                 "primary": {"interval": "1h"}},
        "selection": {
            "universe": [
                {"block": "universe.filter_quote_volume",
                 "params": {"min_quote_volume": 1.0e+7}},
                {"block": "universe.augment_with_indicator",
                 "with": ["universe_bars"],
                 "params": {"indicator": "supertrend", "timeframe": "4h",
                            "output": 1, "as": "st_dir_4h"}},
            ],
            "score": "quoteVolume", "top_k": 5, "dedupe_by": "base_asset",
            "short_when": {"cond": "conditions.multi_timeframe_alignment",
                           "args": ["st_dir_4h", "st_dir_4h"]},
        },
    }
    errors, _warnings = validate_spec(spec)
    assert any("yes/no votes" in error for error in errors), errors
