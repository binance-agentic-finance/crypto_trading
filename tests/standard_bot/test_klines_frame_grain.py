# =============================================================================
# frames.klines 只能裝一條序列 —— 多時框會被靜默併成一條
# =============================================================================
# load_input_bundle 把 klines 收進 MarketBundle 時,key 取自**第一根 bar** 的
# (instrument, timeframe)。所以只要 frame 裡不只一組,全部的 bar 都會被歸到
# 排在最前面的那一組底下,變成「好幾條序列接成一條」,而接縫處時間軸會往回跳。
#
# FRAME_SHAPES 的註解早就寫明這個危害 —— 但只寫了「多 symbol」那一半,
# universe_bars 這個 key 就是為此而生。**多時框是同一行造成的同一個危害**,
# 而它沒有任何防線:99 根 1h + 99 根 4h 會被讀成一條 198 根的「1h」序列,
# 時間軸前進 98 次、往回跳 16.5 天、再前進,而 primary 時框上的每一個指標都是
# 在那上面算的,零錯誤零警告。
#
# 這不是理論:查證時有人想「自己把 4h K 線塞進 klines frame」來湊多時框,
# 拿到的就是這個。
# =============================================================================
from __future__ import annotations

import json

import pytest

from cyqnt_trd.standard_bot.data.input_bundle import load_input_bundle


def _bar(symbol: str, timeframe: str, close_time: int) -> dict:
    return {
        "instrument_id": symbol, "timeframe": timeframe,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 1000.0, "close_time": close_time,
        "open_time": close_time - 1, "available_time": close_time,
    }


def _bundle(rows: list[dict]) -> dict:
    return {
        "schema": "cyqnt.input/v1",
        "snapshot_id": "00000000-0000-5000-8000-000000000000",
        "decision_time": 1_785_000_000_000,
        "frames": {"klines": {"shape": "BarFrame@1.0", "rows": rows}},
    }


def _write(tmp_path, bundle) -> str:
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return str(path)


def test_one_instrument_one_timeframe_still_loads(tmp_path):
    """The accept side. A guard that refused this would break every offline run."""
    rows = [_bar("BTCUSDT", "1h", 1_784_000_000_000 + i * 3_600_000) for i in range(5)]
    snapshot = load_input_bundle(_write(tmp_path, _bundle(rows)))

    assert list(snapshot.market.bars) == ["BTCUSDT|1h"]
    assert len(snapshot.market.bars["BTCUSDT|1h"]) == 5


def test_two_timeframes_in_one_klines_frame_is_refused(tmp_path):
    """The case with no guard before this: same instrument, two timeframes."""
    rows = ([_bar("BTCUSDT", "1h", 1_784_000_000_000 + i * 3_600_000) for i in range(5)]
            + [_bar("BTCUSDT", "4h", 1_784_000_000_000 + i * 14_400_000) for i in range(5)])

    with pytest.raises(ValueError) as excinfo:
        load_input_bundle(_write(tmp_path, _bundle(rows)))

    message = str(excinfo.value)
    assert "1h" in message and "4h" in message, message
    assert "universe_bars" in message, (
        "an error that does not name the frame to use instead just moves the "
        "guesswork one step later")


def test_two_instruments_in_one_klines_frame_is_refused_too(tmp_path):
    """The half the comment already described. It was described, not checked."""
    rows = ([_bar("BTCUSDT", "1h", 1_784_000_000_000 + i * 3_600_000) for i in range(3)]
            + [_bar("ETHUSDT", "1h", 1_784_000_000_000 + i * 3_600_000) for i in range(3)])

    with pytest.raises(ValueError, match="universe_bars"):
        load_input_bundle(_write(tmp_path, _bundle(rows)))


def test_the_refusal_names_what_the_bars_would_have_been_filed_as(tmp_path):
    """The number that makes the bug legible: 10 bars in, one 10-bar series out
    under a name that is true of only half of them."""
    rows = ([_bar("BTCUSDT", "1h", 1_784_000_000_000 + i * 3_600_000) for i in range(5)]
            + [_bar("BTCUSDT", "4h", 1_784_000_000_000 + i * 14_400_000) for i in range(5)])

    with pytest.raises(ValueError, match=r"filed as BTCUSDT\|1h"):
        load_input_bundle(_write(tmp_path, _bundle(rows)))
