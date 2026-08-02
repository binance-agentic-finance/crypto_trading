# =============================================================================
# klines 節點不該宣告一個回應裡不存在的欄位
# =============================================================================
# BarFrame@1.0 要求每一列有 instrument_id。各端點怎麼補上這一欄不一樣,節點用兩個
# 機制宣告:
#
#   column_map  「回應裡**有**這欄,只是名字不同」→ 改名
#   constants   「回應裡**沒有**這欄,但我從請求就知道」→ 直接填
#
# ticker24hr 每一列自帶 symbol,所以 universe 節點用 column_map 是對的。
# klines 的回應是一串**純陣列**,連欄位名都沒有,更沒有 symbol —— 標的在請求的
# query string 裡,幣安不會在回應裡再說一次。所以 klines 只能用 constants。
#
# 這個檔案存在的原因:klines 節點上原本有一份從 universe 節點複製過來的
# column_map={"symbol": "instrument_id"}。它永遠不會觸發,所以資料一直是對的 ——
# 但它讓 input_contract 的「宣告的來源欄不在回應裡」檢查**每一次擷取都為真**,
# 於是每份 bundle 的 warnings 裡都有一條永遠不需要處理的訊息,並且原封不動流進
# 每一份 signal 的 warnings。而那個欄位同時也是 declared assumption 出去的地方,
# 一條永遠都在的警告,教會讀者跳過整個欄位。
#
# 那個「宣告的欄位不存在就警告」的檢查本身是好的,不要因為這件事把它拿掉:它防的是
# 幣安哪天把 symbol 改名、rename 靜默失效、instrument_id 變成整欄 NaN —— 那不會拋
# 例外,只會讓下游 join 對不上,而 join 對不上的結果是一個比較小但每檔都合理的籃子。
# =============================================================================
from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.data import catalog


def _spec(name: str):
    node = catalog.get_node(name)
    return getattr(node, "source", None) or getattr(node, "rest", None) or node


def test_klines_declares_no_rename():
    """The response has no named fields at all, so there is nothing to rename."""
    assert not getattr(_spec("klines"), "column_map", None)


def test_klines_still_gets_its_instrument_id_from_constants():
    """Removing the rename must not remove the column it never actually supplied."""
    constants = getattr(_spec("klines"), "constants", None) or {}
    assert "instrument_id" in constants
    assert "timeframe" in constants


def test_the_node_that_really_does_rename_still_does():
    """ticker24hr carries ``symbol`` per row, so ``universe`` keeps its map.

    Asserted alongside, because the point is not "column_map is bad" — it is
    "declare it only where the response actually has the column".
    """
    assert getattr(_spec("universe"), "column_map", None) == {"symbol": "instrument_id"}


@pytest.mark.parametrize("name", sorted(catalog.list_node_names()))
def test_no_node_declares_a_rename_it_also_pins_as_a_constant(name):
    """The general form of the bug, so the next copy-paste is caught too.

    A node that both renames ``X -> Y`` and pins ``Y`` as a constant is saying
    two incompatible things about where ``Y`` comes from. Whichever is right, the
    other one is either dead or shadowed — and the dead one is what emits a
    permanent warning.
    """
    spec = _spec(name)
    column_map = getattr(spec, "column_map", None) or {}
    constants = getattr(spec, "constants", None) or {}
    overlap = sorted(set(column_map.values()) & set(constants))
    assert not overlap, (
        "%s declares both a rename onto %s and a constant for it; the rename "
        "runs first, so the constant only applies when the rename found nothing "
        "— i.e. exactly when the rename should not have been declared" % (name, overlap))
