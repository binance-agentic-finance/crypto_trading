# =============================================================================
# 根 __init__ 必須是懶的
# =============================================================================
# 0.1.13 唯一的改動就是把根 ``cyqnt_trd/__init__.py`` 從 eager 迴圈改成 PEP 562 的
# 模組級 ``__getattr__``。但**這個行為原本沒有任何測試守著** —— 2420 條測試裡沒有一條
# 會因為它退回 eager 而變紅。
#
# 這不是理論風險,已經付出過代價:0.1.12 的 eager 根讓
#
#     import cyqnt_trd.blocks.indicators          # 只算數,只需要 numpy + pandas
#
# 連帶拉進 aiohttp / binance_common / binance_sdk_derivatives_trading_usds_futures /
# binance_sdk_spot / matplotlib / requests / scipy —— 實測 7 個。下游那個純計算的
# capability sandbox(規則:不 import 交易所 SDK / HTTP 客戶端)因此不得不自己掛一層
# 垫片繞過本檔案,理由正是「不能相信上游的根是乾淨的」。
#
# 所以這裡把契約寫成可執行的:根是懶的、乾淨葉子導入保持乾淨、而「懶」不等於「壞掉」
# (子包該導的時候還是要導得進來,失敗時的降級語義也不能變)。
#
# 為什麼要用子行程:``sys.modules`` 是行程級的。同一個 pytest 行程裡別的測試早就把
# standard_bot / get_data 那些重子包載進來了,在行程內斷言「沒有被載入」永遠是紅的,
# 而斷言「有被載入」則永遠是綠的 —— 兩個方向都測不到東西。
# =============================================================================
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]

#: 純計算路徑上不該出現的東西。前四個是交易所 SDK,接著是 HTTP 客戶端 ——
#: 下游 sandbox 的紅線就是這一批。matplotlib 放進來是因為 0.1.12 真的會拉進它
#: (畫圖函式庫進到一個只做計算的行程裡,純粹是 eager 根的附帶損害)。
SDK_AND_HTTP = (
    "binance_sdk_spot",
    "binance_common",
    "binance_sdk_algo",
    "binance_sdk_derivatives_trading_usds_futures",
    "requests",
    "aiohttp",
    "httpx",
    "websockets",
    "matplotlib",
)

#: 純計算函式庫。回測引擎用 numba 加速、用 scipy 算統計量,所以它們出現是**正常**的,
#: 不列進紅線。列在這裡是為了說明「紅線挑的是 SDK/HTTP,不是『所有重的東西』」。
COMPUTE_LIBRARIES = ("numpy", "pandas", "numba", "scipy")


def _in_clean_interpreter(body: str) -> dict:
    """在一個全新的解釋器裡跑 body,它必須 print 一行 JSON。"""
    code = textwrap.dedent("""
        import importlib, json, sys
        %s
        top_level = {name.split(".")[0] for name in sys.modules}
        cyqnt_submodules = sorted(
            name for name in sys.modules if name.startswith("cyqnt_trd."))
        print(json.dumps({
            "sdk_and_http": sorted(top_level & set(%r)),
            "compute": sorted(top_level & set(%r)),
            "cyqnt_submodules": cyqnt_submodules,
        }))
    """) % (textwrap.dedent(body).strip(), SDK_AND_HTTP, COMPUTE_LIBRARIES)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, f"子行程失敗:\n{result.stderr[-3000:]}"
    return json.loads(result.stdout.strip().splitlines()[-1])


# ── 懶:該不載入的時候真的沒載入 ──────────────────────────────────────────

def test_importing_the_root_does_not_import_any_heavy_submodule():
    """``import cyqnt_trd`` 本身必須幾乎不做事。

    eager 版本在這一步就把 get_data / trading_signal / backtesting / standard_bot /
    strategy_cases / compat 全部導進來了,於是交易所 SDK 與 HTTP 客戶端跟著進來。
    """
    report = _in_clean_interpreter("import cyqnt_trd")

    assert report["sdk_and_http"] == [], (
        "只 import 根就把交易所 SDK / HTTP 客戶端拉進來了:"
        f"{report['sdk_and_http']} —— 根 __init__ 恐怕又變回 eager 導入了")
    # 反向:確認「什麼都沒載入」不是因為連根都沒 import 成功。
    assert report["cyqnt_submodules"] == [], (
        f"根 __init__ 順手導了子包:{report['cyqnt_submodules']}")


def test_a_clean_leaf_import_stays_clean():
    """``import cyqnt_trd.blocks.<leaf>`` 只該多 numpy / pandas。

    這條就是下游純計算 sandbox 依賴的那個契約。0.1.12 上它會拉進 7 個 SDK/HTTP。
    """
    report = _in_clean_interpreter("""
        for name in ("indicators", "conditions", "patterns", "regime", "sizing"):
            importlib.import_module("cyqnt_trd.blocks." + name)
    """)

    assert report["sdk_and_http"] == [], (
        f"乾淨葉子導入洩漏了:{report['sdk_and_http']}")
    # 反向斷言:算數要用的東西確實載入了,否則上面那條是空轉 ——
    # 葉子沒真的 import 進來的話,「沒有洩漏」這句話沒有意義。
    assert "numpy" in report["compute"] and "pandas" in report["compute"]


def test_the_backtest_engine_only_pulls_compute_libraries():
    """回測引擎比葉子更容易洩漏,所以單獨守一條。

    ``standard_bot/simulation/vectorized_backtest.py`` 用的是**絕對**導入
    (``from cyqnt_trd.blocks import indicators as ind``),一定會解析到根 ——
    根只要是 eager 的,這裡就會把交易所 SDK 拖進來。numba / scipy 是它算得快的原因,
    屬於正常開銷。
    """
    report = _in_clean_interpreter(
        'importlib.import_module("cyqnt_trd.standard_bot.simulation.vectorized_backtest")')

    assert report["sdk_and_http"] == [], (
        f"載入回測引擎洩漏了:{report['sdk_and_http']}")
    assert "numba" in report["compute"] or "scipy" in report["compute"], (
        "回測引擎連 numba / scipy 都沒載入,這條大概沒真的 import 到東西")


# ── 但「懶」不等於「壞掉」 ────────────────────────────────────────────────

def test_accessing_a_submodule_actually_imports_it():
    """延後導入,不是不導入。

    只斷言「沒被載入」的話,一個永遠回 None 的 __getattr__ 也能讓上面三條全綠。
    """
    report = _in_clean_interpreter("""
        import cyqnt_trd
        assert cyqnt_trd.standard_bot is not None, "存取子包卻拿到 None"
    """)
    assert any(name.startswith("cyqnt_trd.standard_bot")
               for name in report["cyqnt_submodules"]), (
        "存取了 cyqnt_trd.standard_bot,它卻沒有進 sys.modules")


def test_the_cached_submodule_is_the_same_object_on_every_access():
    """第二次存取要命中快取,不能重新 import 一份。

    每次都重新 import 會產生兩個不同的模組物件,模組級狀態(快取、註冊表)就會分裂。
    """
    report = _in_clean_interpreter("""
        import cyqnt_trd
        first = cyqnt_trd.compat
        second = cyqnt_trd.compat
        assert first is second, "同一個子包存取兩次拿到兩個不同物件"
    """)
    assert report["sdk_and_http"] == []


def test_an_unknown_attribute_still_raises_attribute_error():
    """``__getattr__`` 接管之後,打錯的名字不能靜默變成 None。

    回 None 會讓 ``cyqnt_trd.get_dtaa`` 這種拼錯一路傳下去,直到很遠的地方才炸。
    """
    report = _in_clean_interpreter("""
        import cyqnt_trd
        try:
            cyqnt_trd.no_such_submodule
        except AttributeError:
            pass
        else:
            raise AssertionError("未知屬性沒有拋 AttributeError")
    """)
    assert report["sdk_and_http"] == []


# ── 失敗時的降級語義不能變 ────────────────────────────────────────────────

def test_a_failing_submodule_degrades_to_none_and_records_the_error():
    """子包 import 失敗時:不拋,屬性給 None,異常記進 ``_OPTIONAL_IMPORT_ERRORS``。

    這是 eager 版 ``_safe_import`` 的語義,有調用方按「缺可選依賴時子包是 None」寫的,
    所以改成懶加載時只能改**時機**(從 import 本包時推遲到首次存取),不能改結果。
    """
    import cyqnt_trd

    name = "strategy_cases"
    cached = cyqnt_trd.__dict__.get(name, "<absent>")
    recorded_before = dict(cyqnt_trd._OPTIONAL_IMPORT_ERRORS)
    real_import_module = importlib.import_module

    def boom(target, *args, **kwargs):
        if target == f"cyqnt_trd.{name}":
            raise ImportError("simulated missing optional dependency")
        return real_import_module(target, *args, **kwargs)

    try:
        cyqnt_trd.__dict__.pop(name, None)          # 清掉快取,否則走不到 __getattr__
        sys.modules.pop(f"cyqnt_trd.{name}", None)
        importlib.import_module = boom

        assert getattr(cyqnt_trd, name) is None, "import 失敗時屬性應該是 None"
        assert name in cyqnt_trd._OPTIONAL_IMPORT_ERRORS, "失敗的異常沒有被記下來"
        assert isinstance(cyqnt_trd._OPTIONAL_IMPORT_ERRORS[name], ImportError)
    finally:
        importlib.import_module = real_import_module
        cyqnt_trd._OPTIONAL_IMPORT_ERRORS.clear()
        cyqnt_trd._OPTIONAL_IMPORT_ERRORS.update(recorded_before)
        if cached == "<absent>":
            cyqnt_trd.__dict__.pop(name, None)
        else:
            cyqnt_trd.__dict__[name] = cached


# ── 兩份清單不能各自漂 ────────────────────────────────────────────────────

def test_the_lazy_list_and_all_do_not_drift():
    """``_LAZY_SUBMODULES`` 與 ``__all__`` 裡的子包名必須一致。

    只加進 ``__all__`` 的話,``from cyqnt_trd import *`` 會炸 AttributeError;
    只加進 ``_LAZY_SUBMODULES`` 的話,它不會出現在 ``__all__`` / ``dir()`` 裡,
    等於加了一個沒人知道的公開 API。兩種都不會有錯誤訊息。
    """
    import cyqnt_trd

    exported = {name for name in cyqnt_trd.__all__ if not name.startswith("_")}
    assert exported == set(cyqnt_trd._LAZY_SUBMODULES), (
        "__all__ 與 _LAZY_SUBMODULES 不一致:"
        f"只在 __all__ = {sorted(exported - set(cyqnt_trd._LAZY_SUBMODULES))}, "
        f"只在 _LAZY_SUBMODULES = {sorted(set(cyqnt_trd._LAZY_SUBMODULES) - exported)}")


def test_star_import_reaches_every_exported_name():
    """``__all__`` 每一個名字都要真的取得到 —— 這是上面那條的可執行版本。

    子行程裡任何一個名字取不到就是 AttributeError,子行程非零退出,
    ``_in_clean_interpreter`` 直接斷言失敗。這裡再確認一次它真的把子包導進來了,
    否則一個「每個名字都回 None」的實作也能讓子行程安靜地跑完。
    """
    report = _in_clean_interpreter("""
        import cyqnt_trd
        for name in cyqnt_trd.__all__:
            if not name.startswith("_"):
                assert getattr(cyqnt_trd, name) is not None, name
    """)
    imported_roots = {name.split(".")[1] for name in report["cyqnt_submodules"]
                      if name.count(".") >= 1}
    import cyqnt_trd

    assert set(cyqnt_trd._LAZY_SUBMODULES) <= imported_roots, (
        "取遍 __all__ 之後這些子包還是沒進 sys.modules:"
        f"{sorted(set(cyqnt_trd._LAZY_SUBMODULES) - imported_roots)}")


@pytest.mark.parametrize("name", ["get_data", "standard_bot", "compat", "utils"])
def test_dir_lists_the_lazy_submodules(name):
    """``dir(cyqnt_trd)`` 要列出懶加載的子包。

    PEP 562 的 ``__getattr__`` 對 ``dir()`` 是隱形的,不補一個 ``__dir__`` 的話
    自動補全和 ``inspect`` 就看不到這些名字 —— API 還在,只是沒人找得到。
    """
    import cyqnt_trd

    assert name in dir(cyqnt_trd)
