#!/usr/bin/env python3
"""重算 docs/ARCHITECTURE.md 裡的每一個數字,並印出它的定義。

為什麼需要這支腳本:這個 repo 曾經在四份文件裡對「blocks 有幾個函式」給出四個不同的
答案(146 / 310 / 365 / 393)。四個數字沒有一個是造假的 —— 它們用了四種不同的計法,
而沒有一份文件寫出自己用的是哪一種。所以這裡的規則是:

    每個數字都跟著它的定義一起印,永遠不要只印數字。

用法::

    python docs/count_architecture_facts.py            # 全部
    python docs/count_architecture_facts.py --diff     # 只印與 ARCHITECTURE.md 不一致的

靜態分析為主,不需要網路。只有「YAML 可觸及的 block 數」需要 import cyqnt_trd,
import 失敗時該項會跳過而不是讓整支腳本掛掉。
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# ARCHITECTURE.md 目前寫的值,用來偵測文件是否已經漂走。
# 改了架構就更新這裡與 ARCHITECTURE.md，兩邊一起動。
DOCUMENTED = {
    "blocks.public_def": 330,
    "blocks.files": 25,
    "top_level.modules": 14,
    "entrypoints.runnable": 17,
    "catalog.nodes": 75,
    "catalog.BACKTESTABLE": 7,
    "catalog.FORWARD_ONLY": 38,
    "catalog.INTERNAL_HTTP": 28,
    "universe.blocks": 31,
    "intent.values": 12,
    "signal_schema.properties": 42,
    "entrypoints.files": 19,
    "trading_signal.files": 126,
    "register()": 28,
    "register_selection()": 6,
}

results: dict[str, int] = {}


def report(key, value, definition, note=""):
    results[key] = value
    doc = DOCUMENTED.get(key)
    flag = ""
    if doc is not None and doc != value:
        flag = f"   <<< ARCHITECTURE.md 寫 {doc},已漂走"
    print(f"  {key:<32} = {value:>6}   {definition}{flag}")
    if note:
        print(f"  {'':<32}   {note}")


def blocks_facts():
    print("\n█ blocks —— 「幾個函式」有四種計法,差異全部來自定義")
    files = sorted(p for p in (REPO / "cyqnt_trd/blocks").glob("*.py")
                   if p.stem != "__init__")
    report("blocks.files", len(files), "cyqnt_trd/blocks/*.py(不含 __init__)")

    top_level = 0
    all_exports = 0
    for path in files:
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                top_level += 1
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "__all__" for t in node.targets):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    all_exports += len(node.value.elts)
    report("blocks.public_def", top_level,
           "各模組自己定義的頂層公開函式(ast,不含 class、不含 re-export)",
           "ARCHITECTURE.md §五用這個定義")
    report("blocks.__all__", all_exports, "__all__ 匯出的名稱總數(含 re-export,會重複計)")

    # 第四種:YAML 解譯器可觸及的 block。需要 import,失敗就跳過。
    try:
        sys.path.insert(0, str(REPO))
        import importlib
        from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (  # noqa: WPS433
            SpecError, resolve_block)
        total = ok = 0
        for path in files:
            if path.stem.startswith("_"):
                continue
            mod = importlib.import_module(f"cyqnt_trd.blocks.{path.stem}")
            for name, obj in vars(mod).items():
                if not callable(obj) or name.startswith("_"):
                    continue
                if not getattr(obj, "__module__", "").startswith("cyqnt_trd.blocks"):
                    continue
                total += 1
                try:
                    resolve_block(f"{path.stem}.{name}")
                    ok += 1
                except SpecError:
                    pass
        report("blocks.yaml_visible_total", total,
               "blocks 底下所有公開 callable(含 class、含 re-export)",
               "docs/gen_report_html.py 用這個定義 —— 它是四個數字裡最大的那個")
        report("blocks.yaml_reachable", ok, "上面那些之中,YAML 寫得出來的")
    except Exception as exc:  # pragma: no cover - 只是少一項,不該讓整支腳本掛掉
        print(f"  (跳過 YAML 可觸及數:import 失敗 —— {type(exc).__name__}: {exc})")

    report("universe.blocks",
           sum(1 for node in ast.parse((REPO / "cyqnt_trd/blocks/universe.py").read_text()).body
               if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")),
           "cyqnt_trd/blocks/universe.py 的頂層公開函式(選幣 pipeline 可用的步驟)")


def catalog_facts():
    print("\n█ 資料節點目錄 —— 「有這個資料」不等於「能拿它回測」")
    src = (REPO / "cyqnt_trd/standard_bot/data/catalog.py").read_text()
    report("catalog.nodes", src.count("DataNodeSpec("), "catalog.py 裡的 DataNodeSpec(")

    for kind, pattern in (("availability", r"availability=Availability\.([A-Z_]+)"),
                          ("source_path", r"source_path=SourcePath\.([A-Z_]+)")):
        counts: dict[str, int] = {}
        for name in re.findall(pattern, src):
            counts[name] = counts.get(name, 0) + 1
        print(f"    ── {kind} 分佈 ──")
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            report(f"catalog.{name}", n, f"{kind}={name}")


def module_facts():
    print("\n█ 頂層模組 —— 引用次數 + 引用來源")
    print("    定義:在 cyqnt_trd / strategies 底下被 import 的次數(排除模組自己內部)")
    print("    引用『來源』比『次數』重要 —— 一個模組可以被引用很多次,而引用它的全都是")
    print("    自己也沒有 caller 的腳本區。那不是主線,只是一群死人在互相呼叫。")
    roots = [p.name for p in (REPO / "cyqnt_trd").iterdir()
             if p.is_dir() and not p.name.startswith(("_", "."))
             and any((p).rglob("*.py"))]          # 空目錄不算一個模組
    py_files = [p for d in ("cyqnt_trd", "strategies")
                for p in (REPO / d).rglob("*.py")]
    rows = []
    for mod in sorted(roots):
        own = f"cyqnt_trd/{mod}/"
        hits = 0
        sources: dict[str, int] = {}
        for path in py_files:
            if own in str(path):
                continue
            text = path.read_text(errors="ignore")
            n = (len(re.findall(rf"cyqnt_trd\.{mod}\b", text))
                 + len(re.findall(rf"from \.{{1,2}}{mod}\b", text)))
            if n:
                hits += n
                rel = path.relative_to(REPO)
                # 歸給引用者所屬的頂層模組,而不是個別檔案
                owner = rel.parts[1] if rel.parts[0] == "cyqnt_trd" and len(rel.parts) > 2 \
                    else rel.parts[0]
                sources[owner] = sources.get(owner, 0) + n
        n_files = sum(1 for _ in (REPO / "cyqnt_trd" / mod).rglob("*.py"))
        rows.append((hits, mod, n_files, sources))

    # 誰是「活的引用者」:自己被引用 >0 次的模組。用來判斷一個引用有沒有份量。
    live = {mod for hits, mod, _, _ in rows if hits > 0}
    for hits, mod, n_files, sources in sorted(rows, reverse=True):
        top = ", ".join(f"{k}×{v}" for k, v in
                        sorted(sources.items(), key=lambda kv: -kv[1])[:3]) or "—"
        print(f"  {mod:<20} 引用 {hits:>4}  {n_files:>3} 檔   來源:{top}")
        results[f"module.{mod}.refs"] = hits
        if hits and not (set(sources) & live - {mod}):
            print(f"  {'':<20} ↑ 引用它的模組自己都沒有 caller —— 這不是主線")
    report("trading_signal.files",
           sum(1 for _ in (REPO / "cyqnt_trd/trading_signal").rglob("*.py")),
           "平行的舊因子體系 —— 規模像核心,但 standard_bot 完全不碰它")
    report("top_level.modules", len(roots), "cyqnt_trd/ 底下有 .py 的子目錄")
    print("\n  注意:repo 根還有一個 strategies/,和 cyqnt_trd/strategies/ 是不同的兩處:")
    print(f"    cyqnt_trd/strategies/  {sum(1 for _ in (REPO/'cyqnt_trd/strategies').rglob('*.py')):>3} 檔")
    print(f"    strategies/            {sum(1 for _ in (REPO/'strategies').rglob('*.py')):>3} 檔")


def standard_bot_facts():
    print("\n█ standard_bot 的子系統")
    base = REPO / "cyqnt_trd/standard_bot"
    rows = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()
                      and not p.name.startswith(("_", "."))):
        files = sorted(sub.glob("*.py"))
        lines = sum(len(f.read_text(errors="ignore").splitlines()) for f in files)
        rows.append((lines, sub.name, len(files)))
    for lines, name, n in sorted(rows, reverse=True):
        note = "  (獨立子專案,不走這個架構)" if n == 0 else ""
        print(f"  {name:<32} {n:>3} 檔 {lines:>6} 行{note}")
    top = sorted(base.glob("*.py"))
    print(f"  {'(頂層檔案)':<32} {len(top):>3} 檔 "
          f"{sum(len(f.read_text(errors='ignore').splitlines()) for f in top):>6} 行")
    report("entrypoints.files", len(list((base / 'entrypoints').glob('*.py'))),
           "standard_bot/entrypoints/*.py")
    runnable = sum(1 for f in (base / "entrypoints").glob("*.py")
                   if "__main__" in f.read_text(errors="ignore"))
    report("entrypoints.runnable", runnable, "其中可以直接 python -m 跑的")


def contract_facts():
    print("\n█ 契約")
    import json
    schema = json.loads((REPO / "strategies/_standard/signal.schema.v2.json").read_text())
    report("signal_schema.properties", len(schema.get("properties", {})),
           "signal.schema.v2.json 的頂層欄位(交易與選幣共用同一份)")

    src = (REPO / "cyqnt_trd/standard_bot/core/signal_contract.py").read_text()
    m = re.search(r"class PositionIntent\(.*?\):(.*?)(?=\nclass |\n@|\Z)", src, re.S)
    if m:
        report("intent.values", len(re.findall(r"^\s{4}[A-Z_]+\s*=", m.group(1), re.M)),
               "PositionIntent 的值 —— 唯一決定「要對部位做什麼」的欄位")

    print("\n█ 策略註冊(策略『支數』要數呼叫次數,不是檔數)")
    calls = {"register(": 0, "register_selection(": 0}
    for d in ("strategies", "cyqnt_trd/strategies"):
        for path in (REPO / d).rglob("*.py"):
            text = path.read_text(errors="ignore")
            calls["register_selection("] += text.count("register_selection(")
            calls["register("] += (text.count("strategy.register(")
                                   + text.count("S.register("))
    report("register()", calls["register("], "block path 的交易型策略")
    report("register_selection()", calls["register_selection("], "block path 的選幣型策略")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff", action="store_true",
                        help="只印與 ARCHITECTURE.md 不一致的項")
    args = parser.parse_args()

    print("=" * 96)
    print("docs/ARCHITECTURE.md 的數字重算 —— 每個數字都附它的定義")
    print("=" * 96)
    blocks_facts()
    catalog_facts()
    module_facts()
    standard_bot_facts()
    contract_facts()

    drifted = {k: (DOCUMENTED[k], results[k]) for k in DOCUMENTED
               if k in results and DOCUMENTED[k] != results[k]}
    print("\n" + "=" * 96)
    if drifted:
        print(f"ARCHITECTURE.md 有 {len(drifted)} 個數字已經漂走:")
        for key, (doc, now) in sorted(drifted.items()):
            print(f"  {key:<32} 文件寫 {doc:>6} → 實際 {now:>6}")
        print("\n請更新 docs/ARCHITECTURE.md 與本檔的 DOCUMENTED,兩邊一起動。")
    else:
        print("ARCHITECTURE.md 的數字與現況一致。")
    print("=" * 96)
    print("測試數要另外跑(這支腳本不跑測試):python -m pytest tests -q")
    print("docs/report_flow.html 是實跑生成的,重新產生:python docs/gen_report_html.py")
    return 1 if (drifted and args.diff) else 0


if __name__ == "__main__":
    sys.exit(main())
