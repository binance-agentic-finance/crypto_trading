# atomic_compat — atomic_strategy_lib shim layer

**目的**：讓使用 `from atomic_strategy_lib.X import Y` 的舊 case 程式碼，
不改任何一個字也能在 cyqnt_trd 環境下跑。

## TL;DR

```python
# Case 寫:
from atomic_strategy_lib.scoring.gates import verdict_with_gate

# Python 找到我們這個 shim:
#   atomic_compat/atomic_strategy_lib/scoring/gates.py
# Shim 內部 redirect 到:
#   cyqnt_trd/blocks/verdicts.py
# verdict_with_gate 最終來自 cyqnt_trd
```

## 結構

```
atomic_compat/
└── atomic_strategy_lib/    ← 名字故意跟 atomic 真版一樣
    ├── core/   scoring/   decision/   risk/
    ├── signals/   data/   execution/
    └── monitoring/   orchestration/
```

51 個檔，每個 5-30 行 thin re-export。

## 安裝後位置

`pip install cyqnt-trd>=0.1.9.dev3` 之後，shim 安裝在：

```
<site-packages>/atomic_strategy_lib/   ← shim
<site-packages>/cyqnt_trd/             ← 真實實作
```

兩個都 importable，無需設環境變數。

## 想要更深的細節？

看 `docs/atomic-compat/MIGRATION_HANDOFF.md`。
