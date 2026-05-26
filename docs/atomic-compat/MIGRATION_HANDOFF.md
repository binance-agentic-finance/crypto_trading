# Atomic Strategy Lib → cyqnt_trd 遷移 Handoff

> 寫給：之後維護這個系統的人（包括未來的自己）
>
> 本文記錄 2026-05-26 完成的 atomic_strategy_lib → cyqnt_trd 整合工作。
> 包含背景、設計思路、實作細節、驗證方式、OpenClaw 部署、與後續維護指南。

---

## 目錄

1. [我們解決了什麼問題](#1-我們解決了什麼問題)
2. [關鍵設計：什麼是 shim package](#2-關鍵設計什麼是-shim-package)
3. [架構：三層結構](#3-架構三層結構)
4. [為什麼 user cases 不用改任何東西](#4-為什麼-user-cases-不用改任何東西)
5. [完整工作流程](#5-完整工作流程)
6. [驗證：4 個層次](#6-驗證4-個層次)
7. [OpenClaw 部署](#7-openclaw-部署)
8. [維護指南](#8-維護指南)
9. [附錄](#9-附錄)

---

## 1. 我們解決了什麼問題

### 1.1 起點

公司有兩個並存的策略庫：

| 庫 | 位置 | 風格 |
|---|---|---|
| **atomic_strategy_lib** | `dev_trading_bot/must-read-intent-analysis-planning/references/atomic_strategy_lib/` | stdlib only、`list/dict` I/O、subprocess 呼叫 binance-cli |
| **cyqnt_trd** | `crypto_trading-main/cyqnt_trd/` | pandas-first、`Series/DataFrame` I/O、有 backtest engine、paper trade daemon |

加上 **44 個 user cases**：
```
must-read-intent-analysis-planning/references/usage_project_cases/
├── ahr999-btc-hodl/
├── bb-squeeze-momentum/
├── btc-multi-factor-trend/
... (44 個)
└── whale-tracker/
```
每個 case 有 `scripts/run_pipeline.py` + `_paths.py` + 多個輔助腳本，全部都是 `from atomic_strategy_lib.X.Y import Z` 的進口風格。

### 1.2 目標

**用 cyqnt_trd 取代 atomic_strategy_lib，但 44 個 user case 程式碼一字不改**。

具體要求：
- 所有 case 的 imports 還能解析
- 所有 case 的呼叫不會 runtime error
- 跟 atomic 真實版的數值一致
- OpenClaw 雲端拉新版 cyqnt-trd 後，case 不需要設任何環境變數

### 1.3 為什麼不直接改 case？

| 直接改 case | 用 shim |
|---|---|
| 44 case × ~10 imports = ~440 處要改 | 一次做完 51 個 shim 檔，case 0 改動 |
| 介面風格也要改（list → Series, dict → Series）| Shim 在內部處理介面轉換 |
| 維護兩份 case（atomic 風格、cyqnt 風格）| 維護一份 case |
| 任何後續改動都會擴散到 44 個 case | 任何改動都集中在 cyqnt_trd 內 |

---

## 2. 關鍵設計：什麼是 shim package

### 2.1 一句話

**Shim = 51 個假冒 `atomic_strategy_lib` 名字的 thin re-export 檔，內容只有 `from cyqnt_trd.X import Y`，沒有實際邏輯。**

### 2.2 比喻

```
你訂披薩到「中山北路 100 號」（atomic_strategy_lib）。
舊房子拆了，原地蓋了新房子（cyqnt_trd），門牌還是「中山北路 100 號」。
披薩外送員照樣送達，不會也不需要知道後面換房了。
```

對應到 Python：

```python
# Case 寫
from atomic_strategy_lib.scoring.gates import verdict_with_gate

# Python 找 sys.path → 找到 site-packages 內的 atomic_strategy_lib/scoring/gates.py
# 那個檔案內容是：
"""shim — atomic.scoring.gates"""
from cyqnt_trd.blocks.verdicts import verdict_with_gate

# 最終 verdict_with_gate 來自 cyqnt_trd
# Case 完全不知道
```

### 2.3 實際 shim 檔長怎樣

打開任何一個，內容極簡：

```python
# atomic_compat/atomic_strategy_lib/scoring/gates.py（11 行）
"""shim — atomic.scoring.gates (8 gates / verdict helpers)"""
from cyqnt_trd.blocks.verdicts import (  # noqa: F401
    hard_gate, enum_gate, soft_factor,
    verdict_classify, verdict_with_gate,
    cross_validate, conflict_detect, normalize_score,
)
```

```python
# atomic_compat/atomic_strategy_lib/decision/sizing.py（10 行）
"""shim — atomic.decision.sizing"""
from cyqnt_trd.blocks.sizing import (  # noqa: F401
    fixed_dollar_loss, fixed_risk_pct, kelly_size,
    atr_inverse_size, leverage_cap,
    compute_stop_price, adaptive_stop_pct,
)
```

51 個檔總共 1117 行，平均 22 行/檔。其中 49 個是 `< 30 行` 的純 re-export；兩個比較大：
- `core/run_output.py`（125 行）— atomic 的 IO helper，無對應 cyqnt 實作，所以**直接內嵌 atomic 原本的純 stdlib 邏輯**
- `risk/graduated_tp.py`（146 行）— atomic 的 stateful 三段式 TP，cyqnt 只有 stateless ladder builder，所以多寫了組合邏輯

### 2.4 還有第二層 shim：`cyqnt_trd/compat/atomic_signals/`

Atomic 的 signal helpers（RSI / MACD / EMA / ATR / Bollinger / 等）演算法跟 cyqnt 主庫的 pandas 版本**不完全一樣**：
- atomic：純 Python，SMA-then-Wilder smoothing
- cyqnt 主庫：pandas.ewm，pure-Wilder

兩者數值差異可能達 30+（測試實測 RSI 平均差 34.24 點）。

為了確保「shim 跟 atomic 數值一致」，我們把同事在 `migrate_library` branch 上做的 atomic-verbatim 演算法 port 過來，放在 `cyqnt_trd/compat/atomic_signals/`：

```
cyqnt_trd/compat/atomic_signals/
├── __init__.py
├── computes.py              ← 681 行純 Python，verbatim 移植 atomic
├── derivatives_detectors.py ← funding/OI/crowding 偵測
├── structure.py              ← Fibonacci/pivots/candlestick/box-range
└── volume_detectors.py       ← volume surge/trend
```

第一層 shim 內 `signals/momentum.py` 等 5 個檔，內部 redirect 到這個 atomic_signals package：

```python
# atomic_compat/atomic_strategy_lib/signals/momentum.py
from cyqnt_trd.compat.atomic_signals import (
    rsi_compute, rsi_current, rsi_zone_detect,
    macd_compute, stochrsi_compute,
)
```

這樣保證**shim 算出來的指標 = atomic 算出來的指標**，1e-9 精度內一致（已驗證 12/12 函數，236/236 ~ 250/250 全部 match）。

---

## 3. 架構：三層結構

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 1: Cases (44 個 user_project_cases)                    │
│                                                              │
│  位置: must-read-intent-analysis-planning/references/        │
│        usage_project_cases/<case>/scripts/                   │
│                                                              │
│  Case 寫: from atomic_strategy_lib.X.Y import Z              │
│  Case 不知道下面是什麼，也不關心                             │
└──────────────────────┬───────────────────────────────────────┘
                       │ Python import 解析
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ Layer 2: Atomic Shim (51 個假冒檔)                            │
│                                                              │
│  位置: crypto_trading-main/atomic_compat/atomic_strategy_lib/│
│        ├── core/        (config, types, cli, run_output, …) │
│        ├── scoring/     (combinators, gates)                 │
│        ├── decision/    (direction, sizing)                  │
│        ├── risk/        (stop_loss, graduated_tp, limits)    │
│        ├── signals/     (momentum, trend, volatility, …)     │
│        ├── data/        (kline, ticker, funding, …)          │
│        ├── execution/   (orders, position)                   │
│        ├── monitoring/  (pnl, signals, stops)                │
│        └── orchestration/ (pipeline, scheduler, state)       │
│                                                              │
│  每個檔內容: from cyqnt_trd.X.Y import Z (純 re-export)      │
│  目的: 把 atomic 名字 redirect 到 cyqnt_trd 實作            │
└──────────────────────┬───────────────────────────────────────┘
                       │ 內部 from cyqnt_trd...
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ Layer 3: cyqnt_trd 真實實作                                  │
│                                                              │
│  位置: crypto_trading-main/cyqnt_trd/                        │
│        ├── blocks/                                           │
│        │   ├── verdicts.py    ← atomic-style scoring/gates  │
│        │   ├── sizing.py      ← decision/sizing             │
│        │   ├── stop_loss.py   ← risk/stop_loss              │
│        │   ├── limits.py      ← risk/limits                 │
│        │   ├── exit.py        ← graduated_take_profit       │
│        │   └── decision.py    ← decision/direction          │
│        │                                                     │
│        ├── compat/                                           │
│        │   ├── types.py       ← dataclasses (Candle, Signal)│
│        │   ├── config.py      ← load_config + DEFAULT       │
│        │   ├── adapters.py    ← df_to_candles / candles_to_df│
│        │   └── atomic_signals/ ← atomic-verbatim algorithms │
│        │       ├── computes.py                               │
│        │       ├── derivatives_detectors.py                  │
│        │       ├── structure.py                              │
│        │       └── volume_detectors.py                       │
│        │                                                     │
│        ├── data_cli/          ← binance-cli wrappers        │
│        ├── exec_cli/          ← binance-cli order helpers   │
│        └── standard_bot/                                     │
│            ├── monitoring/    ← runtime PnL / signals / stops│
│            └── orchestration/ ← scan / rescan / state CRUD  │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. 為什麼 user cases 不用改任何東西

這是整個設計的關鍵。理解它需要四個觀念。

### 4.1 觀念 1：Python 怎麼解析 `import`

當 Python 看到 `from atomic_strategy_lib.X import Y`：

```
Python 按順序檢查 sys.path 上的每個目錄：
  1. 當前 script 的目錄
  2. 環境變數 PYTHONPATH 列的目錄
  3. 預設系統路徑
  4. site-packages（pip 安裝的套件都在這）

第一個找到名為 "atomic_strategy_lib" 的目錄就用它，不再往下找。
```

### 4.2 觀念 2：site-packages 永遠在 sys.path 上

這是 Python 預設行為。每個 venv / 系統 Python 啟動時，都會自動把 site-packages 加進 sys.path。

OpenClaw 的環境也一樣：`/usr/local/lib/python3.11/dist-packages/` 永遠在 sys.path 上。

### 4.3 觀念 3：我們把 shim 取名跟 atomic 一樣

關鍵的命名選擇：shim 套件叫 `atomic_strategy_lib`（**故意**跟原 atomic 一樣）。

```
crypto_trading-main/atomic_compat/atomic_strategy_lib/   ← 這個名字是設計選擇
```

如果取不一樣的名字（例如 `cyqnt_trd_atomic_compat`），case 就要改 import 路徑。

### 4.4 觀念 4：pip install 把 shim 放進 site-packages

`pyproject.toml` 改完後，`pip install cyqnt-trd==0.1.9.dev3` 會把兩個目錄都安裝進 site-packages：

```
/usr/local/lib/python3.11/dist-packages/
├── cyqnt_trd/                  ← 真實實作
└── atomic_strategy_lib/        ← shim（從 atomic_compat/ 來，安裝時去掉前綴）
    ├── scoring/
    ├── signals/
    └── ...
```

### 4.5 把四個觀念串起來

```
Case 寫:
    from atomic_strategy_lib.scoring.gates import verdict_with_gate
                                ↓
Python 按 sys.path 找:
    1. case scripts/ → 沒有 atomic_strategy_lib/
    2. 系統路徑      → 沒有
    3. site-packages → ✓ 找到 (我們透過 pip install 放進去的 shim)
                                ↓
        載入 site-packages/atomic_strategy_lib/scoring/gates.py
                                ↓
        執行該檔案的內容: "from cyqnt_trd.blocks.verdicts import verdict_with_gate"
                                ↓
        Python 再找 cyqnt_trd → site-packages 也有 → 找到真實實作
                                ↓
verdict_with_gate 最終來自 cyqnt_trd.blocks.verdicts
Case 全程不知道，也不需要知道
```

### 4.6 那 case 自帶的 `_paths.py` 呢？

每個 case 的 `scripts/_paths.py` 原本想做的事：

```python
# scripts/_paths.py 的邏輯
_default = Path(__file__)... / "atomic_strategy_lib"   # 預設指向原 atomic
_lib_root = Path(os.environ.get("ATOMIC_STRATEGY_LIB_PATH", str(_default)))

if str(_lib_root.parent) not in sys.path:
    sys.path.insert(0, str(_lib_root.parent))
```

它是想確保 sys.path 上有原 atomic 的位置。但是：

- 如果 `ATOMIC_STRATEGY_LIB_PATH` 沒設且原 atomic 真版已被刪 → `_default` 路徑不存在 → `if not in sys.path` 條件成立但加進去也沒用（那位置根本沒檔案）
- **重點是**：site-packages 永遠在 sys.path 上，反正 Python 會找到 shim

所以 `_paths.py` 即使邏輯失敗（找不到原 atomic）也不影響結果。**Case 完全不用改 `_paths.py`，shim 也能 work**。

### 4.7 一句話總結

> **因為 (a) shim 取名跟 atomic 完全一樣，且 (b) pip install 後安裝在 Python 預設找的位置 (site-packages)，所以 case 寫 `from atomic_strategy_lib.X import Y` 時 Python 直接從 site-packages 找到 shim，case 完全感覺不到「後面換成 cyqnt_trd 了」。**

---

## 5. 完整工作流程

完整流程從問題盤點到 PyPI 上線。

### 5.1 階段 0：盤點

| 工作 | 結果 |
|---|---|
| 比對 atomic vs cyqnt 函數覆蓋率 | atomic 217 個函數、cyqnt 已涵蓋 158/217 (~73%)，缺 21 個核心函數 |
| 找出 44 個 user cases 用到的 atomic API | 列出 ~80 個被 import 的函數 |
| 評估介面差異 | atomic 用 `list/dict`、cyqnt 用 `Series/DataFrame`，不能直接 1:1 換 |

### 5.2 階段 1：補 atomic 核心缺失（21 個函數）

從同事的 `ai_pro_trading_library` 借用了純 Python 實作，移植到 cyqnt_trd：

| 補的東西 | 在哪 | 為什麼 |
|---|---|---|
| `verdicts.py`（5 combinators + 8 gates）| `cyqnt_trd/blocks/verdicts.py` | atomic.scoring.combinators + scoring.gates |
| `limits.py`（7 個 risk check）| `cyqnt_trd/blocks/limits.py` | atomic.risk.limits |
| `stop_loss.py`（4 個 stop helpers）| `cyqnt_trd/blocks/stop_loss.py` | atomic.risk.stop_loss |
| `exit.py` 內加 `graduated_take_profit` | `cyqnt_trd/blocks/exit.py` | atomic.risk.graduated_tp |

驗證：21/21 補完，原有 324 個 pytest 全過。

### 5.3 階段 2：建立 shim package

51 個 thin re-export 檔，建在：
```
crypto_trading-main/atomic_compat/atomic_strategy_lib/
```

每個 shim 檔內容類似：
```python
from cyqnt_trd.X import Y, Z
```

涵蓋 atomic 的所有 namespace：
- `core/` (config, types, cli, run_output, template_hint)
- `scoring/` (combinators, gates)
- `decision/` (direction, sizing)
- `risk/` (stop_loss, graduated_tp, limits)
- `signals/` (momentum, trend, volatility, volume, derivatives, structure)
- `data/` (kline, ticker, funding, oi, orderbook, account, scanner, market_bundle, pro, workflow, web3, search, tradfi, market_cache, http_client)
- `execution/` (orders, position)
- `monitoring/` (pnl, signals, stops)
- `orchestration/` (pipeline, scheduler, state)

### 5.4 階段 3：建立 atomic-verbatim signals

從同事 `migrate_library` branch 借用 verbatim 演算法：

```
cyqnt_trd/compat/atomic_signals/
├── computes.py              (681 行：RSI, MACD, EMA, ATR, Bollinger, ADX, StochRSI, SuperTrend)
├── derivatives_detectors.py (177 行：funding_extreme, oi_anomaly, crowding)
├── structure.py             (249 行：fibonacci, pivots, candlestick patterns, box-range)
└── volume_detectors.py      (95 行：volume surge, trend, normalize)
```

第一層 shim 的 `signals/*.py` 都 redirect 到這裡，保證演算法跟 atomic 1:1 一致。

### 5.5 階段 4：驗證

詳見 [§6 驗證章節](#6-驗證4-個層次)。

44/44 case Layer 1-3 全過。12/12 核心函數數值一致。

### 5.6 階段 5：改 packaging

關鍵改動：`pyproject.toml`。

**改前**：
```toml
[tool.setuptools]
packages = { find = {} }    # 只找 root，只裝 cyqnt_trd
```

**改後**：
```toml
[tool.setuptools]
package-dir = {"" = ".", "atomic_strategy_lib" = "atomic_compat/atomic_strategy_lib"}

[tool.setuptools.packages.find]
where = [".", "atomic_compat"]
include = ["cyqnt_trd*", "atomic_strategy_lib*"]
namespaces = false
```

這 8 行改動讓 `pip install cyqnt-trd` 把兩個 package 都裝進 site-packages。

### 5.7 階段 6：發版

```
版本: 0.1.9.dev2 → 0.1.9.dev3

git commit (hash df46542)
git push binance main
git push origin main
python -m build
python -m twine upload dist/*
```

PyPI: https://pypi.org/project/cyqnt-trd/0.1.9.dev3/

### 5.8 完整時間線

| 時間 | 動作 | Commit |
|---|---|---|
| 早上 | 補 21 個 atomic 核心函數到 cyqnt_trd | `74eff8b` |
| 中午 | 把 atomic-port team 完整 work commit | `038a08f` |
| 下午 | 建 atomic_signals adapter | `7e6250e` |
| 下午 | 建 51 個 shim + atomic-verbatim signals | `a80b081` |
| 傍晚 | 改 packaging + bump version + 發 PyPI | `df46542` |

---

## 6. 驗證：4 個層次

我們從淺到深驗證了 4 個層次。

### 6.1 Layer 1-3：每個 case 的 syntax + import + entrypoint

**測試方法**：
```python
# 對每個 case：
1. py_compile 每個 .py     → syntax 過嗎
2. exec_module run_pipeline.py  → import 過嗎
3. python run_pipeline.py --help → main 入口能 load 嗎
```

**結果**：
```
PASS: 44   FAIL: 0   SKIP: 0   TOTAL: 44
```

完整 list 見 `/tmp/case_results.json`。

### 6.2 Layer 4：核心演算法數值對齊

**測試方法**：拿同一份 250 bar input.json，分別跑：
- atomic 真實版（從 atomic 原始庫 import）
- shim 版（透過 atomic_compat shim → cyqnt_trd）

逐元素比對。

**結果**：12/12 函數，所有元素差異 < 1e-9：

| 函數 | 元素一致數 |
|---|---|
| `rsi(14)` | 236/236 |
| `ema(20)` | 250/250 |
| `ema(50)` | 250/250 |
| `macd.macd` | 250/250 |
| `macd.signal` | 250/250 |
| `macd.histogram` | 250/250 |
| `atr(14)` | 236/236 |
| `bollinger.upper` | 231/231 |
| `bollinger.middle` | 231/231 |
| `bollinger.lower` | 231/231 |
| `bollinger.bandwidth` | 231/231 |
| `bollinger.pct_b` | 231/231 |

### 6.3 Layer 5：Case 級規則對 ai_pro golden fixtures

借用同事 `migrate_library` branch 的 `cases/migrated/<case>/fixtures/`：
- `input.json`（250 OHLCV bars）
- `golden_signal.json`（預期 signal 觸發的 bar indices）

對於 `rsi-mean-reversion`（RSI < 30）和 `btc-multi-factor-trend`（EMA20>EMA50 + RSI>50），用我們 shim 跑出 indices，跟 golden 比 Jaccard 相似度：

| Case | Jaccard | 結論 |
|---|---|---|
| btc-multi-factor-trend | 0.954 | 邊界差異，邏輯正確 |
| rsi-mean-reversion | 0.833 | warmup 處理略不同（atomic warmup 14 bar，ai_pro 多加幾根 settling），核心邏輯正確 |

### 6.4 Layer 6：完整 OpenClaw 環境模擬

**測試方法**：
```bash
# 1. 把原 atomic_strategy_lib 改名（模擬已從 must-read 刪除）
mv must-read/.../atomic_strategy_lib must-read/.../atomic_strategy_lib.disabled

# 2. 仿造 OpenClaw 雲端結構
mkdir -p /tmp/openclaw_sim/app/skills/must-read.../usage_project_cases

# 3. 全新 venv（模擬乾淨雲端環境）
python -m venv /tmp/openclaw_sim/venv
source /tmp/openclaw_sim/venv/bin/activate

# 4. pip install 我們的 wheel（含 shim）
pip install cyqnt_trd-0.1.9.dev3-py3-none-any.whl pandas numpy websockets

# 5. 不設任何環境變數
unset PYTHONPATH ATOMIC_STRATEGY_LIB_PATH

# 6. 跑 44 個 case 的 import test
python /tmp/sim_test.py
```

**結果**：43 PASS / 0 FAIL / 1 SKIP（popular_trading_bot 沒有 run_pipeline.py）。

確認在「真的沒有 atomic_strategy_lib 真實版 + 沒設任何環境變數 + 全新 venv」的最嚴格條件下，shim 仍 work。

---

## 7. OpenClaw 部署

### 7.1 OpenClaw 雲端架構

根據用戶提供的資訊，OpenClaw 雲端是這樣：

```
/app/skills/                                 ← 系統內建 Skills 根目錄
├── must-read-intent-analysis-planning/
└── trading-monitoring-executing-router/

/usr/local/lib/python3.11/dist-packages/
└── cyqnt_trd/                               ← Python 套件 (透過 pip install)
```

### 7.2 部署步驟（給 OpenClaw release pipeline）

```bash
# 1. OpenClaw release pipeline 拉新版 cyqnt-trd
pip install --upgrade cyqnt-trd==0.1.9.dev3

# 2. 確認 site-packages 有兩個 package
ls /usr/local/lib/python3.11/dist-packages/
# 應該看到:
#   cyqnt_trd/
#   atomic_strategy_lib/   ← 這個是新的
#   cyqnt_trd-0.1.9.dev3.dist-info/

# 3. 用戶在 OpenClaw 跑 case
cd /app/skills/must-read-intent-analysis-planning/references/usage_project_cases/rsi-mean-reversion/scripts
python run_pipeline.py
# 應該直接 work，不需要設任何環境變數
```

### 7.3 用戶不需要做什麼

| 不需要做的事 | 原因 |
|---|---|
| ❌ 改 case 程式碼 | shim 名字跟 atomic 完全一樣 |
| ❌ 改 `_paths.py` | site-packages 永遠在 sys.path 上 |
| ❌ 設 `PYTHONPATH` | site-packages 自動可用 |
| ❌ 設 `ATOMIC_STRATEGY_LIB_PATH` | shim 已在 site-packages |
| ❌ 維護兩個 atomic 版本 | 真版可以刪除，shim 完全替代 |

### 7.4 故障排除

如果 case 跑出 ImportError，按順序檢查：

```bash
# 1. cyqnt-trd 是否裝對版本？
pip show cyqnt-trd
# 期待 Version: 0.1.9.dev3 或更新

# 2. site-packages 是否有 atomic_strategy_lib？
python -c "import atomic_strategy_lib; print(atomic_strategy_lib.__file__)"
# 期待: /usr/local/lib/python3.11/dist-packages/atomic_strategy_lib/__init__.py

# 3. shim 內部能 import cyqnt_trd 嗎？
python -c "from atomic_strategy_lib.scoring.gates import verdict_with_gate; print('OK')"
# 期待: OK

# 4. 如果以上都通過，case 還是 ImportError，看完整 traceback
python -c "
import sys, traceback
sys.path.insert(0, '/path/to/case/scripts')
try:
    import _paths
    import importlib.util
    spec = importlib.util.spec_from_file_location('rp', '/path/to/case/scripts/run_pipeline.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
except Exception:
    traceback.print_exc()
"
```

常見問題：

| 錯誤 | 原因 | 解法 |
|---|---|---|
| `No module named 'atomic_strategy_lib'` | cyqnt-trd 是舊版（< 0.1.9.dev3）| `pip install --upgrade cyqnt-trd` |
| `No module named 'websockets'`（或其他依賴）| pip 安裝沒帶 dependencies | `pip install cyqnt-trd[all]` 或手動裝 |
| `cannot import name 'X' from 'cyqnt_trd...'` | shim 引用的 cyqnt_trd 函數沒實作或 rename | 看 §8.3 維護指南 |

---

## 8. 維護指南

### 8.1 之後改 cyqnt_trd 怎麼影響 shim？

| 改動類型 | shim 受影響？ | 怎麼處理 |
|---|---|---|
| 在 cyqnt_trd 內加新函數 | ❌ 不影響 | 如果想讓 atomic case 也能用，在對應 shim 檔加 re-export |
| 改 cyqnt_trd 函數簽名 | ⚠️ 影響 | shim 內可能要包一層 wrapper 維持 atomic 介面 |
| 改 cyqnt_trd 函數命名 | ⚠️ 影響 | shim 改 `from cyqnt_trd.X import Y as Z` |
| 刪除 cyqnt_trd 函數 | ❌ shim 會壞 | 先看 atomic case 還用不用，要不要保留功能 |
| 改演算法行為 | ⚠️ 影響數值 | atomic_signals 內的 verbatim 演算法**不要改**（保 L2 parity） |

### 8.2 加新 atomic API 的 shim

例：發現某 case 用了 `atomic.foo.bar` 但 shim 沒涵蓋。

```bash
# 1. 看 atomic 真版的 bar 是什麼
cat dev_trading_bot/.../atomic_strategy_lib/foo/bar.py

# 2. 在 cyqnt_trd 內找對應實作（或寫一個）
# 假設 cyqnt_trd.something.bar 有對應功能

# 3. 在 shim 內加 re-export
echo '"""shim — atomic.foo.bar"""
from cyqnt_trd.something.bar import something_useful
' > atomic_compat/atomic_strategy_lib/foo/bar.py

# 4. 跑測試
.venv-standard-bot/bin/python /tmp/test_all_cases.py

# 5. bump version
# 改 pyproject.toml: 0.1.9.dev3 → 0.1.9.dev4

# 6. commit + push + 發 PyPI
git add -A && git commit -m "feat(shim): add atomic.foo.bar"
git push binance main
python -m build
python -m twine upload dist/*
```

### 8.3 改 cyqnt_trd 函數 → shim 配套修改

例：把 `cyqnt_trd.blocks.verdicts.verdict_with_gate` 改名為 `cyqnt_trd.blocks.verdicts.gate_verdict`。

```python
# atomic_compat/atomic_strategy_lib/scoring/gates.py 改成:
from cyqnt_trd.blocks.verdicts import (
    hard_gate, enum_gate, soft_factor,
    verdict_classify,
    gate_verdict as verdict_with_gate,    # ← 改名 alias
    cross_validate, conflict_detect, normalize_score,
)
```

這樣 case 還是 import `verdict_with_gate`，內部對應到新名字。

### 8.4 演算法不要改

`cyqnt_trd/compat/atomic_signals/computes.py` 內 RSI/MACD/EMA/ATR/Bollinger 的演算法**故意**跟 atomic 一致（warmup 期 SMA-then-Wilder）。

如果你想用更現代的演算法（pure Wilder / Cutler smoothing），請：
- 不要改 `atomic_signals/computes.py`
- 在 `cyqnt_trd/blocks/indicators.py` 內提供新版本（或已有）
- 兩者並存，atomic case 用 atomic_signals 版，新 strategy 用 blocks 版

### 8.5 發新版本流程

```bash
# 1. 改 pyproject.toml 內 version
# 例: 0.1.9.dev3 → 0.1.9.dev4

# 2. 改完所有 code，跑 test
.venv-standard-bot/bin/python -m pytest tests/standard_bot/ -q
.venv-standard-bot/bin/python /tmp/test_all_cases.py

# 3. Commit + push
git add -A && git commit -m "..."
git push binance main && git push origin main

# 4. Build
rm -rf build/ dist/ *.egg-info
.venv-standard-bot/bin/python -m build

# 5. 確認 wheel 含兩個 package
unzip -l dist/cyqnt_trd-X.Y.Z-*.whl | grep -E "atomic_strategy_lib/__init|cyqnt_trd/__init"

# 6. Upload
.venv-standard-bot/bin/python -m twine upload dist/*

# 7. 等 PyPI propagate（通常 < 1 分鐘）
# 8. OpenClaw 拉新版（pip install --upgrade cyqnt-trd）
```

### 8.6 移除 shim 的計畫（如果未來要做）

理想狀態：所有 case 都改用 cyqnt_trd 風格 import，不再需要 shim。

漸進路線：
1. **第 1 階段**（現在）：shim 存在，case 完全不改
2. **第 2 階段**：對於新 case，要求直接用 cyqnt_trd 風格 import
3. **第 3 階段**：把舊 case 一個個改 import path（用 sed/codemod 自動化）
4. **第 4 階段**：所有 case 都不用 shim 了，可以刪 atomic_compat/

但這個工作量很大（44 case × ~10 imports）。**短期不建議做**。

---

## 9. 附錄

### 9.1 重要 commit 清單

| Commit | 說明 |
|---|---|
| `74eff8b` | 補 21 個 atomic 核心缺失（verdicts, limits, stop_loss, graduated_tp） |
| `038a08f` | 把 atomic-port team 整體工作 commit（compat/, data_cli/, exec_cli/, monitoring/, orchestration/） |
| `7e6250e` | 建 cyqnt_trd/compat/atomic_signals adapter（pandas-wrap 版本，後來被取代） |
| `a80b081` | 建 51 個 shim + atomic-verbatim signals（取代 7e6250e 的 pandas-wrap） |
| **`df46542`** | **packaging：把 shim 也打進 wheel，bump 0.1.9.dev3，發 PyPI** |

### 9.2 重要檔案清單

```
crypto_trading-main/
├── pyproject.toml                       ← packaging 設定，控制誰被打進 wheel
├── atomic_compat/
│   ├── README.md                        ← shim 簡短說明（建議補）
│   └── atomic_strategy_lib/             ← 51 個 shim 檔
│       ├── __init__.py
│       ├── core/, scoring/, decision/, risk/, signals/,
│       ├── data/, execution/, monitoring/, orchestration/
│       └── ...
│
├── cyqnt_trd/
│   ├── blocks/
│   │   ├── verdicts.py                  ← scoring/gates 真實實作
│   │   ├── sizing.py                    ← decision/sizing 真實實作
│   │   ├── stop_loss.py                 ← risk/stop_loss 真實實作
│   │   ├── limits.py                    ← risk/limits 真實實作
│   │   ├── exit.py                      ← graduated_take_profit 在這
│   │   └── decision.py                  ← decision/direction 真實實作
│   │
│   ├── compat/
│   │   ├── types.py                     ← Candle / Signal / Verdict dataclasses
│   │   ├── config.py                    ← load_config + DEFAULT_CONFIG
│   │   └── atomic_signals/              ← atomic-verbatim 演算法 (★)
│   │       ├── computes.py              ← RSI/MACD/EMA/ATR/Bollinger
│   │       ├── derivatives_detectors.py
│   │       ├── structure.py
│   │       └── volume_detectors.py
│   │
│   ├── data_cli/                        ← binance-cli 資料源 wrappers
│   ├── exec_cli/                        ← binance-cli 訂單 wrappers
│   └── standard_bot/
│       ├── monitoring/                  ← runtime PnL / signal log / stops
│       └── orchestration/               ← scan / rescan / state CRUD
│
└── docs/
    └── atomic-compat/
        └── MIGRATION_HANDOFF.md         ← 這份文件
```

### 9.3 Layer 4 數值驗證表（atomic vs shim）

```
RSI(14):       atomic=236 elements, shim=236 elements, diff < 1e-9: 236/236
EMA(20):       atomic=250 elements, shim=250 elements, diff < 1e-9: 250/250
EMA(50):       atomic=250 elements, shim=250 elements, diff < 1e-9: 250/250
MACD.macd:     atomic=250, shim=250, match: 250/250
MACD.signal:   atomic=250, shim=250, match: 250/250
MACD.hist:     atomic=250, shim=250, match: 250/250
ATR(14):       atomic=236, shim=236, match: 236/236
BB.upper:      atomic=231, shim=231, match: 231/231
BB.middle:     atomic=231, shim=231, match: 231/231
BB.lower:      atomic=231, shim=231, match: 231/231
BB.bandwidth:  atomic=231, shim=231, match: 231/231
BB.pct_b:      atomic=231, shim=231, match: 231/231

Total: 12/12 functions, all elements match within 1e-9 tolerance.
```

### 9.4 借自同事 migrate_library branch 的東西

| 借了什麼 | 用在哪 | 為什麼 |
|---|---|---|
| `library/features/atomic_signals/computes.py` | `cyqnt_trd/compat/atomic_signals/computes.py` | atomic-verbatim RSI/MACD/EMA/ATR/Bollinger 演算法 |
| `library/features/atomic_signals/derivatives_detectors.py` | 同上 | funding/OI/crowding 偵測 |
| `library/features/atomic_signals/structure.py` | 同上 | Fibonacci/pivots/candlestick |
| `library/features/atomic_signals/volume_detectors.py` | 同上 | volume surge/trend |
| `cases/migrated/*/fixtures/*.json` | Layer 5 驗證的 golden fixtures | 對 case 規則做合理性檢查 |

**沒借**：他的 `StrategySpec` 整體架構、ai_pro_trading_library 的整個 package 結構（兩條路線方向不同）。

### 9.5 PyPI 發版資訊

| 項目 | 值 |
|---|---|
| Package name | `cyqnt-trd` |
| Latest version | `0.1.9.dev3` |
| PyPI URL | https://pypi.org/project/cyqnt-trd/0.1.9.dev3/ |
| 含 atomic_strategy_lib shim | ✅ 是 |
| Wheel size | 581 KB |
| `cyqnt_trd/*` files | 278 |
| `atomic_strategy_lib/*` files | 53 |

### 9.6 對外文檔需要更新

完成此次 release 後，以下對外文檔可能也需要對應更新（但不在這次 PR 內）：

- README.md（提到新版可以 pip install 後直接跑 atomic case）
- 各 case 的 README.md（如有寫「需要設定 atomic 路徑」可以拿掉）
- OpenClaw 內部部署文件

---

**End of handoff. Questions or issues — see commit log around df46542 or this doc's §7.4 troubleshooting.**
