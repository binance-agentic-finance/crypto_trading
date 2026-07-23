# Atomic Compat Shim 修正報告

**日期**: 2026-05-29  
**修正者**: Hank  
**Repo**: `crypto_trading-main` (commit `f58d29a`, branch `main`)  
**驗證環境**: `.venv-standard-bot` (Python 3.11, editable install)

---

## 問題描述

同事回報 `atomic_strategy_lib` 的 data shim 層壞掉，具體例子：

```python
# atomic_strategy_lib/data/open_interest.py 的 shim：
def open_interest_fetch(symbol, ...):
    df = fetch_open_interest(symbol)   # 調 cyqnt_trd.data_cli
    if df is None or df.empty:         # ← 假設返回 DataFrame
        ...
    row = df.iloc[0]                   # ← 用 DataFrame 方式取值
```

但實際上 `cyqnt_trd.data_cli.fetch_open_interest()` 返回的是 **dict**，不是 DataFrame。  
同理 `fetch_funding_rate()` 返回 **float**，不是 DataFrame。

導致所有使用 funding / open_interest 的 usage cases 直接崩潰：

```
AttributeError: 'dict' object has no attribute 'empty'
AttributeError: 'float' object has no attribute 'empty'
```

---

## 根因分析

| 函式 | shim 假設回傳 | 實際回傳 | 結果 |
|---|---|---|---|
| `fetch_open_interest(symbol)` | DataFrame | **dict** | `.empty` 爆炸 |
| `fetch_funding_rate(symbol)` | DataFrame | **float** | `.empty` 爆炸 |
| `fetch_oi_history(symbol, interval=...)` | — | 參數叫 **period** 不是 interval | `TypeError` |
| `scan_with_filter(..., profile=...)` | — | native 不接受 `profile` | `TypeError` |
| `scan_with_filter()` 回傳 | list[dict] | **DataFrame** | case 迭代爆炸 |

**本質**：shim compatibility layer 對 `cyqnt_trd.data_cli` 的回傳型別與函式簽名假設錯誤。

---

## 修正內容

### 修正原則
- **不動 `cyqnt_trd.data_cli` native API**（它的設計是合理的）
- **不動 `blocks` 層**
- **只改 `atomic_compat/` 的 compatibility translation layer**

### 修改的檔案

| 檔案 | 修了什麼 |
|---|---|
| `atomic_compat/atomic_strategy_lib/data/funding.py` | `funding_rate_current()`: float → `FundingRate` dataclass；`funding_rate_fetch()`: 改走 `fetch_funding_history()` → `list[FundingRate]` |
| `atomic_compat/atomic_strategy_lib/data/open_interest.py` | `open_interest_fetch()`: dict → `OpenInterest` dataclass；`oi_history_fetch()`: 修正參數名 `interval` → `period`；`oi_delta_pct()`: 支援雙調用模式 |
| `atomic_compat/atomic_strategy_lib/data/scanner.py` | 加 `profile`/`binary` kwargs 忽略；`scan_with_filter()` 回傳從 DataFrame 轉成 `list[dict]` |

### 型別轉換對照

```
cyqnt_trd.data_cli                    atomic_compat shim output
─────────────────                    ────────────────────────────
fetch_open_interest() → dict    →    OpenInterest dataclass
fetch_funding_rate() → float    →    FundingRate dataclass
fetch_oi_history() → DataFrame  →    list[OIHistoryPoint]
fetch_funding_history() → DataFrame → list[FundingRate]
scan_with_filter() → DataFrame  →    list[dict]
```

---

## 真實資料驗證

用 **真實 binance-cli** 抓取 live data 驗證（不是模擬）：

```python
>>> from atomic_compat.atomic_strategy_lib.data.funding import funding_rate_current
>>> funding_rate_current("BTCUSDT")
FundingRate(symbol='BTCUSDT', rate=0.0001, timestamp=1780012800000, mark_price=72906.82, ...)

>>> from atomic_compat.atomic_strategy_lib.data.open_interest import open_interest_fetch
>>> open_interest_fetch("BTCUSDT")
OpenInterest(symbol='BTCUSDT', oi_base=106338.681, oi_value=7824889709.45, timestamp=...)

>>> from atomic_compat.atomic_strategy_lib.data.scanner import scan_with_filter
>>> scan_with_filter(market="futures", min_volume=5_000_000, profile="prod")
[{'symbol': 'BTCUSDT', 'volume_quote': 12926467942.2}, {'symbol': 'ETHUSDT', ...}, ...]
# 259 instruments returned
```

---

## 全量 Smoke Test 結果

對 `usage_project_cases/` 全部 **44 個 case** 執行三層驗證：

1. **Layer 1** — `py_compile`：語法正確
2. **Layer 2** — import/exec：所有 `from atomic_strategy_lib.*` 能解析
3. **Layer 3** — `--help`：entrypoint 可正常啟動

### 結果

| | 數量 |
|---|---:|
| ✅ PASS | **44** |
| ❌ FAIL | **0** |
| ⏭️ SKIP | 1 (`_templates`，模板目錄) |

### 完整清單

| # | Case | Status |
|---|---|---|
| 1 | ahr999-btc-hodl | ✅ |
| 2 | bb-squeeze-momentum | ✅ |
| 3 | bitfence | ✅ |
| 4 | black-swan-insurance | ✅ |
| 5 | btc-grid-buy-okb | ✅ |
| 6 | btc-multi-factor-trend | ✅ |
| 7 | btc-usdt-swap-defensive-ai | ✅ |
| 8 | case_smart_momentum | ✅ |
| 9 | crypto-swing-signal-analyst | ✅ |
| 10 | dca-bot-parameterizer | ✅ |
| 11 | dcd-auto-trader | ✅ |
| 12 | dual-signal-analyzer | ✅ |
| 13 | earnings-tradfi-scanner | ✅ |
| 14 | funding-rate-scanner | ✅ |
| 15 | golden-ratio-fibonacci | ✅ |
| 16 | kline-indicator-checkup | ✅ |
| 17 | less-is-more | ✅ |
| 18 | macro-event-driven-scanner | ✅ |
| 19 | maker-entry | ✅ |
| 20 | market-intel | ✅ |
| 21 | neurogrid-supertrend-compound | ✅ |
| 22 | ondo-daily | ✅ |
| 23 | pair-spread | ✅ |
| 24 | popular_trading_bot | ✅ |
| 25 | position-sizer | ✅ |
| 26 | prediction_market | ✅ |
| 27 | quadruple-filter-breakout | ✅ |
| 28 | resilience-trader | ✅ |
| 29 | rolling-position | ✅ |
| 30 | rsi-mean-reversion | ✅ |
| 31 | small-account-600u-pnl-percent-v1 | ✅ |
| 32 | sol-rebound-short | ✅ |
| 33 | speed-hunter | ✅ |
| 34 | spot-momentum-scan-validate | ✅ |
| 35 | square-buzz-screener | ✅ |
| 36 | stochrsi-mdi-trend-v1 | ✅ |
| 37 | structure-breakout-priceaction | ✅ |
| 38 | structured-trade-plan | ✅ |
| 39 | three-tier-strategy | ✅ |
| 40 | trend-grid-bot | ✅ |
| 41 | trendline-symmetry-breakout | ✅ |
| 42 | volume-surge-daily-report | ✅ |
| 43 | web3-daily | ✅ |
| 44 | whale-tracker | ✅ |

---

## 如何在你的環境重現

```bash
cd /path/to/crypto_trading-main
git pull origin main          # 拉最新 commit f58d29a
pip install -e . --no-deps    # editable install 到你的 venv

# 驗證
python -c "
from atomic_strategy_lib.data.funding import funding_rate_current
from atomic_strategy_lib.data.open_interest import open_interest_fetch
print(funding_rate_current('BTCUSDT'))
print(open_interest_fetch('BTCUSDT'))
"
```

---

## 注意事項

- 修正只影響 `atomic_compat/` 層，**不影響 `cyqnt_trd` 核心 API**
- 如果你的 venv 裡有舊的 `cyqnt-trd` wheel（非 editable），需要先 `pip uninstall cyqnt-trd` 再重裝
- `scan_with_filter()` 現在回傳 `list[dict]`，如果你有 native caller 直接用 `cyqnt_trd.data_cli.scan_with_filter()` 它仍然回 DataFrame（不受影響）
