# Trading Modes

這份 skill 的交易規則與過去版本不同：

- 不使用 testnet
- OpenClaw 的真實交易，必須以 `paper trade` 產生的信號為來源
- 真實交易不是由 repo 直接下單，而是由 OpenClaw 的交易能力執行
- 但 OpenClaw 下單前，必須先經過 paper signal、餘額確認、`CONFIRM`、時長確認這四個步驟

## 1. 模式 1：只回測

- 使用 `mvp_backtest`
- 不送單
- 只產出回測結果與 JSON

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine python \
  --strategy <strategy_id> \
  --strategy-module strategies.<strategy_module> \
  --symbol BTCUSDT --interval 1h \
  --market-type futures \
  --initial-capital 10000 \
  --commission-bps 4 --slippage-bps 2 \
  --execution-model next_bar_open \
  --historical-dir data/mtf_90d --storage-timeframe 1m --limit 2000
```

## 2. 模式 2：只產生 paper signal（Paper Trade Daemon）

- 使用 `mvp_paper_daemon`（長駐 daemon，持續監聽 Binance REST）
- 不對外下單
- 產出 `state.json`（即時狀態）和 `trades.jsonl`（所有 paper fills）

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \
  --engine python \
  --strategy <strategy_id> \
  --strategy-module strategies.<strategy_module> \
  --symbol BTCUSDT --interval 1h \
  --market-type futures \
  --state-dir ./watcher/<STRATEGY_ID>_BTCUSDT_1h \
  --poll-interval 3570 \
  --warm-up-bars 80 \
  --initial-capital 10000 \
  --fee-bps 4 --slippage-bps 2
```

關鍵輸出：
- `state.json` — 即時持倉、equity、最新 signal
- `trades.jsonl` — 每筆 paper fill（含 `action` 欄位：`open_long`、`flip_to_short` 等）

停止方式：
- `Ctrl+C`
- 或寫入 `state.json` 的 `status` 為 `"stopped"`

## 3. 模式 3：paper signal 驅動的真實交易（Live Trade）

這是 OpenClaw 目前應採用的真實交易模式。

### 技術架構

```
┌─────────────────────────────────────────────────────────┐
│ mvp_paper_daemon (signal source, 與 paper mode 完全相同)  │
│ ├─ PythonLivePaperSession                               │
│ ├─ make_signals(df) — 策略邏輯                           │
│ └─ 寫 trades.jsonl                                      │
└───────────────────────────┬─────────────────────────────┘
                            │ trades.jsonl (action 欄位)
                            ▼
┌─────────────────────────────────────────────────────────┐
│ mvp_live_executor (真實下單)                              │
│ ├─ 讀 trades.jsonl 偵測新 fill                           │
│ ├─ 根據 action 方向 → binance-cli 下單                   │
│ ├─ 支援 flip_to_long / flip_to_short（拆成 close + open）│
│ ├─ 每次下單前 reconcile 真實倉位                         │
│ ├─ 失敗自動 retry（exponential backoff, 最多 3 次）       │
│ └─ 寫 executions.jsonl（audit trail）                    │
└─────────────────────────────────────────────────────────┘
```

### 啟動流程

**Step 1：啟動 paper daemon（訊號來源）**

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \
  --engine python \
  --strategy <strategy_id> \
  --strategy-module strategies.<strategy_module> \
  --symbol BTCUSDT --interval 1h \
  --market-type futures \
  --state-dir ./watcher/<STRATEGY_ID>_BTCUSDT_1h \
  --poll-interval 3570 \
  --warm-up-bars 80 \
  --initial-capital 10000 \
  --fee-bps 4 --slippage-bps 2
```

**Step 2：啟動 live executor（真實下單）— 先 dry-run 驗證**

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor \
  --state-dir ./watcher/<STRATEGY_ID>_BTCUSDT_1h \
  --symbol BTCUSDT \
  --max-notional 200 \
  --notional-fraction 0.95 \
  --dry-run
```

**Step 3：確認 dry-run 輸出正確後，移除 `--dry-run` 正式啟動**

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor \
  --state-dir ./watcher/<STRATEGY_ID>_BTCUSDT_1h \
  --symbol BTCUSDT \
  --max-notional 200 \
  --notional-fraction 0.95
```

### 訊號一致性保證

三種模式都呼叫**完全相同**的 `make_signals(df)` 函式：

| 模式 | Signal 來源 | 執行層 |
|------|------------|--------|
| Backtest | `SnapshotBacktestRunner` → `BlockStrategyPlugin.run()` → `make_signals(df)` | 模擬 fill（next_bar_open） |
| Paper | `PythonLivePaperSession.tick()` → `_compute_latest_target()` → `make_signals(df)` | 模擬 fill（next_bar_open） |
| Live | 同 Paper daemon → `trades.jsonl` → `BinanceCliExecutor` | `binance-cli` 真實下單 |

### Live Executor 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--state-dir` | (必填) | Paper daemon 的 state 目錄（含 trades.jsonl） |
| `--symbol` | BTCUSDT | 交易對 |
| `--max-notional` | 200 | 單筆最大下單金額 USDT |
| `--notional-fraction` | 0.95 | 下單金額佔可用餘額的比例 |
| `--dry-run` | false | 只印出 binance-cli 指令不執行 |
| `--max-retries` | 3 | 下單失敗重試次數 |
| `--poll-interval` | 5 | 輪詢 trades.jsonl 間隔（秒） |

### 緊急停止

```bash
# 方法 1：建立 kill switch 檔案（executor 會取消所有掛單並退出）
touch ./watcher/<STRATEGY_ID>_BTCUSDT_1h/EMERGENCY_STOP

# 方法 2：Ctrl+C 中斷 executor process

# 方法 3：停止 daemon（executor 偵測到 daemon stopped 後自動退出）
# 寫入 state.json：{"status": "stopped"}
```

### Live Executor 支援的 action 類型

Paper daemon 的 `PythonLivePaperSession` 會產生以下 action，executor 全部支援：

| action | binance-cli 操作 |
|--------|-----------------|
| `open_long` | `BUY MARKET` |
| `open_short` | `SELL MARKET` |
| `close_long` | `SELL MARKET --reduce-only true` |
| `close_short` | `BUY MARKET --reduce-only true` |
| `flip_to_long` | `close_short` + 驗證 + `open_long`（兩步） |
| `flip_to_short` | `close_long` + 驗證 + `open_short`（兩步） |

### 核心程式碼位置

| 元件 | 路徑 |
|------|------|
| Live Executor class | `cyqnt_trd/standard_bot/execution/cli_executor.py` |
| Live Executor CLI | `cyqnt_trd/standard_bot/entrypoints/mvp_live_executor.py` |
| Paper Daemon | `cyqnt_trd/standard_bot/entrypoints/mvp_paper_daemon.py` |
| Paper Session engine | `cyqnt_trd/standard_bot/simulation/python_live_paper_session.py` |
| Strategy registration | `cyqnt_trd/blocks/strategy.py` |

### binance-cli 指令語法參考

Executor 使用的 binance-cli 語法（來自 `binance-cli-trading-reference.md`）：

```bash
# 市價開多
binance-cli futures-usds new-order \
  --symbol BTCUSDT --side BUY --type MARKET --quantity 0.002

# 市價開空
binance-cli futures-usds new-order \
  --symbol BTCUSDT --side SELL --type MARKET --quantity 0.002

# 平多倉（reduce-only）
binance-cli futures-usds new-order \
  --symbol BTCUSDT --side SELL --type MARKET \
  --quantity 0.002 --reduce-only true

# 查帳戶餘額
binance-cli futures-usds futures-account-balance-v3

# 查持倉
binance-cli futures-usds position-information-v3 --symbol BTCUSDT

# 查即時價格
binance-cli futures-usds symbol-price-ticker --symbol BTCUSDT

# 取消所有掛單
binance-cli futures-usds cancel-all-open-orders --symbol BTCUSDT
```

## 4. 使用者交易前必須看到的資訊

OpenClaw 在任何真實交易前，必須展示：

- 最新 paper signal：`buy / sell / hold`
- signal 產生時間
- 監聽標的
- 目前價格
- 使用者餘額
- 預計使用的交易金額或數量
- 風險管控方式
- 預計執行時長
- 如何停止

## 5. `CONFIRM` 規則

若未收到 `CONFIRM`：

- 可以回測
- 可以生成 signal
- 可以啟動 paper 監聽
- 不可下真實單

若收到 `CONFIRM`，仍必須先滿足：

- 使用者已看到當前餘額
- 使用者已知道交易標的、方向、金額/數量
- 使用者已看到本次 session 的風險管控方式
- 使用者已給定執行時長

## 6. 時長規則

真實交易 session 必須是有時長限制的。

OpenClaw 不可在未經說明下啟動無限期交易。若使用者沒有給時長，必須先追問，例如：

- `10 分鐘`
- `1 小時`
- `到我手動停止為止`

若使用者選擇「到我手動停止為止」，仍需明確告知 `停止` 指令。
