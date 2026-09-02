# Changelog

本文件記錄 cyqnt_trd 持續整合與功能擴充的時序內容。
編號採日期格式（YYYY-MM-DD）。發版條目的標題會帶上 PyPI 版本號。
版本號的真相來源是 `pyproject.toml` 與 `cyqnt_trd.__version__`（兩者由
`tests/test_version_is_single_sourced.py` 釘住一致）。
這個 repo 目前**不打 git tag**，所以不要去那裡找版本。

---

## 2026-09-02 — PyPI 0.1.15:兩個靜默錯答、選幣市值篩選,與架構設計文件

### 摘要

累積 PR #19 與 #20。兩個修正的共同點是**沒有例外、輸出看起來合理**,所以只能靠人盯著才會
發現 —— 這是這個 repo 最常見的一類缺陷。

此版本對應 `pyproject.toml` → `version = "0.1.15"`。

### ⚠️ 行為變化:`sizing.compute_stop_price` 的 `direction`

原本用裸比較 `direction == "LONG"`,所以傳 `"long"`(小寫)會掉進 short 分支,
**做多的停損被放到進場價上方**。現在改為 `.upper()`,與 `stop_loss.atr_dynamic_stop` /
`fixed_pct_stop` / `limits` 一致。

**升版前請確認**:如果你的程式已經在這個錯答案上疊了補償邏輯(例如自己把方向再翻回來),
升到 0.1.15 之後那層補償會變成新的錯誤。傳大寫的呼叫方不受影響。

這個函式原本零測試,所以壞了沒人發現;現在有 13 條測試釘住「每種拼法停損都落在進場價的
正確一側」以及「兩個算 ATR 停損的函式對同一組輸入給同一個價格」。

### 修正:清算累積檔讀不到時會蓋掉歷史

`standard_bot/data/liquidations.py` 的 `_read_existing_aggregate` 對「檔案不存在」和
「檔案在但讀不到」都回 `None`,呼叫端把 `None` 當成沒有歷史,接著 `os.replace` 換掉唯一
那一份。實測:存好 2 列 → 檔案損壞 → 下一次擷取後只剩 1 列,而且輸出裡沒有任何跡象。

`!forceOrder` 上游不歸檔,刪掉就拿不回來。現在讀不到的檔案會先改名到
`<name>.unreadable-<ms>` 再寫新的,並發 `RuntimeWarning` —— 壞掉的位元組留在磁碟上可以
搶救,這次擷取照樣落地。連移不動就讓錯誤往上拋。

同時把兩個實作上有、docstring 沒寫的取捨補上:合併時取 max 對**部分重疊的觀測會低估**
(一次看到 5 筆、另一次看到 7 筆,真值 12,存的是 7 —— 計數是下界不是普查),
以及**平手時後傳進來的贏**。

### 新增:選幣宇宙的市值篩選(#19)

`universe.augment_with_market_cap` —— 撿回交易所本來就在回的流通量,解除 GAP-MARKET-CAP。
同時修掉三個「dry-run 會過、真正跑必失敗」的整合缺口。

**注意步驟順序**:這個 block 屬於「沒有全市場端點、只能逐幣抓」的那五個之一,排在收窄步驟
之前會直接超出額度;而且合約端有一批非 COIN 標的的流通量回 0,所以它必須排在
`filter_crypto_only` 之後。`validate` 會靜態擋下錯誤順序。
具體檔數與實測呼叫次數見 `AGENTS.md` 的「選幣層」一節。

### 文件

新增 `docs/ARCHITECTURE.md` —— 系統邊界、訊號生命週期、四種 `kind`、能力邊界(用數字說)、
頂層模組地圖與引用次數、`standard_bot` 12 個子系統、同一件事的多個實作哪一個活著、
兩個不變量、已知缺口。`README.md` 與 `AGENTS.md` 都指向它。

新增 `docs/count_architecture_facts.py` —— 起因是發現四份文件對「blocks 有幾個積木」給出
四個不同答案(146 / 310 / 365 / 393),四個都不是造假,它們用了四種計法而沒有一份文件寫出
自己用哪一種。這支腳本重算每個數字**並印出它的定義**,`--diff` 在文件漂走時 exit 1。

順帶修掉三處硬編碼在 `docs/gen_report_html.py` 及其產物裡的公司內網文件連結
(這是公開 repo),並把漏掉的域名補進 `tests/test_no_leaked_secrets.py` 的守門清單。

### 驗證

`2505 passed, 2 skipped`(乾淨 Python 3.13 環境)。前一版基線 2489,新增 16 條測試。

---

## 2026-08-11 — PyPI 0.1.14:支援 Python 3.13

`numpy` 依賴按 Python 版本分岔(`<3.13` 用 `>=1.24,<2.0`;`>=3.13` 用 `>=2.1,<3.0`),
因為 numpy 1.x 沒有 3.13 的 wheel。新增 `tests/test_supported_python_matrix.py` 釘住
`requires-python`、依賴分岔與工具設定的 target 三者一致。

同時補齊測試自己要用的兩個依賴(`pyyaml`、`jsonschema`)—— 原本沒宣告,在乾淨環境裡
`pip install -e ".[dev]"` 之後跑 pytest 會有 11 個模組 collection error,看起來像
「這個 Python 版本不支援」,其實只是缺測試依賴。

---

## 2026-08-10 — PyPI 0.1.13:根 `__init__` 改 PEP 562 懶加載

原本根 `__init__` 用一個 eager 迴圈把全部子套件匯入(失敗就吞進 `_IMPORT_ERRORS`),代價是
**任何** `import cyqnt_trd.*` 都會連帶拉進 `get_data` / `trading_signal` / `backtesting` /
`standard_bot` 的重依賴 —— binance SDK / requests / aiohttp / scipy / matplotlib。
下游的純計算消費者為此得掛檔案系統墊片繞過。改成 PEP 562 `__getattr__` 懶加載後,
只在真正取用時才匯入。新增 `tests/test_root_import_is_lazy.py`(264 行)守住這件事。

---

## 2026-08-03 — PyPI 0.1.12:版本號單一真相來源

`cyqnt_trd.__version__` 寫著 `0.1.9.dev2` 而 `pyproject.toml` 寫 `0.1.11` —— **漂了四個
發版**,而且沒有任何東西比對過兩者。消費端做 `cyqnt_trd.__version__` 拿到的是一個好幾個月
前就不存在於 PyPI 的號碼,那比沒有號碼更糟:它看起來很權威。
新增 `tests/test_version_is_single_sourced.py` 斷言兩處一致。

這一版累積了 47 個 commit(自 0.1.11 起),主要是 YAML pipeline、選幣層與 NL→YAML 那條線;
其中新聞資料層的細節見上面 2026-07-23 的條目。

---

## 2026-07-23 — 新聞資料層：PUBLIC Binance Square 兩層接入（data_cli + lookahead-safe 特徵）

### 摘要

把 PUBLIC Binance Square 新聞／社群資料以「兩層」接進 cyqnt_trd，完全比照現有
`_htf_*` 欄位注入機制，但**只做資料層**，尚未接進 spec/strategy。

1. **資料層** `cyqnt_trd/data_cli/news.py`：`fetch_news / fetch_sentiment /
   fetch_ticker_rank / fetch_topic_trending / fetch_hot_post` → 回傳快取 typed
   DataFrame（形狀比照 `kline.py`）。envelope `code=='000000'` 且 `data==None`
   視為 cache-miss，回「空的 typed DataFrame」而非拋例外。
2. **Vendored PUBLIC client** `cyqnt_trd/data_cli/_vendor/binance_bigdata_client.py`：
   純 stdlib（urllib）、`env='prod'`、`min_interval` 節流、帶 provenance 標頭；
   **只含 7 個 PUBLIC Square 方法**，完全排除 `TradingInsightClient` 與所有
   內網主機上的方法。支援 `CYQNT_BIGDATA_API_PATH` 覆寫改 import 上游。
   不引入任何新 runtime 依賴。
3. **Lookahead-safe 特徵層** `cyqnt_trd/blocks/news_feed.py`：
   `load_pit_index / build_pit_feature_frame / attach_news_features /
   ticker_rank_universe`。對齊一律用**可用性時間**（`captured_at_ms` /
   `capture_completed_at`），**絕不用內容時間**（`news.date` / `generatedAt`）；
   `idx = np.searchsorted(avail_ts, base_ts, 'right') - 1`；news id first-seen
   去重；warmup bar 比率類給 `NaN`、計數/旗標類給 `0.0`；欄位命名 `_news_*`。
4. **選標的** `cyqnt_trd/blocks/universe.py`：`augment_with_news / top_mentioned /
   top_bullish / filter_sentiment`（base ↔ `<BASE>USDT` join）。
5. **測試** `cyqnt_trd/tests/test_news_feed.py`（19 條），含「擾動未來擷取、斷言過去
   bar 不變」的 lookahead 洩漏測試，以及「gating 用可用性時間而非內容時間」的證明。

### 頭號風險與防護

lookahead 洩漏。防護：對齊只用可用性時間；`capture_completed_at` 預設（≥ 任一
per-endpoint 到達時間，最保守）；first-seen 去重確保同一 news id 只計一次；單元測試
直接擾動未來擷取並斷言過去 bar 輸出位元級不變。

---

## 2026-06-04 — PyPI 0.1.11：Live Trade Recovery 強化 + OpenClaw / Docker 驗證 + MA Cross Workspace

### 摘要

本次發版聚焦在兩件事：

1. **補強 live trade 安全恢復機制**，降低 Binance Demo / futures API 在 `502`、timeout、
   reduce-only reject 等情況下造成 paper/live 倉位漂移的風險。
2. **整理一套可直接給 OpenClaw / Binance AI Pro 使用的 MA Cross Strategy workspace**，
   讓策略能從 blocks 組成、回測、paper trade、watcher、到 `binance-cli` live trade
   做完整驗證。

此版本對應：

- `pyproject.toml` → `version = "0.1.11"`

### Live Trade：從一次性下單改為可恢復的 transition/reconcile 流程

`cyqnt_trd/standard_bot/execution/cli_executor.py`

原本的 live executor 對 `flip_to_long` / `flip_to_short` 採一次性執行：

1. 先平舊倉
2. 成功後再開反向新倉

這種做法在交易所端短暫失敗時會偏安全，但容易留下：

- paper 已翻倉
- live 還停在舊方向
- 後續沒有自動補完 transition

本次補上：

- **`live_executor_state.json`**：持久化未完成的 live transition 狀態
- **pending transition / reconcile loop**
  - 每輪先查真實帳戶方向
  - 再決定下一個原子動作（`close_long` / `open_short` 等）
  - 若未達 `desired_position`，下輪繼續補
- **target-direction based recovery**
  - 不再只假設「上一筆指令成功」
  - 改成持續把 `actual_position` 拉回 `target_direction`
- **drift-aware flip recovery**
  - 若 `close_*` 因 `502` / timeout 未完成，不會默默放棄
  - 若帳戶其實已經 flat，會直接補正確的 `open_*`

這讓 live executor 更接近「長時間自動交易的狀態機」，而不是單純的事件觸發器。

### Live Trade 安全性改善

這次的目標不是讓 executor 更激進，而是讓它在錯誤情況下**更保守但更能恢復一致性**：

- **不因平倉失敗就貿然開反向倉**
- **保留 pending 狀態，等待下一輪再次對帳**
- **避免 paper / live 因單次 API 異常長期失去同步**

這對廣大用戶尤其重要，因為 live trade 執行正確性比單次回測結果更關鍵。

### mvp_live_executor：前置檢查、恢復模式、預設值補強

`cyqnt_trd/standard_bot/entrypoints/mvp_live_executor.py`

入口層新增：

- **preflight checks**
  - `state.json` / `trades.jsonl` 是否存在
  - `binance-cli` 是否可用
  - futures 帳戶是否可讀
  - `max_notional` 是否會因 step size round 成 0
- **`--reconcile-only` 模式**
  - 只做 live transition 恢復
  - 不消費新的 paper fills
  - 方便在 OpenClaw / watcher / operator 發現 drift 後做補正
- **可調整的執行參數**
  - `--retry-base-sec`
  - `--heartbeat-interval`
  - `--max-reconcile-cycles`
- **更清楚的啟動摘要**
  - 顯示可用餘額、預估下單量、真實倉位、pending transition 狀態

這讓 `mvp_live_executor` 更適合作為正式長跑的 entrypoint，而不只是一次性測試工具。

### 新增測試：覆蓋 502 / drift recovery 與 preflight

新增測試：

- `tests/standard_bot/test_cli_executor_recovery.py`
- `tests/standard_bot/test_mvp_live_executor.py`

覆蓋場景包括：

- `flip_to_short` 遇到 `502` 後，pending transition 仍被保留
- 下輪先 `close_*` 再 `open_*`，完成 flip recovery
- 帳戶已 flat 時，不再重複 reduce-only close，直接補 `open_*`
- preflight 檢查缺檔、下單量 round-to-zero、reconcile-only 流程

### MA Cross Strategy Workspace：提供給 OpenClaw / Binance AI Pro 的完整範例

新增 workspace：

- `cyqnt_trd/standard_bot/ma_cross_strategy/`

內容包括：

- `strategies/ma_cross_v1.py`
- `strategies/ma_cross_validation_fast.py`
- `strategies/bar_direction_validation.py`
- `scripts/run_strategy.py`
- `scripts/run_paper_daemon.sh`
- `scripts/signal_executor.py`
- `scripts/session_watcher.py`
- `tests/test_strategy_composition.py`
- `README.md`

這套 workspace 的設計目標是：

- 直接讓 OpenClaw 用外部腳本調用 `cyqnt_trd`
- 不需要改套件核心即可完成驗證
- 支援 backtest / paper / live / watcher 全流程

### OpenClaw 驗證友善設計

為了讓 Docker OpenClaw / Binance AI Pro 更順利使用，這次的範例與入口腳本特別強調：

- **`cyqnt_trd` 視為 readonly 套件**
- **透過 `python -m cyqnt_trd...` 與外部 launcher 調用**
- **不依賴 `setup_env`**
- **live trade 直接走 `binance-cli`**
- **watcher 可以從 template 衍生 session runtime，回報 fills / risk / stop**

### MA Cross 驗證預設：ETHUSDT + 1m

為了讓 paper / live / watcher 驗證在短時間內更容易觀察到成交：

- 預設標的改為 `ETHUSDT`
- 預設策略週期改為 `1m`

這使得 OpenClaw 在 Docker 內測試時，更容易：

- 快速產生成交
- 檢查 paper/live 是否同源
- 驗證 watcher 是否能即時通知

### 版本定位

`0.1.11` 可以視為：

- `0.1.9.dev6 ~ dev7` 的 paper / exit / atomic compat 能力之上
- 補上更完整的 **live trade 安全恢復**
- 加入可供 OpenClaw 實際操作的 **workspace / launcher / watcher 範例**

也就是讓 `cyqnt_trd` 更接近：

**「可被 agent 安全調用、可長時間運行、可監控、可恢復」的交易框架。**

---

## 2026-06-02 — Live Trade Executor (binance-cli) + MA Cross Strategy 整合

### 摘要

新增 **BinanceCliExecutor** — 一個 strategy-agnostic 的 live trade 執行器，
讓任何透過 `cyqnt_trd.blocks.strategy.register()` 註冊的策略都能從 paper trade
直接進入 live trade。同時整合 MA cross strategy 作為第一個完整走通
backtest → paper → live 全路徑的範例策略。

### 新增：Live Executor

`cyqnt_trd/standard_bot/execution/cli_executor.py`：

- **BinanceCliExecutor class** — 監聽 paper daemon 的 `trades.jsonl`，將
  每筆 paper fill 的 `action` 翻譯成 `binance-cli futures-usds new-order` 真實下單
- 支援所有 action types：
  - `open_long` / `open_short` / `close_long` / `close_short`
  - `flip_to_long` / `flip_to_short`（拆成 close → verify → open 兩步安全執行）
- **Position reconciliation**：每次下單前查真實倉位，防止重複開倉或空平
- **Retry with exponential backoff**：失敗自動重試（base 2s, max 3 次）
- **Audit trail**：每筆 execution 寫入 `executions.jsonl`
- **Kill switch**：`touch EMERGENCY_STOP` → 取消所有掛單 + 退出
- **Heartbeat**：每 5 分鐘寫一次 alive log
- **Sizing 獨立於 paper**：用真實帳戶餘額計算下單量（`min(balance * fraction, max_notional) / price`）

`cyqnt_trd/standard_bot/entrypoints/mvp_live_executor.py`：

- CLI 入口，可以 `python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor` 呼叫
- 參數：`--state-dir`, `--symbol`, `--max-notional`, `--notional-fraction`, `--dry-run`, `--max-retries`, `--poll-interval`

### Live Trade 架構

```
mvp_paper_daemon (訊號來源，不改)
    │ writes trades.jsonl
    ▼
mvp_live_executor (真實下單)
    │ reads action → binance-cli futures-usds new-order
    ▼
Binance REST API → 撮合引擎
```

兩個 process 配對運行，訊號一致性由「三條路共用同一個 `make_signals(df)`」保證。

### 關鍵修正：flip action 支援

**問題**：MA cross 等 position-flip 策略的 paper daemon 產生 `flip_to_short` /
`flip_to_long` action（佔 ~97% 的交易），但舊版 signal_executor 不認識這些
action 直接 skip。

**驗證**：用 2000 根 1h bars 模擬 paper session，31 筆 fills 中 30 筆是 flip。

**修正**：BinanceCliExecutor 將 flip 拆成兩步：
1. 先平反向倉位（reduce-only）
2. 確認成功後再開新方向倉位
若第一步失敗則不執行第二步（防止裸露風險）。

### MA Cross Strategy 整合

- `strategies/ma_cross_v1.py`：SMA(20) / SMA(60) 金叉死叉策略
- Backtest 驗證：2000 snapshots, 16 trades (long-only), +3.37% → -1.8%
- Paper 驗證：2000 bars, 31 fills (bidirectional with flips)
- Live 驗證：E2E dry-run test 全部 action 正確翻譯成 binance-cli 指令

### 文件更新

- `README.md`：新增 Paper Trade Daemon 和 Live Trade 段落
- `references/trading-modes.md`：Mode 4 加入技術實作路徑

### 使用方式

```bash
# Backtest
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine python --strategy ma_cross_v1 --strategy-module strategies.ma_cross_v1 \
  --symbol BTCUSDT --interval 1h --market-type futures \
  --initial-capital 10000 --commission-bps 4 --slippage-bps 2

# Paper trade
python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \
  --engine python --strategy ma_cross_v1 --strategy-module strategies.ma_cross_v1 \
  --symbol BTCUSDT --interval 1h --market-type futures \
  --state-dir ./watcher/MA_CROSS_V1_BTCUSDT_1h

# Live trade (dry-run → real)
python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor \
  --state-dir ./watcher/MA_CROSS_V1_BTCUSDT_1h \
  --symbol BTCUSDT --max-notional 200 --dry-run
```

### 檔案變更清單

| 檔案 | 動作 |
|------|------|
| `cyqnt_trd/standard_bot/execution/cli_executor.py` | 新增 |
| `cyqnt_trd/standard_bot/entrypoints/mvp_live_executor.py` | 新增 |
| `cyqnt_trd/standard_bot/execution/__init__.py` | 修改（加入 export） |
| `strategies/__init__.py` | 新增 |
| `strategies/ma_cross_v1.py` | 新增 |
| `scripts/run_strategy.py` | 新增 |
| `README.md` | 修改 |
| `references/trading-modes.md` | 修改 |

---

## 2026-05-29 — Paper Trade Exit Management + Scoring/Sizing 擴充 + PyPI 0.1.9.dev6

### 摘要

把 Python engine 的 **paper trade daemon** 補上完整的 TP/SL/max_bars exit
management，讓 `strategy.register(exit_cfg={...})` 的 blocks 策略能在
`mvp_paper_daemon --engine python` 中觸發止盈止損。同步擴充 `scoring.py` /
`sizing.py` 模組，新增 8 個策略範例，修復 atomic data shim 相容性問題，
最終發版 **0.1.9.dev6** 到 PyPI。

### Paper Trade Exit Management

`cyqnt_trd/standard_bot/simulation/python_live_paper_session.py` (+148 lines)：

- 新增 `_position_exit_spec: Optional[Dict]` 狀態欄位
- 新增 `_position_entry_tick: int` 追蹤持倉起始 bar
- 新增 `_check_exit()` 方法：每根 bar 按優先級檢查
  - **SL**: `bar_low <= stop_loss_price` → 觸發止損
  - **TP**: `bar_high >= take_profit_price` → 觸發止盈
  - **max_bars**: `bars_held >= max_bars` → 超時平倉
- 新增 `_build_exit_spec()` 方法：開倉時從 `plugin.exit_cfg` 讀取配置，
  計算絕對價格（`fill_price * (1 ± pct)` 或 `fill_price ± ATR * mult`）
- 觸發後 queue `PendingOrder(target_position=0)` → next-bar-open 模型平倉
- 支援 `strategy.register(exit_cfg={...})` 的 5 種類型：
  - `pct_stop_tp`（百分比止損止盈）
  - `atr_stop_tp`（ATR 倍數止損止盈）
  - `time_only`（純時間止損）
  - `ma_cross_exit`（MA cross 平倉）
  - `opposite_signal`（反向訊號平倉）

Paper trade daemon 用法：

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \
  --engine python \
  --strategy my_strategy_v1 \
  --strategy-module workspace.my_strategy \
  --symbol BTCUSDT --interval 15m \
  --state-dir ./paper_runs/run1/
```

### Atomic Data Shim Fixes

`cyqnt_trd/compat/` 內的 atomic data shim 修正：

| 檔案 | 修正 |
|---|---|
| `funding.py` | `fetch_funding_rate()` 回傳 float（非 DataFrame）|
| `open_interest.py` | `fetch_open_interest()` 回傳 dict（非 DataFrame）|
| `open_interest.py` | `oi_history_fetch()` 參數名修正（period, 非 interval）|
| `scanner.py` | `scan_with_filter()` 回傳 `list[dict]`（atomic 相容）|
| 全部 shim | 接受 `profile=` / `binary=` kwargs 不 raise error |

### blocks/scoring.py 擴充（+281 lines）

擴充評分 combinators 與 gate functions，覆蓋更多 atomic 原版的用法模式。

### blocks/sizing.py 擴充（+372 lines）

擴充倉位管理工具，覆蓋更多 sizing 方案（Kelly / fixed-risk / ATR-inverse 等
在原有基礎上的延伸變體）。

### 新增 8 個策略範例

| 策略 | 行數 | 說明 |
|---|---|---|
| `channel_breakout.py` | 60 | Donchian 通道突破 |
| `ada_usdt_multi_tf.py` | 128 | 多時間框架 ADA 策略 |
| `consecutive_opposite_days.py` | 62 | 連續反向日反轉 |
| `ema_rsi_10pt.py` | 236 | EMA + RSI 含 10% TP/SL |
| `hermes_v12_1h_trend.py` | 75 | 1h 趨勢追蹤 |
| `lana_cron_takeprofit.py` | 240 | Cron 定時止盈策略 |
| `lana_style_momentum.py` | 118 | 動量風格策略 |
| `profitable_3indicators.py` | 120 | 三指標複合策略 |

### Python Engine Phase 1-2（commit f58d29a，同日稍早）

| 功能 | 說明 |
|---|---|
| Multi-TF HTF 自動附加 | `htf_specs` in `strategy.register()` 自動下載 + 附到 snapshot |
| Position sizing | `size` param 控制每次開倉大小 |
| Exit management | `exit_cfg` → `SnapshotBacktestRunner._check_exit()` |
| next_bar_open 執行模型 | `--execution-model` flag，signal bar close → next bar open 成交 |
| Full metrics | Sharpe, MaxDD, WinRate, avg_trade_pnl（via metrics_kernels）|
| 4h resample | `SUPPORTED_RESAMPLE_TIMEFRAMES` 加入 4h |

### Dependency 版本兼容性修復（dev5 遺留）

今天確認所有 13 個 dependency bounded ranges 仍正確：

```
pandas>=2.0.0,<3.0  numpy>=1.24.0,<2.0  polars>=1.0.0,<2.0
numba>=0.60.0,<0.70  pyarrow>=14.0.0,<25.0  matplotlib>=3.7.0,<4.0
scipy>=1.10.0,<2.0  requests>=2.32.0,<3.0  websockets>=15.0.1,<16.0
binance-sdk-spot>=8.2.1,<10.0  binance-sdk-usds-futures>=10.0.1,<11.0
binance-sdk-algo>=2.6.0,<3.0  binance-common>=3.8.0,<4.0
```

### PyPI Release

| 版本 | 內容 |
|---|---|
| **0.1.9.dev6** | 含以上所有改動；wheel 581 KB, 280 cyqnt_trd files + 53 atomic_strategy_lib shim files |

### SKILL.md 更新（dev_trading_bot + GHE）

同日也完成 skill 文件更新：

| 改動 | 位置 |
|---|---|
| dev3→dev5 版本號 (5 處) | SKILL.md + repo-bootstrap.md |
| SKILL.md 6 處加 python engine | description, Rule 10, Reference Map ×2, Workflow §3, Summary |
| `binance-cli spot get-account` 餘額快照規則 | SKILL.md (3 處) + risk-controls.md (section 1.1) |
| 新建 blocks-library.md (295 行) | 完整 blocks 模組清單 + 策略寫法 + 回測格式 |
| 新建 indicator-authoring-guide.md (317 行) | 新指標 inline 寫法 + 4 個必跑驗證 |
| backtest-workflow.md 加 python engine 段 | section 4.1 python engine 命令模板 |
| strategy-routing.md 加 blocks 段 | section 3 cyqnt_trd.blocks + python engine |
| repo-bootstrap.md 加 Available Library Modules | section 1.1 |
| YAML frontmatter 修復 | `## name:` → `name:` + `flow.metadata:` 拆行 + 加 `---` |

### Commits（2026-05-29）

| Hash | Subject | Remote |
|---|---|---|
| `f58d29a` | feat(engine+compat): Python engine Phase 1-2 + atomic data shim fixes | binance + origin |
| `0e2c2c6` | feat(engine): paper trade exit management + scoring/sizing expansion + 8 strategies | binance + origin |
| `4d3a085` | chore(release): bump cyqnt-trd 0.1.9.dev5 → 0.1.9.dev6 | binance + origin |
| `0195965` | docs(skill): mandate binance-cli spot get-account for balance snapshots | dev_trading_bot |
| `0025436` | docs(skill): mention python engine alongside numba in 6 places | dev_trading_bot |
| `9d244ae` | fix(skill): repair YAML frontmatter | dev_trading_bot |
| `b3a458b` | docs(skill): mandate binance-cli (GHE) | GHE feature branch |
| `76bcc25` | docs(skill): mention python engine (GHE) | GHE feature branch |
| `d88e898` | fix(skill): repair YAML frontmatter (GHE) | GHE feature branch |

### 整體統計

| 指標 | 數字 |
|---|---|
| Paper trade 新增 exit management | +148 lines |
| blocks/scoring 擴充 | +281 lines |
| blocks/sizing 擴充 | +372 lines |
| 新增策略範例 | 8 個 (+1,039 lines) |
| Atomic shim fixes | 5 個函數修正 |
| PyPI release | 0.1.9.dev6 |
| SKILL.md + references 更新 | ~800 lines across 7 files |
| Dependency 版本兼容 | 13 packages bounded, 0 upgrade on OpenClaw |

---

## 2026-05-28 — Official Python Engine（SnapshotBacktestRunner）補完 + 正式驗證

### 摘要

今天的核心工作不是新增指標，而是把 `standard_bot` 的 **official Python engine**
（`SnapshotBacktestRunner`）補到足以承接 blocks 策略的真實回測用途，讓後續
Binance AI Pro / OpenClaw 可以**直接調用官方 engine**，而不是再額外維護一套
自寫回測引擎。

整體目標：

1. 讓 `--engine python` 支援 **blocks 多時間框架（HTF）策略**
2. 讓 runner 支援 **position sizing / stop-loss / take-profit / max-bars exit**
3. 讓執行模型改成更貼近真實交易的 **`next_bar_open`**
4. 用官方 engine 重新驗證 strategy-lab 的 top20，確認哪些策略真的有效

同一天也完成了單策略 `r3_conservative_g04` 的官方 engine 驗證、圖表輸出、HTML/Markdown 報告與逐筆交易明細。

### 為什麼要做這個

原本 strategy-lab 的基因搜尋使用自寫的 `strategy_lab/backtest.py` 向量化回測器。
它的優點是很快，適合一次跑 600 個策略；但它不是 official engine，而且後來驗證發現：

- `ma_cross_exit` 在向量化版存在 **1-bar exit lookahead bug**
- 結果會高估 Sharpe / Return、低估回撤
- 因此未來 production / OpenClaw 的可信基準應該是 **official Python engine**，不是 strategy-lab 向量化器

結論：今天把 official Python engine 補完，並把它定為 blocks 策略未來的標準回測基礎。

---

### Phase 1 — Python engine 支援 Multi-TF / HTF Data

#### 1. `cyqnt_trd/blocks/strategy.py`

擴充 `strategy.register(...)`：

```python
strategy.register(
    "my_strategy",
    make_signals,
    htf_specs=[("4h", 200)],
)
```

新增能力：

- `htf_specs=[("4h", 200)]`：宣告策略需要 HTF 4h SMA(200)
- `BlockStrategyPlugin.needed_timeframes()`：回報該策略所需 HTF timeframes
- `_attach_htf_columns()`：在 signal-time 自動把 HTF bar 計算成 `_htf_4h_sma_200` 欄位，並對齊到 base TF DataFrame
- `get_block_plugin(strategy_id)`：讓 entrypoint 可以查詢某個 block strategy 的 HTF 需求

對齊邏輯採 **lookahead-safe**：

- 用 HTF bar 的 `close_time <= base bar open_time` 做 `searchsorted`
- 每根 1h bar 只會看到**已經收盤確認**的 4h bar，不會偷看到未來

#### 2. `cyqnt_trd/standard_bot/entrypoints/mvp_backtest.py`

改 `build_market_query()`：

- 原本只會讀 `args.interval`
- 現在若 strategy 有註冊 `htf_specs`，會自動把 HTF 加入 `MarketQuery.timeframes`

例如：

- base TF = `1h`
- strategy 需要 `4h`
- `MarketQuery.timeframes = ["1h", "4h"]`

#### 3. `cyqnt_trd/standard_bot/data/historical.py`

把：

```python
SUPPORTED_RESAMPLE_TIMEFRAMES = {"5m", "15m", "1h"}
```

擴為：

```python
SUPPORTED_RESAMPLE_TIMEFRAMES = {"5m", "15m", "1h", "4h"}
```

效果：local historical adapter 可以直接從 `1m.parquet` resample 出 `4h`。

#### 4. 驗證

- 既有 `smc_3confluence_v1` 的 baseline 結果 **完全不變**（diff = 0）
- 建立 `strategy_lab/strat_htf_test.py` 驗證：official Python engine 可成功讀取 `4h` 並 attach `_htf_4h_sma_200`
- 全套測試：**424 passed, 1 skipped**

---

### Phase 2 — Python engine 支援 ExitSpec / Sizing / next_bar_open

#### 1. `cyqnt_trd/blocks/strategy.py`

擴充註冊介面：

```python
strategy.register(
    "my_strategy",
    make_signals,
    htf_specs=[("4h", 200)],
    exit_cfg={"type": "ma_cross_exit", "period": 35, "ma_type": "ema", "max_bars": 160},
    size=0.25,
)
```

新增能力：

- `exit_cfg`：策略註冊時可直接宣告出場規則
- `size`：每筆單只動用部分資金（例如 `0.25` = 25% cash）
- `_compute_exit_spec()`：把 `exit_cfg` 轉成 runner 可執行的 `exit_spec`，塞入每個 entry signal payload

支援的 exit types：

- `time_only`
- `pct_stop_tp`
- `atr_stop_tp`
- `ma_cross_exit`
- `opposite_signal`

其中 `atr_stop_tp` 會在 plugin 端預先把 ATR 資訊轉成 `stop_loss_price / take_profit_price`，讓 runner 在 bar 層直接比 high/low。

#### 2. `cyqnt_trd/standard_bot/simulation/runner.py`

`SnapshotBacktestRunner` 進行核心補強：

##### (a) Metrics 補齊

整合 `metrics_kernels.compute_equity_statistics()`，讓 Python engine 現在和 Numba engine 一樣輸出：

- `sharpe_ratio`
- `max_drawdown`
- `mean_bar_return`
- `bar_return_volatility`
- `win_rate`
- `win_count`
- `avg_trade_pnl`
- `total_pnl`

##### (b) Sizing 支援

BUY signal 的 payload 若帶：

```python
{"size": 0.25}
```

則 runner 會用：

```python
target_notional = cash * 0.25
qty = target_notional / execution_price
```

##### (c) Exit management

runner 現在維護：

- `position_exit_spec`
- `position_entry_idx`
- 每根 bar 檢查：
  - `stop_loss_price`
  - `take_profit_price`
  - `max_bars`

##### (d) Execution model

`mvp_backtest.py` 新增：

```bash
--execution-model close_fill | next_bar_open
```

並把 `execution_model` 傳進 `request.extras`。

runner 支援兩種模式：

- `close_fill`（舊行為，legacy）
- `next_bar_open`（新行為，貼近真實交易）

`next_bar_open` 的流程：

1. bar T close 計算 signal
2. signal queue 起來，不立即成交
3. bar T+1 open 才 fill
4. SELL signal 也一樣延到下一根 bar open 平倉

這比之前 close-fill 更貼近「candle 收完後才下單、下一根開盤成交」的實際操作。

##### (e) 內建限制（目前已知）

仍未處理：

- funding rate
- volume-based liquidity cap
- short-side position management（目前 SELL = 平多，不是開空）
- intra-bar 同時 hit SL/TP 時的路徑依賴（目前採固定順序判斷，偏保守）

#### 3. 驗證

- `smc_3confluence_v1` baseline 在沒有 exit_cfg / size 的情況下，結果仍 **完全一致**
- 全套測試：**424 passed, 1 skipped**
- 說明：Phase 2 是 additive / backward-compatible，不破壞舊策略

---

### `r3_conservative_g04` 官方 Python Engine 實驗驗證

#### 1. 策略定義

檔案：

- `strategy_lab/strat_r3_g04_full.py`

策略內容：

- Long entry（四個條件全滿足）
  1. `close > _htf_4h_sma_200`
  2. `EMA(7)` 上穿 `EMA(18)`
  3. `Volume >= 1.3 × Volume MA(20)`
  4. `close > SMA(50)`
- Exit
  - `crossunder(close, EMA35)` 產生 SELL signal
  - `max_bars = 160`
- `size = 0.25`

#### 2. strategy-lab 向量化版的錯誤被正式確認

之前 strategy-lab 向量化回測器的 `ma_cross_exit` 實作存在：

- **同一根 bar 先看 close，再在 open 出場**
- 屬於 1-bar exit lookahead

這會高估 Sharpe / Return，低估 DD。

#### 3. Official engine（12 個月 IS、`next_bar_open`）結果

命令：

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine python \
  --strategy r3_g04_full \
  --strategy-module strategy_lab.strat_r3_g04_full \
  --symbol ETHUSDT --interval 1h \
  --start-ts 1735776000000 --end-ts 1767311940000 \
  --tail-bars 500 \
  --execution-model next_bar_open \
  --market-type futures --historical-dir strategy_lab/data \
  --initial-capital 10000 --commission-bps 4 --slippage-bps 2
```

結果：

| Metric | Official Python Engine（真實） |
|---|---|
| Sharpe | **1.9778** |
| Return | **+10.50%** |
| MaxDD | **-4.23%** |
| Trades | **24** |
| WinRate | **29.2%** |

#### 4. OOS（2026-01-02 ~ 2026-05-26）結果

| Metric | OOS |
|---|---|
| Sharpe | **1.2582** |
| Return | **+2.58%** |
| MaxDD | **-1.39%** |
| Trades | **14** |
| WinRate | **50.0%** |

#### 5. 結論

`r3_conservative_g04` 在 official Python engine 上：

- **IS 有效**（Sharpe ~1.98）
- **OOS 仍有效**（Sharpe ~1.26）
- 相較 strategy-lab 舊結果（Sharpe 3.6）明顯下降，證明舊引擎存在 optimistic bias
- 但策略本身不是無效，而是從「超漂亮」變成「仍然值得用、但現實一點」

這個結果很重要，因為它說明：

> 之後 blocks 策略若交由 Binance AI Pro / OpenClaw 自動生成，應該一律用 official Python engine 作為可信標準，而不是 strategy-lab 的舊向量化器。

---

### Official Python Engine 重跑 Top 20（IS → OOS）

今天也用 official engine（`next_bar_open`）重跑了 top20，並輸出：

- `strategy_lab/results/official_top20_is_oos.csv`
- `strategy_lab/results/equity_curves_official.json`
- `strategy_lab/results/report_python_engine.html`

#### 核心結果

| 指標 | 數值 |
|---|---|
| Top20 中 OOS Sharpe > 0 的策略數 | **13 / 20** |
| OOS 冠軍 | `r3_conservative_g04` |
| OOS 冠軍 Sharpe | **1.2582** |
| OOS 冠軍 Return（5 個月）| **+2.58%** |
| OOS 冠軍 MaxDD | **-1.39%** |

#### 觀察

- conservative 風格策略在 OOS 上最穩定
- 15m scalp 型策略在 official engine 下幾乎完全失效
- 之前 strategy-lab 報告中 OOS Sharpe 3.26 的結果是舊引擎偏差，不應再作為決策依據

---

### 單策略報告輸出

針對 `r3_conservative_g04` 額外生成：

- `strategy_lab/results/r3_g04_single_report.html`
- `strategy_lab/results/r3_g04_single_report.md`
- `strategy_lab/results/r3_g04_single_pnl.png`
- `strategy_lab/results/r3_g04_full_is.json`
- `strategy_lab/results/r3_g04_full_oos.json`

內容包含：

- 策略定義
- official Python engine 實際調用流程
- IS / OOS 指標
- PnL / Drawdown 圖
- 原始策略碼
- IS 24 筆逐筆交易明細
- OOS 14 筆逐筆交易明細

---

### 測試結果

今天所有改動完成後再跑完整測試：

| 測試集 | 結果 |
|---|---|
| `pytest tests/` | **424 passed, 1 skipped** |

也就是說：

- official Python engine 功能大幅增強
- blocks 策略現在可走官方回測路徑
- 既有功能完全沒被破壞

---

### 已知限制 / 下一步

#### 已完成

- Multi-TF / HTF attach
- size fraction
- stop/TP/max_bars exit management
- next_bar_open execution model
- metrics parity with Numba engine

#### 待做（若要更貼近實盤）

- funding rate 整合進 fee / pnl
- volume-based liquidity / partial fill 模擬
- short-side position management
- 同一 bar 內 stop / TP 先後順序的更精細路徑處理
- 用修正後的 official engine 重新做大規模策略搜尋（取代舊 strategy-lab 結果）

---

## 2026-05-27 — TradingView Indicators + SMC Wave A

### 摘要

擴充 cyqnt_trd 指標庫共 24 個函數：16 個 TradingView 主流指標
（兩個批次）+ 8 個 Smart Money Concepts (SMC) 元件。新增 4 個 Binance
真實資料 fixtures 用於 SMC 與一般 indicator 的整合測試。撰寫 SMC 範例
strategy 並用 mvp_backtest 的 python engine 跑通真實 backtest。

整體目標：讓 cyqnt_trd 的指標庫從「上一代傳統 TA」擴充到「現代量化
SMC + Volume + Trend」三大流派的同時覆蓋。

### Indicators 新增（24 個）

#### TradingView Batch 1 — 高優先（8 個）

新增於 `cyqnt_trd/blocks/indicators.py`：

| 函數 | TradingView 對應 | 說明 |
|---|---|---|
| `vwma(df, period=20)` | `ta.vwma` | 成交量加權移動平均 |
| `hma(series, period=20)` | `ta.hma` | Hull Moving Average，比 EMA 反應快 |
| `mfi(df, period=14)` | `ta.mfi` | Money Flow Index，類似 RSI 但加 volume |
| `cci(df, period=20)` | `ta.cci` | Commodity Channel Index |
| `williams_r(df, period=14)` | `ta.wpr` | Williams %R 動量震盪 |
| `keltner(df, period, atr_period, multiplier)` | `ta.keltner` | Keltner Channel（ATR-based 通道）|
| `heikin_ashi(df)` | `ta.heikinashi` | Heikin Ashi K 棒轉換 |
| `cmf(df, period=20)` | `ta.cmf` | Chaikin Money Flow |

#### TradingView Batch 2 — 中優先（8 個）

| 函數 | TradingView 對應 | 說明 |
|---|---|---|
| `tema(series, period=20)` | `ta.tema` | Triple Exponential MA |
| `dema(series, period=20)` | `ta.dema` | Double Exponential MA |
| `aroon(df, period=14)` | `ta.aroon` | Aroon Up/Down/Oscillator 三 Series |
| `trix(series, period=14)` | `ta.trix` | 三重 EMA smoothed ROC（basis points）|
| `awesome_oscillator(df)` | `ta.ao` | Bill Williams 5/34 SMA 差 |
| `pivot_points(df)` | `ta.pivot_point_levels` | Standard 7-level Floor pivots |
| `zigzag(series, deviation_pct=5.0)` | `ZigZag` | 百分比偏差 swing 偵測 |
| `pvt(df)` | `ta.pvt` | Price-Volume Trend 累計值 |

#### SMC Wave A — Smart Money Concepts（8 個）

新增於 `cyqnt_trd/blocks/smc_structure.py`（5 個）：

| 函數 | 說明 |
|---|---|
| `fractal_pivot_high(df, lookback=5)` | N-bar fractal pivot detection |
| `fractal_pivot_low(df, lookback=5)` | 同上對 low |
| `fair_value_gap(df)` | FVG 三 K 線跳空缺口偵測 |
| `order_block_detect(df, swing_lookback=5)` | Order Block（BOS 前最後反向 K 棒）|
| `bos_choch_detect(df, swing_lookback=5)` | Break of Structure / Change of Character 偵測 + trend state machine |

新增於 `cyqnt_trd/blocks/smc_liquidity.py`（3 個）：

| 函數 | 說明 |
|---|---|
| `liquidity_sweep_detect(df, swing_lookback=5)` | Stop hunt / liquidity sweep 偵測 |
| `equal_highs_lows(df, swing_lookback=5, tolerance_pct=0.1, max_pivots=20)` | EQH / EQL 流動性聚集區 |
| `premium_discount_zone(df, swing_lookback=5)` | 當前 swing range 切分 Premium / Discount / Equilibrium |

### Test Fixtures 新增

`tests/blocks/fixtures/`：

| 檔案 | 大小 | 期間 | 用途 |
|---|---|---|---|
| BTCUSDT_1h_500bars.parquet | 35K | 21 天 | 主測試 |
| ETHUSDT_1h_500bars.parquet | 35K | 21 天 | 第二 symbol 對照 |
| BTCUSDT_4h_300bars.parquet | 24K | 50 天 | High TF |
| BTCUSDT_15m_500bars.parquet | 35K | 5 天 | Low TF |

來源：binance-cli futures klines。

### Test Suites 新增

| 檔案 | 行數 | 測試數 |
|---|---|---|
| `tests/blocks/test_tradingview_indicators.py` | 533 | 49 |
| `tests/blocks/test_smc.py` | 424 | 29 |
| **合計** | 957 | **78** |![1779962808740](image/CHANGELOG/1779962808740.png)

整套測試結果：**414 passed, 1 skipped**（無 regression）。

### SMC 範例 Strategy

新增於 `cyqnt_trd/strategies/`：

- `smc_3confluence.py` — SMC 3-confluence 策略（sweep + structure + zone）
- `smc_5confluence.py` — SMC 5-confluence 策略（嚴格版）
- `mega_indicator_smoke.py` — Smoke test，使用全部 24 個新指標的單一策略

### Python Engine 整合驗證

執行真實 binance backtest：

```bash
mvp_backtest --engine python \
  --strategy smc_3confluence_v1 \
  --strategy-module cyqnt_trd.strategies.smc_3confluence \
  --symbol BTCUSDT --interval 1h --limit 1000
```

結果：

| Strategy | 期間 | Trades | Total Return |
|---|---|---|---|
| smc_3confluence_v1 | BTC 1h × 1000 bars (~42 天) | 2 | +1.27% |
| channel_breakout_v1（對照組）| 同上 | 2 | +0.84% |

樣本數小，無法下統計顯著性結論，但證明 python engine 完全支援 SMC strategy。

### Lookahead Bias 驗證

對全部 29 個指標跑「indicator(df[:i+1])[-1] == indicator(df)[i]」測試。

| 類別 | 結果 |
|---|---|
| 完全 lookahead-safe | 25/29 |
| Fractal pivot raw（high/low）| 有 N-bar confirmation lag（fractal 設計本質）|
| ZigZag | 最近 pivot 是 tentative（TradingView 原版亦如此）|

SMC 應用層 6 個函數（FVG、OB、BOS/CHoCH、Sweep、EQH/EQL、P/D Zone）
全部 lookahead-safe，可放心用於 backtest。

### BTCUSDT 1h 500 bars 真實 SMC 統計

| 元件 | 偵測量 |
|---|---|
| Fractal Pivot High | 28 |
| Fractal Pivot Low | 38 |
| Fair Value Gap | 101（48 BULL + 53 BEAR）|
| Order Block | 17（6 BULL + 11 BEAR）|
| BOS / CHoCH | 17 events |
| Liquidity Sweep | 49（29 BULL + 20 BEAR）|
| Equal Highs/Lows max cluster | 3 |
| 最後 zone | DISCOUNT |

### TradingView Scripts 涵蓋率分析

對 TradingView Community Scripts 頁面 21 個熱門 script 分類後：

| 類別 | 數量 | cyqnt_trd 已涵蓋 |
|---|---|---|
| Trend / MA | 8 | ~80%（缺 MDI、velocity pulse 等 composite）|
| Volume Profile / Order Flow | 4 | 25%（缺 anchored VWAP, footprint, TPO）|
| SMC / ICT | 3 | ~90%（缺 AI 學習層、TMF）|
| Momentum Oscillators | 2 | 100% |
| Volatility / Squeeze | 1 | 100% |
| Statistical | 1 | 80% |
| AI / ML | 1 | 0%（不建議做）|
| Other（S/R, ORB, Kinetic）| 4 | 50% |

整體覆蓋率：**~70%**。

### Implementation 細節

- 採用平行 sub-agent 架構：smc-structure-impl + smc-liquidity-impl 並行寫程式碼，smc-tester 寫測試
- 全部 sub-agent 用 claude-4-6-sonnet
- 之前同樣任務用 claude-4-7-opus + max reasoning 啟動的 team 卡 idle 7 小時無進展，
  改用 inline Agent tool + sonnet 後 2 個並行各約 5-10 分鐘完成
- TradingView batches 由 main agent 直接寫（風格一致性更好）

### Commits（2026-05-27）

| Hash（origin/main）| Subject |
|---|---|
| `bcd2089` | feat(indicators): add 8 high-priority TradingView-style indicators |
| `c02c5da` | feat(indicators): add 8 mid-priority TradingView indicators (batch 2) |
| `65092c7` | feat(blocks): add Wave A — Smart Money Concepts (SMC) indicators |

### 已知限制

- Fractal pivot raw 版本有 lookback bars 的 confirmation lag（fractal 本質）
- ZigZag 最近 pivot 是 tentative（隨後續 bar 重畫）
- SMC 5-confluence 在 1h × 500 bars 找不到完整 setup（嚴格版的真實表現）
- Volume Profile / Footprint / TPO 仍未支援
- Tick / orderbook backtest 引擎未實作

---

## 2026-05-26 — Atomic Strategy Lib → cyqnt_trd 整合

### 摘要

把另一個並行策略庫 `atomic_strategy_lib`（位於
`dev_trading_bot/must-read-intent-analysis-planning/references/`）整合進
cyqnt_trd，讓 44 個 user case 不修改任何程式碼也能在新版 cyqnt_trd 環境下
跑。透過命名相同的 shim package + atomic-verbatim 演算法兩層機制，達成
import 介面相容 + 數值一致。

PyPI 上線 cyqnt-trd 兩個版本：0.1.9.dev3（含 shim）→ 0.1.9.dev4（修
websockets 版本衝突）。

### 補完 atomic 核心 21 個函數

| 模組 | 補的內容 |
|---|---|
| `cyqnt_trd/blocks/verdicts.py` | 5 scoring combinators + 8 gates（hard_gate, enum_gate, soft_factor, verdict_classify, verdict_with_gate, cross_validate, conflict_detect, normalize_score）|
| `cyqnt_trd/blocks/limits.py` | 7 risk limits（liquidation, max_positions, max_exposure, daily_loss, price_deviation, circuit_breaker, funding_window）|
| `cyqnt_trd/blocks/stop_loss.py` | 4 stop helpers |
| `cyqnt_trd/blocks/exit.py` | 加 graduated_take_profit（atomic 的 stateful 三段式止盈）|

來源：同事的 `ai_pro_trading_library` migrate_library branch。

### Shim Package（51 個 thin re-export 檔）

新增於 `crypto_trading-main/atomic_compat/atomic_strategy_lib/`，覆蓋
atomic 全部 namespace（core, scoring, decision, risk, signals, data,
execution, monitoring, orchestration）。

每個 shim 檔 5-30 行，內容只有 `from cyqnt_trd.X import Y`。

關鍵設計：shim 取名跟原 atomic 完全一致為 `atomic_strategy_lib`。
Python 解析 import 時會把 shim 當成原 atomic，case 程式碼不需任何改動。

### Atomic-verbatim Signals

新增於 `cyqnt_trd/compat/atomic_signals/`：

| 檔案 | 行數 | 內容 |
|---|---|---|
| `computes.py` | 681 | RSI/EMA/MACD/ATR/Bollinger/StochRSI/SuperTrend/ADX |
| `derivatives_detectors.py` | 177 | funding/OI/crowding 偵測 |
| `structure.py` | 249 | Fibonacci/pivots/candlestick/box-range |
| `volume_detectors.py` | 95 | volume surge/trend |

採 atomic 純 Python verbatim 演算法（SMA-then-Wilder smoothing），
與 atomic 原版逐元素一致到 1e-9 精度內。第一版用 pandas RSI wrap 結果
與 atomic 平均差 34 點（warmup 期演算法不同），改為 verbatim port 後
完全對齊。

### Packaging 改造

`pyproject.toml` 改動：

```toml
[tool.setuptools]
package-dir = {"" = ".", "atomic_strategy_lib" = "atomic_compat/atomic_strategy_lib"}

[tool.setuptools.packages.find]
where = [".", "atomic_compat"]
include = ["cyqnt_trd*", "atomic_strategy_lib*"]
namespaces = false
```

效果：`pip install cyqnt-trd` 後 site-packages 同時得到
`cyqnt_trd/` + `atomic_strategy_lib/` 兩個 top-level package。

### PyPI Release

| 版本 | 日期 | 內容 | 問題 |
|---|---|---|---|
| 0.1.9.dev3 | 2026-05-26 | 首版含 atomic shim | `websockets==16.0` 與 binance-common 衝突 |
| 0.1.9.dev4 | 2026-05-26 | 修 websockets 約束 | — |

`websockets` 從 `==16.0` 放寬為 `>=15.0.1,<17`，與
binance-common / binance-sdk-* 的 `>=15.0.1,<16.0.0` 約束相容。

### Verification（4 層）

| Layer | 內容 | 結果 |
|---|---|---|
| Layer 1 | py_compile 全部 .py | 44/44 過 |
| Layer 2 | exec_module run_pipeline.py | 44/44 過 |
| Layer 3 | python run_pipeline.py --help | 44/44 過 |
| Layer 4 | atomic 真版 vs shim 數值對齊 | 12/12 函數，差異 < 1e-9 |
| Layer 5 | Case 級規則對 ai_pro golden | btc-multi-factor 0.954, rsi-mean-reversion 0.833 |
| Layer 6 | OpenClaw 環境完整模擬（無 atomic 真版 + 無 env var）| 44/44 過 |

### 文檔

新增於 `docs/atomic-compat/`：

- `MIGRATION_HANDOFF.md`（~7000 字）— 完整 handoff 指南：問題說明、設計理念、三層架構圖、為什麼 user cases 不用改、完整工作流程、四層驗證、OpenClaw 部署、維護指南
- `atomic_compat/README.md`（短版快速參考）

### 清理

- `dev_trading_bot/must-read-intent-analysis-planning/references/atomic_strategy_lib/`
  已 git rm + push 到 dev_trading_bot/dev branch（commit `032f340`，刪 61 檔 / 9921 行）
- `crypto_trading-main/ai_pro_trading_library/`（同事 PR #1 merge 進來的）
  已 git rm + push 到 binance + origin（commit `9b0bc0b`，刪 237 檔 / 55,693 行）

### Skill.md 版本更新

`trading-monitoring-executing-router/SKILL.md` 與 `references/repo-bootstrap.md`
在 4 個位置（dev_trade_bot / private-skills / 2c-openclaw skills / .build）
從 `cyqnt-trd==0.1.9.dev2` 更新為 `0.1.9.dev3`（後續再更新到 dev4）。
**尚未 commit** — 待後續批次處理。

### Commits（2026-05-26）

| Hash（origin/main）| Subject |
|---|---|
| `74eff8b` | feat(blocks): close atomic→cyqnt_trd audit gaps with verdicts/limits/stop_loss |
| `038a08f` | feat: complete atomic_strategy_lib port — compat, data_cli, exec_cli, monitoring, orchestration |
| `7e6250e` | feat(compat): add atomic_signals adapter layer for drop-in case migration |
| `a80b081` | feat: complete atomic compat — shim package + atomic-verbatim signals |
| `df46542` | chore(packaging): bundle atomic_strategy_lib shim into cyqnt-trd wheel (v0.1.9.dev3) |
| `cd71a50` | docs(atomic-compat): add migration handoff + shim README |
| `9b0bc0b` | chore: remove ai_pro_trading_library (no longer needed) |
| `6a016bb` | fix(deps): relax websockets pin to be compatible with binance SDKs (v0.1.9.dev4) |

dev_trading_bot/dev branch：

| Hash | Subject |
|---|---|
| `032f340` | chore: remove atomic_strategy_lib (now bundled in cyqnt-trd 0.1.9.dev3) |

### 已知限制

- OpenClaw 雲端 cyqnt-trd 升級到 0.1.9.dev4 待用戶觸發 release pipeline
- OpenClaw skill source 內的 `atomic_strategy_lib`（位於
  `2c-openclaw-customization-feature-add_new_skills_binance_pro/skills/...`）
  尚未刪除（等雲端確認升級後再刪）
- 4 個位置的 SKILL.md 版本號改動已寫盤但尚未 commit / push

---

## 索引：所有新增的指標與函數

### `cyqnt_trd/blocks/indicators.py`

#### 既有
sma, ema, wma, rma, rsi, macd, true_range, atr, adx, bollinger,
donchian, stochastic, vwap, obv, volume_ma, volume_zscore,
ma_direction, ma_alignment, swing_high, swing_low, highest, lowest,
price_change_pct, supertrend, ichimoku, parabolic_sar, rolling_zscore,
rolling_quantile, stochrsi, rsi_zone, atr_ratio, bb_bandwidth,
bb_pct_b, bb_squeeze, dual_speed_atr, volume_surge_ratio,
volume_trend, ema_cross_signal, trend_strength

#### 2026-05-27 新增
- 高優先：vwma, hma, mfi, cci, williams_r, keltner, heikin_ashi, cmf
- 中優先：tema, dema, aroon, trix, awesome_oscillator, pivot_points, zigzag, pvt

### `cyqnt_trd/blocks/smc_structure.py`（2026-05-27 新增）

fractal_pivot_high, fractal_pivot_low, fair_value_gap,
order_block_detect, bos_choch_detect

### `cyqnt_trd/blocks/smc_liquidity.py`（2026-05-27 新增）

liquidity_sweep_detect, equal_highs_lows, premium_discount_zone

### `cyqnt_trd/blocks/verdicts.py`（2026-05-26 新增）

5 combinators：additive_combine, weighted_combine, hierarchical_combine, ...
8 gates：hard_gate, enum_gate, soft_factor, verdict_classify,
verdict_with_gate, cross_validate, conflict_detect, normalize_score

### `cyqnt_trd/blocks/limits.py`（2026-05-26 新增）

is_funding_window, liquidation_check, max_positions_check,
max_exposure_check, daily_loss_check, price_deviation_check,
circuit_breaker_check

### `cyqnt_trd/blocks/stop_loss.py`（2026-05-26 新增）

4 stop helpers

### `cyqnt_trd/blocks/exit.py`（2026-05-26 新增）

graduated_take_profit, TakeProfitStep

### `cyqnt_trd/compat/atomic_signals/`（2026-05-26 新增）

computes.py（RSI/EMA/MACD/ATR/Bollinger/StochRSI/SuperTrend/ADX 等）,
derivatives_detectors.py, structure.py, volume_detectors.py

### `atomic_compat/atomic_strategy_lib/`（2026-05-26 新增）

51 個 thin re-export 檔案，覆蓋 atomic 全部 namespace。

### `cyqnt_trd/strategies/`（2026-05-27 新增）

smc_3confluence.py, smc_5confluence.py, mega_indicator_smoke.py

---

## 整體統計

| 指標 | 數字 |
|---|---|
| 新增 indicator 函數（兩天合計）| 24 個 + 21 個 atomic 補完 = **45 個** |
| 新增 SMC 元件 | 8 個 |
| 新增 shim 檔案 | 51 個 |
| 新增 atomic-verbatim 演算法 | ~20 個函數 |
| 新增 strategy 範例 | 3 個 |
| 新增 test fixtures | 4 個 binance parquet |
| 新增 unit tests | 78 個（49 + 29）|
| 全套測試結果 | 414 passed, 1 skipped |
| PyPI release | 0.1.9.dev3 → 0.1.9.dev4 |
| 文檔 | 2 份（MIGRATION_HANDOFF.md + atomic_compat/README.md）+ 此 CHANGELOG |
| 真實 backtest 驗證 | SMC 1h × 1000 bars: 2 trades, +1.27% |
| Lookahead 驗證 | 25/29 indicators 完全 safe，4/29 文件化的 lag |

---

## 待辦（跨日跟進）

### 即時可做
- 4 個位置的 SKILL.md（dev_trade_bot / private-skills / 2c-openclaw / .build）commit + push
- 觸發 OpenClaw release pipeline 拉 cyqnt-trd 0.1.9.dev4
- 雲端升級驗證後刪除 OpenClaw skill source 內的 atomic 真版

### 可selectorMethod（按需要）
- Wave B: Volume 進階（Anchored VWAP, VWAP StdDev Bands, TMF, CVD）
- Wave C: Composite signal engine（多 filter 整合，Signal Forge 風格）
- Wave D: 進階結構（Auto S/R + 評分, Trend Channels）
- 加 `confirmed=True` 參數到 fractal_pivot 解決 lookahead 半問題
- Lookahead pytest 自動化檢查
- TradingView reference value 對齊驗證（產 fixtures）
- Volume Profile approximation（用 1m sub-bar）

### 不在 scope
- Tick-by-tick backtest engine
- L2 orderbook 模擬
- AI 學習層 indicator
- Footprint 真實 tick 級實作
