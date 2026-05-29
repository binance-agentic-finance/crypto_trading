# Changelog

本文件記錄 cyqnt_trd 持續整合與功能擴充的時序內容。
編號採日期格式（YYYY-MM-DD），不對應 PyPI 版本號。
PyPI 版本變動見 `pyproject.toml` 與 git tag。

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
