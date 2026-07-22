# Strategy YAML Spec v1 — 設計 + 實作 + 驗證報告

**日期**:2026-07-22
**標的**:`cyqnt_trd/standard_bot`(block path,`--engine python`)
**目標**:一份 YAML 規格,讓前端 / bdp-ai-trading-bot 用 blocks 任意組合成策略,並可透過指令跑到 backtest / paper / live。
**狀態**:規格 + 執行器 + 驗證器**已實作並實跑驗證通過**(見 §4)。

---

## 0. 一句話結論(給主管 / Harvin)

> 我們**不需要**新建一層「評分/方向合成/sizing」——standard_bot 的 **block path**(`strategy.register` + `make_signals(df)→(long,short)`)本來就能做多因子、任意巢狀組合、宣告式出場與 sizing,而且 **backtest / paper / live 三條路跑的是完全相同的訊號碼**。這份 YAML 就是把那條路徑「宣告式化」:前端填 YAML → `validate`(靜態 + 合成資料 dry-run)擋掉錯 → `run` 編譯成 block strategy 上線。**已用真實引擎跑出交易、paper daemon 也實際啟動並抓即時行情。**

這修正了上一版報告的判斷:上一版誤把 standard_bot 只看成「typed plugin(一次一個、無評分層)」,漏了 block path 這個完整的組合面。

---

## 1. 交付物

**執行器套件**(新增,`cyqnt_trd/standard_bot/yaml_pipeline/`):
| 檔案 | 作用 |
|---|---|
| `interpreter.py` | YAML → `make_signals(df)`。白名單 blocks 分派、`inspect.signature` 自動判斷 df/series 首參、tuple 輸出取分量、任意巢狀組合樹 |
| `spec.py` | `load_spec` / `validate_spec`(靜態 + **合成 OHLCV dry-run**)/ `register_from_yaml` |
| `cli.py` | `validate` / `run`(backtest 在進程內跑;paper/live 產生正確且安全的指令) |
| `_daemon_loader.py` | 讓**獨立進程**(paper daemon)靠 `CYQNT_YAML_SPEC` 環境變數註冊 YAML 策略 |
| `__init__.py` / `__main__.py` | 對外 API + `python -m` 入口 |

**規格 + 範例**(`docs/strategy_yaml_spec/`):
- `strategy.schema.yaml` — 完整欄位模板 + 常用可組合 blocks 附錄
- `example_single_ma.yaml` — 單因子 MA 交叉(最小可跑)
- `example_multi_factor.yaml` — 多因子 + 巢狀(`any_of(all_of(EMA金叉+ADX+RSI過濾), 突破)`)+ ATR 出場
- 本報告
- `docs/block_registry.json` — 308 筆已驗證 block 目錄(組合時的可用算子清單)

---

## 2. YAML 如何覆蓋 Harvin 列的 8 個欄位

| Harvin 要求 | YAML 區塊 | 對映 standard_bot | 狀態 |
|---|---|---|---|
| 数据源 | `data.symbol/market_type/source` | `MarketQuery` + 資料 adapter | ✅ |
| 每个数据源拉取周期 | `data.primary.poll_interval` + `data.htf[]` | daemon `--poll-interval`;htf 掛 `_htf_*` 欄位 | ✅ |
| 信号计算 | `signals.indicators` + `signals.entry` | `make_signals(df)`(blocks) | ✅ |
| 方向 | `signals.entry.long/short`(給不給決定多空/long-only) | `target_position ±1` | ✅ |
| 仓位决策 | `sizing.size` | `strategy.register(size=)` | ✅ |
| 风控 | `risk.exit` + `risk.fees` + `risk.live_guards` | `exit_cfg` + fee model + executor 上限 | ✅ |
| 真实交易 / paper trade | `run.mode` | backtest / paper daemon / live executor | ✅ |
| 执行的周期 | `data.primary.poll_interval` / `run.duration_end_at` | daemon poll 迴圈 + session 上限 | ✅ |

巢狀組合(`all_of/any_of/not/exclude_when` 任意深度)= Harvin 說的「可以巢狀」。✅

---

## 3. 架構(為什麼是這條路)

```
YAML ──validate(靜態 + 合成資料 dry-run)──► 擋掉 block 不存在 / 簽章錯 / 參數錯
  │
  └─ register_from_yaml ─► interpreter.build_make_signals(spec) ─► make_signals(df)
                                                                      │
                              strategy.register(id, make_signals,     │ 三條路完全相同
                                 htf_specs, exit_cfg, size)           ▼
   ┌──────────────────────────┬───────────────────────────┬──────────────────────────┐
   │ backtest                 │ paper                      │ live                      │
   │ mvp_backtest --engine    │ mvp_paper_daemon --engine  │ paper daemon(訊號)       │
   │ python                   │ python(+ _daemon_loader)   │ + mvp_live_executor(下單)│
   └──────────────────────────┴───────────────────────────┴──────────────────────────┘
```

關鍵設計決策:
1. **走 block path,不走 typed-plugin path。** block path 能組合全部 blocks(不只 11 個 plugin),且有現成的評分/巢狀/出場/sizing——正是「讓所有 blocks 排列組合」的落點。
2. **不新建執行底座,復用既有 entrypoints。** paper→live 只換 executor 層,訊號碼一字不改(standard_bot 原生保證)。
3. **validate 的 dry-run 是防過去 47 bug 的關鍵閘。** Harvin demo 那類「簽章/型別/參數對不上」的錯,在這裡於 `validate` 階段就被真的跑一遍合成資料抓出來,碰真錢前先擋。
4. **安全邊界**:`signals` 只能呼叫白名單 blocks module(不能 import 任意碼);密鑰不入 YAML(`.env`);live 絕不由本 CLI 自動下單。

---

## 4. 驗證結果(全部實跑,非紙上)

環境:`python3.11` venv + `pandas/numpy/pyarrow/requests/pyyaml/websockets`。資料:repo 內 `tests/blocks/fixtures/BTCUSDT_1h_500bars.parquet` 轉成 kline JSON。

| # | 測試 | 結果 |
|---|---|---|
| 1 | `validate example_single_ma.yaml` | ✅ OK(靜態 + dry-run 通過) |
| 2 | `validate example_multi_factor.yaml`(巢狀) | ✅ OK |
| 3 | `run` 回測 single_ma(進程內) | ✅ `snapshots=300 trades=3`,有 pnl |
| 4 | `run` 回測 multi_factor(巢狀,進程內) | ✅ `trades=2`;`size:0.5` 反映在下單量(qty≈單因子一半) |
| 5 | **loader 模組在獨立進程註冊**(= paper daemon 的機制)+ 回測 | ✅ `trades=2`(`--strategy-module _daemon_loader` + `CYQNT_YAML_SPEC`) |
| 6 | validate 擋「block 不存在」(`indicators.emaa`) | ✅ 拒絕,exit=1 |
| 7 | validate 擋「簽章錯」(`ma_cross_above` 少一個 arg) | ✅ `dry-run failed: TypeError: ma_cross_above() missing 1 required positional argument: 'slow'`,exit=1 |
| 8 | **paper daemon 實跑** | ✅ 啟動→匯入 loader→註冊策略→抓 60 根即時 K→輪詢(poll_count=2)→寫 `state.json`(status running, price 66238.3)→SIGTERM 乾淨關閉。**未下任何真實單** |
| 9 | `run` live 派工 | ✅ 印出「paper 先行→查餘額→CONFIRM→時長上限」安全序列 + dry-run-first + max_notional + EMERGENCY_STOP;**本 CLI 不自動下單** |

---

## 5. 誠實的限制 / 邊界(交接必讀)

1. **live 真實下單未執行(刻意)。** 註冊 + 派工 + 兩段式指令都驗過,但實際 futures live 下單需「已驗證 futures CLI 的環境 + 人工 CONFIRM + 真錢」,本次不做也不應由 agent 自動做。預設仍以 `binance-cli spot get-account` 為準。
2. **validate 的 dry-run 是 fail-fast**:一次報第一個致命錯,修完再跑會出下一個(不是一次列全部)。靜態層(block 不存在)則會一次列出。
3. **組合面覆蓋**:`indicators / conditions / entry / exit / sizing` 已接線且實測;`verdicts / scoring / smc_* / patterns` 在白名單內、可被 dry-run 逐一驗證,但我只實測了常用集合,不是 308 個 block 逐一跑過——每個新 block 第一次用時由 `validate` 的 dry-run 把關。
4. **多時框(htf)**:`BlockStrategyPlugin` 原生支援、dry-run 也驗過;但離線 `--input-json` 只有單一時框,htf 腿在離線測試會是 NaN(不報錯)。正式 `binance_rest`/`historical_parquet` 會提供多時框資料。
5. **numba 引擎未使用**:走 `--engine python`(block 組合面)。numba 路徑需 repo 釘選的 `numpy<2/pandas<3`;正式部署請用 repo 的 `requirements.txt + requirements-standard-bot-mvp.txt` 在 3.11 venv 安裝(本次用了較新版,python 引擎路徑可跑)。

---

## 6. 交接指令速查

```bash
# 驗證(碰真錢前的硬閘)
python -m cyqnt_trd.standard_bot.yaml_pipeline validate docs/strategy_yaml_spec/example_multi_factor.yaml

# 回測(離線用 --input-json;正式用 YAML 內 data.source: binance_rest)
python -m cyqnt_trd.standard_bot.yaml_pipeline run docs/strategy_yaml_spec/example_multi_factor.yaml \
  --input-json <klines.json> --output-json result.json

# paper(把 run.mode 改 paper;--start 才真的 spawn daemon)
python -m cyqnt_trd.standard_bot.yaml_pipeline run my_strategy.yaml --start

# live(run.mode: live + risk.live_guards + run.duration_end_at;印出安全序列,人工執行 executor)
python -m cyqnt_trd.standard_bot.yaml_pipeline run my_strategy.yaml
```

---

## 7. 建議下一步

1. 跟 Harvin 對齊:**YAML 編譯目標定在 standard_bot block path**,demo_strategy(atomic)降為概念參考;7 模組概念沿用。
2. 把 `validate` 接進 CI / 前端存檔時 → 存檔即驗證,擋掉 config 類錯誤。
3. 擴充 `sizing`(目前 `size` 比例;可接 `blocks.sizing` 的 Kelly / 固定風險%)與把 `blocks.limits` 的 daily_loss / kill switch 接進 paper daemon 的風控(live 上線前建議)。
4. 正式環境用 repo 釘選版本重跑一次 §4 驗證(numpy<2/pandas<3),並補一次真實 `binance_rest` 多時框 paper 24h 觀察。
