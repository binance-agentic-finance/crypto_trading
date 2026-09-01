# cyqnt_trd 架構

這份文件回答一件事:**這個 repo 由哪些部分組成、哪些部分在主線上、一筆訊號怎麼從資料走到下單**。

三份文件分工:

| 文件 | 回答什麼 |
|---|---|
| `README.md` | 怎麼安裝、怎麼跑起來 |
| **`docs/ARCHITECTURE.md`(本檔)** | **系統長什麼樣、哪條路活著** |
| `AGENTS.md` | 接手前必須知道的陷阱與驗證方法 |

本檔所有數字都附了重算指令(見最後一節)。**數字會過期,指令不會** —— 引用前先自己跑一次。

---

## 一、這個 repo 負責哪一段

整條鏈上有三段,這個 repo 只做中間那段:**把資料變成訊號**。不採集資料,也不負責真正把單送進交易所。

```
上游(不是這裡)              這個 repo                    下游(不是這裡)
─────────────────────   ───────────────────────    ─────────────────────
交易所公開 API       ┐                              ┌→ 回測(歷史重播)
內部 HTTP 節點       ┤                              │
Binance Square       ┤→  一份輸入 JSON              ├→ paper(模擬撮合)
外部 vendor          ┤   cyqnt.input/v1             │
本地 parquet         ┘        ↓                     ├→ live 執行層
                         計算積木組成策略           │  (讀訊號自己下單)
                              ↓                     │
                         一份輸出 JSON  ───────────→└→ 前端 / 通知
                         cyqnt.signal/v2
```

**關鍵設計:輸入被凍結成一份 JSON。** 回測、paper、live 走同一段程式碼,差別只在那份 JSON 來自歷史還是現在去抓。所以「回測會過、實盤不會過」在這個架構裡不該發生 —— 如果發生了,是有人繞過了那份 JSON(見 §五的已知缺口)。

## 二、一筆訊號的生命週期

```
① 取數    決定要哪些資料 → 逐節點抓 → 正規化成 7 種標準形狀之一
          → 過 PIT 閘門(只留「當時真的知道」的列)→ 一份 cyqnt.input/v1
                │
② 組策略  那份 JSON 攤成策略吃得下的表 → 呼叫計算積木(指標 / 條件 / 型態 /
          倉位 / 停損)→ 逐根或截面產出布林序列或候選名單
                │
③ 出訊號  包成 cyqnt.signal/v2:方向、信心、進場、停損、分批止盈、部位意圖、
          證據、警告。建構時就檢查一致性,貼錯標籤直接拋錯
                │
④ 消費    回測算績效 / paper 模擬撮合 / live 執行層讀訊號自己下單
```

四段之間只靠兩份契約溝通,任何一段換掉實作、只要契約不變,其他三段不用動:

| 契約 | 定義處 | 機器可驗的 schema |
|---|---|---|
| 輸入 `cyqnt.input/v1` | `cyqnt_trd/standard_bot/core/input_contract.py` | `strategies/_standard/input.schema.v1.json` |
| 輸出 `cyqnt.signal/v2` | `cyqnt_trd/standard_bot/core/signal_contract.py` | `strategies/_standard/signal.schema.v2.json` |

schema 由 dataclass 自動生成,並有測試釘住(`tests/standard_bot/test_input_contract.py::test_samples_regenerate_identically`)—— 契約改了但沒重生,CI 會紅。

## 三、四種訊號型別

消費端只看 `kind` 一個欄位就知道怎麼處理。**`kind` 由內容推導,不由生產者宣告** —— 貼錯會在 `__post_init__` 直接 `ValueError`。

| `kind` | 內容 | 誰消費 | 現況 |
|---|---|---|---|
| `trade` | 進場價、停損、分批止盈、部位大小 | 回測 / paper / live 執行層 | 完整 |
| `selection` | 候選籃子、宇宙大小、每檔的分數與理由 | 前端 / watchlist / 人工 | 出得來,但變不成訂單(§五)|
| `alert` | 建議動作、摘要 | 通知 / 人工判讀 | 不自動執行,設計如此 |
| 平倉(仍是 `trade`)| 用 `intent` 區分,不是另一種 `kind` | 同 `trade` | 完整 |

「要對部位做什麼」由 `intent` 決定,共 12 個值:`open_long` `open_short` `add_long` `add_short` `reduce_long` `reduce_short` `close_long` `close_short` `flip_to_long` `flip_to_short` `flat` `hold`。

**沒宣告 `reads_positions=True` 的策略不准發 `close` / `reduce` / `flip`** —— 那些指令都在斷言「現在有一個部位」,而一支從沒讀過部位的策略是在猜。

範例輸出在 `docs/standard_bot_io/samples/`,四種 `kind` 各一份;`output_open_long.json` 與 `output_close_short.json` 對照著看,就懂 `intent` 為什麼要 12 個值。

## 四、能力邊界 —— 用數字說

**「有這個資料」和「能拿這個資料回測」是兩件事。** `cyqnt_trd/standard_bot/data/catalog.py` 有 75 個節點,但按可回測性分:

| `availability` | 節點數 | 意思 |
|---|---|---|
| `BACKTESTABLE` | **7** | 有深度時點歷史,可以逐根 walk-forward |
| `SEMI` | 21 | 窗口有限或 T+1,能回測但要知道限制 |
| `FORWARD_ONLY` | **38** | 快照,**沒有時點歷史**。拿去回測會把今天的值貼到每根歷史 bar 上 |
| `EXTERNAL_PENDING` | 9 | 規劃了但還沒接上資料源 |

**能拿來做嚴謹歷史回測的只有 7 個節點。** `validate_nodes(..., for_backtest=True)` 會在編譯期擋下並引用具體陷阱,不會讓它默默跑出一個漂亮的假績效。

同一批節點按來源分:

| `source_path` | 節點數 | 說明 |
|---|---|---|
| `INTERNAL_HTTP` | 28 | 欄位契約在 repo 裡(`standard_bot/data/internal_slots.py`),**client 不隨這個公開套件出貨**;有 client 的部署接上即可用,bundle 結構完全相同,只有 `source_status` 不同 |
| `PUBLIC_BINANCE` | 25 | K 線 / funding / OI / 多空比 / 主動買賣 / 全市場 ticker |
| `EXTERNAL_VENDOR` | 9 | 多數屬上表的 `EXTERNAL_PENDING` |
| `SQUARE_SKILL` | 6 | 新聞 / 熱度 / 情緒 / 話題 |
| `LOCAL_PARQUET` | 3 | 研究與回測用的離線資料 |
| 其他 | 4 | |

一句話總結:**公開資料那條路完整可跑;內部節點的欄位契約已經定好,接上 client 就能用,策略程式碼一行都不用改。**

---

## 五、頂層模組地圖:14 個模組,主線只有 3 個

先知道一件事:**這個 repo 的複雜度幾乎全部來自只加不刪** —— 每次有新需求就在旁邊加一條路,舊的因為在跑真錢而沒有拆。所以「看到兩個名字很像的模組」是常態,而不是每一個都活著。

「被引用」= 在 `cyqnt_trd` / `strategies` 底下被 import 的次數(排除模組自己內部)。**「引用來源」比「引用次數」重要** —— 一個模組可以被引用很多次,而引用它的全都是自己也沒有 caller 的腳本區;那不是主線。

| 模組 | 規模 | 被引用 | 引用來源 | 定位 |
|---|---|---|---|---|
| **`blocks`** | 25 個模組檔 | **137** | `strategies`×71 `standard_bot`×61 | 計算積木。策略的原料 |
| **`data_cli`** | 19 檔 | **73** | `standard_bot`×69 | 取數:包 `binance-cli` / `binance-pro-cli` 的 subprocess wrapper |
| **`standard_bot`** | 118 檔(12 子系統 + 6 頂層檔)| **54** | `strategies`×29 `blocks`×23 | 框架主體:契約、取數、YAML、模擬、執行 |
| `trading_signal` | **126 檔 / 12,397 行** | 23 | `test_script`×10 `online_trading`×10 `get_data`×3 | **平行的舊因子體系。** 引用它的三個模組自己都沒有 caller —— **這不是主線** |
| `backtesting` | 4 檔 | 4 | `trading_signal`×4 | 單因子勝率測試。**引用全部來自 `trading_signal`**,所以和它同屬舊體系 |
| `strategies` | 17 檔(另有 repo 根的 `strategies/` 28 檔)| 3 | `standard_bot`×3 | 策略實作 |
| `utils` | 2 檔 | 1 | `test_script`×1 | 引用者沒有 caller |
| `evolve` | 17 檔 | 0 | — | 獨立 CLI:參數演化。有 2 個 `__main__` 與 3 個測試檔,**是活的工具不是死碼** |
| `get_data` | 5 檔 | 0 | — | 獨立抓數腳本(4 個 `__main__`)|
| `online_trading` | 2 檔 | 0 | — | 獨立腳本(1 個 `__main__`)|
| `test_script` | 3 檔 | 0 | — | 手動測試腳本 |
| `compat` | 9 檔 | 0 | — | **無 caller、無 `__main__`、無測試。** `atomic_strategy_lib` 的型別 shim |
| `exec_cli` | 5 檔 | 0 | — | **無 caller、無 `__main__`、無測試。** `binance-cli` 下單 wrapper |
| `strategy_cases` | 1 檔 | 0 | — | 隨套件出貨的策略案例資產 |

**主線是三個**:`blocks`(算)+ `standard_bot`(組裝與契約)+ `data_cli`(取數)。它們互相引用,而且被 `strategies` 引用 —— 那是唯一一個閉環。

**最容易誤判的是 `trading_signal`** —— 126 檔、12,397 行,規模看起來像核心。但它和 `standard_bot` 沒有任何交集,23 次引用全部來自 `test_script/`、`online_trading/`、`get_data/` 這三個本身沒有 caller 的腳本區。`backtesting` 也在這條線上(4 次引用全來自 `trading_signal`)。

**兩個 `strategies` 不要搞混**:`cyqnt_trd/strategies/`(17 檔,隨套件出貨)和 repo 根的 `strategies/`(28 檔,含 `_standard/` 的契約 schema)。

## 六、`standard_bot` 的 12 個子系統

| 子系統 | 規模 | 負責什麼 |
|---|---|---|
| **`data`** | 21 檔 / 8,649 行 | 節點目錄(75 節點)、取數、PIT 閘門、正規化成 7 種形狀、攤成策略吃的表 |
| **`yaml_pipeline`** | 10 檔 / 7,534 行 | YAML → 策略:解析、靜態驗證、編譯、執行。前端與 bot 的主要入口 |
| **`simulation`** | 9 檔 / 4,035 行 | 三個回測引擎(事件驅動 / 向量化 / numba)|
| `entrypoints` | 19 檔 / 4,183 行 | 所有 CLI 進入點(§八)|
| `signal` | 6 檔 / 3,520 行 | 訊號組裝、證據、警告、reason code |
| **`core`** | 4 檔 / 2,306 行 | **兩份契約的定義處** |
| `execution` | 8 檔 / 1,569 行 | 下單:paper / testnet / mainnet / CLI、冪等、風控規則 |
| `advisory` | 6 檔 / 1,561 行 | 監控與建議型 bot(`kind=alert`)|
| `runtime` | 6 檔 / 1,153 行 | 執行期:編譯後策略要的模組、狀態 |
| `orchestration` | 4 檔 / 799 行 | 多 bot 調度 |
| `monitoring` | 4 檔 / 608 行 | 執行期監控工具 |
| `ma_cross_strategy` | 獨立子專案 | 自帶 README / scripts / strategies / tests 的均線交叉範例,**不走上面的架構** |

加上頂層 6 檔(2,169 行):`bot.py` `adapter.py` `registry.py` `config.py` `universal.py`。其中 `adapter.py` 是兩條策略路徑之間的橋,看它**刻意不做什麼**比看它做什麼有用。

## 七、同一件事有多個實作 —— 哪一個活著

**接手時最省時間的一節。** 下面每個數字都是查呼叫者查出來的,不是從 docstring 推論的。

| 概念 | 實作 | 現況 |
|---|---|---|
| **策略寫法** | 3 條路 | `register()` 28 次 · `register_selection()` 6 次 · StandardBot 子類 7 檔 · **`UniversalBot` 0 個真實 caller**(唯一命中是 `universal.py` 自己 docstring 裡的範例)|
| **輸出格式** | 4 種並存 | `cyqnt.signal/v2`(現行)· `cyqnt.signal/v1`(舊策略還在用,**未標 deprecated**)· `SignalEnvelope block/v1` · `trades.jsonl` |
| **回測引擎** | 3 個都活著 | 事件驅動 `simulation/runner.py`(2 caller)· 向量化 `simulation/vectorized_backtest.py`(5 caller)· `simulation/numba_runner.py`(2 caller)。三者已對拍到逐根一致 |
| **取數路徑** | 5 條 | `data/input_bundle.py`(讀檔)· `data/live_bundle.py`(打 API)· `data/historical.py` · `yaml_pipeline/_data.py` · `runtime/data.py` |

**查呼叫者的方法** —— 這個 repo 一直在改,不要相信任何文件寫的接線狀態,包括本檔:

```bash
# 注意左括號,才分得出「真的呼叫」與 docstring / re-export
grep -rn "build_input_bundle(" --include="*.py" cyqnt_trd strategies \
  | grep -v "data/input_bundle.py"

# 輸出為空 = 只有測試和文件在用 = 那條路現在沒有在跑
```

## 八、怎麼跑起來

`cyqnt_trd/standard_bot/entrypoints/` 有 19 個檔案、17 個可以直接 `-m` 跑。最常用的四條:

```bash
# ① 抓一份現在的快照(照 spec 解出它需要的資料,不多抓)
python -m cyqnt_trd.standard_bot.entrypoints.mvp_input_bundle \
    --strategy-yaml <spec>.yaml --out /tmp/bundle.json

# ② 餵回去 = 離線重播,零網路,結果逐位元可重現
python -m cyqnt_trd.standard_bot.yaml_pipeline run <spec>.yaml --input-json /tmp/bundle.json

# ③ 不給 --input-json = 現在去抓
python -m cyqnt_trd.standard_bot.yaml_pipeline run <spec>.yaml

# ④ 靜態驗證(不打網路,擋掉順序錯誤與不可回測的節點)
python -m cyqnt_trd.standard_bot.yaml_pipeline validate <spec>.yaml
```

其他進入點:`mvp_paper` / `mvp_paper_daemon`(模擬盤)、`mvp_backtest`、`mvp_advisory`(監控)、`mvp_testnet_execution` / `mvp_mainnet_execution`(下單)、`mvp_monitor_http`、`mvp_run_manager`。

## 九、兩個不變量(改動時不要破壞)

**① `event_time` 與 `available_time` 不是同一件事**

`event_time` = 事情何時發生;`available_time` = 我們何時才可能知道。回測只准讀 `available_time <= decision_time` 的列。**把這兩者混為一談就是偷看未來。** PIT 閘門在產生輸入 JSON 時 gate 一次,下游不要再自己做一次 —— 會不一致。

**② `kind` 由內容推導,不由生產者宣告**

貼錯標籤會在建構時直接拋錯。不要改成用宣告值 —— 那等於把「輸出形狀對不對」從編譯期檢查降級成執行期的靜默錯答。

這兩條之所以寫成不變量,是因為**這個 repo 修過的 bug 絕大多數不是崩潰,而是看起來合理的錯答案**:週期不符 → 靜默沒有欄位、策略無聲停止參考未平倉量;參數名打錯一個字母 → 停損整個消失、回測照跑照出數字;某個比例是 NaN → 每個候選都變做空,而 `min(100, nan)` 回 `100.0` → 信心被洗成最高。

**所以改完不要只看有沒有報錯 —— 把你以為在起作用的條件拿掉,看結果有沒有變。沒變就是它從來沒在作用。**

## 十、已知缺口

**這一節會過期,每一條都自己重驗一次再相信。**

| 缺口 | 影響 |
|---|---|
| live 執行層吃 `trades.jsonl`(成交記錄,不是訊號)| **交易所端沒有掛保護單** —— 停損只在程式內模擬判斷,程式掉了部位就沒有保護 |
| `ExitPlan.exchange_managed` | 欄位存在,**沒有任何消費端實作** |
| paper session 的型別檢查 | StandardBot v2 那 7 支**上不了 paper / live** |
| `yaml_pipeline/_data.py` 自己取數 | 沒走 `build_input_bundle`,所以 YAML 回測的 PIT 閘門與 bundle 那條不同源 |
| 選幣無法回測 | 宇宙 / 排名 / 新聞在節點目錄標 `FORWARD_ONLY`,**沒有時點歷史** |
| `build_intents` 跳過非 `trade` | **選幣候選變不成訂單** —— 目前是 export / watchlist |
| `cyqnt.signal/v1` 未標 deprecated | 舊策略還在用,4 種輸出格式並存 |

---

## 十一、重算本檔所有數字

**這一節是本檔最重要的部分。** 文件裡的數字會過期 —— 這個 repo 曾經在四份文件裡對同一件事給出四個不同的數字(blocks 函式數:146 / 310 / 365 / 393),原因是每份文件各自手寫、各自用不同的定義。

跑這個腳本可以一次重算全部,**並且它會印出每個數字的定義**:

```bash
python docs/count_architecture_facts.py
```

`docs/report_flow.html` 是另一條路 —— 它由 `docs/gen_report_html.py` 實跑生成,數字自動跟上:

```bash
python docs/gen_report_html.py          # 重新生成,離線,讀本地 parquet
```

**引用數字前先跑一次。** 如果本檔的數字和腳本輸出不一致,相信腳本,並更新本檔。
