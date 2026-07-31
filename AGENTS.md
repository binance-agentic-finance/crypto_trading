# cyqnt_trd — 給 AI 助手的上手說明

這個 repo 做一件事:**把多來源市場資料組成一份輸入,用積木組出策略,輸出一種格式的訊號**
(交易指令或選幣籃子),給回測 / paper / live 消費。

```
①  取數                      ②  組策略               ③  出訊號                ④  消費
──────────────────────────────────────────────────────────────────────────────────
K 線            ┐                                      ┌─ kind=trade       ┌→ 回測
funding / OI    ┤                                      │                   │
清算            ┤→   一份 JSON      →   blocks 積木  →  ┼─ kind=selection  ┼→ paper
新聞 / 熱度     ┤   cyqnt.input/v1      (365 個函式)    │                   │
選幣宇宙        ┤                                      └─ kind=alert       └→ live
內網 BigData    ┘                                        cyqnt.signal/v2
```

---

## ⚠️ 先讀這一節,否則你會給出自信但錯誤的答案

這個 repo 有三個特性,會讓「讀了程式碼就以為懂了」的助手翻車。

### 1. 同一件事有多個實作,而且不是每一個都活著

| 概念 | 實作數 | 說明 |
|---|---|---|
| 策略寫法 | 3 | block path(20 支策略)、StandardBot v2(7 支)、`UniversalBot`(**0 支、0 測試**) |
| 輸出格式 | 4 | `SignalEnvelope block/v1`、`cyqnt.signal/v1`、`cyqnt.signal/v2`、`trades.jsonl` |
| 回測引擎 | 3 | `cyqnt_trd/standard_bot/simulation/runner.py`(事件驅動)、`cyqnt_trd/standard_bot/simulation/vectorized_backtest.py`、`cyqnt_trd/standard_bot/simulation/numba_runner.py` |
| 取數路徑 | 多條 | `cyqnt_trd/standard_bot/data/input_bundle.py`、`cyqnt_trd/standard_bot/data/live_bundle.py`、`cyqnt_trd/standard_bot/data/historical.py`、`cyqnt_trd/standard_bot/yaml_pipeline/_data.py`、`cyqnt_trd/standard_bot/runtime/data.py` |

**這是歷史累積的結果,不是設計。** 每次有新需求就在旁邊加一條路,舊的因為在跑真錢而沒有拆。

所以看到兩個名字很像的模組時,**先查哪一個有 caller**(方法見下一節),不要假設新的取代了舊的。

### 2. 文件描述的是「契約」,不一定是「目前跑的路」

有些模組寫得很完整、docstring 很清楚、有測試,但**production 沒有任何呼叫者**。

**在你說「系統會做 X」之前,先確認有人呼叫它。** 只掃 production 目錄,不要用
`. | grep -v ./tests` —— `grep -r` 的輸出不一定帶 `./` 前綴,那個排除式會失效:

```bash
# 對:限定範圍,並排除定義處自己
grep -rn "build_input_bundle" --include="*.py" cyqnt_trd strategies \
  | grep -v "data/input_bundle.py"
```

再確認找到的是**真的呼叫**而不是 docstring 或 re-export:

```bash
grep -rn "build_input_bundle(" --include="*.py" cyqnt_trd strategies   # 注意左括號
```

輸出為空 = 只有測試和文件在用 = **那條路現在沒有在跑**。這個檢查做一次,可以省下好幾小時的
錯誤結論 —— 而且這個 repo 正在被改,所以**每次都要重查,不要相信任何文件裡寫的接線狀態,
包括這一份**。

### 3. 錯誤幾乎不會崩潰,而是產出「看起來合理的錯答案」

這個 repo 修過的 bug 絕大多數屬於這一類。實例:

- OI parquet 週期不符 → **靜默沒有欄位**,策略無聲停止參考未平倉量
- `stop_pctt` 打錯一個字母 → **停損整個消失**,回測照跑照出數字
- `bull_ratio` 是 NaN → 每個候選都變做空,而 `min(100, nan)` 回傳 `100.0` → **分數被洗成最高信心**
- paper session 收到 funding/OI、存進 checkpoint、**然後不交給策略**

**所以:改完不要只看「有沒有報錯」。要做反事實檢查** —— 把你以為在起作用的那個條件拿掉,
看結果有沒有變。沒變就是它從來沒在作用。

---

## 按這個順序讀

| 順序 | 檔案 | 為什麼 |
|---|---|---|
| 1 | `strategies/_standard/signal.schema.v2.json` | **輸出契約**。42 個欄位,交易與選幣共用。從 dataclass 自動生成 |
| 2 | `cyqnt_trd/standard_bot/core/signal_contract.py` | 上面那份 schema 的來源。`StandardSignal` + `PositionIntent`(12 值) + `ExitPlan` |
| 3 | `strategies/_standard/input.schema.v1.json` | **輸入契約**。7 種標準 frame 形狀 |
| 4 | `cyqnt_trd/standard_bot/data/input_bundle.py` | 輸入產生器(讀本地檔)。看 docstring 裡的 PIT 說明 |
| 5 | `cyqnt_trd/standard_bot/data/live_bundle.py` | 輸入產生器(打 API)。**與 4 吐出相同 envelope** |
| 6 | `cyqnt_trd/strategies/funding_squeeze_panel.py` | **最有代表性的交易型策略**:論點需要 K 線 + 資金費率 + 未平倉量 + 主動買賣**四個來源同時成立**,缺一個就驗證不了。是多源那條路的活體證明 |
| 7 | `cyqnt_trd/strategies/news_buzz_selector.py` | **最有代表性的選幣型策略**:一次對整個宇宙排名。看它的 docstring 講「門檻要放多低」—— 熱度先於流動性出現 |
| 8 | `cyqnt_trd/standard_bot/yaml_pipeline/interpreter.py` | YAML → 策略。看 `DENIED_NAMESPACES` 的註解理解可用邊界 |
| 9 | `cyqnt_trd/blocks/BLOCKS_API.md` | 365 個積木的清單 |
| 10 | `cyqnt_trd/standard_bot/adapter.py` | 兩條策略路徑之間的橋。看它**刻意不做什麼** |

`docs/full_flow_spec.html` 和 `docs/report_flow.html` 是自動生成的全流程說明,
數字全部來自實跑 —— 但**它們描述契約,不保證接線**(見上面第 2 點)。

### 參考策略:一支交易、一支選幣

寫新策略時**照這兩支的形狀寫**,不要從零開始:

| | `funding_squeeze_panel.py` | `news_buzz_selector.py` |
|---|---|---|
| 型別 | 交易(單一標的、逐根) | 選幣(截面、一次) |
| 註冊 | `strategy.register(id, make_signals, needs={"derivatives": True})` | `strategy.register_selection(id, selection_fn)` |
| 進入點 | `make_signals(df) -> (long, short)` | `selection_fn(universe_df, ticker_rank_df, **_) -> list[dict]` |
| 輸出 kind | `trade` | `selection` |
| 讀什麼 | df 上的 `funding_rate` / `open_interest_value` / 主動買賣量欄位 | `universe` + `ticker_rank` 兩張 frame |

實測(寫這份文件時):

```
funding_squeeze_panel  → 需要 ('rate','oi_value','buy_vol','sell_vol');
                          在 sample bundle 上 0 訊號(四條件沒同時成立 —— 正常)
news_buzz_selector     → 5 檔候選,BTC long / ETH short / SOL long /
                          BNB neutral / XRP short,每檔標流動性等級
```

`funding_squeeze_panel` 回 0 訊號**不是壞掉** —— 它的論點要四個來源同時成立,大多數時候
不成立。判斷它有沒有在作用,要看 `REQUIRED_COLUMNS` 那幾欄有沒有真的進到 df(用下一節
「探測 plugin」的方法),不是看有沒有訊號。

---

## 輸入 / 輸出的範例檔在哪

**規格看 schema,長相看這些檔。** 全部在 `docs/standard_bot_io/samples/`:

### 輸入

| 檔案 | 是什麼 |
|---|---|
| `docs/standard_bot_io/samples/input_bundle_preview.json` | **先看這個**。與完整檔同結構,每個 frame 只留幾列,人讀得完 |
| `docs/standard_bot_io/samples/input_bundle_example.json` | 完整的 `cyqnt.input/v1`(讀本地檔產生)。上面那些驗證腳本都吃這一份 |
| `docs/standard_bot_io/samples/live_bundle_example.json` | 打 API 產生的版本 —— **與上面同一個 envelope**,證明 paper/live 與回放形狀一致 |
| `docs/standard_bot_io/samples/bot_context.json` | bundle 讀回來、策略在 `decide(ctx)` 裡真正拿到的樣子 |
| `docs/standard_bot_io/samples/input_shapes.json` | 7 種標準 frame 形狀的欄位定義 |
| `docs/standard_bot_io/samples/node_shape_table.json` | 每個資料節點 → 對應哪一種形狀 |

### 輸出

| 檔案 | `kind` | 是什麼 |
|---|---|---|
| `docs/standard_bot_io/samples/output_open_long.json` | `trade` | 開多,含完整 `entry` / `exit_plan` / `size` |
| `docs/standard_bot_io/samples/output_close_short.json` | `trade` | 平空 —— 和開多**對照著看**,就懂 `intent` 為什麼要 12 個值 |
| `docs/standard_bot_io/samples/output_selection.json` | `selection` | 選幣籃子。**與上面兩份頂層欄位逐一相同**,只是填了 `candidates` |
| `docs/standard_bot_io/samples/output_advisory.json` | `alert` | 觀察型,不自動執行 |
| `docs/standard_bot_io/samples/output_open_long.execution_request.json` | – | 訊號再往下變成執行請求的樣子 |

這些檔案由 `docs/standard_bot_io/gen_samples.py` 生成,而且**有測試釘住**
(`tests/standard_bot/test_input_contract.py::test_samples_regenerate_identically`)——
契約改了但沒重生,CI 會紅。所以它們永遠與程式碼同步,可以放心當真。

自己比一次最有感:

```bash
.venv-standard-bot/bin/python - <<'PY'
import json, pathlib
d = pathlib.Path("docs/standard_bot_io/samples")
t = json.loads((d / "output_open_long.json").read_text())
s = json.loads((d / "output_selection.json").read_text())
print("trade     kind=%-10s keys=%d" % (t["kind"], len(t)))
print("selection kind=%-10s keys=%d" % (s["kind"], len(s)))
print("欄位差集:", set(t) ^ set(s) or "空 —— 同一份 schema")
PY
```

---

## 跑這些來確認你的理解

全部離線可跑,不需要網路。

```bash
cd <repo>   # 用 .venv-standard-bot/bin/python,不要用系統 python

# 全套測試(基準線,改動前後都要跑)
.venv-standard-bot/bin/python -m pytest tests -q

# 四個範例 YAML:靜態檢查 + 用合成資料實跑一次
for f in docs/strategy_yaml_spec/example_*.yaml; do
  .venv-standard-bot/bin/python -m cyqnt_trd.standard_bot.yaml_pipeline validate $f
done

# 多源策略回測(讀本地 parquet)
.venv-standard-bot/bin/python -m cyqnt_trd.standard_bot.yaml_pipeline \
    run docs/strategy_yaml_spec/example_multi_source.yaml

# 一份 bundle 同時餵交易型與選幣型,證明輸出是同一組欄位
.venv-standard-bot/bin/python - <<'PY'
from cyqnt_trd.standard_bot.bot import BotContext, _frames_from_snapshot
from cyqnt_trd.standard_bot.data import load_input_bundle
from strategies.standard.blocks_reference_bots import BlocksNewsRankBot
from strategies.standard.multi_source_bot import MultiSourceBot

snap = load_input_bundle("docs/standard_bot_io/samples/input_bundle_example.json")
ctx = BotContext(decision_time=snap.meta.decision_as_of, frames=_frames_from_snapshot(snap))
out = {}
for bot in (MultiSourceBot(), BlocksNewsRankBot()):
    sigs = bot.decide_checked(ctx)
    if sigs:
        d = sigs[0].to_dict()
        out[d["bot_id"]] = d
        print("%-26s kind=%-10s keys=%d" % (d["bot_id"], d["kind"], len(d)))
ks = [set(d) for d in out.values()]
print("頂層欄位完全相同:", all(k == ks[0] for k in ks))
PY
```

互動 demo(需外網抓行情;NL→YAML 那半需要公司 VPN):

```bash
PYTHONPATH=<repo> <repo>/.venv-standard-bot/bin/python \
  docs/strategy_yaml_spec/demo/server.py
# 開 http://127.0.0.1:8799
```

---

## 兩個核心不變量(改動時不要破壞)

### `event_time` vs `available_time`

- `event_time` = 事情**何時發生**
- `available_time` = 我們**何時才可能知道**

回測只准讀 `available_time <= decision_time` 的列。**把這兩者混為一談,就是 walk-forward
偷看未來。** PIT 閘門在 bundle 產生時 gate 一次,下游不要再自己做一次(會不一致)。

### `kind` 決定輸出形狀

消費端 switch `kind` 一個欄位:

| `kind` | 填什麼 |
|---|---|
| `trade` | `entry` / `exit_plan` / `size` |
| `selection` | `candidates` / `universe_size` |
| `alert` | `advisory_action` / `summary` |

`kind` **由 payload 推導,不由生產者宣告** —— 貼錯標籤會在 `__post_init__` 直接
`ValueError`。不要改成用宣告值。

另外 `intent`(12 值)是唯一決定「要對部位做什麼」的欄位。**沒宣告
`reads_positions=True` 的策略不准發 `close/reduce/flip`** —— 那些指令都在斷言「現在有一個
部位」,而一支從沒讀過部位的策略是在猜。

---

## 已知缺口(不要以為這些能用)

**這一節會過期 —— 每一條都自己重驗一次再相信。**

| 缺口 | 影響 |
|---|---|
| live 執行層吃 `trades.jsonl`(成交記錄,不是訊號) | **交易所端沒有掛保護單**,停損只在程式內模擬,程式掉了部位就沒保護 |
| `ExitPlan.exchange_managed` | 欄位存在,**沒有任何消費端實作** |
| paper session 型別檢查 `isinstance(plugin, BlockStrategyPlugin)` | v2 StandardBot **上不了 paper/live** |
| `cyqnt_trd/standard_bot/yaml_pipeline/_data.py` 自己取數 | 沒有走 `build_input_bundle`,所以 YAML 回測的 PIT 閘門與 bundle 那條不同源 |
| 選幣無法回測 | `universe`/`ticker_rank`/`news` 在 catalog 標 FORWARD_ONLY,**沒有 PIT 歷史** |
| `build_intents` 跳過 `kind != TRADE` | **選幣候選變不成訂單** |
| `cyqnt.signal/v1` 未標 deprecated | 舊策略還在用,4 種輸出格式並存 |

### 想知道策略「實際」看到什麼,要探測 plugin,不要探測 helper

這是最容易搞錯的一種驗證。直接呼叫 `bars_to_df()` 只會拿到 OHLCV,於是很容易誤判成
「多源資料到不了策略」—— 但 `BlockStrategyPlugin._attach_frame_columns`
(`cyqnt_trd/blocks/strategy.py:424`)會把 `DataSnapshot.frames` as-of merge 到 bar 索引上。
**要量的是 plugin 交給 `make_signals` 的那個 df:**

```bash
.venv-standard-bot/bin/python - <<'PY'
import pandas as pd
from cyqnt_trd.blocks import strategy as BS
from cyqnt_trd.standard_bot.data import load_input_bundle

snap = load_input_bundle("docs/standard_bot_io/samples/input_bundle_example.json")
seen = {}
def probe(df):                      # 假策略,只為了看見 df
    seen["cols"] = list(df.columns)
    empty = pd.Series(False, index=df.index)
    return empty, empty

# MarketBundle 的 key 是 "SYMBOL|TIMEFRAME"(分隔符是 |,不是 :)。
# 別自己拼字串 —— 直接讀第一根 bar 的欄位,它一定對。
bars = snap.require_market().bars
first = bars[list(bars)[0]][0]
BS.build_plugin("probe", probe).step(
    snap, {}, {"instrument_id": first.instrument_id, "timeframe": first.timeframe})

print("bundle frames:", sorted(snap.frames or {}))
print("make_signals 收到 %d 欄:" % len(seen["cols"]), seen["cols"])
PY
```

寫這份文件時的實測結果:bundle 有 7 個 frame,`make_signals` 收到 **22 欄**,其中包含
`funding_rate` / `open_interest` / `open_interest_value`。
**注意 `news`(EventFrame)沒有變成欄位** —— 只有數值型 frame merge 得上 bar 時鐘,
事件型要靠 v2 的 `ctx.frames` 讀。這個差異值得自己再確認一次。

---

## 工作習慣

- **改動前後都跑全套 `pytest tests -q`**,並在報告裡寫出前後數字
- 加新功能時**問一句「這是新增一條路,還是取代一條路?」**。這個 repo 的複雜度幾乎全部來自
  只加不刪
- 對自己找到的結論做反事實驗證。「測試通過」不等於「功能在作用」
- 不確定某條路是否活著 → 用上面的 `grep` 查呼叫者,不要從 docstring 推論
- 講數字要附上產生它的指令
```
