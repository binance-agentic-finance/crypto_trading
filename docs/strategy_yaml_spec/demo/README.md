# NL → YAML → 訊號 / 回測 Demo

一個瀏覽器展示:用**你的語言模型(LiteLLM / OpenAI 相容)**把一句自然語言分流成交易或選幣
YAML。交易策略可回測或產生當下訊號；選幣需求會抓 Binance Square 的 universe / ticker rank，
經 Blocks 排名後輸出 `cyqnt.signal/v2 kind=selection`。這是「語言模型轉換 + 確定性執行」，不是 agent。

## 啟動

```bash
# 用有裝好依賴的 python(pandas/numpy/requests/pyyaml/websockets)
PYTHONPATH=/Users/hankchung/Dev/crypto_trading-main \
  .venv-standard-bot/bin/python docs/strategy_yaml_spec/demo/server.py
# 開瀏覽器:http://127.0.0.1:8799
```

## 用法

1. **LLM 設定**:填 API Base URL(例 `http://localhost:4000/v1`)、API Key、Model。
   後端會打 `{Base}/chat/completions`；key 會轉送至該端點，但不寫入 localStorage。
2. **自然語言**:描述交易規則或選幣需求 → 按「轉換成 YAML」→ LLM 回傳 YAML,並即時驗證。
3. **交易**:YAML 留在交易區，人工確認後可執行回測或產生訊號。
4. **選幣**:YAML 自動送到選幣區，接著抓即時宇宙與 Square 熱度並輸出候選訊號。

LLM 輸出不會被直接信任。後端會先獨立判斷需求是 `trade`、`selection` 或模糊需求，
再核對 YAML 的頂層種類、使用的資料節點、Blocks 參數、候選數量與使用者指定標的。
例如，新聞選幣若沒有真的接入 `ticker_rank` 或沒有用 `news_*` 欄位排名，會被拒絕；
選幣需求若被模型改寫成單一 SUI 技術策略，也會被拒絕且不執行。無法可靠分流時會停止並要求
釐清，不再預設為技術分析。

## 資料來源

- 回測資料:Binance 公開 K 線(依 YAML 的 `symbol` / `interval` / `market_type`)。
- 選幣資料:live bundle 的 `universe` + Binance Square `ticker_rank`（目前為最近 24 小時）。
- 抓不到時,若 symbol/interval 命中 repo fixtures(BTCUSDT 1h/4h/15m、ETHUSDT 1h)會自動離線回退。
- 基準:同區間 BTCUSDT buy & hold,初始資金全押。

## 端點

- `GET /` 前端 · `GET /api/schema` 轉換用 schema
- `POST /api/convert` `{nl, api_base, api_key, model}` → `{yaml, strategy_kind, valid, errors}`
- `POST /api/backtest` `{yaml}` → `{metrics, baseline, chart, trades_sample}`
- `POST /api/signal` `{yaml}` → `cyqnt.signal-batch/v1` + 最新 trade signal（若有觸發）
- `POST /api/selection` `{yaml}` → `cyqnt.signal-batch/v1` + `cyqnt.signal/v2 kind=selection`

## 注意

- 回測用向量化引擎 `run_vectorized_backtest`(訊號向量化 + numpy 出場迴圈)。
- 不涉及真實下單；這個 demo 只產生回測結果或標準訊號。
- 「少見／尚未被市場發現」與「一定會漲」目前沒有可直接證明的資料欄位。系統只能以
  Square 提及量、新聞情緒與流動性作代理，並會把這個限制列為 warning。
- 目前自然語言選幣已接通 Square 熱度／情緒、24h `quote_volume`，以及跨幣別
  funding snapshot（`with: [funding]`）。funding 排名目前是由高到低；「最負／最低」
  會明確回 `unsupported`，直到 schema 有 ascending 排名。跨幣別 open interest 與漲跌幅
  的自然語言路徑仍未接通，也不會被自動替換成新聞排行。單一標的交易策略仍可透過已宣告的
  derivatives frame 使用歷史 funding/OI。
