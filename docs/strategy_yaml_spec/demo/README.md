# NL → YAML → 回測 Demo

一個瀏覽器展示:用**你的語言模型(LiteLLM / OpenAI 相容)**把一句自然語言轉成策略 YAML,
直接跑回測,輸出 Sharpe / 回撤 / 報酬 / **PnL**,並對照 **BTC buy&hold 基準**。這是單純的
「語言模型轉換 + 回測」,不是 agent。

## 啟動

```bash
# 用有裝好依賴的 python(pandas/numpy/requests/pyyaml/websockets)
PYTHONPATH=/Users/hankchung/Dev/crypto_trading-main \
  <venv>/bin/python docs/strategy_yaml_spec/demo/server.py
# 開瀏覽器:http://127.0.0.1:8799
```

本次驗證用的 venv:
`/private/tmp/claude-501/-Users-hankchung-Dev/574ee522-7e94-46bd-bc91-99115831731c/scratchpad/venv311/bin/python`

## 用法

1. **LLM 設定**:填 API Base URL(例 `http://localhost:4000/v1`)、API Key、Model。
   後端會打 `{Base}/chat/completions`;key 只送到本機後端。
2. **自然語言**:描述策略 → 按「轉換成 YAML」→ LLM 回傳 YAML,並即時驗證。
3. **YAML**:可手動改;按「執行回測」。
4. **結果**:指標卡(總報酬 / PnL / BTC 基準 / 超額 / Sharpe / 回撤 / 勝率 / 次數)+ 權益曲線 vs BTC 基準圖。

## 資料來源

- 回測資料:Binance 公開 K 線(依 YAML 的 `symbol` / `interval` / `market_type`)。
- 抓不到時,若 symbol/interval 命中 repo fixtures(BTCUSDT 1h/4h/15m、ETHUSDT 1h)會自動離線回退。
- 基準:同區間 BTCUSDT buy & hold,初始資金全押。

## 端點

- `GET /` 前端 · `GET /api/schema` 轉換用 schema
- `POST /api/convert` `{nl, api_base, api_key, model}` → `{yaml, valid, errors}`
- `POST /api/backtest` `{yaml}` → `{metrics, baseline, chart, trades_sample}`

## 注意

- 回測用向量化引擎 `run_vectorized_backtest`(訊號向量化 + numpy 出場迴圈)。
- 不涉及真實下單;純回測。
