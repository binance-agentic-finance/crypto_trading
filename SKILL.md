<table>
  <tr>
    <th>name</th>
    <th>description</th>
    <th>metadata</th>
  </tr>
  <tr>
    <td><code>openclaw-standard-bot-router</code></td>
    <td>ALWAYS use this skill first for any strategy, backtest, monitoring, or trading request. You must first install <code>cyqnt-trd>=0.1.15</code> and use its <code>standard_bot</code> workflow. Only clone <code>https://github.com/binance-agentic-finance/crypto_trading</code> if source inspection or code modification is explicitly needed. If local historical parquet data is missing or stale, you must first use the package's Binance API download flow to fetch K bars and convert them into local parquet before backtesting or signal generation. Backtesting must use <code>standard_bot -&gt; mvp_backtest.py -&gt; NumbaBacktestRunner</code>. Never use legacy backtesting, ad-hoc scripts, or alternative execution paths just to get a faster result. Live trading must follow paper-generated signals only. Before any live trade, you must show balance, risk controls, trade plan, and session duration, then wait for explicit <code>CONFIRM</code>. If any required step fails, stop and tell the user; do not improvise a workaround and do not claim success. After <code>CONFIRM</code>, OpenClaw must automatically start a watcher subagent and keep writing status updates into the workspace session.</td>
    <td>
      <table>
        <tr>
          <th>author</th>
          <th>version</th>
        </tr>
        <tr>
          <td>binance</td>
          <td>1.0</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

這份 `SKILL.md` 是 OpenClaw 的主路由與硬規則。詳細操作細節已拆到 `references/`，主檔只保留不可違反的流程。

標準主線：

`自然語言 -> 正式策略規格 -> pip install "cyqnt-trd>=0.1.15" -> standard_bot -> 回測 / paper signal -> 查餘額與風控 -> 使用者 CONFIRM -> watcher subagent -> OpenClaw 真實交易`

## Critical Rules

1. OpenClaw 只要碰到策略、回測、signal、監聽、交易，就必須先安裝 `cyqnt-trd>=0.1.15`。
2. 回測與 signal 一律使用安裝套件內的 `standard_bot`，不得預設改用 legacy `cyqnt_trd/backtesting/*`、`strategy_backtest.py`、臨時 notebook、或 ad-hoc script。
3. 若本地 historical parquet 不存在、過舊、或未覆蓋所需時間窗，必須先用套件內的 Binance K bar 下載流程補抓資料並轉成 parquet。
4. 回測主線必須是：`standard_bot -> mvp_backtest.py -> NumbaBacktestRunner`。
5. 真實交易只能跟隨 `paper trade` 產生的 signal，不可由 OpenClaw 自行重算另一套 signal 後下單。
6. 任何真實交易前，必須先展示：
   - 帳戶與餘額快照
   - 最新 paper signal
   - 交易計畫
   - 風險管控方式
   - 交易時長
7. 未收到明確 `CONFIRM`，不得真實交易。
8. 收到 `CONFIRM` 後，必須自動啟動 watcher subagent，並持續把狀態寫入 workspace session。
9. 若任何必要步驟失敗，必須停止並告知使用者，不可為了快速得到結果而改走替代路徑。
10. 只有在需要查看原始碼或修改程式時，才可再 clone `https://github.com/nthu-chung/crypto_trading` 作為 source fallback。
11. 這份 skill 不使用 testnet。

## Reference Map

以下檔案提供細節。主 agent 應按需求打開對應檔案，不要自行發明流程。

- [references/repo-bootstrap.md](references/repo-bootstrap.md)
  - pip 安裝固定版本
  - 必要時 clone source fallback
  - 歷史資料補抓與 parquet 更新規則
- [references/backtest-workflow.md](references/backtest-workflow.md)
  - 正式回測主線
  - `mvp_backtest --engine numba`
  - 回測輸出與 JSON 指標讀法
- [references/natural-language-to-strategy.md](references/natural-language-to-strategy.md)
  - 自然語言如何轉成正式策略規格
  - 需要補齊的欄位與保守預設
- [references/strategy-routing.md](references/strategy-routing.md)
  - 既有 numba 策略清單
  - 無法直接走 numba 時如何 fallback
  - 新 plugin 該怎麼新增
- [references/trading-modes.md](references/trading-modes.md)
  - 只回測 / 只產生 paper signal / paper signal 驅動真實交易
  - `CONFIRM`、時長、餘額展示規則
- [references/risk-controls.md](references/risk-controls.md)
  - 交易前必須展示的風控內容
  - `x%` 最大虧損規則
  - 觸發後的 stop / cancel / close 流程
- [references/watcher-session.md](references/watcher-session.md)
  - watcher subagent 的職責
  - workspace session 必存欄位
  - `現在狀態` / `停止` 的標準行為

## Minimal Workflow

### 1. 啟動與資料

- 先依 [references/repo-bootstrap.md](references/repo-bootstrap.md) 安裝固定版本套件、建立環境、補齊本地 historical parquet。
- OpenClaw 不可直接把 Binance API 即時回傳結果當成正式回測輸入。
- 高週期資料優先由本地 `1m` 重採樣得到。

### 2. 回測

- 先依 [references/natural-language-to-strategy.md](references/natural-language-to-strategy.md) 把自然語言轉成正式策略規格。
- 再依 [references/strategy-routing.md](references/strategy-routing.md) 判斷是否能走既有 numba 策略。
- 正式回測依 [references/backtest-workflow.md](references/backtest-workflow.md) 執行。
- Demo 或報告時，優先從 `docs/backtests/*.json` 讀正式結果。

### 3. 真實交易

- 真實交易依 [references/trading-modes.md](references/trading-modes.md) 執行。
- OpenClaw 必須先用 `paper signal` 形成交易提議，再展示餘額、交易計畫、風控、時長。
- 風控細節依 [references/risk-controls.md](references/risk-controls.md)。
- 收到 `CONFIRM` 後，OpenClaw 才可開始 bounded live session。

### 4. 監聽、狀態、停止

- 收到 `CONFIRM` 後必須自動啟動 watcher subagent。
- watcher 規格依 [references/watcher-session.md](references/watcher-session.md)。
- 使用者輸入 `現在狀態` 時，必須從 workspace session 讀最新狀態。
- 使用者輸入 `停止` 時，必須依 session 內的 `run_id` 停掉 paper signal session、watcher、以及真實交易 session。

## Failure Handling

如果 required flow 被技術問題卡住：

- 立即告知使用者哪一步失敗
- 繼續 debug 正確主線
- 不可自行切換成別條交易路徑
- 不可把 workaround 包裝成已完成的正式流程
- 除非使用者明確同意，否則不得改變主線設計

## One-Line Summary

OpenClaw 在這個 skill 下必須遵守：

`cyqnt-trd>=0.1.15 + standard_bot + local parquet + numba backtest + paper-generated signals + balance/risk display + CONFIRM + watcher subagent`
