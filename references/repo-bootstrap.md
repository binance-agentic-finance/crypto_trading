# Package Bootstrap And Data Refresh

這份文件說明 OpenClaw 如何為 `cyqnt-trd>=0.1.15` 準備執行環境與最新歷史資料。

## 1. Install Fixed Package Version

只要任務碰到：

- 自然語言轉策略
- 回測
- paper signal
- 監聽
- 依照策略訊號交易

就先確保固定版本套件已可用：

```bash
python3 -m venv .venv-standard-bot
source .venv-standard-bot/bin/activate
python -m pip install -U pip
python -m pip install "cyqnt-trd>=0.1.15"
```

## 2. Optional Source Fallback

只有在需要查看原始碼、除錯、或修改程式時，才可額外 clone source repo：

```bash
REPO_DIR="${REPO_DIR:-$PWD/crypto_trading}"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/nthu-chung/crypto_trading "$REPO_DIR"
fi

cd "$REPO_DIR"
git pull --ff-only
```

若 source repo 被 clone，下列命令仍應優先在已啟用的 `.venv-standard-bot` 中執行。

## 3. 歷史資料補齊規則

OpenClaw 在回測、paper signal、monitor 前，必須先確認本地 historical parquet 是否足夠新、且覆蓋所需時間窗。

若本地資料不存在、過舊、或範圍不足：

- 不可直接拿過期 parquet 硬跑
- 不可直接把 Binance API 即時回傳結果當成最終回測輸入
- 必須先使用安裝套件或 source repo 內的資料下載流程抓取 Binance K bar
- 必須先把下載結果存成或更新為本地 parquet
- 高週期資料應優先由本地 `1m` 重採樣產生

標準資料主線：

`Binance API -> package/source downloader -> local parquet -> local resample -> standard_bot`

## 4. 強制原則

OpenClaw 一旦要做回測或 signal 驗證，必須優先使用：

- `python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest`
- `python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper`
- `python -m cyqnt_trd.standard_bot.entrypoints.mvp_monitor_http`
- `python -m cyqnt_trd.standard_bot.entrypoints.mvp_run_manager`

不得預設改用：

- `cyqnt_trd/get_data/*` 當正式回測入口
- `cyqnt_trd/backtesting/*`
- `cyqnt_trd/backtesting/strategy_backtest.py`
- 臨時 notebook / 臨時 pandas 回測腳本

若需要描述回測核心，標準說法應為：

`standard_bot -> mvp_backtest.py -> NumbaBacktestRunner`
