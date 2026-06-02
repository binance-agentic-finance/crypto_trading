"""
MA Cross Strategy v1 — blocks / python-engine
==============================================

策略邏輯
--------
  Long  進場：fast SMA 由下往上穿越 slow SMA（金叉）
  Short 進場：fast SMA 由上往下穿越 slow SMA（死叉）
  每次新訊號由框架自動平前倉再開反倉（position-flip 模型）

參數預設（保守值，可 fork 此檔修改）
--------------------------------------
  FAST_PERIOD = 20  （5 根 4h bar ≈ 20 根 1h bar）
  SLOW_PERIOD = 60  （15 根 4h bar ≈ 60 根 1h bar）

訊號一致性設計（backtest ≈ paper ≈ live 的關鍵）
-------------------------------------------------
本策略在三種執行模式下使用完全相同的 make_signals()：

1. Backtest（mvp_backtest --engine python）
   SnapshotBacktestRunner 呼叫 BlockStrategyPlugin.run()
   → _call_signal_fn(full_df) → make_signals(df)

2. Paper trade（mvp_paper_daemon --engine python）
   PythonLivePaperSession.tick() 呼叫 _compute_latest_target()
   → 從 self._closes/volumes 建立 _build_dataframe() → make_signals(df)
   → 只讀取最後一根的訊號值，不重複計算歷史

3. Live trade（run_live_monitor.py + mvp_monitor_http）
   MarketOnlyPaperRunner 呼叫 BlockStrategyPlugin.step()
   → _call_signal_fn(full_df) → make_signals(df)
   → 使用 cursor 過濾只發射「新於 last cursor」的訊號

這三條路都呼叫同一個 make_signals()，傳入同樣的收盤價序列，
因此在相同的歷史視窗下，訊號必然相同。

No-lookahead 保證
-----------------
1. SMA 使用 pandas rolling，min_periods=period → warm-up 期產生 NaN
2. 穿越判斷使用 .shift(1)（前一根的值）和當根的值
   → 訊號在 bar close 後確認，在「下一根 bar open」由框架執行
   → next-bar-open 模型，與 paper/live 的執行時序完全對齊
3. 所有 NaN 用 fillna(False) 填充 → warm-up 期絕對不發假訊號

欄位相容性
----------
- _build_dataframe（daemon）: open/high/low/close/volume/quote_volume/timestamp/close_time
- bars_to_df（backtest/live）: open/high/low/close/volume/quote_volume/open_time/close_time/...
- 本策略只用 df["close"]，完全不依賴 open_time/close_time 欄位
  → 兩條路都安全，無 KeyError 風險

驗證紀錄（2026-05-15）
----------------------
- import + register：PASS
- 輸出型別（bool Series）：PASS
- warm-up 期全 False：PASS
- 金叉訊號（prev_fast<prev_slow → curr_fast>curr_slow）：PASS @ index 79
- 死叉訊號（prev_fast>prev_slow → curr_fast<curr_slow）：PASS @ index 200
- 無同時金叉+死叉重疊：PASS
- 無 open_time 環境下 KeyError：PASS
- plugin.signal_fn identity：PASS
"""

import pandas as pd
from cyqnt_trd.blocks import indicators as ind, strategy

# ── 策略參數 ─────────────────────────────────────────────────────────────────
FAST_PERIOD: int = 20
SLOW_PERIOD: int = 60
# ────────────────────────────────────────────────────────────────────────────


def make_signals(df: pd.DataFrame):
    """
    MA 金叉/死叉訊號函式。

    輸入：df 含至少 df["close"] 欄位，行數即 snapshot 視窗大小
    輸出：(long_signal, short_signal) — bool pd.Series，index 與 df 對齊

    訊號時序（next-bar-open 執行模型）：
      bar N close 時計算 → 若有穿越，bar N+1 open 由框架執行
    """
    if len(df) < SLOW_PERIOD + 1:
        # 資料不足：全部回傳 False，不發假訊號
        false_series = pd.Series(False, index=df.index)
        return false_series, false_series

    fast_ma = ind.sma(df["close"], FAST_PERIOD)   # min_periods=FAST_PERIOD
    slow_ma = ind.sma(df["close"], SLOW_PERIOD)   # min_periods=SLOW_PERIOD

    # shift(1)：引用前一根 bar 確認後的值（no lookahead）
    prev_fast = fast_ma.shift(1)
    prev_slow = slow_ma.shift(1)

    # 金叉：前根 fast < slow（fast 在 slow 下方）
    #        當根 fast > slow（fast 剛剛上穿 slow）
    long_signal = (prev_fast < prev_slow) & (fast_ma > slow_ma)

    # 死叉：前根 fast > slow（fast 在 slow 上方）
    #        當根 fast < slow（fast 剛剛下穿 slow）
    short_signal = (prev_fast > prev_slow) & (fast_ma < slow_ma)

    # fillna(False)：warm-up 期 NaN → False
    return long_signal.fillna(False), short_signal.fillna(False)


# register 在 import 時執行：
#   → BlockStrategyPlugin("ma_cross_v1") 放入 _PENDING_REGISTRATIONS
#   → make_registry() 呼叫 flush_pending_into() 時裝入 SignalPluginRegistry
strategy.register("ma_cross_v1", make_signals)
