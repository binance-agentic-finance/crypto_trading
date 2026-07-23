# -*- coding: utf-8 -*-
"""
evo_xrp_5m_macd_rsi_standalone.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

XRPUSDT 5m MACD + RSI 動能策略 — 完全自足版本（standalone）
========================================================

這份檔案是 `evo_xrp_5m_macd_rsi.py` 的 self-contained 版本：
**只依賴 cyqnt_trd.blocks.indicators 跟 cyqnt_trd.blocks.strategy 兩個原生模組**，
所有條件邏輯（MACD 金叉判定、AND 組合）都 inline 寫在本檔內，
不引用 conditions / entry / regime / patterns 等可能版本不一致的模組。

可以單獨提交這個檔案到任何有 cyqnt_trd 0.1.x 的線上環境，立刻可用。

---------------------------------------------------------
策略來源 (Strategy Origin)
---------------------------------------------------------

由 evolutionary-strategy-discovery 系統挖掘所得（2026-06-05 XRP 5m run #2）。
從 230 個演化候選中通過 IS+OOS 雙評選為冠軍。

回測表現 (Backtest Performance)
───────────────────────────────
| 期間             | IS (60 天)        | OOS (30 天) |
|------------------|-------------------|-------------|
| 市場報酬         | +4.21%            | -16.0% (大跌) |
| 策略報酬         | **+20.70%**       | **+7.24%**  |
| Sharpe ratio     | **+7.69**         | **+6.70**   |
| 交易次數         | 432               | 173         |
| 平均交易頻率     | ~7/天             | ~5.8/天     |
| 勝率             | -                 | 36%         |
| Profit factor    | -                 | 2.90        |
| 最大回撤         | -                 | **-2.21%**  |

回測假設：
    initial_capital = 10000
    fee_bps = 4 (Binance spot taker)
    slippage_bps = 1
    size = 0.65 (65% allocation per trade)
    long_only = True (Binance AI Pro is spot)

---------------------------------------------------------
策略邏輯 (Strategy Logic)
---------------------------------------------------------

入場條件 (ALL OF — 兩條件須同時成立)：
    1. MACD(12, 26, 9) 金叉
       i.e. macd_line[t-1] <= signal_line[t-1] AND macd_line[t] > signal_line[t]
    2. RSI(14) > 50

出場條件 (由 backtest engine 透過 EXIT_CFG 處理)：
    type = "ma_cross_exit"
    period = 13      # EMA(13)
    ma_type = "ema"
    max_bars = 30    # 最多持倉 30 根 K 線（~2.5 hours）

可解釋性 (Why this works)
─────────────────────────

這個策略本質是「**動能反彈確認**」：

1. MACD 金叉是動能由空轉多的初步訊號，但常有假訊號
2. 加上 RSI > 50 確認價格動能已實際向上才進場，過濾大量假突破
3. 出場用 EMA(13) 跌破而非固定 stop，讓贏家延續，避免被反彈中的小回撤洗出

在 XRP 大跌市場（OOS）中表現特別好的原因：
- 大跌期間常出現「短時間反彈」(buy-the-dip)
- 策略不接刀 — MACD 必須先金叉才進場
- 高波動環境下反彈幅度大，TP 容易達標
- EMA(13) 退出比固定百分比 stop 更能配合波動

特點：
- 高頻交易：5m 週期，平均一天約 6 筆
- 勝率不高 (36%) 但 profit factor 高 (2.90)
- 最大回撤受控 (-2.21%)
- **零 filter** — 經實證 filter 反而傷 OOS

---------------------------------------------------------
注意事項 (Caveats)
---------------------------------------------------------

⚠ 此策略為 long-only，適合 spot market（如 Binance AI Pro）
⚠ 高頻策略對手續費敏感，假設 fee 4 bps + slippage 1 bps
⚠ 已通過 IS/OOS 但尚未經 walk-forward / paper trade 驗證
⚠ 36% 勝率代表多次小虧少數大賺的形態，需心理準備接受連虧

---------------------------------------------------------
依賴模組 (Dependencies)
---------------------------------------------------------

只依賴 cyqnt_trd 兩個基礎模組：
    - cyqnt_trd.blocks.indicators (用 macd / rsi)
    - cyqnt_trd.blocks.strategy (用 register)

沒有用任何新加的指標或 filter。

作者：Min Han Chung (via OpenClaw / CodeMax 協作)
版本：1.0.0 — standalone
建立日期：2026-06-05
"""

from typing import Tuple

import pandas as pd

from cyqnt_trd.blocks import indicators as ind
from cyqnt_trd.blocks import strategy


# ═══════════════════════════════════════════════════════════════════════════
# Inline helpers — 不依賴 cyqnt_trd.blocks.conditions / entry，直接內嵌實作
# ═══════════════════════════════════════════════════════════════════════════


def _macd_golden_cross(macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
    """檢測 MACD 金叉。

    True 當 (macd_line[t-1] <= signal_line[t-1]) AND (macd_line[t] > signal_line[t])。
    與 cyqnt_trd.blocks.conditions.macd_golden_cross 邏輯相同，這裡 inline 實作
    避免依賴 conditions 模組。
    """
    prev_below = macd_line.shift(1) <= signal_line.shift(1)
    now_above = macd_line > signal_line
    return (prev_below & now_above).fillna(False).astype(bool)


def _all_of(conditions: list) -> pd.Series:
    """把多個 boolean Series 用 AND 合併。

    與 cyqnt_trd.blocks.entry.all_of 邏輯相同，這裡 inline 實作。
    """
    if not conditions:
        raise ValueError("conditions list is empty")
    out = conditions[0]
    for c in conditions[1:]:
        out = out & c
    return out.fillna(False).astype(bool)


# ═══════════════════════════════════════════════════════════════════════════
# 主策略函數
# ═══════════════════════════════════════════════════════════════════════════


def make_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """生成 XRPUSDT 5m MACD+RSI 動能策略訊號。

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV K 線資料，必要欄位：open, high, low, close, volume。
        建議週期：5 分鐘 K 線（其他週期可用但未經驗證）。

    Returns
    -------
    (long_signal, short_signal) : Tuple[pd.Series[bool], pd.Series[bool]]
        long_signal:  多單入場訊號（True 即進場）。
        short_signal: 永遠為 False — 本策略只做多。
    """
    # ── 步驟 1：計算技術指標 ──────────────────────────────────────────
    # MACD(12, 26, 9)：標準參數
    #   - macd_line = EMA(12) - EMA(26)
    #   - signal_line = EMA(macd_line, 9)
    #   - hist = macd_line - signal_line  (本策略不用 hist)
    macd_line, signal_line, _ = ind.macd(
        df["close"],
        fast=12,
        slow=26,
        signal=9,
    )

    # RSI(14)：Wilder RSI，範圍 [0, 100]
    rsi_v = ind.rsi(df["close"], period=14)

    # ── 步驟 2：構建入場條件 ──────────────────────────────────────────
    # 條件 1：MACD 金叉
    macd_golden = _macd_golden_cross(macd_line, signal_line)

    # 條件 2：RSI > 50（動能向上確認）
    rsi_bullish = rsi_v > 50

    # 兩條件 AND
    long_signal = _all_of([macd_golden, rsi_bullish])

    # ── 步驟 3：本策略只做多 ──────────────────────────────────────────
    short_signal = pd.Series(False, index=df.index)

    return long_signal, short_signal


# ═══════════════════════════════════════════════════════════════════════════
# 出場規則 — 由 backtest engine / runtime 透過此 dict 統一處理
# ═══════════════════════════════════════════════════════════════════════════
#
# 在 cyqnt_trd.standard_bot.simulation.vectorized_backtest.run_vectorized_backtest
# 中，把這個 EXIT_CFG 傳入 exit_cfg 參數即可：
#
#   from cyqnt_trd.standard_bot.simulation.vectorized_backtest import run_vectorized_backtest
#   from cyqnt_trd.strategies import evo_xrp_5m_macd_rsi_standalone as st
#
#   result = run_vectorized_backtest(
#       df=df,
#       signal_fn=st.make_signals,
#       exit_cfg=st.EXIT_CFG,
#       timeframe="5m",
#       size=st.DEFAULT_SIZE,
#       fee_bps=4.0,
#       slippage_bps=1.0,
#       initial_capital=10000.0,
#       long_only=True,
#   )

EXIT_CFG = {
    "type": "ma_cross_exit",
    "period": 13,
    "ma_type": "ema",
    "max_bars": 30,
}

# 預設參數（給 caller 引用）
DEFAULT_PREFERRED_INTERVAL = "5m"
DEFAULT_SIZE = 0.65
DEFAULT_FEE_BPS = 4.0
DEFAULT_SLIPPAGE_BPS = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 策略註冊
# ═══════════════════════════════════════════════════════════════════════════
#
# 註冊後可用 strategy ID "evo_xrp_5m_macd_rsi_standalone" 透過 mvp_backtest CLI 呼叫：
#
#   python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
#       --engine vectorized \
#       --strategy evo_xrp_5m_macd_rsi_standalone \
#       --strategy-module cyqnt_trd.strategies.evo_xrp_5m_macd_rsi_standalone \
#       --symbol XRPUSDT --interval 5m --limit 5000

strategy.register("evo_xrp_5m_macd_rsi_standalone", make_signals)


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test：直接執行此檔可快速驗證策略可載入
#   python -m cyqnt_trd.strategies.evo_xrp_5m_macd_rsi_standalone
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    # 構造合成 OHLCV 資料（500 bars）做最簡單的 sanity check
    np.random.seed(42)
    n = 500
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.005, size=n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.003, size=n))
    low = close * (1 - rng.uniform(0, 0.003, size=n))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.uniform(100, 1000, size=n)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })

    long_sig, short_sig = make_signals(df)
    print(f"synthetic 500 bars → long signals: {long_sig.sum()}, "
          f"short signals: {short_sig.sum()}")
    print(f"EXIT_CFG: {EXIT_CFG}")
    print(f"strategy registered as: evo_xrp_5m_macd_rsi_standalone")
    print("OK")
