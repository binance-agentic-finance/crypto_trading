# -*- coding: utf-8 -*-
"""
evo_xrp_5m_macd_rsi.py — XRPUSDT 5m MACD+RSI 動能策略
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

策略來源：
    由 evolutionary-strategy-discovery 系統挖掘所得（2026-06-05 XRP 5m run #2）。
    從 230 個演化候選中通過 IS+OOS 雙評選為冠軍，OOS 期間為 -16% 大跌市場
    時仍能獲取 +7.24% 報酬，OOS Sharpe +6.70。

核心邏輯：
    1. MACD(12,26,9) 出現金叉 — 動能由空轉多
    2. RSI(14) > 50 — 確認多頭動能足夠強
    3. 兩條件 AND，皆需成立才進場
    4. 持倉中當收盤價跌破 EMA(13) 即出場（彈性追蹤止損）

為什麼有效（可解釋性）：
    這個策略本質是「**動能反彈確認**」。MACD 金叉是動能轉正的初步訊號，
    但常有假訊號；加上 RSI > 50 確認價格動能已實際向上後才進場，可過濾
    掉很多假突破。出場用 EMA(13) 跌破而非固定百分比，讓贏家可以延續，
    同時避免被反彈中的小回撤洗出。

    在 XRP 5m 大跌市場（OOS）中，此策略表現特別好的原因：
    - 大跌期間常出現「短時間反彈」，MACD 金叉 + RSI>50 抓的就是這種反彈
    - 高波動環境下反彈幅度大，TP 容易達標
    - EMA(13) 退出比固定 stop 更能配合波動

特點：
    - 高頻交易：5m 週期，平均一天 ~6 筆
    - 勝率不高（IS 36%）但盈虧比佳（profit factor 2.90）
    - 最大回撤受控（OOS DD 僅 -2.21%）
    - 不需要任何 regime / volatility filter — 經實證 filter 反而傷 OOS

回測表現：
    IS  (60 天):  +20.70% return, Sharpe +7.69, 432 trades
    OOS (30 天，市場 -16%):
                +7.24% return, Sharpe +6.70, 173 trades, DD -2.21%

注意事項：
    ⚠ 此策略為 long-only（適合 spot market）
    ⚠ 高頻策略對手續費敏感，假設 fee 4 bps + slippage 1 bps 計算
    ⚠ 經過 IS/OOS 驗證但尚未經 walk-forward / paper trade 驗證
    ⚠ 36% 勝率代表多次小虧少數大賺的形態，需要心理準備接受連虧

作者：Evolutionary Discovery System (Min Han Chung)
版本：1.0.0
建立日期：2026-06-05
"""

from typing import Tuple

import pandas as pd

from cyqnt_trd.blocks import conditions as cond
from cyqnt_trd.blocks import indicators as ind
from cyqnt_trd.blocks import strategy
from cyqnt_trd.blocks.entry import all_of


def make_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """生成 XRPUSDT 5m MACD+RSI 動能策略訊號。

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV K 線資料。必要欄位：open, high, low, close, volume。
        建議週期：5 分鐘 K 線。

    Returns
    -------
    (long_signal, short_signal) : Tuple[pd.Series[bool], pd.Series[bool]]
        long_signal: 多單入場訊號（True 即進場）。
        short_signal: 永遠為 False — 本策略只做多。
    """
    # ── 計算技術指標 ──────────────────────────────────────────────
    # MACD(12, 26, 9)：標準參數，捕捉中短期動能轉折
    macd_line, signal_line, _ = ind.macd(df["close"], fast=12, slow=26, signal=9)

    # RSI(14)：確認動能方向
    rsi_v = ind.rsi(df["close"], period=14)

    # ── 入場條件 ───────────────────────────────────────────────────
    # 條件 1: MACD 金叉 — 動能由空轉多
    macd_golden = cond.macd_golden_cross(macd_line, signal_line)

    # 條件 2: RSI > 50 — 動能已確認向上
    rsi_bullish = rsi_v > 50

    # 兩條件 AND
    long_signal = all_of([macd_golden, rsi_bullish])

    # 不做空
    short_signal = pd.Series(False, index=df.index)

    return long_signal.fillna(False).astype(bool), short_signal


# ── 出場規則（由 backtest engine / runtime 統一處理）──────────────────
# 出場類型: ma_cross_exit
# 出場參數: {
#     "type": "ma_cross_exit",
#     "period": 13,
#     "ma_type": "ema",
#     "max_bars": 30,
# }
#
# 即：當 close 跌破 EMA(13) 時平倉，或最多持倉 30 根 K 線（~2.5 hours）。
# 在 cyqnt_trd.standard_bot.simulation.vectorized_backtest 中傳入此 exit_cfg。

EXIT_CFG = {
    "type": "ma_cross_exit",
    "period": 13,
    "ma_type": "ema",
    "max_bars": 30,
}

DEFAULT_PREFERRED_INTERVAL = "5m"
DEFAULT_SIZE = 0.65


# ── 策略註冊 ───────────────────────────────────────────────────────
strategy.register("evo_xrp_5m_macd_rsi", make_signals)
