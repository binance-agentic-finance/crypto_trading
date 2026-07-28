"""T1 — Structured Trade Plan(結構化交易計畫)。

輸入 symbol 的 OHLCV → 立即回一張可執行計畫
(方向 / 信心 / verdict / 進場區 / 停損 / 分批止盈 / 倉位),免跑完整回測。

輸出:cyqnt.signal/v1(kind=trade)。骨架/輸出固定,只有 CONFIG(blocks/參數)可調。
組裝邏輯在 cyqnt_trd.blocks.trade_plan.build_trade_plan。
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
from cyqnt_trd.blocks import trade_plan as TP
from cyqnt_trd.blocks import indicators as I

BOT_ID = "structured_trade_plan"
VERSION = "v1"

# 唯一可調處(轉交 build_trade_plan)
CONFIG: Dict = {
    "ema_fast": 20, "ema_slow": 50, "ema_trend": 200,
    "rsi_period": 14, "adx_period": 14, "atr_period": 14,
    "stop_mult": 2.0, "tp_mults": [1.5, 3.0], "tp_close": [0.5, 0.5],
    "balance": 10000.0, "risk_pct": 1.0, "max_leverage": 5.0,
}


def inputs() -> Dict[str, object]:
    return {"kline": ["<primary>"], "htf": [], "news": False, "funding": False, "oi": False}


def make_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """(long, short) 布林 Series —— 走現有 backtest/paper/live 引擎。

    方向用與 :func:`trade_plan.build_trade_plan` 相同的多因子投票(EMA 排列 +
    RSI + MACD 柱),但逐根向量化:votes>0 進多、votes<0 進空。停損/分批止盈
    由 register 的 exit_cfg 管(此輪為單一 atr_stop_tp;分批 TP 為後續引擎功能)。
    """
    c = CONFIG
    close = df["close"]
    ema_f = I.ema(close, c["ema_fast"]); ema_s = I.ema(close, c["ema_slow"]); ema_t = I.ema(close, c["ema_trend"])
    rsi = I.rsi(close, c["rsi_period"])
    _macd_line, _macd_sig, macd_hist = I.macd(close)
    votes = pd.Series(0, index=df.index)
    votes = votes + ((ema_f > ema_s) & (close > ema_t)).astype(int)
    votes = votes - ((ema_f < ema_s) & (close < ema_t)).astype(int)
    votes = votes + (rsi > 55).astype(int)
    votes = votes - (rsi < 45).astype(int)
    votes = votes + (macd_hist > 0).astype(int)
    votes = votes - (macd_hist < 0).astype(int)
    long = (votes > 0)
    short = (votes < 0)
    return long, short


def generate(df: pd.DataFrame, symbol: str, interval: str,
             market_type: str = "futures") -> List[Dict]:
    """匯出格式(給 JS 消費的 cyqnt.signal/v1 kind=trade 計畫,對最後一根)。

    注意:策略的**執行介面**是 make_signals + register(與既有策略一致);
    這個 generate() 只是框架/前端要用的匯出 JSON,不是作者要走的路線。
    """
    return [TP.build_trade_plan(df, symbol, interval, market_type=market_type, cfg=CONFIG)]


# 走現有引擎(與既有策略同一條路線);單一 ATR 停損停利(分批 TP 為後續引擎功能)。
from cyqnt_trd.blocks import strategy as _strat

_strat.register(
    BOT_ID, make_signals,
    exit_cfg={"type": "atr_stop_tp", "atr_period": CONFIG["atr_period"],
              "stop_mult": CONFIG["stop_mult"], "tp_mult": CONFIG["tp_mults"][-1]},
    size=0.1,
)
