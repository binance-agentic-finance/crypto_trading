"""T1 — Structured Trade Plan(結構化交易計畫)。

輸入 symbol 的 OHLCV → 立即回一張可執行計畫
(方向 / 信心 / verdict / 進場區 / 停損 / 分批止盈 / 倉位),免跑完整回測。

輸出:cyqnt.signal/v1(kind=trade)。骨架/輸出固定,只有 CONFIG(blocks/參數)可調。
組裝邏輯在 cyqnt_trd.blocks.trade_plan.build_trade_plan。
"""
from __future__ import annotations
from typing import Dict, List
import pandas as pd
from cyqnt_trd.blocks import trade_plan as TP

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


def generate(df: pd.DataFrame, symbol: str, interval: str,
             market_type: str = "futures") -> List[Dict]:
    """回傳 [一張當前交易計畫](cyqnt.signal/v1, kind=trade)。"""
    return [TP.build_trade_plan(df, symbol, interval, market_type=market_type, cfg=CONFIG)]
