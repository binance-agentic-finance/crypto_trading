"""T2 — Multi-Timeframe Trend Follow(參數化模版)。

設計重點:**骨架與輸出固定,只有 CONFIG(blocks/參數)可調**。
- 輸入:單標的 OHLCV df(欄位 open/high/low/close/volume/close_time)。
- 輸出:`cyqnt.signal/v1`(kind=trade)—— 對所有標的×週期都是同一個規格。
- 同時提供 make_signals(df)->(long,short) 走現有 backtest/paper/live 引擎。

換標的/週期 = 換傳給 generate() 的 symbol/interval(值,不改結構)。
換策略邏輯 = 只改 CONFIG(indicators/entry/exit 的 blocks 與參數)。
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import pandas as pd
from cyqnt_trd.blocks import indicators as I

BOT_ID = "mtf_trend_follow"
VERSION = "v1"

# ===== 唯一可調處:blocks + 參數(骨架/輸出不動)=====
CONFIG = {
    "ema_fast": 20,
    "ema_slow": 50,
    "ema_trend": 200,     # 趨勢濾網(單框變體;完整 MTF 版用 htf 4h)
    "adx_period": 14,
    "adx_min": 25.0,
    "atr_period": 14,
    "stop_mult": 2.0,
    "tp_mult": 3.0,
    "size_risk_pct": 0.01,
    "leverage": 5,
    "max_notional_usdt": 2000,
}

# interval → time_horizon(值的對映,schema 不變)
_HORIZON = {"1m": "scalp", "3m": "scalp", "5m": "scalp", "15m": "scalp",
            "30m": "intraday", "1h": "intraday", "2h": "intraday", "4h": "swing",
            "6h": "swing", "8h": "swing", "12h": "swing", "1d": "swing"}


def _factors(df: pd.DataFrame, cfg: dict):
    close = df["close"]
    ema_f = I.ema(close, cfg["ema_fast"])
    ema_s = I.ema(close, cfg["ema_slow"])
    ema_t = I.ema(close, cfg["ema_trend"])
    adx = I.adx(df, cfg["adx_period"])[0]           # tuple → 取 adx
    atr = I.atr(df, cfg["atr_period"])
    return close, ema_f, ema_s, ema_t, adx, atr


def make_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """(long, short) 布林 Series —— 走現有引擎。"""
    c, ef, es, et, adx, _atr = _factors(df, CONFIG)
    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    cross_dn = (ef < es) & (ef.shift(1) >= es.shift(1))
    trend_up, trend_dn = c > et, c < et
    strong = adx > CONFIG["adx_min"]
    long = (cross_up & trend_up & strong).fillna(False)
    short = (cross_dn & trend_dn & strong).fillna(False)
    return long, short


def inputs() -> Dict[str, object]:
    """標準輸入宣告。"""
    return {"kline": ["<primary>"], "htf": [], "news": False, "funding": False, "oi": False}


def generate(df: pd.DataFrame, symbol: str, interval: str,
             market_type: str = "futures") -> List[Dict]:
    """標準輸出:list[cyqnt.signal/v1](kind=trade)。規格對所有 symbol×interval 一致。"""
    c, ef, es, et, adx, atr = _factors(df, CONFIG)
    long, short = make_signals(df)
    horizon = _HORIZON.get(interval, "intraday")
    out: List[Dict] = []
    for i in range(len(df)):
        is_long, is_short = bool(long.iloc[i]), bool(short.iloc[i])
        if not (is_long or is_short):
            continue
        px = float(c.iloc[i]); a = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
        side = "long" if is_long else "short"
        if side == "long":
            stop = px - CONFIG["stop_mult"] * a
            tp = px + CONFIG["tp_mult"] * a
        else:
            stop = px + CONFIG["stop_mult"] * a
            tp = px - CONFIG["tp_mult"] * a
        ts = int(df["close_time"].iloc[i])
        out.append({
            "schema": "cyqnt.signal/v1", "kind": "trade",
            "bot_id": BOT_ID, "ts": ts,
            "symbol": symbol.upper(), "market_type": market_type,
            "side": side, "action": "entry",
            "confidence": round(min(1.0, float(adx.iloc[i]) / 50.0), 3),
            "entry": {"type": "market", "zone": [round(px, 6), round(px, 6)]},
            "stop_loss": round(stop, 6),
            "take_profit": [{"price": round(tp, 6), "close_pct": 1.0}],
            "size": {"mode": "risk_pct", "value": CONFIG["size_risk_pct"],
                     "leverage": CONFIG["leverage"], "max_notional_usdt": CONFIG["max_notional_usdt"]},
            "time_horizon": horizon, "valid_until": ts + _interval_ms(interval),
            "reason": f"EMA{CONFIG['ema_fast']}x{CONFIG['ema_slow']} 交叉 + 收盤在 EMA{CONFIG['ema_trend']} {'上' if side=='long' else '下'}方 + ADX>{CONFIG['adx_min']}",
            "provenance": {"strategy_id": BOT_ID, "version": VERSION,
                           "inputs": ["kline"]},
        })
    return out


def _interval_ms(interval: str) -> int:
    unit = interval[-1]; n = int(interval[:-1])
    return n * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}.get(unit, 3_600_000)


# 走現有引擎(backtest/paper/live);exit 用 ATR 停損停利
try:
    from cyqnt_trd.blocks import strategy as _strat
    _strat.register(BOT_ID, make_signals,
                    exit_cfg={"type": "atr_stop_tp", "atr_period": CONFIG["atr_period"],
                              "stop_mult": CONFIG["stop_mult"], "tp_mult": CONFIG["tp_mult"]},
                    size=0.1)
except Exception:
    pass  # 匯入即註冊;獨立 generate() 不依賴註冊
