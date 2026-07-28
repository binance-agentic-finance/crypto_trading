"""T3 — Derivatives Positioning(資金費率 / OI 定位訊號)。

使用合約專屬數據(資金費率 / 未平倉量)產生訊號。
做法:把 funding_rate + open_interest 掛進 K 線 df(glue),用 OI×價格背離定方向、
funding 極值做加碼確認。輸出 kind=trade(cyqnt.signal/v1)。

註:run loop 不供 funding/OI 欄 → 用 attach_derivatives() 這段 glue 先掛欄。
"""
from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
from cyqnt_trd.blocks import derivatives as D
from cyqnt_trd.blocks import indicators as I

BOT_ID = "deriv_positioning"
VERSION = "v1"

CONFIG = {
    "oi_lookback": 20,
    "funding_high_bps": 0.5,   # 真實 funding 常很小,門檻放寬才可用
    "funding_low_bps": -0.5,
    "atr_period": 14, "stop_mult": 2.0, "tp_mult": 3.0,
    "size_risk_pct": 0.01, "leverage": 3, "max_notional_usdt": 2000,
}
# oi 背離 → 方向
_LONG = {"bullish_buildup", "short_squeeze"}
_SHORT = {"bearish_buildup", "long_squeeze"}


def attach_derivatives(df: pd.DataFrame, funding_df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
    """glue:把 funding_rate / open_interest 以 lookahead-safe(backward asof)掛到 df。"""
    out = df.sort_values("close_time").copy()
    f = funding_df[["timestamp", "funding_rate"]].rename(columns={"timestamp": "close_time"}).sort_values("close_time")
    o = oi_df[["timestamp", "open_interest"]].rename(columns={"timestamp": "close_time"}).sort_values("close_time")
    out = pd.merge_asof(out, f, on="close_time", direction="backward")
    out = pd.merge_asof(out, o, on="close_time", direction="backward")
    return out


def inputs() -> Dict[str, object]:
    return {"kline": ["<primary>"], "funding": True, "oi": True, "htf": [], "news": False}


def _side_series(df: pd.DataFrame, cfg: dict) -> pd.Series:
    if "open_interest" not in df.columns:
        raise ValueError("df 缺 open_interest 欄 —— 請先 attach_derivatives()")
    div = D.oi_price_divergence(df["close"], df["open_interest"], lookback=cfg["oi_lookback"])
    return div.map(lambda s: "long" if s in _LONG else ("short" if s in _SHORT else "flat")), div


def make_signals(df: pd.DataFrame):
    """(long, short):在方向『轉變』的那一根觸發(避免每根重複)。"""
    side, _div = _side_series(df, CONFIG)
    changed = side != side.shift(1)
    long = ((side == "long") & changed).fillna(False)
    short = ((side == "short") & changed).fillna(False)
    return long, short


def generate(df: pd.DataFrame, symbol: str, interval: str,
             market_type: str = "futures", cfg: Optional[dict] = None) -> List[Dict]:
    c = dict(CONFIG); c.update(cfg or {})
    side, div = _side_series(df, c)
    changed = side != side.shift(1)
    atr = I.atr(df, c["atr_period"])
    fstate = D.funding_rate_state(df["funding_rate"], high_threshold_bps=c["funding_high_bps"],
                                  low_threshold_bps=c["funding_low_bps"]) if "funding_rate" in df.columns else None
    horizon = {"1h": "intraday", "4h": "swing", "15m": "scalp"}.get(interval, "intraday")
    out: List[Dict] = []
    for i in range(len(df)):
        s = side.iloc[i]
        if s == "flat" or not bool(changed.iloc[i]):
            continue
        px = float(df["close"].iloc[i]); a = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
        stop = px - c["stop_mult"] * a if s == "long" else px + c["stop_mult"] * a
        tp = px + c["tp_mult"] * a if s == "long" else px - c["tp_mult"] * a
        conf = 0.6
        fs = str(fstate.iloc[i]) if fstate is not None and pd.notna(fstate.iloc[i]) else "neutral"
        # funding 確認:負費率(bearish_squeeze,空頭付錢)偏多;正費率(bullish_squeeze,多頭付錢)偏空
        if (s == "long" and fs == "bearish_squeeze") or (s == "short" and fs == "bullish_squeeze"):
            conf = 0.75
        ts = int(df["close_time"].iloc[i])
        out.append({
            "schema": "cyqnt.signal/v1", "kind": "trade", "bot_id": BOT_ID, "ts": ts,
            "symbol": symbol.upper(), "market_type": market_type, "side": s, "action": "entry",
            "confidence": conf,
            "entry": {"type": "market", "zone": [round(px, 6), round(px, 6)]},
            "stop_loss": round(stop, 6),
            "take_profit": [{"price": round(tp, 6), "close_pct": 1.0}],
            "size": {"mode": "risk_pct", "value": c["size_risk_pct"],
                     "leverage": c["leverage"], "max_notional_usdt": c["max_notional_usdt"]},
            "time_horizon": horizon, "valid_until": ts + 3_600_000,
            "reason": f"OI×價格背離={div.iloc[i]};funding_state={fs}",
            "features": {"oi_divergence": str(div.iloc[i]), "funding_state": fs,
                         "open_interest": round(float(df['open_interest'].iloc[i]), 2) if 'open_interest' in df else None},
            "provenance": {"strategy_id": BOT_ID, "version": VERSION, "inputs": ["kline", "funding", "oi"]},
        })
    return out
