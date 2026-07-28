"""N2 — Social Heat Breakout(社群熱度暴增 + 技術突破選幣)。

依社群熱度暴增 + 技術突破選幣(敘事爆發型)。
兩段:① 選:Square 提及量『環比暴增』(heat delta)取 top-K;
      ② 確認:對候選拉 K 線,要求 Donchian 突破 + 量能放大才給多單。
輸出 kind=selection(cyqnt.signal/v1),通過技術確認的候選內嵌 trade 計畫。
"""
from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
from cyqnt_trd.blocks import indicators as I

BOT_ID = "social_heat_breakout"
VERSION = "v1"

CONFIG = {
    "top_k": 8,             # 熱度榜取前 K 做技術確認
    "min_delta": 50,        # 提及量環比至少 +N 才算暴增
    "breakout_window": 20,  # Donchian 突破視窗
    "vol_surge_mult": 1.5,  # 量能 > N×均量
    "atr_period": 14, "stop_mult": 2.0, "tp_mult": 3.0,
    "size_risk_pct": 0.01, "leverage": 3, "max_notional_usdt": 1500,
}


def inputs() -> Dict[str, object]:
    return {"universe": True, "news": True, "kline": ["<candidate 1h>"]}


def _base(sym: str) -> str:
    for q in ("USDT", "USDC", "USD"):
        if sym.upper().endswith(q):
            return sym.upper()[:-len(q)]
    return sym.upper()


def _breakout(df: pd.DataFrame, cfg: dict):
    """回傳 (confirmed, entry, stop, tp, feat)。Donchian 上緣突破 + 量能放大。"""
    if df is None or len(df) < cfg["breakout_window"] + cfg["atr_period"] + 2:
        return False, None, None, None, {"reason": "K線不足"}
    close = df["close"]; high = df["high"]; vol = df["volume"]
    donch_hi = high.rolling(cfg["breakout_window"]).max().shift(1)
    vol_ma = vol.rolling(cfg["breakout_window"]).mean()
    atr = I.atr(df, cfg["atr_period"])
    i = len(df) - 1
    px = float(close.iloc[i]); a = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
    brk = bool(pd.notna(donch_hi.iloc[i]) and px > float(donch_hi.iloc[i]))
    vsurge = bool(pd.notna(vol_ma.iloc[i]) and vol.iloc[i] > cfg["vol_surge_mult"] * float(vol_ma.iloc[i]))
    feat = {"breakout": brk, "vol_surge": vsurge,
            "donchian_hi": round(float(donch_hi.iloc[i]), 6) if pd.notna(donch_hi.iloc[i]) else None}
    if brk and vsurge:
        return True, px, round(px - cfg["stop_mult"] * a, 6), round(px + cfg["tp_mult"] * a, 6), feat
    return False, px, None, None, feat


def generate_selection(rank_now: pd.DataFrame, rank_prev: pd.DataFrame,
                       klines: Optional[Dict[str, pd.DataFrame]] = None, *,
                       as_of_ms: int = 0, market_type: str = "futures",
                       cfg: Optional[dict] = None) -> List[Dict]:
    c = dict(CONFIG); c.update(cfg or {})
    klines = klines or {}
    prev = rank_prev.set_index("ticker")["mention_count"].to_dict()
    rows = []
    for _, r in rank_now.iterrows():
        tk = str(r["ticker"]).upper(); now = float(r["mention_count"])
        delta = now - float(prev.get(tk, 0.0))
        if delta >= c["min_delta"]:
            rows.append((delta, tk, now))
    rows.sort(key=lambda x: -x[0])
    max_delta = rows[0][0] if rows else 1.0

    candidates: List[Dict] = []
    for rank, (delta, tk, now) in enumerate(rows[:c["top_k"]], 1):
        symbol = f"{tk}USDT"
        df = klines.get(symbol)
        if df is None:
            df = klines.get(tk)
        confirmed, entry, stop, tp, feat = _breakout(df, c)
        cand = {
            "symbol": symbol, "rank": rank, "score": round(delta / max_delta, 4),
            "side": "long",
            "reason": f"提及量環比 +{int(delta)}(暴增);技術突破確認={'是' if confirmed else '否'}",
            "features": {"mention_delta": int(delta), "mention_now": int(now), **feat},
        }
        if confirmed:
            cand["trade"] = {
                "entry": {"type": "market", "zone": [round(entry, 6), round(entry, 6)]},
                "stop_loss": stop, "take_profit": [{"price": tp, "close_pct": 1.0}],
                "size": {"mode": "risk_pct", "value": c["size_risk_pct"],
                         "leverage": c["leverage"], "max_notional_usdt": c["max_notional_usdt"]},
            }
        candidates.append(cand)
    return [{
        "schema": "cyqnt.signal/v1", "kind": "selection", "bot_id": BOT_ID,
        "ts": int(as_of_ms), "as_of": int(as_of_ms), "universe_size": int(len(rank_now)),
        "candidates": candidates,
        "reason": f"Square 提及量環比暴增(≥+{c['min_delta']})取 top{c['top_k']},再要求 Donchian{c['breakout_window']} 突破+量能×{c['vol_surge_mult']} 才給進場",
        "provenance": {"strategy_id": BOT_ID, "version": VERSION, "inputs": ["universe", "news", "kline"]},
    }]
