"""trade_plan — 把既有 blocks 組裝成一張「可執行交易計畫」。

給 T1(結構化交易計畫)用:輸入單標的 OHLCV df,對**最後一根**產出方向/信心/
進場區/停損/分批止盈/倉位建議。純函式、不下單、不連網。

註:本檔自算停損(long: entry-mult*ATR),不呼叫 sizing.compute_stop_price。
原因是當初實測 direction="long" 會回 entry+mult*ATR(停損跑到進場價上方,方向反了)——
那個 bug 已修(該函式現在大小寫不敏感),所以這裡改用它也會拿到同樣的值;維持自算只是
不想在沒有必要時動已驗證過的計畫邏輯。倉位用 sizing.fixed_risk_pct(已驗證正確)。
"""
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
from cyqnt_trd.blocks import indicators as I
from cyqnt_trd.blocks import sizing as S

DEFAULTS = {
    "ema_fast": 20, "ema_slow": 50, "ema_trend": 200,
    "rsi_period": 14, "adx_period": 14, "atr_period": 14,
    "stop_mult": 2.0, "tp_mults": [1.5, 3.0], "tp_close": [0.5, 0.5],
    "balance": 10000.0, "risk_pct": 1.0, "max_leverage": 5.0,
}


def _verdict(conf: float) -> str:
    if conf >= 0.75: return "STRONG_CANDIDATE"
    if conf >= 0.60: return "CANDIDATE"
    if conf >= 0.45: return "WATCHLIST"
    if conf >= 0.30: return "SKIP"
    return "AVOID"


def build_trade_plan(df: pd.DataFrame, symbol: str, interval: str,
                     market_type: str = "futures", cfg: Optional[dict] = None) -> Dict:
    """回傳一張 cyqnt.signal/v1 (kind=trade) 計畫(對 df 最後一根)。"""
    c = dict(DEFAULTS); c.update(cfg or {})
    if len(df) < max(c["ema_trend"], c["adx_period"]) + 2:
        raise ValueError(f"df 太短({len(df)} 根),需 ≥ {c['ema_trend']+2}")
    close = df["close"]
    ema_f = I.ema(close, c["ema_fast"]); ema_s = I.ema(close, c["ema_slow"]); ema_t = I.ema(close, c["ema_trend"])
    rsi = I.rsi(close, c["rsi_period"])
    macd_line, macd_sig, macd_hist = I.macd(close)
    adx = I.adx(df, c["adx_period"])[0]
    atr = I.atr(df, c["atr_period"])
    i = len(df) - 1
    px = float(close.iloc[i]); a = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0

    # ---- 多因子投票(自算,避免 direction_from_multi_factor 的字串詞彙陷阱)----
    votes, reasons = 0, []
    if ema_f.iloc[i] > ema_s.iloc[i] and px > ema_t.iloc[i]:
        votes += 1; reasons.append(f"趨勢多(EMA{c['ema_fast']}>EMA{c['ema_slow']} 且收盤>EMA{c['ema_trend']})")
    elif ema_f.iloc[i] < ema_s.iloc[i] and px < ema_t.iloc[i]:
        votes -= 1; reasons.append("趨勢空")
    r = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else 50.0
    if r > 55: votes += 1; reasons.append(f"RSI {r:.0f}>55")
    elif r < 45: votes -= 1; reasons.append(f"RSI {r:.0f}<45")
    h = float(macd_hist.iloc[i]) if pd.notna(macd_hist.iloc[i]) else 0.0
    if h > 0: votes += 1; reasons.append("MACD 柱>0")
    elif h < 0: votes -= 1; reasons.append("MACD 柱<0")
    adx_v = float(adx.iloc[i]) if pd.notna(adx.iloc[i]) else 0.0

    side = "long" if votes > 0 else ("short" if votes < 0 else "flat")
    strength = min(1.0, abs(votes) / 3.0)                 # 因子一致度
    trend_conf = min(1.0, adx_v / 40.0)                   # 趨勢強度
    confidence = round(0.5 * strength + 0.5 * trend_conf, 3)

    # ---- 停損/分批止盈(自算,方向正確)----
    if side == "long":
        stop = px - c["stop_mult"] * a
        tps = [px + m * a for m in c["tp_mults"]]
    elif side == "short":
        stop = px + c["stop_mult"] * a
        tps = [px - m * a for m in c["tp_mults"]]
    else:
        stop = px; tps = [px]
    take_profit = [{"price": round(tp, 6), "close_pct": cp}
                   for tp, cp in zip(tps, c["tp_close"])]

    # ---- 倉位(fixed_risk_pct,已驗證正確)----
    stop_dist_pct = abs(px - stop) / px * 100 if px and side != "flat" else 0.0
    size = {"mode": "risk_pct", "value": c["risk_pct"] / 100.0,
            "leverage": c["max_leverage"], "max_notional_usdt": None}
    if stop_dist_pct > 0:
        sz = S.fixed_risk_pct(c["balance"], risk_pct=c["risk_pct"],
                              max_leverage=c["max_leverage"], stop_distance_pct=stop_dist_pct)
        size["max_notional_usdt"] = round(sz["notional"], 2)
        size["leverage"] = round(sz["leverage"], 2)

    ts = int(df["close_time"].iloc[i]) if "close_time" in df.columns else 0
    return {
        "schema": "cyqnt.signal/v1", "kind": "trade",
        "bot_id": "structured_trade_plan", "ts": ts,
        "symbol": symbol.upper(), "market_type": market_type,
        "side": side, "action": "entry" if side != "flat" else "adjust",
        "confidence": confidence, "verdict": _verdict(confidence),
        "entry": {"type": "market", "zone": [round(px, 6), round(px, 6)]},
        "stop_loss": round(stop, 6),
        "take_profit": take_profit,
        "size": size,
        "time_horizon": {"1m": "scalp", "5m": "scalp", "15m": "scalp",
                         "1h": "intraday", "4h": "swing", "1d": "swing"}.get(interval, "intraday"),
        "reason": "; ".join(reasons) if reasons else "無明確方向(因子中性)",
        "features": {"rsi": round(r, 2), "adx": round(adx_v, 2),
                     "macd_hist": round(h, 6), "atr": round(a, 6), "votes": votes},
        "provenance": {"strategy_id": "structured_trade_plan", "version": "v1",
                       "inputs": ["kline"]},
    }
