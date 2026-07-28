"""N1 — News Catalyst Selector(新聞催化劑選幣)。

依新聞/社群訊號選幣——「哪些幣有利好/被提及」。
做法:全市場 universe → 用 Binance Square 提及量+情緒(getTickerRank)打分排名 →
輸出 top-K 候選(附方向)。輸出 kind=selection(cyqnt.signal/v1)。

- 骨架/輸出固定;CONFIG(門檻/權重)可調。
- 資料:universe(24h ticker)+ ticker_rank(Square)。live 自動 fetch;回測/驗證可傳入。
"""
from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
from cyqnt_trd.blocks import universe as U

BOT_ID = "news_catalyst_selector"
VERSION = "v1"

CONFIG = {
    "min_quote_volume": 1e8,   # 流動性門檻
    "min_mentions": 30,        # 至少 N 次 Square 提及(濾雜訊)
    "top_k": 5,                # 輸出候選數
    "long_ratio": 0.55,        # bull_ratio ≥ → long
    "short_ratio": 0.45,       # bull_ratio ≤ → short
}


def inputs() -> Dict[str, object]:
    return {"universe": True, "news": True, "sentiment": True, "kline": []}


def generate_selection(universe_df: pd.DataFrame, ticker_rank_df: Optional[pd.DataFrame] = None,
                       *, as_of_ms: int = 0, market_type: str = "futures",
                       cfg: Optional[dict] = None) -> List[Dict]:
    """回傳 [一個 selection 訊號](cyqnt.signal/v1, kind=selection)。"""
    c = dict(CONFIG); c.update(cfg or {})
    uni = U.filter_quote_volume(universe_df, c["min_quote_volume"])
    uni = U.augment_with_news(uni, ticker_rank_df)                  # 掛 news_* 欄
    uni = uni.dropna(subset=["news_mention_count"])
    uni = uni[uni["news_mention_count"] >= c["min_mentions"]]
    universe_size = int(len(universe_df))

    candidates: List[Dict] = []
    if len(uni):
        max_m = float(uni["news_mention_count"].max()) or 1.0
        rows = []
        for _, r in uni.iterrows():
            br = float(r.get("news_bull_ratio")) if pd.notna(r.get("news_bull_ratio")) else 0.5
            mentions = float(r["news_mention_count"])
            conviction = max(br, 1.0 - br)                          # 情緒一面倒程度
            score = round((mentions / max_m) * conviction, 4)
            side = "long" if br >= c["long_ratio"] else ("short" if br <= c["short_ratio"] else "long")
            rows.append((score, r["symbol"], br, mentions, side,
                         int(r.get("news_unique_authors")) if pd.notna(r.get("news_unique_authors")) else 0))
        rows.sort(key=lambda x: -x[0])
        for rank, (score, sym, br, mentions, side, authors) in enumerate(rows[:c["top_k"]], 1):
            candidates.append({
                "symbol": str(sym).upper(), "rank": rank, "score": score, "side": side,
                "reason": f"Square 提及 {int(mentions)} 次、情緒多空比 {br:.2f}",
                "features": {"news_mention_count": int(mentions), "news_bull_ratio": round(br, 3),
                             "news_unique_authors": authors},
            })
    return [{
        "schema": "cyqnt.signal/v1", "kind": "selection",
        "bot_id": BOT_ID, "ts": int(as_of_ms), "as_of": int(as_of_ms),
        "universe_size": universe_size,
        "candidates": candidates,
        "reason": f"Square 提及量×情緒打分,取流動性≥{c['min_quote_volume']:.0e}、提及≥{c['min_mentions']} 的 top {c['top_k']}",
        "provenance": {"strategy_id": BOT_ID, "version": VERSION,
                       "inputs": ["universe", "news", "sentiment"]},
    }]


def generate_live(*, market_type: str = "futures", cfg: Optional[dict] = None) -> List[Dict]:
    """live:自動抓 universe + ticker_rank(需連網)。"""
    uni = U.fetch_perpetual_universe(market_type=market_type)
    from cyqnt_trd.data_cli import fetch_ticker_rank
    rank = fetch_ticker_rank(window="24h", limit=100)
    return generate_selection(uni, rank, market_type=market_type, cfg=cfg)
