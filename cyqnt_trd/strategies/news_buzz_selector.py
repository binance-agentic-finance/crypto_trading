"""新聞熱度選幣 —— 一次對整個幣種宇宙排名,選出一籃候選

策略邏輯
--------
1. 流動性過濾:24h 成交額 < 1 億 USDT 的直接剔除(選出來也吃不下)
2. 掛上 Square 熱度:提及量、多空情緒比
3. 依提及量由大到小排名,取前 K 名
4. 方向由情緒決定:多空比 ≥ 0.55 做多、≤ 0.45 做空,中間不表態

介面
----
選幣型走 ``selection_fn(universe_df, ticker_rank_df, ...) -> candidates``
+ ``strategy.register_selection(...)``,和交易型的 ``make_signals`` 並列。
框架把回傳的候選清單包成 ``cyqnt.signal/v2``(``kind=selection``)。

一個必須處理的坑:同一資產的不同計價對
--------------------------------------
Square 的提及量是按 **base token** 統計的(``BTC``),join 回來時會貼到
``BTCUSDT`` / ``BTCUSDC`` **每一個交易對**上,分數完全相同。不處理的話
``top_k=5`` 會變成「3 個資產佔 5 個位置」,BTC 與 SOL 各拿雙倍權重 ——
拿去下單就是無意識的加倉。

所以這裡按 base asset 去重,同分時**留成交額大的那一對**:分數既然相同,
決定的就不是選誰,而是訂單會在哪裡成交。

誠實邊界
--------
Square 熱度是 ``FORWARD_ONLY``:**沒有時點歷史**。這支今天回測不了 ——
重播會把當下的熱度榜貼到每一根歷史 bar 上。它能跑 live / paper,
edge 在前向收集夠久之前只是假設。
"""

from cyqnt_trd.blocks import strategy as strat, universe as U

BOT_ID = "news_buzz_selector"

CONFIG = {
    "min_quote_volume": 1e8,    # 24h 成交額下限(USDT)
    "top_k": 5,
    "min_mentions": 100,        # 提及量太少的排名沒有意義
    "long_ratio": 0.55,         # 多空比 ≥ 這個值做多
    "short_ratio": 0.45,        # ≤ 這個值做空;中間不表態
    "dedupe_by_base_asset": True,
}

#: 常見計價幣,用來把 BTCUSDT / BTCUSDC 收斂成 BTC
_QUOTES = ("USDT", "USDC", "FDUSD", "TUSD", "BUSD", "USD")


def _base_asset(symbol: str) -> str:
    value = str(symbol).upper()
    for quote in _QUOTES:
        if value.endswith(quote) and len(value) > len(quote):
            return value[: -len(quote)]
    return value


def selection_fn(universe_df, ticker_rank_df=None, **_):
    """回傳候選 dict 清單:symbol / rank / score / side / reason / features。"""
    import pandas as pd

    c = CONFIG
    if universe_df is None or not len(universe_df):
        return []

    # 1. 流動性 → 2. 掛熱度
    uni = U.filter_quote_volume(universe_df, c["min_quote_volume"])
    uni = U.augment_with_news(uni, ticker_rank_df)
    if "news_mention_count" not in uni.columns:
        return []                       # 熱度榜沒回來,沒有東西可以排
    uni = uni.dropna(subset=["news_mention_count"])
    uni = uni[uni["news_mention_count"] >= c["min_mentions"]]
    if not len(uni):
        return []

    # 3. 排名 —— 去重要在取前 K 名「之前」,否則籃子裡是重複的資產
    ranked = uni.sort_values("news_mention_count", ascending=False)
    if c["dedupe_by_base_asset"]:
        volume_col = next((col for col in ("quoteVolume", "quote_volume")
                           if col in ranked.columns), None)
        if volume_col is not None:
            # 同一 token 的每個交易對分數相同,所以 tie-break 決定的是成交地點
            ranked = ranked.sort_values(
                ["news_mention_count", volume_col], ascending=False)
        base = ranked["symbol"].map(_base_asset)
        ranked = ranked[~base.duplicated(keep="first")]
    ranked = ranked.head(int(c["top_k"]))

    # 4. 方向由情緒決定
    cands = []
    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        ratio = row.get("news_bull_ratio")
        ratio = None if ratio is None or pd.isna(ratio) else float(ratio)
        if ratio is None:
            side = "neutral"            # 有熱度但沒有情緒 → 只列入觀察
        elif ratio >= c["long_ratio"]:
            side = "long"
        elif ratio <= c["short_ratio"]:
            side = "short"
        else:
            side = "neutral"

        mentions = int(row["news_mention_count"])
        cands.append({
            "symbol": str(row["symbol"]).upper(),
            "rank": rank,
            "score": round(float(mentions), 2),
            "side": side,
            "reason": ("提及量 %d、多空比 %s" % (
                mentions, "無資料" if ratio is None else "%.2f" % ratio)),
            "features": {
                "news_mention_count": mentions,
                "news_bull_ratio": None if ratio is None else round(ratio, 3),
                "quote_volume": float(row.get("quoteVolume")
                                      or row.get("quote_volume") or 0.0),
                "base_asset": _base_asset(row["symbol"]),
            },
        })
    return cands


strat.register_selection(BOT_ID, selection_fn)
