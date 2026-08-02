"""箱體整理選幣 —— 直接寫 Python 的 selection_fn,不經過 YAML。

這是什麼
--------
語料裡的一筆真實需求(``strategy_test_queue.csv`` 第 13 名,14 個對話問過同一件事)::

    找币，帮找币安平台里永续合约里的最近 7 天或者 10 天/15 天/20 天以上做箱体运动
    的代币。需求:
      1. 找龙头币(龙头币里没有可以放宽条件找非龙头)
      2. 每天价格段和之前每一天高度重叠的,而不是只和前一天,而是每天
      3. 当天上下振幅在 5% 以上
      4. 平均天成交额在 2000 万以上

為什麼這一筆不走 YAML
---------------------
條件 2 是**同一檔幣內部、逐日兩兩比對、再聚合成一個布林**。selection 層的
``augment_with_indicator`` 只能把一檔幣的一段 K 線壓成**一個數字**,而它的 ``agg``
字彙是 ``last / min / max / mean / any_negative / all_negative`` —— 沒有任何一個
表達得出「每一天都和其他每一天重疊」。``blocks.indicators`` 裡也沒有箱體指標。

所以這份 spec 在 YAML 表面是 ``not_expressible``。而它在 Python 裡是六行。

寫 Python 不代表跳出契約
------------------------
YAML 那條路最後也只是產生一個 ``selection_fn``(``interpreter.build_selection_fn``
回傳的就是它),再交給 ``blocks.strategy.register_selection``。這裡直接提供同一個
形狀的 callable,所以:

* 收窄仍然用現成的 ``blocks.universe`` block —— 能用既有 block 表達的部分不重寫,
  否則就變成第二套跟第一套不一致的篩選邏輯;
* 輸出仍然是 ``cyqnt.signal/v2``,``kind=selection``,下游一個字都不用改;
* 一樣吃凍結的 ``cyqnt.input/v1`` bundle,所以離線、決定性、可重播。

換來的是:任意 Python 都能當條件。付出的是:這份策略的邏輯不再能被
``validate_spec`` 靜態檢查,也不會被 nl2yaml 的 G1a–G1e 閘門看到 —— 正確性完全
靠這個檔案自己的斷言與下面的 ``--explain`` 輸出。

用法
----
    python -m strategies.selection.box_range_screener \\
        --bundle tests/standard_bot/fixtures/universe_derivatives.json --explain
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from cyqnt_trd.blocks import universe as ub

# ---------------------------------------------------------------------------
# 這份 spec 替使用者選的讀法 —— 跟著輸出走,不是註解
# ---------------------------------------------------------------------------
#: 和 ``strategy.assumptions[]`` 同一個形狀與同一個理由:需求裡有兩處沒有可計算
#: 的定義,系統的預設是「取比較可能的讀法然後說出來」,而說出來只有在跟著產出物
#: 一起走的時候才算數。這幾條會進 signal 的 warnings。
ASSUMPTIONS = (
    {
        "cid": "overlap",
        "reading": "「每天價格段和之前每一天高度重疊」讀作:窗口內所有日 K 的"
                   "[low, high] 區間**共同交集**非空,且交集寬度 / 箱體總寬度 ≥ "
                   "min_overlap(預設 0.5)",
        "alternatives": ["逐對重疊各自 ≥ 門檻(較寬鬆,允許存在一天與另一天完全不交)"],
        "basis": "「每一天都和其他每一天重疊」在數學上等價於 max(lows) <= min(highs),"
                 "也就是共同交集非空;「高度」沒有可計算定義,用交集佔箱體的比例量化",
    },
    {
        "cid": "leader",
        "reading": "「龍頭幣」無可用資料來源(市值/流通量在公開 API 完全沒有),"
                   "本 spec **不表達**這一條",
        "alternatives": ["拿成交額當龍頭的代理"],
        "basis": "使用者自己寫了「龍頭幣裡沒有可以放寬條件找非龍頭」,所以放寬是他"
                 "授權的;拿成交額代理則是換一個問題回答,不做",
    },
)


# ---------------------------------------------------------------------------
# 條件 2:箱體重疊 —— YAML 表面寫不出來的那一條
# ---------------------------------------------------------------------------
def box_overlap(daily: pd.DataFrame) -> Optional[Dict[str, float]]:
    """一檔幣在給定日 K 窗口上的箱體特徵,資料不足回 None。

    「每天都和之前每一天重疊」看起來是 O(n²) 的兩兩比對,實際上不是:兩個區間重疊
    的條件是 ``max(low_i, low_j) <= min(high_i, high_j)``,而**所有**配對都成立
    等價於 ``max(所有 low) <= min(所有 high)`` —— 也就是全體的共同交集非空。一次
    掃描就夠,而且它同時給出「重疊有多深」的量:交集寬度佔箱體總寬度的比例。

    這個等價關係是這條需求能便宜實作的原因,也是它在 YAML 裡寫不出來的原因:
    ``agg`` 能把一段 K 線壓成一個數字,但壓法只有六種,沒有一種是「取 low 的最大值
    和 high 的最小值再相除」。

    門檻與日振幅是綁在一起的,這點在調參前要先知道:``overlap_ratio`` 要達到 0.5,
    日內振幅必須顯著大於多日中心漂移 —— 中心漂移剛好等於半個日振幅時,交集就退化
    成一個刀口(ratio ≈ 0)。所以「日振幅 ≥ 5% 且 overlap ≥ 0.5」合起來讀是:
    **每天上下震得夠大、但十五天下來哪裡也沒去**。那正是箱體的意思,不過它比
    「十五天漲跌幅很小」嚴格得多,後者會把緩慢陰跌也算進來。
    """
    if daily.empty:
        return None
    lows = pd.to_numeric(daily["low"], errors="coerce")
    highs = pd.to_numeric(daily["high"], errors="coerce")
    if lows.isna().any() or highs.isna().any():
        return None
    floor, ceiling = float(lows.max()), float(highs.min())
    box_low, box_high = float(lows.min()), float(highs.max())
    span = box_high - box_low
    if span <= 0 or box_low <= 0:
        return None
    return {
        # 負值代表有某兩天完全不相交 —— 不是箱體
        "overlap_ratio": (ceiling - floor) / span,
        "box_width_pct": span / box_low * 100.0,
        "box_low": box_low,
        "box_high": box_high,
    }


def latest_amplitude_pct(daily: pd.DataFrame) -> Optional[float]:
    """條件 3:當天上下振幅。用最後一根日 K 的 (high - low) / low。"""
    if daily.empty:
        return None
    last = daily.iloc[-1]
    low = float(last["low"])
    if low <= 0:
        return None
    return (float(last["high"]) - low) / low * 100.0


def mean_daily_turnover(daily: pd.DataFrame) -> Optional[float]:
    """條件 4:平均日成交額。

    優先用 K 線自己帶的 quote_volume;沒有就用 close × volume 估。兩者不同,所以
    哪一個被用到會寫進候選的 features,而不是靜默替換。
    """
    if daily.empty:
        return None
    if "quote_volume" in daily.columns and daily["quote_volume"].notna().all():
        return float(pd.to_numeric(daily["quote_volume"]).mean())
    close = pd.to_numeric(daily["close"], errors="coerce")
    volume = pd.to_numeric(daily["volume"], errors="coerce")
    if close.isna().any() or volume.isna().any():
        return None
    return float((close * volume).mean())


# ---------------------------------------------------------------------------
# selection_fn —— 與 build_selection_fn(spec) 回傳的是同一個形狀
# ---------------------------------------------------------------------------
def build(
    *,
    window_days: int = 15,
    min_overlap: float = 0.5,
    min_amplitude_pct: float = 5.0,
    min_mean_turnover: float = 20_000_000.0,
    top_k: int = 10,
) -> Callable[..., List[Dict[str, Any]]]:
    """做出一個 selection_fn。參數就是這份策略的旋鈕,全部有預設值且明寫。"""

    def selection_fn(universe_df, ticker_rank_df=None, *, frames=None,
                     **_: Any) -> List[Dict[str, Any]]:
        frames = dict(frames or {})
        frame = universe_df
        if frame is None or not len(frame):
            return []

        # --- 第一段:免費的截面收窄。能用現成 block 的就不自己寫 -------------
        # 這一段決定第二段要付多少錢:日 K 是逐幣抓的,名單多長就是幾個 request。
        meta = frames.get("contract_meta")
        if meta is not None:
            frame = ub.augment_with_contract_meta(frame, meta)
            frame = ub.filter_crypto_only(frame)
        frame = ub.filter_quote_volume(frame, min_quote_volume=min_mean_turnover)

        # --- 第二段:日 K 進來,箱體測試在這裡 -------------------------------
        bars = frames.get("universe_bars")
        if bars is None or not len(bars):
            return []
        bars = pd.DataFrame(bars)
        daily_all = bars[bars["timeframe"].astype(str) == "1d"]
        if daily_all.empty:
            return []

        symbol_col = next((c for c in ("symbol", "instrument_id") if c in frame.columns), None)
        if symbol_col is None:
            return []
        survivors = set(frame[symbol_col].astype(str).str.upper())

        kept: List[Dict[str, Any]] = []
        for symbol, group in daily_all.groupby(daily_all["instrument_id"].astype(str).str.upper()):
            if symbol not in survivors:
                continue
            group = group.sort_values("open_time")
            # 不足窗口長度就跳過,不是用比較短的窗口回答一個比較寬的問題 ——
            # 較短的窗口箱體一定比較緊,那會讓新上市的幣系統性地看起來像箱體。
            if len(group) < window_days:
                continue
            window = group.tail(window_days)

            box = box_overlap(window)
            amplitude = latest_amplitude_pct(window)
            turnover = mean_daily_turnover(window)
            if box is None or amplitude is None or turnover is None:
                continue
            if box["overlap_ratio"] < min_overlap:
                continue
            if amplitude < min_amplitude_pct:
                continue
            if turnover < min_mean_turnover:
                continue

            kept.append({
                "symbol": symbol,
                # 排名鍵:重疊愈深、箱體愈明確。刻意不用成交額 —— 那會讓籃子永遠
                # 由最大的幣佔滿,而「哪個箱體最乾淨」才是這個需求問的事。
                "score": round(box["overlap_ratio"] * 100.0, 6),
                "side": "neutral",
                "features": {
                    "overlap_ratio": round(box["overlap_ratio"], 4),
                    "box_width_pct": round(box["box_width_pct"], 3),
                    "box_low": box["box_low"],
                    "box_high": box["box_high"],
                    "latest_amplitude_pct": round(amplitude, 3),
                    "mean_daily_turnover": round(turnover, 2),
                    "turnover_source": ("quote_volume"
                                        if "quote_volume" in window.columns
                                        and window["quote_volume"].notna().all()
                                        else "close_x_volume"),
                    "window_days": window_days,
                },
            })

        kept.sort(key=lambda row: -row["score"])
        scored_pool = len(kept)
        kept = kept[:top_k]
        for rank, row in enumerate(kept, start=1):
            row["rank"] = rank
            row["scored_pool"] = scored_pool
            row["reason"] = (
                "overlap_ratio=%.3f over %dd (box %.4g-%.4g, %.1f%% wide), "
                "latest amplitude %.2f%%, mean turnover %.3g, rank %d of %d"
                % (row["features"]["overlap_ratio"], window_days,
                   row["features"]["box_low"], row["features"]["box_high"],
                   row["features"]["box_width_pct"],
                   row["features"]["latest_amplitude_pct"],
                   row["features"]["mean_daily_turnover"], rank, len(kept)))
        return kept

    return selection_fn


# ---------------------------------------------------------------------------
# 跑起來:餵凍結 bundle,拿到 cyqnt.signal/v2
# ---------------------------------------------------------------------------
def run(bundle_path: str, **kwargs: Any) -> Dict[str, Any]:
    """在凍結的 cyqnt.input/v1 上跑一次,回傳那一份 cyqnt.signal/v2。

    ``batch_to_signals`` 回的是 ``list[StandardSignal]`` —— 選幣一次只發一個訊號,
    所以取第一個。空 list 代表 plugin 連 envelope 都沒發,那是 bug 不是空籃子
    (空籃子仍會發一個 candidates=[] 的訊號),所以 raise 而不是回 None。"""
    from cyqnt_trd.blocks import strategy as blocks_strategy
    from cyqnt_trd.standard_bot.adapter import batch_to_signals
    from cyqnt_trd.standard_bot.data import load_input_bundle

    snapshot = load_input_bundle(bundle_path)
    plugin = blocks_strategy.build_selection_plugin(
        "box_range_screener", build(**kwargs),
        assumptions=tuple(
            "assumption (%s): %s — not: %s [%s]"
            % (a["cid"], a["reading"], "; ".join(a["alternatives"]), a["basis"])
            for a in ASSUMPTIONS),
    )
    batch = plugin.run(snapshot, {})
    signals = batch_to_signals(batch)
    if not signals:
        raise RuntimeError(
            "selection plugin emitted no envelope at all. An empty basket still "
            "emits one signal with candidates=[], so this is a wiring failure, "
            "not 'no coin qualified'.")
    return signals[0].to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--window-days", type=int, default=15)
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--min-amplitude-pct", type=float, default=5.0)
    ap.add_argument("--min-turnover", type=float, default=20_000_000.0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--explain", action="store_true",
                    help="印出可讀的候選表而不是原始 JSON")
    a = ap.parse_args()

    out = run(a.bundle, window_days=a.window_days, min_overlap=a.min_overlap,
              min_amplitude_pct=a.min_amplitude_pct,
              min_mean_turnover=a.min_turnover, top_k=a.top_k)
    if not a.explain:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    signal = out
    print("kind=%s  scope=%s  candidates=%d" % (
        signal["kind"], signal["market_scope"], len(signal["candidates"])))
    print("summary: %s" % signal["summary"])
    for line in signal["warnings"]:
        if line.startswith("assumption"):
            print("  " + line[:160])
    print()
    if not signal["candidates"]:
        print("(空籃子 —— reason_codes=%s)" % signal["reason_codes"])
        return
    print("%-4s %-12s %8s %9s %9s %14s" % (
        "rank", "symbol", "overlap", "box%", "amp%", "turnover"))
    for candidate in signal["candidates"]:
        f = candidate["features"]
        print("%-4d %-12s %8.3f %9.2f %9.2f %14.3g" % (
            candidate["rank"], candidate["symbol"], f["overlap_ratio"],
            f["box_width_pct"], f["latest_amplitude_pct"], f["mean_daily_turnover"]))


if __name__ == "__main__":
    main()
